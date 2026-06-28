from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cse2026.ong_solver import SolverConfig
from cse2026.ong_solver.deeprmsa import (
    DEEPRMSA_CANDIDATE_FEATURE_COLUMNS,
    deeprmsa_features_from_arrays,
)

from ..config import ExperimentConfig
from .ong_solver_eval import _finite_float
from .train_dqn import (
    _device,
    _iter_batches,
    _load_split,
    _raw_float,
    _raw_int,
    _raw_str,
    _splits,
    _target_index_for_key,
    _target_params,
    _target_params_for_metrics,
    _transition_limit,
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_features(split: Any, key: tuple[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = split.arrays_for_key(key)
    q_head_scores = split.candidate_metric_vector(key, "q_head_score")
    candidate_features, context_features = deeprmsa_features_from_arrays(
        node_features=arrays["node_features"],
        global_features=arrays["global_features"],
        request_features=arrays["request_features"],
        action_features=arrays["action_features"],
        q_head_scores=q_head_scores,
        candidate_mask=arrays["candidate_mask"],
    )
    return candidate_features, context_features, arrays["candidate_mask"].astype(np.bool_)


def _zero_features(candidate_dim: int, context_dim: int, n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((n_max, candidate_dim), dtype=np.float32),
        np.zeros((context_dim,), dtype=np.float32),
        np.zeros((n_max,), dtype=np.bool_),
    )


def _batch_tensors(
    split: Any,
    row_indices: np.ndarray,
    *,
    device: str,
    torch: Any,
    gamma: float,
    n_step_return: int,
    target_mode: str,
    target_params: dict[str, Any],
    candidate_dim: int,
    context_dim: int,
) -> dict[str, Any]:
    current_candidates: list[np.ndarray] = []
    current_contexts: list[np.ndarray] = []
    current_masks: list[np.ndarray] = []
    next_candidates: list[np.ndarray] = []
    next_contexts: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []
    selected_indices: list[int] = []
    target_indices: list[int] = []
    rewards: list[float] = []
    discounts: list[float] = []
    done_flags: list[bool] = []
    next_available: list[bool] = []
    n_max = int(split.cfg.n_max)
    zero_candidate, zero_context, zero_mask = _zero_features(candidate_dim, context_dim, n_max)

    for row_position, row in zip(row_indices, split.dqn.iloc[row_indices].itertuples(index=False)):
        key = (str(row.episode_id), int(row.request_id))
        cand, ctx, mask = _state_features(split, key)
        current_candidates.append(cand)
        current_contexts.append(ctx)
        current_masks.append(mask)
        selected_indices.append(int(row.selected_candidate_index))
        target_indices.append(
            _target_index_for_key(
                split,
                key,
                stored_best=int(row.best_candidate_index),
                selected=int(row.selected_candidate_index),
                mode=target_mode,
                params=target_params,
            )
        )
        reward, discount, done, next_key = split.n_step_target_view(
            int(row_position),
            gamma=gamma,
            n_step_return=n_step_return,
        )
        rewards.append(float(reward))
        discounts.append(float(discount))
        done_flags.append(bool(done))
        if (
            done
            or next_key is None
            or next_key not in split.sample_index_by_key
            or next_key not in split.candidate_groups
            or next_key not in split.states
        ):
            next_candidates.append(zero_candidate)
            next_contexts.append(zero_context)
            next_masks.append(zero_mask)
            next_available.append(False)
            continue
        next_cand, next_ctx, next_mask = _state_features(split, next_key)
        next_candidates.append(next_cand)
        next_contexts.append(next_ctx)
        next_masks.append(next_mask)
        next_available.append(bool(next_mask.any()))

    return {
        "candidate_features": torch.as_tensor(np.stack(current_candidates), dtype=torch.float32, device=device),
        "context_features": torch.as_tensor(np.stack(current_contexts), dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(np.stack(current_masks), dtype=torch.bool, device=device),
        "next_candidate_features": torch.as_tensor(np.stack(next_candidates), dtype=torch.float32, device=device),
        "next_context_features": torch.as_tensor(np.stack(next_contexts), dtype=torch.float32, device=device),
        "next_candidate_mask": torch.as_tensor(np.stack(next_masks), dtype=torch.bool, device=device),
        "selected_index": torch.as_tensor(selected_indices, dtype=torch.long, device=device),
        "target_index": torch.as_tensor(target_indices, dtype=torch.long, device=device),
        "reward": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "discount": torch.as_tensor(discounts, dtype=torch.float32, device=device),
        "done": torch.as_tensor(done_flags, dtype=torch.bool, device=device),
        "next_available": torch.as_tensor(next_available, dtype=torch.bool, device=device),
    }


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1e9)


def _entropy(masked_logits: Any, mask: Any, torch: Any) -> Any:
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
    probs = torch.exp(log_probs) * mask.to(dtype=log_probs.dtype)
    return -(probs * log_probs).sum(dim=1).mean()


def _candidate_metric(split: Any, row_position: int, action_index: int, metric: str, default: float) -> float:
    row = split.dqn.iloc[int(row_position)]
    key = (str(row.episode_id), int(row.request_id))
    candidate = split.candidate_row(key, int(action_index))
    if candidate is None:
        return default
    return _finite_float(candidate.get(metric), default=default)


def _apply_clamp50_reward_override(split: Any, config: ExperimentConfig) -> dict[str, Any] | None:
    mode = str(config.resolved.get("reward_override_mode", config.raw.get("reward_override_mode", "stored"))).strip().lower()
    if mode in {"", "stored", "none"}:
        return None
    if mode != "problem_shaped":
        raise ValueError(f"Unsupported reward_override_mode: {mode}")

    dqn = split.dqn.copy()
    rewards: list[float] = []
    accepted_count = 0
    blocked_count = 0
    for row in dqn.itertuples(index=False):
        selected = int(row.selected_candidate_index)
        accepted = selected >= 0 and not bool(row.blocked)
        key = (str(row.episode_id), int(row.request_id))
        candidate = split.candidate_row(key, selected) if accepted else None
        if candidate is None:
            rewards.append(_raw_float(config, "block_penalty", -1.5))
            blocked_count += 1
            continue
        accepted_count += 1
        energy_norm = _finite_float(candidate.get("energy_increment_norm"), default=math.nan)
        if not math.isfinite(energy_norm):
            energy_norm = _finite_float(candidate.get("energy_increment"), default=0.0) / max(
                _raw_float(config, "reward_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)),
                1e-9,
            )
        delay_norm = _finite_float(candidate.get("delay_norm"), default=math.nan)
        if not math.isfinite(delay_norm):
            delay_norm = _finite_float(candidate.get("delay_ms"), default=0.0) / max(
                _raw_float(config, "reward_delay_norm_ms", _raw_float(config, "delay_norm_ms", 50.0)),
                1e-9,
            )
        reward = (
            _raw_float(config, "accepted_service_reward", 1.0)
            - _raw_float(config, "reward_energy_weight", 0.25) * energy_norm
            - _raw_float(config, "reward_fragmentation_weight", 0.55)
            * _finite_float(candidate.get("fragmentation_after"), default=1.0)
            + _raw_float(config, "reward_qot_margin_weight", 0.15)
            * _finite_float(candidate.get("qot_margin_norm"), default=0.0)
            - _raw_float(config, "reward_delay_weight", 0.10) * delay_norm
        )
        rewards.append(float(reward))

    original_reward = dqn["reward"].to_numpy(dtype=np.float64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    dqn["reward"] = reward_array
    split.dqn = dqn
    return {
        "mode": mode,
        "split": split.split,
        "mean_original_reward": float(np.mean(original_reward)) if original_reward.size else None,
        "mean_override_reward": float(np.mean(reward_array)) if reward_array.size else None,
        "min_override_reward": float(np.min(reward_array)) if reward_array.size else None,
        "max_override_reward": float(np.max(reward_array)) if reward_array.size else None,
        "accepted_rows": int(accepted_count),
        "blocked_rows": int(blocked_count),
    }


def _evaluate(
    *,
    model: Any,
    split: Any,
    batch_size: int,
    max_batches: int,
    device: str,
    torch: Any,
    gamma: float,
    n_step_return: int,
    target_mode: str,
    target_params: dict[str, Any],
    candidate_dim: int,
    context_dim: int,
) -> dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(0)
    indices = split.valid_indices
    batches = _iter_batches(len(indices), batch_size, shuffle=False, rng=rng)
    if max_batches > 0:
        batches = batches[:max_batches]

    total = 0
    target_agree = 0
    selected_agree = 0
    mask_violations = 0
    value_losses: list[float] = []
    greedy_energy: list[float] = []
    greedy_fragmentation: list[float] = []
    greedy_delay: list[float] = []
    greedy_qot: list[float] = []
    huber = torch.nn.SmoothL1Loss(reduction="none")

    with torch.no_grad():
        for positions in batches:
            row_indices = indices[positions]
            batch = _batch_tensors(
                split,
                row_indices,
                device=device,
                torch=torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
                candidate_dim=candidate_dim,
                context_dim=context_dim,
            )
            logits, values = model(batch["candidate_features"], batch["context_features"])
            next_logits, next_values = model(batch["next_candidate_features"], batch["next_context_features"])
            del next_logits
            bootstrap = batch["next_available"] & (~batch["done"])
            target_values = batch["reward"] + batch["discount"] * torch.where(
                bootstrap,
                next_values,
                torch.zeros_like(next_values),
            )
            value_losses.extend(float(value) for value in huber(values, target_values).detach().cpu().numpy())
            masked = _masked_logits(logits, batch["candidate_mask"])
            greedy = masked.argmax(dim=1)
            invalid = ~batch["candidate_mask"].gather(1, greedy[:, None]).squeeze(1)
            mask_violations += int(invalid.sum().detach().cpu())
            target = batch["target_index"]
            selected = batch["selected_index"]
            target_valid = (target >= 0) & (target < batch["candidate_mask"].shape[1])
            selected_valid = selected >= 0
            target_agree += int(((greedy == target) & target_valid).sum().detach().cpu())
            selected_agree += int(((greedy == selected) & selected_valid).sum().detach().cpu())
            total += int(len(row_indices))

            for row_position, action_index in zip(row_indices, greedy.detach().cpu().numpy()):
                greedy_energy.append(_candidate_metric(split, int(row_position), int(action_index), "energy_increment", math.nan))
                greedy_fragmentation.append(
                    _candidate_metric(split, int(row_position), int(action_index), "fragmentation_after", math.nan)
                )
                greedy_delay.append(_candidate_metric(split, int(row_position), int(action_index), "delay_ms", math.nan))
                greedy_qot.append(_candidate_metric(split, int(row_position), int(action_index), "qot_margin_norm", math.nan))

    def mean(values: list[float]) -> float | None:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return None
        return float(np.mean(np.asarray(finite, dtype=np.float64)))

    return {
        "samples": int(total),
        "learning_target": target_mode,
        "value_huber_loss": mean(value_losses),
        "greedy_matches_learning_target_index": None if total == 0 else float(target_agree / total),
        "greedy_matches_recorded_selected_candidate_index": None if total == 0 else float(selected_agree / total),
        "mask_violations": int(mask_violations),
        "mean_greedy_energy_increment": mean(greedy_energy),
        "mean_greedy_fragmentation_after": mean(greedy_fragmentation),
        "mean_greedy_delay_ms": mean(greedy_delay),
        "mean_greedy_qot_margin_norm": mean(greedy_qot),
        "generated_state_cache_size": int(len(split.cache)),
    }


def _validation_score(metrics: dict[str, Any], config: ExperimentConfig) -> float:
    miss = 1.0 - float(metrics.get("greedy_matches_learning_target_index") or 0.0)
    energy = float(metrics.get("mean_greedy_energy_increment") or 1200.0)
    fragmentation = float(metrics.get("mean_greedy_fragmentation_after") or 1.0)
    delay = float(metrics.get("mean_greedy_delay_ms") or 50.0)
    qot = float(metrics.get("mean_greedy_qot_margin_norm") or 0.0)
    return float(
        _raw_float(config, "validation_best_miss_weight", 1.0) * miss
        + _raw_float(config, "validation_energy_weight", 0.35) * (energy / max(_raw_float(config, "validation_energy_norm_w", 1200.0), 1e-9))
        + _raw_float(config, "validation_fragmentation_weight", 1.25) * fragmentation
        + _raw_float(config, "validation_delay_weight", 0.05) * (delay / max(_raw_float(config, "validation_delay_norm_ms", 50.0), 1e-9))
        - _raw_float(config, "validation_qot_margin_weight", 0.35) * qot
    )


def run_train_deeprmsa_a3c(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_deeprmsa_a3c requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    from cse2026.ong_solver.models import DeepRmsaA3CNetwork, require_torch

    torch = require_torch()
    device = _device(config, torch)
    cfg = SolverConfig(n_max=_raw_int(config, "n_max", 32), rng_seed=int(config.seed), device=device)
    train_split, val_split, test_split = _splits(config)
    transition_limits = {
        "train": _transition_limit(config, train_split),
        "val": _transition_limit(config, val_split),
        "test": _transition_limit(config, test_split),
    }
    train = _load_split(config.dataset_path, train_split, cfg, transition_limit=transition_limits["train"], seed=config.seed)
    val = _load_split(config.dataset_path, val_split, cfg, transition_limit=transition_limits["val"], seed=config.seed + 1)
    test = _load_split(config.dataset_path, test_split, cfg, transition_limit=transition_limits["test"], seed=config.seed + 2)
    reward_override_stats = {
        "train": _apply_clamp50_reward_override(train, config),
        "val": _apply_clamp50_reward_override(val, config),
        "test": _apply_clamp50_reward_override(test, config),
    }

    sample_key = (str(train.dqn.iloc[int(train.valid_indices[0])].episode_id), int(train.dqn.iloc[int(train.valid_indices[0])].request_id))
    sample_candidate, sample_context, _sample_mask = _state_features(train, sample_key)
    candidate_dim = int(sample_candidate.shape[1])
    context_dim = int(sample_context.shape[0])
    hidden_dim = _raw_int(config, "hidden_dim", 128)
    layers = _raw_int(config, "deeprmsa_layers", 5)
    dropout = _raw_float(config, "dropout", 0.05)
    if DeepRmsaA3CNetwork is None:
        raise RuntimeError("DeepRmsaA3CNetwork is unavailable because PyTorch is not installed")
    model = DeepRmsaA3CNetwork(
        n_max=int(cfg.n_max),
        candidate_feature_dim=candidate_dim,
        context_feature_dim=context_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_raw_float(config, "learning_rate", 2e-4),
        weight_decay=_raw_float(config, "weight_decay", 1e-4),
    )
    batch_size = int(config.batch_size)
    max_batches = int(config.max_batches)
    epochs = _raw_int(config, "epochs", 4)
    patience = _raw_int(config, "patience", 2)
    gamma = _raw_float(config, "gamma", 0.95)
    n_step_return = max(1, _raw_int(config, "n_step_return", 3))
    target_mode = _raw_str(config, "learning_target", _raw_str(config, "imitation_target", "best"))
    target_params = _target_params(config)
    pg_weight = _raw_float(config, "policy_gradient_loss_weight", 0.25)
    imitation_weight = _raw_float(config, "imitation_loss_weight", 1.0)
    value_weight = _raw_float(config, "value_loss_weight", 0.50)
    entropy_weight = _raw_float(config, "entropy_loss_weight", 0.01)
    progress_every_batches = _raw_int(config, "progress_every_batches", 0)
    rng = np.random.default_rng(config.seed)
    huber = torch.nn.SmoothL1Loss()
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = -1
    stale = 0
    best_path = run_path / "deeprmsa_a3c_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _iter_batches(len(train.valid_indices), batch_size, shuffle=True, rng=rng)
        if max_batches > 0:
            batches = batches[:max_batches]
        losses: list[float] = []
        policy_losses: list[float] = []
        imitation_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        target_correct = 0
        target_total = 0

        for batch_index, positions in enumerate(batches, start=1):
            row_indices = train.valid_indices[positions]
            batch = _batch_tensors(
                train,
                row_indices,
                device=device,
                torch=torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
                candidate_dim=candidate_dim,
                context_dim=context_dim,
            )
            logits, values = model(batch["candidate_features"], batch["context_features"])
            with torch.no_grad():
                _next_logits, next_values = model(batch["next_candidate_features"], batch["next_context_features"])
                bootstrap = batch["next_available"] & (~batch["done"])
                target_values = batch["reward"] + batch["discount"] * torch.where(
                    bootstrap,
                    next_values,
                    torch.zeros_like(next_values),
                )
            masked = _masked_logits(logits, batch["candidate_mask"])
            log_probs = torch.nn.functional.log_softmax(masked, dim=1)
            target = batch["target_index"]
            selected = batch["selected_index"]
            target_valid = (target >= 0) & (target < masked.shape[1])
            safe_target = target.clamp(0, masked.shape[1] - 1)
            target_valid = target_valid & batch["candidate_mask"].gather(1, safe_target[:, None]).squeeze(1)
            selected_valid = (selected >= 0) & (selected < masked.shape[1])
            safe_selected = selected.clamp(0, masked.shape[1] - 1)
            selected_valid = selected_valid & batch["candidate_mask"].gather(1, safe_selected[:, None]).squeeze(1)
            if not bool(target_valid.any()) and not bool(selected_valid.any()):
                continue
            advantage = target_values - values
            value_loss = huber(values, target_values)
            loss = float(value_weight) * value_loss
            if bool(selected_valid.any()) and pg_weight != 0.0:
                selected_log_prob = log_probs.gather(1, safe_selected[:, None]).squeeze(1)
                pg_loss = -(selected_log_prob[selected_valid] * advantage.detach()[selected_valid]).mean()
                loss = loss + float(pg_weight) * pg_loss
                policy_losses.append(float(pg_loss.detach().cpu()))
            if bool(target_valid.any()) and imitation_weight != 0.0:
                imitation_loss = torch.nn.functional.cross_entropy(masked[target_valid], target[target_valid])
                loss = loss + float(imitation_weight) * imitation_loss
                imitation_losses.append(float(imitation_loss.detach().cpu()))
                greedy = masked[target_valid].argmax(dim=1)
                target_correct += int((greedy == target[target_valid]).sum().detach().cpu())
                target_total += int(target_valid.sum().detach().cpu())
            entropy = _entropy(masked, batch["candidate_mask"], torch)
            loss = loss - float(entropy_weight) * entropy
            entropies.append(float(entropy.detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "deeprmsa_a3c_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(batches),
                            "loss_so_far": None if not losses else float(np.mean(np.asarray(losses, dtype=np.float64))),
                            "target_accuracy_so_far": None if target_total == 0 else float(target_correct / target_total),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        val_metrics = _evaluate(
            model=model,
            split=val,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            torch=torch,
            gamma=gamma,
            n_step_return=n_step_return,
            target_mode=target_mode,
            target_params=target_params,
            candidate_dim=candidate_dim,
            context_dim=context_dim,
        )
        val_score = _validation_score(val_metrics, config)
        row = {
            "phase": "deeprmsa_a3c",
            "epoch": epoch,
            "train_loss": None if not losses else float(np.mean(np.asarray(losses, dtype=np.float64))),
            "train_policy_loss": None if not policy_losses else float(np.mean(np.asarray(policy_losses, dtype=np.float64))),
            "train_imitation_loss": None if not imitation_losses else float(np.mean(np.asarray(imitation_losses, dtype=np.float64))),
            "train_value_loss": None if not value_losses else float(np.mean(np.asarray(value_losses, dtype=np.float64))),
            "train_entropy": None if not entropies else float(np.mean(np.asarray(entropies, dtype=np.float64))),
            "train_target_accuracy": None if target_total == 0 else float(target_correct / target_total),
            "val_score": float(val_score),
            "val": val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_score < best_score:
            best_score = float(val_score)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                    "val_score": float(best_score),
                    "config": config.resolved,
                    "n_max": int(cfg.n_max),
                    "candidate_feature_dim": int(candidate_dim),
                    "context_feature_dim": int(context_dim),
                    "hidden_dim": int(hidden_dim),
                    "layers": int(layers),
                    "dropout": float(dropout),
                    "candidate_feature_columns": list(DEEPRMSA_CANDIDATE_FEATURE_COLUMNS),
                    "learning_target": target_mode,
                    "target_params": _target_params_for_metrics(target_params),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    if best_epoch >= 0:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "stage": "train_deeprmsa_a3c",
        "dataset_path": str(config.dataset_path),
        "device": device,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "train_valid_transitions": int(len(train.valid_indices)),
        "val_valid_transitions": int(len(val.valid_indices)),
        "test_valid_transitions": int(len(test.valid_indices)),
        "transition_limits": transition_limits,
        "reward_override": reward_override_stats,
        "gamma": float(gamma),
        "n_step_return": int(n_step_return),
        "learning_target": target_mode,
        "target_params": _target_params_for_metrics(target_params),
        "policy_gradient_loss_weight": float(pg_weight),
        "imitation_loss_weight": float(imitation_weight),
        "value_loss_weight": float(value_weight),
        "entropy_loss_weight": float(entropy_weight),
        "history": history,
        "train": _evaluate(
            model=model,
            split=train,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            torch=torch,
            gamma=gamma,
            n_step_return=n_step_return,
            target_mode=target_mode,
            target_params=target_params,
            candidate_dim=candidate_dim,
            context_dim=context_dim,
        ),
        "val": _evaluate(
            model=model,
            split=val,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            torch=torch,
            gamma=gamma,
            n_step_return=n_step_return,
            target_mode=target_mode,
            target_params=target_params,
            candidate_dim=candidate_dim,
            context_dim=context_dim,
        ),
        "test": _evaluate(
            model=model,
            split=test,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            torch=torch,
            gamma=gamma,
            n_step_return=n_step_return,
            target_mode=target_mode,
            target_params=target_params,
            candidate_dim=candidate_dim,
            context_dim=context_dim,
        ),
    }
    _write_json(run_path / "metrics.json", final_metrics)
    return final_metrics
