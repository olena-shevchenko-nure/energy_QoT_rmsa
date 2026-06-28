from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from cse2026.ong_solver import SolverConfig

from ..config import ExperimentConfig
from .ong_solver_eval import _finite_float
from .train_dqn import (
    _apply_reward_override,
    _batch_tensors,
    _device,
    _iter_batches,
    _load_split,
    _raw_bool,
    _raw_float,
    _raw_int,
    _raw_str,
    _splits,
    _target_params,
    _target_params_for_metrics,
    _transition_limit,
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model_forward(model: Any, tensors: dict[str, Any], edge_index: Any) -> tuple[Any, Any]:
    return model(
        link_features=tensors["link_features"],
        edge_index=edge_index,
        global_features=tensors["global_features"],
        request_features=tensors["request_features"],
        action_features=tensors["action_features"],
        route_link_mask=tensors["route_link_mask"],
    )


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1e9)


def _entropy(masked_logits: Any, mask: Any, torch: Any) -> Any:
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
    probs = torch.exp(log_probs) * mask.to(dtype=log_probs.dtype)
    return -(probs * log_probs).sum(dim=1).mean()


def _valid_mass_loss(logits: Any, mask: Any, torch: Any) -> Any:
    probs = torch.nn.functional.softmax(logits, dim=1)
    valid_mass = (probs * mask.to(dtype=probs.dtype)).sum(dim=1).clamp_min(1e-8)
    return -torch.log(valid_mass).mean()


def _old_selected_log_prob(
    *,
    q_head_scores: Any,
    candidate_mask: Any,
    selected_index: Any,
    mode: str,
    temperature: float,
    torch: Any,
) -> Any:
    behavior_mode = str(mode or "q_head_softmax").strip().lower()
    if behavior_mode in {"uniform", "uniform_valid"}:
        behavior_logits = torch.zeros_like(q_head_scores)
    elif behavior_mode in {"q_head", "q_head_softmax", "qhead", "qhead_softmax"}:
        safe_scores = torch.where(torch.isfinite(q_head_scores), q_head_scores, torch.zeros_like(q_head_scores))
        behavior_logits = safe_scores / max(float(temperature), 1e-6)
    else:
        raise ValueError(f"Unsupported xlron behavior_policy_mode: {mode}")
    behavior_logits = behavior_logits.masked_fill(~candidate_mask, -1e9)
    behavior_log_probs = torch.nn.functional.log_softmax(behavior_logits, dim=1)
    safe_selected = selected_index.clamp(0, behavior_logits.shape[1] - 1)
    return behavior_log_probs.gather(1, safe_selected[:, None]).squeeze(1)


def _candidate_metric(split: Any, row_position: int, action_index: int, metric: str, default: float) -> float:
    row = split.dqn.iloc[int(row_position)]
    key = (str(row.episode_id), int(row.request_id))
    candidate = split.candidate_row(key, int(action_index))
    if candidate is None:
        return default
    return _finite_float(candidate.get(metric), default=default)


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


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
) -> dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(0)
    indices = split.valid_indices
    batches = _iter_batches(len(indices), batch_size, shuffle=False, rng=rng)
    if max_batches > 0:
        batches = batches[:max_batches]

    edge_index = torch.as_tensor(split.edge_index, dtype=torch.long, device=device)
    huber = torch.nn.SmoothL1Loss(reduction="none")
    total = 0
    target_agree = 0
    selected_agree = 0
    mask_violations = 0
    value_losses: list[float] = []
    greedy_energy: list[float] = []
    greedy_fragmentation: list[float] = []
    greedy_delay: list[float] = []
    greedy_qot: list[float] = []

    with torch.no_grad():
        for positions in batches:
            row_indices = indices[positions]
            batch = _batch_tensors(
                split,
                row_indices,
                device,
                torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
            )
            logits, values = _model_forward(model, batch["current"], edge_index)
            _next_logits, next_values = _model_forward(model, batch["next"], edge_index)
            bootstrap = batch["next_available"] & (~batch["done"])
            target_values = batch["reward"] + batch["discount"] * torch.where(
                bootstrap,
                next_values,
                torch.zeros_like(next_values),
            )
            value_losses.extend(float(value) for value in huber(values, target_values).detach().cpu().numpy())

            current_mask = batch["current"]["candidate_mask"]
            masked = _masked_logits(logits, current_mask)
            greedy = masked.argmax(dim=1)
            invalid = ~current_mask.gather(1, greedy[:, None]).squeeze(1)
            mask_violations += int(invalid.sum().detach().cpu())
            target = batch["best_index"]
            selected = batch["selected_index"]
            target_valid = (target >= 0) & (target < current_mask.shape[1])
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

    return {
        "samples": int(total),
        "learning_target": target_mode,
        "value_huber_loss": _mean(value_losses),
        "greedy_matches_learning_target_index": None if total == 0 else float(target_agree / total),
        "greedy_matches_recorded_selected_candidate_index": None if total == 0 else float(selected_agree / total),
        "mask_violations": int(mask_violations),
        "mean_greedy_energy_increment": _mean(greedy_energy),
        "mean_greedy_fragmentation_after": _mean(greedy_fragmentation),
        "mean_greedy_delay_ms": _mean(greedy_delay),
        "mean_greedy_qot_margin_norm": _mean(greedy_qot),
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


def run_train_xlron_graph_transformer_ppo(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_xlron_graph_transformer_ppo requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    from cse2026.ong_solver.models import XlronGraphTransformerPpoNetwork, require_torch

    torch = require_torch()
    if XlronGraphTransformerPpoNetwork is None:
        raise RuntimeError("XlronGraphTransformerPpoNetwork is unavailable because PyTorch is not installed")
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
    if len(train.valid_indices) == 0:
        raise RuntimeError("train_xlron_graph_transformer_ppo found no usable train transitions")
    reward_override_stats = {
        "train": _apply_reward_override(train, config),
        "val": _apply_reward_override(val, config),
        "test": _apply_reward_override(test, config),
    }

    sample_row = train.dqn.iloc[int(train.valid_indices[0])]
    sample_key = (str(sample_row.episode_id), int(sample_row.request_id))
    sample = train.arrays_for_key(sample_key)
    action_feature_dim = int(sample["action_features"].shape[1])
    link_feature_dim = int(sample["link_features"].shape[1])
    global_feature_dim = int(sample["global_features"].shape[0])
    request_feature_dim = int(sample["request_features"].shape[0])
    embedding_dim = _raw_int(config, "transformer_embedding_size", _raw_int(config, "hidden_dim", 128))
    num_layers = _raw_int(config, "transformer_num_layers", 2)
    num_heads = _raw_int(config, "transformer_num_heads", 8)
    dropout = _raw_float(config, "dropout", 0.05)
    position_dim = _raw_int(config, "transformer_position_dim", 8)
    model = XlronGraphTransformerPpoNetwork(
        action_feature_dim=action_feature_dim,
        link_feature_dim=link_feature_dim,
        global_feature_dim=global_feature_dim,
        request_feature_dim=request_feature_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        position_dim=position_dim,
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
    target_mode = _raw_str(config, "learning_target", "blocking_sensitive_hybrid")
    target_params = _target_params(config)
    ppo_weight = _raw_float(config, "ppo_loss_weight", 1.0)
    imitation_weight = _raw_float(config, "imitation_loss_weight", 0.75)
    value_weight = _raw_float(config, "value_loss_weight", 0.50)
    entropy_weight = _raw_float(config, "entropy_loss_weight", 0.01)
    valid_mass_weight = _raw_float(config, "valid_mass_loss_coef", 0.001)
    clip_epsilon = _raw_float(config, "ppo_clip_epsilon", 0.20)
    behavior_mode = _raw_str(config, "behavior_policy_mode", "q_head_softmax")
    behavior_temperature = _raw_float(config, "behavior_policy_temperature", 0.35)
    normalize_advantage = _raw_bool(config, "normalize_advantage", True)
    progress_every_batches = _raw_int(config, "progress_every_batches", 0)
    edge_index = torch.as_tensor(train.edge_index, dtype=torch.long, device=device)
    huber = torch.nn.SmoothL1Loss()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = -1
    stale = 0
    best_path = run_path / "xlron_graph_transformer_ppo_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _iter_batches(len(train.valid_indices), batch_size, shuffle=True, rng=rng)
        if max_batches > 0:
            batches = batches[:max_batches]
        losses: list[float] = []
        ppo_losses: list[float] = []
        imitation_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        valid_mass_losses: list[float] = []
        target_correct = 0
        target_total = 0

        for batch_index, positions in enumerate(batches, start=1):
            row_indices = train.valid_indices[positions]
            batch = _batch_tensors(
                train,
                row_indices,
                device,
                torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
            )
            logits, values = _model_forward(model, batch["current"], edge_index)
            with torch.no_grad():
                _next_logits, next_values = _model_forward(model, batch["next"], edge_index)
                bootstrap = batch["next_available"] & (~batch["done"])
                target_values = batch["reward"] + batch["discount"] * torch.where(
                    bootstrap,
                    next_values,
                    torch.zeros_like(next_values),
                )
                advantage = target_values - values
                if normalize_advantage and advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)

            current_mask = batch["current"]["candidate_mask"]
            masked = _masked_logits(logits, current_mask)
            log_probs = torch.nn.functional.log_softmax(masked, dim=1)
            selected = batch["selected_index"]
            selected_valid = (selected >= 0) & (selected < masked.shape[1])
            safe_selected = selected.clamp(0, masked.shape[1] - 1)
            selected_valid = selected_valid & current_mask.gather(1, safe_selected[:, None]).squeeze(1)
            target = batch["best_index"]
            target_valid = (target >= 0) & (target < masked.shape[1])
            safe_target = target.clamp(0, masked.shape[1] - 1)
            target_valid = target_valid & current_mask.gather(1, safe_target[:, None]).squeeze(1)
            if not bool(selected_valid.any()) and not bool(target_valid.any()):
                continue

            loss = logits.new_tensor(0.0)
            if bool(selected_valid.any()) and ppo_weight != 0.0:
                selected_log_prob = log_probs.gather(1, safe_selected[:, None]).squeeze(1)
                old_selected_log_prob = _old_selected_log_prob(
                    q_head_scores=batch["current_q_head_scores"],
                    candidate_mask=current_mask,
                    selected_index=selected,
                    mode=behavior_mode,
                    temperature=behavior_temperature,
                    torch=torch,
                )
                ratio = torch.exp(selected_log_prob[selected_valid] - old_selected_log_prob[selected_valid])
                adv = advantage.detach()[selected_valid]
                unclipped = ratio * adv
                clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * adv
                ppo_loss = -torch.minimum(unclipped, clipped).mean()
                loss = loss + float(ppo_weight) * ppo_loss
                ppo_losses.append(float(ppo_loss.detach().cpu()))

            value_loss = huber(values, target_values)
            loss = loss + float(value_weight) * value_loss
            value_losses.append(float(value_loss.detach().cpu()))

            if bool(target_valid.any()) and imitation_weight != 0.0:
                imitation_loss = torch.nn.functional.cross_entropy(masked[target_valid], target[target_valid])
                loss = loss + float(imitation_weight) * imitation_loss
                imitation_losses.append(float(imitation_loss.detach().cpu()))
                greedy = masked[target_valid].argmax(dim=1)
                target_correct += int((greedy == target[target_valid]).sum().detach().cpu())
                target_total += int(target_valid.sum().detach().cpu())

            entropy = _entropy(masked, current_mask, torch)
            loss = loss - float(entropy_weight) * entropy
            entropies.append(float(entropy.detach().cpu()))
            if valid_mass_weight != 0.0:
                mass_loss = _valid_mass_loss(logits, current_mask, torch)
                loss = loss + float(valid_mass_weight) * mass_loss
                valid_mass_losses.append(float(mass_loss.detach().cpu()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "xlron_graph_transformer_ppo_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(batches),
                            "loss_so_far": _mean(losses),
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
        )
        val_score = _validation_score(val_metrics, config)
        row = {
            "phase": "xlron_graph_transformer_ppo",
            "epoch": epoch,
            "train_loss": _mean(losses),
            "train_ppo_loss": _mean(ppo_losses),
            "train_imitation_loss": _mean(imitation_losses),
            "train_value_loss": _mean(value_losses),
            "train_entropy": _mean(entropies),
            "train_valid_mass_loss": _mean(valid_mass_losses),
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
                    "action_feature_dim": int(action_feature_dim),
                    "link_feature_dim": int(link_feature_dim),
                    "global_feature_dim": int(global_feature_dim),
                    "request_feature_dim": int(request_feature_dim),
                    "embedding_dim": int(embedding_dim),
                    "hidden_dim": int(embedding_dim),
                    "transformer_num_layers": int(num_layers),
                    "transformer_num_heads": int(num_heads),
                    "dropout": float(dropout),
                    "position_dim": int(position_dim),
                    "learning_target": target_mode,
                    "target_params": _target_params_for_metrics(target_params),
                    "policy": "xlron_graph_transformer_ppo",
                    "adaptation": "link-token Graph Transformer PPO over CSE2026 Top-N candidates",
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
        "stage": "train_xlron_graph_transformer_ppo",
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
        "ppo_clip_epsilon": float(clip_epsilon),
        "ppo_loss_weight": float(ppo_weight),
        "imitation_loss_weight": float(imitation_weight),
        "value_loss_weight": float(value_weight),
        "entropy_loss_weight": float(entropy_weight),
        "valid_mass_loss_coef": float(valid_mass_weight),
        "behavior_policy_mode": behavior_mode,
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
        ),
    }
    _write_json(run_path / "metrics.json", final_metrics)
    return final_metrics

