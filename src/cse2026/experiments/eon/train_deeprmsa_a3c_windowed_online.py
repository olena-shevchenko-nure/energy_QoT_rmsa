from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import Candidate, CandidateBatch, GnnCnnDqnOngSolver, SolverConfig
from cse2026.ong_solver.deeprmsa import deeprmsa_candidate_feature_columns, deeprmsa_features_from_batch

from ..config import ExperimentConfig
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _raw_str,
    _solver_config,
    _traffic_jsonl_for_episode,
)


@dataclass
class WindowTransition:
    candidate_features: np.ndarray
    context_features: np.ndarray
    candidate_mask: np.ndarray
    action_index: int
    reward: float
    done: bool


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_optional_path(config: ExperimentConfig, key: str) -> Path | None:
    value = config.resolved.get(key, config.raw.get(key))
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _device(config: ExperimentConfig, torch: Any) -> str:
    requested = str(config.resolved.get("device", config.device))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _sum(values: list[Any]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(np.sum(np.asarray(finite, dtype=np.float64)))


def _batch_features(batch: CandidateBatch, cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_features, context_features = deeprmsa_features_from_batch(
        batch,
        cfg.n_max,
        cfg.deeprmsa_prior_score,
    )
    return (
        candidate_features.astype(np.float32, copy=False),
        context_features.astype(np.float32, copy=False),
        batch.candidate_mask.astype(np.bool_, copy=False),
    )


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1.0e9)


def _state_value(
    *,
    model: Any,
    batch: CandidateBatch,
    cfg: SolverConfig,
    device: str,
    torch: Any,
) -> float:
    candidate_features, context_features, _mask = _batch_features(batch, cfg)
    model.eval()
    with torch.no_grad():
        _logits, value = model(
            torch.as_tensor(candidate_features[None, ...], dtype=torch.float32, device=device),
            torch.as_tensor(context_features[None, ...], dtype=torch.float32, device=device),
        )
    return float(value.detach().cpu().reshape(-1)[0])


def _discounted_returns(transitions: list[WindowTransition], *, gamma: float, bootstrap_value: float) -> np.ndarray:
    returns = np.zeros((len(transitions),), dtype=np.float32)
    running = float(bootstrap_value)
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        running = float(item.reward) + float(gamma) * running * (0.0 if item.done else 1.0)
        returns[index] = float(running)
    return returns


def _window_tensors(
    transitions: list[WindowTransition],
    *,
    returns: np.ndarray,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    return {
        "candidate_features": torch.as_tensor(
            np.stack([item.candidate_features for item in transitions], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "context_features": torch.as_tensor(
            np.stack([item.context_features for item in transitions], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "candidate_mask": torch.as_tensor(
            np.stack([item.candidate_mask for item in transitions], axis=0),
            dtype=torch.bool,
            device=device,
        ),
        "action_index": torch.as_tensor([item.action_index for item in transitions], dtype=torch.long, device=device),
        "return": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }


def _update_window(
    *,
    model: Any,
    optimizer: Any,
    transitions: list[WindowTransition],
    bootstrap_value: float,
    gamma: float,
    value_loss_weight: float,
    entropy_loss_weight: float,
    grad_clip_norm: float,
    device: str,
    torch: Any,
) -> dict[str, float]:
    if not transitions:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "mean_return": 0.0,
            "mean_value": 0.0,
            "action_rows": 0.0,
        }
    returns = _discounted_returns(transitions, gamma=gamma, bootstrap_value=bootstrap_value)
    batch = _window_tensors(transitions, returns=returns, device=device, torch=torch)
    model.train()
    logits, values = model(batch["candidate_features"], batch["context_features"])
    value_loss = 0.5 * torch.nn.functional.mse_loss(values, batch["return"])
    advantage = batch["return"] - values.detach()

    valid_action_rows = (batch["action_index"] >= 0) & batch["candidate_mask"].any(dim=1)
    policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy = torch.zeros((), dtype=torch.float32, device=device)
    if bool(valid_action_rows.any()):
        masked = _masked_logits(logits[valid_action_rows], batch["candidate_mask"][valid_action_rows])
        log_probs = torch.nn.functional.log_softmax(masked, dim=1)
        probs = torch.nn.functional.softmax(masked, dim=1)
        selected_log_probs = log_probs.gather(1, batch["action_index"][valid_action_rows, None]).squeeze(1)
        policy_loss = -(selected_log_probs * advantage[valid_action_rows]).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()

    loss = policy_loss + float(value_loss_weight) * value_loss - float(entropy_loss_weight) * entropy
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "mean_return": float(np.mean(returns)) if returns.size else 0.0,
        "mean_value": float(values.detach().mean().cpu()),
        "action_rows": float(valid_action_rows.sum().detach().cpu()),
    }


def _problem_reward(config: ExperimentConfig, candidate: Candidate | None, accepted: bool, env_reward: float) -> float:
    mode = _raw_str(config, "online_reward_mode", "problem_shaped").strip().lower()
    if mode in {"ong", "env", "environment"}:
        return float(env_reward)
    if mode not in {"problem_shaped", "problem-shaped"}:
        raise ValueError(f"Unsupported online_reward_mode: {mode}")
    if not accepted or candidate is None:
        return _raw_float(config, "block_penalty", -1.5)
    energy_norm = max(_raw_float(config, "energy_norm_w", 1200.0), 1.0e-9)
    delay_norm = max(_raw_float(config, "delay_bound_ms", 50.0), 1.0e-9)
    return float(
        _raw_float(config, "accepted_service_reward", 1.0)
        - _raw_float(config, "reward_energy_weight", 0.25) * (float(candidate.energy_increment) / energy_norm)
        - _raw_float(config, "reward_fragmentation_weight", 0.55) * float(candidate.fragmentation_after)
        + _raw_float(config, "reward_qot_margin_weight", 0.15) * float(candidate.qot_margin_norm)
        - _raw_float(config, "reward_delay_weight", 0.10) * (float(candidate.delay_ms) / delay_norm)
    )


def _epsilon_for_step(config: ExperimentConfig, step: int) -> float:
    start = _raw_float(config, "epsilon_start", 0.80)
    end = _raw_float(config, "epsilon_end", 0.05)
    decay_steps = max(1, _raw_int(config, "epsilon_decay_steps", 60000))
    fraction = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return float(start + fraction * (end - start))


def _select_action_index(
    *,
    model: Any,
    candidate_features: np.ndarray,
    context_features: np.ndarray,
    candidate_mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: str,
    torch: Any,
    config: ExperimentConfig,
) -> tuple[int, str]:
    valid = np.flatnonzero(candidate_mask.astype(bool))
    if valid.size == 0:
        return -1, "block"
    with torch.no_grad():
        logits, _value = model(
            torch.as_tensor(candidate_features[None, ...], dtype=torch.float32, device=device),
            torch.as_tensor(context_features[None, ...], dtype=torch.float32, device=device),
        )
        masked = _masked_logits(logits, torch.as_tensor(candidate_mask[None, :], dtype=torch.bool, device=device))
        probs = torch.nn.functional.softmax(masked, dim=1).detach().cpu().numpy().reshape(-1)
    if rng.random() < float(epsilon):
        exploration = _raw_str(config, "epsilon_exploration_mode", "policy_sample").strip().lower()
        if exploration == "random":
            return int(rng.choice(valid)), "epsilon_random"
        valid_probs = probs[valid].astype(np.float64)
        total = float(valid_probs.sum())
        if not math.isfinite(total) or total <= 0.0:
            return int(rng.choice(valid)), "epsilon_random_fallback"
        valid_probs = valid_probs / total
        return int(rng.choice(valid, p=valid_probs)), "epsilon_policy_sample"
    return int(valid[int(np.argmax(probs[valid]))]), "greedy"


def _rollout_greedy(
    *,
    model: Any,
    traffic: pd.DataFrame,
    episode_ids: tuple[str, ...],
    run_path: Path,
    config: ExperimentConfig,
    cfg: SolverConfig,
    device: str,
    torch: Any,
    tag: str,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    max_requests_per_episode = _raw_int(config, "validation_max_requests_per_episode", 0)
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(max_requests_per_episode).reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"{tag}_{episode_id}", episode)
        env = _make_env(
            episode_id=f"{tag}_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        solver = GnnCnnDqnOngSolver(cfg)
        requests = 0
        accepted = 0
        rewards: list[float] = []
        selected_topn: list[float] = []
        selected_energy: list[float] = []
        selected_fragmentation: list[float] = []
        while True:
            batch = solver.candidate_batch(env)
            candidate_features, context_features, candidate_mask = _batch_features(batch, cfg)
            selected_index, _mode = _select_action_index(
                model=model,
                candidate_features=candidate_features,
                context_features=context_features,
                candidate_mask=candidate_mask,
                epsilon=0.0,
                rng=np.random.default_rng(int(config.seed)),
                device=device,
                torch=torch,
                config=config,
            )
            if selected_index < 0:
                candidate = None
                action = int(solver.adapter(env).block_action(env))
            else:
                candidate = batch.topn[int(selected_index)]
                action = int(candidate.action)
                selected_topn.append(float(selected_index))
                selected_energy.append(float(candidate.energy_increment))
                selected_fragmentation.append(float(candidate.fragmentation_after))
            _observation, env_reward, terminated, truncated, info = env.step(int(action))
            accepted_flag = bool(info.get("accepted", False))
            rewards.append(_problem_reward(config, candidate, accepted_flag, float(env_reward)))
            requests += 1
            accepted += int(accepted_flag)
            if bool(terminated) or bool(truncated):
                break
        rows.append(
            {
                "episode_id": episode_id,
                "requests": int(requests),
                "accepted": int(accepted),
                "blocked": int(requests - accepted),
                "blocking_rate": float((requests - accepted) / max(requests, 1)),
                "total_reward": _sum(rewards),
                "mean_reward": _mean(rewards),
                "mean_selected_topn_index": _mean(selected_topn),
                "mean_selected_energy_increment": _mean(selected_energy),
                "mean_selected_fragmentation_after": _mean(selected_fragmentation),
                "traffic_scenario": str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else "",
                "load_name": str(episode["load_name"].iloc[0]) if "load_name" in episode else "",
                "seed": int(episode["seed"].iloc[0]) if "seed" in episode else None,
            }
        )
    requests = int(sum(row["requests"] for row in rows))
    accepted = int(sum(row["accepted"] for row in rows))
    return {
        "episodes": int(len(rows)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": _sum([row["total_reward"] for row in rows]),
        "mean_episode_reward": _mean([row["total_reward"] for row in rows]),
        "mean_reward": float(_sum([row["total_reward"] for row in rows]) / max(requests, 1)),
        "mean_selected_topn_index": _mean([row["mean_selected_topn_index"] for row in rows]),
        "mean_selected_energy_increment": _mean([row["mean_selected_energy_increment"] for row in rows]),
        "mean_selected_fragmentation_after": _mean([row["mean_selected_fragmentation_after"] for row in rows]),
        "episodes_detail": rows,
    }


def _select_episode_ids(
    traffic: pd.DataFrame,
    *,
    max_episodes: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> tuple[str, ...]:
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    if shuffle:
        episode_ids = tuple(str(value) for value in rng.permutation(np.asarray(episode_ids, dtype=object)).tolist())
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    return episode_ids


def _infer_feature_dims(
    *,
    traffic: pd.DataFrame,
    episode_ids: tuple[str, ...],
    run_path: Path,
    config: ExperimentConfig,
    cfg: SolverConfig,
) -> tuple[int, int]:
    solver = GnnCnnDqnOngSolver(cfg)
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"infer_{episode_id}", episode)
        env = _make_env(
            episode_id=f"infer_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        while True:
            batch = solver.candidate_batch(env)
            candidate_features, context_features, _mask = _batch_features(batch, cfg)
            if candidate_features.size and context_features.size:
                return int(candidate_features.shape[1]), int(context_features.shape[0])
            action = int(solver.adapter(env).block_action(env))
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if bool(terminated) or bool(truncated):
                break
    raise RuntimeError("Could not infer DeepRMSA feature dimensions from online rollout")


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    config: ExperimentConfig,
    cfg: SolverConfig,
    candidate_feature_dim: int,
    context_feature_dim: int,
    step: int,
    update_count: int,
    metrics: dict[str, Any],
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_max": int(cfg.n_max),
            "candidate_feature_dim": int(candidate_feature_dim),
            "context_feature_dim": int(context_feature_dim),
            "hidden_dim": _raw_int(config, "hidden_dim", 128),
            "layers": _raw_int(config, "deeprmsa_layers", 5),
            "dropout": _raw_float(config, "dropout", 0.05),
            "candidate_feature_columns": list(deeprmsa_candidate_feature_columns(cfg.deeprmsa_prior_score)),
            "deeprmsa_prior_score": str(cfg.deeprmsa_prior_score),
            "step": int(step),
            "update_count": int(update_count),
            "config": config.resolved,
            "solver_config": asdict(cfg),
            "metrics": metrics,
            "training_mode": "windowed_online_a3c",
        },
        path,
    )


def run_train_deeprmsa_a3c_windowed_online(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_deeprmsa_a3c_windowed_online requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)

    from cse2026.ong_solver.models import DeepRmsaA3CNetwork, require_torch

    torch = require_torch()
    if DeepRmsaA3CNetwork is None:
        raise RuntimeError("DeepRmsaA3CNetwork is unavailable because PyTorch is not installed")
    device = _device(config, torch)
    rng = np.random.default_rng(int(config.seed))
    cfg = _solver_config(config, neural=False)

    train_split = _raw_str(config, "online_train_split", "train")
    val_split = _raw_str(config, "validation_split", "val")
    train_traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{train_split}.parquet")
    val_traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{val_split}.parquet")
    train_episode_ids = _select_episode_ids(
        train_traffic,
        max_episodes=_raw_int(config, "online_max_episodes", 0),
        rng=rng,
        shuffle=_raw_bool(config, "online_shuffle_episodes", True),
    )
    val_episode_ids = _select_episode_ids(
        val_traffic,
        max_episodes=_raw_int(config, "validation_max_episodes", 16),
        rng=np.random.default_rng(int(config.seed) + 17),
        shuffle=False,
    )
    candidate_feature_dim, context_feature_dim = _infer_feature_dims(
        traffic=train_traffic,
        episode_ids=train_episode_ids,
        run_path=run_path,
        config=config,
        cfg=cfg,
    )

    model = DeepRmsaA3CNetwork(
        n_max=int(cfg.n_max),
        candidate_feature_dim=candidate_feature_dim,
        context_feature_dim=context_feature_dim,
        hidden_dim=_raw_int(config, "hidden_dim", 128),
        layers=_raw_int(config, "deeprmsa_layers", 5),
        dropout=_raw_float(config, "dropout", 0.05),
    ).to(device)

    optimizer_name = _raw_str(config, "optimizer", "rmsprop").strip().lower()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable,
            lr=_raw_float(config, "learning_rate", 1.0e-4),
            weight_decay=_raw_float(config, "weight_decay", 0.0),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            trainable,
            lr=_raw_float(config, "learning_rate", 1.0e-4),
            weight_decay=_raw_float(config, "weight_decay", 0.0),
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            trainable,
            lr=_raw_float(config, "learning_rate", 7.0e-5),
            alpha=_raw_float(config, "rmsprop_alpha", 0.99),
            eps=_raw_float(config, "rmsprop_eps", 1.0e-5),
            weight_decay=_raw_float(config, "weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    batch_size = max(1, int(config.batch_size))
    update_interval = max(1, _raw_int(config, "deeprmsa_update_interval", 2 * batch_size - 1))
    overlap = max(0, min(update_interval - 1, _raw_int(config, "deeprmsa_window_overlap", batch_size - 1)))
    gamma = _raw_float(config, "gamma", 0.95)
    value_loss_weight = _raw_float(config, "value_loss_weight", 0.5)
    entropy_loss_weight = _raw_float(config, "entropy_loss_weight", 0.01)
    grad_clip_norm = _raw_float(config, "grad_clip_norm", 40.0)
    epochs = max(1, _raw_int(config, "epochs", 3))
    patience = _raw_int(config, "patience", 0)
    max_requests_per_episode = _raw_int(config, "online_max_requests_per_episode", 0)
    progress_every = max(1, _raw_int(config, "progress_every_episodes", 5))

    train_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    global_step = 0
    update_count = 0
    best_score: tuple[int, float] | None = None
    best_epoch = 0
    stale_epochs = 0
    best_path = run_path / "deeprmsa_a3c_windowed_online_best.pt"
    final_path = run_path / "deeprmsa_a3c_windowed_online_final.pt"

    for epoch in range(1, epochs + 1):
        epoch_episode_ids = train_episode_ids
        if _raw_bool(config, "online_shuffle_episodes_each_epoch", True):
            epoch_episode_ids = tuple(
                str(value) for value in rng.permutation(np.asarray(train_episode_ids, dtype=object)).tolist()
            )
        for episode_index, episode_id in enumerate(epoch_episode_ids, start=1):
            episode = train_traffic[train_traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
            if max_requests_per_episode > 0:
                episode = episode.head(max_requests_per_episode).reset_index(drop=True)
            traffic_path = _traffic_jsonl_for_episode(run_path, f"train_e{epoch}_{episode_id}", episode)
            env = _make_env(
                episode_id=f"train_e{epoch}_{episode_id}",
                traffic_path=traffic_path,
                request_count=len(episode),
                seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
                config=config,
            )
            env.reset(seed=int(config.seed))
            solver = GnnCnnDqnOngSolver(cfg)
            buffer: list[WindowTransition] = []
            requests = 0
            accepted = 0
            rewards: list[float] = []
            update_losses: list[float] = []
            action_modes: dict[str, int] = {}
            selected_topn: list[float] = []

            while True:
                batch = solver.candidate_batch(env)
                candidate_features, context_features, candidate_mask = _batch_features(batch, cfg)
                epsilon = _epsilon_for_step(config, global_step)
                selected_index, action_mode = _select_action_index(
                    model=model,
                    candidate_features=candidate_features,
                    context_features=context_features,
                    candidate_mask=candidate_mask,
                    epsilon=epsilon,
                    rng=rng,
                    device=device,
                    torch=torch,
                    config=config,
                )
                action_modes[action_mode] = int(action_modes.get(action_mode, 0) + 1)
                if selected_index < 0:
                    candidate = None
                    action = int(solver.adapter(env).block_action(env))
                else:
                    candidate = batch.topn[int(selected_index)]
                    action = int(candidate.action)
                    selected_topn.append(float(selected_index))

                _observation, env_reward, terminated, truncated, info = env.step(int(action))
                done = bool(terminated) or bool(truncated)
                accepted_flag = bool(info.get("accepted", False))
                reward = _problem_reward(config, candidate, accepted_flag, float(env_reward))
                buffer.append(
                    WindowTransition(
                        candidate_features=candidate_features,
                        context_features=context_features,
                        candidate_mask=candidate_mask,
                        action_index=int(selected_index),
                        reward=float(reward),
                        done=bool(done),
                    )
                )
                rewards.append(float(reward))
                requests += 1
                accepted += int(accepted_flag)

                if len(buffer) >= update_interval:
                    bootstrap_value = 0.0
                    if not done:
                        next_batch = solver.candidate_batch(env)
                        bootstrap_value = _state_value(
                            model=model,
                            batch=next_batch,
                            cfg=cfg,
                            device=device,
                            torch=torch,
                        )
                    metrics = _update_window(
                        model=model,
                        optimizer=optimizer,
                        transitions=buffer,
                        bootstrap_value=bootstrap_value,
                        gamma=gamma,
                        value_loss_weight=value_loss_weight,
                        entropy_loss_weight=entropy_loss_weight,
                        grad_clip_norm=grad_clip_norm,
                        device=device,
                        torch=torch,
                    )
                    update_count += 1
                    update_losses.append(float(metrics["loss"]))
                    update_row = {
                        "epoch": int(epoch),
                        "episode_index": int(episode_index),
                        "episode_id": episode_id,
                        "step": int(global_step),
                        "update": int(update_count),
                        "window_size": int(len(buffer)),
                        "bootstrap_value": float(bootstrap_value),
                        **metrics,
                    }
                    update_rows.append(update_row)
                    if overlap > 0 and not done:
                        buffer = buffer[-overlap:]
                    else:
                        buffer = []

                global_step += 1
                if done:
                    if buffer:
                        metrics = _update_window(
                            model=model,
                            optimizer=optimizer,
                            transitions=buffer,
                            bootstrap_value=0.0,
                            gamma=gamma,
                            value_loss_weight=value_loss_weight,
                            entropy_loss_weight=entropy_loss_weight,
                            grad_clip_norm=grad_clip_norm,
                            device=device,
                            torch=torch,
                        )
                        update_count += 1
                        update_losses.append(float(metrics["loss"]))
                        update_rows.append(
                            {
                                "epoch": int(epoch),
                                "episode_index": int(episode_index),
                                "episode_id": episode_id,
                                "step": int(global_step),
                                "update": int(update_count),
                                "window_size": int(len(buffer)),
                                "bootstrap_value": 0.0,
                                **metrics,
                            }
                        )
                    break

            train_row = {
                "epoch": int(epoch),
                "episode_index": int(episode_index),
                "episode_id": episode_id,
                "requests": int(requests),
                "accepted": int(accepted),
                "blocked": int(requests - accepted),
                "blocking_rate": float((requests - accepted) / max(requests, 1)),
                "mean_reward": _mean(rewards),
                "mean_update_loss": _mean(update_losses),
                "mean_selected_topn_index": _mean(selected_topn),
                "epsilon_end": float(_epsilon_for_step(config, global_step)),
                "updates": int(update_count),
                "action_modes": dict(action_modes),
                "traffic_scenario": str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else "",
                "load_name": str(episode["load_name"].iloc[0]) if "load_name" in episode else "",
                "seed": int(episode["seed"].iloc[0]) if "seed" in episode else None,
            }
            train_rows.append(train_row)
            if episode_index % progress_every == 0 or episode_index == len(epoch_episode_ids):
                print(
                    json.dumps(
                        {
                            "event": "deeprmsa_windowed_online_episode",
                            **train_row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        validation = _rollout_greedy(
            model=model,
            traffic=val_traffic,
            episode_ids=val_episode_ids,
            run_path=run_path,
            config=config,
            cfg=cfg,
            device=device,
            torch=torch,
            tag=f"val_e{epoch}",
        )
        validation_row = {"epoch": int(epoch), **{key: value for key, value in validation.items() if key != "episodes_detail"}}
        validation_rows.append(validation_row)
        print(json.dumps({"event": "deeprmsa_windowed_online_validation", **validation_row}, sort_keys=True), flush=True)
        score = (int(validation["accepted"]), float(validation["total_reward"]))
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = int(epoch)
            stale_epochs = 0
            _save_checkpoint(
                path=best_path,
                model=model,
                config=config,
                cfg=cfg,
                candidate_feature_dim=candidate_feature_dim,
                context_feature_dim=context_feature_dim,
                step=global_step,
                update_count=update_count,
                metrics=validation,
                torch=torch,
            )
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break

    final_validation = _rollout_greedy(
        model=model,
        traffic=val_traffic,
        episode_ids=val_episode_ids,
        run_path=run_path,
        config=config,
        cfg=cfg,
        device=device,
        torch=torch,
        tag="val_final",
    )
    _save_checkpoint(
        path=final_path,
        model=model,
        config=config,
        cfg=cfg,
        candidate_feature_dim=candidate_feature_dim,
        context_feature_dim=context_feature_dim,
        step=global_step,
        update_count=update_count,
        metrics=final_validation,
        torch=torch,
    )
    pd.DataFrame(train_rows).to_csv(run_path / "online_episode_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_path / "online_update_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(run_path / "validation_metrics.csv", index=False)

    metrics = {
        "stage": "train_deeprmsa_a3c_windowed_online",
        "training_mode": "windowed_online_a3c",
        "dataset_path": str(config.dataset_path),
        "ong_source_path": ong_source,
        "device": device,
        "solver_config": asdict(cfg),
        "train_split": train_split,
        "validation_split": val_split,
        "train_episodes": int(len(train_episode_ids)),
        "validation_episodes": int(len(val_episode_ids)),
        "batch_size": int(batch_size),
        "update_interval": int(update_interval),
        "window_overlap": int(overlap),
        "gamma": float(gamma),
        "optimizer": optimizer_name,
        "learning_rate": _raw_float(config, "learning_rate", 7.0e-5),
        "reward_mode": _raw_str(config, "online_reward_mode", "problem_shaped"),
        "loss_weights": {
            "value_loss_weight": float(value_loss_weight),
            "entropy_loss_weight": float(entropy_loss_weight),
        },
        "feature_dims": {
            "candidate_feature_dim": int(candidate_feature_dim),
            "context_feature_dim": int(context_feature_dim),
            "n_max": int(cfg.n_max),
        },
        "updates": int(update_count),
        "steps": int(global_step),
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "best_score": None if best_score is None else {"accepted": int(best_score[0]), "total_reward": float(best_score[1])},
        "final_validation": final_validation,
        "validation_history": validation_rows,
        "train_summary": {
            "episodes": int(len(train_rows)),
            "requests": int(sum(row["requests"] for row in train_rows)),
            "accepted": int(sum(row["accepted"] for row in train_rows)),
            "blocked": int(sum(row["blocked"] for row in train_rows)),
            "blocking_rate": float(
                sum(row["blocked"] for row in train_rows) / max(sum(row["requests"] for row in train_rows), 1)
            ),
            "mean_reward": _mean([row["mean_reward"] for row in train_rows]),
            "mean_update_loss": _mean([row["mean_update_loss"] for row in train_rows]),
            "mean_selected_topn_index": _mean([row["mean_selected_topn_index"] for row in train_rows]),
        },
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
