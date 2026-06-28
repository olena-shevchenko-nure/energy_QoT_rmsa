from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.ong_solver import Candidate, GnnCnnDqnOngSolver, SolverConfig

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
from .train_deeprmsa_a3c_windowed_online import (
    _device,
    _mean,
    _problem_reward,
    _select_episode_ids,
    _sum,
    _write_json,
)
from .train_dqn import _batch_to_arrays, _stack_state_arrays


@dataclass
class Top32PpoTransition:
    arrays: dict[str, np.ndarray]
    action_index: int
    reward: float
    done: bool
    old_masked_log_prob: float
    old_valid_mass: float
    value: float
    valid_count: int
    action_mode: str


def _model_forward(model: Any, tensors: dict[str, Any], edge_index: Any) -> tuple[Any, Any]:
    return model(
        link_features=tensors["link_features"],
        edge_index=edge_index,
        global_features=tensors["global_features"],
        request_features=tensors["request_features"],
        action_features=tensors["action_features"],
        route_link_mask=tensors["route_link_mask"],
        spectrum_tensors=tensors["spectrum_tensors"],
        route_basic_features=tensors["route_basic_features"],
        block_bounds=tensors["block_bounds"],
        candidate_mask=tensors["candidate_mask"],
    )


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1.0e9)


def _valid_mass_from_logits(logits: Any, mask: Any, torch: Any) -> Any:
    probs = torch.nn.functional.softmax(logits, dim=1)
    return (probs * mask.to(dtype=probs.dtype)).sum(dim=1).clamp_min(1.0e-8)


def _masked_entropy(masked_logits: Any, mask: Any, torch: Any) -> Any:
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
    probs = torch.exp(log_probs) * mask.to(dtype=log_probs.dtype)
    return -(probs * log_probs).sum(dim=1)


def _xlron_architecture_kwargs(config: ExperimentConfig, model_shapes: dict[str, int]) -> dict[str, Any]:
    architecture = _raw_str(config, "xlron_architecture", "link_transformer").strip().lower()
    full_default = architecture == "full"
    return {
        "architecture": architecture,
        "spectrum_channels": int(model_shapes.get("spectrum_channels", 6)),
        "route_basic_dim": int(model_shapes.get("route_basic_feature_dim", 2)),
        "candidate_transformer_layers": _raw_int(config, "xlron_candidate_transformer_layers", 0 if not full_default else 1),
        "candidate_transformer_heads": _raw_int(config, "xlron_candidate_transformer_heads", 4),
        "enable_spectrum_branch": _raw_bool(config, "xlron_enable_spectrum_branch", full_default),
        "enable_candidate_attention": _raw_bool(config, "xlron_enable_candidate_attention", full_default),
        "enable_base_relative_branch": _raw_bool(config, "xlron_enable_base_relative_branch", full_default),
        "enable_auxiliary_heads": _raw_bool(config, "xlron_enable_auxiliary_heads", False),
    }


def _gae_returns(
    transitions: list[Top32PpoTransition],
    *,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros((len(transitions),), dtype=np.float32)
    returns = np.zeros((len(transitions),), dtype=np.float32)
    next_value = float(bootstrap_value)
    running_advantage = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        not_done = 0.0 if bool(item.done) else 1.0
        delta = float(item.reward) + float(gamma) * next_value * not_done - float(item.value)
        running_advantage = delta + float(gamma) * float(gae_lambda) * not_done * running_advantage
        advantages[index] = float(running_advantage)
        returns[index] = float(running_advantage + float(item.value))
        next_value = float(item.value)
    return advantages, returns


def _state_value(
    *,
    model: Any,
    arrays: dict[str, np.ndarray],
    edge_index: Any,
    device: str,
    torch: Any,
) -> float:
    model.eval()
    with torch.no_grad():
        tensors = _stack_state_arrays([arrays], device, torch)
        _logits, value = _model_forward(model, tensors, edge_index)
    return float(value.detach().cpu().reshape(-1)[0])


def _select_action_index(
    *,
    model: Any,
    arrays: dict[str, np.ndarray],
    edge_index: Any,
    rng: np.random.Generator,
    device: str,
    torch: Any,
    config: ExperimentConfig,
    greedy: bool = False,
) -> tuple[int, dict[str, float | int | str]]:
    candidate_mask = arrays["candidate_mask"].astype(bool)
    valid = np.flatnonzero(candidate_mask)
    if valid.size == 0:
        return -1, {
            "mode": "block",
            "old_masked_log_prob": 0.0,
            "old_valid_mass": 0.0,
            "value": 0.0,
            "valid_count": 0,
        }

    model.eval()
    with torch.no_grad():
        tensors = _stack_state_arrays([arrays], device, torch)
        logits, value = _model_forward(model, tensors, edge_index)
        valid_mass = _valid_mass_from_logits(logits, tensors["candidate_mask"], torch)
        masked = _masked_logits(logits, tensors["candidate_mask"])
        masked_log_probs = torch.nn.functional.log_softmax(masked, dim=1)
        masked_probs = torch.nn.functional.softmax(masked, dim=1).detach().cpu().numpy().reshape(-1)
        logits_np = logits.detach().cpu().numpy().reshape(-1)

    if valid.size == 1:
        selected = int(valid[0])
        mode = "forced_single"
    elif greedy:
        selected = int(valid[int(np.argmax(logits_np[valid]))])
        mode = "greedy"
    else:
        epsilon = _raw_float(config, "epsilon_exploration", 0.0)
        if epsilon > 0.0 and rng.random() < epsilon:
            selected = int(rng.choice(valid))
            mode = "epsilon_random"
        else:
            valid_probs = masked_probs[valid].astype(np.float64)
            total = float(valid_probs.sum())
            if not np.isfinite(total) or total <= 0.0:
                selected = int(rng.choice(valid))
                mode = "sample_fallback"
            else:
                selected = int(rng.choice(valid, p=valid_probs / total))
                mode = "sample_masked"

    old_log_prob = float(masked_log_probs[0, int(selected)].detach().cpu())
    return int(selected), {
        "mode": mode,
        "old_masked_log_prob": old_log_prob,
        "old_valid_mass": float(valid_mass.detach().cpu().reshape(-1)[0]),
        "value": float(value.detach().cpu().reshape(-1)[0]),
        "valid_count": int(valid.size),
    }


def _ppo_update(
    *,
    model: Any,
    optimizer: Any,
    transitions: list[Top32PpoTransition],
    bootstrap_value: float,
    edge_index: Any,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_loss_weight: float,
    entropy_loss_weight: float,
    valid_mass_loss_coef: float,
    valid_mass_target: float,
    valid_mass_hard_gate: float,
    normalize_advantage: bool,
    grad_clip_norm: float,
    rng: np.random.Generator,
    device: str,
    torch: Any,
) -> dict[str, float]:
    if not transitions:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "valid_mass_loss": 0.0,
            "valid_mass": 0.0,
            "mean_advantage": 0.0,
            "actor_rows": 0.0,
            "updates": 0.0,
        }

    advantages_np, returns_np = _gae_returns(
        transitions,
        bootstrap_value=float(bootstrap_value),
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
    )
    if normalize_advantage and advantages_np.size > 1:
        std = float(np.std(advantages_np))
        if math.isfinite(std) and std > 1.0e-6:
            advantages_np = (advantages_np - float(np.mean(advantages_np))) / std

    arrays = [item.arrays for item in transitions]
    action_index_np = np.asarray([item.action_index for item in transitions], dtype=np.int64)
    old_log_prob_np = np.asarray([item.old_masked_log_prob for item in transitions], dtype=np.float32)
    valid_count_np = np.asarray([item.valid_count for item in transitions], dtype=np.int64)

    tensors = _stack_state_arrays(arrays, device, torch)
    action_index = torch.as_tensor(action_index_np, dtype=torch.long, device=device)
    old_log_prob = torch.as_tensor(old_log_prob_np, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)
    valid_count = torch.as_tensor(valid_count_np, dtype=torch.long, device=device)

    rows = np.arange(len(transitions), dtype=np.int64)
    ppo_epochs = max(1, int(ppo_epochs))
    minibatch_size = max(1, min(int(minibatch_size), len(transitions)))

    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    valid_mass_losses: list[float] = []
    valid_masses: list[float] = []
    actor_rows_total = 0
    updates = 0

    for _epoch in range(ppo_epochs):
        rng.shuffle(rows)
        for start in range(0, len(rows), minibatch_size):
            batch_rows_np = rows[start : start + minibatch_size]
            batch_rows = torch.as_tensor(batch_rows_np, dtype=torch.long, device=device)
            batch_tensors = {key: value[batch_rows] for key, value in tensors.items()}
            logits, values = _model_forward(model, batch_tensors, edge_index)
            mask = batch_tensors["candidate_mask"]
            safe_action = action_index[batch_rows].clamp(0, logits.shape[1] - 1)
            action_valid = (action_index[batch_rows] >= 0) & mask.gather(1, safe_action[:, None]).squeeze(1)
            actor_mask = action_valid & (valid_count[batch_rows] > 1)

            valid_mass = _valid_mass_from_logits(logits, mask, torch)
            valid_mass_loss = -torch.log(valid_mass).mean()
            value_loss = 0.5 * torch.nn.functional.mse_loss(values, returns[batch_rows])
            loss = float(value_loss_weight) * value_loss + float(valid_mass_loss_coef) * valid_mass_loss

            policy_loss = torch.zeros((), dtype=torch.float32, device=device)
            entropy = torch.zeros((), dtype=torch.float32, device=device)
            if bool(actor_mask.any()):
                masked_logits = _masked_logits(logits, mask)
                masked_log_probs = torch.nn.functional.log_softmax(masked_logits, dim=1)
                new_log_prob = masked_log_probs.gather(1, safe_action[:, None]).squeeze(1)
                ratio = torch.exp(new_log_prob[actor_mask] - old_log_prob[batch_rows][actor_mask])
                adv = advantages[batch_rows][actor_mask]
                unclipped = ratio * adv
                clipped = ratio.clamp(1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)) * adv

                current_mass = valid_mass[actor_mask].detach()
                damp = (current_mass / max(float(valid_mass_target), 1.0e-8)).clamp(0.0, 1.0)
                if float(valid_mass_hard_gate) > 0.0:
                    damp = torch.where(
                        current_mass >= float(valid_mass_hard_gate),
                        damp,
                        torch.zeros_like(damp),
                    )
                policy_loss = -(damp * torch.minimum(unclipped, clipped)).mean()

                masked = masked_logits[actor_mask]
                entropy_values = _masked_entropy(masked, mask[actor_mask], torch)
                entropy = (damp * entropy_values).mean()
                loss = loss + policy_loss - float(entropy_loss_weight) * entropy
                actor_rows_total += int(actor_mask.sum().detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            policy_losses.append(float(policy_loss.detach().cpu()))
            value_losses.append(float(value_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
            valid_mass_losses.append(float(valid_mass_loss.detach().cpu()))
            valid_masses.append(float(valid_mass.detach().mean().cpu()))
            updates += 1

    return {
        "loss": _mean(losses) or 0.0,
        "policy_loss": _mean(policy_losses) or 0.0,
        "value_loss": _mean(value_losses) or 0.0,
        "entropy": _mean(entropies) or 0.0,
        "valid_mass_loss": _mean(valid_mass_losses) or 0.0,
        "valid_mass": _mean(valid_masses) or 0.0,
        "mean_advantage": float(np.mean(advantages_np)) if advantages_np.size else 0.0,
        "actor_rows": float(actor_rows_total),
        "updates": float(updates),
    }


def _infer_model_shapes(
    *,
    traffic: pd.DataFrame,
    episode_ids: tuple[str, ...],
    run_path: Path,
    config: ExperimentConfig,
    cfg: SolverConfig,
) -> tuple[dict[str, int], np.ndarray]:
    solver = GnnCnnDqnOngSolver(cfg)
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"infer_top32_xlron_{episode_id}", episode)
        env = _make_env(
            episode_id=f"infer_top32_xlron_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        while True:
            batch = solver.candidate_batch(env)
            arrays = _batch_to_arrays(batch, cfg)
            if bool(arrays["candidate_mask"].any()):
                return (
                    {
                        "action_feature_dim": int(arrays["action_features"].shape[1]),
                        "link_feature_dim": int(arrays["link_features"].shape[1]),
                        "global_feature_dim": int(arrays["global_features"].shape[0]),
                        "request_feature_dim": int(arrays["request_features"].shape[0]),
                        "spectrum_channels": int(arrays["spectrum_tensors"].shape[1]),
                        "route_basic_feature_dim": int(arrays["route_basic_features"].shape[1]),
                    },
                    np.asarray(batch.state.edge_index, dtype=np.int64),
                )
            action = int(solver.adapter(env).block_action(env))
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if bool(terminated) or bool(truncated):
                break
    raise RuntimeError("Could not infer Top32 XLRON model shapes from online rollout")


def _rollout_greedy(
    *,
    model: Any,
    traffic: pd.DataFrame,
    episode_ids: tuple[str, ...],
    run_path: Path,
    config: ExperimentConfig,
    cfg: SolverConfig,
    edge_index: Any,
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
            arrays = _batch_to_arrays(batch, cfg)
            selected_index, _details = _select_action_index(
                model=model,
                arrays=arrays,
                edge_index=edge_index,
                rng=np.random.default_rng(int(config.seed)),
                device=device,
                torch=torch,
                config=config,
                greedy=True,
            )
            if selected_index < 0:
                candidate: Candidate | None = None
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
    total_reward = _sum([row["total_reward"] for row in rows])
    return {
        "episodes": int(len(rows)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": float(total_reward),
        "mean_episode_reward": _mean([row["total_reward"] for row in rows]),
        "mean_reward": float(total_reward / max(requests, 1)),
        "mean_selected_topn_index": _mean([row["mean_selected_topn_index"] for row in rows]),
        "mean_selected_energy_increment": _mean([row["mean_selected_energy_increment"] for row in rows]),
        "mean_selected_fragmentation_after": _mean([row["mean_selected_fragmentation_after"] for row in rows]),
        "episodes_detail": rows,
    }


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    config: ExperimentConfig,
    cfg: SolverConfig,
    model_shapes: dict[str, int],
    step: int,
    update_count: int,
    metrics: dict[str, Any],
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "top32_xlron_stabilized_ppo",
            "n_max": int(cfg.n_max),
            "action_feature_dim": int(model_shapes["action_feature_dim"]),
            "link_feature_dim": int(model_shapes["link_feature_dim"]),
            "global_feature_dim": int(model_shapes["global_feature_dim"]),
            "request_feature_dim": int(model_shapes["request_feature_dim"]),
            "embedding_dim": _raw_int(config, "transformer_embedding_size", _raw_int(config, "hidden_dim", 128)),
            "hidden_dim": _raw_int(config, "transformer_embedding_size", _raw_int(config, "hidden_dim", 128)),
            "transformer_num_layers": _raw_int(config, "transformer_num_layers", 2),
            "transformer_num_heads": _raw_int(config, "transformer_num_heads", 8),
            "dropout": _raw_float(config, "dropout", 0.05),
            "position_dim": _raw_int(config, "transformer_position_dim", 8),
            **_xlron_architecture_kwargs(config, model_shapes),
            "step": int(step),
            "update_count": int(update_count),
            "config": config.resolved,
            "solver_config": asdict(cfg),
            "metrics": metrics,
            "training_mode": "top32_online_stabilized_ppo",
        },
        path,
    )


def run_train_top32_xlron_stabilized_ppo(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_top32_xlron_stabilized_ppo requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)

    from cse2026.ong_solver.models import XlronGraphTransformerPpoNetwork, require_torch

    torch = require_torch()
    if XlronGraphTransformerPpoNetwork is None:
        raise RuntimeError("XlronGraphTransformerPpoNetwork is unavailable because PyTorch is not installed")
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
        max_episodes=_raw_int(config, "validation_max_episodes", 8),
        rng=np.random.default_rng(int(config.seed) + 17),
        shuffle=False,
    )
    model_shapes, edge_index_np = _infer_model_shapes(
        traffic=train_traffic,
        episode_ids=train_episode_ids,
        run_path=run_path,
        config=config,
        cfg=cfg,
    )
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)

    model = XlronGraphTransformerPpoNetwork(
        action_feature_dim=int(model_shapes["action_feature_dim"]),
        link_feature_dim=int(model_shapes["link_feature_dim"]),
        global_feature_dim=int(model_shapes["global_feature_dim"]),
        request_feature_dim=int(model_shapes["request_feature_dim"]),
        embedding_dim=_raw_int(config, "transformer_embedding_size", _raw_int(config, "hidden_dim", 128)),
        num_layers=_raw_int(config, "transformer_num_layers", 2),
        num_heads=_raw_int(config, "transformer_num_heads", 8),
        dropout=_raw_float(config, "dropout", 0.05),
        position_dim=_raw_int(config, "transformer_position_dim", 8),
        **_xlron_architecture_kwargs(config, model_shapes),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=_raw_float(config, "learning_rate", 2.0e-4),
        weight_decay=_raw_float(config, "weight_decay", 1.0e-4),
    )

    rollout_steps = max(1, _raw_int(config, "rollout_steps", int(config.batch_size)))
    minibatch_size = max(1, int(config.batch_size))
    ppo_epochs = max(1, _raw_int(config, "ppo_epochs", 4))
    gamma = _raw_float(config, "gamma", 0.995)
    gae_lambda = _raw_float(config, "gae_lambda", 0.95)
    clip_epsilon = _raw_float(config, "ppo_clip_epsilon", 0.20)
    value_loss_weight = _raw_float(config, "value_loss_weight", 0.50)
    entropy_loss_weight = _raw_float(config, "entropy_loss_weight", 0.01)
    valid_mass_loss_coef = _raw_float(config, "valid_mass_loss_coef", 0.001)
    valid_mass_target = _raw_float(config, "valid_mass_target", 0.80)
    valid_mass_hard_gate = _raw_float(config, "valid_mass_hard_gate", 0.05)
    normalize_advantage = _raw_bool(config, "normalize_advantage", True)
    grad_clip_norm = _raw_float(config, "grad_clip_norm", 1.0)
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
    best_path = run_path / "top32_xlron_stabilized_ppo_best.pt"
    final_path = run_path / "top32_xlron_stabilized_ppo_final.pt"

    for epoch in range(1, epochs + 1):
        epoch_episode_ids = train_episode_ids
        if _raw_bool(config, "online_shuffle_episodes_each_epoch", True):
            epoch_episode_ids = tuple(str(value) for value in rng.permutation(np.asarray(train_episode_ids, dtype=object)).tolist())
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
            buffer: list[Top32PpoTransition] = []
            requests = 0
            accepted = 0
            rewards: list[float] = []
            update_losses: list[float] = []
            selected_topn: list[float] = []
            valid_masses: list[float] = []
            action_modes: dict[str, int] = {}

            while True:
                batch = solver.candidate_batch(env)
                arrays = _batch_to_arrays(batch, cfg)
                selected_index, details = _select_action_index(
                    model=model,
                    arrays=arrays,
                    edge_index=edge_index,
                    rng=rng,
                    device=device,
                    torch=torch,
                    config=config,
                    greedy=False,
                )
                action_mode = str(details["mode"])
                action_modes[action_mode] = int(action_modes.get(action_mode, 0) + 1)
                valid_masses.append(float(details["old_valid_mass"]))
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
                    Top32PpoTransition(
                        arrays=arrays,
                        action_index=int(selected_index),
                        reward=float(reward),
                        done=bool(done),
                        old_masked_log_prob=float(details["old_masked_log_prob"]),
                        old_valid_mass=float(details["old_valid_mass"]),
                        value=float(details["value"]),
                        valid_count=int(details["valid_count"]),
                        action_mode=action_mode,
                    )
                )
                rewards.append(float(reward))
                requests += 1
                accepted += int(accepted_flag)

                should_update = len(buffer) >= rollout_steps or done
                if should_update:
                    bootstrap_value = 0.0
                    if not done:
                        next_batch = solver.candidate_batch(env)
                        bootstrap_value = _state_value(
                            model=model,
                            arrays=_batch_to_arrays(next_batch, cfg),
                            edge_index=edge_index,
                            device=device,
                            torch=torch,
                        )
                    metrics = _ppo_update(
                        model=model,
                        optimizer=optimizer,
                        transitions=buffer,
                        bootstrap_value=bootstrap_value,
                        edge_index=edge_index,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        ppo_epochs=ppo_epochs,
                        minibatch_size=minibatch_size,
                        clip_epsilon=clip_epsilon,
                        value_loss_weight=value_loss_weight,
                        entropy_loss_weight=entropy_loss_weight,
                        valid_mass_loss_coef=valid_mass_loss_coef,
                        valid_mass_target=valid_mass_target,
                        valid_mass_hard_gate=valid_mass_hard_gate,
                        normalize_advantage=normalize_advantage,
                        grad_clip_norm=grad_clip_norm,
                        rng=rng,
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
                            "rollout_size": int(len(buffer)),
                            "bootstrap_value": float(bootstrap_value),
                            **metrics,
                        }
                    )
                    buffer = []

                global_step += 1
                if done:
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
                "mean_valid_mass": _mean(valid_masses),
                "updates": int(update_count),
                "action_modes": dict(action_modes),
                "traffic_scenario": str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else "",
                "load_name": str(episode["load_name"].iloc[0]) if "load_name" in episode else "",
                "seed": int(episode["seed"].iloc[0]) if "seed" in episode else None,
            }
            train_rows.append(train_row)
            if episode_index % progress_every == 0 or episode_index == len(epoch_episode_ids):
                print(json.dumps({"event": "top32_xlron_stabilized_ppo_episode", **train_row}, sort_keys=True), flush=True)

        validation = _rollout_greedy(
            model=model,
            traffic=val_traffic,
            episode_ids=val_episode_ids,
            run_path=run_path,
            config=config,
            cfg=cfg,
            edge_index=edge_index,
            device=device,
            torch=torch,
            tag=f"val_e{epoch}",
        )
        validation_row = {"epoch": int(epoch), **{key: value for key, value in validation.items() if key != "episodes_detail"}}
        validation_rows.append(validation_row)
        print(json.dumps({"event": "top32_xlron_stabilized_ppo_validation", **validation_row}, sort_keys=True), flush=True)
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
                model_shapes=model_shapes,
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
        edge_index=edge_index,
        device=device,
        torch=torch,
        tag="val_final",
    )
    _save_checkpoint(
        path=final_path,
        model=model,
        config=config,
        cfg=cfg,
        model_shapes=model_shapes,
        step=global_step,
        update_count=update_count,
        metrics=final_validation,
        torch=torch,
    )
    pd.DataFrame(train_rows).to_csv(run_path / "online_episode_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_path / "online_update_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(run_path / "validation_metrics.csv", index=False)

    metrics = {
        "stage": "train_top32_xlron_stabilized_ppo",
        "training_mode": "top32_online_stabilized_ppo",
        "dataset_path": str(config.dataset_path),
        "ong_source_path": ong_source,
        "device": device,
        "solver_config": asdict(cfg),
        "train_split": train_split,
        "validation_split": val_split,
        "train_episodes": int(len(train_episode_ids)),
        "validation_episodes": int(len(val_episode_ids)),
        "model_shapes": dict(model_shapes),
        "hyperparameters": {
            "rollout_steps": int(rollout_steps),
            "minibatch_size": int(minibatch_size),
            "ppo_epochs": int(ppo_epochs),
            "gamma": float(gamma),
            "gae_lambda": float(gae_lambda),
            "clip_epsilon": float(clip_epsilon),
            "value_loss_weight": float(value_loss_weight),
            "entropy_loss_weight": float(entropy_loss_weight),
            "valid_mass_loss_coef": float(valid_mass_loss_coef),
            "valid_mass_target": float(valid_mass_target),
            "valid_mass_hard_gate": float(valid_mass_hard_gate),
            "normalize_advantage": bool(normalize_advantage),
            "grad_clip_norm": float(grad_clip_norm),
            "learning_rate": _raw_float(config, "learning_rate", 2.0e-4),
            "weight_decay": _raw_float(config, "weight_decay", 1.0e-4),
            "reward_mode": _raw_str(config, "online_reward_mode", "problem_shaped"),
            **_xlron_architecture_kwargs(config, model_shapes),
        },
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "global_step": int(global_step),
        "update_count": int(update_count),
        "best_validation": None if best_score is None else {"accepted": int(best_score[0]), "total_reward": float(best_score[1])},
        "final_validation": final_validation,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
