from __future__ import annotations

import json
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
    _discounted_returns,
    _epsilon_for_step,
    _mean,
    _problem_reward,
    _select_episode_ids,
    _sum,
    _write_json,
)
from .train_dqn import _batch_to_arrays, _stack_state_arrays


@dataclass
class FullEncoderWindowTransition:
    arrays: dict[str, np.ndarray]
    action_index: int
    reward: float
    done: bool


def _masked_logits(logits: Any, mask: Any) -> Any:
    return logits.masked_fill(~mask, -1.0e9)


def _model_forward(model: Any, tensors: dict[str, Any], edge_index: Any) -> tuple[Any, Any]:
    return model(
        node_features=tensors["node_features"],
        link_features=tensors["link_features"],
        global_features=tensors["global_features"],
        edge_index=edge_index,
        request_features=tensors["request_features"],
        spectrum_tensors=tensors["spectrum_tensors"],
        action_features=tensors["action_features"],
        route_link_mask=tensors["route_link_mask"],
        route_basic_features=tensors["route_basic_features"],
        block_bounds=tensors["block_bounds"],
        candidate_mask=tensors["candidate_mask"],
    )


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


def _update_window(
    *,
    model: Any,
    optimizer: Any,
    transitions: list[FullEncoderWindowTransition],
    edge_index: Any,
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
    tensors = _stack_state_arrays([item.arrays for item in transitions], device, torch)
    action_index = torch.as_tensor([item.action_index for item in transitions], dtype=torch.long, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

    model.train()
    logits, values = _model_forward(model, tensors, edge_index)
    value_loss = 0.5 * torch.nn.functional.mse_loss(values, return_tensor)
    advantage = return_tensor - values.detach()

    candidate_mask = tensors["candidate_mask"]
    valid_action_rows = (action_index >= 0) & candidate_mask.any(dim=1)
    policy_loss = torch.zeros((), dtype=torch.float32, device=device)
    entropy = torch.zeros((), dtype=torch.float32, device=device)
    if bool(valid_action_rows.any()):
        masked = _masked_logits(logits[valid_action_rows], candidate_mask[valid_action_rows])
        log_probs = torch.nn.functional.log_softmax(masked, dim=1)
        probs = torch.nn.functional.softmax(masked, dim=1)
        selected_log_probs = log_probs.gather(1, action_index[valid_action_rows, None]).squeeze(1)
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


def _select_action_index(
    *,
    model: Any,
    arrays: dict[str, np.ndarray],
    edge_index: Any,
    epsilon: float,
    rng: np.random.Generator,
    device: str,
    torch: Any,
    config: ExperimentConfig,
) -> tuple[int, str]:
    candidate_mask = arrays["candidate_mask"].astype(bool)
    valid = np.flatnonzero(candidate_mask)
    if valid.size == 0:
        return -1, "block"
    with torch.no_grad():
        tensors = _stack_state_arrays([arrays], device, torch)
        logits, _value = _model_forward(model, tensors, edge_index)
        masked = _masked_logits(logits, tensors["candidate_mask"])
        probs = torch.nn.functional.softmax(masked, dim=1).detach().cpu().numpy().reshape(-1)
    if rng.random() < float(epsilon):
        exploration = _raw_str(config, "epsilon_exploration_mode", "policy_sample").strip().lower()
        if exploration == "random":
            return int(rng.choice(valid)), "epsilon_random"
        valid_probs = probs[valid].astype(np.float64)
        total = float(valid_probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            return int(rng.choice(valid)), "epsilon_random_fallback"
        valid_probs = valid_probs / total
        return int(rng.choice(valid, p=valid_probs)), "epsilon_policy_sample"
    return int(valid[int(np.argmax(probs[valid]))]), "greedy"


def _infer_model_shapes(
    *,
    traffic: pd.DataFrame,
    episode_ids: tuple[str, ...],
    run_path: Path,
    config: ExperimentConfig,
    cfg: SolverConfig,
) -> tuple[int, np.ndarray]:
    solver = GnnCnnDqnOngSolver(cfg)
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"infer_gnncnn_a3c_{episode_id}", episode)
        env = _make_env(
            episode_id=f"infer_gnncnn_a3c_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        while True:
            batch = solver.candidate_batch(env)
            arrays = _batch_to_arrays(batch, cfg)
            if arrays["action_features"].size:
                return int(arrays["action_features"].shape[1]), np.asarray(batch.state.edge_index, dtype=np.int64)
            action = int(solver.adapter(env).block_action(env))
            _observation, _reward, terminated, truncated, _info = env.step(action)
            if bool(terminated) or bool(truncated):
                break
    raise RuntimeError("Could not infer GNN+CNN A3C model shapes from online rollout")


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
            selected_index, _mode = _select_action_index(
                model=model,
                arrays=arrays,
                edge_index=edge_index,
                epsilon=0.0,
                rng=np.random.default_rng(int(config.seed)),
                device=device,
                torch=torch,
                config=config,
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
    action_feature_dim: int,
    step: int,
    update_count: int,
    metrics: dict[str, Any],
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "gnn_cnn_a3c",
            "n_max": int(cfg.n_max),
            "action_feature_dim": int(action_feature_dim),
            "hidden_dim": _raw_int(config, "hidden_dim", 128),
            "step": int(step),
            "update_count": int(update_count),
            "config": config.resolved,
            "solver_config": asdict(cfg),
            "metrics": metrics,
            "training_mode": "full_encoder_windowed_online_a3c",
        },
        path,
    )


def run_train_gnn_cnn_a3c_windowed_online(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_gnn_cnn_a3c_windowed_online requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)

    from cse2026.ong_solver.models import GnnCnnA3CNetwork, require_torch

    torch = require_torch()
    if GnnCnnA3CNetwork is None:
        raise RuntimeError("GnnCnnA3CNetwork is unavailable because PyTorch is not installed")
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
    action_feature_dim, edge_index_np = _infer_model_shapes(
        traffic=train_traffic,
        episode_ids=train_episode_ids,
        run_path=run_path,
        config=config,
        cfg=cfg,
    )
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)

    model = GnnCnnA3CNetwork(
        action_feature_dim=action_feature_dim,
        hidden_dim=_raw_int(config, "hidden_dim", 128),
    ).to(device)

    optimizer_name = _raw_str(config, "optimizer", "adamw").strip().lower()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable,
            lr=_raw_float(config, "learning_rate", 5.0e-5),
            weight_decay=_raw_float(config, "weight_decay", 1.0e-4),
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            trainable,
            lr=_raw_float(config, "learning_rate", 5.0e-5),
            alpha=_raw_float(config, "rmsprop_alpha", 0.99),
            eps=_raw_float(config, "rmsprop_eps", 1.0e-5),
            weight_decay=_raw_float(config, "weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    batch_size = max(1, int(config.batch_size))
    update_interval = max(1, _raw_int(config, "a3c_update_interval", 2 * batch_size - 1))
    overlap = max(0, min(update_interval - 1, _raw_int(config, "a3c_window_overlap", batch_size - 1)))
    gamma = _raw_float(config, "gamma", 0.95)
    value_loss_weight = _raw_float(config, "value_loss_weight", 0.5)
    entropy_loss_weight = _raw_float(config, "entropy_loss_weight", 0.01)
    grad_clip_norm = _raw_float(config, "grad_clip_norm", 10.0)
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
    best_path = run_path / "gnn_cnn_a3c_windowed_online_best.pt"
    final_path = run_path / "gnn_cnn_a3c_windowed_online_final.pt"

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
            buffer: list[FullEncoderWindowTransition] = []
            requests = 0
            accepted = 0
            rewards: list[float] = []
            update_losses: list[float] = []
            action_modes: dict[str, int] = {}
            selected_topn: list[float] = []

            while True:
                batch = solver.candidate_batch(env)
                arrays = _batch_to_arrays(batch, cfg)
                epsilon = _epsilon_for_step(config, global_step)
                selected_index, action_mode = _select_action_index(
                    model=model,
                    arrays=arrays,
                    edge_index=edge_index,
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
                    FullEncoderWindowTransition(
                        arrays=arrays,
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
                            arrays=_batch_to_arrays(next_batch, cfg),
                            edge_index=edge_index,
                            device=device,
                            torch=torch,
                        )
                    metrics = _update_window(
                        model=model,
                        optimizer=optimizer,
                        transitions=buffer,
                        edge_index=edge_index,
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
                    update_rows.append(
                        {
                            "epoch": int(epoch),
                            "episode_index": int(episode_index),
                            "episode_id": episode_id,
                            "step": int(global_step),
                            "update": int(update_count),
                            "window_size": int(len(buffer)),
                            "bootstrap_value": float(bootstrap_value),
                            **metrics,
                        }
                    )
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
                            edge_index=edge_index,
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
                print(json.dumps({"event": "gnn_cnn_a3c_windowed_online_episode", **train_row}, sort_keys=True), flush=True)

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
        print(json.dumps({"event": "gnn_cnn_a3c_windowed_online_validation", **validation_row}, sort_keys=True), flush=True)
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
                action_feature_dim=action_feature_dim,
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
        action_feature_dim=action_feature_dim,
        step=global_step,
        update_count=update_count,
        metrics=final_validation,
        torch=torch,
    )
    pd.DataFrame(train_rows).to_csv(run_path / "online_episode_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_path / "online_update_metrics.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(run_path / "validation_metrics.csv", index=False)

    metrics = {
        "stage": "train_gnn_cnn_a3c_windowed_online",
        "training_mode": "full_encoder_windowed_online_a3c",
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
        "learning_rate": _raw_float(config, "learning_rate", 5.0e-5),
        "reward_mode": _raw_str(config, "online_reward_mode", "problem_shaped"),
        "loss_weights": {
            "value_loss_weight": float(value_loss_weight),
            "entropy_loss_weight": float(entropy_loss_weight),
        },
        "model_shapes": {
            "action_feature_dim": int(action_feature_dim),
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
        },
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
