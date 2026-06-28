from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import Candidate, CandidateBatch, SolverConfig
from cse2026.ong_solver.common import pad_q_scores

from ..config import ExperimentConfig
from .lookahead_override_features import select_q_head_index
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _safe_dqn_index,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from .train_dqn import (
    _batch_to_arrays,
    _build_model,
    _model_forward,
    _score_values,
    _stack_state_arrays,
)


@dataclass
class OnlineTransition:
    current_arrays: dict[str, np.ndarray]
    next_arrays: dict[str, np.ndarray]
    current_q_head_scores: np.ndarray
    next_q_head_scores: np.ndarray
    selected_index: int
    q_head_index: int
    reward: float
    done: bool
    next_available: bool


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


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _q_head_scores(batch: CandidateBatch, n_max: int) -> np.ndarray:
    if not batch.topn:
        return np.full((n_max,), -np.inf, dtype=np.float32)
    scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
    return pad_q_scores(scores, n_max).astype(np.float32)


def _q_values_np(
    *,
    model: Any,
    batch: CandidateBatch,
    cfg: SolverConfig,
    device: str,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    torch: Any,
) -> np.ndarray:
    arrays = _batch_to_arrays(batch, cfg)
    tensors = _stack_state_arrays([arrays], device, torch)
    q_head = torch.as_tensor(_q_head_scores(batch, cfg.n_max)[None, :], dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(batch.state.edge_index, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        raw = _model_forward(model, tensors, edge_index)
        scored = _score_values(
            raw,
            q_head,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        )
    return scored.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _problem_reward(config: ExperimentConfig, candidate: Candidate | None, accepted: bool, env_reward: float) -> float:
    mode = str(config.resolved.get("online_reward_mode", config.raw.get("online_reward_mode", "problem_shaped")))
    if mode == "ong":
        return float(env_reward)
    if not accepted or candidate is None:
        return _raw_float(config, "block_penalty", -1.5)
    energy_norm = max(_raw_float(config, "energy_norm_w", 1200.0), 1e-9)
    delay_norm = max(_raw_float(config, "delay_bound_ms", 50.0), 1e-9)
    return float(
        _raw_float(config, "accepted_service_reward", 1.0)
        - _raw_float(config, "reward_energy_weight", 0.25) * (candidate.energy_increment / energy_norm)
        - _raw_float(config, "reward_fragmentation_weight", 0.55) * candidate.fragmentation_after
        + _raw_float(config, "reward_qot_margin_weight", 0.15) * candidate.qot_margin_norm
        - _raw_float(config, "reward_delay_weight", 0.10) * (candidate.delay_ms / delay_norm)
    )


def _sample_tensors(
    samples: list[OnlineTransition],
    *,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    return {
        "current": _stack_state_arrays([item.current_arrays for item in samples], device, torch),
        "next": _stack_state_arrays([item.next_arrays for item in samples], device, torch),
        "current_q_head_scores": torch.as_tensor(
            np.stack([item.current_q_head_scores for item in samples], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "next_q_head_scores": torch.as_tensor(
            np.stack([item.next_q_head_scores for item in samples], axis=0),
            dtype=torch.float32,
            device=device,
        ),
        "selected_index": torch.as_tensor([item.selected_index for item in samples], dtype=torch.long, device=device),
        "q_head_index": torch.as_tensor([item.q_head_index for item in samples], dtype=torch.long, device=device),
        "reward": torch.as_tensor([item.reward for item in samples], dtype=torch.float32, device=device),
        "done": torch.as_tensor([item.done for item in samples], dtype=torch.bool, device=device),
        "next_available": torch.as_tensor([item.next_available for item in samples], dtype=torch.bool, device=device),
    }


def _online_update(
    *,
    online: Any,
    target: Any,
    optimizer: Any,
    replay: list[OnlineTransition],
    rng: np.random.Generator,
    edge_index: Any,
    batch_size: int,
    device: str,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    gamma: float,
    td_loss_weight: float,
    imitation_loss_weight: float,
    residual_l2_weight: float,
    torch: Any,
) -> dict[str, float]:
    sample_size = min(int(batch_size), len(replay))
    indices = rng.choice(np.arange(len(replay), dtype=np.int64), size=sample_size, replace=False)
    batch = _sample_tensors([replay[int(index)] for index in indices], device=device, torch=torch)
    online.train()

    current_raw = _model_forward(online, batch["current"], edge_index)
    current_scores = _score_values(
        current_raw,
        batch["current_q_head_scores"],
        score_mode=score_mode,
        residual_scale=residual_scale,
        residual_delta_clip=residual_delta_clip,
        torch=torch,
    ).masked_fill(~batch["current"]["candidate_mask"], -1e9)
    selected_q = current_scores.gather(1, batch["selected_index"][:, None]).squeeze(1)

    with torch.no_grad():
        next_mask = batch["next"]["candidate_mask"] & batch["next_available"][:, None] & (~batch["done"])[:, None]
        next_online_raw = _model_forward(online, batch["next"], edge_index)
        next_online_scores = _score_values(
            next_online_raw,
            batch["next_q_head_scores"],
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        ).masked_fill(~next_mask, -1e9)
        next_action = next_online_scores.argmax(dim=1)
        next_target_raw = _model_forward(target, batch["next"], edge_index)
        next_target_scores = _score_values(
            next_target_raw,
            batch["next_q_head_scores"],
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        ).masked_fill(~next_mask, -1e9)
        next_q = next_target_scores.gather(1, next_action[:, None]).squeeze(1)
        next_q = torch.where(next_mask.any(dim=1), next_q, torch.zeros_like(next_q))
        expected = batch["reward"] + float(gamma) * next_q

    huber = torch.nn.SmoothL1Loss()
    td_loss = huber(selected_q, expected)
    loss = float(td_loss_weight) * td_loss

    valid_q_head = batch["q_head_index"] >= 0
    imitation_loss = torch.zeros((), dtype=torch.float32, device=device)
    if bool(valid_q_head.any()) and imitation_loss_weight > 0.0:
        imitation_loss = torch.nn.functional.cross_entropy(current_scores[valid_q_head], batch["q_head_index"][valid_q_head])
        loss = loss + float(imitation_loss_weight) * imitation_loss

    residual_l2 = current_raw.masked_select(batch["current"]["candidate_mask"]).square().mean()
    if residual_l2_weight > 0.0:
        loss = loss + float(residual_l2_weight) * residual_l2

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "td_loss": float(td_loss.detach().cpu()),
        "td_abs_error": float((selected_q.detach() - expected).abs().mean().cpu()),
        "imitation_loss": float(imitation_loss.detach().cpu()),
        "residual_l2_loss": float(residual_l2.detach().cpu()),
    }


def _save_checkpoint(
    *,
    path: Path,
    online: Any,
    target: Any,
    config: ExperimentConfig,
    cfg: SolverConfig,
    step: int,
    episode_count: int,
    metrics: dict[str, Any],
) -> None:
    import torch

    torch.save(
        {
            "model_state_dict": online.state_dict(),
            "target_model_state_dict": target.state_dict(),
            "step": int(step),
            "episode_count": int(episode_count),
            "config": config.resolved,
            "solver_config": {
                "n_max": int(cfg.n_max),
                "hidden_dim": int(cfg.hidden_dim),
                "device": cfg.device,
                "q_score_mode": str(config.resolved.get("q_score_mode", config.raw.get("q_score_mode", "q_head_residual"))),
                "residual_scale": _raw_float(config, "residual_scale", 1.0),
                "residual_delta_clip": _raw_float(config, "residual_delta_clip", 0.10),
            },
            "metrics": metrics,
        },
        path,
    )


def run_train_dqn_online(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_dqn_online requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)

    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    cfg = _solver_config(config, neural=False)
    online, pretrained = _build_model(config, device, torch)
    target = copy.deepcopy(online).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = [parameter for parameter in online.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=_raw_float(config, "learning_rate", 5e-5),
        weight_decay=_raw_float(config, "weight_decay", 1e-4),
    )
    score_mode = str(config.resolved.get("q_score_mode", config.raw.get("q_score_mode", "q_head_residual")))
    residual_scale = _raw_float(config, "residual_scale", 1.0)
    residual_delta_clip = _raw_float(config, "residual_delta_clip", 0.10)
    gamma = _raw_float(config, "gamma", 0.95)
    td_loss_weight = _raw_float(config, "td_loss_weight", 1.0)
    imitation_loss_weight = _raw_float(config, "imitation_loss_weight", 0.20)
    residual_l2_weight = _raw_float(config, "residual_l2_weight", 0.05)
    replay_capacity = _raw_int(config, "replay_capacity", 5000)
    replay_min_size = _raw_int(config, "replay_min_size", 256)
    update_every = max(1, _raw_int(config, "online_update_every", 4))
    gradient_steps = max(1, _raw_int(config, "gradient_steps_per_update", 1))
    target_update_interval = max(1, _raw_int(config, "target_update_interval", 200))
    max_updates = _raw_int(config, "online_max_updates", 0)
    batch_size = int(config.batch_size)
    rng = np.random.default_rng(int(config.seed))

    split = str(config.resolved.get("online_train_split", config.raw.get("online_train_split", "train")))
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    if _raw_bool(config, "online_shuffle_episodes", True):
        episode_ids = tuple(str(value) for value in rng.permutation(np.asarray(episode_ids, dtype=object)).tolist())
    max_episodes = _raw_int(config, "online_max_episodes", 0)
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    max_requests_per_episode = _raw_int(config, "online_max_requests_per_episode", 0)

    replay: list[OnlineTransition] = []
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    global_step = 0
    update_count = 0
    edge_index_tensor: Any | None = None

    for episode_index, episode_id in enumerate(episode_ids, start=1):
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(max_requests_per_episode).reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"online_{episode_id}", episode)
        env = _make_env(
            episode_id=f"online_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        solver = __import__("cse2026.ong_solver", fromlist=["GnnCnnDqnOngSolver"]).GnnCnnDqnOngSolver(cfg)

        accepted = 0
        requests = 0
        override_requests = 0
        override_applied = 0
        rewards: list[float] = []
        env_rewards: list[float] = []
        update_losses: list[float] = []

        while True:
            batch = solver.candidate_batch(env)
            if edge_index_tensor is None:
                edge_index_tensor = torch.as_tensor(batch.state.edge_index, dtype=torch.long, device=device)
            if not batch.has_real_candidates:
                action = solver.adapter(env).block_action(env)
                selected = None
                selected_index = -1
                did_override = False
            else:
                q_values = _q_values_np(
                    model=online,
                    batch=batch,
                    cfg=cfg,
                    device=device,
                    score_mode=score_mode,
                    residual_scale=residual_scale,
                    residual_delta_clip=residual_delta_clip,
                    torch=torch,
                )
                selected_index, did_override, _margin = _safe_dqn_index(
                    batch=batch,
                    q_values=q_values,
                    config=config,
                    n_max=cfg.n_max,
                )
                selected = batch.topn[int(selected_index)]
                action = int(selected.action)
                override_requests += 1
                override_applied += int(bool(did_override))

            _observation, env_reward, terminated, truncated, info = env.step(int(action))
            accepted_flag = bool(info.get("accepted", False))
            reward = _problem_reward(config, selected, accepted_flag, float(env_reward))
            rewards.append(float(reward))
            env_rewards.append(float(env_reward))
            requests += 1
            accepted += int(accepted_flag)
            done = bool(terminated) or bool(truncated)

            if selected is not None and selected_index >= 0:
                current_arrays = _batch_to_arrays(batch, cfg)
                current_q_head = _q_head_scores(batch, cfg.n_max)
                q_head_index = select_q_head_index(batch, cfg.n_max)
                if done:
                    next_arrays = current_arrays
                    next_q_head = np.full((cfg.n_max,), -np.inf, dtype=np.float32)
                    next_available = False
                else:
                    next_batch = solver.candidate_batch(env)
                    next_arrays = _batch_to_arrays(next_batch, cfg)
                    next_q_head = _q_head_scores(next_batch, cfg.n_max)
                    next_available = bool(next_batch.has_real_candidates)
                replay.append(
                    OnlineTransition(
                        current_arrays=current_arrays,
                        next_arrays=next_arrays,
                        current_q_head_scores=current_q_head,
                        next_q_head_scores=next_q_head,
                        selected_index=int(selected_index),
                        q_head_index=int(q_head_index),
                        reward=float(reward),
                        done=bool(done),
                        next_available=bool(next_available),
                    )
                )
                if len(replay) > replay_capacity:
                    del replay[: len(replay) - replay_capacity]

            if len(replay) >= replay_min_size and global_step % update_every == 0 and (max_updates <= 0 or update_count < max_updates):
                for _ in range(gradient_steps):
                    metrics = _online_update(
                        online=online,
                        target=target,
                        optimizer=optimizer,
                        replay=replay,
                        rng=rng,
                        edge_index=edge_index_tensor,
                        batch_size=batch_size,
                        device=device,
                        score_mode=score_mode,
                        residual_scale=residual_scale,
                        residual_delta_clip=residual_delta_clip,
                        gamma=gamma,
                        td_loss_weight=td_loss_weight,
                        imitation_loss_weight=imitation_loss_weight,
                        residual_l2_weight=residual_l2_weight,
                        torch=torch,
                    )
                    update_count += 1
                    update_losses.append(float(metrics["loss"]))
                    update_rows.append({"step": int(global_step), "update": int(update_count), **metrics})
                    if update_count % target_update_interval == 0:
                        target.load_state_dict(online.state_dict())
                    if max_updates > 0 and update_count >= max_updates:
                        break

            global_step += 1
            if done:
                break

        row = {
            "episode_index": int(episode_index),
            "episode_id": episode_id,
            "requests": int(requests),
            "accepted": int(accepted),
            "blocked": int(requests - accepted),
            "blocking_rate": float((requests - accepted) / max(requests, 1)),
            "mean_reward": _mean(rewards),
            "mean_env_reward": _mean(env_rewards),
            "mean_update_loss": _mean(update_losses),
            "override_requests": int(override_requests),
            "override_applied": int(override_applied),
            "override_rate": float(override_applied / max(override_requests, 1)),
            "replay_size": int(len(replay)),
            "updates": int(update_count),
            "traffic_scenario": str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else "",
            "load_name": str(episode["load_name"].iloc[0]) if "load_name" in episode else "",
            "seed": int(episode["seed"].iloc[0]) if "seed" in episode else None,
        }
        episode_rows.append(row)
        print(json.dumps({"event": "online_episode", **row}, sort_keys=True), flush=True)

    target.load_state_dict(online.state_dict())
    final_path = run_path / "dqn_online_final.pt"
    train_summary = {
        "episodes": int(len(episode_rows)),
        "requests": int(sum(row["requests"] for row in episode_rows)),
        "accepted": int(sum(row["accepted"] for row in episode_rows)),
        "blocked": int(sum(row["blocked"] for row in episode_rows)),
        "blocking_rate": float(sum(row["blocked"] for row in episode_rows) / max(sum(row["requests"] for row in episode_rows), 1)),
        "mean_episode_reward": _mean([row["mean_reward"] for row in episode_rows if row["mean_reward"] is not None]),
        "override_requests": int(sum(row["override_requests"] for row in episode_rows)),
        "override_applied": int(sum(row["override_applied"] for row in episode_rows)),
        "override_rate": float(
            sum(row["override_applied"] for row in episode_rows) / max(sum(row["override_requests"] for row in episode_rows), 1)
        ),
        "updates": int(update_count),
        "replay_size": int(len(replay)),
    }
    metrics = {
        "stage": "train_dqn_online",
        "mode": "constrained_q_head_safe_override",
        "dataset_path": str(config.dataset_path),
        "split": split,
        "ong_source_path": ong_source,
        "device": device,
        "solver_config": asdict(cfg),
        "pretrained": pretrained,
        "initial_dqn_checkpoint": str(_resolve_optional_path(config, "initial_dqn_checkpoint")),
        "final_checkpoint": str(final_path),
        "score_mode": score_mode,
        "residual_scale": float(residual_scale),
        "residual_delta_clip": float(residual_delta_clip),
        "gamma": float(gamma),
        "replay_capacity": int(replay_capacity),
        "replay_min_size": int(replay_min_size),
        "online_update_every": int(update_every),
        "gradient_steps_per_update": int(gradient_steps),
        "target_update_interval": int(target_update_interval),
        "online_max_updates": int(max_updates),
        "loss_weights": {
            "td_loss_weight": float(td_loss_weight),
            "imitation_loss_weight": float(imitation_loss_weight),
            "residual_l2_weight": float(residual_l2_weight),
        },
        "dqn_safe_gate": {
            "margin": _raw_float(config, "dqn_gate_margin", 0.005),
            "fragmentation_slack": _raw_float(config, "dqn_gate_fragmentation_slack", 0.02),
            "small_gap_slack": _raw_float(config, "dqn_gate_small_gap_slack", 0.02),
            "lmax_slack_slots": _raw_int(config, "dqn_gate_lmax_slack_slots", 4),
            "qot_margin_slack": _raw_float(config, "dqn_gate_qot_margin_slack", 0.08),
            "energy_slack_w": _raw_float(config, "dqn_gate_energy_slack_w", 80.0),
            "delay_slack_ms": _raw_float(config, "dqn_gate_delay_slack_ms", 1.0),
        },
        "train_summary": train_summary,
        "episodes": episode_rows,
        "updates": update_rows,
    }
    _save_checkpoint(
        path=final_path,
        online=online,
        target=target,
        config=config,
        cfg=cfg,
        step=global_step,
        episode_count=len(episode_rows),
        metrics=train_summary,
    )
    pd.DataFrame(episode_rows).to_csv(run_path / "online_episode_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(run_path / "online_update_metrics.csv", index=False)
    _write_json(run_path / "metrics.json", metrics)
    return metrics
