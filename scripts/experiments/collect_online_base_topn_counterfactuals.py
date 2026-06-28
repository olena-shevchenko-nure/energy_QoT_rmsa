from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.dagger_tree_ranker import _simulate_candidate
from cse2026.experiments.eon.lookahead_override_features import OVERRIDE_FEATURE_NAMES, candidate_feature_matrix
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_int,
    _raw_str,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.train_dqn import _batch_to_arrays, _device, _stack_state_arrays
from cse2026.experiments.eon.tree_ranker_runtime import select_tree_base_index
from cse2026.ong_solver import GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax

from train_top32_xlron_full_dqn_distill import (
    _load_xlron_checkpoint_model,
    _resolve_cli_path,
    _xlron_forward,
)


NEURAL_STATE_KEYS = (
    "node_features",
    "link_features",
    "global_features",
    "request_features",
    "spectrum_tensors",
    "action_features",
    "route_link_mask",
    "route_basic_features",
    "block_bounds",
    "candidate_mask",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalized_policy(policy: str) -> str:
    return str(policy or "").strip().lower().replace("_", "-")


def _uses_dqn_policy(*policies: str) -> bool:
    return any(_normalized_policy(policy) in {"gnn-cnn-dqn"} for policy in policies)


def _uses_top32_xlron_policy(*policies: str) -> bool:
    return any(
        _normalized_policy(policy)
        in {"top32-xlron", "top32-xlron-stabilized-ppo", "xlron-top32", "student-xlron"}
        for policy in policies
    )


def _xlron_index(
    *,
    batch: Any,
    solver: GnnCnnDqnOngSolver,
    model: Any,
    device: str,
    torch: Any,
) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1
    arrays = _batch_to_arrays(batch, solver.config)
    tensors = _stack_state_arrays([arrays], device, torch)
    edge_index = torch.as_tensor(np.asarray(batch.state.edge_index, dtype=np.int64), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        raw_logits, _value = _xlron_forward(model, tensors, edge_index)
        scores = raw_logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
    selected = int(masked_argmax(scores, batch.candidate_mask))
    if selected < 0 or not bool(batch.candidate_mask[int(selected)]):
        return int(valid[0])
    return int(selected)


def _policy_index(
    *,
    batch: Any,
    solver: GnnCnnDqnOngSolver,
    policy: str,
    top32_xlron_model: Any | None,
    device: str,
    torch: Any | None,
) -> int:
    normalized = _normalized_policy(policy)
    if normalized == "gnn-cnn-dqn":
        return int(masked_argmax(solver.q_values(batch), batch.candidate_mask))
    if normalized in {"top32-xlron", "top32-xlron-stabilized-ppo", "xlron-top32", "student-xlron"}:
        if top32_xlron_model is None or torch is None:
            raise ValueError(f"{policy} requires --top32-xlron-checkpoint")
        return _xlron_index(
            batch=batch,
            solver=solver,
            model=top32_xlron_model,
            device=device,
            torch=torch,
        )
    return int(select_tree_base_index(batch, solver.config.n_max, normalized))


def _solver_rng_state(solver: GnnCnnDqnOngSolver) -> Any | None:
    rng = getattr(solver, "rng", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is None:
        return None
    return copy.deepcopy(bit_generator.state)


def _restore_solver_rng_state(solver: GnnCnnDqnOngSolver, state: Any | None) -> None:
    if state is None:
        return
    rng = getattr(solver, "rng", None)
    bit_generator = getattr(rng, "bit_generator", None)
    if bit_generator is not None:
        bit_generator.state = copy.deepcopy(state)


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _simulate_candidate_with_xlron(
    *,
    env: Any,
    candidate: Any,
    solver: GnnCnnDqnOngSolver,
    rollout_policy: str,
    horizon: int,
    top32_xlron_model: Any | None,
    device: str,
    torch: Any | None,
) -> dict[str, Any]:
    if not _uses_top32_xlron_policy(rollout_policy):
        return _simulate_candidate(
            env=env,
            candidate=candidate,
            solver=solver,
            rollout_policy=rollout_policy,
            horizon=int(horizon),
        )

    clone = copy.deepcopy(env)
    accepted = 0
    requests = 0
    env_rewards: list[float] = []
    selected_energy: list[float] = []
    selected_fragmentation: list[float] = []
    selected_qot: list[float] = []
    selected_delay: list[float] = []

    rng_state = _solver_rng_state(solver)
    try:
        action = int(candidate.action)
        selected_candidate: Any | None = candidate
        for step in range(max(1, int(horizon))):
            if step > 0:
                batch = solver.candidate_batch(clone)
                valid = np.flatnonzero(batch.candidate_mask.astype(bool))
                if valid.size == 0:
                    action = int(solver.adapter(clone).block_action(clone))
                    selected_candidate = None
                else:
                    selected_index = _policy_index(
                        batch=batch,
                        solver=solver,
                        policy=rollout_policy,
                        top32_xlron_model=top32_xlron_model,
                        device=device,
                        torch=torch,
                    )
                    if selected_index < 0 or not bool(batch.candidate_mask[int(selected_index)]):
                        selected_index = int(valid[0])
                    selected_candidate = batch.topn[int(selected_index)]
                    action = int(selected_candidate.action)
            if selected_candidate is not None:
                selected_energy.append(float(selected_candidate.energy_increment))
                selected_fragmentation.append(float(selected_candidate.fragmentation_after))
                selected_qot.append(float(selected_candidate.qot_margin_norm))
                selected_delay.append(float(selected_candidate.delay_ms))
            _observation, reward, terminated, truncated, info = clone.step(int(action))
            accepted += int(bool(info.get("accepted", False)))
            requests += 1
            env_rewards.append(float(reward))
            if bool(terminated) or bool(truncated):
                break
    finally:
        _restore_solver_rng_state(solver, rng_state)

    return {
        "future_requests": int(requests),
        "future_accepted": int(accepted),
        "future_blocked": int(requests - accepted),
        "future_env_reward": float(np.sum(np.asarray(env_rewards, dtype=np.float64))) if env_rewards else 0.0,
        "future_selected_count": int(len(selected_energy)),
        "future_energy_increment_sum": float(np.sum(np.asarray(selected_energy, dtype=np.float64))) if selected_energy else 0.0,
        "future_energy_increment_mean": _mean(selected_energy),
        "future_fragmentation_after_mean": _mean(selected_fragmentation),
        "future_qot_margin_norm_mean": _mean(selected_qot),
        "future_delay_ms_mean": _mean(selected_delay),
    }


def _candidate_pool_indices(batch: Any, *, base_index: int, top_k: int, candidate_pool: str) -> list[int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return []
    pool = str(candidate_pool or "energy_topk_hybrid").strip().lower().replace("-", "_")
    if pool in {"all", "all_topn"}:
        selected = [int(index) for index in valid]
    elif pool in {"energy_topk", "energy_topk_hybrid", "quick_topk_hybrid", "quick_top8_hybrid"}:
        selected_set: set[int] = set()
        if int(base_index) >= 0:
            selected_set.add(int(base_index))
        energy_order = sorted((int(index) for index in valid), key=lambda index: (float(batch.topn[index].energy_increment_norm), index))
        selected_set.update(energy_order[: max(1, min(int(top_k), len(energy_order)))])
        if pool != "energy_topk":
            selected_set.add(int(valid[0]))
            selected_set.add(min((int(index) for index in valid), key=lambda index: (float(batch.topn[index].fragmentation_after), index)))
            selected_set.add(
                max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].largest_free_block_after), -index))
            )
            selected_set.add(max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].qot_margin_norm), -index)))
        selected = [int(index) for index in valid if int(index) in selected_set]
    else:
        raise ValueError(f"Unsupported candidate_pool: {candidate_pool}")
    if int(base_index) >= 0 and int(base_index) not in selected:
        selected.append(int(base_index))
    return selected


def _secondary_delta(candidate_sim: dict[str, Any], base_sim: dict[str, Any], candidate: Any, base: Any) -> float:
    energy_delta = float(candidate_sim["future_energy_increment_sum"]) - float(base_sim["future_energy_increment_sum"])
    reward_delta = float(candidate_sim["future_env_reward"]) - float(base_sim["future_env_reward"])
    frag_delta = float(candidate.fragmentation_after) - float(base.fragmentation_after)
    qot_delta = float(candidate.qot_margin_norm) - float(base.qot_margin_norm)
    return float(0.05 * reward_delta - 0.001 * energy_delta - 0.25 * frag_delta + 0.10 * qot_delta)


def _parse_bucket_set(text: str) -> set[str]:
    result: set[str] = set()
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bucket must use traffic_scenario:load_name format, got {item!r}")
        result.add(item)
    return result


def _select_episode_ids(
    episodes: pd.DataFrame,
    *,
    max_episodes: int,
    episode_selection: str,
    include_buckets: str,
) -> tuple[str, ...]:
    selected_episodes = episodes.copy()
    bucket_filter = _parse_bucket_set(include_buckets)
    if bucket_filter and {"traffic_scenario", "load_name"}.issubset(selected_episodes.columns):
        bucket = selected_episodes["traffic_scenario"].astype(str) + ":" + selected_episodes["load_name"].astype(str)
        selected_episodes = selected_episodes[bucket.isin(bucket_filter)].reset_index(drop=True)

    episode_ids = tuple(str(value) for value in selected_episodes["episode_id"].tolist())
    if max_episodes <= 0:
        return episode_ids
    normalized = str(episode_selection).strip().lower().replace("-", "_")
    if normalized == "first":
        return episode_ids[: int(max_episodes)]
    if normalized != "stratified":
        raise ValueError(f"Unsupported episode selection mode: {episode_selection}")

    keys = ["traffic_scenario", "load_name"]
    if not all(key in selected_episodes.columns for key in keys):
        return episode_ids[: int(max_episodes)]
    groups = [
        [str(value) for value in group["episode_id"].tolist()]
        for _group_key, group in selected_episodes.groupby(keys, sort=False)
    ]
    selected: list[str] = []
    round_index = 0
    while len(selected) < int(max_episodes):
        added = False
        for group in groups:
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) >= int(max_episodes):
                    break
        if not added:
            break
        round_index += 1
    return tuple(selected)


def collect_online_base_topn(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    base_policy: str,
    rollout_policy: str,
    top32_xlron_checkpoint: Path | None,
    candidate_pool: str,
    top_k: int,
    lookahead_horizon: int,
    episode_selection: str,
    include_buckets: str,
    collection_stride: int,
    max_groups_per_episode: int,
    max_collected_groups: int,
    progress_every: int,
    save_neural_states: bool,
) -> dict[str, Any]:
    _add_ong_source_path(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _raw_str(config, "rollout_split", "test")
    max_episodes = _raw_int(config, "max_episodes", 0)
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episodes = traffic.drop_duplicates("episode_id").reset_index(drop=True)
    episode_ids = _select_episode_ids(
        episodes,
        max_episodes=int(max_episodes),
        episode_selection=str(episode_selection),
        include_buckets=str(include_buckets),
    )

    use_neural = _uses_dqn_policy(base_policy, rollout_policy)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=use_neural))
    top32_xlron_model = None
    torch = None
    device = "cpu"
    if _uses_top32_xlron_policy(base_policy, rollout_policy):
        if top32_xlron_checkpoint is None:
            raise ValueError("top32_xlron policy requires --top32-xlron-checkpoint")
        from cse2026.ong_solver.models import require_torch

        torch = require_torch()
        device = _device(config, torch)
        top32_xlron_model, _checkpoint = _load_xlron_checkpoint_model(top32_xlron_checkpoint, device=device, torch=torch)
        top32_xlron_model.eval()
        for parameter in top32_xlron_model.parameters():
            parameter.requires_grad_(False)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    group_sizes: list[int] = []
    metadata: list[dict[str, Any]] = []
    neural_states: dict[str, list[np.ndarray]] = {key: [] for key in NEURAL_STATE_KEYS}
    neural_group_ids: list[int] = []
    neural_base_indices: list[int] = []
    neural_edge_index: np.ndarray | None = None
    neural_accepted_delta: list[np.ndarray] = []
    neural_secondary_delta: list[np.ndarray] = []
    neural_target_delta: list[np.ndarray] = []
    neural_label_mask: list[np.ndarray] = []
    group_id = 0
    visited_requests = 0
    valid_states = 0
    collected_states = 0
    no_candidate_requests = 0
    skipped_small_pool = 0

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(int(max_requests_per_episode)).reset_index(drop=True)
        if episode.empty:
            continue
        traffic_path = _traffic_jsonl_for_episode(output_dir, f"online_base_topn_{episode_id}", episode)
        env = _make_env(
            episode_id=f"online_base_topn_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        episode_collected = 0
        episode_valid_seen = 0

        for position, request in episode.iterrows():
            batch = solver.candidate_batch(env)
            valid = np.flatnonzero(batch.candidate_mask.astype(bool))
            if valid.size == 0:
                no_candidate_requests += 1
                action = int(solver.adapter(env).block_action(env))
                _observation, _reward, terminated, truncated, _info = env.step(action)
                visited_requests += 1
                if bool(terminated) or bool(truncated):
                    break
                continue

            valid_states += 1
            episode_valid_seen += 1
            base_index = _policy_index(
                batch=batch,
                solver=solver,
                policy=base_policy,
                top32_xlron_model=top32_xlron_model,
                device=device,
                torch=torch,
            )
            if base_index < 0:
                action = int(solver.adapter(env).block_action(env))
                _observation, _reward, terminated, truncated, _info = env.step(action)
                visited_requests += 1
                if bool(terminated) or bool(truncated):
                    break
                continue

            should_collect = (episode_valid_seen - 1) % max(int(collection_stride), 1) == 0
            if max_groups_per_episode > 0 and episode_collected >= int(max_groups_per_episode):
                should_collect = False
            if max_collected_groups > 0 and collected_states >= int(max_collected_groups):
                should_collect = False

            if should_collect:
                candidate_indices = _candidate_pool_indices(
                    batch,
                    base_index=int(base_index),
                    top_k=int(top_k),
                    candidate_pool=candidate_pool,
                )
                features, kept_indices = candidate_feature_matrix(
                    batch=batch,
                    candidate_indices=candidate_indices,
                    n_max=solver.config.n_max,
                    reference_index=int(base_index),
                )
                if features.shape[0] >= 2 and int(base_index) in kept_indices:
                    base_position = kept_indices.index(int(base_index))
                    state_arrays = _batch_to_arrays(batch, solver.config) if save_neural_states else None
                    sims: list[dict[str, Any]] = []
                    for candidate_index in kept_indices:
                        sims.append(
                            _simulate_candidate_with_xlron(
                                env=env,
                                candidate=batch.topn[int(candidate_index)],
                                solver=solver,
                                rollout_policy=rollout_policy,
                                horizon=int(lookahead_horizon),
                                top32_xlron_model=top32_xlron_model,
                                device=device,
                                torch=torch,
                            )
                        )
                    base_sim = sims[int(base_position)]
                    base_candidate = batch.topn[int(base_index)]
                    accepted_delta_values: list[int] = []
                    secondary_delta_values: list[float] = []
                    target_delta_values: list[float] = []
                    for feature_row, candidate_index, simulation in zip(features, kept_indices, sims):
                        candidate = batch.topn[int(candidate_index)]
                        accepted_delta = int(simulation["future_accepted"]) - int(base_sim["future_accepted"])
                        reward_delta = float(simulation["future_env_reward"]) - float(base_sim["future_env_reward"])
                        energy_delta = float(simulation["future_energy_increment_sum"]) - float(base_sim["future_energy_increment_sum"])
                        secondary_delta = _secondary_delta(simulation, base_sim, candidate, base_candidate)
                        target = float(accepted_delta if accepted_delta != 0 else secondary_delta)
                        accepted_delta_values.append(int(accepted_delta))
                        secondary_delta_values.append(float(secondary_delta))
                        target_delta_values.append(float(target))
                        rows.append(np.asarray(feature_row, dtype=np.float32))
                        targets.append(float(target))
                        metadata.append(
                            {
                                "split": str(split),
                                "group_id": int(group_id),
                                "episode_id": str(episode_id),
                                "request_id": int(request["request_id"]),
                                "position": int(position),
                                "traffic_scenario": str(request.get("traffic_scenario", "")),
                                "load_name": str(request.get("load_name", "")),
                                "seed": int(request["seed"]) if "seed" in request else None,
                                "base_policy": str(base_policy),
                                "counterfactual_rollout_policy": str(rollout_policy),
                                "top32_xlron_checkpoint": None
                                if top32_xlron_checkpoint is None
                                else str(top32_xlron_checkpoint),
                                "candidate_pool": str(candidate_pool),
                                "candidate_pool_top_k": int(top_k),
                                "lookahead_horizon": int(lookahead_horizon),
                                "valid_candidates": int(valid.size),
                                "candidate_pool_size": int(len(kept_indices)),
                                "candidate_index": int(candidate_index),
                                "base_index": int(base_index),
                                "is_base": bool(int(candidate_index) == int(base_index)),
                                "target": float(target),
                                "accepted_delta_vs_base": int(accepted_delta),
                                "secondary_delta_vs_base": float(secondary_delta),
                                "future_env_reward_delta_vs_base": float(reward_delta),
                                "future_energy_increment_delta_vs_base": float(energy_delta),
                                "future_accepted": int(simulation["future_accepted"]),
                                "base_future_accepted": int(base_sim["future_accepted"]),
                                "future_blocked": int(simulation["future_blocked"]),
                                "future_requests": int(simulation["future_requests"]),
                                "future_env_reward": float(simulation["future_env_reward"]),
                                "base_future_env_reward": float(base_sim["future_env_reward"]),
                                "future_energy_increment_sum": float(simulation["future_energy_increment_sum"]),
                                "energy_increment": float(candidate.energy_increment),
                                "base_energy_increment": float(base_candidate.energy_increment),
                                "fragmentation_after": float(candidate.fragmentation_after),
                                "base_fragmentation_after": float(base_candidate.fragmentation_after),
                                "qot_margin_norm": float(candidate.qot_margin_norm),
                                "base_qot_margin_norm": float(base_candidate.qot_margin_norm),
                            }
                        )
                    group_sizes.append(int(features.shape[0]))
                    if save_neural_states and state_arrays is not None:
                        n_max = int(solver.config.n_max)
                        accepted_vector = np.full((n_max,), np.nan, dtype=np.float32)
                        secondary_vector = np.full((n_max,), np.nan, dtype=np.float32)
                        target_vector = np.full((n_max,), np.nan, dtype=np.float32)
                        label_mask = np.zeros((n_max,), dtype=np.bool_)
                        for candidate_index, accepted_delta, secondary_delta, target_delta in zip(
                            kept_indices,
                            accepted_delta_values,
                            secondary_delta_values,
                            target_delta_values,
                        ):
                            index = int(candidate_index)
                            if 0 <= index < n_max:
                                accepted_vector[index] = float(accepted_delta)
                                secondary_vector[index] = float(secondary_delta)
                                target_vector[index] = float(target_delta)
                                label_mask[index] = True
                        for key in NEURAL_STATE_KEYS:
                            neural_states[key].append(np.asarray(state_arrays[key]))
                        neural_group_ids.append(int(group_id))
                        neural_base_indices.append(int(base_index))
                        neural_accepted_delta.append(accepted_vector)
                        neural_secondary_delta.append(secondary_vector)
                        neural_target_delta.append(target_vector)
                        neural_label_mask.append(label_mask)
                        if neural_edge_index is None:
                            neural_edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
                    group_id += 1
                    collected_states += 1
                    episode_collected += 1
                    if progress_every > 0 and (
                        collected_states == 1 or collected_states % int(progress_every) == 0
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "output_dir": str(output_dir),
                                    "episode_id": str(episode_id),
                                    "visited_requests": int(visited_requests),
                                    "valid_states": int(valid_states),
                                    "collected_states": int(collected_states),
                                    "max_collected_groups": int(max_collected_groups),
                                    "lookahead_horizon": int(lookahead_horizon),
                                },
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    skipped_small_pool += 1

            action = int(batch.topn[int(base_index)].action)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            visited_requests += 1
            if max_collected_groups > 0 and collected_states >= int(max_collected_groups):
                break
            if bool(terminated) or bool(truncated):
                break
        if max_collected_groups > 0 and collected_states >= int(max_collected_groups):
            break

    metadata_df = pd.DataFrame(metadata)
    x = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    group_sizes_array = np.asarray(group_sizes, dtype=np.int32)
    metadata_df.to_csv(output_dir / "online_base_topn_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "online_base_topn_examples.npz",
        features=x,
        targets=y,
        group_sizes=group_sizes_array,
        feature_names=np.asarray(OVERRIDE_FEATURE_NAMES, dtype=object),
    )
    neural_states_path = output_dir / "online_base_topn_neural_states.npz"
    if save_neural_states:
        neural_payload: dict[str, np.ndarray] = {
            "group_ids": np.asarray(neural_group_ids, dtype=np.int64),
            "base_index": np.asarray(neural_base_indices, dtype=np.int64),
            "accepted_delta_vs_base": np.asarray(neural_accepted_delta, dtype=np.float32),
            "secondary_delta_vs_base": np.asarray(neural_secondary_delta, dtype=np.float32),
            "target_delta": np.asarray(neural_target_delta, dtype=np.float32),
            "label_mask": np.asarray(neural_label_mask, dtype=np.bool_),
            "edge_index": np.zeros((2, 0), dtype=np.int64) if neural_edge_index is None else neural_edge_index,
        }
        for key, values in neural_states.items():
            if values:
                dtype = np.bool_ if key == "candidate_mask" else np.float32
                neural_payload[key] = np.stack(values, axis=0).astype(dtype)
            else:
                neural_payload[key] = np.asarray([], dtype=np.bool_ if key == "candidate_mask" else np.float32)
        np.savez_compressed(neural_states_path, **neural_payload)
    if not metadata_df.empty:
        non_base = metadata_df[~metadata_df["is_base"].astype(bool)]
        accepted = non_base["accepted_delta_vs_base"].astype(float) if not non_base.empty else pd.Series(dtype=float)
        positive_groups = int(metadata_df.groupby("group_id")["accepted_delta_vs_base"].max().gt(0).sum())
        negative_groups = int(metadata_df.groupby("group_id")["accepted_delta_vs_base"].min().lt(0).sum())
        label_summary = {
            "groups": int(metadata_df["group_id"].nunique()),
            "rows": int(len(metadata_df)),
            "non_base_rows": int(len(non_base)),
            "win_rows": int((accepted > 0).sum()) if not accepted.empty else 0,
            "loss_rows": int((accepted < 0).sum()) if not accepted.empty else 0,
            "tie_rows": int((accepted == 0).sum()) if not accepted.empty else 0,
            "groups_with_win": int(positive_groups),
            "groups_with_loss": int(negative_groups),
            "non_base_total_delta": int(round(float(accepted.sum()))) if not accepted.empty else 0,
        }
    else:
        label_summary = {
            "groups": 0,
            "rows": 0,
            "non_base_rows": 0,
            "win_rows": 0,
            "loss_rows": 0,
            "tie_rows": 0,
            "groups_with_win": 0,
            "groups_with_loss": 0,
            "non_base_total_delta": 0,
        }
    summary = {
        "output_dir": str(output_dir),
        "dataset_path": str(config.dataset_path),
        "split": str(split),
        "episodes": list(episode_ids),
        "visited_requests": int(visited_requests),
        "valid_states": int(valid_states),
        "collected_states": int(collected_states),
        "no_candidate_requests": int(no_candidate_requests),
        "skipped_small_pool": int(skipped_small_pool),
        "base_policy": str(base_policy),
        "counterfactual_rollout_policy": str(rollout_policy),
        "top32_xlron_checkpoint": None if top32_xlron_checkpoint is None else str(top32_xlron_checkpoint),
        "dqn_checkpoint": solver.config.checkpoint_path,
        "candidate_pool": str(candidate_pool),
        "candidate_pool_top_k": int(top_k),
        "lookahead_horizon": int(lookahead_horizon),
        "episode_selection": str(episode_selection),
        "include_buckets": sorted(_parse_bucket_set(str(include_buckets))),
        "collection_stride": int(collection_stride),
        "max_groups_per_episode": int(max_groups_per_episode),
        "max_collected_groups": int(max_collected_groups),
        "save_neural_states": bool(save_neural_states),
        "neural_states_path": str(neural_states_path) if save_neural_states else None,
        "feature_names": list(OVERRIDE_FEATURE_NAMES),
        "label_summary": label_summary,
    }
    _write_json(output_dir / "online_base_topn_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect base-trajectory Top-N counterfactual examples vs energy-aware base.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-policy", default="energy-aware-ksp-bm-ff")
    parser.add_argument("--rollout-policy", default="")
    parser.add_argument("--top32-xlron-checkpoint", default="")
    parser.add_argument("--candidate-pool", default="energy_topk_hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--lookahead-horizon", type=int, default=8)
    parser.add_argument("--episode-selection", choices=("first", "stratified"), default="first")
    parser.add_argument(
        "--include-buckets",
        default="",
        help="Comma-separated traffic_scenario:load_name bucket filter applied before episode selection.",
    )
    parser.add_argument("--collection-stride", type=int, default=15)
    parser.add_argument("--max-groups-per-episode", type=int, default=25)
    parser.add_argument("--max-collected-groups", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-save-neural-states", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    base_policy = str(args.base_policy)
    rollout_policy = str(args.rollout_policy or base_policy)
    top32_xlron_checkpoint = _resolve_cli_path(str(args.top32_xlron_checkpoint or ""))
    summary = collect_online_base_topn(
        config=config,
        output_dir=Path(args.output_dir),
        base_policy=base_policy,
        rollout_policy=rollout_policy,
        top32_xlron_checkpoint=top32_xlron_checkpoint,
        candidate_pool=str(args.candidate_pool),
        top_k=int(args.top_k),
        lookahead_horizon=int(args.lookahead_horizon),
        episode_selection=str(args.episode_selection),
        include_buckets=str(args.include_buckets),
        collection_stride=int(args.collection_stride),
        max_groups_per_episode=int(args.max_groups_per_episode),
        max_collected_groups=int(args.max_collected_groups),
        progress_every=int(args.progress_every),
        save_neural_states=not bool(args.no_save_neural_states),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
