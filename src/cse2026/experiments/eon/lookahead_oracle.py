from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.ong_solver import Candidate, GnnCnnDqnOngSolver, SolverConfig
from cse2026.ong_solver.common import masked_argmax, pad_q_scores

from ..config import ExperimentConfig
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_float,
    _raw_int,
    _raw_list,
    _traffic_jsonl_for_episode,
)
from .tree_ranker_runtime import select_tree_base_index


def _solver_config(config: ExperimentConfig) -> SolverConfig:
    return SolverConfig(
        n_max=_raw_int(config, "n_max", 32),
        random_starts_per_route=_raw_int(config, "random_starts_per_route", 2),
        rng_seed=int(config.seed),
        use_neural=False,
        device=str(config.resolved.get("device", config.device)),
        hidden_dim=_raw_int(config, "hidden_dim", 128),
        slot_capacity_gbps_at_1bpshz=_raw_float(config, "slot_capacity_gbps_at_1bpshz", 12.5),
        guard_band_slots=_raw_int(config, "guard_band_slots", 1),
        mean_holding_time=_raw_float(config, "mean_holding_time", 14.0),
        min_useful_slot_width=_raw_int(config, "min_useful_slot_width", 4),
        delay_bound_ms=_raw_float(config, "delay_bound_ms", 50.0),
        span_length_km=_raw_float(config, "span_length_km", 80.0),
        amplifier_power_w=_raw_float(config, "amplifier_power_w", 20.0),
        roadm_switch_power_w=_raw_float(config, "roadm_switch_power_w", 5.0),
        transponder_power_w=_raw_float(config, "transponder_power_w", 100.0),
        energy_norm_w=_raw_float(config, "energy_norm_w", 1200.0),
        max_route_length_norm_km=_raw_float(config, "max_route_length_norm_km", 6000.0),
    )


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _select_q_head(batch: Any, n_max: int) -> int:
    if not np.asarray(batch.candidate_mask, dtype=bool).any():
        return -1
    scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
    return masked_argmax(pad_q_scores(scores, n_max), batch.candidate_mask)


def _select_j_total(batch: Any) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    return -1 if valid.size == 0 else int(valid[0])


def _select_rollout_index(batch: Any, policy: str, n_max: int) -> int:
    policy_name = str(policy).strip().lower().replace("_", "-")
    if policy_name in {"j-total-heuristic", "j-total", "jtotal"}:
        return _select_j_total(batch)
    if policy_name in {"q-head-heuristic", "q-head", "qhead"}:
        return _select_q_head(batch, n_max)
    return select_tree_base_index(batch, n_max, policy)


def _candidate_indices(batch: Any) -> list[int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return []
    return [int(index) for index in valid]


def _simulate_first_action(
    *,
    env: Any,
    first_action: int,
    solver: GnnCnnDqnOngSolver,
    rollout_policy: str,
    horizon: int,
) -> dict[str, Any]:
    clone = copy.deepcopy(env)
    total_reward = 0.0
    accepted = 0
    requests = 0
    selected_energy: list[float] = []
    selected_fragmentation: list[float] = []
    selected_qot: list[float] = []
    selected_delay: list[float] = []

    action = int(first_action)
    for step in range(max(1, int(horizon))):
        candidate: Candidate | None = None
        if step > 0:
            batch = solver.candidate_batch(clone)
            index = _select_rollout_index(batch, rollout_policy, solver.config.n_max)
            if index < 0:
                action = int(solver.adapter(clone).block_action(clone))
            else:
                candidate = batch.topn[index]
                action = int(candidate.action)
        observation, reward, terminated, truncated, info = clone.step(action)
        del observation
        requests += 1
        total_reward += float(reward)
        accepted += int(bool(info.get("accepted", False)))
        if candidate is not None:
            selected_energy.append(float(candidate.energy_increment))
            selected_fragmentation.append(float(candidate.fragmentation_after))
            selected_qot.append(float(candidate.qot_margin_norm))
            selected_delay.append(float(candidate.delay_ms))
        if bool(terminated) or bool(truncated):
            break
    return {
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "reward": float(total_reward),
        "reward_per_request": float(total_reward / max(requests, 1)),
        "mean_energy": _mean(selected_energy),
        "mean_fragmentation": _mean(selected_fragmentation),
        "mean_qot": _mean(selected_qot),
        "mean_delay": _mean(selected_delay),
    }


def _score_simulation(result: dict[str, Any], config: ExperimentConfig) -> float:
    accepted_weight = _raw_float(config, "lookahead_accepted_weight", 1.0)
    reward_weight = _raw_float(config, "lookahead_reward_weight", 0.05)
    fragmentation_weight = _raw_float(config, "lookahead_fragmentation_weight", 0.0)
    energy_weight = _raw_float(config, "lookahead_energy_weight", 0.0)
    score = accepted_weight * float(result["accepted"]) + reward_weight * float(result["reward"])
    if result.get("mean_fragmentation") is not None:
        score -= fragmentation_weight * float(result["mean_fragmentation"])
    if result.get("mean_energy") is not None:
        score -= energy_weight * float(result["mean_energy"]) / 1000.0
    return float(score)


def _run_episode(
    *,
    episode_id: str,
    episode: pd.DataFrame,
    traffic_path: Path,
    config: ExperimentConfig,
    solver: GnnCnnDqnOngSolver,
) -> list[dict[str, Any]]:
    env = _make_env(
        episode_id=episode_id,
        traffic_path=traffic_path,
        request_count=len(episode),
        seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
        config=config,
    )
    env.reset(seed=int(config.seed))
    horizon = _raw_int(config, "lookahead_horizon", 12)
    base_policy = str(
        config.resolved.get(
            "lookahead_base_policy",
            config.raw.get(
                "lookahead_base_policy",
                config.resolved.get("tree_ranker_base_policy", config.raw.get("tree_ranker_base_policy", "energy-aware-ksp-bm-ff")),
            ),
        )
    )
    rollout_policy = str(config.resolved.get("lookahead_rollout_policy", config.raw.get("lookahead_rollout_policy", base_policy)))
    state_policy = str(config.resolved.get("lookahead_state_policy", config.raw.get("lookahead_state_policy", base_policy)))
    rows: list[dict[str, Any]] = []
    position = 0

    while position < len(episode):
        batch = solver.candidate_batch(env)
        valid = np.flatnonzero(batch.candidate_mask.astype(bool))
        if valid.size == 0:
            action = int(solver.adapter(env).block_action(env))
            observation, reward, terminated, truncated, info = env.step(action)
            del observation, reward, info
            position += 1
            if bool(terminated) or bool(truncated):
                break
            continue

        q_head_index = _select_q_head(batch, solver.config.n_max)
        j_total_index = _select_j_total(batch)
        base_index = select_tree_base_index(batch, solver.config.n_max, base_policy)
        eval_indices = _candidate_indices(batch)
        if base_index >= 0 and base_index not in eval_indices:
            eval_indices.append(int(base_index))
        simulations: dict[int, dict[str, Any]] = {}
        for index in eval_indices:
            simulations[index] = _simulate_first_action(
                env=env,
                first_action=int(batch.topn[index].action),
                solver=solver,
                rollout_policy=rollout_policy,
                horizon=horizon,
            )
        oracle_index = max(eval_indices, key=lambda index: (_score_simulation(simulations[index], config), -int(index)))
        q_head_result = simulations.get(q_head_index)
        if q_head_result is None:
            q_head_result = _simulate_first_action(
                env=env,
                first_action=int(batch.topn[q_head_index].action),
                solver=solver,
                rollout_policy=rollout_policy,
                horizon=horizon,
            )
        j_total_result = simulations.get(j_total_index)
        if j_total_result is None:
            j_total_result = _simulate_first_action(
                env=env,
                first_action=int(batch.topn[j_total_index].action),
                solver=solver,
                rollout_policy=rollout_policy,
                horizon=horizon,
            )
        base_result = simulations.get(base_index)
        if base_result is None:
            base_result = _simulate_first_action(
                env=env,
                first_action=int(batch.topn[base_index].action),
                solver=solver,
                rollout_policy=rollout_policy,
                horizon=horizon,
            )
        oracle_result = simulations[oracle_index]
        request = episode.iloc[position]
        rows.append(
            {
                "episode_id": str(episode_id),
                "request_id": int(request["request_id"]),
                "traffic_scenario": str(request.get("traffic_scenario", "")),
                "load_name": str(request.get("load_name", "")),
                "seed": int(request["seed"]) if "seed" in request else None,
                "valid_candidates": int(valid.size),
                "evaluated_candidates": int(len(eval_indices)),
                "base_policy": str(base_policy),
                "base_index": int(base_index),
                "q_head_index": int(q_head_index),
                "j_total_index": int(j_total_index),
                "oracle_index": int(oracle_index),
                "oracle_differs_base": bool(int(oracle_index) != int(base_index)),
                "oracle_differs_q_head": bool(int(oracle_index) != int(q_head_index)),
                "oracle_differs_j_total": bool(int(oracle_index) != int(j_total_index)),
                "q_head_q_score": float(batch.topn[q_head_index].q_head_score),
                "oracle_q_score": float(batch.topn[oracle_index].q_head_score),
                "base_lookahead_accepted": int(base_result["accepted"]),
                "q_head_lookahead_accepted": int(q_head_result["accepted"]),
                "j_total_lookahead_accepted": int(j_total_result["accepted"]),
                "oracle_lookahead_accepted": int(oracle_result["accepted"]),
                "oracle_accepted_delta_vs_base": int(oracle_result["accepted"]) - int(base_result["accepted"]),
                "oracle_accepted_delta_vs_q_head": int(oracle_result["accepted"]) - int(q_head_result["accepted"]),
                "oracle_accepted_delta_vs_j_total": int(oracle_result["accepted"]) - int(j_total_result["accepted"]),
                "base_lookahead_reward": float(base_result["reward"]),
                "q_head_lookahead_reward": float(q_head_result["reward"]),
                "j_total_lookahead_reward": float(j_total_result["reward"]),
                "oracle_lookahead_reward": float(oracle_result["reward"]),
                "oracle_reward_delta_vs_base": float(oracle_result["reward"]) - float(base_result["reward"]),
                "oracle_reward_delta_vs_q_head": float(oracle_result["reward"]) - float(q_head_result["reward"]),
                "oracle_reward_delta_vs_j_total": float(oracle_result["reward"]) - float(j_total_result["reward"]),
            }
        )

        state_index = _select_rollout_index(batch, state_policy, solver.config.n_max)
        if state_index < 0:
            action = int(solver.adapter(env).block_action(env))
        else:
            action = int(batch.topn[state_index].action)
        observation, reward, terminated, truncated, info = env.step(action)
        del observation, reward, info
        position += 1
        if bool(terminated) or bool(truncated):
            break
    return rows


def _aggregate(rows: list[dict[str, Any]], group_keys: list[str] | None = None) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    if group_keys:
        grouped = frame.groupby(group_keys, dropna=False)
    else:
        grouped = [((), frame)]
    output: list[dict[str, Any]] = []
    for key, group in grouped:
        record: dict[str, Any] = {}
        if group_keys:
            key_tuple = key if isinstance(key, tuple) else (key,)
            for name, value in zip(group_keys, key_tuple):
                record[name] = value
        samples = int(len(group))
        accepted_delta_col = "oracle_accepted_delta_vs_base" if "oracle_accepted_delta_vs_base" in group else "oracle_accepted_delta_vs_q_head"
        reward_delta_col = "oracle_reward_delta_vs_base" if "oracle_reward_delta_vs_base" in group else "oracle_reward_delta_vs_q_head"
        differs_col = "oracle_differs_base" if "oracle_differs_base" in group else "oracle_differs_q_head"
        base_index_col = "base_index" if "base_index" in group else "q_head_index"
        record.update(
            {
                "samples": samples,
                "mean_valid_candidates": float(group["valid_candidates"].mean()),
                "mean_evaluated_candidates": float(group["evaluated_candidates"].mean()),
                "oracle_differs_base_rate": float(group[differs_col].mean()),
                "positive_accepted_delta_vs_base_rate": float((group[accepted_delta_col] > 0).mean()),
                "mean_accepted_delta_vs_base": float(group[accepted_delta_col].mean()),
                "total_accepted_delta_vs_base": int(group[accepted_delta_col].sum()),
                "positive_reward_delta_vs_base_rate": float((group[reward_delta_col] > 1e-9).mean()),
                "mean_reward_delta_vs_base": float(group[reward_delta_col].mean()),
                "total_reward_delta_vs_base": float(group[reward_delta_col].sum()),
                "oracle_differs_j_total_rate": float(group["oracle_differs_j_total"].mean()),
                "mean_accepted_delta_vs_j_total": float(group["oracle_accepted_delta_vs_j_total"].mean()),
                "total_accepted_delta_vs_j_total": int(group["oracle_accepted_delta_vs_j_total"].sum()),
                "mean_base_index": float(group[base_index_col].mean()),
                "mean_oracle_index": float(group["oracle_index"].mean()),
            }
        )
        output.append(record)
    return output


def run_lookahead_oracle_eval(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("lookahead_oracle_eval requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)
    solver = GnnCnnDqnOngSolver(_solver_config(config))

    split = str(config.resolved.get("rollout_split", config.raw.get("rollout_split", "test")))
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    scenario_filter = set(_raw_list(config, "traffic_scenarios", ()))
    load_filter = set(_raw_list(config, "load_names", ()))
    if scenario_filter:
        traffic = traffic[traffic["traffic_scenario"].astype(str).isin(scenario_filter)]
    if load_filter:
        traffic = traffic[traffic["load_name"].astype(str).isin(load_filter)]
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    max_episodes = _raw_int(config, "max_episodes", 0)
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)

    rows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(max_requests_per_episode).reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, episode_id, episode)
        episode_rows = _run_episode(
            episode_id=episode_id,
            episode=episode,
            traffic_path=traffic_path,
            config=config,
            solver=solver,
        )
        rows.extend(episode_rows)
        episode_summary = _aggregate(episode_rows)[0] if episode_rows else {"samples": 0}
        episode_summary["episode_id"] = episode_id
        print(json.dumps(episode_summary, sort_keys=True))

    detail = pd.DataFrame(rows)
    detail.to_csv(run_path / "lookahead_oracle_samples.csv", index=False)
    by_scenario_load = _aggregate(rows, ["traffic_scenario", "load_name"])
    pd.DataFrame(by_scenario_load).to_csv(run_path / "lookahead_oracle_by_scenario_load.csv", index=False)
    summary = _aggregate(rows)[0] if rows else {"samples": 0}
    base_policy = str(
        config.resolved.get(
            "lookahead_base_policy",
            config.raw.get(
                "lookahead_base_policy",
                config.resolved.get(
                    "tree_ranker_base_policy",
                    config.raw.get("tree_ranker_base_policy", "energy-aware-ksp-bm-ff"),
                ),
            ),
        )
    )
    metrics = {
        "stage": "lookahead_oracle_eval",
        "dataset_path": str(config.dataset_path),
        "split": split,
        "ong_source_path": ong_source,
        "episodes": list(episode_ids),
        "parameters": {
            "candidate_pool": "all_topn",
            "lookahead_base_policy": base_policy,
            "lookahead_horizon": _raw_int(config, "lookahead_horizon", 12),
            "lookahead_rollout_policy": str(
                config.resolved.get("lookahead_rollout_policy", config.raw.get("lookahead_rollout_policy", base_policy))
            ),
            "lookahead_state_policy": str(
                config.resolved.get("lookahead_state_policy", config.raw.get("lookahead_state_policy", base_policy))
            ),
            "lookahead_accepted_weight": _raw_float(config, "lookahead_accepted_weight", 1.0),
            "lookahead_reward_weight": _raw_float(config, "lookahead_reward_weight", 0.05),
        },
        "summary": summary,
        "by_scenario_load": by_scenario_load,
        "sample_metrics_path": str(run_path / "lookahead_oracle_samples.csv"),
    }
    return metrics
