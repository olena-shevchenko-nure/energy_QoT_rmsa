from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.ong_solver import Candidate, GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax

from ..config import ExperimentConfig
from .lookahead_override_features import (
    OVERRIDE_FEATURE_NAMES,
    candidate_feature_matrix,
    candidate_indices_for_topn,
)
from .lookahead_tree_ranker import _predict as _predict_ranker
from .lookahead_tree_ranker import _train_lightgbm, _train_xgboost
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _traffic_jsonl_for_episode,
)
from .tree_ranker_runtime import (
    ADVANTAGE_BASE_RAW_FEATURE_INDICES,
    ADVANTAGE_FEATURE_NAMES,
    TreeCandidateRanker,
    select_tree_base_index,
)
from .lookahead_oracle import _solver_config


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _raw_str(config: ExperimentConfig, key: str, default: str) -> str:
    return str(config.resolved.get(key, config.raw.get(key, default)))


def _raw_optional_int(config: ExperimentConfig, key: str, default: int | None = None) -> int | None:
    value = config.resolved.get(key, config.raw.get(key, default))
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _policy_index(
    *,
    batch: Any,
    solver: GnnCnnDqnOngSolver,
    policy: str,
    tree_ranker: TreeCandidateRanker | None = None,
) -> tuple[int, float]:
    policy_norm = str(policy or "energy-aware-ksp-bm-ff").strip().lower().replace("_", "-")
    if policy_norm in {"tree-ranker", "xgboost-candidate-ranker", "lightgbm-candidate-ranker"}:
        if tree_ranker is None:
            return select_tree_base_index(batch, solver.config.n_max, "energy-aware-ksp-bm-ff"), 0.0
        return tree_ranker.select_index(batch, solver.config.n_max)
    if policy_norm == "gnn-cnn-dqn":
        q_values = solver.q_values(batch)
        selected = int(masked_argmax(q_values, batch.candidate_mask))
        if selected < 0:
            return -1, 0.0
        valid = np.flatnonzero(batch.candidate_mask.astype(bool))
        if valid.size <= 1:
            return selected, 0.0
        masked = np.asarray(q_values, dtype=np.float32).copy()
        masked[int(selected)] = -np.inf
        return selected, float(q_values[int(selected)] - np.max(masked[valid]))
    return select_tree_base_index(batch, solver.config.n_max, policy_norm), 0.0


def _candidate_indices(
    *,
    batch: Any,
) -> list[int]:
    return candidate_indices_for_topn(batch)


def _step_policy(
    *,
    env: Any,
    solver: GnnCnnDqnOngSolver,
    policy: str,
    tree_ranker: TreeCandidateRanker | None = None,
) -> tuple[int, Candidate | None, int, float]:
    batch = solver.candidate_batch(env)
    if not np.asarray(batch.candidate_mask, dtype=bool).any():
        return int(solver.adapter(env).block_action(env)), None, -1, 0.0
    index, margin = _policy_index(batch=batch, solver=solver, policy=policy, tree_ranker=tree_ranker)
    if index < 0:
        return int(solver.adapter(env).block_action(env)), None, -1, float(margin)
    candidate = batch.topn[int(index)]
    return int(candidate.action), candidate, int(index), float(margin)


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


def _simulate_candidate(
    *,
    env: Any,
    candidate: Candidate,
    solver: GnnCnnDqnOngSolver,
    rollout_policy: str,
    horizon: int,
) -> dict[str, Any]:
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
        selected_candidate: Candidate | None = candidate
        for step in range(max(1, int(horizon))):
            if step > 0:
                action, selected_candidate, _index, _margin = _step_policy(
                    env=clone,
                    solver=solver,
                    policy=rollout_policy,
                    tree_ranker=None,
                )
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


def _simulation_metric(simulation: dict[str, Any], key: str, fallback: float) -> float:
    value = simulation.get(key)
    if value is None:
        return float(fallback)
    value_float = float(value)
    return value_float if math.isfinite(value_float) else float(fallback)


def _secondary_tiebreak_score(candidate: Candidate, simulation: dict[str, Any], config: ExperimentConfig) -> float:
    energy_norm_w = max(_raw_float(config, "dagger_utility_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)), 1e-9)
    energy_value = _simulation_metric(simulation, "future_energy_increment_sum", float(candidate.energy_increment)) / energy_norm_w
    fragmentation = _simulation_metric(simulation, "future_fragmentation_after_mean", float(candidate.fragmentation_after))
    qot_margin = _simulation_metric(simulation, "future_qot_margin_norm_mean", float(candidate.qot_margin_norm))
    qot_margin = min(
        _raw_float(config, "dagger_utility_qot_clip_max", 1.0),
        max(_raw_float(config, "dagger_utility_qot_clip_min", 0.0), qot_margin),
    )
    return float(
        - _raw_float(config, "dagger_utility_energy_weight", 0.25) * energy_value
        - _raw_float(config, "dagger_utility_fragmentation_weight", 0.80) * fragmentation
        + _raw_float(config, "dagger_utility_qot_weight", 0.20) * qot_margin
    )


def _utility(candidate: Candidate, simulation: dict[str, Any], config: ExperimentConfig) -> float:
    mode = _raw_str(config, "dagger_utility_mode", "linear").strip().lower()
    if mode in {"accepted_tiebreak", "lexicographic", "lexicographic_advantage"}:
        return float(
            _raw_float(config, "dagger_utility_accepted_weight", 100.0) * float(simulation["future_accepted"])
            + _secondary_tiebreak_score(candidate, simulation, config)
        )

    energy_norm_w = max(_raw_float(config, "dagger_utility_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)), 1e-9)
    energy_value = float(candidate.energy_increment) / energy_norm_w
    return float(
        _raw_float(config, "dagger_utility_accepted_weight", 2.0) * float(simulation["future_accepted"])
        - _raw_float(config, "dagger_utility_block_penalty", 1.5) * float(simulation["future_blocked"])
        - _raw_float(config, "dagger_utility_energy_weight", 0.25) * energy_value
        - _raw_float(config, "dagger_utility_fragmentation_weight", 0.80) * float(candidate.fragmentation_after)
        + _raw_float(config, "dagger_utility_qot_weight", 0.20) * float(candidate.qot_margin_norm)
    )


def _advantage_delta_target(
    *,
    accepted_delta: int,
    secondary_delta: float,
    reward_delta: float,
    utility_delta: float,
    config: ExperimentConfig,
) -> float:
    mode = _raw_str(config, "advantage_gate_delta_target_mode", "accepted_tiebreak").strip().lower()
    if mode in {"accepted", "accepted_only"}:
        return float(accepted_delta)
    if mode in {"utility", "utility_delta"}:
        return float(utility_delta)
    if mode in {"accepted_reward", "accepted_plus_reward"}:
        return float(accepted_delta) + _raw_float(config, "advantage_gate_delta_reward_weight", 0.0) * float(reward_delta)
    if int(accepted_delta) != 0:
        return float(accepted_delta)
    return _raw_float(config, "advantage_gate_delta_tiebreak_weight", 1.0) * float(secondary_delta)


def _target_values(utilities: list[float], config: ExperimentConfig) -> list[float]:
    mode = _raw_str(config, "dagger_rank_target_mode", "shifted_utility").strip().lower()
    values = np.asarray(utilities, dtype=np.float64)
    if values.size == 0:
        return []
    if mode == "raw_utility":
        return [float(value) for value in values]
    if mode == "rank_grade":
        order = np.argsort(-values)
        grades = np.zeros_like(values)
        for rank, position in enumerate(order):
            grades[int(position)] = max(0.0, float(values.size - rank - 1))
        return [float(value) for value in grades]
    shifted = values - float(np.min(values))
    return [float(value) for value in shifted]


def _accepted_safe_targets(
    *,
    utilities: list[float],
    simulations: list[dict[str, Any]],
    base_position: int,
    config: ExperimentConfig,
) -> tuple[list[float], list[float], int]:
    """Rank only candidates that do not lose accepted requests vs the base policy."""
    if not utilities:
        return [], [], -1
    values = np.asarray(utilities, dtype=np.float64)
    base_accepted = int(simulations[int(base_position)]["future_accepted"]) if base_position >= 0 else int(
        max(int(sim["future_accepted"]) for sim in simulations)
    )
    accepted = np.asarray([int(sim["future_accepted"]) for sim in simulations], dtype=np.int32)
    safe = accepted >= int(base_accepted)
    label_scores = np.full(values.shape, float(np.min(values)) - 1.0, dtype=np.float64)
    label_scores[safe] = values[safe]

    grades = np.zeros_like(values, dtype=np.float64)
    safe_positions = np.flatnonzero(safe)
    if safe_positions.size:
        order = safe_positions[np.argsort(-values[safe_positions])]
        for rank, position in enumerate(order):
            grades[int(position)] = float(safe_positions.size - rank)
    else:
        order = np.argsort(-values)
        for rank, position in enumerate(order):
            grades[int(position)] = max(0.0, float(values.size - rank - 1))
        label_scores = values.copy()

    mode = _raw_str(config, "dagger_rank_target_mode", "shifted_utility").strip().lower()
    if mode == "accepted_safe_raw_utility":
        targets = label_scores.copy()
    elif mode == "accepted_safe_shifted_utility":
        targets = label_scores - float(np.min(label_scores))
    else:
        targets = grades
    best_position = int(np.argmax(label_scores))
    return [float(value) for value in targets], [float(value) for value in label_scores], best_position


def _dagger_targets(
    *,
    utilities: list[float],
    simulations: list[dict[str, Any]],
    base_position: int,
    config: ExperimentConfig,
) -> tuple[list[float], list[float], int]:
    mode = _raw_str(config, "dagger_rank_target_mode", "shifted_utility").strip().lower()
    if mode.startswith("accepted_safe"):
        return _accepted_safe_targets(
            utilities=utilities,
            simulations=simulations,
            base_position=base_position,
            config=config,
        )
    targets = _target_values(utilities, config)
    values = [float(value) for value in utilities]
    best_position = int(np.argmax(np.asarray(values, dtype=np.float64))) if values else -1
    return targets, values, best_position


def _collect_examples(
    *,
    config: ExperimentConfig,
    split: str,
    solver: GnnCnnDqnOngSolver,
    run_path: Path,
    iteration: int,
    state_policy: str,
    tree_ranker: TreeCandidateRanker | None,
    group_id_start: int,
    max_episodes_key: str,
    max_requests_key: str,
) -> dict[str, Any]:
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    max_episodes = _raw_optional_int(config, max_episodes_key, _raw_int(config, "max_episodes", 0))
    if max_episodes and max_episodes > 0:
        episode_ids = episode_ids[: int(max_episodes)]
    max_requests = _raw_optional_int(config, max_requests_key, _raw_int(config, "max_requests_per_episode", 0))

    base_policy = _raw_str(config, "dagger_base_policy", "energy-aware-ksp-bm-ff")
    rollout_policy = _raw_str(config, "dagger_lookahead_rollout_policy", base_policy)
    horizon = _raw_int(config, "dagger_lookahead_horizon", 12)

    rows: list[np.ndarray] = []
    targets: list[float] = []
    group_sizes: list[int] = []
    metadata: list[dict[str, Any]] = []
    skipped_groups = 0
    group_id = int(group_id_start)

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests and max_requests > 0:
            episode = episode.head(int(max_requests)).reset_index(drop=True)
        if episode.empty:
            continue
        traffic_path = _traffic_jsonl_for_episode(run_path, f"{split}_iter{iteration}_{episode_id}", episode)
        env = _make_env(
            episode_id=f"{split}_iter{iteration}_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))

        for _, request in episode.iterrows():
            batch = solver.candidate_batch(env)
            if np.asarray(batch.candidate_mask, dtype=bool).any():
                candidate_indices = _candidate_indices(
                    batch=batch,
                )
                base_index = select_tree_base_index(batch, solver.config.n_max, base_policy)
                features, kept_indices = candidate_feature_matrix(
                    batch=batch,
                    candidate_indices=candidate_indices,
                    n_max=solver.config.n_max,
                    reference_index=base_index,
                )
                if features.shape[0] >= 2:
                    utilities: list[float] = []
                    secondary_scores: list[float] = []
                    sims: list[dict[str, Any]] = []
                    for candidate_index in kept_indices:
                        candidate = batch.topn[int(candidate_index)]
                        simulation = _simulate_candidate(
                            env=env,
                            candidate=candidate,
                            solver=solver,
                            rollout_policy=rollout_policy,
                            horizon=horizon,
                        )
                        sims.append(simulation)
                        secondary_scores.append(_secondary_tiebreak_score(candidate, simulation, config))
                        utilities.append(_utility(candidate, simulation, config))
                    base_position = kept_indices.index(base_index) if base_index in kept_indices else 0
                    relevance, label_scores, best_position = _dagger_targets(
                        utilities=utilities,
                        simulations=sims,
                        base_position=base_position,
                        config=config,
                    )
                    best_index = int(kept_indices[best_position])
                    group_sizes.append(int(features.shape[0]))
                    for feature_row, candidate_index, simulation, utility, secondary_score, label_score, target in zip(
                        features,
                        kept_indices,
                        sims,
                        utilities,
                        secondary_scores,
                        label_scores,
                        relevance,
                    ):
                        candidate = batch.topn[int(candidate_index)]
                        rows.append(feature_row)
                        targets.append(float(target))
                        metadata.append(
                            {
                                "split": split,
                                "iteration": int(iteration),
                                "group_id": int(group_id),
                                "episode_id": episode_id,
                                "request_id": int(request["request_id"]),
                                "traffic_scenario": str(request.get("traffic_scenario", "")),
                                "load_name": str(request.get("load_name", "")),
                                "candidate_index": int(candidate_index),
                                "base_index": int(base_index),
                                "utility_best_index": int(best_index),
                                "utility": float(utility),
                                "secondary_score": float(secondary_score),
                                "secondary_delta_vs_base": float(secondary_score) - float(secondary_scores[base_position]),
                                "label_score": float(label_score),
                                "target": float(target),
                                "accepted_delta_vs_base": int(simulation["future_accepted"]) - int(sims[base_position]["future_accepted"]),
                                "future_env_reward_delta_vs_base": float(simulation["future_env_reward"])
                                - float(sims[base_position]["future_env_reward"]),
                                "future_energy_increment_delta_vs_base": float(simulation["future_energy_increment_sum"])
                                - float(sims[base_position]["future_energy_increment_sum"]),
                                "future_fragmentation_after_delta_vs_base": float(
                                    _simulation_metric(
                                        simulation,
                                        "future_fragmentation_after_mean",
                                        float(candidate.fragmentation_after),
                                    )
                                )
                                - float(
                                    _simulation_metric(
                                        sims[base_position],
                                        "future_fragmentation_after_mean",
                                        float(batch.topn[int(kept_indices[base_position])].fragmentation_after),
                                    )
                                ),
                                "future_qot_margin_delta_vs_base": float(
                                    _simulation_metric(simulation, "future_qot_margin_norm_mean", float(candidate.qot_margin_norm))
                                )
                                - float(
                                    _simulation_metric(
                                        sims[base_position],
                                        "future_qot_margin_norm_mean",
                                        float(batch.topn[int(kept_indices[base_position])].qot_margin_norm),
                                    )
                                ),
                                "future_accepted": int(simulation["future_accepted"]),
                                "future_blocked": int(simulation["future_blocked"]),
                                "future_requests": int(simulation["future_requests"]),
                                "future_env_reward": float(simulation["future_env_reward"]),
                                "future_selected_count": int(simulation["future_selected_count"]),
                                "future_energy_increment_sum": float(simulation["future_energy_increment_sum"]),
                                "future_energy_increment_mean": float(
                                    _simulation_metric(
                                        simulation,
                                        "future_energy_increment_mean",
                                        float(candidate.energy_increment),
                                    )
                                ),
                                "future_fragmentation_after_mean": float(
                                    _simulation_metric(
                                        simulation,
                                        "future_fragmentation_after_mean",
                                        float(candidate.fragmentation_after),
                                    )
                                ),
                                "future_qot_margin_norm_mean": float(
                                    _simulation_metric(simulation, "future_qot_margin_norm_mean", float(candidate.qot_margin_norm))
                                ),
                                "future_delay_ms_mean": float(_simulation_metric(simulation, "future_delay_ms_mean", float(candidate.delay_ms))),
                                "energy_increment": float(candidate.energy_increment),
                                "energy_increment_norm": float(candidate.energy_increment_norm),
                                "fragmentation_after": float(candidate.fragmentation_after),
                                "qot_margin_norm": float(candidate.qot_margin_norm),
                            }
                        )
                    group_id += 1
                else:
                    skipped_groups += 1

            action, _candidate, _index, _margin = _step_policy(
                env=env,
                solver=solver,
                policy=state_policy,
                tree_ranker=tree_ranker,
            )
            _observation, _reward, terminated, truncated, _info = env.step(int(action))
            if bool(terminated) or bool(truncated):
                break

    if rows:
        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
    else:
        x = np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32)
    return {
        "x": x,
        "y": y,
        "group_sizes": np.asarray(group_sizes, dtype=np.int32),
        "metadata": pd.DataFrame(metadata),
        "skipped_groups": int(skipped_groups),
        "next_group_id": int(group_id),
    }


def _merge_datasets(items: list[dict[str, Any]]) -> dict[str, Any]:
    nonempty = [item for item in items if item["x"].shape[0] > 0]
    if not nonempty:
        return {
            "x": np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32),
            "y": np.zeros((0,), dtype=np.float32),
            "group_sizes": np.zeros((0,), dtype=np.int32),
            "metadata": pd.DataFrame(),
            "skipped_groups": int(sum(int(item.get("skipped_groups", 0)) for item in items)),
        }
    return {
        "x": np.concatenate([item["x"] for item in nonempty], axis=0),
        "y": np.concatenate([item["y"] for item in nonempty], axis=0),
        "group_sizes": np.concatenate([item["group_sizes"] for item in nonempty], axis=0),
        "metadata": pd.concat([item["metadata"] for item in nonempty], ignore_index=True),
        "skipped_groups": int(sum(int(item.get("skipped_groups", 0)) for item in items)),
    }


def _write_counterfactual_mining_outputs(
    *,
    data: dict[str, Any],
    config: ExperimentConfig,
    run_path: Path,
    prefix: str,
) -> dict[str, Any]:
    metadata = data["metadata"].copy()
    if metadata.empty:
        return {
            "groups": 0,
            "rows": 0,
            "hard_positive_rows": 0,
            "hard_positive_groups": 0,
            "hard_negative_rows": 0,
            "hard_negative_groups": 0,
        }

    win_min = _raw_int(config, "advantage_gate_win_min_accepted_delta", 1)
    loss_min = _raw_int(config, "advantage_gate_loss_min_accepted_delta", 1)
    metadata["candidate_differs_base"] = metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)
    if "secondary_delta_vs_base" not in metadata:
        metadata["secondary_delta_vs_base"] = 0.0
    hard_positive = metadata[
        metadata["candidate_differs_base"] & (metadata["accepted_delta_vs_base"].astype(int) >= int(win_min))
    ].copy()
    hard_negative = metadata[
        metadata["candidate_differs_base"] & (metadata["accepted_delta_vs_base"].astype(int) <= -int(loss_min))
    ].copy()

    order_columns = ["group_id", "accepted_delta_vs_base", "secondary_delta_vs_base", "future_env_reward_delta_vs_base", "candidate_index"]
    best = metadata.sort_values(
        order_columns,
        ascending=[True, False, False, False, True],
    ).groupby("group_id", sort=False).head(1)
    best_positive = best[best["accepted_delta_vs_base"].astype(int) >= int(win_min)].copy()

    hard_positive.to_csv(run_path / f"{prefix}_counterfactual_hard_positive_examples.csv", index=False)
    hard_negative.to_csv(run_path / f"{prefix}_counterfactual_hard_negative_examples.csv", index=False)
    best_positive.to_csv(run_path / f"{prefix}_counterfactual_best_positive_groups.csv", index=False)

    return {
        "groups": int(metadata["group_id"].nunique()),
        "rows": int(len(metadata)),
        "hard_positive_rows": int(len(hard_positive)),
        "hard_positive_groups": int(hard_positive["group_id"].nunique()) if not hard_positive.empty else 0,
        "hard_negative_rows": int(len(hard_negative)),
        "hard_negative_groups": int(hard_negative["group_id"].nunique()) if not hard_negative.empty else 0,
        "best_positive_groups": int(len(best_positive)),
        "max_accepted_delta_vs_base": int(metadata["accepted_delta_vs_base"].max()),
        "min_accepted_delta_vs_base": int(metadata["accepted_delta_vs_base"].min()),
        "mean_positive_accepted_delta": None
        if hard_positive.empty
        else float(hard_positive["accepted_delta_vs_base"].astype(float).mean()),
        "mean_positive_secondary_delta": None
        if hard_positive.empty
        else float(hard_positive["secondary_delta_vs_base"].astype(float).mean()),
        "outputs": {
            "hard_positive_examples": str(run_path / f"{prefix}_counterfactual_hard_positive_examples.csv"),
            "hard_negative_examples": str(run_path / f"{prefix}_counterfactual_hard_negative_examples.csv"),
            "best_positive_groups": str(run_path / f"{prefix}_counterfactual_best_positive_groups.csv"),
        },
    }


def _ranker_metrics(metadata: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    if metadata.empty:
        return {
            "groups": 0,
            "rows": 0,
            "utility_top1_accuracy": None,
            "base_utility_top1_accuracy": None,
            "label_top1_accuracy": None,
            "base_label_top1_accuracy": None,
            "override_rate_vs_base": None,
            "mean_selected_utility_delta_vs_base": None,
            "unsafe_override_rate_vs_base": None,
        }
    scored = metadata.copy()
    scored["score"] = np.asarray(scores, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for _, group in scored.groupby("group_id", sort=False):
        selected = group.sort_values(["score", "candidate_index"], ascending=[False, True]).iloc[0]
        best = group.sort_values(["utility", "candidate_index"], ascending=[False, True]).iloc[0]
        label_column = "label_score" if "label_score" in group else "utility"
        label_best = group.sort_values([label_column, "candidate_index"], ascending=[False, True]).iloc[0]
        base_rows = group[group["candidate_index"].astype(int) == int(selected["base_index"])]
        base = base_rows.iloc[0] if not base_rows.empty else group.iloc[0]
        rows.append(
            {
                "selected_best": int(selected["candidate_index"]) == int(best["candidate_index"]),
                "base_best": int(base["candidate_index"]) == int(best["candidate_index"]),
                "selected_label_best": int(selected["candidate_index"]) == int(label_best["candidate_index"]),
                "base_label_best": int(base["candidate_index"]) == int(label_best["candidate_index"]),
                "selected_differs_base": int(selected["candidate_index"]) != int(base["candidate_index"]),
                "selected_utility_delta_vs_base": float(selected["utility"]) - float(base["utility"]),
                "selected_future_accepted_delta_vs_base": int(selected["future_accepted"]) - int(base["future_accepted"]),
                "unsafe_override_vs_base": int(selected["candidate_index"]) != int(base["candidate_index"])
                and int(selected["future_accepted"]) < int(base["future_accepted"]),
            }
        )
    table = pd.DataFrame(rows)
    return {
        "groups": int(len(table)),
        "rows": int(len(scored)),
        "utility_top1_accuracy": float(table["selected_best"].mean()),
        "base_utility_top1_accuracy": float(table["base_best"].mean()),
        "label_top1_accuracy": float(table["selected_label_best"].mean()),
        "base_label_top1_accuracy": float(table["base_label_best"].mean()),
        "override_rate_vs_base": float(table["selected_differs_base"].mean()),
        "mean_selected_utility_delta_vs_base": float(table["selected_utility_delta_vs_base"].mean()),
        "mean_selected_future_accepted_delta_vs_base": float(table["selected_future_accepted_delta_vs_base"].mean()),
        "unsafe_override_rate_vs_base": float(table["unsafe_override_vs_base"].mean()),
    }


def _advantage_gate_enabled(config: ExperimentConfig, selection_mode: str) -> bool:
    return _raw_bool(
        config,
        "advantage_gate_enabled",
        str(selection_mode).strip().lower() in {"advantage", "positive_advantage"},
    )


def _advantage_gate_thresholds(config: ExperimentConfig) -> dict[str, float]:
    return {
        "min_win_prob": _raw_float(config, "advantage_gate_min_win_prob", 0.35),
        "max_loss_prob": _raw_float(config, "advantage_gate_max_loss_prob", 0.04),
        "min_delta_pred": _raw_float(config, "advantage_gate_min_delta_pred", 0.01),
        "win_weight": _raw_float(config, "advantage_gate_win_weight", 1.0),
        "loss_weight": _raw_float(config, "advantage_gate_loss_weight", 2.0),
        "delta_weight": _raw_float(config, "advantage_gate_delta_weight", 1.0),
        "ranker_margin_weight": _raw_float(config, "advantage_gate_ranker_margin_weight", 0.0),
    }


def _raw_float_grid(config: ExperimentConfig, key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = config.resolved.get(key, config.raw.get(key))
    if value is None:
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    return (float(value),)


def _advantage_score(
    *,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    ranker_margin: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return (
        float(thresholds["delta_weight"]) * np.asarray(delta_pred, dtype=np.float32)
        + float(thresholds["win_weight"]) * np.asarray(win_prob, dtype=np.float32)
        - float(thresholds["loss_weight"]) * np.asarray(loss_prob, dtype=np.float32)
        + float(thresholds["ranker_margin_weight"]) * np.asarray(ranker_margin, dtype=np.float32)
    )


def _build_advantage_dataset(
    *,
    data: dict[str, Any],
    ranker_scores: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True).copy()
    features = np.asarray(data["x"], dtype=np.float32)
    scores = np.asarray(ranker_scores, dtype=np.float32)
    if metadata.empty or features.size == 0:
        return {
            "x": np.zeros((0, len(ADVANTAGE_FEATURE_NAMES)), dtype=np.float32),
            "win_y": np.zeros((0,), dtype=np.float32),
            "loss_y": np.zeros((0,), dtype=np.float32),
            "delta_y": np.zeros((0,), dtype=np.float32),
            "metadata": pd.DataFrame(),
        }

    win_min_accepted = _raw_int(config, "advantage_gate_win_min_accepted_delta", 1)
    loss_min_accepted = _raw_int(config, "advantage_gate_loss_min_accepted_delta", 1)

    rows: list[np.ndarray] = []
    win_targets: list[float] = []
    loss_targets: list[float] = []
    delta_targets: list[float] = []
    pair_metadata: list[dict[str, Any]] = []

    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = [int(index) for index in group.index.to_numpy()]
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        base_row_index = int(base_rows.index[0]) if not base_rows.empty else int(group_indices[0])
        base_features = features[base_row_index]
        base_raw_features = base_features[list(ADVANTAGE_BASE_RAW_FEATURE_INDICES)]
        base_score = float(scores[base_row_index])
        base_future_accepted = int(metadata.at[base_row_index, "future_accepted"])
        base_future_reward = float(metadata.at[base_row_index, "future_env_reward"])
        base_utility = float(metadata.at[base_row_index, "utility"])
        base_secondary = float(metadata.at[base_row_index, "secondary_score"]) if "secondary_score" in metadata else 0.0

        for row_index in group_indices:
            if int(row_index) == int(base_row_index):
                continue
            row = metadata.loc[row_index]
            candidate_features = features[row_index]
            ranker_score = float(scores[row_index])
            ranker_margin = float(ranker_score - base_score)
            accepted_delta = int(row.get("accepted_delta_vs_base", int(row["future_accepted"]) - base_future_accepted))
            reward_delta = float(row.get("future_env_reward_delta_vs_base", float(row["future_env_reward"]) - base_future_reward))
            utility_delta = float(row["utility"]) - base_utility
            secondary_delta = float(row.get("secondary_delta_vs_base", float(row.get("secondary_score", 0.0)) - base_secondary))
            target_delta = _advantage_delta_target(
                accepted_delta=int(accepted_delta),
                secondary_delta=float(secondary_delta),
                reward_delta=float(reward_delta),
                utility_delta=float(utility_delta),
                config=config,
            )
            rows.append(
                np.concatenate(
                    [
                        candidate_features,
                        base_raw_features,
                        np.asarray([ranker_score, ranker_margin], dtype=np.float32),
                    ]
                )
            )
            win_targets.append(float(accepted_delta >= int(win_min_accepted)))
            loss_targets.append(float(accepted_delta <= -int(loss_min_accepted)))
            delta_targets.append(float(target_delta))
            pair_metadata.append(
                {
                    "split": str(row.get("split", "")),
                    "iteration": int(row.get("iteration", 0)),
                    "group_id": int(row["group_id"]),
                    "episode_id": str(row.get("episode_id", "")),
                    "request_id": int(row.get("request_id", 0)),
                    "traffic_scenario": str(row.get("traffic_scenario", "")),
                    "load_name": str(row.get("load_name", "")),
                    "candidate_index": int(row["candidate_index"]),
                    "base_index": int(base_index),
                    "ranker_score": float(ranker_score),
                    "ranker_margin": float(ranker_margin),
                    "accepted_delta_vs_base": int(accepted_delta),
                    "future_env_reward_delta_vs_base": float(reward_delta),
                    "utility_delta_vs_base": float(utility_delta),
                    "secondary_delta_vs_base": float(secondary_delta),
                    "target_delta": float(target_delta),
                    "is_win": bool(accepted_delta >= int(win_min_accepted)),
                    "is_loss": bool(accepted_delta <= -int(loss_min_accepted)),
                }
            )

    if not rows:
        x = np.zeros((0, len(ADVANTAGE_FEATURE_NAMES)), dtype=np.float32)
    else:
        x = np.asarray(rows, dtype=np.float32)
    return {
        "x": x,
        "win_y": np.asarray(win_targets, dtype=np.float32),
        "loss_y": np.asarray(loss_targets, dtype=np.float32),
        "delta_y": np.asarray(delta_targets, dtype=np.float32),
        "metadata": pd.DataFrame(pair_metadata),
    }


def _pos_weight(y: np.ndarray) -> float:
    positives = int((np.asarray(y, dtype=np.float32) > 0.5).sum())
    negatives = int(y.size - positives)
    if positives <= 0:
        return 1.0
    return float(max(1.0, negatives / max(positives, 1)))


def _advantage_sample_weights(
    *,
    dataset: dict[str, Any],
    config: ExperimentConfig,
    target_key: str,
) -> np.ndarray | None:
    if not _raw_bool(config, "advantage_gate_use_sample_weights", True):
        return None
    metadata = dataset.get("metadata")
    if metadata is None or metadata.empty:
        return None
    if "accepted_delta_vs_base" not in metadata:
        return None

    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    weights = np.full(
        accepted_delta.shape,
        _raw_float(config, "advantage_gate_neutral_weight", 0.35),
        dtype=np.float32,
    )
    hard_positive = accepted_delta >= float(_raw_int(config, "advantage_gate_win_min_accepted_delta", 1))
    hard_negative = accepted_delta <= -float(_raw_int(config, "advantage_gate_loss_min_accepted_delta", 1))

    if target_key == "win_y":
        weights[hard_positive] = _raw_float(config, "advantage_gate_win_positive_weight", 8.0)
        weights[hard_negative] = _raw_float(config, "advantage_gate_win_hard_negative_weight", 2.0)
    elif target_key == "loss_y":
        weights[hard_negative] = _raw_float(config, "advantage_gate_loss_positive_weight", 10.0)
        weights[hard_positive] = _raw_float(config, "advantage_gate_loss_hard_positive_weight", 2.0)
    elif target_key == "delta_y":
        weights[hard_positive] = _raw_float(config, "advantage_gate_delta_hard_positive_weight", 6.0)
        weights[hard_negative] = _raw_float(config, "advantage_gate_delta_hard_negative_weight", 8.0)
        tie_break = np.isclose(accepted_delta, 0.0)
        weights[tie_break] = _raw_float(config, "advantage_gate_delta_neutral_weight", 0.75)

    if _raw_bool(config, "advantage_gate_group_normalize_weights", True) and "group_id" in metadata:
        group_codes = pd.Categorical(metadata["group_id"], ordered=False).codes.astype(np.int32)
        group_sums = np.bincount(group_codes, weights=weights.astype(np.float64))
        group_counts = np.bincount(group_codes)
        group_mean = np.divide(
            group_sums,
            np.maximum(group_counts, 1),
            out=np.ones_like(group_sums, dtype=np.float64),
            where=group_counts > 0,
        )
        weights = (weights / np.maximum(group_mean[group_codes].astype(np.float32), 1e-6)).astype(np.float32)

    mean = float(np.mean(weights)) if weights.size else 1.0
    if mean > 0.0:
        weights = (weights / mean).astype(np.float32)
    return weights


def _xgboost_device_params(config: ExperimentConfig) -> dict[str, Any]:
    import xgboost as xgb

    tree_method = str(
        config.resolved.get(
            "advantage_gate_tree_method",
            config.raw.get("advantage_gate_tree_method", config.resolved.get("tree_ranker_tree_method", "hist")),
        )
    )
    device = str(
        config.resolved.get(
            "advantage_gate_device",
            config.raw.get("advantage_gate_device", config.resolved.get("tree_ranker_device", "")),
        )
    ).strip()
    version_parts = tuple(int(part) for part in str(getattr(xgb, "__version__", "0")).split(".")[:2] if part.isdigit())
    if device and device.lower().startswith(("cuda", "gpu")) and version_parts and version_parts < (2, 0):
        tree_method = str(
            config.resolved.get(
                "advantage_gate_gpu_tree_method",
                config.raw.get("advantage_gate_gpu_tree_method", "gpu_hist"),
            )
        )
    params: dict[str, Any] = {"tree_method": tree_method}
    if device and (not version_parts or version_parts >= (2, 0)):
        params["device"] = device
    predictor = str(config.resolved.get("advantage_gate_predictor", config.raw.get("advantage_gate_predictor", ""))).strip()
    if predictor:
        params["predictor"] = predictor
    return params


def _train_advantage_xgboost(
    *,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
    target_key: str,
    task: str,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for advantage gate training") from exc

    train_weights = _advantage_sample_weights(dataset=train, config=config, target_key=target_key)
    dtrain = xgb.DMatrix(
        train["x"],
        label=train[target_key],
        weight=train_weights,
        feature_names=ADVANTAGE_FEATURE_NAMES,
    )
    evals = [(dtrain, "train")]
    if val is not None and val["x"].shape[0] > 0:
        val_weights = _advantage_sample_weights(dataset=val, config=config, target_key=target_key)
        dval = xgb.DMatrix(
            val["x"],
            label=val[target_key],
            weight=val_weights,
            feature_names=ADVANTAGE_FEATURE_NAMES,
        )
        evals.append((dval, "val"))

    objective = "binary:logistic" if task == "binary" else "reg:squarederror"
    params = {
        "objective": objective,
        "eval_metric": "logloss" if task == "binary" else "rmse",
        "eta": _raw_float(config, "advantage_gate_learning_rate", _raw_float(config, "tree_ranker_learning_rate", 0.04)),
        "max_depth": _raw_int(config, "advantage_gate_max_depth", _raw_int(config, "tree_ranker_max_depth", 4)),
        "min_child_weight": _raw_float(
            config,
            "advantage_gate_min_child_weight",
            _raw_float(config, "tree_ranker_min_child_weight", 1.0),
        ),
        "subsample": _raw_float(config, "advantage_gate_subsample", _raw_float(config, "tree_ranker_subsample", 0.9)),
        "colsample_bytree": _raw_float(
            config,
            "advantage_gate_colsample_bytree",
            _raw_float(config, "tree_ranker_colsample_bytree", 0.9),
        ),
        "seed": int(config.seed),
        "nthread": _raw_int(config, "advantage_gate_nthread", _raw_int(config, "tree_ranker_nthread", 4)),
    }
    params.update(_xgboost_device_params(config))
    if task == "binary" and _raw_bool(config, "advantage_gate_auto_scale_pos_weight", True):
        params["scale_pos_weight"] = _pos_weight(train[target_key])

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=_raw_int(
            config,
            "advantage_gate_num_boost_round",
            _raw_int(config, "tree_ranker_num_boost_round", 160),
        ),
        evals=evals,
        verbose_eval=False,
    )
    booster.save_model(str(model_path))
    return booster


def _lightgbm_device_params(config: ExperimentConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    device_type = str(
        config.resolved.get(
            "advantage_gate_device_type",
            config.raw.get("advantage_gate_device_type", config.resolved.get("tree_ranker_device_type", "")),
        )
    ).strip()
    if device_type:
        params["device_type"] = device_type
    max_bin = config.resolved.get(
        "advantage_gate_max_bin",
        config.raw.get("advantage_gate_max_bin", config.resolved.get("tree_ranker_max_bin")),
    )
    if max_bin is not None:
        params["max_bin"] = int(max_bin)
    gpu_platform_id = config.resolved.get(
        "advantage_gate_gpu_platform_id",
        config.raw.get("advantage_gate_gpu_platform_id", config.resolved.get("tree_ranker_gpu_platform_id")),
    )
    if gpu_platform_id is not None:
        params["gpu_platform_id"] = int(gpu_platform_id)
    gpu_device_id = config.resolved.get(
        "advantage_gate_gpu_device_id",
        config.raw.get("advantage_gate_gpu_device_id", config.resolved.get("tree_ranker_gpu_device_id")),
    )
    if gpu_device_id is not None:
        params["gpu_device_id"] = int(gpu_device_id)
    gpu_use_dp = config.resolved.get(
        "advantage_gate_gpu_use_dp",
        config.raw.get("advantage_gate_gpu_use_dp", config.resolved.get("tree_ranker_gpu_use_dp")),
    )
    if gpu_use_dp is not None:
        params["gpu_use_dp"] = _raw_bool(config, "advantage_gate_gpu_use_dp", False)
    return params


def _train_advantage_lightgbm(
    *,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
    target_key: str,
    task: str,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is required for advantage gate training") from exc

    dtrain = lgb.Dataset(
        train["x"],
        label=train[target_key],
        weight=_advantage_sample_weights(dataset=train, config=config, target_key=target_key),
        feature_name=ADVANTAGE_FEATURE_NAMES,
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    if val is not None and val["x"].shape[0] > 0:
        dval = lgb.Dataset(
            val["x"],
            label=val[target_key],
            weight=_advantage_sample_weights(dataset=val, config=config, target_key=target_key),
            feature_name=ADVANTAGE_FEATURE_NAMES,
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets.append(dval)
        valid_names.append("val")

    params = {
        "objective": "binary" if task == "binary" else "regression",
        "metric": "binary_logloss" if task == "binary" else "rmse",
        "learning_rate": _raw_float(config, "advantage_gate_learning_rate", _raw_float(config, "tree_ranker_learning_rate", 0.04)),
        "num_leaves": _raw_int(config, "advantage_gate_num_leaves", _raw_int(config, "tree_ranker_num_leaves", 15)),
        "max_depth": _raw_int(config, "advantage_gate_max_depth", _raw_int(config, "tree_ranker_max_depth", 4)),
        "min_data_in_leaf": _raw_int(
            config,
            "advantage_gate_min_data_in_leaf",
            _raw_int(config, "tree_ranker_min_data_in_leaf", 20),
        ),
        "feature_fraction": _raw_float(
            config,
            "advantage_gate_feature_fraction",
            _raw_float(config, "tree_ranker_feature_fraction", 0.9),
        ),
        "bagging_fraction": _raw_float(
            config,
            "advantage_gate_bagging_fraction",
            _raw_float(config, "tree_ranker_bagging_fraction", 0.9),
        ),
        "bagging_freq": _raw_int(config, "advantage_gate_bagging_freq", _raw_int(config, "tree_ranker_bagging_freq", 1)),
        "seed": int(config.seed),
        "num_threads": _raw_int(config, "advantage_gate_nthread", _raw_int(config, "tree_ranker_nthread", 4)),
        "verbosity": -1,
    }
    params.update(_lightgbm_device_params(config))
    if task == "binary" and _raw_bool(config, "advantage_gate_auto_scale_pos_weight", True):
        params["scale_pos_weight"] = _pos_weight(train[target_key])

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=_raw_int(
            config,
            "advantage_gate_num_boost_round",
            _raw_int(config, "tree_ranker_num_boost_round", 160),
        ),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    booster.save_model(str(model_path))
    return booster


def _train_advantage_model(
    *,
    backend: str,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
    target_key: str,
    task: str,
) -> Any:
    if backend == "xgboost":
        return _train_advantage_xgboost(
            train=train,
            val=val,
            config=config,
            model_path=model_path,
            target_key=target_key,
            task=task,
        )
    if backend == "lightgbm":
        return _train_advantage_lightgbm(
            train=train,
            val=val,
            config=config,
            model_path=model_path,
            target_key=target_key,
            task=task,
        )
    raise ValueError(f"Unsupported tree_ranker_backend: {backend}")


def _predict_advantage(backend: str, model: Any, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if backend == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(x, feature_names=ADVANTAGE_FEATURE_NAMES)
        return np.asarray(model.predict(matrix), dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def _advantage_gate_metrics(
    *,
    dataset: dict[str, Any],
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    config: ExperimentConfig,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    metadata = dataset["metadata"].copy()
    if metadata.empty:
        return {
            "groups": 0,
            "rows": 0,
            "win_positive_rate": None,
            "loss_positive_rate": None,
            "candidate_gate_pass_rate": None,
            "candidate_win_precision_when_passed": None,
            "candidate_loss_rate_when_passed": None,
            "mean_candidate_accepted_delta_when_passed": None,
            "override_rate_vs_base": None,
            "mean_selected_accepted_delta_vs_base": None,
        }
    thresholds = dict(_advantage_gate_thresholds(config) if thresholds is None else thresholds)
    metadata["win_prob"] = np.asarray(win_prob, dtype=np.float32)
    metadata["loss_prob"] = np.asarray(loss_prob, dtype=np.float32)
    metadata["delta_pred"] = np.asarray(delta_pred, dtype=np.float32)
    metadata["gate_score"] = _advantage_score(
        win_prob=metadata["win_prob"].to_numpy(dtype=np.float32),
        loss_prob=metadata["loss_prob"].to_numpy(dtype=np.float32),
        delta_pred=metadata["delta_pred"].to_numpy(dtype=np.float32),
        ranker_margin=metadata["ranker_margin"].to_numpy(dtype=np.float32),
        thresholds=thresholds,
    )
    metadata["passes_gate"] = (
        (metadata["win_prob"] >= float(thresholds["min_win_prob"]))
        & (metadata["loss_prob"] <= float(thresholds["max_loss_prob"]))
        & (metadata["delta_pred"] >= float(thresholds["min_delta_pred"]))
    )

    passed = metadata[metadata["passes_gate"]]
    selected_rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        passed_group = group[group["passes_gate"]]
        if passed_group.empty:
            selected_rows.append(
                {
                    "override": False,
                    "selected_accepted_delta_vs_base": 0,
                    "selected_reward_delta_vs_base": 0.0,
                    "selected_utility_delta_vs_base": 0.0,
                    "selected_loss": False,
                    "selected_win": False,
                }
            )
            continue
        selected = passed_group.sort_values(["gate_score", "candidate_index"], ascending=[False, True]).iloc[0]
        selected_rows.append(
            {
                "override": True,
                "selected_accepted_delta_vs_base": int(selected["accepted_delta_vs_base"]),
                "selected_reward_delta_vs_base": float(selected["future_env_reward_delta_vs_base"]),
                "selected_utility_delta_vs_base": float(selected["utility_delta_vs_base"]),
                "selected_loss": bool(selected["is_loss"]),
                "selected_win": bool(selected["is_win"]),
            }
        )
    selected_table = pd.DataFrame(selected_rows)
    return {
        "groups": int(metadata["group_id"].nunique()),
        "rows": int(len(metadata)),
        "win_positive_rows": int(metadata["is_win"].sum()),
        "win_positive_rate": float(metadata["is_win"].mean()),
        "loss_positive_rows": int(metadata["is_loss"].sum()),
        "loss_positive_rate": float(metadata["is_loss"].mean()),
        "candidate_gate_pass_rows": int(len(passed)),
        "candidate_gate_pass_rate": float(metadata["passes_gate"].mean()),
        "candidate_win_precision_when_passed": None if passed.empty else float(passed["is_win"].mean()),
        "candidate_loss_rate_when_passed": None if passed.empty else float(passed["is_loss"].mean()),
        "mean_candidate_accepted_delta_when_passed": None
        if passed.empty
        else float(passed["accepted_delta_vs_base"].mean()),
        "mean_candidate_reward_delta_when_passed": None
        if passed.empty
        else float(passed["future_env_reward_delta_vs_base"].mean()),
        "override_rate_vs_base": float(selected_table["override"].mean()),
        "selected_win_rate_when_overridden": None
        if not bool(selected_table["override"].any())
        else float(selected_table[selected_table["override"]]["selected_win"].mean()),
        "selected_loss_rate_when_overridden": None
        if not bool(selected_table["override"].any())
        else float(selected_table[selected_table["override"]]["selected_loss"].mean()),
        "mean_selected_accepted_delta_vs_base": float(selected_table["selected_accepted_delta_vs_base"].mean()),
        "total_selected_accepted_delta_vs_base": int(selected_table["selected_accepted_delta_vs_base"].sum()),
        "mean_selected_reward_delta_vs_base": float(selected_table["selected_reward_delta_vs_base"].mean()),
        "mean_selected_utility_delta_vs_base": float(selected_table["selected_utility_delta_vs_base"].mean()),
        "thresholds": thresholds,
    }


def _fast_gate_selection_metrics(
    *,
    metadata: pd.DataFrame,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if metadata.empty:
        return {
            "candidate_gate_pass_rows": 0,
            "override_count": 0,
            "override_rate_vs_base": 0.0,
            "selected_loss_rate_when_overridden": None,
            "selected_loss_count": 0,
            "candidate_win_precision_when_passed": None,
            "total_selected_accepted_delta_vs_base": 0,
            "mean_selected_reward_delta_vs_base": 0.0,
        }
    group_codes = pd.Categorical(metadata["group_id"], ordered=False).codes.astype(np.int32)
    group_count = int(np.max(group_codes)) + 1 if group_codes.size else 0
    ranker_margin = metadata["ranker_margin"].to_numpy(dtype=np.float32)
    candidate_index = metadata["candidate_index"].to_numpy(dtype=np.int32)
    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    reward_delta = metadata["future_env_reward_delta_vs_base"].to_numpy(dtype=np.float32)
    is_win = metadata["is_win"].to_numpy(dtype=bool)
    is_loss = metadata["is_loss"].to_numpy(dtype=bool)
    gate_score = _advantage_score(
        win_prob=win_prob,
        loss_prob=loss_prob,
        delta_pred=delta_pred,
        ranker_margin=ranker_margin,
        thresholds=thresholds,
    )
    passed = (
        (np.asarray(win_prob, dtype=np.float32) >= float(thresholds["min_win_prob"]))
        & (np.asarray(loss_prob, dtype=np.float32) <= float(thresholds["max_loss_prob"]))
        & (np.asarray(delta_pred, dtype=np.float32) >= float(thresholds["min_delta_pred"]))
    )
    passed_indices = np.flatnonzero(passed)
    if passed_indices.size == 0 or group_count == 0:
        return {
            "candidate_gate_pass_rows": int(passed_indices.size),
            "override_count": 0,
            "override_rate_vs_base": 0.0,
            "selected_loss_rate_when_overridden": None,
            "selected_loss_count": 0,
            "candidate_win_precision_when_passed": None,
            "total_selected_accepted_delta_vs_base": 0,
            "mean_selected_reward_delta_vs_base": 0.0,
        }
    order = np.lexsort(
        (
            candidate_index[passed_indices],
            -gate_score[passed_indices],
            group_codes[passed_indices],
        )
    )
    sorted_indices = passed_indices[order]
    sorted_groups = group_codes[sorted_indices]
    first = np.ones((sorted_indices.size,), dtype=bool)
    first[1:] = sorted_groups[1:] != sorted_groups[:-1]
    selected = sorted_indices[first]
    override_count = int(selected.size)
    selected_loss_count = int(is_loss[selected].sum()) if override_count else 0
    return {
        "candidate_gate_pass_rows": int(passed_indices.size),
        "override_count": int(override_count),
        "override_rate_vs_base": float(override_count / max(group_count, 1)),
        "selected_loss_rate_when_overridden": float(is_loss[selected].mean()) if override_count else None,
        "selected_loss_count": int(selected_loss_count),
        "candidate_win_precision_when_passed": float(is_win[passed_indices].mean()),
        "total_selected_accepted_delta_vs_base": int(np.sum(accepted_delta[selected]).round()),
        "mean_selected_reward_delta_vs_base": float(np.sum(reward_delta[selected]) / max(group_count, 1)),
    }


def _no_override_advantage_thresholds(base: dict[str, float]) -> dict[str, float]:
    thresholds = dict(base)
    thresholds.update(
        {
            "min_win_prob": 1.000001,
            "max_loss_prob": -0.000001,
            "min_delta_pred": max(float(base.get("min_delta_pred", 0.0)), 1.0e9),
            "fallback_no_override": 1.0,
            "tune_found_feasible": 0.0,
        }
    )
    return thresholds


def _tune_advantage_thresholds(
    *,
    dataset: dict[str, Any],
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float]:
    base = _advantage_gate_thresholds(config)
    if not _raw_bool(config, "advantage_gate_auto_tune_thresholds", True) or dataset["metadata"].empty:
        return base

    max_loss_rate = _raw_float(config, "advantage_gate_tune_max_loss_rate", 0.0)
    min_override_rate = _raw_float(config, "advantage_gate_tune_min_override_rate", 0.0)
    min_override_count = _raw_int(config, "advantage_gate_tune_min_override_count", 1)
    min_total_delta = _raw_float(config, "advantage_gate_tune_min_total_delta", 0.0)
    best_thresholds: dict[str, float] | None = None
    best_key: tuple[float, float, float, float, float] | None = None

    for min_win_prob in _raw_float_grid(config, "advantage_gate_tune_min_win_prob_grid", (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)):
        for max_loss_prob in _raw_float_grid(config, "advantage_gate_tune_max_loss_prob_grid", (0.01, 0.02, 0.04, 0.08, 0.12)):
            for min_delta_pred in _raw_float_grid(config, "advantage_gate_tune_min_delta_pred_grid", (-0.02, 0.0, 0.01, 0.03, 0.05, 0.10)):
                thresholds = dict(base)
                thresholds.update(
                    {
                        "min_win_prob": float(min_win_prob),
                        "max_loss_prob": float(max_loss_prob),
                        "min_delta_pred": float(min_delta_pred),
                    }
                )
                metrics = _fast_gate_selection_metrics(
                    metadata=dataset["metadata"],
                    win_prob=win_prob,
                    loss_prob=loss_prob,
                    delta_pred=delta_pred,
                    thresholds=thresholds,
                )
                candidate_pass_rows = int(metrics.get("candidate_gate_pass_rows") or 0)
                override_count = int(metrics.get("override_count") or 0)
                override_rate = float(metrics.get("override_rate_vs_base") or 0.0)
                loss_rate = metrics.get("selected_loss_rate_when_overridden")
                loss_rate_value = 0.0 if loss_rate is None else float(loss_rate)
                if candidate_pass_rows <= 0:
                    continue
                if override_count < int(min_override_count):
                    continue
                if override_rate < float(min_override_rate):
                    continue
                if loss_rate_value > float(max_loss_rate):
                    continue
                total_delta = float(metrics.get("total_selected_accepted_delta_vs_base") or 0.0)
                if total_delta <= float(min_total_delta):
                    continue
                mean_reward_delta = float(metrics.get("mean_selected_reward_delta_vs_base") or 0.0)
                candidate_precision = metrics.get("candidate_win_precision_when_passed")
                precision_value = 0.0 if candidate_precision is None else float(candidate_precision)
                key = (total_delta, override_rate, mean_reward_delta, precision_value, -loss_rate_value)
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = thresholds
    if best_thresholds is None:
        return _no_override_advantage_thresholds(base)
    best_thresholds = dict(best_thresholds)
    best_thresholds.update({"fallback_no_override": 0.0, "tune_found_feasible": 1.0})
    return best_thresholds


def _tree_ranker_guard_defaults(mode: str) -> dict[str, float | int | bool | str]:
    normalized = str(mode or "strict").strip().lower().replace("_", "-")
    if normalized in {"emergency", "emergency-risk", "extreme", "extreme-risk"}:
        return {
            "mode": "emergency",
            "check_fragmentation": False,
            "check_small_gap": False,
            "check_lmax": False,
            "check_qot_margin": True,
            "check_energy": True,
            "check_delay": True,
            "fragmentation_slack": 0.50,
            "small_gap_slack": 1.0,
            "lmax_slack_slots": 40,
            "qot_margin_slack": 0.25,
            "energy_slack_w": 480.0,
            "delay_slack_ms": 10.0,
        }
    return {
        "mode": "strict",
        "check_fragmentation": True,
        "check_small_gap": True,
        "check_lmax": True,
        "check_qot_margin": True,
        "check_energy": True,
        "check_delay": True,
        "fragmentation_slack": 0.04,
        "small_gap_slack": 0.04,
        "lmax_slack_slots": 6,
        "qot_margin_slack": 0.12,
        "energy_slack_w": 120.0,
        "delay_slack_ms": 1.5,
    }


def _safety_guard_from_config(config: ExperimentConfig) -> dict[str, float | int | bool | str]:
    mode = _raw_str(config, "tree_ranker_safety_guard_mode", "strict")
    defaults = _tree_ranker_guard_defaults(mode)
    return {
        "enabled": _raw_bool(config, "tree_ranker_safety_guard", True),
        "mode": str(defaults["mode"]),
        "check_fragmentation": _raw_bool(
            config,
            "tree_ranker_guard_check_fragmentation",
            bool(defaults["check_fragmentation"]),
        ),
        "check_small_gap": _raw_bool(config, "tree_ranker_guard_check_small_gap", bool(defaults["check_small_gap"])),
        "check_lmax": _raw_bool(config, "tree_ranker_guard_check_lmax", bool(defaults["check_lmax"])),
        "check_qot_margin": _raw_bool(
            config,
            "tree_ranker_guard_check_qot_margin",
            bool(defaults["check_qot_margin"]),
        ),
        "check_energy": _raw_bool(config, "tree_ranker_guard_check_energy", bool(defaults["check_energy"])),
        "check_delay": _raw_bool(config, "tree_ranker_guard_check_delay", bool(defaults["check_delay"])),
        "fragmentation_slack": _raw_float(
            config,
            "tree_ranker_guard_fragmentation_slack",
            float(defaults["fragmentation_slack"]),
        ),
        "small_gap_slack": _raw_float(config, "tree_ranker_guard_small_gap_slack", float(defaults["small_gap_slack"])),
        "lmax_slack_slots": _raw_int(config, "tree_ranker_guard_lmax_slack_slots", int(defaults["lmax_slack_slots"])),
        "qot_margin_slack": _raw_float(
            config,
            "tree_ranker_guard_qot_margin_slack",
            float(defaults["qot_margin_slack"]),
        ),
        "energy_slack_w": _raw_float(config, "tree_ranker_guard_energy_slack_w", float(defaults["energy_slack_w"])),
        "delay_slack_ms": _raw_float(config, "tree_ranker_guard_delay_slack_ms", float(defaults["delay_slack_ms"])),
    }


def _train_backend(
    *,
    backend: str,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
) -> Any:
    if backend == "xgboost":
        return _train_xgboost(train=train, val=val, config=config, model_path=model_path)
    if backend == "lightgbm":
        return _train_lightgbm(train=train, val=val, config=config, model_path=model_path)
    raise ValueError(f"Unsupported tree_ranker_backend: {backend}")


def run_train_dagger_tree_ranker(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_dagger_tree_ranker requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)
    solver = GnnCnnDqnOngSolver(_solver_config(config))

    backend = _raw_str(config, "tree_ranker_backend", "lightgbm").strip().lower()
    model_suffix = "json" if backend == "xgboost" else "txt"
    model_path = run_path / f"{backend}_dagger_tree_ranker.{model_suffix}"
    ranker_path = run_path / "tree_ranker.json"
    base_policy = _raw_str(config, "dagger_base_policy", "energy-aware-ksp-bm-ff")
    selection_mode = _raw_str(config, "tree_ranker_selection_mode", "guarded")
    dagger_follow_selection_mode = _raw_str(
        config,
        "dagger_follow_selection_mode",
        "guarded" if str(selection_mode).strip().lower() in {"advantage", "positive_advantage"} else selection_mode,
    )
    residual_beta = _raw_float(config, "tree_ranker_residual_beta", 0.05)
    selection_margin = _raw_float(config, "tree_ranker_selection_margin", 0.005)
    safety_guard = _safety_guard_from_config(config)

    iterations = max(1, _raw_int(config, "dagger_iterations", 1))
    follow_trained = _raw_bool(config, "dagger_follow_trained_ranker", True)
    train_split = _raw_str(config, "dagger_train_split", "train")
    eval_split = _raw_str(config, "dagger_eval_split", "val")
    train_parts: list[dict[str, Any]] = []
    ranker: TreeCandidateRanker | None = None
    group_id = 0
    history: list[dict[str, Any]] = []

    for iteration in range(iterations):
        state_policy = base_policy if iteration == 0 or not follow_trained or ranker is None else "tree_ranker"
        part = _collect_examples(
            config=config,
            split=train_split,
            solver=solver,
            run_path=run_path,
            iteration=iteration,
            state_policy=state_policy,
            tree_ranker=ranker,
            group_id_start=group_id,
            max_episodes_key="dagger_train_max_episodes",
            max_requests_key="dagger_train_max_requests_per_episode",
        )
        group_id = int(part["next_group_id"])
        train_parts.append(part)
        train_data = _merge_datasets(train_parts)
        if train_data["x"].shape[0] == 0:
            raise ValueError("No DAgger tree-ranker examples were generated")

        ranker_model = _train_backend(
            backend=backend,
            train=train_data,
            val=None,
            config=config,
            model_path=model_path,
        )
        train_scores = _predict_ranker(backend, ranker_model, train_data["x"])
        train_metrics = _ranker_metrics(train_data["metadata"], train_scores)
        ranker = TreeCandidateRanker(
            backend=backend,
            model=ranker_model,
            selection_mode=dagger_follow_selection_mode,
            residual_beta=residual_beta,
            selection_margin=selection_margin,
            base_policy=base_policy,
            safety_guard=safety_guard,
        )
        history.append(
            {
                "iteration": int(iteration),
                "state_policy": state_policy,
                "train_groups": int(train_data["group_sizes"].size),
                "train_rows": int(train_data["x"].shape[0]),
                "train": train_metrics,
            }
        )
        print(json.dumps({"event": "dagger_tree_ranker_iteration", **history[-1]}, sort_keys=True), flush=True)

    train_data = _merge_datasets(train_parts)
    eval_data: dict[str, Any] | None = None
    eval_metrics: dict[str, Any] | None = None
    eval_scores: np.ndarray | None = None
    if eval_split:
        eval_data = _collect_examples(
            config=config,
            split=eval_split,
            solver=solver,
            run_path=run_path,
            iteration=0,
            state_policy=base_policy,
            tree_ranker=None,
            group_id_start=group_id,
            max_episodes_key="dagger_eval_max_episodes",
            max_requests_key="dagger_eval_max_requests_per_episode",
        )
        if eval_data["x"].shape[0] > 0:
            final_model = _train_backend(
                backend=backend,
                train=train_data,
                val=eval_data,
                config=config,
                model_path=model_path,
            )
            eval_scores = _predict_ranker(backend, final_model, eval_data["x"])
            eval_metrics = _ranker_metrics(eval_data["metadata"], eval_scores)
        else:
            final_model = _train_backend(
                backend=backend,
                train=train_data,
                val=None,
                config=config,
                model_path=model_path,
            )
    else:
        final_model = _train_backend(
            backend=backend,
            train=train_data,
            val=None,
            config=config,
            model_path=model_path,
        )

    train_scores = _predict_ranker(backend, final_model, train_data["x"])
    final_train_metrics = _ranker_metrics(train_data["metadata"], train_scores)
    train_data["metadata"].to_csv(run_path / "train_dagger_tree_ranker_examples.csv", index=False)
    counterfactual_mining: dict[str, Any] = {
        "train": _write_counterfactual_mining_outputs(
            data=train_data,
            config=config,
            run_path=run_path,
            prefix="train",
        )
    }
    np.savez_compressed(
        run_path / "train_dagger_tree_ranker_examples.npz",
        features=train_data["x"].astype(np.float32),
        targets=train_data["y"].astype(np.float32),
        group_sizes=train_data["group_sizes"].astype(np.int32),
        feature_names=np.asarray(OVERRIDE_FEATURE_NAMES, dtype=object),
    )
    if eval_data is not None and eval_data["x"].shape[0] > 0:
        if eval_scores is None:
            eval_scores = _predict_ranker(backend, final_model, eval_data["x"])
            if eval_metrics is None:
                eval_metrics = _ranker_metrics(eval_data["metadata"], eval_scores)
        eval_data["metadata"].to_csv(run_path / "eval_dagger_tree_ranker_examples.csv", index=False)
        counterfactual_mining["eval"] = _write_counterfactual_mining_outputs(
            data=eval_data,
            config=config,
            run_path=run_path,
            prefix="eval",
        )
        np.savez_compressed(
            run_path / "eval_dagger_tree_ranker_examples.npz",
            features=eval_data["x"].astype(np.float32),
            targets=eval_data["y"].astype(np.float32),
            group_sizes=eval_data["group_sizes"].astype(np.int32),
            feature_names=np.asarray(OVERRIDE_FEATURE_NAMES, dtype=object),
        )

    advantage_gate_meta: dict[str, Any] = {"enabled": False}
    advantage_metrics: dict[str, Any] | None = None
    if _advantage_gate_enabled(config, selection_mode):
        train_advantage = _build_advantage_dataset(data=train_data, ranker_scores=train_scores, config=config)
        eval_advantage = (
            _build_advantage_dataset(data=eval_data, ranker_scores=eval_scores, config=config)
            if eval_data is not None and eval_scores is not None and eval_data["x"].shape[0] > 0
            else None
        )
        if train_advantage["x"].shape[0] == 0:
            raise ValueError("No advantage gate examples were generated")

        train_advantage["metadata"].to_csv(run_path / "train_advantage_gate_examples.csv", index=False)
        np.savez_compressed(
            run_path / "train_advantage_gate_examples.npz",
            features=train_advantage["x"].astype(np.float32),
            win_targets=train_advantage["win_y"].astype(np.float32),
            loss_targets=train_advantage["loss_y"].astype(np.float32),
            delta_targets=train_advantage["delta_y"].astype(np.float32),
            win_weights=_advantage_sample_weights(dataset=train_advantage, config=config, target_key="win_y"),
            loss_weights=_advantage_sample_weights(dataset=train_advantage, config=config, target_key="loss_y"),
            delta_weights=_advantage_sample_weights(dataset=train_advantage, config=config, target_key="delta_y"),
            feature_names=np.asarray(ADVANTAGE_FEATURE_NAMES, dtype=object),
        )
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0:
            eval_advantage["metadata"].to_csv(run_path / "eval_advantage_gate_examples.csv", index=False)
            np.savez_compressed(
                run_path / "eval_advantage_gate_examples.npz",
                features=eval_advantage["x"].astype(np.float32),
                win_targets=eval_advantage["win_y"].astype(np.float32),
                loss_targets=eval_advantage["loss_y"].astype(np.float32),
                delta_targets=eval_advantage["delta_y"].astype(np.float32),
                win_weights=_advantage_sample_weights(dataset=eval_advantage, config=config, target_key="win_y"),
                loss_weights=_advantage_sample_weights(dataset=eval_advantage, config=config, target_key="loss_y"),
                delta_weights=_advantage_sample_weights(dataset=eval_advantage, config=config, target_key="delta_y"),
                feature_names=np.asarray(ADVANTAGE_FEATURE_NAMES, dtype=object),
            )

        win_model_path = run_path / f"{backend}_advantage_win.{model_suffix}"
        loss_model_path = run_path / f"{backend}_advantage_loss.{model_suffix}"
        delta_model_path = run_path / f"{backend}_advantage_delta.{model_suffix}"
        win_model = _train_advantage_model(
            backend=backend,
            train=train_advantage,
            val=eval_advantage,
            config=config,
            model_path=win_model_path,
            target_key="win_y",
            task="binary",
        )
        loss_model = _train_advantage_model(
            backend=backend,
            train=train_advantage,
            val=eval_advantage,
            config=config,
            model_path=loss_model_path,
            target_key="loss_y",
            task="binary",
        )
        delta_model = _train_advantage_model(
            backend=backend,
            train=train_advantage,
            val=eval_advantage,
            config=config,
            model_path=delta_model_path,
            target_key="delta_y",
            task="regression",
        )

        train_win_prob = _predict_advantage(backend, win_model, train_advantage["x"])
        train_loss_prob = _predict_advantage(backend, loss_model, train_advantage["x"])
        train_delta_pred = _predict_advantage(backend, delta_model, train_advantage["x"])
        eval_win_prob = None
        eval_loss_prob = None
        eval_delta_pred = None
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0:
            eval_win_prob = _predict_advantage(backend, win_model, eval_advantage["x"])
            eval_loss_prob = _predict_advantage(backend, loss_model, eval_advantage["x"])
            eval_delta_pred = _predict_advantage(backend, delta_model, eval_advantage["x"])

        tuned_thresholds = _tune_advantage_thresholds(
            dataset=eval_advantage if eval_advantage is not None and eval_advantage["x"].shape[0] > 0 else train_advantage,
            win_prob=eval_win_prob if eval_win_prob is not None else train_win_prob,
            loss_prob=eval_loss_prob if eval_loss_prob is not None else train_loss_prob,
            delta_pred=eval_delta_pred if eval_delta_pred is not None else train_delta_pred,
            config=config,
        )
        train_advantage_metrics = _advantage_gate_metrics(
            dataset=train_advantage,
            win_prob=train_win_prob,
            loss_prob=train_loss_prob,
            delta_pred=train_delta_pred,
            config=config,
            thresholds=tuned_thresholds,
        )
        eval_advantage_metrics = None
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0:
            eval_advantage_metrics = _advantage_gate_metrics(
                dataset=eval_advantage,
                win_prob=eval_win_prob if eval_win_prob is not None else np.zeros((0,), dtype=np.float32),
                loss_prob=eval_loss_prob if eval_loss_prob is not None else np.zeros((0,), dtype=np.float32),
                delta_pred=eval_delta_pred if eval_delta_pred is not None else np.zeros((0,), dtype=np.float32),
                config=config,
                thresholds=tuned_thresholds,
            )
        advantage_gate_meta = {
            "enabled": True,
            "backend": backend,
            "feature_names": list(ADVANTAGE_FEATURE_NAMES),
            "win_model_path": str(win_model_path),
            "loss_model_path": str(loss_model_path),
            "delta_model_path": str(delta_model_path),
            "win_min_accepted_delta": _raw_int(config, "advantage_gate_win_min_accepted_delta", 1),
            "loss_min_accepted_delta": _raw_int(config, "advantage_gate_loss_min_accepted_delta", 1),
            "delta_target_mode": _raw_str(config, "advantage_gate_delta_target_mode", "accepted_tiebreak"),
            "delta_tiebreak_weight": _raw_float(config, "advantage_gate_delta_tiebreak_weight", 1.0),
            "delta_reward_weight": _raw_float(config, "advantage_gate_delta_reward_weight", 0.0),
            "sample_weights": {
                "enabled": _raw_bool(config, "advantage_gate_use_sample_weights", True),
                "neutral_weight": _raw_float(config, "advantage_gate_neutral_weight", 0.35),
                "win_positive_weight": _raw_float(config, "advantage_gate_win_positive_weight", 8.0),
                "win_hard_negative_weight": _raw_float(config, "advantage_gate_win_hard_negative_weight", 2.0),
                "loss_positive_weight": _raw_float(config, "advantage_gate_loss_positive_weight", 10.0),
                "loss_hard_positive_weight": _raw_float(config, "advantage_gate_loss_hard_positive_weight", 2.0),
                "delta_hard_positive_weight": _raw_float(config, "advantage_gate_delta_hard_positive_weight", 6.0),
                "delta_hard_negative_weight": _raw_float(config, "advantage_gate_delta_hard_negative_weight", 8.0),
                "delta_neutral_weight": _raw_float(config, "advantage_gate_delta_neutral_weight", 0.75),
                "group_normalize": _raw_bool(config, "advantage_gate_group_normalize_weights", True),
            },
            "auto_tuned_thresholds": _raw_bool(config, "advantage_gate_auto_tune_thresholds", True),
            "tuning_constraints": {
                "max_loss_rate": _raw_float(config, "advantage_gate_tune_max_loss_rate", 0.0),
                "min_override_rate": _raw_float(config, "advantage_gate_tune_min_override_rate", 0.0),
                "min_override_count": _raw_int(config, "advantage_gate_tune_min_override_count", 1),
                "min_total_delta": _raw_float(config, "advantage_gate_tune_min_total_delta", 0.0),
            },
            **tuned_thresholds,
        }
        advantage_metrics = {
            "train": train_advantage_metrics,
            "eval": eval_advantage_metrics,
            "train_rows": int(train_advantage["x"].shape[0]),
            "eval_rows": None if eval_advantage is None else int(eval_advantage["x"].shape[0]),
        }
        print(
            json.dumps(
                {
                    "event": "dagger_tree_ranker_advantage_gate",
                    "train": train_advantage_metrics,
                    "eval": eval_advantage_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ranker_meta = {
        "backend": backend,
        "model_path": str(model_path),
        "feature_names": list(OVERRIDE_FEATURE_NAMES),
        "candidate_pool": "all_topn",
        "selection_mode": selection_mode,
        "residual_beta": float(residual_beta),
        "selection_margin": float(selection_margin),
        "base_policy": base_policy,
        "safety_guard": safety_guard,
        "advantage_gate": advantage_gate_meta,
        "dagger": {
            "iterations": int(iterations),
            "follow_trained_ranker": bool(follow_trained),
            "follow_selection_mode": dagger_follow_selection_mode,
            "train_split": train_split,
            "eval_split": eval_split,
            "lookahead_horizon": _raw_int(config, "dagger_lookahead_horizon", 12),
            "lookahead_rollout_policy": _raw_str(config, "dagger_lookahead_rollout_policy", base_policy),
            "rank_target_mode": _raw_str(config, "dagger_rank_target_mode", "shifted_utility"),
        },
        "utility": {
            "mode": _raw_str(config, "dagger_utility_mode", "linear"),
            "accepted_weight": _raw_float(config, "dagger_utility_accepted_weight", 2.0),
            "block_penalty": _raw_float(config, "dagger_utility_block_penalty", 1.5),
            "energy_weight": _raw_float(config, "dagger_utility_energy_weight", 0.25),
            "fragmentation_weight": _raw_float(config, "dagger_utility_fragmentation_weight", 0.80),
            "qot_weight": _raw_float(config, "dagger_utility_qot_weight", 0.20),
            "qot_clip_min": _raw_float(config, "dagger_utility_qot_clip_min", 0.0),
            "qot_clip_max": _raw_float(config, "dagger_utility_qot_clip_max", 1.0),
            "energy_norm_w": _raw_float(config, "dagger_utility_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)),
        },
    }
    _write_json(ranker_path, ranker_meta)

    metrics = {
        "stage": "train_dagger_tree_ranker",
        "dataset_path": str(config.dataset_path),
        "ong_source_path": ong_source,
        "backend": backend,
        "model_path": str(model_path),
        "ranker_path": str(ranker_path),
        "base_policy": base_policy,
        "selection_mode": selection_mode,
        "safety_guard": safety_guard,
        "history": history,
        "train": final_train_metrics,
        "eval": eval_metrics,
        "train_groups": int(train_data["group_sizes"].size),
        "train_rows": int(train_data["x"].shape[0]),
        "eval_groups": None if eval_data is None else int(eval_data["group_sizes"].size),
        "eval_rows": None if eval_data is None else int(eval_data["x"].shape[0]),
        "counterfactual_mining": counterfactual_mining,
        "advantage_gate": advantage_metrics,
        "ranker": ranker_meta,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
