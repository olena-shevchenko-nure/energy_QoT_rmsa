from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.dagger_tree_ranker import _simulate_candidate
from cse2026.experiments.eon.lookahead_oracle import _solver_config
from cse2026.experiments.eon.lookahead_override_features import candidate_feature_matrix
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_int,
    _raw_str,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.tree_ranker_runtime import (
    DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES,
    TreeCandidateRanker,
    _append_runtime_features,
    select_tree_base_index,
)
from cse2026.ong_solver import GnnCnnDqnOngSolver


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


def _proposal_for_state(
    *,
    batch: Any,
    ranker: TreeCandidateRanker,
    n_max: int,
) -> dict[str, Any] | None:
    base_index = select_tree_base_index(batch, n_max, ranker.base_policy)
    if base_index < 0:
        return None
    candidate_indices = ranker._candidate_indices(batch, base_index)
    features, kept_indices = candidate_feature_matrix(
        batch=batch,
        candidate_indices=candidate_indices,
        n_max=n_max,
        reference_index=base_index,
    )
    if not kept_indices or int(base_index) not in kept_indices:
        return None
    base_position = kept_indices.index(int(base_index))
    ranker_features = _append_runtime_features(
        features=features,
        kept_indices=kept_indices,
        base_index=base_index,
        feature_names=ranker.feature_names,
    )
    scores = ranker.scores(ranker_features)
    selected_index, selector_value = ranker.select_index(batch, n_max)
    if selected_index < 0 or int(selected_index) not in kept_indices:
        return None
    selected_position = kept_indices.index(int(selected_index))
    selected_score = float(scores[int(selected_position)])
    base_score = float(scores[int(base_position)])
    order = np.argsort(scores, kind="mergesort")
    best_position = int(order[-1])
    second_score = float(scores[int(order[-2])]) if len(order) > 1 else float(scores[best_position])
    best_score = float(scores[best_position])
    other_best = second_score if int(selected_position) == best_position else best_score
    margin = float(selected_score - base_score)
    gate_features = np.concatenate(
        [
            ranker_features[int(selected_position)].astype(np.float32),
            np.asarray(
                [
                    selected_score,
                    margin,
                    float(selected_score - other_best),
                    float(margin - ranker.selection_margin),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    return {
        "base_index": int(base_index),
        "selected_index": int(selected_index),
        "base_position": int(base_position),
        "selected_position": int(selected_position),
        "selected_score": selected_score,
        "base_score": base_score,
        "score_margin": margin,
        "selector_value": float(selector_value),
        "candidate_pool_size": int(len(kept_indices)),
        "gate_features": gate_features,
    }


def collect_online_overrides(
    *,
    config: ExperimentConfig,
    artifact_path: Path,
    output_dir: Path,
    max_collected_overrides: int,
) -> dict[str, Any]:
    _add_ong_source_path(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _raw_str(config, "rollout_split", "test")
    max_episodes = _raw_int(config, "max_episodes", 0)
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)
    base_policy = _raw_str(config, "online_hardcase_base_policy", "energy-aware-ksp-bm-ff")
    behavior_policy = _raw_str(config, "online_hardcase_behavior_policy", "proposal")
    rollout_policy = _raw_str(config, "online_hardcase_counterfactual_rollout_policy", base_policy)
    horizon = _raw_int(config, "online_hardcase_lookahead_horizon", _raw_int(config, "dagger_lookahead_horizon", 12))

    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    if max_episodes > 0:
        episode_ids = episode_ids[: int(max_episodes)]

    solver = GnnCnnDqnOngSolver(_solver_config(config))
    ranker = TreeCandidateRanker.load(artifact_path)
    if ranker.base_policy != base_policy:
        ranker.base_policy = str(base_policy)

    feature_names = list(ranker.feature_names) + list(DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES)
    rows: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    group_id = 0
    visited_requests = 0
    raw_override_count = 0
    no_candidate_requests = 0

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(int(max_requests_per_episode)).reset_index(drop=True)
        if episode.empty:
            continue
        traffic_path = _traffic_jsonl_for_episode(output_dir, f"online_hardcase_{episode_id}", episode)
        env = _make_env(
            episode_id=f"online_hardcase_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
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

            proposal = _proposal_for_state(batch=batch, ranker=ranker, n_max=solver.config.n_max)
            if proposal is None:
                selected_index = select_tree_base_index(batch, solver.config.n_max, base_policy)
                action = int(batch.topn[int(selected_index)].action) if selected_index >= 0 else int(solver.adapter(env).block_action(env))
                _observation, _reward, terminated, truncated, _info = env.step(action)
                visited_requests += 1
                if bool(terminated) or bool(truncated):
                    break
                continue

            base_index = int(proposal["base_index"])
            selected_index = int(proposal["selected_index"])
            did_override = int(selected_index) != int(base_index)
            if did_override:
                raw_override_count += 1
                base_candidate = batch.topn[int(base_index)]
                selected_candidate = batch.topn[int(selected_index)]
                selected_sim = _simulate_candidate(
                    env=env,
                    candidate=selected_candidate,
                    solver=solver,
                    rollout_policy=rollout_policy,
                    horizon=horizon,
                )
                base_sim = _simulate_candidate(
                    env=env,
                    candidate=base_candidate,
                    solver=solver,
                    rollout_policy=rollout_policy,
                    horizon=horizon,
                )
                accepted_delta = int(selected_sim["future_accepted"]) - int(base_sim["future_accepted"])
                reward_delta = float(selected_sim["future_env_reward"]) - float(base_sim["future_env_reward"])
                energy_delta = float(selected_sim["future_energy_increment_sum"]) - float(base_sim["future_energy_increment_sum"])
                features.append(np.asarray(proposal["gate_features"], dtype=np.float32))
                rows.append(
                    {
                        "group_id": int(group_id),
                        "episode_id": str(episode_id),
                        "request_id": int(request["request_id"]),
                        "position": int(position),
                        "traffic_scenario": str(request.get("traffic_scenario", "")),
                        "load_name": str(request.get("load_name", "")),
                        "seed": int(request["seed"]) if "seed" in request else None,
                        "base_policy": str(base_policy),
                        "proposal_artifact": str(artifact_path),
                        "behavior_policy": str(behavior_policy),
                        "counterfactual_rollout_policy": str(rollout_policy),
                        "lookahead_horizon": int(horizon),
                        "valid_candidates": int(valid.size),
                        "candidate_pool_size": int(proposal["candidate_pool_size"]),
                        "base_index": int(base_index),
                        "selected_index": int(selected_index),
                        "selected_score": float(proposal["selected_score"]),
                        "base_score": float(proposal["base_score"]),
                        "score_margin": float(proposal["score_margin"]),
                        "selector_value": float(proposal["selector_value"]),
                        "accepted_delta_vs_base": int(accepted_delta),
                        "future_env_reward_delta_vs_base": float(reward_delta),
                        "future_energy_increment_delta_vs_base": float(energy_delta),
                        "selected_future_accepted": int(selected_sim["future_accepted"]),
                        "base_future_accepted": int(base_sim["future_accepted"]),
                        "selected_future_blocked": int(selected_sim["future_blocked"]),
                        "base_future_blocked": int(base_sim["future_blocked"]),
                        "selected_future_reward": float(selected_sim["future_env_reward"]),
                        "base_future_reward": float(base_sim["future_env_reward"]),
                        "selected_energy_increment": float(selected_candidate.energy_increment),
                        "base_energy_increment": float(base_candidate.energy_increment),
                        "selected_fragmentation_after": float(selected_candidate.fragmentation_after),
                        "base_fragmentation_after": float(base_candidate.fragmentation_after),
                        "selected_qot_margin_norm": float(selected_candidate.qot_margin_norm),
                        "base_qot_margin_norm": float(base_candidate.qot_margin_norm),
                    }
                )
                group_id += 1

            if behavior_policy.strip().lower() == "base":
                step_index = int(base_index)
            else:
                step_index = int(selected_index)
            action = int(batch.topn[int(step_index)].action)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            visited_requests += 1
            if max_collected_overrides > 0 and len(rows) >= int(max_collected_overrides):
                break
            if bool(terminated) or bool(truncated):
                break
        if max_collected_overrides > 0 and len(rows) >= int(max_collected_overrides):
            break

    metadata = pd.DataFrame(rows)
    x = np.vstack(features).astype(np.float32) if features else np.zeros((0, len(feature_names)), dtype=np.float32)
    metadata.to_csv(output_dir / "online_override_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "online_override_examples.npz",
        features=x,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    if not metadata.empty:
        accepted = metadata["accepted_delta_vs_base"].astype(float)
        label_summary = {
            "examples": int(len(metadata)),
            "wins": int((accepted > 0).sum()),
            "losses": int((accepted < 0).sum()),
            "ties": int((accepted == 0).sum()),
            "total_accepted_delta": int(round(float(accepted.sum()))),
            "loss_rate": float((accepted < 0).mean()),
            "win_rate": float((accepted > 0).mean()),
        }
    else:
        label_summary = {
            "examples": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "total_accepted_delta": 0,
            "loss_rate": None,
            "win_rate": None,
        }
    summary = {
        "artifact_path": str(artifact_path),
        "output_dir": str(output_dir),
        "dataset_path": str(config.dataset_path),
        "split": split,
        "episodes": list(episode_ids),
        "visited_requests": int(visited_requests),
        "raw_override_count": int(raw_override_count),
        "no_candidate_requests": int(no_candidate_requests),
        "feature_names": list(feature_names),
        "label_summary": label_summary,
    }
    _write_json(output_dir / "online_override_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect online override hard cases with paired counterfactual labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-collected-overrides", type=int, default=0)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = collect_online_overrides(
        config=config,
        artifact_path=Path(args.artifact),
        output_dir=Path(args.output_dir),
        max_collected_overrides=int(args.max_collected_overrides),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
