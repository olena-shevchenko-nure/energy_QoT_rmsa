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
from cse2026.experiments.eon.neural_three_head_runtime import NeuralThreeHeadOverridePolicy
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _device,
    _make_env,
    _raw_int,
    _raw_str,
    _resolve_optional_path,
    _solver_config,
    _traffic_jsonl_for_episode,
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


def _distribution(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None}
    quantiles = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "mean": float(arr.mean()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
    }


def _candidate_record(prefix: str, candidate: Any) -> dict[str, Any]:
    return {
        f"{prefix}_route_id": int(candidate.route_id),
        f"{prefix}_b_start": int(candidate.b_start),
        f"{prefix}_width": int(candidate.w),
        f"{prefix}_modulation_index": int(candidate.modulation_index),
        f"{prefix}_modulation_name": str(candidate.modulation_name),
        f"{prefix}_energy_increment": float(candidate.energy_increment),
        f"{prefix}_energy_increment_norm": float(candidate.energy_increment_norm),
        f"{prefix}_fragmentation_after": float(candidate.fragmentation_after),
        f"{prefix}_delta_fragmentation": float(candidate.delta_fragmentation),
        f"{prefix}_largest_free_block_after": int(candidate.largest_free_block_after),
        f"{prefix}_small_gap_penalty": float(candidate.small_gap_penalty),
        f"{prefix}_qot_margin_norm": float(candidate.qot_margin_norm),
        f"{prefix}_qot_risk": float(candidate.qot_risk),
        f"{prefix}_delay_ms": float(candidate.delay_ms),
        f"{prefix}_j_total": float(candidate.j_total),
    }


def _prediction_record(prefix: str, pred: dict[str, np.ndarray], index: int) -> dict[str, float | int]:
    if index < 0:
        return {
            f"{prefix}_win_prob": 0.0,
            f"{prefix}_loss_prob": 1.0,
            f"{prefix}_delta_pred": 0.0,
            f"{prefix}_score": 0.0,
        }
    return {
        f"{prefix}_win_prob": float(pred["win_prob"][int(index)]),
        f"{prefix}_loss_prob": float(pred["loss_prob"][int(index)]),
        f"{prefix}_delta_pred": float(pred["delta_pred"][int(index)]),
        f"{prefix}_score": float(pred["score"][int(index)]),
    }


def _rank_desc(values: np.ndarray, valid_indices: np.ndarray, selected_index: int) -> int | None:
    if selected_index < 0 or valid_indices.size == 0:
        return None
    ordered = sorted((int(index) for index in valid_indices), key=lambda index: (-float(values[int(index)]), index))
    try:
        return int(ordered.index(int(selected_index)) + 1)
    except ValueError:
        return None


def _summary_for_rows(metadata: pd.DataFrame) -> dict[str, Any]:
    if metadata.empty:
        return {
            "examples": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "win_rate": None,
            "loss_rate": None,
            "total_accepted_delta": 0,
            "total_reward_delta": 0.0,
        }
    accepted = metadata["accepted_delta_vs_base"].astype(float)
    reward = metadata["future_env_reward_delta_vs_base"].astype(float)
    return {
        "examples": int(len(metadata)),
        "wins": int((accepted > 0).sum()),
        "losses": int((accepted < 0).sum()),
        "ties": int((accepted == 0).sum()),
        "win_rate": float((accepted > 0).mean()),
        "loss_rate": float((accepted < 0).mean()),
        "tie_rate": float((accepted == 0).mean()),
        "total_accepted_delta": int(round(float(accepted.sum()))),
        "mean_accepted_delta": float(accepted.mean()),
        "total_reward_delta": float(reward.sum()),
        "mean_reward_delta": float(reward.mean()),
        "accepted_delta_distribution": _distribution(accepted),
        "selected_win_prob_distribution": _distribution(metadata["selected_win_prob"].astype(float)),
        "selected_loss_prob_distribution": _distribution(metadata["selected_loss_prob"].astype(float)),
        "selected_delta_pred_distribution": _distribution(metadata["selected_delta_pred"].astype(float)),
    }


def _context_summary(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if metadata.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (scenario, load), group in metadata.groupby(["traffic_scenario", "load_name"], sort=True):
        metrics = _summary_for_rows(group)
        rows.append(
            {
                "traffic_scenario": str(scenario),
                "load_name": str(load),
                "examples": int(metrics["examples"]),
                "wins": int(metrics["wins"]),
                "losses": int(metrics["losses"]),
                "ties": int(metrics["ties"]),
                "win_rate": metrics["win_rate"],
                "loss_rate": metrics["loss_rate"],
                "total_accepted_delta": metrics["total_accepted_delta"],
                "total_reward_delta": metrics["total_reward_delta"],
            }
        )
    rows.sort(key=lambda row: (float(row["total_accepted_delta"]), str(row["traffic_scenario"]), str(row["load_name"])))
    return rows


def _veto_sweep(metadata: pd.DataFrame, *, min_kept: int) -> list[dict[str, Any]]:
    if metadata.empty:
        return []
    win_values = sorted(set([0.6, 0.65, 0.7, 0.75, 0.8, 0.85]))
    loss_values = sorted(set([0.20, 0.18, 0.16, 0.14, 0.12, 0.10]), reverse=True)
    delta_values = sorted(set([0.2, 0.4, 0.6, 0.8, 1.0]))
    rows: list[dict[str, Any]] = []
    for win_threshold in win_values:
        for loss_threshold in loss_values:
            for delta_margin in delta_values:
                keep = (
                    (metadata["selected_win_prob"].astype(float) >= float(win_threshold))
                    & (metadata["selected_loss_prob"].astype(float) <= float(loss_threshold))
                    & (metadata["selected_delta_pred"].astype(float) >= float(delta_margin))
                )
                kept = metadata[keep]
                if len(kept) < int(min_kept):
                    continue
                metrics = _summary_for_rows(kept)
                rows.append(
                    {
                        "win_threshold": float(win_threshold),
                        "loss_threshold": float(loss_threshold),
                        "delta_margin": float(delta_margin),
                        "kept": int(metrics["examples"]),
                        "wins": int(metrics["wins"]),
                        "losses": int(metrics["losses"]),
                        "loss_rate": metrics["loss_rate"],
                        "total_accepted_delta": metrics["total_accepted_delta"],
                        "total_reward_delta": metrics["total_reward_delta"],
                    }
                )
    rows.sort(
        key=lambda row: (
            float(row["total_accepted_delta"]),
            -float(row["losses"]),
            float(row["kept"]),
            float(row["win_threshold"]),
        ),
        reverse=True,
    )
    return rows[:20]


def audit_live_overrides(
    *,
    config: ExperimentConfig,
    checkpoint_path: Path,
    output_dir: Path,
    base_policy: str,
    rollout_policy: str,
    lookahead_horizon: int,
    max_collected_overrides: int,
    max_overrides_per_episode: int,
    progress_every: int,
) -> dict[str, Any]:
    _add_ong_source_path(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = _raw_str(config, "rollout_split", "test")
    max_episodes = _raw_int(config, "max_episodes", 0)
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)

    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    if max_episodes > 0:
        episode_ids = episode_ids[: int(max_episodes)]

    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    policy = NeuralThreeHeadOverridePolicy.load(checkpoint_path, device=_device(config), base_policy=base_policy)

    rows: list[dict[str, Any]] = []
    visited_requests = 0
    valid_states = 0
    no_candidate_requests = 0
    raw_override_count = 0
    skipped_episode_cap = 0

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(int(max_requests_per_episode)).reset_index(drop=True)
        if episode.empty:
            continue
        traffic_path = _traffic_jsonl_for_episode(output_dir, f"neural_three_head_live_{episode_id}", episode)
        env = _make_env(
            episode_id=f"neural_three_head_live_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        episode_overrides = 0

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
            details = policy.decision_details(batch, solver.config)
            pred = details["prediction"]
            base_index = int(details["base_index"])
            selected_index = int(details["selected_index"])
            if selected_index < 0:
                selected_index = int(valid[0])
            did_override = bool(details["override_applied"])

            if did_override:
                raw_override_count += 1
                should_collect = max_overrides_per_episode <= 0 or episode_overrides < int(max_overrides_per_episode)
                if should_collect:
                    episode_overrides += 1
                    base_candidate = batch.topn[int(base_index)]
                    selected_candidate = batch.topn[int(selected_index)]
                    selected_sim = _simulate_candidate(
                        env=env,
                        candidate=selected_candidate,
                        solver=solver,
                        rollout_policy=rollout_policy,
                        horizon=int(lookahead_horizon),
                    )
                    base_sim = _simulate_candidate(
                        env=env,
                        candidate=base_candidate,
                        solver=solver,
                        rollout_policy=rollout_policy,
                        horizon=int(lookahead_horizon),
                    )
                    accepted_delta = int(selected_sim["future_accepted"]) - int(base_sim["future_accepted"])
                    reward_delta = float(selected_sim["future_env_reward"]) - float(base_sim["future_env_reward"])
                    energy_delta = (
                        float(selected_sim["future_energy_increment_sum"])
                        - float(base_sim["future_energy_increment_sum"])
                    )
                    valid_indices = np.asarray(details["valid_indices"], dtype=np.int64)
                    record = {
                        "override_id": int(len(rows)),
                        "split": str(split),
                        "episode_id": str(episode_id),
                        "request_id": int(request["request_id"]),
                        "position": int(position),
                        "traffic_scenario": str(request.get("traffic_scenario", "")),
                        "load_name": str(request.get("load_name", "")),
                        "seed": int(request["seed"]) if "seed" in request else None,
                        "base_policy": str(base_policy),
                        "counterfactual_rollout_policy": str(rollout_policy),
                        "lookahead_horizon": int(lookahead_horizon),
                        "valid_candidates": int(valid.size),
                        "eligible_count": int(details["eligible_count"]),
                        "base_index": int(base_index),
                        "selected_index": int(selected_index),
                        "selected_score_rank": _rank_desc(np.asarray(pred["score"], dtype=np.float32), valid_indices, selected_index),
                        "base_score_rank": _rank_desc(np.asarray(pred["score"], dtype=np.float32), valid_indices, base_index),
                        "accepted_delta_vs_base": int(accepted_delta),
                        "future_env_reward_delta_vs_base": float(reward_delta),
                        "future_energy_increment_delta_vs_base": float(energy_delta),
                        "selected_future_accepted": int(selected_sim["future_accepted"]),
                        "base_future_accepted": int(base_sim["future_accepted"]),
                        "selected_future_blocked": int(selected_sim["future_blocked"]),
                        "base_future_blocked": int(base_sim["future_blocked"]),
                        "selected_future_reward": float(selected_sim["future_env_reward"]),
                        "base_future_reward": float(base_sim["future_env_reward"]),
                    }
                    record.update(_prediction_record("selected", pred, selected_index))
                    record.update(_prediction_record("base", pred, base_index))
                    record.update(_candidate_record("selected", selected_candidate))
                    record.update(_candidate_record("base", base_candidate))
                    rows.append(record)
                    if progress_every > 0 and (len(rows) == 1 or len(rows) % int(progress_every) == 0):
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "episode_id": str(episode_id),
                                    "visited_requests": int(visited_requests),
                                    "collected_overrides": int(len(rows)),
                                    "accepted_delta_sum": int(sum(int(row["accepted_delta_vs_base"]) for row in rows)),
                                },
                                sort_keys=True,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    skipped_episode_cap += 1

            action = int(batch.topn[int(selected_index)].action)
            _observation, _reward, terminated, truncated, _info = env.step(action)
            visited_requests += 1
            if max_collected_overrides > 0 and len(rows) >= int(max_collected_overrides):
                break
            if bool(terminated) or bool(truncated):
                break
        print(
            json.dumps(
                {
                    "event": "episode_done",
                    "episode_id": str(episode_id),
                    "episode_overrides": int(episode_overrides),
                    "collected_overrides": int(len(rows)),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        if max_collected_overrides > 0 and len(rows) >= int(max_collected_overrides):
            break

    metadata = pd.DataFrame(rows)
    metadata.to_csv(output_dir / "neural_three_head_live_override_audit.csv", index=False)
    label_summary = _summary_for_rows(metadata)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "dataset_path": str(config.dataset_path),
        "split": str(split),
        "episodes": list(episode_ids),
        "visited_requests": int(visited_requests),
        "valid_states": int(valid_states),
        "no_candidate_requests": int(no_candidate_requests),
        "raw_override_count": int(raw_override_count),
        "skipped_episode_cap": int(skipped_episode_cap),
        "base_policy": str(base_policy),
        "counterfactual_rollout_policy": str(rollout_policy),
        "lookahead_horizon": int(lookahead_horizon),
        "thresholds": dict(policy.thresholds),
        "label_summary": label_summary,
        "context_summary": _context_summary(metadata),
        "observed_veto_sweep": _veto_sweep(metadata, min_kept=max(5, min(20, int(len(metadata) * 0.05)))),
    }
    _write_json(output_dir / "neural_three_head_live_override_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit live neural three-head overrides with paired counterfactual labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-policy", default="energy-aware-ksp-bm-ff")
    parser.add_argument("--rollout-policy", default="")
    parser.add_argument("--lookahead-horizon", type=int, default=50)
    parser.add_argument("--max-collected-overrides", type=int, default=0)
    parser.add_argument("--max-overrides-per-episode", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _resolve_optional_path(config, "neural_three_head_override_path")
    if checkpoint is None:
        raise ValueError("--checkpoint or neural_three_head_override_path is required")
    rollout_policy = str(args.rollout_policy or args.base_policy)
    summary = audit_live_overrides(
        config=config,
        checkpoint_path=checkpoint,
        output_dir=Path(args.output_dir),
        base_policy=str(args.base_policy),
        rollout_policy=rollout_policy,
        lookahead_horizon=int(args.lookahead_horizon),
        max_collected_overrides=int(args.max_collected_overrides),
        max_overrides_per_episode=int(args.max_overrides_per_episode),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
