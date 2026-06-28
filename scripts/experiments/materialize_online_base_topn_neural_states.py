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
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_int,
    _raw_str,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.train_dqn import _batch_to_arrays
from cse2026.experiments.eon.tree_ranker_runtime import select_tree_base_index
from cse2026.ong_solver import GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax


STATE_KEYS = (
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


def _normalized_policy(policy: str) -> str:
    return str(policy or "").strip().lower().replace("_", "-")


def _uses_dqn_policy(*policies: str) -> bool:
    return any(_normalized_policy(policy) in {"gnn-cnn-dqn"} for policy in policies)


def _base_index(*, batch: Any, solver: GnnCnnDqnOngSolver, base_policy: str) -> int:
    if _normalized_policy(base_policy) == "gnn-cnn-dqn":
        return int(masked_argmax(solver.q_values(batch), batch.candidate_mask))
    return int(select_tree_base_index(batch, solver.config.n_max, base_policy))


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


def _group_label_vectors(group: pd.DataFrame, n_max: int) -> dict[str, np.ndarray]:
    accepted = np.full((n_max,), np.nan, dtype=np.float32)
    secondary = np.full((n_max,), np.nan, dtype=np.float32)
    target = np.full((n_max,), np.nan, dtype=np.float32)
    label_mask = np.zeros((n_max,), dtype=np.bool_)
    for row in group.itertuples(index=False):
        index = int(getattr(row, "candidate_index"))
        if not (0 <= index < n_max):
            continue
        accepted_delta = float(getattr(row, "accepted_delta_vs_base"))
        secondary_delta = float(getattr(row, "secondary_delta_vs_base"))
        accepted[index] = accepted_delta
        secondary[index] = secondary_delta
        target[index] = accepted_delta if accepted_delta != 0.0 else secondary_delta
        label_mask[index] = True
    return {
        "accepted_delta_vs_base": accepted,
        "secondary_delta_vs_base": secondary,
        "target_delta": target,
        "label_mask": label_mask,
    }


def materialize(
    *,
    config: ExperimentConfig,
    input_dir: Path,
    output_path: Path,
    base_policy: str,
    progress_every: int,
) -> dict[str, Any]:
    _add_ong_source_path(config)
    metadata_path = input_dir / "online_base_topn_examples.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    if metadata.empty:
        raise ValueError(f"Metadata is empty: {metadata_path}")
    metadata["is_base"] = metadata["is_base"].astype(bool)
    if not base_policy:
        base_policy = str(metadata["base_policy"].iloc[0]) if "base_policy" in metadata else "gnn_cnn_dqn"

    split = _raw_str(config, "rollout_split", "test")
    max_episodes = _raw_int(config, "max_episodes", 0)
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    if max_episodes > 0:
        episode_ids = episode_ids[: int(max_episodes)]

    groups_by_key: dict[tuple[str, int], list[pd.DataFrame]] = {}
    group_order: list[int] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        episode_id = str(group["episode_id"].iloc[0])
        request_id = int(group["request_id"].iloc[0])
        groups_by_key.setdefault((episode_id, request_id), []).append(group.reset_index(drop=True))
        group_order.append(int(group_id))

    use_neural = _uses_dqn_policy(base_policy)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=use_neural))
    states: dict[int, dict[str, np.ndarray]] = {}
    base_indices: dict[int, int] = {}
    label_vectors: dict[int, dict[str, np.ndarray]] = {}
    edge_index: np.ndarray | None = None
    visited_requests = 0
    materialized = 0
    missing_keys: set[tuple[str, int]] = set(groups_by_key.keys())
    base_mismatches = 0

    work_dir = output_path.parent / "_materialize_replay"
    work_dir.mkdir(parents=True, exist_ok=True)
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.head(int(max_requests_per_episode)).reset_index(drop=True)
        if episode.empty:
            continue
        traffic_path = _traffic_jsonl_for_episode(work_dir, f"materialize_{episode_id}", episode)
        env = _make_env(
            episode_id=f"materialize_{episode_id}",
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        for _position, request in episode.iterrows():
            batch = solver.candidate_batch(env)
            valid = np.flatnonzero(batch.candidate_mask.astype(bool))
            if valid.size == 0:
                action = int(solver.adapter(env).block_action(env))
            else:
                selected_base = _base_index(batch=batch, solver=solver, base_policy=base_policy)
                if selected_base < 0:
                    action = int(solver.adapter(env).block_action(env))
                else:
                    key = (str(episode_id), int(request["request_id"]))
                    if key in groups_by_key:
                        arrays = _batch_to_arrays(batch, solver.config)
                        if edge_index is None:
                            edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
                        for group in groups_by_key[key]:
                            group_id = int(group["group_id"].iloc[0])
                            states[group_id] = {state_key: np.asarray(arrays[state_key]) for state_key in STATE_KEYS}
                            base_indices[group_id] = int(selected_base)
                            label_vectors[group_id] = _group_label_vectors(group, int(solver.config.n_max))
                            stored_base = int(group["base_index"].iloc[0])
                            if stored_base != int(selected_base):
                                base_mismatches += 1
                            materialized += 1
                        missing_keys.discard(key)
                        if progress_every > 0 and (materialized == 1 or materialized % int(progress_every) == 0):
                            print(
                                json.dumps(
                                    {
                                        "event": "progress",
                                        "materialized_groups": int(materialized),
                                        "target_groups": int(metadata["group_id"].nunique()),
                                        "episode_id": str(episode_id),
                                        "request_id": int(request["request_id"]),
                                    },
                                    sort_keys=True,
                                ),
                                file=sys.stderr,
                                flush=True,
                            )
                    action = int(batch.topn[int(selected_base)].action)

            _observation, _reward, terminated, truncated, _info = env.step(action)
            visited_requests += 1
            if bool(terminated) or bool(truncated):
                break
        if not missing_keys:
            break

    missing_group_ids = [group_id for group_id in group_order if group_id not in states]
    if missing_group_ids:
        raise RuntimeError(f"Failed to materialize {len(missing_group_ids)} groups, first={missing_group_ids[:5]}")

    ordered_group_ids = np.asarray(group_order, dtype=np.int64)
    payload: dict[str, np.ndarray] = {
        "group_ids": ordered_group_ids,
        "base_index": np.asarray([base_indices[int(group_id)] for group_id in ordered_group_ids], dtype=np.int64),
        "edge_index": np.zeros((2, 0), dtype=np.int64) if edge_index is None else np.asarray(edge_index, dtype=np.int64),
    }
    for state_key in STATE_KEYS:
        dtype = np.bool_ if state_key == "candidate_mask" else np.float32
        payload[state_key] = np.stack([states[int(group_id)][state_key] for group_id in ordered_group_ids], axis=0).astype(dtype)
    for label_key in ("accepted_delta_vs_base", "secondary_delta_vs_base", "target_delta", "label_mask"):
        dtype = np.bool_ if label_key == "label_mask" else np.float32
        payload[label_key] = np.stack([label_vectors[int(group_id)][label_key] for group_id in ordered_group_ids], axis=0).astype(dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    summary = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "base_policy": str(base_policy),
        "dqn_checkpoint": solver.config.checkpoint_path,
        "groups": int(len(ordered_group_ids)),
        "metadata_rows": int(len(metadata)),
        "visited_requests": int(visited_requests),
        "base_mismatches": int(base_mismatches),
        "missing_keys": int(len(missing_keys)),
        "state_shapes": {key: list(value.shape) for key, value in payload.items() if key in STATE_KEYS},
    }
    _write_json(output_path.parent / "materialize_neural_states_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay base trajectory and materialize neural tensors for online Top-N examples.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--base-policy", default="")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path) if args.output_path else input_dir / "online_base_topn_neural_states.npz"
    summary = materialize(
        config=config,
        input_dir=input_dir,
        output_path=output_path,
        base_policy=str(args.base_policy),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
