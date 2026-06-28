from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import read_json

from ..config import ExperimentConfig


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits(config: ExperimentConfig) -> list[str]:
    if config.splits:
        return list(config.splits)
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    manifest = read_json(config.dataset_path / "manifest.json")
    return list(manifest["splits"].keys())


def _verify_selected_blocks(x_spec: np.ndarray, cnn_index: pd.DataFrame, max_rows: int = 1000) -> bool:
    if x_spec.shape[0] != len(cnn_index):
        return False
    slots = x_spec.shape[2]
    for row in cnn_index.head(max_rows).itertuples(index=False):
        expected = np.zeros(slots, dtype=x_spec.dtype)
        expected[int(row.b_start) : int(row.b_start) + int(row.w)] = 1
        if not np.array_equal(x_spec[int(row.sample_id), 1, :], expected):
            return False
    return True


def run_dataloader_smoke(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("dataloader_smoke requires dataset_path")
    run_path = Path(run_dir)
    split_metrics: list[dict[str, Any]] = []
    for split in _splits(config):
        graphs = np.load(config.dataset_path / "gnn" / f"{split}_graphs.npz")
        x_spec = np.load(config.dataset_path / "cnn" / f"{split}_tensors.npz")["X_spec"]
        candidates = pd.read_parquet(config.dataset_path / "candidates" / f"{split}.parquet")
        dqn = pd.read_parquet(config.dataset_path / "dqn" / f"{split}_transitions.parquet")
        cnn_index = pd.read_parquet(config.dataset_path / "cnn" / f"{split}_index.parquet")

        candidate_group_sizes = candidates.groupby(["episode_id", "request_id"]).size()
        candidate_mask_sums = candidates.groupby(["episode_id", "request_id"])["candidate_mask"].sum()
        if candidate_group_sizes.nunique() > 1:
            raise RuntimeError(f"{split}: candidate groups have inconsistent N_max sizes")
        if x_spec.ndim != 3 or x_spec.shape[1] != 6:
            raise RuntimeError(f"{split}: CNN tensor shape must be [N, 6, slots], got {x_spec.shape}")
        if not _verify_selected_blocks(x_spec, cnn_index):
            raise RuntimeError(f"{split}: selected_block_indicator channel check failed")

        batch_size = int(config.batch_size)
        max_batches = int(config.max_batches)
        batch_shapes: list[dict[str, Any]] = []
        total = int(min(len(dqn), max_batches * batch_size))
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch_shapes.append(
                {
                    "batch_start": int(start),
                    "batch_stop": int(stop),
                    "gnn_node_features": list(graphs["node_features"][start:stop].shape),
                    "gnn_link_features": list(graphs["link_features"][start:stop].shape),
                    "cnn_X_spec": list(x_spec[start : min(stop, len(x_spec))].shape),
                    "dqn_transitions": int(stop - start),
                }
            )
        print(f"{split}: node_features={graphs['node_features'].shape}")
        print(f"{split}: link_features={graphs['link_features'].shape}")
        print(f"{split}: CNN X_spec={x_spec.shape}")
        print(f"{split}: DQN transitions={len(dqn)}")
        split_metrics.append(
            {
                "split": split,
                "gnn_node_features_shape": list(graphs["node_features"].shape),
                "gnn_link_features_shape": list(graphs["link_features"].shape),
                "gnn_global_features_shape": list(graphs["global_features"].shape),
                "cnn_X_spec_shape": list(x_spec.shape),
                "candidate_rows": int(len(candidates)),
                "candidate_group_size": int(candidate_group_sizes.iloc[0]) if len(candidate_group_sizes) else 0,
                "mean_candidate_mask_sum": float(candidate_mask_sums.mean()) if len(candidate_mask_sums) else 0.0,
                "dqn_transitions": int(len(dqn)),
                "mini_batches_checked": len(batch_shapes),
                "batch_shapes": batch_shapes,
            }
        )

    metrics = {
        "stage": "dataloader_smoke",
        "dataset_path": str(config.dataset_path),
        "splits": split_metrics,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
