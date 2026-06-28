from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from cse2026.data_generation.io_utils import read_json

from ..config import ExperimentConfig


def _finite_or_none(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def _mean(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    value = float(series.mean())
    return None if not math.isfinite(value) else value


def evaluate_topn_baseline(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("topn_baseline_eval requires dataset_path")
    rows: list[dict[str, Any]] = []
    run_path = Path(run_dir)
    for split in _splits(config):
        candidates = pd.read_parquet(config.dataset_path / "candidates" / f"{split}.parquet")
        dqn_path = config.dataset_path / "dqn" / f"{split}_transitions.parquet"
        dqn = pd.read_parquet(dqn_path) if dqn_path.exists() else pd.DataFrame()
        selected_rows: list[pd.Series] = []
        request_count = 0
        blocked = 0
        mask_sums: list[float] = []
        group_cols = ["episode_id", "request_id"]
        for _key, group in candidates.groupby(group_cols, sort=False):
            request_count += 1
            real = group[group["candidate_mask"] == 1]
            mask_sums.append(float(real["candidate_mask"].sum()))
            if real.empty:
                blocked += 1
                continue
            selected_rows.append(real.sort_values(["j_total", "energy_increment", "route_id", "b_start"]).iloc[0])
        selected = pd.DataFrame(selected_rows)
        if "num_feasible_before_topn" in dqn.columns:
            mean_num_feasible = _mean(dqn["num_feasible_before_topn"])
        else:
            mean_num_feasible = None
        row = {
            "split": split,
            "request_count": int(request_count),
            "blocking_rate": float(blocked / request_count) if request_count else None,
            "mean_energy_increment": _mean(selected["energy_increment"]) if not selected.empty else None,
            "mean_fragmentation_after": _mean(selected["fragmentation_after"]) if not selected.empty else None,
            "mean_qot_margin": _mean(selected["qot_margin"]) if not selected.empty else None,
            "mean_delay_ms": _mean(selected["delay_ms"]) if not selected.empty else None,
            "mean_candidate_mask_sum": float(sum(mask_sums) / len(mask_sums)) if mask_sums else None,
            "mean_num_feasible_before_topn": mean_num_feasible,
        }
        rows.append({key: _finite_or_none(value) for key, value in row.items()})

    metrics = {
        "stage": "topn_baseline_eval",
        "dataset_path": str(config.dataset_path),
        "splits": rows,
    }
    metrics_frame = pd.DataFrame(rows)
    run_path.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(run_path / "metrics.csv", index=False)
    _write_json(run_path / "metrics.json", metrics)
    return metrics
