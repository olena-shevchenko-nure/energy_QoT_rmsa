from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quick_exception_ranker_ab import _add_runtime_features, _filter_small_pool, _load_split


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_index(feature_names: list[str]) -> dict[str, int]:
    return {name: int(position) for position, name in enumerate(feature_names)}


def _condition_masks(data: dict[str, Any]) -> dict[str, np.ndarray]:
    x = np.asarray(data["x"], dtype=np.float32)
    fidx = _feature_index(data["feature_names"])
    return {
        "qot_margin_emergency": x[:, fidx["qot_margin_delta"]] >= -0.25,
        "energy_emergency": x[:, fidx["energy_delta"]] <= 0.40,
        "delay_emergency": x[:, fidx["delay_delta_norm"]] <= 0.20,
    }


def _all_pass(condition_masks: dict[str, np.ndarray]) -> np.ndarray:
    masks = list(condition_masks.values())
    if not masks:
        return np.zeros((0,), dtype=bool)
    result = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        result &= mask
    return result


def _rate(mask: np.ndarray) -> float | None:
    if mask.size == 0:
        return None
    return float(np.mean(mask))


def _summary_for_subset(
    *,
    name: str,
    subset_mask: np.ndarray,
    metadata: pd.DataFrame,
    condition_masks: dict[str, np.ndarray],
    pass_all: np.ndarray,
) -> dict[str, Any]:
    rows = int(np.sum(subset_mask))
    groups = int(metadata.loc[subset_mask, "group_id"].nunique()) if rows else 0
    result: dict[str, Any] = {
        "name": name,
        "rows": rows,
        "groups": groups,
        "pass_all_rows": int(np.sum(subset_mask & pass_all)),
        "pass_all_rate": _rate(pass_all[subset_mask]),
    }
    per_condition: dict[str, Any] = {}
    for condition, mask in condition_masks.items():
        fail = ~mask
        per_condition[condition] = {
            "pass_rate": _rate(mask[subset_mask]),
            "fail_rows": int(np.sum(subset_mask & fail)),
            "fail_rate": _rate(fail[subset_mask]),
            "alone_blocks_rows": int(np.sum(subset_mask & fail & _all_other_pass(condition_masks, condition))),
        }
    result["conditions"] = per_condition
    return result


def _all_other_pass(condition_masks: dict[str, np.ndarray], excluded: str) -> np.ndarray:
    selected = [mask for name, mask in condition_masks.items() if name != excluded]
    if not selected:
        return np.zeros((0,), dtype=bool)
    result = np.ones_like(selected[0], dtype=bool)
    for mask in selected:
        result &= mask
    return result


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"p01": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None}
    quantiles = np.quantile(values.astype(np.float64), [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
    }


def _delta_distributions(data: dict[str, Any], subset_mask: np.ndarray) -> dict[str, Any]:
    x = np.asarray(data["x"], dtype=np.float32)
    fidx = _feature_index(data["feature_names"])
    fields = [
        "energy_delta",
        "fragmentation_delta",
        "small_gap_delta",
        "largest_free_block_delta_norm",
        "qot_margin_delta",
        "delay_delta_norm",
    ]
    return {field: _distribution(x[subset_mask, fidx[field]]) for field in fields}


def audit_split(original: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    data = _add_runtime_features(_filter_small_pool(original, top_k=top_k))
    metadata = data["metadata"].reset_index(drop=True)
    nonbase = (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()
    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    hard_positive = nonbase & (accepted_delta > 0)
    hard_negative = nonbase & (accepted_delta < 0)
    tie = nonbase & (accepted_delta == 0)
    condition_masks = _condition_masks(data)
    pass_all = _all_pass(condition_masks)
    summaries = {
        "all_nonbase": _summary_for_subset(
            name="all_nonbase",
            subset_mask=nonbase,
            metadata=metadata,
            condition_masks=condition_masks,
            pass_all=pass_all,
        ),
        "hard_positive": _summary_for_subset(
            name="hard_positive",
            subset_mask=hard_positive,
            metadata=metadata,
            condition_masks=condition_masks,
            pass_all=pass_all,
        ),
        "hard_negative": _summary_for_subset(
            name="hard_negative",
            subset_mask=hard_negative,
            metadata=metadata,
            condition_masks=condition_masks,
            pass_all=pass_all,
        ),
        "tie": _summary_for_subset(
            name="tie",
            subset_mask=tie,
            metadata=metadata,
            condition_masks=condition_masks,
            pass_all=pass_all,
        ),
    }
    return {
        "groups": int(metadata["group_id"].nunique()),
        "rows": int(len(metadata)),
        "avg_pool_size": float(len(metadata) / max(int(metadata["group_id"].nunique()), 1)),
        "summaries": summaries,
        "delta_distributions": {
            "hard_positive": _delta_distributions(data, hard_positive),
            "hard_negative": _delta_distributions(data, hard_negative),
            "tie": _delta_distributions(data, tie),
        },
    }


def run_audit(*, run_dir: Path, output_path: Path, top_k: int) -> dict[str, Any]:
    result = {
        "run_dir": str(run_dir),
        "top_k": int(top_k),
        "train": audit_split(_load_split(run_dir, "train"), top_k=top_k),
        "eval": audit_split(_load_split(run_dir, "eval"), top_k=top_k),
    }
    _write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit safety guard conditions on small exception pool.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    run_audit(run_dir=Path(args.run_dir), output_path=Path(args.output), top_k=int(args.top_k))


if __name__ == "__main__":
    main()
