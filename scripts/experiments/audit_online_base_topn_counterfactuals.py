from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(float(numerator) / denominator)


def _distribution(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None}
    quantiles = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "mean": float(np.mean(arr)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
    }


def _state_key_columns(metadata: pd.DataFrame) -> list[str]:
    candidates = ["split", "episode_id", "request_id", "position"]
    return [name for name in candidates if name in metadata.columns]


def _candidate_key_columns(metadata: pd.DataFrame) -> list[str]:
    columns = _state_key_columns(metadata)
    columns.append("candidate_index")
    return columns


def _sign(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    result = np.zeros(arr.shape, dtype=np.int8)
    result[arr > 0.0] = 1
    result[arr < 0.0] = -1
    return result


def _load_metadata(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "online_base_topn_examples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing collector metadata: {path}")
    metadata = pd.read_csv(path)
    if metadata.empty:
        return metadata
    metadata["is_base"] = metadata["is_base"].astype(bool)
    metadata["accepted_delta_vs_base"] = metadata["accepted_delta_vs_base"].astype(float)
    metadata["secondary_delta_vs_base"] = metadata["secondary_delta_vs_base"].astype(float)
    return metadata


def _group_oracle_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_columns = _state_key_columns(metadata)
    for group_id, group in metadata.groupby("group_id", sort=False):
        base_rows = group[group["is_base"]]
        base_future_accepted = float(base_rows["base_future_accepted"].iloc[0]) if not base_rows.empty else None
        nonbase = group[~group["is_base"]]
        if nonbase.empty:
            continue
        ordered = group.sort_values(["accepted_delta_vs_base", "secondary_delta_vs_base"], ascending=[False, False])
        best = ordered.iloc[0]
        record = {
            "group_id": int(group_id),
            "rows": int(len(group)),
            "nonbase_rows": int(len(nonbase)),
            "has_win": bool((nonbase["accepted_delta_vs_base"] > 0.0).any()),
            "has_loss": bool((nonbase["accepted_delta_vs_base"] < 0.0).any()),
            "best_accepted_delta": float(best["accepted_delta_vs_base"]),
            "best_secondary_delta": float(best["secondary_delta_vs_base"]),
            "best_candidate_index": int(best["candidate_index"]),
            "base_future_accepted": base_future_accepted,
            "oracle_future_accepted": None if base_future_accepted is None else float(base_future_accepted + best["accepted_delta_vs_base"]),
            "traffic_scenario": str(group["traffic_scenario"].iloc[0]) if "traffic_scenario" in group else "",
            "load_name": str(group["load_name"].iloc[0]) if "load_name" in group else "",
        }
        for key in key_columns:
            record[key] = group[key].iloc[0]
        rows.append(record)
    return pd.DataFrame(rows)


def _context_summary(groups: pd.DataFrame) -> list[dict[str, Any]]:
    if groups.empty or "traffic_scenario" not in groups or "load_name" not in groups:
        return []
    rows: list[dict[str, Any]] = []
    for (scenario, load), group in groups.groupby(["traffic_scenario", "load_name"], sort=True):
        rows.append(
            {
                "traffic_scenario": str(scenario),
                "load_name": str(load),
                "groups": int(len(group)),
                "state_positive_rate": _safe_rate(int(group["has_win"].sum()), len(group)),
                "groups_with_win": int(group["has_win"].sum()),
                "oracle_gain_accepted_sum": float(group["best_accepted_delta"].sum()),
                "mean_best_accepted_delta": float(group["best_accepted_delta"].mean()) if len(group) else None,
                "p95_best_accepted_delta": float(np.quantile(group["best_accepted_delta"].to_numpy(dtype=np.float64), 0.95))
                if len(group)
                else None,
            }
        )
    rows.sort(key=lambda row: (-float(row["state_positive_rate"] or 0.0), str(row["traffic_scenario"]), str(row["load_name"])))
    return rows


def _summarize_horizon(metadata: pd.DataFrame, *, secondary_margin: float, min_positive_delta: float) -> dict[str, Any]:
    if metadata.empty:
        return {"rows": 0, "groups": 0}
    groups = _group_oracle_table(metadata)
    nonbase = metadata[~metadata["is_base"]].reset_index(drop=True)
    wins = nonbase["accepted_delta_vs_base"] > 0.0
    losses = nonbase["accepted_delta_vs_base"] < 0.0
    ties = nonbase["accepted_delta_vs_base"] == 0.0
    hard_positive = nonbase["accepted_delta_vs_base"] >= float(min_positive_delta)
    uncertain_secondary = ties & (nonbase["secondary_delta_vs_base"].abs() < float(secondary_margin))
    return {
        "rows": int(len(metadata)),
        "groups": int(metadata["group_id"].nunique()),
        "non_base_rows": int(len(nonbase)),
        "avg_pool_size": float(len(metadata) / max(int(metadata["group_id"].nunique()), 1)),
        "win_rows": int(wins.sum()),
        "loss_rows": int(losses.sum()),
        "tie_rows": int(ties.sum()),
        "hard_positive_rows": int(hard_positive.sum()),
        "uncertain_tie_rows": int(uncertain_secondary.sum()),
        "action_positive_rate": _safe_rate(int(wins.sum()), len(nonbase)),
        "action_loss_rate": _safe_rate(int(losses.sum()), len(nonbase)),
        "action_tie_rate": _safe_rate(int(ties.sum()), len(nonbase)),
        "hard_positive_rate": _safe_rate(int(hard_positive.sum()), len(nonbase)),
        "uncertain_tie_rate": _safe_rate(int(uncertain_secondary.sum()), len(nonbase)),
        "groups_with_win": int(groups["has_win"].sum()) if not groups.empty else 0,
        "groups_with_loss": int(groups["has_loss"].sum()) if not groups.empty else 0,
        "state_positive_rate": _safe_rate(int(groups["has_win"].sum()) if not groups.empty else 0, len(groups)),
        "state_loss_available_rate": _safe_rate(int(groups["has_loss"].sum()) if not groups.empty else 0, len(groups)),
        "oracle_gain_accepted_sum": float(groups["best_accepted_delta"].sum()) if not groups.empty else 0.0,
        "mean_best_accepted_delta": float(groups["best_accepted_delta"].mean()) if not groups.empty else None,
        "best_accepted_delta_distribution": _distribution(groups["best_accepted_delta"] if not groups.empty else np.asarray([])),
        "accepted_delta_distribution_nonbase": _distribution(nonbase["accepted_delta_vs_base"]),
        "secondary_delta_distribution_ties": _distribution(nonbase.loc[ties, "secondary_delta_vs_base"]),
        "context": _context_summary(groups),
    }


def _candidate_stability(short: pd.DataFrame, ref: pd.DataFrame) -> dict[str, Any]:
    key_columns = _candidate_key_columns(short)
    short_nonbase = short[~short["is_base"]].copy()
    ref_nonbase = ref[~ref["is_base"]].copy()
    merged = short_nonbase.merge(
        ref_nonbase,
        on=key_columns,
        how="inner",
        suffixes=("_short", "_ref"),
    )
    if merged.empty:
        return {"common_candidate_rows": 0}
    sign_short = _sign(merged["accepted_delta_vs_base_short"])
    sign_ref = _sign(merged["accepted_delta_vs_base_ref"])
    short_wins = sign_short > 0
    ref_wins = sign_ref > 0
    nonzero_both = (sign_short != 0) & (sign_ref != 0)
    opposite_nonzero = nonzero_both & (sign_short != sign_ref)
    delta_short = merged["accepted_delta_vs_base_short"].to_numpy(dtype=np.float64)
    delta_ref = merged["accepted_delta_vs_base_ref"].to_numpy(dtype=np.float64)
    corr = None
    if len(merged) > 1 and np.std(delta_short) > 0.0 and np.std(delta_ref) > 0.0:
        corr = float(np.corrcoef(delta_short, delta_ref)[0, 1])
    return {
        "common_candidate_rows": int(len(merged)),
        "sign_mismatch_rate": float(np.mean(sign_short != sign_ref)),
        "opposite_nonzero_sign_rate": _safe_rate(int(opposite_nonzero.sum()), int(nonzero_both.sum())),
        "short_win_not_ref_win_rate": _safe_rate(int((short_wins & ~ref_wins).sum()), int(short_wins.sum())),
        "ref_win_missed_by_short_rate": _safe_rate(int((ref_wins & ~short_wins).sum()), int(ref_wins.sum())),
        "short_win_rows": int(short_wins.sum()),
        "ref_win_rows": int(ref_wins.sum()),
        "delta_correlation": corr,
        "short_delta_distribution": _distribution(delta_short),
        "ref_delta_distribution": _distribution(delta_ref),
    }


def _state_stability(short: pd.DataFrame, ref: pd.DataFrame) -> dict[str, Any]:
    short_groups = _group_oracle_table(short)
    ref_groups = _group_oracle_table(ref)
    key_columns = _state_key_columns(short_groups)
    merged = short_groups.merge(ref_groups, on=key_columns, how="inner", suffixes=("_short", "_ref"))
    if merged.empty:
        return {"common_states": 0}
    short_has_win = merged["has_win_short"].astype(bool).to_numpy()
    ref_has_win = merged["has_win_ref"].astype(bool).to_numpy()
    short_best_sign = _sign(merged["best_accepted_delta_short"])
    ref_best_sign = _sign(merged["best_accepted_delta_ref"])
    best_short = merged["best_accepted_delta_short"].to_numpy(dtype=np.float64)
    best_ref = merged["best_accepted_delta_ref"].to_numpy(dtype=np.float64)
    corr = None
    if len(merged) > 1 and np.std(best_short) > 0.0 and np.std(best_ref) > 0.0:
        corr = float(np.corrcoef(best_short, best_ref)[0, 1])
    return {
        "common_states": int(len(merged)),
        "has_win_mismatch_rate": float(np.mean(short_has_win != ref_has_win)),
        "short_has_win_not_ref_rate": _safe_rate(int((short_has_win & ~ref_has_win).sum()), int(short_has_win.sum())),
        "ref_has_win_missed_by_short_rate": _safe_rate(int((ref_has_win & ~short_has_win).sum()), int(ref_has_win.sum())),
        "best_sign_mismatch_rate": float(np.mean(short_best_sign != ref_best_sign)),
        "best_delta_correlation": corr,
        "short_oracle_gain_accepted_sum": float(merged["best_accepted_delta_short"].sum()),
        "ref_oracle_gain_accepted_sum": float(merged["best_accepted_delta_ref"].sum()),
    }


def _parse_run(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise ValueError("--run must use HORIZON:PATH format")
    horizon, path = value.split(":", 1)
    return int(horizon), Path(path)


def run_audit(
    *,
    runs: list[tuple[int, Path]],
    output_path: Path,
    reference_horizon: int | None,
    secondary_margin: float,
    min_positive_delta: float,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one --run is required")
    loaded = {int(horizon): _load_metadata(path) for horizon, path in runs}
    reference = int(reference_horizon) if reference_horizon is not None else max(loaded)
    if reference not in loaded:
        raise ValueError(f"Reference horizon {reference} was not provided")
    result: dict[str, Any] = {
        "runs": {str(horizon): str(path) for horizon, path in runs},
        "reference_horizon": int(reference),
        "secondary_margin": float(secondary_margin),
        "min_positive_delta": float(min_positive_delta),
        "horizons": {},
        "stability_vs_reference": {},
    }
    for horizon, metadata in loaded.items():
        result["horizons"][str(horizon)] = _summarize_horizon(
            metadata,
            secondary_margin=float(secondary_margin),
            min_positive_delta=float(min_positive_delta),
        )
    reference_metadata = loaded[int(reference)]
    for horizon, metadata in loaded.items():
        if int(horizon) == int(reference):
            continue
        result["stability_vs_reference"][str(horizon)] = {
            "candidate_level": _candidate_stability(metadata, reference_metadata),
            "state_level": _state_stability(metadata, reference_metadata),
        }
    _write_json(output_path, result)
    print(json.dumps(_json_safe(result), sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit online base Top-N counterfactual label quality and pool coverage.")
    parser.add_argument("--run", action="append", default=[], help="Collector run in HORIZON:PATH format.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-horizon", type=int, default=None)
    parser.add_argument("--secondary-margin", type=float, default=0.05)
    parser.add_argument("--min-positive-delta", type=float, default=1.0)
    args = parser.parse_args()
    run_audit(
        runs=[_parse_run(value) for value in args.run],
        output_path=Path(args.output),
        reference_horizon=args.reference_horizon,
        secondary_margin=float(args.secondary_margin),
        min_positive_delta=float(args.min_positive_delta),
    )


if __name__ == "__main__":
    main()
