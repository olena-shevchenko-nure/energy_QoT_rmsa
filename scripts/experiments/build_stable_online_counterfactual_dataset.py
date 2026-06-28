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


def _load_dataset(input_dir: Path) -> dict[str, Any]:
    metadata_path = input_dir / "online_base_topn_examples.csv"
    npz_path = input_dir / "online_base_topn_examples.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz file: {npz_path}")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    npz = np.load(npz_path, allow_pickle=True)
    features = np.asarray(npz["features"], dtype=np.float32)
    targets = np.asarray(npz["targets"], dtype=np.float32)
    feature_names = np.asarray(npz["feature_names"], dtype=object)
    if len(metadata) != int(features.shape[0]) or len(metadata) != int(targets.shape[0]):
        raise ValueError(f"Row mismatch in {input_dir}: metadata={len(metadata)}, features={features.shape}, targets={targets.shape}")
    metadata["is_base"] = metadata["is_base"].astype(bool)
    return {"metadata": metadata, "features": features, "targets": targets, "feature_names": feature_names}


def _load_neural_states(input_dir: Path) -> dict[str, np.ndarray] | None:
    path = input_dir / "online_base_topn_neural_states.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    return {str(key): np.asarray(npz[key]) for key in npz.files}


def _key_columns(metadata: pd.DataFrame) -> list[str]:
    columns = ["split", "episode_id", "request_id", "position", "candidate_index"]
    return [column for column in columns if column in metadata.columns]


def _group_sizes(metadata: pd.DataFrame) -> np.ndarray:
    return metadata.groupby("group_id", sort=False).size().to_numpy(dtype=np.int32)


def _reason_counts(metadata: pd.DataFrame) -> dict[str, int]:
    if "stable_filter_reason" not in metadata:
        return {}
    return {str(key): int(value) for key, value in metadata["stable_filter_reason"].value_counts(dropna=False).sort_index().items()}


def _summarize_rows(metadata: pd.DataFrame, *, name: str) -> dict[str, Any]:
    if metadata.empty:
        return {
            "name": name,
            "rows": 0,
            "groups": 0,
            "non_base_rows": 0,
            "win_rows": 0,
            "loss_rows": 0,
            "tie_rows": 0,
            "groups_with_win": 0,
            "groups_with_loss": 0,
            "reason_counts": {},
        }
    nonbase = metadata[~metadata["is_base"].astype(bool)]
    accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
    return {
        "name": name,
        "rows": int(len(metadata)),
        "groups": int(metadata["group_id"].nunique()),
        "non_base_rows": int(len(nonbase)),
        "win_rows": int((accepted > 0.0).sum()) if not accepted.empty else 0,
        "loss_rows": int((accepted < 0.0).sum()) if not accepted.empty else 0,
        "tie_rows": int((accepted == 0.0).sum()) if not accepted.empty else 0,
        "groups_with_win": int(nonbase[accepted > 0.0]["group_id"].nunique()) if not nonbase.empty else 0,
        "groups_with_loss": int(nonbase[accepted < 0.0]["group_id"].nunique()) if not nonbase.empty else 0,
        "accepted_delta_sum": float(accepted.sum()) if not accepted.empty else 0.0,
        "accepted_delta_mean": float(accepted.mean()) if not accepted.empty else None,
        "reason_counts": _reason_counts(metadata),
    }


def _context_summary(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if metadata.empty or not {"traffic_scenario", "load_name"}.issubset(metadata.columns):
        return []
    rows: list[dict[str, Any]] = []
    group_head = metadata.drop_duplicates("group_id")
    for (scenario, load), groups in group_head.groupby(["traffic_scenario", "load_name"], sort=True):
        group_ids = set(int(value) for value in groups["group_id"].to_numpy(dtype=np.int64))
        subset = metadata[metadata["group_id"].astype(int).isin(group_ids)]
        nonbase = subset[~subset["is_base"].astype(bool)]
        accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
        rows.append(
            {
                "traffic_scenario": str(scenario),
                "load_name": str(load),
                "groups": int(len(group_ids)),
                "non_base_rows": int(len(nonbase)),
                "win_rows": int((accepted > 0.0).sum()) if not accepted.empty else 0,
                "loss_rows": int((accepted < 0.0).sum()) if not accepted.empty else 0,
                "groups_with_win": int(nonbase[accepted > 0.0]["group_id"].nunique()) if not nonbase.empty else 0,
                "accepted_delta_sum": float(accepted.sum()) if not accepted.empty else 0.0,
            }
        )
    rows.sort(key=lambda row: (-int(row["groups_with_win"]), -float(row["accepted_delta_sum"]), str(row["traffic_scenario"])))
    return rows


def _write_stable_neural_states(
    *,
    primary_neural: dict[str, np.ndarray] | None,
    kept_metadata: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    if primary_neural is None:
        return None
    if kept_metadata.empty:
        return None
    if "group_ids" not in primary_neural:
        raise ValueError("Neural states are missing group_ids")

    source_group_ids = [int(value) for value in np.asarray(primary_neural["group_ids"], dtype=np.int64).tolist()]
    source_by_group = {int(group_id): int(position) for position, group_id in enumerate(source_group_ids)}
    kept_group_ids = [int(value) for value in kept_metadata.drop_duplicates("group_id")["group_id"].to_numpy(dtype=np.int64)]
    source_indices: list[int] = []
    for group_id in kept_group_ids:
        if group_id not in source_by_group:
            raise ValueError(f"Neural states are missing kept group_id={group_id}")
        source_indices.append(int(source_by_group[group_id]))
    source_indices_array = np.asarray(source_indices, dtype=np.int64)
    n_states = int(len(source_indices_array))
    n_max = int(np.asarray(primary_neural["candidate_mask"])[source_indices_array].shape[1])

    label_mask = np.zeros((n_states, n_max), dtype=np.bool_)
    accepted_delta = np.full((n_states, n_max), np.nan, dtype=np.float32)
    secondary_delta = np.full((n_states, n_max), np.nan, dtype=np.float32)
    target_delta = np.full((n_states, n_max), np.nan, dtype=np.float32)
    group_position = {int(group_id): int(position) for position, group_id in enumerate(kept_group_ids)}

    for row in kept_metadata.itertuples(index=False):
        group_id = int(getattr(row, "group_id"))
        candidate_index = int(getattr(row, "candidate_index"))
        if not (0 <= candidate_index < n_max):
            continue
        position = int(group_position[group_id])
        accepted = float(getattr(row, "accepted_delta_vs_base"))
        secondary = float(getattr(row, "secondary_delta_vs_base"))
        label_mask[position, candidate_index] = True
        accepted_delta[position, candidate_index] = accepted
        secondary_delta[position, candidate_index] = secondary
        target_delta[position, candidate_index] = accepted if accepted != 0.0 else secondary

    payload: dict[str, np.ndarray] = {
        "group_ids": np.asarray(kept_group_ids, dtype=np.int64),
        "label_mask": label_mask,
        "accepted_delta_vs_base": accepted_delta,
        "secondary_delta_vs_base": secondary_delta,
        "target_delta": target_delta,
    }
    for key, values in primary_neural.items():
        if key in {"group_ids", "label_mask", "accepted_delta_vs_base", "secondary_delta_vs_base", "target_delta"}:
            continue
        if key == "edge_index":
            payload[key] = np.asarray(values)
            continue
        payload[key] = np.asarray(values)[source_indices_array]

    path = output_dir / "online_base_topn_neural_states.npz"
    np.savez_compressed(path, **payload)
    return str(path)


def _with_stable_filter(
    *,
    primary: pd.DataFrame,
    confirmation: pd.DataFrame,
    min_win_delta: float,
    min_secondary_abs: float,
) -> pd.DataFrame:
    key_columns = _key_columns(primary)
    confirm_columns = key_columns + ["accepted_delta_vs_base", "secondary_delta_vs_base"]
    merged = primary.merge(
        confirmation[confirm_columns],
        on=key_columns,
        how="left",
        suffixes=("", "_confirm"),
        validate="one_to_one",
    )
    if merged["accepted_delta_vs_base_confirm"].isna().any():
        missing = int(merged["accepted_delta_vs_base_confirm"].isna().sum())
        raise ValueError(f"Confirmation dataset is missing {missing} primary rows")

    primary_delta = merged["accepted_delta_vs_base"].astype(float)
    confirm_delta = merged["accepted_delta_vs_base_confirm"].astype(float)
    primary_secondary = merged["secondary_delta_vs_base"].astype(float)
    is_base = merged["is_base"].astype(bool)

    reason = np.full((len(merged),), "drop_conflict", dtype=object)
    keep = np.zeros((len(merged),), dtype=bool)
    label_kind = np.full((len(merged),), "conflict", dtype=object)

    base_mask = is_base.to_numpy(dtype=bool)
    keep[base_mask] = True
    reason[base_mask] = "keep_base"
    label_kind[base_mask] = "base"

    hard_positive = (~is_base) & (primary_delta >= float(min_win_delta)) & (confirm_delta >= 0.0)
    hard_negative = (~is_base) & (primary_delta <= -float(min_win_delta)) & (confirm_delta <= 0.0)
    stable_tie = (
        (~is_base)
        & (primary_delta == 0.0)
        & (confirm_delta == 0.0)
        & (primary_secondary.abs() >= float(min_secondary_abs))
    )
    weak_tie = (~is_base) & (primary_delta == 0.0) & (confirm_delta == 0.0) & (primary_secondary.abs() < float(min_secondary_abs))

    keep[hard_positive.to_numpy(dtype=bool)] = True
    reason[hard_positive.to_numpy(dtype=bool)] = "keep_h100_win_h50_nonloss"
    label_kind[hard_positive.to_numpy(dtype=bool)] = "hard_positive"

    keep[hard_negative.to_numpy(dtype=bool)] = True
    reason[hard_negative.to_numpy(dtype=bool)] = "keep_h100_loss_h50_nonwin"
    label_kind[hard_negative.to_numpy(dtype=bool)] = "hard_negative"

    keep[stable_tie.to_numpy(dtype=bool)] = True
    reason[stable_tie.to_numpy(dtype=bool)] = "keep_stable_secondary_tie"
    label_kind[stable_tie.to_numpy(dtype=bool)] = "stable_secondary_tie"

    reason[weak_tie.to_numpy(dtype=bool)] = "drop_uncertain_weak_tie"

    merged["confirm_accepted_delta_vs_base"] = confirm_delta
    merged["confirm_secondary_delta_vs_base"] = merged["secondary_delta_vs_base_confirm"].astype(float)
    merged["stable_keep"] = keep
    merged["stable_label_kind"] = label_kind
    merged["stable_filter_reason"] = reason
    return merged


def build_stable_dataset(
    *,
    primary_dir: Path,
    confirmation_dir: Path,
    output_dir: Path,
    min_win_delta: float,
    min_secondary_abs: float,
) -> dict[str, Any]:
    primary = _load_dataset(primary_dir)
    confirmation = _load_dataset(confirmation_dir)
    primary_neural = _load_neural_states(primary_dir)
    if list(primary["feature_names"]) != list(confirmation["feature_names"]):
        raise ValueError("Primary and confirmation feature layouts differ")

    metadata = primary["metadata"].reset_index(drop=True)
    filtered = _with_stable_filter(
        primary=metadata,
        confirmation=confirmation["metadata"].reset_index(drop=True),
        min_win_delta=float(min_win_delta),
        min_secondary_abs=float(min_secondary_abs),
    )
    kept_nonbase = filtered[filtered["stable_keep"] & ~filtered["is_base"].astype(bool)]
    kept_group_ids = set(int(value) for value in kept_nonbase["group_id"].to_numpy(dtype=np.int64))
    kept_mask = filtered["group_id"].astype(int).isin(kept_group_ids) & filtered["stable_keep"].astype(bool)
    kept_indices = np.flatnonzero(kept_mask.to_numpy(dtype=bool)).astype(np.int64)
    dropped = filtered[~filtered.index.isin(set(int(value) for value in kept_indices))].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    kept_metadata = filtered.iloc[kept_indices].reset_index(drop=True).copy()
    kept_features = primary["features"][kept_indices].astype(np.float32)
    kept_targets = primary["targets"][kept_indices].astype(np.float32)
    kept_metadata.to_csv(output_dir / "online_base_topn_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "online_base_topn_examples.npz",
        features=kept_features,
        targets=kept_targets,
        group_sizes=_group_sizes(kept_metadata),
        feature_names=np.asarray(primary["feature_names"], dtype=object),
    )
    stable_neural_path = _write_stable_neural_states(
        primary_neural=primary_neural,
        kept_metadata=kept_metadata,
        output_dir=output_dir,
    )
    filtered.to_csv(output_dir / "stable_filter_all_rows.csv", index=False)
    dropped.to_csv(output_dir / "stable_filter_dropped_rows.csv", index=False)

    summary = {
        "primary_dir": str(primary_dir),
        "confirmation_dir": str(confirmation_dir),
        "output_dir": str(output_dir),
        "primary_horizon": int(metadata["lookahead_horizon"].iloc[0]) if "lookahead_horizon" in metadata and not metadata.empty else None,
        "confirmation_horizon": int(confirmation["metadata"]["lookahead_horizon"].iloc[0])
        if "lookahead_horizon" in confirmation["metadata"] and not confirmation["metadata"].empty
        else None,
        "min_win_delta": float(min_win_delta),
        "min_secondary_abs": float(min_secondary_abs),
        "primary_has_neural_states": bool(primary_neural is not None),
        "stable_neural_states_path": stable_neural_path,
        "raw": _summarize_rows(filtered, name="raw_primary_with_filter_flags"),
        "kept": _summarize_rows(kept_metadata, name="stable_kept"),
        "dropped": _summarize_rows(dropped, name="dropped"),
        "kept_context": _context_summary(kept_metadata),
    }
    _write_json(output_dir / "stable_online_counterfactual_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stable online counterfactual dataset from primary and confirmation horizons.")
    parser.add_argument("--primary-dir", required=True, help="Primary collector output, usually h100 or full-tail.")
    parser.add_argument("--confirmation-dir", required=True, help="Confirmation collector output, usually h50.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-win-delta", type=float, default=1.0)
    parser.add_argument("--min-secondary-abs", type=float, default=0.05)
    args = parser.parse_args()
    summary = build_stable_dataset(
        primary_dir=Path(args.primary_dir),
        confirmation_dir=Path(args.confirmation_dir),
        output_dir=Path(args.output_dir),
        min_win_delta=float(args.min_win_delta),
        min_secondary_abs=float(args.min_secondary_abs),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
