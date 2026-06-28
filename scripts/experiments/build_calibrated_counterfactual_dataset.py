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


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def _load_optional_npz(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    return {str(key): np.asarray(npz[key]) for key in npz.files}


def _load_dataset(input_dir: Path) -> dict[str, Any]:
    metadata_path = input_dir / "online_base_topn_examples.csv"
    neural_path = input_dir / "online_base_topn_neural_states.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not neural_path.exists():
        raise FileNotFoundError(f"Missing neural state file: {neural_path}")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    if "group_id" not in metadata or "candidate_index" not in metadata or "is_base" not in metadata:
        raise ValueError("Metadata must include group_id, candidate_index, and is_base")
    metadata["is_base"] = _bool_series(metadata["is_base"])
    metadata["source_row_index"] = np.arange(len(metadata), dtype=np.int64)
    for column in ("accepted_delta_vs_base", "secondary_delta_vs_base"):
        if column not in metadata:
            raise ValueError(f"Metadata is missing {column}")
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce").fillna(0.0).astype(float)
    neural = _load_optional_npz(neural_path)
    if neural is None or "group_ids" not in neural or "candidate_mask" not in neural:
        raise ValueError(f"Neural state file is missing group_ids or candidate_mask: {neural_path}")
    row_features = _load_optional_npz(input_dir / "online_base_topn_examples.npz")
    return {"metadata": metadata, "neural": neural, "row_features": row_features}


def _label_kind(metadata: pd.DataFrame) -> pd.Series:
    if "stable_label_kind" in metadata:
        return metadata["stable_label_kind"].astype(str)
    accepted = metadata["accepted_delta_vs_base"].astype(float)
    secondary = metadata["secondary_delta_vs_base"].astype(float)
    kind = np.full((len(metadata),), "stable_secondary_tie", dtype=object)
    kind[metadata["is_base"].to_numpy(dtype=bool)] = "base"
    kind[(~metadata["is_base"]) & (accepted > 0.0)] = "hard_positive"
    kind[(~metadata["is_base"]) & (accepted < 0.0)] = "hard_negative"
    kind[(~metadata["is_base"]) & (accepted == 0.0) & (secondary.abs() == 0.0)] = "zero_tie"
    return pd.Series(kind, index=metadata.index)


def _bucket_columns(metadata: pd.DataFrame, columns_text: str) -> list[str]:
    requested = [item.strip() for item in str(columns_text or "").split(",") if item.strip()]
    return [column for column in requested if column in metadata.columns]


def _bucket_keys(metadata: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.full(len(metadata), "", dtype=object), index=metadata.index)
    values = metadata[columns].astype(str)
    return values.apply(lambda row: ":".join(str(item) for item in row), axis=1)


def _parse_bucket_set(text: str) -> set[str]:
    return {item.strip() for item in str(text or "").split(",") if item.strip()}


def _bucket_delta_sum(metadata: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.zeros(len(metadata), dtype=np.float32), index=metadata.index)
    nonbase = metadata[~metadata["is_base"].astype(bool)]
    if nonbase.empty:
        return pd.Series(np.zeros(len(metadata), dtype=np.float32), index=metadata.index)
    sums = nonbase.groupby(columns, sort=False)["accepted_delta_vs_base"].sum().rename("calibrated_bucket_delta_sum")
    joined = metadata[columns].merge(sums.reset_index(), on=columns, how="left")
    return joined["calibrated_bucket_delta_sum"].fillna(0.0).astype(float)


def _calibrate_weights(metadata: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    calibrated = metadata.copy()
    is_base = calibrated["is_base"].astype(bool)
    accepted = calibrated["accepted_delta_vs_base"].astype(float)
    secondary = calibrated["secondary_delta_vs_base"].astype(float)
    kind = _label_kind(calibrated)
    bucket_cols = _bucket_columns(calibrated, str(args.bucket_columns))
    bucket_key = _bucket_keys(calibrated, bucket_cols)
    bucket_sum = _bucket_delta_sum(calibrated, bucket_cols)
    protected_buckets = _parse_bucket_set(str(args.protected_buckets))
    fragile_buckets = _parse_bucket_set(str(args.fragile_buckets))
    protected_bucket = bucket_key.isin(protected_buckets).to_numpy(dtype=bool) if protected_buckets else np.zeros(len(calibrated), dtype=bool)
    fragile_bucket = bucket_key.isin(fragile_buckets).to_numpy(dtype=bool) if fragile_buckets else np.zeros(len(calibrated), dtype=bool)

    weight = np.zeros(len(calibrated), dtype=np.float32)
    reason = np.full((len(calibrated),), "drop_low_weight", dtype=object)

    base_mask = is_base.to_numpy(dtype=bool)
    positive_mask = ((~is_base) & (accepted > 0.0)).to_numpy(dtype=bool)
    negative_mask = ((~is_base) & (accepted < 0.0)).to_numpy(dtype=bool)
    tie_mask = ((~is_base) & (accepted == 0.0)).to_numpy(dtype=bool)

    magnitude = np.minimum(np.abs(accepted.to_numpy(dtype=np.float32)), float(args.magnitude_cap))
    weight[base_mask] = float(args.base_weight)
    weight[positive_mask] = float(args.positive_weight) + float(args.magnitude_weight) * magnitude[positive_mask]
    weight[negative_mask] = float(args.negative_weight) + float(args.negative_magnitude_weight) * magnitude[negative_mask]
    weight[tie_mask] = float(args.tie_weight)

    confirm = None
    if "confirm_accepted_delta_vs_base" in calibrated:
        confirm = pd.to_numeric(calibrated["confirm_accepted_delta_vs_base"], errors="coerce").fillna(0.0).astype(float)
    elif "accepted_delta_vs_base_confirm" in calibrated:
        confirm = pd.to_numeric(calibrated["accepted_delta_vs_base_confirm"], errors="coerce").fillna(0.0).astype(float)

    if confirm is not None:
        confirm_positive = ((~is_base) & (accepted > 0.0) & (confirm > 0.0)).to_numpy(dtype=bool)
        primary_only_positive = ((~is_base) & (accepted > 0.0) & (confirm == 0.0)).to_numpy(dtype=bool)
        conflict = (
            ((~is_base) & (accepted > 0.0) & (confirm < 0.0))
            | ((~is_base) & (accepted < 0.0) & (confirm > 0.0))
        ).to_numpy(dtype=bool)
        weight[confirm_positive] *= float(args.confirmed_positive_boost)
        weight[primary_only_positive] *= float(args.primary_only_positive_multiplier)
        weight[conflict] *= float(args.conflict_multiplier)

    negative_bucket = (bucket_sum < 0.0).to_numpy(dtype=bool) & (~base_mask)
    weight[negative_bucket & positive_mask] *= float(args.positive_negative_bucket_multiplier)
    weight[negative_bucket & (~positive_mask)] *= float(args.negative_bucket_multiplier)

    if protected_buckets:
        weight[protected_bucket & base_mask] *= float(args.protected_base_multiplier)
        weight[protected_bucket & positive_mask] *= float(args.protected_positive_multiplier)
        weight[protected_bucket & negative_mask] *= float(args.protected_negative_multiplier)
        weight[protected_bucket & tie_mask] *= float(args.protected_tie_multiplier)
    if fragile_buckets:
        weight[fragile_bucket & base_mask] *= float(args.fragile_base_multiplier)
        weight[fragile_bucket & positive_mask] *= float(args.fragile_positive_multiplier)
        weight[fragile_bucket & negative_mask] *= float(args.fragile_negative_multiplier)
        weight[fragile_bucket & tie_mask] *= float(args.fragile_tie_multiplier)

    if protected_buckets or fragile_buckets:
        strong_positive = (~base_mask) & positive_mask & (accepted.to_numpy(dtype=np.float32) >= float(args.protected_strong_positive_min_delta))
        if confirm is not None:
            strong_positive &= confirm.to_numpy(dtype=np.float32) >= float(args.protected_strong_confirm_min_delta)
        strong_positive &= protected_bucket | fragile_bucket
        weight[strong_positive] *= float(args.protected_strong_positive_multiplier)

    keep_label = weight >= float(args.min_label_weight)
    keep_label[base_mask] = True
    reason[keep_label] = "keep_weighted_label"
    reason[base_mask] = "keep_base"
    reason[negative_bucket & keep_label] = "keep_weighted_negative_bucket"
    reason[positive_mask & keep_label] = "keep_weighted_positive"
    reason[protected_bucket & keep_label & ~base_mask] = "keep_weighted_protected_bucket"
    reason[fragile_bucket & keep_label & ~base_mask] = "keep_weighted_fragile_bucket"

    calibrated["calibrated_label_kind"] = kind
    calibrated["calibrated_bucket_key"] = bucket_key
    calibrated["calibrated_bucket_delta_sum"] = bucket_sum.to_numpy(dtype=np.float32)
    calibrated["calibrated_protected_bucket"] = protected_bucket
    calibrated["calibrated_fragile_bucket"] = fragile_bucket
    calibrated["calibrated_label_weight"] = weight.astype(np.float32)
    calibrated["calibrated_keep_label"] = keep_label
    calibrated["calibrated_filter_reason"] = reason
    return calibrated


def _group_sizes(metadata: pd.DataFrame) -> np.ndarray:
    return metadata.groupby("group_id", sort=False).size().to_numpy(dtype=np.int32)


def _target_from_row(row: Any) -> float:
    accepted = float(getattr(row, "accepted_delta_vs_base"))
    if accepted != 0.0:
        return accepted
    return float(getattr(row, "secondary_delta_vs_base"))


def _write_row_features(row_features: dict[str, np.ndarray] | None, kept_metadata: pd.DataFrame, output_dir: Path) -> str | None:
    if row_features is None:
        return None
    if "features" not in row_features or "targets" not in row_features:
        return None
    source_indices = kept_metadata["source_row_index"].to_numpy(dtype=np.int64)
    features = np.asarray(row_features["features"])[source_indices].astype(np.float32)
    targets = np.asarray(row_features["targets"])[source_indices].astype(np.float32)
    payload: dict[str, np.ndarray] = {
        "features": features,
        "targets": targets,
        "group_sizes": _group_sizes(kept_metadata),
    }
    if "feature_names" in row_features:
        payload["feature_names"] = np.asarray(row_features["feature_names"], dtype=object)
    path = output_dir / "online_base_topn_examples.npz"
    np.savez_compressed(path, **payload)
    return str(path)


def _write_neural_states(
    *,
    neural: dict[str, np.ndarray],
    kept_metadata: pd.DataFrame,
    output_dir: Path,
) -> str:
    source_group_ids = [int(value) for value in np.asarray(neural["group_ids"], dtype=np.int64).tolist()]
    source_by_group = {int(group_id): int(position) for position, group_id in enumerate(source_group_ids)}
    kept_group_ids = [int(value) for value in kept_metadata.drop_duplicates("group_id")["group_id"].to_numpy(dtype=np.int64)]
    source_indices: list[int] = []
    for group_id in kept_group_ids:
        if group_id not in source_by_group:
            raise ValueError(f"Neural states are missing kept group_id={group_id}")
        source_indices.append(int(source_by_group[group_id]))
    source_indices_array = np.asarray(source_indices, dtype=np.int64)
    n_states = int(len(source_indices_array))
    n_max = int(np.asarray(neural["candidate_mask"])[source_indices_array].shape[1])

    label_mask = np.zeros((n_states, n_max), dtype=np.bool_)
    label_weight = np.zeros((n_states, n_max), dtype=np.float32)
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
        label_mask[position, candidate_index] = True
        accepted_delta[position, candidate_index] = float(getattr(row, "accepted_delta_vs_base"))
        secondary_delta[position, candidate_index] = float(getattr(row, "secondary_delta_vs_base"))
        target_delta[position, candidate_index] = _target_from_row(row)
        label_weight[position, candidate_index] = float(getattr(row, "calibrated_label_weight"))

    payload: dict[str, np.ndarray] = {
        "group_ids": np.asarray(kept_group_ids, dtype=np.int64),
        "label_mask": label_mask,
        "label_weight": label_weight,
        "accepted_delta_vs_base": accepted_delta,
        "secondary_delta_vs_base": secondary_delta,
        "target_delta": target_delta,
    }
    source_group_count = int(len(source_group_ids))
    overwritten = {
        "group_ids",
        "label_mask",
        "label_weight",
        "accepted_delta_vs_base",
        "secondary_delta_vs_base",
        "target_delta",
    }
    for key, values in neural.items():
        if key in overwritten:
            continue
        array = np.asarray(values)
        if key == "edge_index":
            payload[key] = array
        elif array.ndim > 0 and int(array.shape[0]) == source_group_count:
            payload[key] = array[source_indices_array]
        else:
            payload[key] = array

    path = output_dir / "online_base_topn_neural_states.npz"
    np.savez_compressed(path, **payload)
    return str(path)


def _summarize_rows(metadata: pd.DataFrame, *, name: str) -> dict[str, Any]:
    if metadata.empty:
        return {"name": name, "rows": 0, "groups": 0, "non_base_rows": 0}
    nonbase = metadata[~metadata["is_base"].astype(bool)]
    accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
    weights = metadata["calibrated_label_weight"].astype(float) if "calibrated_label_weight" in metadata else pd.Series(dtype=float)
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
        "weighted_accepted_delta_sum": float((accepted * nonbase["calibrated_label_weight"].astype(float)).sum())
        if not nonbase.empty and "calibrated_label_weight" in nonbase
        else 0.0,
        "label_weight_mean": float(weights.mean()) if not weights.empty else None,
        "label_weight_quantiles": [float(weights.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
        if not weights.empty
        else [],
    }


def _by_context(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if metadata.empty or not {"traffic_scenario", "load_name"}.issubset(metadata.columns):
        return []
    rows: list[dict[str, Any]] = []
    for (scenario, load), group in metadata.groupby(["traffic_scenario", "load_name"], sort=True):
        nonbase = group[~group["is_base"].astype(bool)]
        accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
        rows.append(
            {
                "traffic_scenario": str(scenario),
                "load_name": str(load),
                "groups": int(group["group_id"].nunique()),
                "non_base_rows": int(len(nonbase)),
                "win_rows": int((accepted > 0.0).sum()) if not accepted.empty else 0,
                "loss_rows": int((accepted < 0.0).sum()) if not accepted.empty else 0,
                "accepted_delta_sum": float(accepted.sum()) if not accepted.empty else 0.0,
                "weighted_accepted_delta_sum": float((accepted * nonbase["calibrated_label_weight"].astype(float)).sum())
                if not nonbase.empty and "calibrated_label_weight" in nonbase
                else 0.0,
                "label_weight_mean": float(group["calibrated_label_weight"].astype(float).mean()),
                "protected_bucket": bool(group["calibrated_protected_bucket"].astype(bool).any())
                if "calibrated_protected_bucket" in group
                else False,
                "fragile_bucket": bool(group["calibrated_fragile_bucket"].astype(bool).any())
                if "calibrated_fragile_bucket" in group
                else False,
            }
        )
    rows.sort(key=lambda row: (-int(row["win_rows"]), -float(row["accepted_delta_sum"]), str(row["traffic_scenario"])))
    return rows


def _oracle_audit(metadata: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        nonbase = group[~group["is_base"].astype(bool)]
        if nonbase.empty:
            continue
        target = nonbase["accepted_delta_vs_base"].where(
            nonbase["accepted_delta_vs_base"].astype(float) != 0.0,
            nonbase["secondary_delta_vs_base"],
        ).astype(float)
        accepted = nonbase["accepted_delta_vs_base"].astype(float)
        best_index = int(target.to_numpy(dtype=np.float32).argmax())
        rows.append(
            {
                "group_id": float(group_id),
                "best_target_delta": float(target.iloc[best_index]),
                "best_accepted_delta": float(accepted.iloc[best_index]),
                "max_accepted_delta": float(accepted.max()),
            }
        )
    if not rows:
        return {"groups": 0}
    table = pd.DataFrame(rows)
    return {
        "groups": int(len(table)),
        "groups_with_accepted_win": int((table["max_accepted_delta"] > 0.0).sum()),
        "best_target_total_delta": float(table["best_target_delta"].sum()),
        "best_target_accepted_total_delta": float(table["best_accepted_delta"].sum()),
        "max_accepted_total_positive_delta": float(table["max_accepted_delta"].clip(lower=0.0).sum()),
    }


def _by_bucket_role(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if metadata.empty or "calibrated_label_weight" not in metadata:
        return []
    role = np.full((len(metadata),), "unprotected", dtype=object)
    if "calibrated_protected_bucket" in metadata:
        role[metadata["calibrated_protected_bucket"].astype(bool).to_numpy(dtype=bool)] = "protected"
    if "calibrated_fragile_bucket" in metadata:
        role[metadata["calibrated_fragile_bucket"].astype(bool).to_numpy(dtype=bool)] = "fragile"
    table = metadata.copy()
    table["bucket_role"] = role
    rows: list[dict[str, Any]] = []
    for bucket_role, group in table.groupby("bucket_role", sort=True):
        nonbase = group[~group["is_base"].astype(bool)]
        accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
        rows.append(
            {
                "bucket_role": str(bucket_role),
                "groups": int(group["group_id"].nunique()),
                "rows": int(len(group)),
                "non_base_rows": int(len(nonbase)),
                "win_rows": int((accepted > 0.0).sum()) if not accepted.empty else 0,
                "loss_rows": int((accepted < 0.0).sum()) if not accepted.empty else 0,
                "accepted_delta_sum": float(accepted.sum()) if not accepted.empty else 0.0,
                "weighted_accepted_delta_sum": float((accepted * nonbase["calibrated_label_weight"].astype(float)).sum())
                if not nonbase.empty
                else 0.0,
                "label_weight_mean": float(group["calibrated_label_weight"].astype(float).mean()),
            }
        )
    return rows


def build_calibrated_dataset(*, input_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    loaded = _load_dataset(input_dir)
    metadata = loaded["metadata"]
    calibrated = _calibrate_weights(metadata, args)
    base_by_group = calibrated[calibrated["is_base"].astype(bool)].groupby("group_id", sort=False).size()
    nonbase_keep_by_group = calibrated[
        calibrated["calibrated_keep_label"].astype(bool) & ~calibrated["is_base"].astype(bool)
    ].groupby("group_id", sort=False).size()
    group_keep_ids = {
        int(group_id)
        for group_id, count in nonbase_keep_by_group.items()
        if int(count) >= int(args.min_nonbase_labels) and int(base_by_group.get(group_id, 0)) > 0
    }
    keep_rows = calibrated["calibrated_keep_label"].astype(bool) & calibrated["group_id"].astype(int).isin(group_keep_ids)
    kept = calibrated[keep_rows].reset_index(drop=True).copy()
    dropped = calibrated[~keep_rows].reset_index(drop=True).copy()
    if kept.empty:
        raise RuntimeError("Calibration removed all groups")

    output_dir.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_dir / "online_base_topn_examples.csv", index=False)
    calibrated.to_csv(output_dir / "calibration_all_rows.csv", index=False)
    dropped.to_csv(output_dir / "calibration_dropped_rows.csv", index=False)
    row_features_path = _write_row_features(loaded["row_features"], kept, output_dir)
    neural_path = _write_neural_states(neural=loaded["neural"], kept_metadata=kept, output_dir=output_dir)

    by_kind = (
        kept.groupby("calibrated_label_kind", sort=True)
        .agg(
            rows=("group_id", "size"),
            groups=("group_id", "nunique"),
            mean_weight=("calibrated_label_weight", "mean"),
            accepted_delta_sum=("accepted_delta_vs_base", "sum"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "stage": "build_calibrated_counterfactual_dataset",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "neural_states_path": neural_path,
        "row_features_path": row_features_path,
        "args": vars(args),
        "raw": _summarize_rows(calibrated, name="raw_stable"),
        "kept": _summarize_rows(kept, name="calibrated_kept"),
        "dropped": _summarize_rows(dropped, name="calibrated_dropped"),
        "by_kind": by_kind,
        "by_context": _by_context(kept),
        "by_bucket_role": _by_bucket_role(kept),
        "oracle_audit": _oracle_audit(kept),
    }
    _write_json(output_dir / "calibrated_counterfactual_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add calibrated label weights to a stable online counterfactual dataset.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket-columns", default="traffic_scenario,load_name")
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=1.6)
    parser.add_argument("--negative-weight", type=float, default=0.30)
    parser.add_argument("--tie-weight", type=float, default=0.12)
    parser.add_argument("--magnitude-weight", type=float, default=0.20)
    parser.add_argument("--negative-magnitude-weight", type=float, default=0.05)
    parser.add_argument("--magnitude-cap", type=float, default=4.0)
    parser.add_argument("--confirmed-positive-boost", type=float, default=1.35)
    parser.add_argument("--primary-only-positive-multiplier", type=float, default=0.75)
    parser.add_argument("--conflict-multiplier", type=float, default=0.10)
    parser.add_argument("--negative-bucket-multiplier", type=float, default=0.55)
    parser.add_argument("--positive-negative-bucket-multiplier", type=float, default=0.85)
    parser.add_argument("--protected-buckets", default="")
    parser.add_argument("--fragile-buckets", default="")
    parser.add_argument("--protected-base-multiplier", type=float, default=1.0)
    parser.add_argument("--protected-positive-multiplier", type=float, default=1.0)
    parser.add_argument("--protected-negative-multiplier", type=float, default=1.0)
    parser.add_argument("--protected-tie-multiplier", type=float, default=1.0)
    parser.add_argument("--fragile-base-multiplier", type=float, default=1.0)
    parser.add_argument("--fragile-positive-multiplier", type=float, default=1.0)
    parser.add_argument("--fragile-negative-multiplier", type=float, default=1.0)
    parser.add_argument("--fragile-tie-multiplier", type=float, default=1.0)
    parser.add_argument("--protected-strong-positive-min-delta", type=float, default=2.0)
    parser.add_argument("--protected-strong-confirm-min-delta", type=float, default=1.0)
    parser.add_argument("--protected-strong-positive-multiplier", type=float, default=1.0)
    parser.add_argument("--min-label-weight", type=float, default=0.05)
    parser.add_argument("--min-nonbase-labels", type=int, default=1)
    args = parser.parse_args()
    summary = build_calibrated_dataset(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir), args=args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
