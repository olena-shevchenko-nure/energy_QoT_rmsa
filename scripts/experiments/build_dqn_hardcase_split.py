from __future__ import annotations

import argparse
import json
import math
import shutil
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


def _load_split(source_dir: Path, split: str) -> dict[str, Any]:
    csv_path = source_dir / f"{split}_dagger_tree_ranker_examples.csv"
    npz_path = source_dir / f"{split}_dagger_tree_ranker_examples.npz"
    metadata = pd.read_csv(csv_path).reset_index(drop=True)
    npz = np.load(npz_path, allow_pickle=True)
    features = np.asarray(npz["features"], dtype=np.float32)
    targets = np.asarray(npz["targets"], dtype=np.float32) if "targets" in npz else np.zeros((len(metadata),), dtype=np.float32)
    feature_names = np.asarray(npz["feature_names"], dtype=object)
    if len(metadata) != int(features.shape[0]):
        raise ValueError(f"{split}: metadata rows ({len(metadata)}) != feature rows ({features.shape[0]})")
    if len(metadata) != int(targets.shape[0]):
        raise ValueError(f"{split}: metadata rows ({len(metadata)}) != target rows ({targets.shape[0]})")
    return {
        "metadata": metadata,
        "features": features,
        "targets": targets,
        "feature_names": feature_names,
    }


def _group_sizes(metadata: pd.DataFrame) -> np.ndarray:
    return metadata.groupby("group_id", sort=False).size().to_numpy(dtype=np.int32)


def _write_split(output_dir: Path, split: str, source: dict[str, Any], row_indices: np.ndarray) -> None:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    metadata = source["metadata"].iloc[row_indices].reset_index(drop=True).copy()
    features = source["features"][row_indices].astype(np.float32)
    targets = source["targets"][row_indices].astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / f"{split}_dagger_tree_ranker_examples.csv", index=False)
    np.savez_compressed(
        output_dir / f"{split}_dagger_tree_ranker_examples.npz",
        features=features,
        targets=targets,
        group_sizes=_group_sizes(metadata),
        feature_names=np.asarray(source["feature_names"], dtype=object),
    )


def _sample_stratified(
    group_info: pd.DataFrame,
    group_ids: set[int],
    count: int,
    rng: np.random.Generator,
    *,
    strata: tuple[str, ...] = ("traffic_scenario", "load_name"),
) -> list[int]:
    if count <= 0 or not group_ids:
        return []
    pool = group_info[group_info["group_id"].astype(int).isin(group_ids)].copy()
    count = min(int(count), int(len(pool)))
    if count <= 0:
        return []
    if not all(column in pool.columns for column in strata):
        values = pool["group_id"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(values)
        return [int(value) for value in values[:count]]

    sampled: list[int] = []
    grouped = list(pool.groupby(list(strata), sort=False))
    total = max(len(pool), 1)
    quotas: list[tuple[int, float, pd.DataFrame]] = []
    used = 0
    for _, group in grouped:
        raw = count * len(group) / total
        floor = min(int(math.floor(raw)), len(group))
        quotas.append((floor, raw - floor, group))
        used += floor
    remaining = count - used
    order = sorted(range(len(quotas)), key=lambda index: (-quotas[index][1], index))
    quota_counts = [item[0] for item in quotas]
    for index in order:
        if remaining <= 0:
            break
        capacity = len(quotas[index][2]) - quota_counts[index]
        if capacity <= 0:
            continue
        add = min(capacity, remaining)
        quota_counts[index] += add
        remaining -= add

    for quota, _, group in [(quota_counts[i], quotas[i][1], quotas[i][2]) for i in range(len(quotas))]:
        if quota <= 0:
            continue
        values = group["group_id"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(values)
        sampled.extend(int(value) for value in values[:quota])
    if len(sampled) < count:
        selected = set(sampled)
        rest = pool[~pool["group_id"].astype(int).isin(selected)]["group_id"].to_numpy(dtype=np.int64).copy()
        rng.shuffle(rest)
        sampled.extend(int(value) for value in rest[: count - len(sampled)])
    return sampled[:count]


def _group_info(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        nonbase = group[group["candidate_index"].astype(int) != group["base_index"].astype(int)]
        accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
        zero_rows = nonbase[nonbase["accepted_delta_vs_base"].astype(float) == 0.0] if not nonbase.empty else nonbase
        secondary = (
            zero_rows["secondary_delta_vs_base"].astype(float)
            if "secondary_delta_vs_base" in zero_rows.columns and not zero_rows.empty
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "group_id": int(group_id),
                "traffic_scenario": str(group["traffic_scenario"].iloc[0]) if "traffic_scenario" in group else "",
                "load_name": str(group["load_name"].iloc[0]) if "load_name" in group else "",
                "episode_id": str(group["episode_id"].iloc[0]) if "episode_id" in group else "",
                "request_id": int(group["request_id"].iloc[0]) if "request_id" in group else -1,
                "rows": int(len(group)),
                "has_positive": bool((accepted > 0.0).any()) if not accepted.empty else False,
                "has_negative": bool((accepted < 0.0).any()) if not accepted.empty else False,
                "max_positive_delta": float(accepted.max()) if not accepted.empty else 0.0,
                "min_negative_delta": float(accepted.min()) if not accepted.empty else 0.0,
                "max_tie_secondary_delta": float(secondary.max()) if not secondary.empty else float("-inf"),
            }
        )
    return pd.DataFrame(rows)


def _summarize_split(metadata: pd.DataFrame, *, name: str) -> dict[str, Any]:
    nonbase = metadata[metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)]
    accepted = nonbase["accepted_delta_vs_base"].astype(float) if not nonbase.empty else pd.Series(dtype=float)
    groups = metadata["group_id"].nunique()
    pos_groups = nonbase[nonbase["accepted_delta_vs_base"].astype(float) > 0]["group_id"].nunique() if not nonbase.empty else 0
    neg_groups = nonbase[nonbase["accepted_delta_vs_base"].astype(float) < 0]["group_id"].nunique() if not nonbase.empty else 0
    by_context = (
        metadata.drop_duplicates("group_id")
        .groupby(["traffic_scenario", "load_name"], dropna=False)
        .size()
        .reset_index(name="groups")
        .to_dict(orient="records")
        if {"traffic_scenario", "load_name"}.issubset(metadata.columns)
        else []
    )
    return {
        "name": name,
        "rows": int(len(metadata)),
        "groups": int(groups),
        "positive_groups": int(pos_groups),
        "positive_group_rate": float(pos_groups / max(groups, 1)),
        "negative_groups": int(neg_groups),
        "negative_group_rate": float(neg_groups / max(groups, 1)),
        "accepted_delta_counts": {str(key): int(value) for key, value in accepted.value_counts().sort_index().items()},
        "by_context": by_context,
    }


def build_split(
    *,
    source_dir: Path,
    output_dir: Path,
    hard_positive_fraction: float,
    hard_negative_fraction: float,
    high_secondary_tie_fraction: float,
    normal_fraction: float,
    calibration_fraction: float,
    high_secondary_quantile: float,
    seed: int,
) -> dict[str, Any]:
    if abs((hard_positive_fraction + hard_negative_fraction + high_secondary_tie_fraction + normal_fraction) - 1.0) > 1e-6:
        raise ValueError("train bucket fractions must sum to 1.0")
    rng = np.random.default_rng(int(seed))
    train = _load_split(source_dir, "train")
    eval_split = _load_split(source_dir, "eval")
    metadata = train["metadata"].reset_index(drop=True)
    group_info = _group_info(metadata)
    all_groups = set(int(value) for value in group_info["group_id"].to_numpy(dtype=np.int64))

    calibration_count = int(round(float(calibration_fraction) * len(all_groups)))
    calibration_groups = set(_sample_stratified(group_info, all_groups, calibration_count, rng))
    available = all_groups - calibration_groups
    available_info = group_info[group_info["group_id"].astype(int).isin(available)].copy()

    hard_positive = set(available_info[available_info["has_positive"]]["group_id"].astype(int).tolist())
    non_positive = available - hard_positive
    hard_negative_pool = set(
        available_info[
            available_info["group_id"].astype(int).isin(non_positive) & available_info["has_negative"]
        ]["group_id"].astype(int).tolist()
    )
    tie_candidates = available_info[
        available_info["group_id"].astype(int).isin(non_positive - hard_negative_pool)
        & np.isfinite(available_info["max_tie_secondary_delta"].astype(float))
    ].copy()
    tie_threshold = (
        float(tie_candidates["max_tie_secondary_delta"].quantile(float(high_secondary_quantile)))
        if not tie_candidates.empty
        else float("inf")
    )
    high_secondary_tie_pool = set(
        tie_candidates[tie_candidates["max_tie_secondary_delta"].astype(float) >= tie_threshold]["group_id"].astype(int).tolist()
    )
    normal_pool = available - hard_positive - hard_negative_pool - high_secondary_tie_pool

    target_total = int(math.ceil(len(hard_positive) / max(float(hard_positive_fraction), 1.0e-9)))
    target_total = min(target_total, len(available))
    target_negative = int(round(target_total * float(hard_negative_fraction)))
    target_tie = int(round(target_total * float(high_secondary_tie_fraction)))
    target_normal = max(0, target_total - len(hard_positive) - target_negative - target_tie)

    selected: dict[str, list[int]] = {
        "hard_positive": _sample_stratified(group_info, hard_positive, len(hard_positive), rng),
        "hard_negative": _sample_stratified(group_info, hard_negative_pool, target_negative, rng),
        "high_secondary_tie": _sample_stratified(group_info, high_secondary_tie_pool, target_tie, rng),
        "normal": _sample_stratified(group_info, normal_pool, target_normal, rng),
    }
    selected_groups = set(value for values in selected.values() for value in values)
    shortage = max(0, target_total - len(selected_groups))
    if shortage > 0:
        fallback_pool = available - selected_groups
        selected["fallback_fill"] = _sample_stratified(group_info, fallback_pool, shortage, rng)
        selected_groups.update(selected["fallback_fill"])

    group_to_bucket: dict[int, str] = {}
    for bucket, groups in selected.items():
        for group_id in groups:
            group_to_bucket[int(group_id)] = bucket

    row_group_ids = metadata["group_id"].astype(int).to_numpy()
    train_indices = np.flatnonzero(np.isin(row_group_ids, np.asarray(sorted(selected_groups), dtype=np.int64)))
    calibration_indices = np.flatnonzero(
        np.isin(row_group_ids, np.asarray(sorted(calibration_groups), dtype=np.int64))
    )
    eval_indices = np.arange(len(eval_split["metadata"]), dtype=np.int64)

    _write_split(output_dir, "train", train, train_indices)
    _write_split(output_dir, "calibration", train, calibration_indices)
    _write_split(output_dir, "eval", eval_split, eval_indices)

    assignments = group_info[group_info["group_id"].astype(int).isin(selected_groups | calibration_groups)].copy()
    assignments["split_assignment"] = assignments["group_id"].astype(int).map(
        lambda value: "calibration" if int(value) in calibration_groups else "train"
    )
    assignments["train_bucket"] = assignments["group_id"].astype(int).map(lambda value: group_to_bucket.get(int(value), ""))
    assignments.to_csv(output_dir / "group_assignments.csv", index=False)

    train_metadata = train["metadata"].iloc[train_indices].reset_index(drop=True)
    calibration_metadata = train["metadata"].iloc[calibration_indices].reset_index(drop=True)
    eval_metadata = eval_split["metadata"].iloc[eval_indices].reset_index(drop=True)
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "fractions": {
            "hard_positive": float(hard_positive_fraction),
            "hard_negative": float(hard_negative_fraction),
            "high_secondary_tie": float(high_secondary_tie_fraction),
            "normal": float(normal_fraction),
            "calibration": float(calibration_fraction),
        },
        "high_secondary_quantile": float(high_secondary_quantile),
        "tie_threshold": tie_threshold,
        "available_groups_after_calibration": int(len(available)),
        "pool_groups": {
            "hard_positive": int(len(hard_positive)),
            "hard_negative": int(len(hard_negative_pool)),
            "high_secondary_tie": int(len(high_secondary_tie_pool)),
            "normal": int(len(normal_pool)),
        },
        "selected_train_groups_by_bucket": {key: int(len(value)) for key, value in selected.items()},
        "splits": {
            "train": _summarize_split(train_metadata, name="train_hardcase_enriched"),
            "calibration": _summarize_split(calibration_metadata, name="calibration_full_distribution"),
            "eval": _summarize_split(eval_metadata, name="eval_full_distribution"),
        },
    }
    _write_json(output_dir / "hardcase_split_summary.json", summary)

    for name in ("config.yaml", "config.resolved.yaml", "metrics.json", "seed_info.json", "git_info.json"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / f"source_{name}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard-case enriched DQN split from DAgger counterfactual data.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hard-positive-fraction", type=float, default=0.50)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.25)
    parser.add_argument("--high-secondary-tie-fraction", type=float, default=0.15)
    parser.add_argument("--normal-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--high-secondary-quantile", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()
    summary = build_split(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        hard_positive_fraction=float(args.hard_positive_fraction),
        hard_negative_fraction=float(args.hard_negative_fraction),
        high_secondary_tie_fraction=float(args.high_secondary_tie_fraction),
        normal_fraction=float(args.normal_fraction),
        calibration_fraction=float(args.calibration_fraction),
        high_secondary_quantile=float(args.high_secondary_quantile),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
