from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.eon.lookahead_override_features import OVERRIDE_FEATURE_NAMES
from cse2026.experiments.eon.tree_ranker_runtime import (
    ADVANTAGE_BASE_RAW_FEATURE_INDICES,
    ADVANTAGE_FEATURE_NAMES,
)


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
    metadata = pd.read_csv(input_dir / "online_base_topn_examples.csv").reset_index(drop=True)
    npz = np.load(input_dir / "online_base_topn_examples.npz", allow_pickle=True)
    features = np.asarray(npz["features"], dtype=np.float32)
    targets = np.asarray(npz["targets"], dtype=np.float32)
    feature_names = [str(value) for value in npz["feature_names"].tolist()]
    if feature_names != list(OVERRIDE_FEATURE_NAMES):
        raise ValueError("online_base_topn feature layout does not match OVERRIDE_FEATURE_NAMES")
    if len(metadata) != int(features.shape[0]) or len(metadata) != int(targets.shape[0]):
        raise ValueError("metadata/features/targets row count mismatch")
    return {"metadata": metadata, "x": features, "target": targets, "feature_names": feature_names}


def _make_group_split(
    metadata: pd.DataFrame,
    *,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    group_rows: list[dict[str, Any]] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        non_base = group[~group["is_base"].astype(bool)]
        max_delta = float(non_base["accepted_delta_vs_base"].max()) if not non_base.empty else 0.0
        min_delta = float(non_base["accepted_delta_vs_base"].min()) if not non_base.empty else 0.0
        bucket = "win_available" if max_delta > 0.0 else ("loss_only" if min_delta < 0.0 else "tie_only")
        group_rows.append({"group_id": int(group_id), "bucket": bucket})
    group_table = pd.DataFrame(group_rows)
    rng = np.random.default_rng(int(seed))
    split_for_group: dict[int, str] = {}
    for _, bucket in group_table.groupby("bucket", sort=False):
        values = bucket["group_id"].astype(int).to_numpy(copy=True)
        rng.shuffle(values)
        train_end = int(round(len(values) * float(train_fraction)))
        cal_end = train_end + int(round(len(values) * float(calibration_fraction)))
        for value in values[:train_end]:
            split_for_group[int(value)] = "train"
        for value in values[train_end:cal_end]:
            split_for_group[int(value)] = "calibration"
        for value in values[cal_end:]:
            split_for_group[int(value)] = "eval"
    split = metadata["group_id"].astype(int).map(split_for_group).fillna("eval").to_numpy()
    return {
        "train": np.flatnonzero(split == "train").astype(np.int64),
        "calibration": np.flatnonzero(split == "calibration").astype(np.int64),
        "eval": np.flatnonzero(split == "eval").astype(np.int64),
    }


def _row_weights(metadata: pd.DataFrame, *, base_weight: float, win_weight: float, loss_weight: float, tie_weight: float) -> np.ndarray:
    is_base = metadata["is_base"].astype(bool).to_numpy()
    delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    weights = np.full((len(metadata),), float(tie_weight), dtype=np.float32)
    weights[delta > 0.0] = float(win_weight)
    weights[delta < 0.0] = float(loss_weight)
    weights[is_base] = float(base_weight)
    group_codes = pd.Categorical(metadata["group_id"], ordered=False).codes.astype(np.int32)
    group_sums = np.bincount(group_codes, weights=weights.astype(np.float64))
    group_counts = np.bincount(group_codes)
    group_mean = np.divide(group_sums, np.maximum(group_counts, 1), out=np.ones_like(group_sums), where=group_counts > 0)
    weights = weights / np.maximum(group_mean[group_codes].astype(np.float32), 1e-6)
    mean = float(np.mean(weights)) if weights.size else 1.0
    return (weights / max(mean, 1e-6)).astype(np.float32)


def _pos_weight(y: np.ndarray) -> float:
    positives = float(np.sum(np.asarray(y, dtype=np.float32) > 0.5))
    negatives = float(max(int(y.size) - int(positives), 0))
    return float(negatives / max(positives, 1.0))


def _train_xgboost_regressor(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    model_path: Path,
    num_boost_round: int,
    seed: int,
) -> Any:
    import xgboost as xgb

    matrix = xgb.DMatrix(x, label=y, weight=weights, feature_names=feature_names)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.04,
        "max_depth": 4,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": int(seed),
        "verbosity": 0,
    }
    model = xgb.train(params, matrix, num_boost_round=int(num_boost_round), verbose_eval=False)
    model.save_model(str(model_path))
    return model


def _train_xgboost_binary(
    *,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    model_path: Path,
    num_boost_round: int,
    seed: int,
) -> Any:
    import xgboost as xgb

    matrix = xgb.DMatrix(x, label=y, weight=weights, feature_names=feature_names)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.04,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "scale_pos_weight": _pos_weight(y),
        "seed": int(seed),
        "verbosity": 0,
    }
    model = xgb.train(params, matrix, num_boost_round=int(num_boost_round), verbose_eval=False)
    model.save_model(str(model_path))
    return model


def _predict_xgboost(model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    import xgboost as xgb

    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(model.predict(xgb.DMatrix(x, feature_names=feature_names)), dtype=np.float32)


def _build_advantage_dataset(metadata: pd.DataFrame, features: np.ndarray, ranker_scores: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    win_y: list[float] = []
    loss_y: list[float] = []
    delta_y: list[float] = []
    adv_metadata: list[dict[str, Any]] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        group_indices = [int(index) for index in group.index.to_numpy()]
        base_rows = group[group["is_base"].astype(bool)]
        if base_rows.empty:
            continue
        base_row_index = int(base_rows.index[0])
        base_features = features[int(base_row_index)]
        base_raw_features = base_features[list(ADVANTAGE_BASE_RAW_FEATURE_INDICES)]
        base_score = float(ranker_scores[int(base_row_index)])
        for row_index in group_indices:
            if int(row_index) == int(base_row_index):
                continue
            row = metadata.loc[int(row_index)]
            ranker_score = float(ranker_scores[int(row_index)])
            ranker_margin = float(ranker_score - base_score)
            accepted_delta = int(row["accepted_delta_vs_base"])
            feature_row = np.concatenate(
                [
                    features[int(row_index)].astype(np.float32),
                    base_raw_features.astype(np.float32),
                    np.asarray([ranker_score, ranker_margin], dtype=np.float32),
                ]
            )
            rows.append(feature_row.astype(np.float32))
            win_y.append(float(accepted_delta > 0))
            loss_y.append(float(accepted_delta < 0))
            delta_y.append(float(target[int(row_index)]))
            adv_metadata.append(
                {
                    "group_id": int(group_id),
                    "episode_id": str(row.get("episode_id", "")),
                    "request_id": int(row.get("request_id", 0)),
                    "traffic_scenario": str(row.get("traffic_scenario", "")),
                    "load_name": str(row.get("load_name", "")),
                    "candidate_index": int(row["candidate_index"]),
                    "base_index": int(row["base_index"]),
                    "ranker_score": float(ranker_score),
                    "ranker_margin": float(ranker_margin),
                    "accepted_delta_vs_base": int(accepted_delta),
                    "future_env_reward_delta_vs_base": float(row.get("future_env_reward_delta_vs_base", 0.0)),
                    "secondary_delta_vs_base": float(row.get("secondary_delta_vs_base", 0.0)),
                    "target_delta": float(target[int(row_index)]),
                    "is_win": bool(accepted_delta > 0),
                    "is_loss": bool(accepted_delta < 0),
                }
            )
    x = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, len(ADVANTAGE_FEATURE_NAMES)), dtype=np.float32)
    return {
        "x": x,
        "win_y": np.asarray(win_y, dtype=np.float32),
        "loss_y": np.asarray(loss_y, dtype=np.float32),
        "delta_y": np.asarray(delta_y, dtype=np.float32),
        "metadata": pd.DataFrame(adv_metadata),
    }


def _adv_weights(metadata: pd.DataFrame, target: str) -> np.ndarray:
    if metadata.empty:
        return np.zeros((0,), dtype=np.float32)
    delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    weights = np.full((len(metadata),), 0.35, dtype=np.float32)
    if target == "win":
        weights[delta > 0.0] = 8.0
        weights[delta < 0.0] = 2.0
    elif target == "loss":
        weights[delta < 0.0] = 10.0
        weights[delta > 0.0] = 2.0
    else:
        weights[delta > 0.0] = 6.0
        weights[delta < 0.0] = 8.0
        weights[np.isclose(delta, 0.0)] = 0.75
    group_codes = pd.Categorical(metadata["group_id"], ordered=False).codes.astype(np.int32)
    group_sums = np.bincount(group_codes, weights=weights.astype(np.float64))
    group_counts = np.bincount(group_codes)
    group_mean = np.divide(group_sums, np.maximum(group_counts, 1), out=np.ones_like(group_sums), where=group_counts > 0)
    weights = weights / np.maximum(group_mean[group_codes].astype(np.float32), 1e-6)
    return (weights / max(float(weights.mean()), 1e-6)).astype(np.float32)


def _gate_score(
    *,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    ranker_margin: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    return (
        float(thresholds.get("delta_weight", 1.0)) * np.asarray(delta_pred, dtype=np.float32)
        + float(thresholds.get("win_weight", 1.0)) * np.asarray(win_prob, dtype=np.float32)
        - float(thresholds.get("loss_weight", 2.0)) * np.asarray(loss_prob, dtype=np.float32)
        + float(thresholds.get("ranker_margin_weight", 0.0)) * np.asarray(ranker_margin, dtype=np.float32)
    )


def _selection_metrics(
    metadata: pd.DataFrame,
    *,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if metadata.empty:
        return {
            "groups": 0,
            "rows": 0,
            "override_count": 0,
            "override_rate_vs_base": 0.0,
            "total_selected_accepted_delta_vs_base": 0,
            "selected_loss_rate_when_overridden": None,
        }
    table = metadata.copy().reset_index(drop=True)
    table["win_prob"] = np.asarray(win_prob, dtype=np.float32)
    table["loss_prob"] = np.asarray(loss_prob, dtype=np.float32)
    table["delta_pred"] = np.asarray(delta_pred, dtype=np.float32)
    table["gate_score"] = _gate_score(
        win_prob=table["win_prob"].to_numpy(dtype=np.float32),
        loss_prob=table["loss_prob"].to_numpy(dtype=np.float32),
        delta_pred=table["delta_pred"].to_numpy(dtype=np.float32),
        ranker_margin=table["ranker_margin"].to_numpy(dtype=np.float32),
        thresholds=thresholds,
    )
    table["passes_gate"] = (
        (table["win_prob"] >= float(thresholds["min_win_prob"]))
        & (table["loss_prob"] <= float(thresholds["max_loss_prob"]))
        & (table["delta_pred"] >= float(thresholds["min_delta_pred"]))
    )
    selected_rows: list[dict[str, Any]] = []
    for _, group in table.groupby("group_id", sort=False):
        passed = group[group["passes_gate"]]
        if passed.empty:
            selected_rows.append(
                {
                    "override": False,
                    "selected_accepted_delta_vs_base": 0,
                    "selected_reward_delta_vs_base": 0.0,
                    "selected_loss": False,
                    "selected_win": False,
                }
            )
            continue
        selected = passed.sort_values(["gate_score", "candidate_index"], ascending=[False, True]).iloc[0]
        selected_rows.append(
            {
                "override": True,
                "selected_accepted_delta_vs_base": int(selected["accepted_delta_vs_base"]),
                "selected_reward_delta_vs_base": float(selected["future_env_reward_delta_vs_base"]),
                "selected_loss": bool(selected["is_loss"]),
                "selected_win": bool(selected["is_win"]),
            }
        )
    selected_table = pd.DataFrame(selected_rows)
    passed_table = table[table["passes_gate"]]
    override = selected_table[selected_table["override"].astype(bool)]
    return {
        "groups": int(table["group_id"].nunique()),
        "rows": int(len(table)),
        "win_rows": int(table["is_win"].sum()),
        "loss_rows": int(table["is_loss"].sum()),
        "candidate_gate_pass_rows": int(len(passed_table)),
        "candidate_gate_pass_rate": float(len(passed_table) / max(len(table), 1)),
        "candidate_win_precision_when_passed": None if passed_table.empty else float(passed_table["is_win"].mean()),
        "candidate_loss_rate_when_passed": None if passed_table.empty else float(passed_table["is_loss"].mean()),
        "override_count": int(len(override)),
        "override_rate_vs_base": float(len(override) / max(table["group_id"].nunique(), 1)),
        "selected_win_rate_when_overridden": None if override.empty else float(override["selected_win"].mean()),
        "selected_loss_rate_when_overridden": None if override.empty else float(override["selected_loss"].mean()),
        "total_selected_accepted_delta_vs_base": int(override["selected_accepted_delta_vs_base"].sum()) if not override.empty else 0,
        "mean_selected_accepted_delta_vs_base": float(selected_table["selected_accepted_delta_vs_base"].mean()),
        "mean_selected_reward_delta_vs_base": float(selected_table["selected_reward_delta_vs_base"].mean()),
        "thresholds": dict(thresholds),
    }


def _grid(values: np.ndarray, defaults: tuple[float, ...], *, lower: float | None = None, upper: float | None = None) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    items = set(float(value) for value in defaults)
    if finite.size:
        for q in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
            items.add(float(np.quantile(finite, q)))
    result = sorted(items)
    if lower is not None:
        result = [value for value in result if value >= float(lower)]
    if upper is not None:
        result = [value for value in result if value <= float(upper)]
    return result


def _tune_thresholds(
    metadata: pd.DataFrame,
    *,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    base = {
        "delta_weight": 1.0,
        "win_weight": 1.0,
        "loss_weight": 2.0,
        "ranker_margin_weight": 0.0,
    }
    best_thresholds: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_thresholds: dict[str, float] | None = None
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None

    win_grid = _grid(np.asarray(win_prob), (0.20, 0.35, 0.50, 0.65, 0.80), lower=0.0, upper=1.0)
    loss_grid = _grid(np.asarray(loss_prob), (0.01, 0.03, 0.05, 0.08, 0.12, 0.20), lower=0.0, upper=1.0)
    delta_grid = _grid(np.asarray(delta_pred), (-0.05, 0.0, 0.03, 0.08, 0.15), upper=10.0)
    for min_win_prob in win_grid:
        for max_loss_prob in loss_grid:
            for min_delta_pred in delta_grid:
                thresholds = dict(base)
                thresholds.update(
                    {
                        "min_win_prob": float(min_win_prob),
                        "max_loss_prob": float(max_loss_prob),
                        "min_delta_pred": float(min_delta_pred),
                    }
                )
                metrics = _selection_metrics(
                    metadata,
                    win_prob=win_prob,
                    loss_prob=loss_prob,
                    delta_pred=delta_pred,
                    thresholds=thresholds,
                )
                loss_rate = metrics.get("selected_loss_rate_when_overridden")
                loss_value = 0.0 if loss_rate is None else float(loss_rate)
                total_delta = float(metrics.get("total_selected_accepted_delta_vs_base") or 0.0)
                override_count = int(metrics.get("override_count") or 0)
                override_rate = float(metrics.get("override_rate_vs_base") or 0.0)
                key = (
                    total_delta,
                    float(metrics.get("mean_selected_reward_delta_vs_base") or 0.0),
                    -loss_value,
                    float(override_count),
                )
                if fallback_key is None or key > fallback_key:
                    fallback_key = key
                    fallback_thresholds = thresholds
                    fallback_metrics = dict(metrics)
                if override_count < int(min_override_count):
                    continue
                if total_delta < float(min_total_delta):
                    continue
                if loss_value > float(max_loss_rate):
                    continue
                if override_rate > float(max_override_rate):
                    continue
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = thresholds
                    best_metrics = dict(metrics)
    if best_thresholds is None or best_metrics is None:
        assert fallback_thresholds is not None and fallback_metrics is not None
        fallback_thresholds = dict(fallback_thresholds)
        fallback_thresholds["fallback_no_override"] = 0.0
        fallback_thresholds["tune_found_feasible"] = 0.0
        fallback_metrics["constraints_satisfied"] = False
        return fallback_thresholds, fallback_metrics
    best_thresholds = dict(best_thresholds)
    best_thresholds["fallback_no_override"] = 0.0
    best_thresholds["tune_found_feasible"] = 1.0
    best_metrics["constraints_satisfied"] = True
    return best_thresholds, best_metrics


def _emergency_safety_guard() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "emergency",
        "check_fragmentation": False,
        "check_small_gap": False,
        "check_lmax": False,
        "check_qot_margin": True,
        "check_energy": True,
        "check_delay": True,
        "fragmentation_slack": 0.50,
        "small_gap_slack": 1.0,
        "lmax_slack_slots": 40,
        "qot_margin_slack": 0.25,
        "energy_slack_w": 480.0,
        "delay_slack_ms": 10.0,
    }


def _export_artifact(
    *,
    output_dir: Path,
    ranker_model_path: Path,
    win_model_path: Path,
    loss_model_path: Path,
    delta_model_path: Path,
    thresholds: dict[str, float],
    candidate_pool: str,
    top_k: int,
    training_summary: dict[str, Any],
) -> Path:
    meta = {
        "backend": "xgboost",
        "model_path": ranker_model_path.name,
        "feature_names": list(OVERRIDE_FEATURE_NAMES),
        "candidate_pool": str(candidate_pool),
        "candidate_pool_top_k": int(top_k),
        "selection_mode": "positive_advantage",
        "residual_beta": 0.0,
        "selection_margin": 0.0,
        "base_policy": "energy-aware-ksp-bm-ff",
        "safety_guard": _emergency_safety_guard(),
        "risk_selector": {"enabled": False},
        "advantage_gate": {
            "enabled": True,
            "backend": "xgboost",
            "feature_names": list(ADVANTAGE_FEATURE_NAMES),
            "win_model_path": win_model_path.name,
            "loss_model_path": loss_model_path.name,
            "delta_model_path": delta_model_path.name,
            "win_min_accepted_delta": 1,
            "loss_min_accepted_delta": 1,
            "delta_target_mode": "accepted_or_secondary",
            **dict(thresholds),
        },
        "training": dict(training_summary),
    }
    artifact_path = output_dir / "xgboost_online_base_topn_exception_tree_ranker.json"
    _write_json(artifact_path, meta)
    return artifact_path


def train_exception_ranker(
    *,
    input_dir: Path,
    output_dir: Path,
    candidate_pool: str,
    top_k: int,
    train_fraction: float,
    calibration_fraction: float,
    secondary_scale: float,
    num_boost_round: int,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    seed: int,
) -> dict[str, Any]:
    data = _load_dataset(input_dir)
    metadata = data["metadata"].reset_index(drop=True)
    x = data["x"].astype(np.float32)
    raw_target = data["target"].astype(np.float32)
    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    secondary_delta = metadata["secondary_delta_vs_base"].to_numpy(dtype=np.float32)
    target = np.where(np.abs(accepted_delta) > 0.0, accepted_delta, float(secondary_scale) * secondary_delta).astype(np.float32)
    if metadata.empty:
        raise ValueError("No online base Top-N examples found")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = _make_group_split(
        metadata,
        train_fraction=float(train_fraction),
        calibration_fraction=float(calibration_fraction),
        seed=int(seed),
    )
    train_idx = splits["train"]
    calibration_idx = splits["calibration"]
    eval_idx = splits["eval"]
    ranker_model_path = output_dir / "xgboost_online_base_topn_ranker.json"
    ranker = _train_xgboost_regressor(
        x=x[train_idx],
        y=target[train_idx],
        weights=_row_weights(metadata.iloc[train_idx].reset_index(drop=True), base_weight=0.40, win_weight=8.0, loss_weight=6.0, tie_weight=0.50),
        feature_names=list(OVERRIDE_FEATURE_NAMES),
        model_path=ranker_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )
    scores = _predict_xgboost(ranker, x, list(OVERRIDE_FEATURE_NAMES))
    train_adv = _build_advantage_dataset(
        metadata.iloc[train_idx].reset_index(drop=True),
        x[train_idx],
        scores[train_idx],
        target[train_idx],
    )
    calibration_adv = _build_advantage_dataset(
        metadata.iloc[calibration_idx].reset_index(drop=True),
        x[calibration_idx],
        scores[calibration_idx],
        target[calibration_idx],
    )
    eval_adv = _build_advantage_dataset(
        metadata.iloc[eval_idx].reset_index(drop=True),
        x[eval_idx],
        scores[eval_idx],
        target[eval_idx],
    )
    if train_adv["x"].shape[0] == 0:
        raise ValueError("No non-base advantage examples found in train split")

    win_model_path = output_dir / "xgboost_online_base_topn_win.json"
    loss_model_path = output_dir / "xgboost_online_base_topn_loss.json"
    delta_model_path = output_dir / "xgboost_online_base_topn_delta.json"
    win_model = _train_xgboost_binary(
        x=train_adv["x"],
        y=train_adv["win_y"],
        weights=_adv_weights(train_adv["metadata"], "win"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=win_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )
    loss_model = _train_xgboost_binary(
        x=train_adv["x"],
        y=train_adv["loss_y"],
        weights=_adv_weights(train_adv["metadata"], "loss"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=loss_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )
    delta_model = _train_xgboost_regressor(
        x=train_adv["x"],
        y=train_adv["delta_y"],
        weights=_adv_weights(train_adv["metadata"], "delta"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=delta_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )

    train_pred = {
        "win": _predict_xgboost(win_model, train_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "loss": _predict_xgboost(loss_model, train_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "delta": _predict_xgboost(delta_model, train_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
    }
    cal_pred = {
        "win": _predict_xgboost(win_model, calibration_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "loss": _predict_xgboost(loss_model, calibration_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "delta": _predict_xgboost(delta_model, calibration_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
    }
    eval_pred = {
        "win": _predict_xgboost(win_model, eval_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "loss": _predict_xgboost(loss_model, eval_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "delta": _predict_xgboost(delta_model, eval_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
    }
    thresholds, calibration_metrics = _tune_thresholds(
        calibration_adv["metadata"],
        win_prob=cal_pred["win"],
        loss_prob=cal_pred["loss"],
        delta_pred=cal_pred["delta"],
        max_loss_rate=float(max_loss_rate),
        min_override_count=int(min_override_count),
        min_total_delta=float(min_total_delta),
        max_override_rate=float(max_override_rate),
    )
    train_metrics = _selection_metrics(
        train_adv["metadata"],
        win_prob=train_pred["win"],
        loss_prob=train_pred["loss"],
        delta_pred=train_pred["delta"],
        thresholds=thresholds,
    )
    eval_metrics = _selection_metrics(
        eval_adv["metadata"],
        win_prob=eval_pred["win"],
        loss_prob=eval_pred["loss"],
        delta_pred=eval_pred["delta"],
        thresholds=thresholds,
    )

    train_adv["metadata"].to_csv(output_dir / "train_online_base_topn_advantage_examples.csv", index=False)
    calibration_adv["metadata"].to_csv(output_dir / "calibration_online_base_topn_advantage_examples.csv", index=False)
    eval_adv["metadata"].to_csv(output_dir / "eval_online_base_topn_advantage_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "online_base_topn_advantage_splits.npz",
        train_x=train_adv["x"].astype(np.float32),
        train_win=train_adv["win_y"].astype(np.float32),
        train_loss=train_adv["loss_y"].astype(np.float32),
        train_delta=train_adv["delta_y"].astype(np.float32),
        calibration_x=calibration_adv["x"].astype(np.float32),
        eval_x=eval_adv["x"].astype(np.float32),
        feature_names=np.asarray(ADVANTAGE_FEATURE_NAMES, dtype=object),
    )
    training_summary = {
        "input_dir": str(input_dir),
        "candidate_pool": str(candidate_pool),
        "candidate_pool_top_k": int(top_k),
        "train_fraction": float(train_fraction),
        "calibration_fraction": float(calibration_fraction),
        "secondary_scale": float(secondary_scale),
        "num_boost_round": int(num_boost_round),
        "max_loss_rate": float(max_loss_rate),
        "min_override_count": int(min_override_count),
        "min_total_delta": float(min_total_delta),
        "max_override_rate": float(max_override_rate),
        "seed": int(seed),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "target": {
            "mean": float(np.mean(target)),
            "std": float(np.std(target)),
            "positive_rows": int(np.sum(target > 1.0e-6)),
            "negative_rows": int(np.sum(target < -1.0e-6)),
            "zero_rows": int(np.sum(np.abs(target) <= 1.0e-6)),
            "raw_target_mean": float(np.mean(raw_target)),
        },
    }
    artifact_path = _export_artifact(
        output_dir=output_dir,
        ranker_model_path=ranker_model_path,
        win_model_path=win_model_path,
        loss_model_path=loss_model_path,
        delta_model_path=delta_model_path,
        thresholds=thresholds,
        candidate_pool=str(candidate_pool),
        top_k=int(top_k),
        training_summary=training_summary,
    )
    summary = {
        "artifact_path": str(artifact_path),
        "ranker_model_path": str(ranker_model_path),
        "win_model_path": str(win_model_path),
        "loss_model_path": str(loss_model_path),
        "delta_model_path": str(delta_model_path),
        "thresholds": dict(thresholds),
        "training": training_summary,
        "label_counts": {
            "groups": int(metadata["group_id"].nunique()),
            "rows": int(len(metadata)),
            "non_base_rows": int((~metadata["is_base"].astype(bool)).sum()),
            "win_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) > 0).sum()),
            "loss_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) < 0).sum()),
            "tie_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) == 0).sum()),
        },
        "gate": {
            "train": train_metrics,
            "calibration": calibration_metrics,
            "eval": eval_metrics,
        },
    }
    _write_json(output_dir / "online_base_topn_exception_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an online base-trajectory Top-N exception ranker.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-pool", default="energy_topk_hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--secondary-scale", type=float, default=0.25)
    parser.add_argument("--num-boost-round", type=int, default=120)
    parser.add_argument("--max-loss-rate", type=float, default=0.05)
    parser.add_argument("--min-override-count", type=int, default=5)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()
    summary = train_exception_ranker(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        candidate_pool=str(args.candidate_pool),
        top_k=int(args.top_k),
        train_fraction=float(args.train_fraction),
        calibration_fraction=float(args.calibration_fraction),
        secondary_scale=float(args.secondary_scale),
        num_boost_round=int(args.num_boost_round),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
