from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


LIVE_OVERRIDE_GATE_FEATURE_NAMES = (
    "valid_candidates_norm",
    "eligible_count_norm",
    "selected_index_norm",
    "base_index_norm",
    "selected_score_rank_norm",
    "base_score_rank_norm",
    "selected_win_prob",
    "selected_loss_prob",
    "selected_delta_pred",
    "selected_score",
    "base_win_prob",
    "base_loss_prob",
    "base_delta_pred",
    "base_score",
    "win_prob_delta",
    "loss_prob_delta",
    "delta_pred_delta",
    "score_delta",
    "selected_energy_increment_norm",
    "base_energy_increment_norm",
    "energy_increment_norm_delta",
    "selected_fragmentation_after",
    "base_fragmentation_after",
    "fragmentation_after_delta",
    "selected_delta_fragmentation",
    "base_delta_fragmentation",
    "delta_fragmentation_delta",
    "selected_largest_free_block_norm",
    "base_largest_free_block_norm",
    "largest_free_block_delta_norm",
    "selected_small_gap_penalty",
    "base_small_gap_penalty",
    "small_gap_delta",
    "selected_qot_margin_norm",
    "base_qot_margin_norm",
    "qot_margin_delta",
    "selected_qot_risk",
    "base_qot_risk",
    "qot_risk_delta",
    "selected_delay_norm",
    "base_delay_norm",
    "delay_delta_norm",
    "selected_j_total",
    "base_j_total",
    "j_total_delta",
    "selected_width_norm",
    "base_width_norm",
    "width_delta_norm",
    "selected_route_id_norm",
    "base_route_id_norm",
    "route_id_delta_norm",
    "same_route",
    "same_modulation",
    "selected_is_j_total",
    "base_is_j_total",
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


def _safe(value: float, scale: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(value) / max(float(scale), 1.0e-9)


def _f(row: pd.Series, name: str, default: float = 0.0) -> float:
    if name not in row:
        return float(default)
    value = row[name]
    if pd.isna(value):
        return float(default)
    return float(value)


def _feature_values_from_row(row: pd.Series, *, n_max: int, slots: int, delay_bound_ms: float) -> dict[str, float]:
    valid_candidates = max(int(_f(row, "valid_candidates", 1.0)), 1)
    selected_index = int(_f(row, "selected_index", 0.0))
    base_index = int(_f(row, "base_index", 0.0))
    route_scale = 8.0
    values = {
        "valid_candidates_norm": _safe(valid_candidates, n_max),
        "eligible_count_norm": _safe(_f(row, "eligible_count"), n_max),
        "selected_index_norm": _safe(selected_index, max(n_max - 1, 1)),
        "base_index_norm": _safe(base_index, max(n_max - 1, 1)),
        "selected_score_rank_norm": _safe(_f(row, "selected_score_rank", valid_candidates), valid_candidates),
        "base_score_rank_norm": _safe(_f(row, "base_score_rank", valid_candidates), valid_candidates),
        "selected_win_prob": _f(row, "selected_win_prob"),
        "selected_loss_prob": _f(row, "selected_loss_prob"),
        "selected_delta_pred": _f(row, "selected_delta_pred"),
        "selected_score": _f(row, "selected_score"),
        "base_win_prob": _f(row, "base_win_prob"),
        "base_loss_prob": _f(row, "base_loss_prob"),
        "base_delta_pred": _f(row, "base_delta_pred"),
        "base_score": _f(row, "base_score"),
        "selected_energy_increment_norm": _f(row, "selected_energy_increment_norm"),
        "base_energy_increment_norm": _f(row, "base_energy_increment_norm"),
        "selected_fragmentation_after": _f(row, "selected_fragmentation_after"),
        "base_fragmentation_after": _f(row, "base_fragmentation_after"),
        "selected_delta_fragmentation": _f(row, "selected_delta_fragmentation"),
        "base_delta_fragmentation": _f(row, "base_delta_fragmentation"),
        "selected_largest_free_block_norm": _safe(_f(row, "selected_largest_free_block_after"), slots),
        "base_largest_free_block_norm": _safe(_f(row, "base_largest_free_block_after"), slots),
        "selected_small_gap_penalty": _f(row, "selected_small_gap_penalty"),
        "base_small_gap_penalty": _f(row, "base_small_gap_penalty"),
        "selected_qot_margin_norm": _f(row, "selected_qot_margin_norm"),
        "base_qot_margin_norm": _f(row, "base_qot_margin_norm"),
        "selected_qot_risk": _f(row, "selected_qot_risk"),
        "base_qot_risk": _f(row, "base_qot_risk"),
        "selected_delay_norm": _safe(_f(row, "selected_delay_ms"), delay_bound_ms),
        "base_delay_norm": _safe(_f(row, "base_delay_ms"), delay_bound_ms),
        "selected_j_total": _f(row, "selected_j_total"),
        "base_j_total": _f(row, "base_j_total"),
        "selected_width_norm": _safe(_f(row, "selected_width"), slots),
        "base_width_norm": _safe(_f(row, "base_width"), slots),
        "selected_route_id_norm": _safe(_f(row, "selected_route_id"), route_scale),
        "base_route_id_norm": _safe(_f(row, "base_route_id"), route_scale),
        "same_route": 1.0 if int(_f(row, "selected_route_id")) == int(_f(row, "base_route_id")) else 0.0,
        "same_modulation": 1.0
        if int(_f(row, "selected_modulation_index")) == int(_f(row, "base_modulation_index"))
        else 0.0,
        "selected_is_j_total": 1.0 if selected_index == 0 else 0.0,
        "base_is_j_total": 1.0 if base_index == 0 else 0.0,
    }
    values.update(
        {
            "win_prob_delta": values["selected_win_prob"] - values["base_win_prob"],
            "loss_prob_delta": values["selected_loss_prob"] - values["base_loss_prob"],
            "delta_pred_delta": values["selected_delta_pred"] - values["base_delta_pred"],
            "score_delta": values["selected_score"] - values["base_score"],
            "energy_increment_norm_delta": values["selected_energy_increment_norm"] - values["base_energy_increment_norm"],
            "fragmentation_after_delta": values["selected_fragmentation_after"] - values["base_fragmentation_after"],
            "delta_fragmentation_delta": values["selected_delta_fragmentation"] - values["base_delta_fragmentation"],
            "largest_free_block_delta_norm": values["selected_largest_free_block_norm"] - values["base_largest_free_block_norm"],
            "small_gap_delta": values["selected_small_gap_penalty"] - values["base_small_gap_penalty"],
            "qot_margin_delta": values["selected_qot_margin_norm"] - values["base_qot_margin_norm"],
            "qot_risk_delta": values["selected_qot_risk"] - values["base_qot_risk"],
            "delay_delta_norm": values["selected_delay_norm"] - values["base_delay_norm"],
            "j_total_delta": values["selected_j_total"] - values["base_j_total"],
            "width_delta_norm": values["selected_width_norm"] - values["base_width_norm"],
            "route_id_delta_norm": values["selected_route_id_norm"] - values["base_route_id_norm"],
        }
    )
    return values


def _feature_matrix(metadata: pd.DataFrame, *, n_max: int, slots: int, delay_bound_ms: float) -> np.ndarray:
    rows: list[list[float]] = []
    for _, row in metadata.iterrows():
        values = _feature_values_from_row(row, n_max=int(n_max), slots=int(slots), delay_bound_ms=float(delay_bound_ms))
        rows.append([float(values.get(name, 0.0)) for name in LIVE_OVERRIDE_GATE_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float32)


def _episode_folds(metadata: pd.DataFrame, folds: int, seed: int) -> list[np.ndarray]:
    table: list[dict[str, Any]] = []
    for episode_id, group in metadata.groupby("episode_id", sort=False):
        accepted = group["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
        table.append(
            {
                "episode_id": str(episode_id),
                "rows": int(len(group)),
                "wins": int((accepted > 0.0).sum()),
                "losses": int((accepted < 0.0).sum()),
                "abs_delta": float(abs(float(accepted.sum()))),
            }
        )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(table)
    table.sort(key=lambda row: (int(row["wins"]) + int(row["losses"]), int(row["rows"]), float(row["abs_delta"])), reverse=True)
    buckets: list[list[str]] = [[] for _ in range(int(folds))]
    stats = [{"rows": 0, "wins": 0, "losses": 0} for _ in range(int(folds))]
    for row in table:
        best = min(range(int(folds)), key=lambda index: (stats[index]["rows"], stats[index]["wins"], stats[index]["losses"]))
        buckets[best].append(str(row["episode_id"]))
        stats[best]["rows"] += int(row["rows"])
        stats[best]["wins"] += int(row["wins"])
        stats[best]["losses"] += int(row["losses"])
    result: list[np.ndarray] = []
    for bucket in buckets:
        mask = metadata["episode_id"].astype(str).isin(bucket).to_numpy()
        result.append(np.flatnonzero(mask).astype(np.int64))
    return result


def _pos_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y > 0.5))
    negatives = float(max(int(y.size) - int(positives), 0))
    return float(negatives / max(positives, 1.0))


def _train_xgboost(
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    objective: str,
    sample_weight: np.ndarray,
    num_boost_round: int,
    seed: int,
    device: str,
) -> Any:
    import xgboost as xgb

    matrix = xgb.DMatrix(x.astype(np.float32), label=y.astype(np.float32), weight=sample_weight.astype(np.float32), feature_names=feature_names)
    params: dict[str, Any] = {
        "objective": str(objective),
        "eta": 0.04,
        "max_depth": 3,
        "min_child_weight": 2.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": int(seed),
        "verbosity": 0,
    }
    if str(device).strip():
        params["device"] = str(device)
    if str(objective) == "binary:logistic":
        params["eval_metric"] = "logloss"
        params["scale_pos_weight"] = _pos_weight(y)
    else:
        params["eval_metric"] = "rmse"
    return xgb.train(params, matrix, num_boost_round=int(num_boost_round), verbose_eval=False)


def _predict_xgboost(model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    import xgboost as xgb

    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    matrix = xgb.DMatrix(x.astype(np.float32), feature_names=feature_names)
    return np.asarray(model.predict(matrix), dtype=np.float32).reshape(-1)


def _sample_weight(accepted_delta: np.ndarray, *, win_weight: float, loss_weight: float, tie_weight: float) -> np.ndarray:
    return np.where(
        accepted_delta > 0.0,
        float(win_weight),
        np.where(accepted_delta < 0.0, float(loss_weight), float(tie_weight)),
    ).astype(np.float32)


def _metrics(metadata: pd.DataFrame, allowed: np.ndarray) -> dict[str, Any]:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    reward = metadata["future_env_reward_delta_vs_base"].to_numpy(dtype=np.float32)
    allowed = np.asarray(allowed, dtype=bool)
    selected_delta = np.where(allowed, accepted, 0.0)
    selected_reward = np.where(allowed, reward, 0.0)
    override_count = int(allowed.sum())
    selected = metadata.loc[allowed]
    loss_rate = None if override_count == 0 else float((selected["accepted_delta_vs_base"].astype(float) < 0.0).mean())
    win_rate = None if override_count == 0 else float((selected["accepted_delta_vs_base"].astype(float) > 0.0).mean())
    tie_rate = None if override_count == 0 else float((selected["accepted_delta_vs_base"].astype(float) == 0.0).mean())
    return {
        "examples": int(len(metadata)),
        "override_count": override_count,
        "override_rate": float(override_count / max(len(metadata), 1)),
        "vetoed_count": int(len(metadata) - override_count),
        "total_accepted_delta": int(round(float(selected_delta.sum()))),
        "mean_accepted_delta_all": float(selected_delta.mean()) if len(metadata) else 0.0,
        "total_reward_delta": float(selected_reward.sum()),
        "mean_reward_delta_all": float(selected_reward.mean()) if len(metadata) else 0.0,
        "selected_loss_rate": loss_rate,
        "selected_win_rate": win_rate,
        "selected_tie_rate": tie_rate,
        "selected_wins": int((selected["accepted_delta_vs_base"].astype(float) > 0.0).sum()) if override_count else 0,
        "selected_losses": int((selected["accepted_delta_vs_base"].astype(float) < 0.0).sum()) if override_count else 0,
        "selected_ties": int((selected["accepted_delta_vs_base"].astype(float) == 0.0).sum()) if override_count else 0,
    }


def _raw_metrics(metadata: pd.DataFrame) -> dict[str, Any]:
    return _metrics(metadata, np.ones((len(metadata),), dtype=bool))


def _allowed(
    *,
    win_pred: np.ndarray,
    loss_pred: np.ndarray,
    delta_pred: np.ndarray,
    thresholds: dict[str, float],
    score_weights: dict[str, float],
) -> np.ndarray:
    combined = (
        float(score_weights.get("win", 1.0)) * win_pred
        - float(score_weights.get("loss", 1.0)) * loss_pred
        + float(score_weights.get("delta", 1.0)) * delta_pred
    )
    return (
        (win_pred >= float(thresholds.get("win_threshold", -math.inf)))
        & (loss_pred <= float(thresholds.get("loss_threshold", math.inf)))
        & (delta_pred >= float(thresholds.get("delta_threshold", -math.inf)))
        & (combined >= float(thresholds.get("combined_threshold", -math.inf)))
    )


def _grid(values: np.ndarray, fixed: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    candidates = list(fixed)
    if arr.size:
        candidates.extend(float(value) for value in np.quantile(arr, np.linspace(0.05, 0.95, 10)).tolist())
    return sorted(set(round(float(value), 6) for value in candidates))


def _tune_thresholds(
    metadata: pd.DataFrame,
    *,
    win_pred: np.ndarray,
    loss_pred: np.ndarray,
    delta_pred: np.ndarray,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    score_weights: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    win_grid = _grid(win_pred, [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60])
    loss_grid = _grid(loss_pred, [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    delta_grid = _grid(delta_pred, [-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0])
    combined = (
        float(score_weights.get("win", 1.0)) * win_pred
        - float(score_weights.get("loss", 1.0)) * loss_pred
        + float(score_weights.get("delta", 1.0)) * delta_pred
    )
    combined_grid = _grid(combined, [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
    best_thresholds: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_thresholds: dict[str, float] | None = None
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    leaderboard: list[dict[str, Any]] = []
    for win_threshold in win_grid:
        for loss_threshold in loss_grid:
            for delta_threshold in delta_grid:
                for combined_threshold in combined_grid:
                    thresholds = {
                        "win_threshold": float(win_threshold),
                        "loss_threshold": float(loss_threshold),
                        "delta_threshold": float(delta_threshold),
                        "combined_threshold": float(combined_threshold),
                    }
                    metrics = _metrics(
                        metadata,
                        _allowed(
                            win_pred=win_pred,
                            loss_pred=loss_pred,
                            delta_pred=delta_pred,
                            thresholds=thresholds,
                            score_weights=score_weights,
                        ),
                    )
                    loss_rate = metrics["selected_loss_rate"]
                    loss_value = float(loss_rate if loss_rate is not None else 0.0)
                    key = (
                        float(metrics["total_accepted_delta"]),
                        float(metrics["total_reward_delta"]),
                        -loss_value,
                        float(metrics["override_count"]),
                        -float(combined_threshold),
                    )
                    row = {**thresholds, **metrics}
                    leaderboard.append(row)
                    if fallback_key is None or key > fallback_key:
                        fallback_key = key
                        fallback_thresholds = dict(thresholds)
                        fallback_metrics = dict(metrics)
                    if int(metrics["override_count"]) < int(min_override_count):
                        continue
                    if float(metrics["override_rate"]) > float(max_override_rate):
                        continue
                    if float(metrics["total_accepted_delta"]) < float(min_total_delta):
                        continue
                    if loss_rate is not None and float(loss_rate) > float(max_loss_rate):
                        continue
                    if best_key is None or key > best_key:
                        best_key = key
                        best_thresholds = dict(thresholds)
                        best_metrics = dict(metrics)
    leaderboard.sort(
        key=lambda row: (
            float(row["total_accepted_delta"]),
            float(row["total_reward_delta"]),
            -float(row["selected_loss_rate"] if row["selected_loss_rate"] is not None else 0.0),
            float(row["override_count"]),
        ),
        reverse=True,
    )
    top_rows = leaderboard[:50]
    if best_thresholds is None or best_metrics is None:
        assert fallback_thresholds is not None and fallback_metrics is not None
        fallback_metrics["constraints_satisfied"] = False
        return fallback_thresholds, fallback_metrics, top_rows
    best_metrics["constraints_satisfied"] = True
    return best_thresholds, best_metrics, top_rows


def train_live_selector(
    *,
    input_csv: Path,
    output_dir: Path,
    folds: int,
    num_boost_round: int,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    win_weight: float,
    loss_weight: float,
    tie_weight: float,
    delta_weight: float,
    seed: int,
    device: str,
    n_max: int,
    slots: int,
    delay_bound_ms: float,
) -> dict[str, Any]:
    metadata = pd.read_csv(input_csv).reset_index(drop=True)
    if metadata.empty:
        raise ValueError(f"No live override labels found in {input_csv}")
    x = _feature_matrix(metadata, n_max=int(n_max), slots=int(slots), delay_bound_ms=float(delay_bound_ms))
    feature_names = list(LIVE_OVERRIDE_GATE_FEATURE_NAMES)
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    reward = metadata["future_env_reward_delta_vs_base"].to_numpy(dtype=np.float32)
    win_y = (accepted > 0.0).astype(np.float32)
    loss_y = (accepted < 0.0).astype(np.float32)
    delta_y = accepted.astype(np.float32)
    sample_weight = _sample_weight(accepted, win_weight=win_weight, loss_weight=loss_weight, tie_weight=tie_weight)
    delta_sample_weight = sample_weight * float(delta_weight)

    fold_indices = _episode_folds(metadata, folds=int(folds), seed=int(seed))
    oof_win = np.full((len(metadata),), np.nan, dtype=np.float32)
    oof_loss = np.full((len(metadata),), np.nan, dtype=np.float32)
    oof_delta = np.full((len(metadata),), np.nan, dtype=np.float32)
    fold_summaries: list[dict[str, Any]] = []
    all_indices = np.arange(len(metadata), dtype=np.int64)
    for fold_id, val_idx in enumerate(fold_indices):
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        win_model = _train_xgboost(
            x=x[train_idx],
            y=win_y[train_idx],
            feature_names=feature_names,
            objective="binary:logistic",
            sample_weight=sample_weight[train_idx],
            num_boost_round=int(num_boost_round),
            seed=int(seed) + 1000 + int(fold_id),
            device=str(device),
        )
        loss_model = _train_xgboost(
            x=x[train_idx],
            y=loss_y[train_idx],
            feature_names=feature_names,
            objective="binary:logistic",
            sample_weight=sample_weight[train_idx],
            num_boost_round=int(num_boost_round),
            seed=int(seed) + 2000 + int(fold_id),
            device=str(device),
        )
        delta_model = _train_xgboost(
            x=x[train_idx],
            y=delta_y[train_idx],
            feature_names=feature_names,
            objective="reg:squarederror",
            sample_weight=delta_sample_weight[train_idx],
            num_boost_round=int(num_boost_round),
            seed=int(seed) + 3000 + int(fold_id),
            device=str(device),
        )
        oof_win[val_idx] = _predict_xgboost(win_model, x[val_idx], feature_names)
        oof_loss[val_idx] = _predict_xgboost(loss_model, x[val_idx], feature_names)
        oof_delta[val_idx] = _predict_xgboost(delta_model, x[val_idx], feature_names)
        fold_meta = metadata.iloc[val_idx]
        fold_summaries.append(
            {
                "fold": int(fold_id),
                "train_rows": int(len(train_idx)),
                "val_rows": int(len(val_idx)),
                "val_episodes": sorted(str(value) for value in fold_meta["episode_id"].unique()),
                "val_wins": int((fold_meta["accepted_delta_vs_base"].astype(float) > 0.0).sum()),
                "val_losses": int((fold_meta["accepted_delta_vs_base"].astype(float) < 0.0).sum()),
                "val_total_delta": int(round(float(fold_meta["accepted_delta_vs_base"].astype(float).sum()))),
            }
        )
    if np.isnan(oof_win).any() or np.isnan(oof_loss).any() or np.isnan(oof_delta).any():
        raise RuntimeError("OOF prediction generation left missing rows")

    score_weights = {"win": 1.0, "loss": 1.0, "delta": 1.0}
    thresholds, oof_metrics, leaderboard = _tune_thresholds(
        metadata,
        win_pred=oof_win,
        loss_pred=oof_loss,
        delta_pred=oof_delta,
        max_loss_rate=float(max_loss_rate),
        min_override_count=int(min_override_count),
        min_total_delta=float(min_total_delta),
        max_override_rate=float(max_override_rate),
        score_weights=score_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = metadata.copy()
    predictions["oof_win_score"] = oof_win.astype(np.float32)
    predictions["oof_loss_score"] = oof_loss.astype(np.float32)
    predictions["oof_delta_score"] = oof_delta.astype(np.float32)
    predictions["oof_allow"] = _allowed(
        win_pred=oof_win,
        loss_pred=oof_loss,
        delta_pred=oof_delta,
        thresholds=thresholds,
        score_weights=score_weights,
    )
    predictions.to_csv(output_dir / "live_override_selector_oof_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "live_override_selector_oof_predictions.npz",
        features=x.astype(np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
        accepted_delta=accepted.astype(np.float32),
        reward_delta=reward.astype(np.float32),
        oof_win=oof_win.astype(np.float32),
        oof_loss=oof_loss.astype(np.float32),
        oof_delta=oof_delta.astype(np.float32),
    )

    final_win = _train_xgboost(
        x=x,
        y=win_y,
        feature_names=feature_names,
        objective="binary:logistic",
        sample_weight=sample_weight,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 11,
        device=str(device),
    )
    final_loss = _train_xgboost(
        x=x,
        y=loss_y,
        feature_names=feature_names,
        objective="binary:logistic",
        sample_weight=sample_weight,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 12,
        device=str(device),
    )
    final_delta = _train_xgboost(
        x=x,
        y=delta_y,
        feature_names=feature_names,
        objective="reg:squarederror",
        sample_weight=delta_sample_weight,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 13,
        device=str(device),
    )
    model_paths = {
        "win": "xgboost_live_override_win.json",
        "loss": "xgboost_live_override_loss.json",
        "delta": "xgboost_live_override_delta.json",
    }
    final_win.save_model(str(output_dir / model_paths["win"]))
    final_loss.save_model(str(output_dir / model_paths["loss"]))
    final_delta.save_model(str(output_dir / model_paths["delta"]))
    artifact = {
        "backend": "xgboost",
        "model_paths": model_paths,
        "feature_names": feature_names,
        "thresholds": thresholds,
        "score_weights": score_weights,
        "training": {
            "input_csv": str(input_csv),
            "folds": int(folds),
            "num_boost_round": int(num_boost_round),
            "seed": int(seed),
            "device": str(device),
            "n_max": int(n_max),
            "slots": int(slots),
            "delay_bound_ms": float(delay_bound_ms),
            "win_weight": float(win_weight),
            "loss_weight": float(loss_weight),
            "tie_weight": float(tie_weight),
            "delta_weight": float(delta_weight),
        },
    }
    artifact_path = output_dir / "neural_three_head_live_override_gate.json"
    _write_json(artifact_path, artifact)

    label_counts = {
        "examples": int(len(metadata)),
        "wins": int((accepted > 0.0).sum()),
        "losses": int((accepted < 0.0).sum()),
        "ties": int((accepted == 0.0).sum()),
        "raw_total_accepted_delta": int(round(float(accepted.sum()))),
        "raw_total_reward_delta": float(reward.sum()),
        "raw_win_rate": float((accepted > 0.0).mean()),
        "raw_loss_rate": float((accepted < 0.0).mean()),
    }
    summary = {
        "artifact_path": str(artifact_path),
        "output_dir": str(output_dir),
        "input_csv": str(input_csv),
        "label_counts": label_counts,
        "raw_metrics": _raw_metrics(metadata),
        "oof_thresholds": thresholds,
        "oof_metrics": oof_metrics,
        "oof_leaderboard_top": leaderboard[:20],
        "folds": fold_summaries,
        "feature_names": feature_names,
        "training": artifact["training"],
    }
    _write_json(output_dir / "neural_three_head_live_override_selector_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a live-labeled gate for neural three-head override proposals.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=120)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.20)
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--loss-weight", type=float, default=10.0)
    parser.add_argument("--tie-weight", type=float, default=0.5)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-max", type=int, default=32)
    parser.add_argument("--slots", type=int, default=100)
    parser.add_argument("--delay-bound-ms", type=float, default=50.0)
    args = parser.parse_args()
    summary = train_live_selector(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        folds=int(args.folds),
        num_boost_round=int(args.num_boost_round),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        win_weight=float(args.win_weight),
        loss_weight=float(args.loss_weight),
        tie_weight=float(args.tie_weight),
        delta_weight=float(args.delta_weight),
        seed=int(args.seed),
        device=str(args.device),
        n_max=int(args.n_max),
        slots=int(args.slots),
        delay_bound_ms=float(args.delay_bound_ms),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
