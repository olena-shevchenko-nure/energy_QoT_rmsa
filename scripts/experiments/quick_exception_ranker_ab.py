from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_split(run_dir: Path, split: str) -> dict[str, Any]:
    npz = np.load(run_dir / f"{split}_dagger_tree_ranker_examples.npz", allow_pickle=True)
    metadata = pd.read_csv(run_dir / f"{split}_dagger_tree_ranker_examples.csv").reset_index(drop=True)
    feature_names = [str(value) for value in npz["feature_names"].tolist()]
    features = np.asarray(npz["features"], dtype=np.float32)
    if len(metadata) != int(features.shape[0]):
        raise ValueError(f"{split}: metadata rows ({len(metadata)}) != feature rows ({features.shape[0]})")
    return {"x": features, "metadata": metadata, "feature_names": feature_names}


def _feature_index(feature_names: list[str]) -> dict[str, int]:
    required = [
        "energy_norm",
        "is_j_total",
        "fragmentation_after",
        "largest_free_block_norm",
        "qot_margin",
        "fragmentation_delta",
        "small_gap_delta",
        "largest_free_block_delta_norm",
        "qot_margin_delta",
        "energy_delta",
        "delay_delta_norm",
    ]
    index = {name: int(position) for position, name in enumerate(feature_names)}
    missing = [name for name in required if name not in index]
    if missing:
        raise ValueError(f"Missing required feature names: {missing}")
    return index


def _safe_delta_target(metadata: pd.DataFrame) -> np.ndarray:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    secondary = metadata.get("secondary_delta_vs_base", pd.Series(np.zeros((len(metadata),), dtype=np.float32))).to_numpy(
        dtype=np.float32
    )
    return np.where(accepted != 0.0, accepted, secondary).astype(np.float32)


def _rank_target(metadata: pd.DataFrame) -> np.ndarray:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    secondary = metadata.get("secondary_delta_vs_base", pd.Series(np.zeros((len(metadata),), dtype=np.float32))).to_numpy(
        dtype=np.float32
    )
    target = np.zeros((len(metadata),), dtype=np.float32)
    target = np.where(accepted > 0, 2.0 + np.minimum(accepted, 5.0), target)
    target = np.where((accepted == 0) & (secondary > 0), 1.0, target)
    return target.astype(np.float32)


def _pool_mask_for_group(
    *,
    group: pd.DataFrame,
    features: np.ndarray,
    fidx: dict[str, int],
    top_k: int,
) -> list[int]:
    positions = np.asarray(group.index.to_numpy(), dtype=np.int64)
    candidate_indices = group["candidate_index"].to_numpy(dtype=np.int64)
    selected: set[int] = set()

    base_index = int(group["base_index"].iloc[0])
    selected.add(base_index)

    energy_order = np.argsort(features[positions, fidx["energy_norm"]], kind="mergesort")
    selected.update(int(candidate_indices[position]) for position in energy_order[: max(1, min(int(top_k), len(positions)))])

    is_j_total = features[positions, fidx["is_j_total"]]
    j_positions = np.flatnonzero(is_j_total > 0.5)
    if j_positions.size:
        selected.update(int(candidate_indices[position]) for position in j_positions)
    else:
        selected.add(0)

    selected.add(int(candidate_indices[int(np.argmin(features[positions, fidx["fragmentation_after"]]))]))
    selected.add(int(candidate_indices[int(np.argmax(features[positions, fidx["largest_free_block_norm"]]))]))
    selected.add(int(candidate_indices[int(np.argmax(features[positions, fidx["qot_margin"]]))]))

    keep = group[group["candidate_index"].astype(int).isin(selected)].index.to_numpy(dtype=np.int64)
    return [int(position) for position in keep]


def _filter_small_pool(data: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    metadata = data["metadata"].copy()
    features = data["x"]
    fidx = _feature_index(data["feature_names"])
    keep_positions: list[int] = []
    for _, group in metadata.groupby("group_id", sort=False):
        keep_positions.extend(_pool_mask_for_group(group=group, features=features, fidx=fidx, top_k=top_k))
    keep = np.asarray(sorted(set(keep_positions)), dtype=np.int64)
    pooled_metadata = metadata.iloc[keep].reset_index(drop=True).copy()
    pooled_features = features[keep].astype(np.float32)
    pooled_metadata["is_base"] = (
        pooled_metadata["candidate_index"].astype(int) == pooled_metadata["base_index"].astype(int)
    )
    return {
        "x": pooled_features,
        "metadata": pooled_metadata,
        "feature_names": list(data["feature_names"]),
    }


def _add_runtime_features(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True).copy()
    features = np.asarray(data["x"], dtype=np.float32)
    fidx = _feature_index(data["feature_names"])
    extra = np.zeros((features.shape[0], 4), dtype=np.float32)
    for _, group in metadata.groupby("group_id", sort=False):
        positions = np.asarray(group.index.to_numpy(), dtype=np.int64)
        energy_values = features[positions, fidx["energy_norm"]]
        order = np.argsort(energy_values, kind="mergesort")
        ranks = np.empty((len(positions),), dtype=np.float32)
        ranks[order] = np.arange(len(positions), dtype=np.float32)
        denom = max(float(len(positions) - 1), 1.0)
        base_rows = np.flatnonzero(group["candidate_index"].to_numpy(dtype=int) == int(group["base_index"].iloc[0]))
        base_local = int(base_rows[0]) if base_rows.size else 0
        base_rank = float(ranks[base_local])
        extra[positions, 0] = (group["candidate_index"].to_numpy(dtype=int) == int(group["base_index"].iloc[0])).astype(
            np.float32
        )
        extra[positions, 1] = ranks / denom
        extra[positions, 2] = (ranks - base_rank) / denom
        extra[positions, 3] = len(positions) / 16.0
    names = list(data["feature_names"]) + ["is_base_runtime", "energy_rank_norm", "energy_rank_delta", "pool_size_norm"]
    return {"x": np.concatenate([features, extra], axis=1), "metadata": metadata, "feature_names": names}


def _split_train_threshold(data: dict[str, Any], *, threshold_fraction: float, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    groups = np.asarray(data["metadata"]["group_id"].drop_duplicates().to_numpy(), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    threshold_count = max(1, int(round(len(shuffled) * float(threshold_fraction))))
    threshold_groups = set(int(value) for value in shuffled[:threshold_count])
    mask = data["metadata"]["group_id"].astype(int).isin(threshold_groups).to_numpy()

    def subset(row_mask: np.ndarray) -> dict[str, Any]:
        return {
            "x": data["x"][row_mask].astype(np.float32),
            "metadata": data["metadata"].loc[row_mask].reset_index(drop=True).copy(),
            "feature_names": list(data["feature_names"]),
        }

    return subset(~mask), subset(mask)


def _non_base_dataset(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    mask = (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()
    selected = metadata.loc[mask].reset_index(drop=True).copy()
    win_y = (selected["accepted_delta_vs_base"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
    loss_y = (selected["accepted_delta_vs_base"].to_numpy(dtype=np.float32) < 0).astype(np.float32)
    delta_y = _safe_delta_target(selected)
    return {
        "x": data["x"][mask].astype(np.float32),
        "metadata": selected,
        "win_y": win_y,
        "loss_y": loss_y,
        "delta_y": delta_y,
        "feature_names": list(data["feature_names"]),
    }


def _group_sizes(metadata: pd.DataFrame) -> list[int]:
    return [int(len(group)) for _, group in metadata.groupby("group_id", sort=False)]


def _pos_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y > 0.5))
    negatives = float(max(int(y.size) - int(positives), 0))
    return float(negatives / max(positives, 1.0))


def _xgb_predict(model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    import xgboost as xgb

    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(model.predict(xgb.DMatrix(x, feature_names=feature_names)), dtype=np.float32)


def _lgb_predict(model: Any, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def _train_xgboost_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    objective: str,
    num_boost_round: int,
    seed: int,
    groups: list[int] | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import xgboost as xgb

    dtrain = xgb.DMatrix(x, label=y, weight=sample_weight, feature_names=feature_names)
    if groups is not None:
        dtrain.set_group(groups)
    params: dict[str, Any] = {
        "objective": objective,
        "eta": 0.05,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": int(seed),
        "verbosity": 0,
    }
    if objective == "binary:logistic":
        params["eval_metric"] = "logloss"
        params["scale_pos_weight"] = _pos_weight(y)
    elif objective.startswith("rank:"):
        params["eval_metric"] = "ndcg"
        params["ndcg_exp_gain"] = False
    else:
        params["eval_metric"] = "rmse"
    return xgb.train(params, dtrain, num_boost_round=int(num_boost_round), verbose_eval=False)


def _train_lightgbm_model(
    *,
    x: np.ndarray,
    y: np.ndarray,
    objective: str,
    num_boost_round: int,
    seed: int,
    groups: list[int] | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    import lightgbm as lgb

    params: dict[str, Any] = {
        "objective": objective,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 4,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": int(seed),
        "num_threads": 4,
        "verbosity": -1,
    }
    if objective == "binary":
        params["metric"] = "binary_logloss"
        params["scale_pos_weight"] = _pos_weight(y)
    elif objective == "lambdarank":
        params["metric"] = "ndcg"
        max_label = int(np.max(y)) if y.size else 1
        params["label_gain"] = list(range(max_label + 1))
    else:
        params["metric"] = "rmse"
    train = lgb.Dataset(x, label=y, weight=sample_weight, group=groups, free_raw_data=False)
    return lgb.train(params, train, num_boost_round=int(num_boost_round), callbacks=[lgb.log_evaluation(period=0)])


def _train_backend_heads(
    *,
    backend: str,
    train: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> dict[str, Any]:
    x = train["x"]
    feature_names = train["feature_names"]
    hard_weight = np.where(train["win_y"] > 0.5, 8.0, np.where(train["loss_y"] > 0.5, 6.0, 0.75)).astype(np.float32)
    if backend == "xgboost":
        return {
            "win": _train_xgboost_model(
                x=x,
                y=train["win_y"],
                feature_names=feature_names,
                objective="binary:logistic",
                num_boost_round=num_boost_round,
                seed=seed,
                sample_weight=hard_weight,
            ),
            "loss": _train_xgboost_model(
                x=x,
                y=train["loss_y"],
                feature_names=feature_names,
                objective="binary:logistic",
                num_boost_round=num_boost_round,
                seed=seed + 1,
                sample_weight=hard_weight,
            ),
            "delta": _train_xgboost_model(
                x=x,
                y=train["delta_y"],
                feature_names=feature_names,
                objective="reg:squarederror",
                num_boost_round=num_boost_round,
                seed=seed + 2,
                sample_weight=hard_weight,
            ),
        }
    if backend == "lightgbm":
        return {
            "win": _train_lightgbm_model(
                x=x,
                y=train["win_y"],
                objective="binary",
                num_boost_round=num_boost_round,
                seed=seed,
                sample_weight=hard_weight,
            ),
            "loss": _train_lightgbm_model(
                x=x,
                y=train["loss_y"],
                objective="binary",
                num_boost_round=num_boost_round,
                seed=seed + 1,
                sample_weight=hard_weight,
            ),
            "delta": _train_lightgbm_model(
                x=x,
                y=train["delta_y"],
                objective="regression",
                num_boost_round=num_boost_round,
                seed=seed + 2,
                sample_weight=hard_weight,
            ),
        }
    raise ValueError(f"Unsupported backend: {backend}")


def _train_loss_head(
    *,
    backend: str,
    train: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> Any:
    x = train["x"]
    hard_weight = np.where(train["win_y"] > 0.5, 8.0, np.where(train["loss_y"] > 0.5, 6.0, 0.75)).astype(np.float32)
    if backend == "xgboost":
        return _train_xgboost_model(
            x=x,
            y=train["loss_y"],
            feature_names=train["feature_names"],
            objective="binary:logistic",
            num_boost_round=num_boost_round,
            seed=seed,
            sample_weight=hard_weight,
        )
    if backend == "lightgbm":
        return _train_lightgbm_model(
            x=x,
            y=train["loss_y"],
            objective="binary",
            num_boost_round=num_boost_round,
            seed=seed,
            sample_weight=hard_weight,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _predict_loss_head(backend: str, model: Any, data: dict[str, Any]) -> np.ndarray:
    if backend == "xgboost":
        return _xgb_predict(model, data["x"], data["feature_names"])
    return _lgb_predict(model, data["x"])


def _predict_heads(backend: str, models: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    if backend == "xgboost":
        return {
            name: _xgb_predict(model, data["x"], data["feature_names"])
            for name, model in models.items()
        }
    return {name: _lgb_predict(model, data["x"]) for name, model in models.items()}


def _train_ranker(
    *,
    backend: str,
    train: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> Any:
    y = _rank_target(train["metadata"])
    groups = _group_sizes(train["metadata"])
    if backend == "xgboost":
        return _train_xgboost_model(
            x=train["x"],
            y=y,
            feature_names=train["feature_names"],
            objective="rank:ndcg",
            num_boost_round=num_boost_round,
            seed=seed,
            groups=groups,
        )
    if backend == "lightgbm":
        return _train_lightgbm_model(
            x=train["x"],
            y=y,
            objective="lambdarank",
            num_boost_round=num_boost_round,
            seed=seed,
            groups=groups,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _predict_ranker(backend: str, model: Any, data: dict[str, Any]) -> np.ndarray:
    if backend == "xgboost":
        return _xgb_predict(model, data["x"], data["feature_names"])
    return _lgb_predict(model, data["x"])


def _safety_mask(data: dict[str, Any], *, enabled: bool) -> np.ndarray:
    metadata = data["metadata"]
    mask = (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy().copy()
    if not enabled:
        return mask
    features = data["x"]
    fidx = _feature_index(data["feature_names"])
    # Emergency guard only. Frag/lmax/small-gap stay as learned risk features.
    mask &= features[:, fidx["qot_margin_delta"]] >= -0.25
    mask &= features[:, fidx["energy_delta"]] <= 0.40
    mask &= features[:, fidx["delay_delta_norm"]] <= 0.20
    return mask


def _selection_metrics(selected_rows: list[dict[str, Any]], metadata: pd.DataFrame) -> dict[str, Any]:
    selected = pd.DataFrame(selected_rows)
    groups = int(metadata["group_id"].nunique())
    override = selected[selected["override"]]
    wins = metadata[
        (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int))
        & (metadata["accepted_delta_vs_base"].astype(float) > 0)
    ]
    win_groups = int(wins["group_id"].nunique()) if not wins.empty else 0
    captured_win_groups = int(override[override["accepted_delta_vs_base"] > 0]["group_id"].nunique()) if not override.empty else 0
    selected_loss_rate = None if override.empty else float((override["accepted_delta_vs_base"] < 0).mean())
    selected_win_rate = None if override.empty else float((override["accepted_delta_vs_base"] > 0).mean())
    return {
        "groups": groups,
        "rows": int(len(metadata)),
        "win_groups": win_groups,
        "win_group_rate": float(win_groups / max(groups, 1)),
        "override_count": int(len(override)),
        "override_rate": float(len(override) / max(groups, 1)),
        "selected_win_rate_when_overridden": selected_win_rate,
        "selected_loss_rate_when_overridden": selected_loss_rate,
        "selected_tie_rate_when_overridden": None
        if override.empty
        else float((override["accepted_delta_vs_base"] == 0).mean()),
        "total_selected_accepted_delta_vs_base": int(round(float(selected["accepted_delta_vs_base"].sum()))),
        "mean_selected_accepted_delta_vs_base": float(selected["accepted_delta_vs_base"].mean()) if not selected.empty else 0.0,
        "mean_selected_reward_delta_vs_base": float(selected["reward_delta_vs_base"].mean()) if not selected.empty else 0.0,
        "mean_selected_utility_delta_vs_base": float(selected["utility_delta_vs_base"].mean()) if not selected.empty else 0.0,
        "captured_win_groups": captured_win_groups,
        "captured_win_group_rate": float(captured_win_groups / max(win_groups, 1)),
    }


def _no_override_row(group: pd.DataFrame) -> dict[str, Any]:
    return {
        "group_id": int(group["group_id"].iloc[0]),
        "override": False,
        "row_index": None,
        "candidate_index": None,
        "accepted_delta_vs_base": 0.0,
        "reward_delta_vs_base": 0.0,
        "utility_delta_vs_base": 0.0,
        "win_prob": None,
        "loss_prob": None,
        "delta_pred": None,
        "selector_score": None,
        "ranker_score": None,
    }


def _override_row(
    *,
    row: pd.Series,
    group: pd.DataFrame,
    row_index: int,
    win_prob: float,
    loss_prob: float,
    delta_pred: float,
    selector_score: float,
    ranker_score: float | None = None,
) -> dict[str, Any]:
    return {
        "group_id": int(row["group_id"]),
        "override": True,
        "row_index": int(row_index),
        "candidate_index": int(row["candidate_index"]),
        "accepted_delta_vs_base": float(row["accepted_delta_vs_base"]),
        "reward_delta_vs_base": float(row.get("future_env_reward_delta_vs_base", 0.0)),
        "utility_delta_vs_base": float(row["utility"]) - _group_base_utility(group),
        "win_prob": float(win_prob),
        "loss_prob": float(loss_prob),
        "delta_pred": float(delta_pred),
        "selector_score": float(selector_score),
        "ranker_score": None if ranker_score is None else float(ranker_score),
    }


def _three_head_selected_rows(
    *,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
    apply_loss_threshold: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = _safety_mask(data, enabled=safety_enabled)
    win_prob = np.asarray(preds["win"], dtype=np.float32)
    loss_prob = np.asarray(preds["loss"], dtype=np.float32)
    delta_pred = np.asarray(preds["delta"], dtype=np.float32)
    score = delta_pred + win_prob - 2.0 * loss_prob
    loss_limit = float(thresholds.get("max_loss_prob", thresholds.get("loss_prob_cutoff", 1.0)))
    passed = (
        eligible
        & (win_prob >= float(thresholds["min_win_prob"]))
        & (delta_pred >= float(thresholds["min_delta_pred"]))
    )
    if apply_loss_threshold:
        passed &= loss_prob <= loss_limit
    rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        passed_indices = group_indices[passed[group_indices]]
        if passed_indices.size == 0:
            rows.append(_no_override_row(group))
            continue
        best = int(
            min(
                (int(index) for index in passed_indices),
                key=lambda index: (-float(score[index]), int(metadata.at[index, "candidate_index"])),
            )
        )
        row = metadata.loc[best]
        rows.append(
            _override_row(
                row=row,
                group=group,
                row_index=best,
                win_prob=float(win_prob[best]),
                loss_prob=float(loss_prob[best]),
                delta_pred=float(delta_pred[best]),
                selector_score=float(score[best]),
            )
        )
    diagnostics = {
        "candidate_gate_pass_rows": int(passed.sum()),
        "candidate_win_precision_when_passed": None
        if not passed.any()
        else float((metadata.loc[passed, "accepted_delta_vs_base"] > 0).mean()),
        "candidate_loss_rate_when_passed": None
        if not passed.any()
        else float((metadata.loc[passed, "accepted_delta_vs_base"] < 0).mean()),
    }
    return rows, diagnostics


def _select_three_head(
    *,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows, diagnostics = _three_head_selected_rows(
        data=data,
        preds=preds,
        thresholds=thresholds,
        safety_enabled=safety_enabled,
        apply_loss_threshold=True,
    )
    metrics = _selection_metrics(rows, metadata)
    metrics.update(diagnostics)
    return metrics


def _group_base_utility(group: pd.DataFrame) -> float:
    base_rows = group[group["candidate_index"].astype(int) == int(group["base_index"].iloc[0])]
    if base_rows.empty:
        return float(group["utility"].iloc[0])
    return float(base_rows["utility"].iloc[0])


def _ranker_gate_selected_rows(
    *,
    data: dict[str, Any],
    ranker_scores: np.ndarray,
    gate_preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
    apply_loss_threshold: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible_override = _safety_mask(data, enabled=safety_enabled)
    win_prob = np.asarray(gate_preds["win"], dtype=np.float32)
    loss_prob = np.asarray(gate_preds["loss"], dtype=np.float32)
    delta_pred = np.asarray(gate_preds["delta"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    candidate_gate_checked = 0
    candidate_gate_passed = 0
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        selectable = []
        for index in group_indices:
            is_base = int(metadata.at[int(index), "candidate_index"]) == int(metadata.at[int(index), "base_index"])
            if is_base or bool(eligible_override[int(index)]):
                selectable.append(int(index))
        if not selectable:
            rows.append(_no_override_row(group))
            continue
        best = int(
            min(
                selectable,
                key=lambda index: (-float(ranker_scores[index]), int(metadata.at[index, "candidate_index"])),
            )
        )
        row = metadata.loc[best]
        is_base = int(row["candidate_index"]) == int(row["base_index"])
        if is_base:
            rows.append(_no_override_row(group))
            continue
        candidate_gate_checked += 1
        passed = (
            float(win_prob[best]) >= float(thresholds["min_win_prob"])
            and float(delta_pred[best]) >= float(thresholds["min_delta_pred"])
        )
        if apply_loss_threshold:
            passed = passed and float(loss_prob[best]) <= float(
                thresholds.get("max_loss_prob", thresholds.get("loss_prob_cutoff", 1.0))
            )
        if not passed:
            rows.append(_no_override_row(group))
            continue
        candidate_gate_passed += 1
        rows.append(
            _override_row(
                row=row,
                group=group,
                row_index=best,
                win_prob=float(win_prob[best]),
                loss_prob=float(loss_prob[best]),
                delta_pred=float(delta_pred[best]),
                selector_score=float(ranker_scores[best]),
                ranker_score=float(ranker_scores[best]),
            )
        )
    diagnostics = {
        "ranker_nonbase_checked": int(candidate_gate_checked),
        "candidate_gate_pass_rows": int(candidate_gate_passed),
    }
    return rows, diagnostics


def _select_ranker_gate(
    *,
    data: dict[str, Any],
    ranker_scores: np.ndarray,
    gate_preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows, diagnostics = _ranker_gate_selected_rows(
        data=data,
        ranker_scores=ranker_scores,
        gate_preds=gate_preds,
        thresholds=thresholds,
        safety_enabled=safety_enabled,
        apply_loss_threshold=True,
    )
    metrics = _selection_metrics(rows, metadata)
    metrics.update(diagnostics)
    return metrics


def _threshold_grid() -> list[dict[str, float]]:
    min_win = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
    max_loss = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    min_delta = [-2.0, -1.0, -0.20, -0.05, 0.0, 0.05, 0.10]
    return [
        {"min_win_prob": float(w), "max_loss_prob": float(l), "min_delta_pred": float(d)}
        for w in min_win
        for l in max_loss
        for d in min_delta
    ]


def _veto_threshold_grid() -> list[dict[str, float]]:
    min_win = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
    min_delta = [-2.0, -1.0, -0.20, -0.05, 0.0, 0.05, 0.10]
    return [
        {"min_win_prob": float(w), "min_delta_pred": float(d)}
        for w in min_win
        for d in min_delta
    ]


def _tune_thresholds(
    *,
    selector: Any,
    data: dict[str, Any],
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    best_thresholds: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for thresholds in _threshold_grid():
        metrics = selector(data=data, thresholds=thresholds)
        override_count = int(metrics.get("override_count", 0))
        if override_count < int(min_override_count):
            continue
        loss_rate = metrics.get("selected_loss_rate_when_overridden")
        loss_value = 0.0 if loss_rate is None else float(loss_rate)
        if loss_value > float(max_loss_rate):
            continue
        total_delta = float(metrics.get("total_selected_accepted_delta_vs_base", 0.0))
        if total_delta <= float(min_total_delta):
            continue
        key = (
            total_delta,
            float(metrics.get("mean_selected_reward_delta_vs_base", 0.0)),
            -loss_value,
            float(metrics.get("override_count", 0.0)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = dict(thresholds)
            best_metrics = dict(metrics)
    if best_thresholds is None or best_metrics is None:
        thresholds = {"min_win_prob": 1.000001, "max_loss_prob": -0.000001, "min_delta_pred": 1.0e9}
        return thresholds, selector(data=data, thresholds=thresholds) | {"tune_found_feasible": 0}
    best_thresholds["tune_found_feasible"] = 1.0
    best_metrics["tune_found_feasible"] = 1
    return best_thresholds, best_metrics


def _loss_vetoed_rows(selected_rows: list[dict[str, Any]], *, loss_prob_cutoff: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff = float(loss_prob_cutoff)
    for row in selected_rows:
        current = dict(row)
        if not bool(current.get("override", False)):
            current["vetoed_by_empirical_loss"] = False
            rows.append(current)
            continue
        loss_prob = current.get("loss_prob")
        loss_value = math.inf if loss_prob is None else float(loss_prob)
        if not math.isfinite(loss_value) or loss_value > cutoff:
            current["override"] = False
            current["accepted_delta_vs_base"] = 0.0
            current["reward_delta_vs_base"] = 0.0
            current["utility_delta_vs_base"] = 0.0
            current["vetoed_by_empirical_loss"] = True
        else:
            current["vetoed_by_empirical_loss"] = False
        rows.append(current)
    return rows


def _loss_vetoed_metrics(
    selected_rows: list[dict[str, Any]],
    metadata: pd.DataFrame,
    *,
    loss_prob_cutoff: float,
) -> dict[str, Any]:
    raw_override_count = int(sum(1 for row in selected_rows if bool(row.get("override", False))))
    vetoed_rows = _loss_vetoed_rows(selected_rows, loss_prob_cutoff=loss_prob_cutoff)
    metrics = _selection_metrics(vetoed_rows, metadata)
    metrics["empirical_loss_cutoff"] = float(loss_prob_cutoff)
    metrics["empirical_veto_raw_override_count"] = raw_override_count
    metrics["empirical_veto_vetoed_override_count"] = int(raw_override_count - int(metrics["override_count"]))
    return metrics


def _empirical_veto_prefix_metrics(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    override_rows = [
        row
        for row in selected_rows
        if bool(row.get("override", False))
        and row.get("loss_prob") is not None
        and math.isfinite(float(row["loss_prob"]))
    ]
    if not override_rows:
        return []
    loss_prob = np.asarray([float(row["loss_prob"]) for row in override_rows], dtype=np.float64)
    accepted = np.asarray([float(row["accepted_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    reward = np.asarray([float(row["reward_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    utility = np.asarray([float(row["utility_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    order = np.argsort(loss_prob, kind="mergesort")
    loss_prob = loss_prob[order]
    accepted = accepted[order]
    reward = reward[order]
    utility = utility[order]
    losses = (accepted < 0.0).astype(np.float64)
    wins = (accepted > 0.0).astype(np.float64)
    ties = (accepted == 0.0).astype(np.float64)
    accepted_cum = np.cumsum(accepted)
    reward_cum = np.cumsum(reward)
    utility_cum = np.cumsum(utility)
    loss_cum = np.cumsum(losses)
    win_cum = np.cumsum(wins)
    tie_cum = np.cumsum(ties)
    unique_ends = np.flatnonzero(np.r_[loss_prob[1:] != loss_prob[:-1], True])
    metrics: list[dict[str, Any]] = []
    for end in unique_ends:
        count = int(end) + 1
        metrics.append(
            {
                "loss_prob_cutoff": float(loss_prob[end]),
                "override_count": count,
                "selected_loss_rate_when_overridden": float(loss_cum[end] / max(count, 1)),
                "selected_win_rate_when_overridden": float(win_cum[end] / max(count, 1)),
                "selected_tie_rate_when_overridden": float(tie_cum[end] / max(count, 1)),
                "total_selected_accepted_delta_vs_base": float(accepted_cum[end]),
                "mean_selected_reward_delta_vs_base": float(reward_cum[end] / max(count, 1)),
                "mean_selected_utility_delta_vs_base": float(utility_cum[end] / max(count, 1)),
            }
        )
    return metrics


def _tune_empirical_loss_veto(
    *,
    row_selector: Any,
    data: dict[str, Any],
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    best_thresholds: dict[str, float] | None = None
    best_rows: list[dict[str, Any]] | None = None
    best_diagnostics: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    cutoffs_checked = 0
    for thresholds in _veto_threshold_grid():
        rows, diagnostics = row_selector(data=data, thresholds=thresholds)
        raw_override_count = int(sum(1 for row in rows if bool(row.get("override", False))))
        if raw_override_count < int(min_override_count):
            continue
        for prefix in _empirical_veto_prefix_metrics(rows):
            cutoffs_checked += 1
            override_count = int(prefix["override_count"])
            if override_count < int(min_override_count):
                continue
            loss_rate = float(prefix["selected_loss_rate_when_overridden"])
            if loss_rate > float(max_loss_rate):
                continue
            total_delta = float(prefix["total_selected_accepted_delta_vs_base"])
            if total_delta <= float(min_total_delta):
                continue
            key = (
                total_delta,
                float(prefix["mean_selected_reward_delta_vs_base"]),
                -loss_rate,
                float(override_count),
            )
            if best_key is None or key > best_key:
                best_key = key
                cutoff = float(prefix["loss_prob_cutoff"])
                best_thresholds = dict(thresholds)
                best_thresholds["loss_prob_cutoff"] = cutoff
                best_thresholds["max_loss_prob"] = cutoff
                best_rows = rows
                best_diagnostics = dict(diagnostics)
    if best_thresholds is None or best_rows is None or best_diagnostics is None:
        thresholds = {
            "min_win_prob": 1.000001,
            "min_delta_pred": 1.0e9,
            "loss_prob_cutoff": -0.000001,
            "max_loss_prob": -0.000001,
        }
        empty_rows = [_no_override_row(group) for _, group in metadata.groupby("group_id", sort=False)]
        metrics = _loss_vetoed_metrics(empty_rows, metadata, loss_prob_cutoff=-0.000001)
        metrics["tune_found_feasible"] = 0
        metrics["empirical_loss_cutoffs_checked"] = int(cutoffs_checked)
        return thresholds, metrics
    best_thresholds["tune_found_feasible"] = 1.0
    metrics = _loss_vetoed_metrics(
        best_rows,
        metadata,
        loss_prob_cutoff=float(best_thresholds["loss_prob_cutoff"]),
    )
    metrics.update(best_diagnostics)
    metrics["tune_found_feasible"] = 1
    metrics["empirical_loss_cutoffs_checked"] = int(cutoffs_checked)
    return best_thresholds, metrics


def _select_three_head_empirical_veto(
    *,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows, diagnostics = _three_head_selected_rows(
        data=data,
        preds=preds,
        thresholds=thresholds,
        safety_enabled=safety_enabled,
        apply_loss_threshold=False,
    )
    metrics = _loss_vetoed_metrics(
        rows,
        metadata,
        loss_prob_cutoff=float(thresholds["loss_prob_cutoff"]),
    )
    metrics.update(diagnostics)
    return metrics


def _select_ranker_gate_empirical_veto(
    *,
    data: dict[str, Any],
    ranker_scores: np.ndarray,
    gate_preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
    safety_enabled: bool,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows, diagnostics = _ranker_gate_selected_rows(
        data=data,
        ranker_scores=ranker_scores,
        gate_preds=gate_preds,
        thresholds=thresholds,
        safety_enabled=safety_enabled,
        apply_loss_threshold=False,
    )
    metrics = _loss_vetoed_metrics(
        rows,
        metadata,
        loss_prob_cutoff=float(thresholds["loss_prob_cutoff"]),
    )
    metrics.update(diagnostics)
    return metrics


def _nonbase_mask(data: dict[str, Any]) -> np.ndarray:
    metadata = data["metadata"].reset_index(drop=True)
    return (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()


def _percentile_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        return np.ones_like(np.asarray(values, dtype=np.float64), dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float64)
    positions = np.searchsorted(ref, values_arr, side="right")
    return (positions / float(ref.size)).astype(np.float32)


def _ranker_context_features(data: dict[str, Any], ranker_scores: np.ndarray) -> tuple[np.ndarray, list[str]]:
    metadata = data["metadata"].reset_index(drop=True)
    scores = np.asarray(ranker_scores, dtype=np.float32)
    extra = np.zeros((len(metadata), 2), dtype=np.float32)
    for _, group in metadata.groupby("group_id", sort=False):
        positions = np.asarray(group.index.to_numpy(), dtype=np.int64)
        group_scores = scores[positions]
        candidate_indices = group["candidate_index"].to_numpy(dtype=int)
        base_index = int(group["base_index"].iloc[0])
        base_positions = np.flatnonzero(candidate_indices == base_index)
        base_score = float(group_scores[int(base_positions[0])]) if base_positions.size else float(np.min(group_scores))
        extra[positions, 0] = group_scores - base_score
        if len(positions) == 1:
            extra[positions, 1] = 0.0
            continue
        order = np.argsort(group_scores, kind="mergesort")
        best_local = int(order[-1])
        second_local = int(order[-2])
        best_score = float(group_scores[best_local])
        second_score = float(group_scores[second_local])
        for local_position, row_position in enumerate(positions):
            other_best = second_score if int(local_position) == best_local else best_score
            extra[int(row_position), 1] = float(group_scores[int(local_position)] - other_best)
    return extra, ["risk_ranker_delta_vs_base", "risk_ranker_margin_to_next"]


def _risk_selector_dataset(
    *,
    data: dict[str, Any],
    heads: dict[str, np.ndarray],
    ranker_scores: np.ndarray,
    loss_percentile: np.ndarray,
) -> dict[str, Any]:
    ranker_extra, ranker_extra_names = _ranker_context_features(data, ranker_scores)
    extra = np.column_stack(
        [
            np.asarray(heads["win"], dtype=np.float32),
            np.asarray(heads["loss"], dtype=np.float32),
            np.asarray(loss_percentile, dtype=np.float32),
            np.asarray(heads["delta"], dtype=np.float32),
            np.asarray(ranker_scores, dtype=np.float32),
            ranker_extra,
        ]
    ).astype(np.float32)
    names = list(data["feature_names"]) + [
        "risk_head_win_prob",
        "risk_head_loss_prob",
        "risk_head_loss_percentile",
        "risk_head_delta_pred",
        "risk_ranker_score",
        *ranker_extra_names,
    ]
    return {
        "x": np.concatenate([data["x"].astype(np.float32), extra], axis=1),
        "metadata": data["metadata"].reset_index(drop=True).copy(),
        "feature_names": names,
    }


def _train_risk_selector(
    *,
    backend: str,
    risk_data: dict[str, Any],
    num_boost_round: int,
    seed: int,
) -> Any:
    metadata = risk_data["metadata"].reset_index(drop=True)
    mask = _nonbase_mask(risk_data)
    selected = metadata.loc[mask].reset_index(drop=True).copy()
    y = (selected["accepted_delta_vs_base"].to_numpy(dtype=np.float32) < 0.0).astype(np.float32)
    accepted = selected["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    sample_weight = np.where(y > 0.5, 8.0, np.where(accepted > 0.0, 3.0, 0.75)).astype(np.float32)
    if backend == "xgboost":
        return _train_xgboost_model(
            x=risk_data["x"][mask].astype(np.float32),
            y=y,
            feature_names=risk_data["feature_names"],
            objective="binary:logistic",
            num_boost_round=num_boost_round,
            seed=seed,
            sample_weight=sample_weight,
        )
    if backend == "lightgbm":
        return _train_lightgbm_model(
            x=risk_data["x"][mask].astype(np.float32),
            y=y,
            objective="binary",
            num_boost_round=num_boost_round,
            seed=seed,
            sample_weight=sample_weight,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _predict_risk_selector(backend: str, model: Any, risk_data: dict[str, Any]) -> np.ndarray:
    if backend == "xgboost":
        return _xgb_predict(model, risk_data["x"], risk_data["feature_names"])
    return _lgb_predict(model, risk_data["x"])


def _score_vetoed_rows(
    selected_rows: list[dict[str, Any]],
    *,
    score_cutoff: float,
    score_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cutoff = float(score_cutoff)
    for row in selected_rows:
        current = dict(row)
        if not bool(current.get("override", False)):
            current["vetoed_by_score"] = False
            rows.append(current)
            continue
        score = current.get(score_field)
        score_value = math.inf if score is None else float(score)
        if not math.isfinite(score_value) or score_value > cutoff:
            current["override"] = False
            current["accepted_delta_vs_base"] = 0.0
            current["reward_delta_vs_base"] = 0.0
            current["utility_delta_vs_base"] = 0.0
            current["vetoed_by_score"] = True
        else:
            current["vetoed_by_score"] = False
        rows.append(current)
    return rows


def _attach_score_to_rows(
    selected_rows: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    score_field: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_values = np.asarray(scores, dtype=np.float32)
    for row in selected_rows:
        current = dict(row)
        row_index = current.get("row_index")
        if bool(current.get("override", False)) and row_index is not None:
            current[score_field] = float(score_values[int(row_index)])
        else:
            current[score_field] = None
        rows.append(current)
    return rows


def _score_prefix_metrics(
    selected_rows: list[dict[str, Any]],
    *,
    score_field: str,
) -> list[dict[str, Any]]:
    override_rows = [
        row
        for row in selected_rows
        if bool(row.get("override", False))
        and row.get(score_field) is not None
        and math.isfinite(float(row[score_field]))
    ]
    if not override_rows:
        return []
    risk_score = np.asarray([float(row[score_field]) for row in override_rows], dtype=np.float64)
    accepted = np.asarray([float(row["accepted_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    reward = np.asarray([float(row["reward_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    utility = np.asarray([float(row["utility_delta_vs_base"]) for row in override_rows], dtype=np.float64)
    order = np.argsort(risk_score, kind="mergesort")
    risk_score = risk_score[order]
    accepted = accepted[order]
    reward = reward[order]
    utility = utility[order]
    losses = (accepted < 0.0).astype(np.float64)
    wins = (accepted > 0.0).astype(np.float64)
    ties = (accepted == 0.0).astype(np.float64)
    accepted_cum = np.cumsum(accepted)
    reward_cum = np.cumsum(reward)
    utility_cum = np.cumsum(utility)
    loss_cum = np.cumsum(losses)
    win_cum = np.cumsum(wins)
    tie_cum = np.cumsum(ties)
    unique_ends = np.flatnonzero(np.r_[risk_score[1:] != risk_score[:-1], True])
    metrics: list[dict[str, Any]] = []
    for end in unique_ends:
        count = int(end) + 1
        metrics.append(
            {
                "score_cutoff": float(risk_score[end]),
                "override_count": count,
                "selected_loss_rate_when_overridden": float(loss_cum[end] / max(count, 1)),
                "selected_win_rate_when_overridden": float(win_cum[end] / max(count, 1)),
                "selected_tie_rate_when_overridden": float(tie_cum[end] / max(count, 1)),
                "total_selected_accepted_delta_vs_base": float(accepted_cum[end]),
                "mean_selected_reward_delta_vs_base": float(reward_cum[end] / max(count, 1)),
                "mean_selected_utility_delta_vs_base": float(utility_cum[end] / max(count, 1)),
            }
        )
    return metrics


def _score_vetoed_metrics(
    selected_rows: list[dict[str, Any]],
    metadata: pd.DataFrame,
    *,
    score_cutoff: float,
    score_field: str,
) -> dict[str, Any]:
    raw_override_count = int(sum(1 for row in selected_rows if bool(row.get("override", False))))
    vetoed_rows = _score_vetoed_rows(selected_rows, score_cutoff=score_cutoff, score_field=score_field)
    metrics = _selection_metrics(vetoed_rows, metadata)
    metrics["score_veto_field"] = str(score_field)
    metrics["score_veto_cutoff"] = float(score_cutoff)
    metrics["score_veto_raw_override_count"] = raw_override_count
    metrics["score_veto_vetoed_override_count"] = int(raw_override_count - int(metrics["override_count"]))
    return metrics


def _tune_score_veto(
    *,
    row_selector: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    score_field: str,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    best_thresholds: dict[str, float] | None = None
    best_rows: list[dict[str, Any]] | None = None
    best_diagnostics: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float] | None = None
    cutoffs_checked = 0
    for thresholds in _veto_threshold_grid():
        rows, diagnostics = row_selector(data=data, thresholds=thresholds)
        rows = _attach_score_to_rows(rows, scores, score_field=score_field)
        raw_override_count = int(sum(1 for row in rows if bool(row.get("override", False))))
        if raw_override_count < int(min_override_count):
            continue
        for prefix in _score_prefix_metrics(rows, score_field=score_field):
            cutoffs_checked += 1
            override_count = int(prefix["override_count"])
            if override_count < int(min_override_count):
                continue
            loss_rate = float(prefix["selected_loss_rate_when_overridden"])
            if loss_rate > float(max_loss_rate):
                continue
            total_delta = float(prefix["total_selected_accepted_delta_vs_base"])
            if total_delta <= float(min_total_delta):
                continue
            key = (
                total_delta,
                float(prefix["mean_selected_reward_delta_vs_base"]),
                -loss_rate,
                float(override_count),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = dict(thresholds)
                best_thresholds["score_veto_cutoff"] = float(prefix["score_cutoff"])
                best_rows = rows
                best_diagnostics = dict(diagnostics)
    if best_thresholds is None or best_rows is None or best_diagnostics is None:
        thresholds = {
            "min_win_prob": 1.000001,
            "min_delta_pred": 1.0e9,
            "score_veto_cutoff": -0.000001,
        }
        empty_rows = [_no_override_row(group) for _, group in metadata.groupby("group_id", sort=False)]
        empty_rows = _attach_score_to_rows(empty_rows, scores, score_field=score_field)
        metrics = _score_vetoed_metrics(
            empty_rows,
            metadata,
            score_cutoff=-0.000001,
            score_field=score_field,
        )
        metrics["tune_found_feasible"] = 0
        metrics["score_veto_cutoffs_checked"] = int(cutoffs_checked)
        return thresholds, metrics
    best_thresholds["tune_found_feasible"] = 1.0
    best_thresholds["score_veto_field"] = str(score_field)
    metrics = _score_vetoed_metrics(
        best_rows,
        metadata,
        score_cutoff=float(best_thresholds["score_veto_cutoff"]),
        score_field=score_field,
    )
    metrics.update(best_diagnostics)
    metrics["tune_found_feasible"] = 1
    metrics["score_veto_cutoffs_checked"] = int(cutoffs_checked)
    return best_thresholds, metrics


def _select_with_score_veto(
    *,
    row_selector: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    thresholds: dict[str, float],
    score_field: str,
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows, diagnostics = row_selector(data=data, thresholds=thresholds)
    rows = _attach_score_to_rows(rows, scores, score_field=score_field)
    metrics = _score_vetoed_metrics(
        rows,
        metadata,
        score_cutoff=float(thresholds["score_veto_cutoff"]),
        score_field=score_field,
    )
    metrics.update(diagnostics)
    return metrics


def _coverage_metrics(data: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    metadata = data["metadata"]
    original_metadata = original["metadata"]
    groups = int(original_metadata["group_id"].nunique())
    win_original = original_metadata[
        (original_metadata["candidate_index"].astype(int) != original_metadata["base_index"].astype(int))
        & (original_metadata["accepted_delta_vs_base"].astype(float) > 0)
    ]
    win_groups = set(int(value) for value in win_original["group_id"].unique())
    pooled_win = metadata[
        (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int))
        & (metadata["accepted_delta_vs_base"].astype(float) > 0)
    ]
    pooled_win_groups = set(int(value) for value in pooled_win["group_id"].unique())
    return {
        "groups": groups,
        "original_rows": int(len(original_metadata)),
        "pool_rows": int(len(metadata)),
        "avg_pool_size": float(len(metadata) / max(int(metadata["group_id"].nunique()), 1)),
        "win_groups": int(len(win_groups)),
        "win_group_rate": float(len(win_groups) / max(groups, 1)),
        "coverage_any_win_given_win": float(len(win_groups & pooled_win_groups) / max(len(win_groups), 1)),
        "pool_has_win_all_groups": float(len(pooled_win_groups) / max(groups, 1)),
    }


def _subset_nonbase(data: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    return {
        "x": data["x"][mask].astype(np.float32),
        "metadata": data["metadata"].loc[mask].reset_index(drop=True).copy(),
        "win_y": data["win_y"][mask].astype(np.float32),
        "loss_y": data["loss_y"][mask].astype(np.float32),
        "delta_y": data["delta_y"][mask].astype(np.float32),
        "feature_names": list(data["feature_names"]),
    }


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    positives = float(np.sum(y > 0.5))
    negatives = float(y.size - positives)
    if positives <= 0.0 or negatives <= 0.0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[y > 0.5]))
    return float((pos_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def _binary_average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    positives = float(np.sum(y > 0.5))
    if positives <= 0.0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    precision = np.cumsum(y_sorted > 0.5) / np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    return float(np.sum(precision * (y_sorted > 0.5)) / positives)


def _loss_probability_bins(y: np.ndarray, score: np.ndarray, accepted_delta: np.ndarray) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, 11)
    bins: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        left = float(edges[index])
        right = float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (score >= left) & (score <= right)
        else:
            mask = (score >= left) & (score < right)
        count = int(np.sum(mask))
        if count <= 0:
            bins.append({"left": left, "right": right, "count": 0})
            continue
        values = accepted_delta[mask]
        bins.append(
            {
                "left": left,
                "right": right,
                "count": count,
                "mean_pred_loss": float(np.mean(score[mask])),
                "actual_loss_rate": float(np.mean(y[mask] > 0.5)),
                "actual_win_rate": float(np.mean(values > 0.0)),
                "mean_accepted_delta_vs_base": float(np.mean(values)),
            }
        )
    return bins


def _low_predicted_loss_prefixes(
    *,
    metadata: pd.DataFrame,
    score: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50),
) -> list[dict[str, Any]]:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float64)
    order = np.argsort(np.asarray(score, dtype=np.float64), kind="mergesort")
    prefixes: list[dict[str, Any]] = []
    for fraction in fractions:
        count = max(1, int(round(len(order) * float(fraction))))
        count = min(count, len(order))
        selected = order[:count]
        selected_accepted = accepted[selected]
        prefixes.append(
            {
                "fraction": float(fraction),
                "count": int(count),
                "loss_prob_cutoff": float(np.max(score[selected])),
                "actual_loss_rate": float(np.mean(selected_accepted < 0.0)),
                "actual_win_rate": float(np.mean(selected_accepted > 0.0)),
                "actual_tie_rate": float(np.mean(selected_accepted == 0.0)),
                "total_accepted_delta_vs_base": int(round(float(np.sum(selected_accepted)))),
                "mean_accepted_delta_vs_base": float(np.mean(selected_accepted)),
            }
        )
    return prefixes


def _loss_head_prediction_metrics(
    *,
    metadata: pd.DataFrame,
    loss_prob: np.ndarray,
) -> dict[str, Any]:
    metadata = metadata.reset_index(drop=True)
    y = (metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float64) < 0.0).astype(np.float64)
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float64)
    score = np.clip(np.asarray(loss_prob, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    logloss = -float(np.mean(y * np.log(score) + (1.0 - y) * np.log(1.0 - score))) if y.size else None
    brier = float(np.mean((score - y) ** 2)) if y.size else None
    return {
        "rows": int(len(metadata)),
        "groups": int(metadata["group_id"].nunique()) if "group_id" in metadata else None,
        "loss_count": int(np.sum(y > 0.5)),
        "loss_rate": float(np.mean(y > 0.5)) if y.size else None,
        "win_count": int(np.sum(accepted > 0.0)),
        "win_rate": float(np.mean(accepted > 0.0)) if y.size else None,
        "mean_pred_loss": float(np.mean(score)) if y.size else None,
        "auc_loss": _binary_auc(y, score),
        "average_precision_loss": _binary_average_precision(y, score),
        "logloss": logloss,
        "brier": brier,
        "low_predicted_loss_prefixes": _low_predicted_loss_prefixes(metadata=metadata, score=score) if y.size else [],
        "probability_bins": _loss_probability_bins(y, score, accepted) if y.size else [],
    }


def _oof_loss_head_audit(
    *,
    backend: str,
    train_pool: dict[str, Any],
    eval_pool: dict[str, Any],
    folds: int,
    num_boost_round: int,
    seed: int,
) -> dict[str, Any]:
    train_nonbase = _non_base_dataset(train_pool)
    eval_nonbase = _non_base_dataset(eval_pool)
    metadata = train_nonbase["metadata"].reset_index(drop=True)
    groups = np.asarray(metadata["group_id"].drop_duplicates().to_numpy(), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    folds = max(2, min(int(folds), int(len(shuffled))))
    group_to_fold = {int(group_id): int(position % folds) for position, group_id in enumerate(shuffled)}
    row_folds = metadata["group_id"].astype(int).map(group_to_fold).to_numpy(dtype=np.int64)
    oof_pred = np.full((len(metadata),), np.nan, dtype=np.float32)
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(folds):
        val_mask = row_folds == int(fold)
        train_mask = ~val_mask
        fold_train = _subset_nonbase(train_nonbase, train_mask)
        fold_val = _subset_nonbase(train_nonbase, val_mask)
        model = _train_loss_head(
            backend=backend,
            train=fold_train,
            num_boost_round=num_boost_round,
            seed=seed + 100 + fold,
        )
        pred = _predict_loss_head(backend, model, fold_val)
        oof_pred[val_mask] = pred.astype(np.float32)
        fold_summary = _loss_head_prediction_metrics(metadata=fold_val["metadata"], loss_prob=pred)
        fold_summary["fold"] = int(fold)
        fold_metrics.append(fold_summary)
    if np.any(~np.isfinite(oof_pred)):
        raise RuntimeError("OOF loss-head prediction contains missing folds.")
    final_model = _train_loss_head(
        backend=backend,
        train=train_nonbase,
        num_boost_round=num_boost_round,
        seed=seed + 500,
    )
    eval_pred = _predict_loss_head(backend, final_model, eval_nonbase)
    return {
        "folds": int(folds),
        "train_oof": _loss_head_prediction_metrics(metadata=metadata, loss_prob=oof_pred),
        "eval_final_loss_head": _loss_head_prediction_metrics(metadata=eval_nonbase["metadata"], loss_prob=eval_pred),
        "fold_metrics": fold_metrics,
    }


def run_quick_ab(
    *,
    run_dir: Path,
    backend: str,
    output_path: Path,
    top_k: int,
    num_boost_round: int,
    threshold_fraction: float,
    seed: int,
    max_loss_rates: list[float],
    min_override_count: int,
    min_total_delta: float,
    safety_proxy: bool,
    empirical_loss_veto: bool,
    oof_loss_folds: int,
    oof_loss_only: bool,
    risk_selector_ab: bool,
    risk_ab_only: bool,
) -> dict[str, Any]:
    original_train = _load_split(run_dir, "train")
    original_eval = _load_split(run_dir, "eval")
    train_pool = _add_runtime_features(_filter_small_pool(original_train, top_k=top_k))
    eval_pool = _add_runtime_features(_filter_small_pool(original_eval, top_k=top_k))
    oof_loss_head = None
    if int(oof_loss_folds) > 1:
        oof_loss_head = _oof_loss_head_audit(
            backend=backend,
            train_pool=train_pool,
            eval_pool=eval_pool,
            folds=int(oof_loss_folds),
            num_boost_round=num_boost_round,
            seed=seed,
        )
    if oof_loss_only:
        result = {
            "run_dir": str(run_dir),
            "backend": backend,
            "top_k": int(top_k),
            "num_boost_round": int(num_boost_round),
            "threshold_fraction": float(threshold_fraction),
            "seed": int(seed),
            "safety_proxy": bool(safety_proxy),
            "empirical_loss_veto": bool(empirical_loss_veto),
            "oof_loss_folds": int(oof_loss_folds),
            "oof_loss_only": True,
            "risk_selector_ab": bool(risk_selector_ab),
            "risk_ab_only": bool(risk_ab_only),
            "train_coverage": _coverage_metrics(train_pool, original_train),
            "eval_coverage": _coverage_metrics(eval_pool, original_eval),
            "oof_loss_head": oof_loss_head,
            "budgets": {},
        }
        _write_json(output_path, result)
        print(json.dumps(result, sort_keys=True))
        return result
    train_inner, threshold_val = _split_train_threshold(train_pool, threshold_fraction=threshold_fraction, seed=seed)
    train_nonbase = _non_base_dataset(train_inner)

    heads = _train_backend_heads(backend=backend, train=train_nonbase, num_boost_round=num_boost_round, seed=seed)
    ranker = _train_ranker(backend=backend, train=train_inner, num_boost_round=num_boost_round, seed=seed + 10)

    def nonbase_view(data: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        metadata = data["metadata"].reset_index(drop=True)
        mask = (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()
        subset = {
            "x": data["x"][mask].astype(np.float32),
            "metadata": metadata.loc[mask].reset_index(drop=True).copy(),
            "feature_names": list(data["feature_names"]),
        }
        positions = np.flatnonzero(mask)
        return subset, positions

    def full_head_preds(data: dict[str, Any]) -> dict[str, np.ndarray]:
        subset, positions = nonbase_view(data)
        preds_subset = _predict_heads(backend, heads, subset)
        result = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
        result["loss"].fill(1.0)
        result["delta"].fill(-1.0)
        for name, values in preds_subset.items():
            result[name][positions] = values
        return result

    threshold_heads = full_head_preds(threshold_val)
    eval_heads = full_head_preds(eval_pool)
    train_heads = full_head_preds(train_inner)
    threshold_ranker_scores = _predict_ranker(backend, ranker, threshold_val)
    eval_ranker_scores = _predict_ranker(backend, ranker, eval_pool)
    train_ranker_scores = _predict_ranker(backend, ranker, train_inner)

    risk_ab_data: dict[str, Any] | None = None
    if risk_selector_ab:
        reference_loss_scores = train_heads["loss"][_nonbase_mask(train_inner)]
        train_loss_percentile = _percentile_from_reference(train_heads["loss"], reference_loss_scores)
        threshold_loss_percentile = _percentile_from_reference(threshold_heads["loss"], reference_loss_scores)
        eval_loss_percentile = _percentile_from_reference(eval_heads["loss"], reference_loss_scores)
        train_risk_data = _risk_selector_dataset(
            data=train_inner,
            heads=train_heads,
            ranker_scores=train_ranker_scores,
            loss_percentile=train_loss_percentile,
        )
        threshold_risk_data = _risk_selector_dataset(
            data=threshold_val,
            heads=threshold_heads,
            ranker_scores=threshold_ranker_scores,
            loss_percentile=threshold_loss_percentile,
        )
        eval_risk_data = _risk_selector_dataset(
            data=eval_pool,
            heads=eval_heads,
            ranker_scores=eval_ranker_scores,
            loss_percentile=eval_loss_percentile,
        )
        risk_selector = _train_risk_selector(
            backend=backend,
            risk_data=train_risk_data,
            num_boost_round=num_boost_round,
            seed=seed + 20,
        )
        risk_ab_data = {
            "loss_percentile": {
                "train": train_loss_percentile,
                "threshold": threshold_loss_percentile,
                "eval": eval_loss_percentile,
            },
            "learned_risk": {
                "train": _predict_risk_selector(backend, risk_selector, train_risk_data),
                "threshold": _predict_risk_selector(backend, risk_selector, threshold_risk_data),
                "eval": _predict_risk_selector(backend, risk_selector, eval_risk_data),
            },
        }

    by_budget: dict[str, Any] = {}
    for max_loss_rate in max_loss_rates:
        budget_key = f"loss_{max_loss_rate:g}"

        if risk_selector_ab and risk_ab_only:
            assert risk_ab_data is not None

            def three_row_selector(
                *,
                data: dict[str, Any],
                thresholds: dict[str, float],
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
                return _three_head_selected_rows(
                    data=data,
                    preds=preds,
                    thresholds=thresholds,
                    safety_enabled=safety_proxy,
                    apply_loss_threshold=False,
                )

            def ranker_row_selector(
                *,
                data: dict[str, Any],
                thresholds: dict[str, float],
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                if data is threshold_val:
                    scores = threshold_ranker_scores
                    preds = threshold_heads
                elif data is eval_pool:
                    scores = eval_ranker_scores
                    preds = eval_heads
                else:
                    scores = train_ranker_scores
                    preds = train_heads
                return _ranker_gate_selected_rows(
                    data=data,
                    ranker_scores=scores,
                    gate_preds=preds,
                    thresholds=thresholds,
                    safety_enabled=safety_proxy,
                    apply_loss_threshold=False,
                )

            def score_array(score_kind: str, data: dict[str, Any]) -> np.ndarray:
                split = "threshold" if data is threshold_val else "eval" if data is eval_pool else "train"
                return np.asarray(risk_ab_data[score_kind][split], dtype=np.float32)

            variants: dict[str, Any] = {}
            for score_kind, score_field, suffix in [
                ("loss_percentile", "loss_head_percentile_score", "loss_percentile_veto"),
                ("learned_risk", "learned_risk_selector_score", "learned_risk_selector"),
            ]:
                for proposer_name, row_selector in [
                    ("three_head", three_row_selector),
                    ("ranker_plus_gate", ranker_row_selector),
                ]:
                    thresholds, threshold_metrics = _tune_score_veto(
                        row_selector=row_selector,
                        data=threshold_val,
                        scores=score_array(score_kind, threshold_val),
                        score_field=score_field,
                        max_loss_rate=max_loss_rate,
                        min_override_count=min_override_count,
                        min_total_delta=min_total_delta,
                    )
                    train_metrics = _select_with_score_veto(
                        row_selector=row_selector,
                        data=train_inner,
                        scores=score_array(score_kind, train_inner),
                        thresholds=thresholds,
                        score_field=score_field,
                    )
                    eval_metrics = _select_with_score_veto(
                        row_selector=row_selector,
                        data=eval_pool,
                        scores=score_array(score_kind, eval_pool),
                        thresholds=thresholds,
                        score_field=score_field,
                    )
                    variants[f"{proposer_name}_{suffix}"] = {
                        "thresholds": thresholds,
                        "threshold_val": threshold_metrics,
                        "train_inner": train_metrics,
                        "eval": eval_metrics,
                    }
            by_budget[budget_key] = variants
            continue

        if empirical_loss_veto:

            def three_row_selector(
                *,
                data: dict[str, Any],
                thresholds: dict[str, float],
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
                return _three_head_selected_rows(
                    data=data,
                    preds=preds,
                    thresholds=thresholds,
                    safety_enabled=safety_proxy,
                    apply_loss_threshold=False,
                )

            three_thresholds, three_val = _tune_empirical_loss_veto(
                row_selector=three_row_selector,
                data=threshold_val,
                max_loss_rate=max_loss_rate,
                min_override_count=min_override_count,
                min_total_delta=min_total_delta,
            )
            three_train = _select_three_head_empirical_veto(
                data=train_inner,
                preds=train_heads,
                thresholds=three_thresholds,
                safety_enabled=safety_proxy,
            )
            three_eval = _select_three_head_empirical_veto(
                data=eval_pool,
                preds=eval_heads,
                thresholds=three_thresholds,
                safety_enabled=safety_proxy,
            )

            def ranker_row_selector(
                *,
                data: dict[str, Any],
                thresholds: dict[str, float],
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                if data is threshold_val:
                    scores = threshold_ranker_scores
                    preds = threshold_heads
                elif data is eval_pool:
                    scores = eval_ranker_scores
                    preds = eval_heads
                else:
                    scores = train_ranker_scores
                    preds = train_heads
                return _ranker_gate_selected_rows(
                    data=data,
                    ranker_scores=scores,
                    gate_preds=preds,
                    thresholds=thresholds,
                    safety_enabled=safety_proxy,
                    apply_loss_threshold=False,
                )

            ranker_thresholds, ranker_val = _tune_empirical_loss_veto(
                row_selector=ranker_row_selector,
                data=threshold_val,
                max_loss_rate=max_loss_rate,
                min_override_count=min_override_count,
                min_total_delta=min_total_delta,
            )
            ranker_train = _select_ranker_gate_empirical_veto(
                data=train_inner,
                ranker_scores=train_ranker_scores,
                gate_preds=train_heads,
                thresholds=ranker_thresholds,
                safety_enabled=safety_proxy,
            )
            ranker_eval = _select_ranker_gate_empirical_veto(
                data=eval_pool,
                ranker_scores=eval_ranker_scores,
                gate_preds=eval_heads,
                thresholds=ranker_thresholds,
                safety_enabled=safety_proxy,
            )

            by_budget[budget_key] = {
                "three_head_exception_empirical_veto": {
                    "thresholds": three_thresholds,
                    "threshold_val": three_val,
                    "train_inner": three_train,
                    "eval": three_eval,
                },
                "ranker_plus_gate_empirical_veto": {
                    "thresholds": ranker_thresholds,
                    "threshold_val": ranker_val,
                    "train_inner": ranker_train,
                    "eval": ranker_eval,
                },
            }
            continue

        def three_selector(*, data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
            preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
            return _select_three_head(data=data, preds=preds, thresholds=thresholds, safety_enabled=safety_proxy)

        three_thresholds, three_val = _tune_thresholds(
            selector=three_selector,
            data=threshold_val,
            max_loss_rate=max_loss_rate,
            min_override_count=min_override_count,
            min_total_delta=min_total_delta,
        )
        three_train = _select_three_head(
            data=train_inner,
            preds=train_heads,
            thresholds=three_thresholds,
            safety_enabled=safety_proxy,
        )
        three_eval = _select_three_head(
            data=eval_pool,
            preds=eval_heads,
            thresholds=three_thresholds,
            safety_enabled=safety_proxy,
        )

        def ranker_selector(*, data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
            if data is threshold_val:
                scores = threshold_ranker_scores
                preds = threshold_heads
            elif data is eval_pool:
                scores = eval_ranker_scores
                preds = eval_heads
            else:
                scores = train_ranker_scores
                preds = train_heads
            return _select_ranker_gate(
                data=data,
                ranker_scores=scores,
                gate_preds=preds,
                thresholds=thresholds,
                safety_enabled=safety_proxy,
            )

        ranker_thresholds, ranker_val = _tune_thresholds(
            selector=ranker_selector,
            data=threshold_val,
            max_loss_rate=max_loss_rate,
            min_override_count=min_override_count,
            min_total_delta=min_total_delta,
        )
        ranker_train = _select_ranker_gate(
            data=train_inner,
            ranker_scores=train_ranker_scores,
            gate_preds=train_heads,
            thresholds=ranker_thresholds,
            safety_enabled=safety_proxy,
        )
        ranker_eval = _select_ranker_gate(
            data=eval_pool,
            ranker_scores=eval_ranker_scores,
            gate_preds=eval_heads,
            thresholds=ranker_thresholds,
            safety_enabled=safety_proxy,
        )

        by_budget[budget_key] = {
            "three_head_exception": {
                "thresholds": three_thresholds,
                "threshold_val": three_val,
                "train_inner": three_train,
                "eval": three_eval,
            },
            "ranker_plus_gate": {
                "thresholds": ranker_thresholds,
                "threshold_val": ranker_val,
                "train_inner": ranker_train,
                "eval": ranker_eval,
            },
        }

    result = {
        "run_dir": str(run_dir),
        "backend": backend,
        "top_k": int(top_k),
        "num_boost_round": int(num_boost_round),
        "threshold_fraction": float(threshold_fraction),
        "seed": int(seed),
        "safety_proxy": bool(safety_proxy),
        "empirical_loss_veto": bool(empirical_loss_veto),
        "oof_loss_folds": int(oof_loss_folds),
        "oof_loss_only": False,
        "risk_selector_ab": bool(risk_selector_ab),
        "risk_ab_only": bool(risk_ab_only),
        "oof_loss_head": oof_loss_head,
        "train_coverage": _coverage_metrics(train_pool, original_train),
        "eval_coverage": _coverage_metrics(eval_pool, original_eval),
        "train_inner_groups": int(train_inner["metadata"]["group_id"].nunique()),
        "threshold_val_groups": int(threshold_val["metadata"]["group_id"].nunique()),
        "budgets": by_budget,
    }
    _write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick offline A/B for energy-aware base exception rankers.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend", choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=120)
    parser.add_argument("--threshold-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--max-loss-rates", default="0.005,0.01,0.02,0.05")
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--safety-proxy", action="store_true")
    parser.add_argument("--empirical-loss-veto", action="store_true")
    parser.add_argument("--oof-loss-folds", type=int, default=0)
    parser.add_argument("--oof-loss-only", action="store_true")
    parser.add_argument("--risk-selector-ab", action="store_true")
    parser.add_argument("--risk-ab-only", action="store_true")
    args = parser.parse_args()
    max_loss_rates = [float(item.strip()) for item in str(args.max_loss_rates).split(",") if item.strip()]
    run_quick_ab(
        run_dir=Path(args.run_dir),
        backend=str(args.backend),
        output_path=Path(args.output),
        top_k=int(args.top_k),
        num_boost_round=int(args.num_boost_round),
        threshold_fraction=float(args.threshold_fraction),
        seed=int(args.seed),
        max_loss_rates=max_loss_rates,
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        safety_proxy=bool(args.safety_proxy),
        empirical_loss_veto=bool(args.empirical_loss_veto),
        oof_loss_folds=int(args.oof_loss_folds),
        oof_loss_only=bool(args.oof_loss_only),
        risk_selector_ab=bool(args.risk_selector_ab),
        risk_ab_only=bool(args.risk_ab_only),
    )


if __name__ == "__main__":
    main()
