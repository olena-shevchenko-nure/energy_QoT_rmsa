from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES = [
    "risk_dqn_score",
    "risk_dqn_delta_vs_base",
    "risk_dqn_margin_to_next",
    "risk_dqn_margin_over_threshold",
]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _save_model(backend: str, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "xgboost":
        model.save_model(str(path))
        return
    if backend == "lightgbm":
        model.save_model(str(path))
        return
    raise ValueError(f"Unsupported backend: {backend}")


def _model_suffix(backend: str) -> str:
    if backend == "xgboost":
        return "json"
    if backend == "lightgbm":
        return "txt"
    raise ValueError(f"Unsupported backend: {backend}")


def _resolve_model_path(meta_path: Path, value: str | None) -> Path:
    if not value:
        raise ValueError(f"{meta_path} does not define model_path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return meta_path.parent / path


def _load_split_pool(quick_module: Any, run_dir: Path, split: str, top_k: int) -> dict[str, Any]:
    return quick_module._add_runtime_features(
        quick_module._filter_small_pool(quick_module._load_split(run_dir, split), top_k=int(top_k))
    )


def _subset_by_groups(data: dict[str, Any], group_ids: set[int]) -> tuple[dict[str, Any], np.ndarray]:
    metadata = data["metadata"].reset_index(drop=True)
    mask = metadata["group_id"].astype(int).isin(group_ids).to_numpy()
    positions = np.flatnonzero(mask).astype(np.int64)
    subset = {
        "x": data["x"][positions].astype(np.float32),
        "metadata": metadata.loc[positions].reset_index(drop=True).copy(),
        "feature_names": list(data["feature_names"]),
    }
    return subset, positions


def _make_group_folds(metadata: pd.DataFrame, folds: int, seed: int) -> list[set[int]]:
    rng = np.random.default_rng(int(seed))
    group_labels: dict[str, list[int]] = {"win_loss": [], "win": [], "loss": [], "tie": []}
    for group_id, group in metadata.groupby("group_id", sort=False):
        nonbase = group[group["candidate_index"].astype(int) != group["base_index"].astype(int)]
        has_win = bool((nonbase["accepted_delta_vs_base"].astype(float) > 0.0).any())
        has_loss = bool((nonbase["accepted_delta_vs_base"].astype(float) < 0.0).any())
        if has_win and has_loss:
            bucket = "win_loss"
        elif has_win:
            bucket = "win"
        elif has_loss:
            bucket = "loss"
        else:
            bucket = "tie"
        group_labels[bucket].append(int(group_id))

    result: list[list[int]] = [[] for _ in range(int(folds))]
    for values in group_labels.values():
        shuffled = list(values)
        rng.shuffle(shuffled)
        for index, group_id in enumerate(shuffled):
            result[index % int(folds)].append(int(group_id))
    return [set(values) for values in result]


def _train_fold_dqn(
    *,
    torch: Any,
    dqn_module: Any,
    train_module: Any,
    data: dict[str, Any],
    secondary_scale: float,
    target_clip: float,
    hidden_dim: int,
    depth: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    pair_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    pair_loss_weight: float,
    rank_loss_weight: float,
    max_pairs_per_group: int,
    base_weight: float,
    win_weight: float,
    loss_weight: float,
    tie_weight: float,
    secondary_weight: float,
    seed: int,
    device: str,
) -> Any:
    metadata = data["metadata"].reset_index(drop=True)
    target = dqn_module._target_from_metadata(
        metadata,
        secondary_scale=float(secondary_scale),
        target_clip=float(target_clip),
    )
    row_weight = dqn_module._sample_weights(
        metadata,
        target,
        base_weight=float(base_weight),
        win_weight=float(win_weight),
        loss_weight=float(loss_weight),
        tie_weight=float(tie_weight),
        secondary_weight=float(secondary_weight),
    )
    pair_left, pair_right, pair_target, pair_weight = dqn_module._pair_indices(
        metadata,
        target,
        max_pairs_per_group=int(max_pairs_per_group),
        pair_target_clip=float(target_clip),
    )
    return dqn_module._train_model(
        torch=torch,
        train_module=train_module,
        x=data["x"].astype(np.float32),
        target=target,
        row_weight=row_weight,
        pair_left=pair_left,
        pair_right=pair_right,
        pair_target=pair_target,
        pair_weight=pair_weight,
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
        epochs=int(epochs),
        batch_size=int(batch_size),
        pair_batch_size=int(pair_batch_size),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        pair_loss_weight=float(pair_loss_weight),
        rank_loss_weight=float(rank_loss_weight),
        seed=int(seed),
        device=str(device),
    )


def _oof_dqn_scores(
    *,
    torch: Any,
    dqn_module: Any,
    train_module: Any,
    train_pool: dict[str, Any],
    folds: int,
    seed: int,
    device: str,
    dqn_params: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    metadata = train_pool["metadata"].reset_index(drop=True)
    oof_scores = np.full((len(metadata),), np.nan, dtype=np.float32)
    fold_groups = _make_group_folds(metadata, folds=int(folds), seed=int(seed))
    all_groups = set(int(value) for value in metadata["group_id"].unique())
    fold_summaries: list[dict[str, Any]] = []
    for fold_index, val_groups in enumerate(fold_groups):
        train_groups = all_groups - set(val_groups)
        fold_train, _ = _subset_by_groups(train_pool, train_groups)
        fold_val, positions = _subset_by_groups(train_pool, set(val_groups))
        model = _train_fold_dqn(
            torch=torch,
            dqn_module=dqn_module,
            train_module=train_module,
            data=fold_train,
            seed=int(seed) + 1000 + int(fold_index),
            device=str(device),
            **dqn_params,
        )
        oof_scores[positions] = dqn_module._predict(torch, model, fold_val["x"])
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "train_groups": int(fold_train["metadata"]["group_id"].nunique()),
                "train_rows": int(len(fold_train["metadata"])),
                "val_groups": int(fold_val["metadata"]["group_id"].nunique()),
                "val_rows": int(len(fold_val["metadata"])),
            }
        )
    if np.isnan(oof_scores).any():
        missing = int(np.isnan(oof_scores).sum())
        raise RuntimeError(f"OOF scoring left {missing} train rows without prediction")
    return oof_scores.astype(np.float32), fold_summaries


def _load_torch_artifact_model(torch: Any, artifact_path: Path) -> tuple[dict[str, Any], Any, Path]:
    meta = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_path = _resolve_model_path(artifact_path, str(meta.get("model_path", "")))
    torch.set_num_threads(1)
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    return meta, model, model_path


def _risk_feature_matrix(
    *,
    data: dict[str, Any],
    scores: np.ndarray,
    selection_margin: float,
) -> tuple[np.ndarray, list[str]]:
    metadata = data["metadata"].reset_index(drop=True)
    score_values = np.asarray(scores, dtype=np.float32)
    extra = np.zeros((len(metadata), len(DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES)), dtype=np.float32)
    for _, group in metadata.groupby("group_id", sort=False):
        positions = np.asarray(group.index.to_numpy(), dtype=np.int64)
        group_scores = score_values[positions]
        candidate_indices = group["candidate_index"].astype(int).to_numpy()
        base_index = int(group["base_index"].iloc[0])
        base_positions = np.flatnonzero(candidate_indices == base_index)
        base_score = float(group_scores[int(base_positions[0])]) if base_positions.size else float(np.min(group_scores))
        if len(positions) == 1:
            best_local = 0
            second_score = float(group_scores[0])
        else:
            order = np.argsort(group_scores, kind="mergesort")
            best_local = int(order[-1])
            second_score = float(group_scores[int(order[-2])])
        best_score = float(group_scores[int(best_local)])
        for local_position, row_position in enumerate(positions):
            score = float(group_scores[int(local_position)])
            other_best = second_score if int(local_position) == best_local else best_score
            margin = float(score - base_score)
            extra[int(row_position)] = np.asarray(
                [
                    score,
                    margin,
                    float(score - other_best),
                    float(margin - selection_margin),
                ],
                dtype=np.float32,
            )
    return (
        np.concatenate([data["x"].astype(np.float32), extra], axis=1).astype(np.float32),
        list(data["feature_names"]) + list(DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES),
    )


def _candidate_proposal_indices(
    quick_module: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    *,
    selection_margin: float,
    margin_slack: float,
) -> np.ndarray:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = quick_module._safety_mask(data, enabled=True)
    selected: list[int] = []
    margin_floor = -math.inf if not math.isfinite(float(margin_slack)) else float(selection_margin) - float(margin_slack)
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        if base_rows.empty:
            continue
        base_score = float(scores[int(base_rows.index[0])])
        selectable = [int(index) for index in group_indices if bool(eligible[int(index)])]
        if not selectable:
            continue
        best = int(min(selectable, key=lambda index: (-float(scores[index]), int(metadata.at[index, "candidate_index"]))))
        if float(scores[best] - base_score) >= margin_floor:
            selected.append(best)
    return np.asarray(selected, dtype=np.int64)


def _selected_rows_with_risk(
    quick_module: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    risk_scores: np.ndarray,
    *,
    selection_margin: float,
    score_cutoff: float,
) -> list[dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = quick_module._safety_mask(data, enabled=True)
    rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        if base_rows.empty:
            rows.append(quick_module._no_override_row(group))
            continue
        base_row_index = int(base_rows.index[0])
        base_score = float(scores[base_row_index])
        selectable = [int(index) for index in group_indices if bool(eligible[int(index)])]
        if not selectable:
            rows.append(quick_module._no_override_row(group))
            continue
        best = int(min(selectable, key=lambda index: (-float(scores[index]), int(metadata.at[index, "candidate_index"]))))
        score_margin = float(scores[best] - base_score)
        if score_margin < float(selection_margin):
            rows.append(quick_module._no_override_row(group))
            continue
        risk_score = float(risk_scores[best])
        if not math.isfinite(risk_score) or risk_score > float(score_cutoff):
            row = quick_module._no_override_row(group)
            row["vetoed_by_risk_selector"] = True
            row["risk_selector_score"] = risk_score
            rows.append(row)
            continue
        row = metadata.loc[best]
        selected = quick_module._override_row(
            row=row,
            group=group,
            row_index=best,
            win_prob=0.0,
            loss_prob=risk_score,
            delta_pred=score_margin,
            selector_score=score_margin,
            ranker_score=float(scores[best]),
        )
        selected["risk_selector_score"] = risk_score
        selected["vetoed_by_risk_selector"] = False
        rows.append(selected)
    return rows


def _metrics_with_risk(
    quick_module: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    risk_scores: np.ndarray,
    *,
    selection_margin: float,
    score_cutoff: float,
) -> dict[str, Any]:
    raw_rows = _selected_rows_with_risk(
        quick_module,
        data,
        scores,
        np.zeros((len(scores),), dtype=np.float32),
        selection_margin=selection_margin,
        score_cutoff=math.inf,
    )
    rows = _selected_rows_with_risk(
        quick_module,
        data,
        scores,
        risk_scores,
        selection_margin=selection_margin,
        score_cutoff=score_cutoff,
    )
    metadata = data["metadata"].reset_index(drop=True)
    metrics = quick_module._selection_metrics(rows, metadata)
    raw_override_count = int(sum(1 for row in raw_rows if bool(row.get("override", False))))
    metrics["risk_selector_score_cutoff"] = float(score_cutoff)
    metrics["risk_selector_raw_override_count"] = raw_override_count
    metrics["risk_selector_vetoed_override_count"] = int(raw_override_count - int(metrics["override_count"]))
    return metrics


def _tune_risk_cutoff(
    quick_module: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    risk_scores: np.ndarray,
    *,
    selection_margin: float,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
) -> tuple[float, dict[str, Any]]:
    proposed = _candidate_proposal_indices(
        quick_module,
        data,
        scores,
        selection_margin=selection_margin,
        margin_slack=0.0,
    )
    finite_scores = np.asarray([float(risk_scores[int(index)]) for index in proposed if math.isfinite(float(risk_scores[int(index)]))])
    if finite_scores.size == 0:
        metrics = _metrics_with_risk(
            quick_module,
            data,
            scores,
            risk_scores,
            selection_margin=selection_margin,
            score_cutoff=-math.inf,
        )
        metrics["risk_selector_constraints_satisfied"] = False
        metrics["risk_selector_cutoffs_checked"] = 0
        return -math.inf, metrics
    cutoffs = sorted(set(float(value) for value in finite_scores.tolist()))
    cutoffs = [float(np.min(finite_scores) - 1.0e-6)] + cutoffs + [float(np.max(finite_scores) + 1.0e-6)]
    best_cutoff: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_cutoff: float | None = None
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    for cutoff in cutoffs:
        metrics = _metrics_with_risk(
            quick_module,
            data,
            scores,
            risk_scores,
            selection_margin=selection_margin,
            score_cutoff=float(cutoff),
        )
        loss_rate = metrics.get("selected_loss_rate_when_overridden")
        loss_value = float(loss_rate if loss_rate is not None else 0.0)
        total_delta = float(metrics.get("total_selected_accepted_delta_vs_base", 0.0))
        override_count = int(metrics.get("override_count", 0))
        override_rate = float(metrics.get("override_rate", 0.0))
        key = (
            total_delta,
            float(metrics.get("mean_selected_reward_delta_vs_base", 0.0)),
            -loss_value,
            float(override_count),
            -float(cutoff),
        )
        if fallback_key is None or key > fallback_key:
            fallback_key = key
            fallback_cutoff = float(cutoff)
            fallback_metrics = dict(metrics)
        if override_count < int(min_override_count):
            continue
        if override_rate > float(max_override_rate):
            continue
        if loss_rate is not None and float(loss_rate) > float(max_loss_rate):
            continue
        if total_delta < float(min_total_delta):
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_cutoff = float(cutoff)
            best_metrics = dict(metrics)
    if best_cutoff is None or best_metrics is None:
        assert fallback_cutoff is not None and fallback_metrics is not None
        fallback_metrics["risk_selector_constraints_satisfied"] = False
        fallback_metrics["risk_selector_cutoffs_checked"] = int(len(cutoffs))
        return float(fallback_cutoff), fallback_metrics
    best_metrics["risk_selector_constraints_satisfied"] = True
    best_metrics["risk_selector_cutoffs_checked"] = int(len(cutoffs))
    return float(best_cutoff), best_metrics


def _train_risk_model(
    quick_module: Any,
    *,
    backend: str,
    x: np.ndarray,
    y: np.ndarray,
    accepted_delta: np.ndarray,
    feature_names: list[str],
    num_boost_round: int,
    seed: int,
    loss_sample_weight: float,
    win_sample_weight: float,
    tie_sample_weight: float,
) -> Any:
    sample_weight = np.where(
        y > 0.5,
        float(loss_sample_weight),
        np.where(accepted_delta > 0.0, float(win_sample_weight), float(tie_sample_weight)),
    ).astype(np.float32)
    if backend == "xgboost":
        return quick_module._train_xgboost_model(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            feature_names=feature_names,
            objective="binary:logistic",
            num_boost_round=int(num_boost_round),
            seed=int(seed),
            sample_weight=sample_weight,
        )
    if backend == "lightgbm":
        return quick_module._train_lightgbm_model(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            objective="binary",
            num_boost_round=int(num_boost_round),
            seed=int(seed),
            sample_weight=sample_weight,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _predict_risk_model(quick_module: Any, backend: str, model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if backend == "xgboost":
        return quick_module._xgb_predict(model, x.astype(np.float32), feature_names)
    return quick_module._lgb_predict(model, x.astype(np.float32))


def train_and_export(
    *,
    run_dir: Path,
    dqn_artifact: Path,
    output_dir: Path,
    backend: str,
    top_k: int | None,
    oof_folds: int,
    oof_epochs: int,
    threshold_fraction: float,
    train_margin_slack: float,
    min_risk_examples: int,
    min_loss_examples: int,
    num_boost_round: int,
    max_loss_rate: float,
    hard_eval_max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    secondary_scale: float,
    target_clip: float,
    hidden_dim: int,
    depth: int,
    dropout: float,
    batch_size: int,
    pair_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    pair_loss_weight: float,
    rank_loss_weight: float,
    max_pairs_per_group: int,
    base_weight: float,
    win_weight: float,
    loss_weight: float,
    tie_weight: float,
    secondary_weight: float,
    loss_sample_weight: float,
    win_sample_weight: float,
    tie_sample_weight: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    script_dir = Path(__file__).resolve().parent
    quick_module = _load_module("quick_exception_ranker_ab", script_dir / "quick_exception_ranker_ab.py")
    dqn_module = _load_module("train_dqn_base_relative_ranker", script_dir / "train_dqn_base_relative_ranker.py")
    train_module = _load_module("train_distilled_dqn_ranker", script_dir / "train_distilled_dqn_ranker.py")

    dqn_meta, dqn_model, dqn_model_path = _load_torch_artifact_model(torch, dqn_artifact)
    resolved_top_k = int(top_k if top_k is not None else dqn_meta.get("candidate_pool_top_k", 8))
    selection_margin = float(dqn_meta.get("selection_margin", 0.0))

    train_pool = _load_split_pool(quick_module, run_dir, "train", resolved_top_k)
    eval_pool = _load_split_pool(quick_module, run_dir, "eval", resolved_top_k)
    calibration_npz = run_dir / "calibration_dagger_tree_ranker_examples.npz"
    calibration_csv = run_dir / "calibration_dagger_tree_ranker_examples.csv"
    if calibration_npz.exists() and calibration_csv.exists():
        threshold_val = _load_split_pool(quick_module, run_dir, "calibration", resolved_top_k)
        calibration_source = "calibration_split"
    else:
        train_pool, threshold_val = quick_module._split_train_threshold(
            train_pool,
            threshold_fraction=float(threshold_fraction),
            seed=int(seed),
        )
        calibration_source = "train_threshold_fraction"

    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    dqn_params = {
        "secondary_scale": float(secondary_scale),
        "target_clip": float(target_clip),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "dropout": float(dropout),
        "epochs": int(oof_epochs),
        "batch_size": int(batch_size),
        "pair_batch_size": int(pair_batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "pair_loss_weight": float(pair_loss_weight),
        "rank_loss_weight": float(rank_loss_weight),
        "max_pairs_per_group": int(max_pairs_per_group),
        "base_weight": float(base_weight),
        "win_weight": float(win_weight),
        "loss_weight": float(loss_weight),
        "tie_weight": float(tie_weight),
        "secondary_weight": float(secondary_weight),
    }
    oof_scores, fold_summaries = _oof_dqn_scores(
        torch=torch,
        dqn_module=dqn_module,
        train_module=train_module,
        train_pool=train_pool,
        folds=int(oof_folds),
        seed=int(seed),
        device=requested_device,
        dqn_params=dqn_params,
    )

    oof_risk_x, risk_feature_names = _risk_feature_matrix(
        data=train_pool,
        scores=oof_scores,
        selection_margin=selection_margin,
    )
    slack_candidates = [float(train_margin_slack), max(float(train_margin_slack), 1.5), math.inf]
    proposal_indices = np.zeros((0,), dtype=np.int64)
    selected_slack = float(train_margin_slack)
    proposal_summary: dict[str, Any] = {}
    train_meta = train_pool["metadata"].reset_index(drop=True)
    for slack in slack_candidates:
        proposal_indices = _candidate_proposal_indices(
            quick_module,
            train_pool,
            oof_scores,
            selection_margin=selection_margin,
            margin_slack=float(slack),
        )
        accepted = train_meta.loc[proposal_indices, "accepted_delta_vs_base"].to_numpy(dtype=np.float32)
        loss_count = int(np.sum(accepted < 0.0))
        win_count = int(np.sum(accepted > 0.0))
        proposal_summary = {
            "margin_slack": None if not math.isfinite(float(slack)) else float(slack),
            "examples": int(len(proposal_indices)),
            "loss_examples": loss_count,
            "win_examples": win_count,
            "tie_examples": int(len(proposal_indices) - loss_count - win_count),
        }
        selected_slack = float(slack)
        if len(proposal_indices) >= int(min_risk_examples) and loss_count >= int(min_loss_examples):
            break
    if len(proposal_indices) == 0:
        raise RuntimeError("No OOF proposals available for risk selector training")

    accepted_delta = train_meta.loc[proposal_indices, "accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    y = (accepted_delta < 0.0).astype(np.float32)
    risk_model = _train_risk_model(
        quick_module,
        backend=str(backend),
        x=oof_risk_x[proposal_indices],
        y=y,
        accepted_delta=accepted_delta,
        feature_names=risk_feature_names,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 5000,
        loss_sample_weight=float(loss_sample_weight),
        win_sample_weight=float(win_sample_weight),
        tie_sample_weight=float(tie_sample_weight),
    )

    train_scores = dqn_module._predict(torch, dqn_model, train_pool["x"])
    threshold_scores = dqn_module._predict(torch, dqn_model, threshold_val["x"])
    eval_scores = dqn_module._predict(torch, dqn_model, eval_pool["x"])
    train_risk_x, _ = _risk_feature_matrix(data=train_pool, scores=train_scores, selection_margin=selection_margin)
    threshold_risk_x, _ = _risk_feature_matrix(data=threshold_val, scores=threshold_scores, selection_margin=selection_margin)
    eval_risk_x, _ = _risk_feature_matrix(data=eval_pool, scores=eval_scores, selection_margin=selection_margin)
    train_risk_scores = _predict_risk_model(quick_module, str(backend), risk_model, train_risk_x, risk_feature_names)
    threshold_risk_scores = _predict_risk_model(quick_module, str(backend), risk_model, threshold_risk_x, risk_feature_names)
    eval_risk_scores = _predict_risk_model(quick_module, str(backend), risk_model, eval_risk_x, risk_feature_names)

    raw_train_rows = dqn_module._selected_rows_for_margin(quick_module, train_pool, train_scores, selection_margin)
    raw_threshold_rows = dqn_module._selected_rows_for_margin(quick_module, threshold_val, threshold_scores, selection_margin)
    raw_eval_rows = dqn_module._selected_rows_for_margin(quick_module, eval_pool, eval_scores, selection_margin)
    raw_train_metrics = quick_module._selection_metrics(raw_train_rows, train_pool["metadata"].reset_index(drop=True))
    raw_threshold_metrics = quick_module._selection_metrics(raw_threshold_rows, threshold_val["metadata"].reset_index(drop=True))
    raw_eval_metrics = quick_module._selection_metrics(raw_eval_rows, eval_pool["metadata"].reset_index(drop=True))

    score_cutoff, threshold_metrics = _tune_risk_cutoff(
        quick_module,
        threshold_val,
        threshold_scores,
        threshold_risk_scores,
        selection_margin=selection_margin,
        max_loss_rate=float(max_loss_rate),
        min_override_count=int(min_override_count),
        min_total_delta=float(min_total_delta),
        max_override_rate=float(max_override_rate),
    )
    train_metrics = _metrics_with_risk(
        quick_module,
        train_pool,
        train_scores,
        train_risk_scores,
        selection_margin=selection_margin,
        score_cutoff=score_cutoff,
    )
    eval_metrics = _metrics_with_risk(
        quick_module,
        eval_pool,
        eval_scores,
        eval_risk_scores,
        selection_margin=selection_margin,
        score_cutoff=score_cutoff,
    )
    eval_loss = eval_metrics.get("selected_loss_rate_when_overridden")
    eval_constraints_satisfied = bool(
        eval_loss is None or float(eval_loss) <= float(hard_eval_max_loss_rate)
    ) and float(eval_metrics.get("total_selected_accepted_delta_vs_base", 0.0)) >= float(min_total_delta)

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_model_path = output_dir / f"{backend}_dqn_base_relative_oof_risk_selector.{_model_suffix(str(backend))}"
    _save_model(str(backend), risk_model, risk_model_path)
    copied_dqn_model_path = output_dir / dqn_model_path.name
    if dqn_model_path.resolve() != copied_dqn_model_path.resolve():
        shutil.copy2(dqn_model_path, copied_dqn_model_path)
    runtime_meta = dict(dqn_meta)
    runtime_meta["model_path"] = copied_dqn_model_path.name
    runtime_meta["selection_mode"] = "base_residual"
    runtime_meta["selection_margin"] = float(selection_margin)
    runtime_meta["risk_selector"] = {
        "enabled": True,
        "backend": str(backend),
        "model_path": risk_model_path.name,
        "feature_kind": "dqn_base_residual",
        "feature_names": list(risk_feature_names),
        "score_cutoff": float(score_cutoff),
        "label": "accepted_delta_vs_base < 0",
    }
    runtime_meta.setdefault("training", {})
    runtime_meta["risk_selector_training"] = {
        "run_dir": str(run_dir),
        "dqn_artifact": str(dqn_artifact),
        "backend": str(backend),
        "top_k": int(resolved_top_k),
        "selection_margin": float(selection_margin),
        "oof_folds": int(oof_folds),
        "oof_epochs": int(oof_epochs),
        "calibration_source": calibration_source,
        "train_margin_slack": None if not math.isfinite(selected_slack) else float(selected_slack),
        "num_boost_round": int(num_boost_round),
        "max_loss_rate": float(max_loss_rate),
        "hard_eval_max_loss_rate": float(hard_eval_max_loss_rate),
        "seed": int(seed),
    }
    artifact_path = output_dir / "torch_dqn_base_relative_oof_risk_tree_ranker.json"
    _write_json(artifact_path, runtime_meta)

    summary = {
        "artifact_path": str(artifact_path),
        "risk_model_path": str(risk_model_path),
        "selection_margin": float(selection_margin),
        "risk_selector_score_cutoff": float(score_cutoff),
        "training": runtime_meta["risk_selector_training"],
        "oof_fold_summaries": fold_summaries,
        "oof_proposal_summary": proposal_summary,
        "risk_train_label_rate": {
            "examples": int(len(proposal_indices)),
            "loss_rate": float(np.mean(y)) if y.size else None,
            "loss_examples": int(np.sum(y > 0.5)),
            "nonloss_examples": int(np.sum(y <= 0.5)),
        },
        "raw_dqn": {
            "train": raw_train_metrics,
            "threshold_val": raw_threshold_metrics,
            "eval": raw_eval_metrics,
        },
        "risk_selector": {
            "train": train_metrics,
            "threshold_val_calibration": threshold_metrics,
            "eval": eval_metrics,
            "hard_eval_constraints_satisfied": eval_constraints_satisfied,
        },
    }
    _write_json(output_dir / "torch_dqn_base_relative_oof_risk_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOF loss-veto selector on top of a base-relative DQN ranker.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dqn-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--oof-epochs", type=int, default=30)
    parser.add_argument("--threshold-fraction", type=float, default=0.15)
    parser.add_argument("--train-margin-slack", type=float, default=1.0)
    parser.add_argument("--min-risk-examples", type=int, default=250)
    parser.add_argument("--min-loss-examples", type=int, default=25)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--max-loss-rate", type=float, default=0.08)
    parser.add_argument("--hard-eval-max-loss-rate", type=float, default=0.12)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.35)
    parser.add_argument("--secondary-scale", type=float, default=0.20)
    parser.add_argument("--target-clip", type=float, default=4.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--pair-loss-weight", type=float, default=0.7)
    parser.add_argument("--rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-group", type=int, default=4)
    parser.add_argument("--base-weight", type=float, default=4.0)
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--loss-weight", type=float, default=10.0)
    parser.add_argument("--tie-weight", type=float, default=0.5)
    parser.add_argument("--secondary-weight", type=float, default=1.0)
    parser.add_argument("--loss-sample-weight", type=float, default=12.0)
    parser.add_argument("--win-sample-weight", type=float, default=2.0)
    parser.add_argument("--tie-sample-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_and_export(
        run_dir=Path(args.run_dir),
        dqn_artifact=Path(args.dqn_artifact),
        output_dir=Path(args.output_dir),
        backend=str(args.backend),
        top_k=args.top_k,
        oof_folds=int(args.oof_folds),
        oof_epochs=int(args.oof_epochs),
        threshold_fraction=float(args.threshold_fraction),
        train_margin_slack=float(args.train_margin_slack),
        min_risk_examples=int(args.min_risk_examples),
        min_loss_examples=int(args.min_loss_examples),
        num_boost_round=int(args.num_boost_round),
        max_loss_rate=float(args.max_loss_rate),
        hard_eval_max_loss_rate=float(args.hard_eval_max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        secondary_scale=float(args.secondary_scale),
        target_clip=float(args.target_clip),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        pair_batch_size=int(args.pair_batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        pair_loss_weight=float(args.pair_loss_weight),
        rank_loss_weight=float(args.rank_loss_weight),
        max_pairs_per_group=int(args.max_pairs_per_group),
        base_weight=float(args.base_weight),
        win_weight=float(args.win_weight),
        loss_weight=float(args.loss_weight),
        tie_weight=float(args.tie_weight),
        secondary_weight=float(args.secondary_weight),
        loss_sample_weight=float(args.loss_sample_weight),
        win_sample_weight=float(args.win_sample_weight),
        tie_sample_weight=float(args.tie_sample_weight),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
