#!/usr/bin/env python
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
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.train_dqn import _device

from top32_xlron_live_risk_features import (
    DEFAULT_BUCKETS,
    live_risk_feature_names,
    live_risk_feature_vector,
    parse_bucket_set,
)
from train_neural_stable_override_selector import _batch_tensors, _iter_batches, _json_safe, _load_dataset, _write_json
from train_top32_xlron_full_dqn_distill import _load_xlron_checkpoint_model, _xlron_forward


STATE_ARRAY_KEYS = (
    "action_features",
    "route_basic_features",
    "global_features",
    "request_features",
)


def _resolve_cli_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def _masked_argmax(scores: np.ndarray, mask: np.ndarray) -> int:
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    if valid.size == 0:
        return -1
    values = np.asarray(scores, dtype=np.float32)[valid]
    return int(valid[int(np.argmax(values))])


def _predict_scores(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    torch: Any,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_indices in _iter_batches(indices, int(batch_size), shuffle=False, rng=np.random.default_rng(0)):
            tensors = _batch_tensors(data, np.asarray(batch_indices, dtype=np.int64), device=device, torch=torch)
            raw_logits, _value = _xlron_forward(model, tensors, edge_index)
            chunks.append(raw_logits.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, int(np.asarray(data["candidate_mask"]).shape[1])), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _group_context(metadata: pd.DataFrame) -> dict[int, dict[str, Any]]:
    context: dict[int, dict[str, Any]] = {}
    for group_id, group in metadata.groupby("group_id", sort=False):
        first = group.iloc[0]
        bucket = f"{first.get('traffic_scenario', '')}:{first.get('load_name', '')}"
        split = str(first.get("split", "train"))
        context[int(group_id)] = {
            "bucket": bucket,
            "split": split,
            "traffic_scenario": str(first.get("traffic_scenario", "")),
            "load_name": str(first.get("load_name", "")),
        }
    return context


def _state_arrays(data: dict[str, np.ndarray], position: int) -> dict[str, np.ndarray]:
    return {key: np.asarray(data[key])[int(position)] for key in STATE_ARRAY_KEYS if key in data}


def _build_live_rows(
    *,
    metadata: pd.DataFrame,
    data: dict[str, np.ndarray],
    scores: np.ndarray,
    bucket_vocab: list[str],
    protected_buckets: set[str],
    training_buckets: set[str],
    proposal_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[str], dict[str, Any]]:
    group_ids = np.asarray(data["group_ids"], dtype=np.int64)
    candidate_mask = np.asarray(data["candidate_mask"], dtype=bool)
    label_mask = np.asarray(data["label_mask"], dtype=bool)
    accepted = np.nan_to_num(np.asarray(data["accepted_delta_vs_base"], dtype=np.float32), nan=0.0)
    base_index = np.asarray(data["base_index"], dtype=np.int64)
    label_weight = np.asarray(data.get("label_weight", np.ones_like(accepted, dtype=np.float32)), dtype=np.float32)
    context = _group_context(metadata)
    action_dim = int(np.asarray(data["action_features"]).shape[-1])
    route_basic_dim = int(np.asarray(data.get("route_basic_features", np.zeros((len(group_ids), candidate_mask.shape[1], 0)))).shape[-1])
    global_dim = int(np.asarray(data["global_features"]).shape[-1])
    request_dim = int(np.asarray(data["request_features"]).shape[-1])
    feature_names = live_risk_feature_names(
        action_dim=action_dim,
        route_basic_dim=route_basic_dim,
        global_dim=global_dim,
        request_dim=request_dim,
        bucket_vocab=bucket_vocab,
    )

    features: list[np.ndarray] = []
    labels: list[float] = []
    weights: list[float] = []
    rows: list[dict[str, Any]] = []
    skipped = {
        "base_selected": 0,
        "unlabeled_selected": 0,
        "no_valid_candidate": 0,
        "outside_training_bucket": 0,
        "no_labeled_nonbase_candidate": 0,
    }
    for local_position, group_id in enumerate(group_ids):
        mask = candidate_mask[local_position]
        live_selected = _masked_argmax(scores[local_position], mask)
        if live_selected < 0:
            skipped["no_valid_candidate"] += 1
            continue
        base = int(base_index[local_position])
        if base < 0 or base >= mask.shape[0] or not bool(mask[base]):
            valid = np.flatnonzero(mask)
            base = int(valid[0]) if valid.size else 0
        ctx = context.get(int(group_id), {"bucket": "unknown:unknown", "split": "train"})
        bucket = str(ctx["bucket"])
        if training_buckets and bucket not in training_buckets:
            skipped["outside_training_bucket"] += 1
            continue
        if str(proposal_mode) == "all_labeled":
            candidate_indices = [
                int(index)
                for index in np.flatnonzero(mask & label_mask[local_position])
                if int(index) != int(base)
            ]
            if not candidate_indices:
                skipped["no_labeled_nonbase_candidate"] += 1
                continue
        else:
            if live_selected == base:
                skipped["base_selected"] += 1
                continue
            if not bool(label_mask[local_position, live_selected]):
                skipped["unlabeled_selected"] += 1
                continue
            candidate_indices = [int(live_selected)]

        for selected in candidate_indices:
            delta = float(accepted[local_position, selected])
            risk_label = 1.0 if delta < 0.0 else 0.0
            feature = live_risk_feature_vector(
                arrays=_state_arrays(data, local_position),
                scores=scores[local_position],
                candidate_mask=mask,
                selected_index=int(selected),
                base_index=int(base),
                bucket=bucket,
                bucket_vocab=bucket_vocab,
            )
            features.append(feature)
            labels.append(risk_label)
            weights.append(float(max(label_weight[local_position, selected], 0.05)))
            rows.append(
                {
                    "position": int(local_position),
                    "group_id": int(group_id),
                    "split": str(ctx.get("split", "train")),
                    "bucket": bucket,
                    "protected_bucket": bool(bucket in protected_buckets),
                    "proposal_mode": str(proposal_mode),
                    "live_selected_index": int(live_selected),
                    "selected_index": int(selected),
                    "base_index": int(base),
                    "selected_delta": delta,
                    "risk_label": int(risk_label),
                    "label_weight": float(weights[-1]),
                    "score_margin_vs_base": float(scores[local_position, selected] - scores[local_position, base]),
                }
            )
    matrix = np.asarray(features, dtype=np.float32) if features else np.zeros((0, len(feature_names)), dtype=np.float32)
    return matrix, np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float32), rows, feature_names, skipped


def _split_indices(rows: list[dict[str, Any]], *, seed: int, train_fraction: float, calibration_fraction: float) -> dict[str, np.ndarray]:
    splits: dict[str, list[int]] = {"train": [], "calibration": [], "eval": []}
    for index, row in enumerate(rows):
        split = str(row.get("split", "train"))
        if split not in splits:
            split = "train"
        splits[split].append(int(index))
    if not splits["calibration"] and not splits["eval"] and len(splits["train"]) > 2:
        by_group: dict[int, list[int]] = {}
        for index, row in enumerate(rows):
            by_group.setdefault(int(row.get("group_id", index)), []).append(int(index))
        groups = np.asarray(sorted(by_group), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(groups)
        train_end = int(round(len(groups) * float(train_fraction)))
        cal_end = train_end + int(round(len(groups) * float(calibration_fraction)))
        train_groups = set(int(value) for value in groups[:train_end])
        cal_groups = set(int(value) for value in groups[train_end:cal_end])
        eval_groups = set(int(value) for value in groups[cal_end:])
        splits = {"train": [], "calibration": [], "eval": []}
        for group_id, indices in by_group.items():
            if int(group_id) in train_groups:
                splits["train"].extend(indices)
            elif int(group_id) in cal_groups:
                splits["calibration"].extend(indices)
            elif int(group_id) in eval_groups:
                splits["eval"].extend(indices)
            else:
                splits["train"].extend(indices)
    elif not splits["calibration"] and splits["eval"]:
        splits["calibration"] = splits["eval"][: max(1, len(splits["eval"]) // 2)]
        splits["eval"] = splits["eval"][max(1, len(splits["eval"]) // 2) :]
    return {key: np.asarray(values, dtype=np.int64) for key, values in splits.items()}


def _metrics_for_threshold(rows: list[dict[str, Any]], predictions: np.ndarray, indices: np.ndarray, threshold: float) -> dict[str, Any]:
    selected_rows = [rows[int(index)] for index in np.asarray(indices, dtype=np.int64)]
    if not selected_rows:
        return {"examples": 0, "threshold": float(threshold)}
    pred = np.asarray(predictions, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    delta = np.asarray([float(row["selected_delta"]) for row in selected_rows], dtype=np.float32)
    veto = pred >= float(threshold)
    kept = ~veto
    raw_loss = delta < 0.0
    raw_win = delta > 0.0
    kept_delta = np.where(kept, delta, 0.0)
    protected = np.asarray([bool(row.get("protected_bucket", False)) for row in selected_rows], dtype=bool)
    return {
        "examples": int(len(selected_rows)),
        "threshold": float(threshold),
        "raw_total_delta": float(delta.sum()),
        "kept_total_delta": float(kept_delta.sum()),
        "delta_improvement": float(kept_delta.sum() - delta.sum()),
        "raw_loss_count": int(raw_loss.sum()),
        "raw_win_count": int(raw_win.sum()),
        "raw_loss_rate": float(raw_loss.mean()),
        "veto_count": int(veto.sum()),
        "veto_rate": float(veto.mean()),
        "veto_loss_count": int((veto & raw_loss).sum()),
        "veto_win_count": int((veto & raw_win).sum()),
        "veto_tie_count": int((veto & ~(raw_loss | raw_win)).sum()),
        "veto_win_rate": float((veto & raw_win).sum() / max(int(veto.sum()), 1)),
        "kept_count": int(kept.sum()),
        "kept_loss_count": int((kept & raw_loss).sum()),
        "kept_win_count": int((kept & raw_win).sum()),
        "kept_loss_rate": float((kept & raw_loss).sum() / max(int(kept.sum()), 1)),
        "protected_examples": int(protected.sum()),
        "protected_delta_improvement": float((np.where(veto & protected, 0.0, delta) - delta)[protected].sum()) if protected.any() else 0.0,
        "protected_veto_count": int((veto & protected).sum()),
    }


def _threshold_grid(predictions: np.ndarray) -> list[float]:
    values = np.asarray(predictions, dtype=np.float32)
    finite = values[np.isfinite(values)]
    grid = {0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95}
    if finite.size:
        for q in (0.25, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98):
            grid.add(float(np.quantile(finite, q)))
    return sorted(float(max(0.0, min(1.0, item))) for item in grid)


def _tune_threshold(
    *,
    rows: list[dict[str, Any]],
    predictions: np.ndarray,
    indices: np.ndarray,
    max_veto_rate: float,
    max_veto_win_rate: float,
    min_delta_improvement: float,
    min_veto_count: int,
) -> tuple[float, dict[str, Any]]:
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    for threshold in _threshold_grid(np.asarray(predictions)[np.asarray(indices, dtype=np.int64)]):
        metrics = _metrics_for_threshold(rows, predictions, indices, threshold)
        key = (
            float(metrics.get("delta_improvement", 0.0)),
            -float(metrics.get("veto_win_count", 0)),
            -float(metrics.get("veto_rate", 0.0)),
        )
        if fallback_key is None or key > fallback_key:
            fallback_key = key
            fallback = dict(metrics)
        if int(metrics.get("veto_count", 0)) < int(min_veto_count):
            continue
        if float(metrics.get("veto_rate", 0.0)) > float(max_veto_rate):
            continue
        if float(metrics.get("veto_win_rate", 0.0)) > float(max_veto_win_rate):
            continue
        if float(metrics.get("delta_improvement", 0.0)) < float(min_delta_improvement):
            continue
        if best_key is None or key > best_key:
            best_key = key
            best = dict(metrics)
    if best is None:
        assert fallback is not None
        fallback["constraints_satisfied"] = False
        return float(fallback["threshold"]), fallback
    best["constraints_satisfied"] = True
    return float(best["threshold"]), best


def _bucket_summary(rows: list[dict[str, Any]], indices: np.ndarray, predictions: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_bucket: dict[str, list[int]] = {}
    for index in np.asarray(indices, dtype=np.int64):
        by_bucket.setdefault(str(rows[int(index)]["bucket"]), []).append(int(index))
    for bucket, bucket_indices in sorted(by_bucket.items()):
        metrics = _metrics_for_threshold(rows, predictions, np.asarray(bucket_indices, dtype=np.int64), threshold)
        metrics["bucket"] = bucket
        out.append(metrics)
    return out


def train(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    import xgboost as xgb
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    torch.manual_seed(int(args.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    input_dir = _resolve_cli_path(args.input_dir)
    if input_dir is None:
        raise ValueError("--input-dir is required")
    checkpoint_path = _resolve_cli_path(args.xlron_checkpoint)
    if checkpoint_path is None:
        raise ValueError("--xlron-checkpoint is required")
    output_dir = _resolve_cli_path(args.output_dir)
    if output_dir is None:
        raise ValueError("--output-dir is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = _load_dataset(input_dir)
    metadata = loaded["metadata"]
    data = loaded["neural"]
    model, checkpoint = _load_xlron_checkpoint_model(checkpoint_path, device=device, torch=torch)
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    all_indices = np.arange(int(np.asarray(data["group_ids"]).shape[0]), dtype=np.int64)
    scores = _predict_scores(
        model=model,
        data=data,
        indices=all_indices,
        edge_index=edge_index,
        batch_size=int(args.batch_size),
        device=device,
        torch=torch,
    )
    protected_buckets = parse_bucket_set(str(args.protected_buckets))
    apply_buckets = parse_bucket_set(str(args.apply_buckets)) if str(args.apply_buckets).strip() else set(protected_buckets)
    training_buckets = parse_bucket_set(str(args.training_buckets))
    bucket_vocab = sorted(
        set(DEFAULT_BUCKETS)
        .union(parse_bucket_set(str(args.bucket_vocab)))
        .union(protected_buckets)
        .union(apply_buckets)
        .union(training_buckets)
    )
    features, labels, label_weights, rows, feature_names, skipped = _build_live_rows(
        metadata=metadata,
        data=data,
        scores=scores,
        bucket_vocab=bucket_vocab,
        protected_buckets=protected_buckets,
        training_buckets=training_buckets,
        proposal_mode="live_selected" if str(args.proposal_mode) == "live_or_all" else str(args.proposal_mode),
    )
    proposal_mode_used = "live_selected" if str(args.proposal_mode) == "live_or_all" else str(args.proposal_mode)
    if features.shape[0] == 0 and str(args.proposal_mode) == "live_or_all":
        features, labels, label_weights, rows, feature_names, skipped = _build_live_rows(
            metadata=metadata,
            data=data,
            scores=scores,
            bucket_vocab=bucket_vocab,
            protected_buckets=protected_buckets,
            training_buckets=training_buckets,
            proposal_mode="all_labeled",
        )
        proposal_mode_used = "all_labeled"
    if features.shape[0] == 0:
        raise RuntimeError("No live non-base labeled selections found for risk selector")
    if np.unique(labels).size < 2:
        raise RuntimeError(f"Risk selector needs both classes, got labels={sorted(set(labels.tolist()))}")

    splits = _split_indices(
        rows,
        seed=int(args.seed),
        train_fraction=float(args.train_fraction),
        calibration_fraction=float(args.calibration_fraction),
    )
    train_idx = splits["train"]
    cal_idx = splits["calibration"] if len(splits["calibration"]) else train_idx
    eval_idx = splits["eval"] if len(splits["eval"]) else cal_idx
    sample_weight = np.asarray(label_weights, dtype=np.float32).copy()
    sample_weight *= np.where(labels > 0.5, float(args.loss_sample_weight), float(args.nonloss_sample_weight))

    dtrain = xgb.DMatrix(features[train_idx], label=labels[train_idx], weight=sample_weight[train_idx], feature_names=feature_names)
    watchlist = [(dtrain, "train")]
    if len(cal_idx):
        dcal = xgb.DMatrix(features[cal_idx], label=labels[cal_idx], weight=sample_weight[cal_idx], feature_names=feature_names)
        watchlist.append((dcal, "calibration"))
    scale_pos_weight = float(max((labels[train_idx] <= 0.5).sum(), 1) / max((labels[train_idx] > 0.5).sum(), 1))
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss"],
        "eta": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "min_child_weight": float(args.min_child_weight),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "lambda": float(args.reg_lambda),
        "alpha": float(args.reg_alpha),
        "tree_method": str(args.tree_method),
        "device": str(args.xgboost_device),
        "scale_pos_weight": scale_pos_weight,
        "seed": int(args.seed),
    }
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=int(args.num_boost_round),
        evals=watchlist,
        early_stopping_rounds=int(args.early_stopping_rounds) if int(args.early_stopping_rounds) > 0 else None,
        verbose_eval=bool(args.verbose_eval),
    )
    dall = xgb.DMatrix(features, feature_names=feature_names)
    predictions = booster.predict(dall).astype(np.float32)
    threshold, cal_metrics = _tune_threshold(
        rows=rows,
        predictions=predictions,
        indices=cal_idx,
        max_veto_rate=float(args.max_veto_rate),
        max_veto_win_rate=float(args.max_veto_win_rate),
        min_delta_improvement=float(args.min_delta_improvement),
        min_veto_count=int(args.min_veto_count),
    )
    train_metrics = _metrics_for_threshold(rows, predictions, train_idx, threshold)
    eval_metrics = _metrics_for_threshold(rows, predictions, eval_idx, threshold)
    all_metrics = _metrics_for_threshold(rows, predictions, np.arange(len(rows), dtype=np.int64), threshold)
    model_path = output_dir / "xgboost_xlron_live_risk_selector.json"
    booster.save_model(str(model_path))
    artifact = {
        "stage": "top32_xlron_live_risk_selector",
        "model_type": "xgboost_binary_loss_veto",
        "model_path": model_path.name,
        "feature_names": feature_names,
        "bucket_vocab": bucket_vocab,
        "protected_buckets": sorted(protected_buckets),
        "apply_buckets": sorted(apply_buckets),
        "training_buckets": sorted(training_buckets),
        "proposal_mode": proposal_mode_used,
        "threshold": float(threshold),
        "base_index": 0,
        "xlron_checkpoint": str(checkpoint_path),
        "checkpoint_architecture": str(checkpoint.get("architecture", "")) if isinstance(checkpoint, dict) else "",
    }
    artifact_path = output_dir / "xlron_live_risk_selector_artifact.json"
    _write_json(artifact_path, artifact)

    rows_path = output_dir / "xlron_live_risk_selector_rows.csv"
    pd.DataFrame(rows).assign(risk_prediction=predictions).to_csv(rows_path, index=False)
    summary = {
        "stage": "train_top32_xlron_live_risk_selector",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "model_path": str(model_path),
        "xlron_checkpoint": str(checkpoint_path),
        "examples": int(features.shape[0]),
        "features": int(features.shape[1]),
        "proposal_mode": proposal_mode_used,
        "labels": {
            "loss": int((labels > 0.5).sum()),
            "nonloss": int((labels <= 0.5).sum()),
            "loss_rate": float((labels > 0.5).mean()),
        },
        "splits": {key: int(len(value)) for key, value in splits.items()},
        "skipped": skipped,
        "params": params,
        "best_iteration": int(getattr(booster, "best_iteration", int(args.num_boost_round) - 1)),
        "threshold": float(threshold),
        "calibration_metrics": cal_metrics,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "all_metrics": all_metrics,
        "eval_by_bucket": _bucket_summary(rows, eval_idx, predictions, threshold),
    }
    _write_json(output_dir / "xlron_live_risk_selector_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an XGBoost live loss-veto selector for Top32 XLRON runtime choices.")
    parser.add_argument("--config", default="configs/experiments/eon/remote_collect_online_dqn_base_alltopn_h100_train_stratified.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--xlron-checkpoint", required=True)
    parser.add_argument("--protected-buckets", default="bursty:high,bursty:medium,bursty:overload,hotspot:high,hotspot:medium,nonuniform:high,nonuniform:medium")
    parser.add_argument("--apply-buckets", default="")
    parser.add_argument("--training-buckets", default="")
    parser.add_argument("--proposal-mode", choices=("live_selected", "all_labeled", "live_or_all"), default="live_or_all")
    parser.add_argument("--bucket-vocab", default="")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--xgboost-device", default="cpu")
    parser.add_argument("--loss-sample-weight", type=float, default=8.0)
    parser.add_argument("--nonloss-sample-weight", type=float, default=1.0)
    parser.add_argument("--max-veto-rate", type=float, default=0.12)
    parser.add_argument("--max-veto-win-rate", type=float, default=0.40)
    parser.add_argument("--min-delta-improvement", type=float, default=0.0)
    parser.add_argument("--min-veto-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--verbose-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    train(config, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
