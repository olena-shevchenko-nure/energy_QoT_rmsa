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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.train_dqn import _device
from train_neural_stable_override_selector import (
    _batch_tensors,
    _iter_batches,
    _json_safe,
    _load_dataset,
    _resolve_cli_path,
    _resolve_path,
    _write_json,
)
from train_neural_three_head_override_selector import (
    _build_three_head_model,
    _forward_three_head,
    _grid,
    _loss,
    _predict,
    _selection_metrics,
)


def _group_bucket(metadata: pd.DataFrame, group_id: int) -> str:
    group = metadata[metadata["group_id"].astype(int) == int(group_id)]
    if group.empty:
        return "missing"
    nonbase = group[~group["is_base"].astype(bool)]
    if nonbase.empty:
        return "base_only"
    max_delta = float(nonbase["accepted_delta_vs_base"].max())
    min_delta = float(nonbase["accepted_delta_vs_base"].min())
    if max_delta > 0.0 and min_delta < 0.0:
        return "win_and_loss"
    if max_delta > 0.0:
        return "win_available"
    if min_delta < 0.0:
        return "loss_only"
    return "tie_only"


def _make_stratified_folds(
    *,
    metadata: pd.DataFrame,
    group_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    if int(n_folds) < 2:
        raise ValueError("n_folds must be >= 2")
    rows = [
        {"position": int(position), "bucket": _group_bucket(metadata, int(group_id))}
        for position, group_id in enumerate(np.asarray(group_ids, dtype=np.int64))
    ]
    table = pd.DataFrame(rows)
    fold_values = np.full((len(group_ids),), -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for _, bucket in table.groupby("bucket", sort=False):
        values = bucket["position"].to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(values)
        for offset, position in enumerate(values):
            fold_values[int(position)] = int(offset % int(n_folds))
    if np.any(fold_values < 0):
        raise RuntimeError("Some groups were not assigned to a fold")
    folds = [np.flatnonzero(fold_values == fold).astype(np.int64) for fold in range(int(n_folds))]
    empty = [index for index, fold in enumerate(folds) if len(fold) == 0]
    if empty:
        raise ValueError(f"Empty OOF folds: {empty}")
    return folds, fold_values


def _train_one_model(
    *,
    config: ExperimentConfig,
    data: dict[str, np.ndarray],
    initial_checkpoint: Path | None,
    train_indices: np.ndarray,
    edge_index: Any,
    device: str,
    torch: Any,
    args: argparse.Namespace,
    seed: int,
    fold: int | None,
    predict_indices: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], list[dict[str, float]], list[dict[str, np.ndarray]]]:
    torch.manual_seed(int(seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model, model_info = _build_three_head_model(
        config=config,
        data=data,
        initial_checkpoint=initial_checkpoint,
        device=device,
        freeze_encoders=bool(args.freeze_encoders),
        freeze_trunk=bool(args.freeze_trunk),
        init_from_q_head=bool(args.init_from_q_head),
        torch=torch,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable model parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(seed))
    epoch_losses: list[dict[str, float]] = []
    heldout_predictions: list[dict[str, np.ndarray]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rows: list[dict[str, float]] = []
        for batch_indices in _iter_batches(train_indices, int(args.batch_size), shuffle=True, rng=rng):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            outputs = _forward_three_head(model, tensors, edge_index)
            loss, parts = _loss(
                outputs=outputs,
                tensors=tensors,
                win_bce_weight=float(args.win_bce_weight),
                loss_bce_weight=float(args.loss_bce_weight),
                delta_weight=float(args.delta_weight),
                ce_weight=float(args.ce_weight),
                pairwise_weight=float(args.pairwise_weight),
                pairwise_margin=float(args.pairwise_margin),
                order_epsilon=float(args.order_epsilon),
                target_scale=float(args.target_scale),
                max_pos_weight=float(args.max_pos_weight),
                torch=torch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
            optimizer.step()
            rows.append(parts)
        loss_row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean([item["loss"] for item in rows])) if rows else 0.0,
            "train_win_bce": float(np.mean([item["win_bce"] for item in rows])) if rows else 0.0,
            "train_loss_bce": float(np.mean([item["loss_bce"] for item in rows])) if rows else 0.0,
            "train_delta_regression": float(np.mean([item["delta_regression"] for item in rows])) if rows else 0.0,
            "train_ce": float(np.mean([item["ce"] for item in rows])) if rows else 0.0,
            "train_pairwise": float(np.mean([item["pairwise"] for item in rows])) if rows else 0.0,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in rows])) if rows else 0.0,
        }
        epoch_losses.append(loss_row)
        if predict_indices is not None:
            prediction = _predict(
                model=model,
                data=data,
                indices=predict_indices,
                edge_index=edge_index,
                batch_size=int(args.batch_size),
                device=device,
                target_scale=float(args.target_scale),
                torch=torch,
            )
            heldout_predictions.append(prediction)
        print(
            json.dumps(
                _json_safe(
                    {
                        "event": "epoch",
                        "fold": None if fold is None else int(fold),
                        "epoch": int(epoch),
                        **loss_row,
                    }
                ),
                sort_keys=True,
            ),
            flush=True,
        )
    return model, model_info, epoch_losses, heldout_predictions


def _tune_thresholds_from_predictions(
    *,
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    win_score_weight: float,
    loss_score_weight: float,
    max_loss_rate: float,
    max_override_rate: float,
    min_total_delta: float,
    min_overrides: int,
    min_delta_floor: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(indices) == 0:
        thresholds = {"win_threshold": 1.0, "loss_threshold": 0.0, "delta_margin": float(min_delta_floor)}
        return thresholds, {"groups": 0, "constraints_satisfied": False}

    mask = (
        np.asarray(data["candidate_mask"])[indices].astype(bool)
        & np.asarray(data["label_mask"])[indices].astype(bool)
    )
    base_index = np.asarray(data["base_index"])[indices].astype(np.int64)
    row = np.arange(len(indices))
    mask[row, np.clip(base_index, 0, mask.shape[1] - 1)] = False
    win_values = predictions["win_prob"][mask]
    loss_values = predictions["loss_prob"][mask]
    delta_values = predictions["delta_pred"][mask]
    win_grid = _grid(
        win_values,
        [0.50, 0.53, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98],
        (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    )
    loss_grid = _grid(
        loss_values,
        [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
        (0.05, 0.10, 0.20, 0.30, 0.50),
    )
    delta_grid = _grid(
        delta_values,
        [float(min_delta_floor), 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0],
        (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
        lower=float(min_delta_floor),
    )

    best_thresholds: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_thresholds: dict[str, float] | None = None
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    for win_threshold in win_grid:
        for loss_threshold in loss_grid:
            for delta_margin in delta_grid:
                metrics = _selection_metrics(
                    predictions=predictions,
                    data=data,
                    indices=indices,
                    win_threshold=float(win_threshold),
                    loss_threshold=float(loss_threshold),
                    delta_margin=float(delta_margin),
                    win_score_weight=float(win_score_weight),
                    loss_score_weight=float(loss_score_weight),
                )
                loss_rate = metrics.get("selected_loss_rate")
                loss_value = 0.0 if loss_rate is None else float(loss_rate)
                total_delta = float(metrics.get("selected_total_delta") or 0.0)
                override_rate = float(metrics.get("override_rate") or 0.0)
                override_count = int(metrics.get("override_count") or 0)
                key = (total_delta, -loss_value, float(override_count), -float(delta_margin), float(win_threshold))
                if fallback_key is None or key > fallback_key:
                    fallback_key = key
                    fallback_thresholds = {
                        "win_threshold": float(win_threshold),
                        "loss_threshold": float(loss_threshold),
                        "delta_margin": float(delta_margin),
                    }
                    fallback_metrics = dict(metrics)
                if total_delta < float(min_total_delta):
                    continue
                if loss_value > float(max_loss_rate):
                    continue
                if override_rate > float(max_override_rate):
                    continue
                if override_count < int(min_overrides):
                    continue
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = {
                        "win_threshold": float(win_threshold),
                        "loss_threshold": float(loss_threshold),
                        "delta_margin": float(delta_margin),
                    }
                    best_metrics = dict(metrics)
    if best_metrics is None:
        assert fallback_metrics is not None and fallback_thresholds is not None
        fallback_metrics["constraints_satisfied"] = False
        fallback_metrics["tune_found_feasible"] = False
        return fallback_thresholds, fallback_metrics
    best_metrics["constraints_satisfied"] = True
    best_metrics["tune_found_feasible"] = True
    return best_thresholds or {}, best_metrics


def _epoch_score(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["oof_metrics"]
    feasible_bonus = 1000.0 if bool(metrics.get("constraints_satisfied")) else 0.0
    return (
        feasible_bonus + float(metrics.get("selected_total_delta") or 0.0),
        -float(metrics.get("selected_loss_rate") or 0.0),
        float(metrics.get("override_count") or 0.0),
        -float(row.get("epoch") or 0.0),
    )


def run(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    loaded = _load_dataset(Path(args.input_dir))
    data = loaded["neural"]
    metadata = loaded["metadata"]
    group_ids = np.asarray(data["group_ids"], dtype=np.int64)
    n_groups = int(len(group_ids))
    n_max = int(np.asarray(data["candidate_mask"]).shape[1])
    folds, fold_assignment = _make_stratified_folds(
        metadata=metadata,
        group_ids=group_ids,
        n_folds=int(args.folds),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = _resolve_cli_path(args.initial_checkpoint) or _resolve_path(config, "dqn_checkpoint")
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)

    oof_predictions = {
        "win_prob": np.full((int(args.epochs), n_groups, n_max), np.nan, dtype=np.float32),
        "loss_prob": np.full((int(args.epochs), n_groups, n_max), np.nan, dtype=np.float32),
        "delta_pred": np.full((int(args.epochs), n_groups, n_max), np.nan, dtype=np.float32),
    }
    fold_summaries: list[dict[str, Any]] = []
    all_indices = np.arange(n_groups, dtype=np.int64)
    model_info: dict[str, Any] | None = None

    for fold_index, heldout in enumerate(folds):
        train_indices = np.setdiff1d(all_indices, heldout, assume_unique=False).astype(np.int64)
        print(
            json.dumps(
                {
                    "event": "fold_start",
                    "fold": int(fold_index),
                    "train_groups": int(len(train_indices)),
                    "heldout_groups": int(len(heldout)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        _model, fold_model_info, losses, heldout_preds = _train_one_model(
            config=config,
            data=data,
            initial_checkpoint=initial_checkpoint,
            train_indices=train_indices,
            edge_index=edge_index,
            device=device,
            torch=torch,
            args=args,
            seed=int(args.seed) + int(fold_index),
            fold=int(fold_index),
            predict_indices=heldout,
        )
        model_info = fold_model_info
        for epoch_index, prediction in enumerate(heldout_preds):
            for key in oof_predictions:
                oof_predictions[key][epoch_index, heldout, :] = prediction[key]
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "train_groups": int(len(train_indices)),
                "heldout_groups": int(len(heldout)),
                "heldout_group_ids": group_ids[heldout].astype(int).tolist(),
                "last_loss": losses[-1] if losses else None,
            }
        )

    if any(np.isnan(value).any() for value in oof_predictions.values()):
        raise RuntimeError("OOF predictions contain missing values")

    history: list[dict[str, Any]] = []
    for epoch_index in range(int(args.epochs)):
        epoch_predictions = {key: value[epoch_index] for key, value in oof_predictions.items()}
        thresholds, metrics = _tune_thresholds_from_predictions(
            predictions=epoch_predictions,
            data=data,
            indices=all_indices,
            win_score_weight=float(args.win_score_weight),
            loss_score_weight=float(args.loss_score_weight),
            max_loss_rate=float(args.max_loss_rate),
            max_override_rate=float(args.max_override_rate),
            min_total_delta=float(args.min_total_delta),
            min_overrides=int(args.min_overrides),
            min_delta_floor=float(args.min_delta_floor),
        )
        row = {
            "epoch": int(epoch_index + 1),
            "thresholds": thresholds,
            "oof_metrics": metrics,
        }
        history.append(row)
        print(json.dumps(_json_safe({"event": "oof_epoch", **row}), sort_keys=True), flush=True)

    best_row = max(history, key=_epoch_score)
    best_epoch = int(best_row["epoch"])
    best_thresholds = dict(best_row["thresholds"])
    best_predictions = {key: value[best_epoch - 1] for key, value in oof_predictions.items()}

    np.savez_compressed(
        output_dir / "neural_three_head_oof_predictions.npz",
        group_ids=group_ids.astype(np.int64),
        fold_assignment=fold_assignment.astype(np.int64),
        win_prob=oof_predictions["win_prob"],
        loss_prob=oof_predictions["loss_prob"],
        delta_pred=oof_predictions["delta_pred"],
    )
    np.savez_compressed(
        output_dir / "neural_three_head_oof_best_predictions.npz",
        group_ids=group_ids.astype(np.int64),
        fold_assignment=fold_assignment.astype(np.int64),
        win_prob=best_predictions["win_prob"],
        loss_prob=best_predictions["loss_prob"],
        delta_pred=best_predictions["delta_pred"],
    )

    final_checkpoint_path: str | None = None
    final_model_info: dict[str, Any] | None = None
    if bool(args.train_final):
        print(
            json.dumps(
                {"event": "final_train_start", "groups": int(n_groups), "epochs": int(best_epoch)},
                sort_keys=True,
            ),
            flush=True,
        )
        final_args = argparse.Namespace(**vars(args))
        final_args.epochs = int(best_epoch)
        final_model, final_model_info, final_losses, _pred = _train_one_model(
            config=config,
            data=data,
            initial_checkpoint=initial_checkpoint,
            train_indices=all_indices,
            edge_index=edge_index,
            device=device,
            torch=torch,
            args=final_args,
            seed=int(args.seed) + 100000,
            fold=None,
            predict_indices=None,
        )
        final_checkpoint = output_dir / "neural_three_head_oof_final.pt"
        torch.save(
            {
                "model_state_dict": final_model.state_dict(),
                "epoch": int(best_epoch),
                "thresholds": best_thresholds,
                "config": {
                    "hidden_dim": int((final_model_info or model_info or {}).get("hidden_dim", 128)),
                    "n_max": int(n_max),
                    "target_scale": float(args.target_scale),
                    "win_score_weight": float(args.win_score_weight),
                    "loss_score_weight": float(args.loss_score_weight),
                    "stage": "neural_three_head_oof_override_selector",
                },
                "model_info": final_model_info,
                "oof_best": best_row,
                "final_losses": final_losses,
            },
            final_checkpoint,
        )
        final_checkpoint_path = str(final_checkpoint)

    summary = {
        "stage": "run_neural_three_head_oof_calibration",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": final_checkpoint_path,
        "device": str(device),
        "groups": int(n_groups),
        "n_max": int(n_max),
        "folds": int(args.folds),
        "fold_sizes": [int(len(fold)) for fold in folds],
        "fold_summaries": fold_summaries,
        "model_info": final_model_info or model_info,
        "best_epoch": int(best_epoch),
        "best_thresholds": best_thresholds,
        "best_oof": best_row,
        "history": history,
        "args": vars(args),
    }
    _write_json(output_dir / "neural_three_head_oof_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run K-fold OOF calibration for the neural three-head override selector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--win-bce-weight", type=float, default=1.5)
    parser.add_argument("--loss-bce-weight", type=float, default=8.0)
    parser.add_argument("--delta-weight", type=float, default=0.25)
    parser.add_argument("--ce-weight", type=float, default=0.10)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--order-epsilon", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=4.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--win-score-weight", type=float, default=1.0)
    parser.add_argument("--loss-score-weight", type=float, default=3.0)
    parser.add_argument("--max-loss-rate", type=float, default=0.0)
    parser.add_argument("--max-override-rate", type=float, default=0.10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--min-overrides", type=int, default=1)
    parser.add_argument("--min-delta-floor", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--freeze-encoders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-trunk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-from-q-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = run(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
