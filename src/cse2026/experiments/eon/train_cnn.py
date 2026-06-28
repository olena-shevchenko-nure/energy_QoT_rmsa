from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ExperimentConfig


REGRESSION_LABELS = (
    "delta_frag",
    "frag_after",
    "lmax_after_norm",
    "nseg_after_norm",
    "compactness",
    "placement_score",
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits(config: ExperimentConfig) -> tuple[str, str, str]:
    values = tuple(config.splits or ("train", "val", "test"))
    train = values[0] if len(values) >= 1 else "train"
    val = values[1] if len(values) >= 2 else "val"
    test = values[2] if len(values) >= 3 else "test"
    return train, val, test


def _device(config: ExperimentConfig, torch: Any) -> str:
    requested = str(config.resolved.get("device", config.device))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _raw_int(config: ExperimentConfig, key: str, default: int) -> int:
    return int(config.resolved.get(key, config.raw.get(key, default)))


def _raw_float(config: ExperimentConfig, key: str, default: float) -> float:
    return float(config.resolved.get(key, config.raw.get(key, default)))


def _load_split(dataset_path: Path, split: str) -> dict[str, Any]:
    tensors = np.load(dataset_path / "cnn" / f"{split}_tensors.npz")["X_spec"].astype(np.float32)
    index = pd.read_parquet(dataset_path / "cnn" / f"{split}_index.parquet")
    labels = index[list(REGRESSION_LABELS)].to_numpy(dtype=np.float32)
    small_gap = index["created_small_gap"].to_numpy(dtype=np.float32)
    block_bounds = index[["b_start", "w"]].to_numpy(dtype=np.float32)
    group_ids = index["group_id"].astype(str).to_numpy()
    rank_targets = (-index["J_total"].to_numpy(dtype=np.float32)).astype(np.float32)
    return {
        "x": tensors,
        "labels": labels,
        "small_gap": small_gap,
        "block_bounds": block_bounds,
        "group_ids": group_ids,
        "rank_targets": rank_targets,
    }


def _iter_batches(size: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def _ranking_loss(rank_scores: Any, group_ids: np.ndarray, targets: Any, torch: Any) -> Any:
    losses = []
    for group_id in np.unique(group_ids):
        positions_np = np.flatnonzero(group_ids == group_id)
        if positions_np.size < 2:
            continue
        positions = torch.as_tensor(positions_np, dtype=torch.long, device=rank_scores.device)
        group_scores = rank_scores.index_select(0, positions)
        group_targets = targets.index_select(0, positions)
        diff_target = group_targets[:, None] - group_targets[None, :]
        good = diff_target > 1e-6
        if not bool(good.any()):
            continue
        diff_pred = group_scores[:, None] - group_scores[None, :]
        losses.append(torch.nn.functional.softplus(-diff_pred[good]).mean())
    if not losses:
        return rank_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


def _classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = probs >= 0.5
    truth = labels.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    total = max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _ranking_metrics(scores: np.ndarray, targets: np.ndarray, group_ids: np.ndarray) -> dict[str, float | None]:
    pair_correct = 0
    pair_total = 0
    top1 = 0
    top3 = 0
    groups = 0
    for group_id in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == group_id)
        if idx.size < 2:
            continue
        groups += 1
        group_scores = scores[idx]
        group_targets = targets[idx]
        best = int(np.argmax(group_targets))
        order = np.argsort(-group_scores)
        top1 += int(order[0] == best)
        top3 += int(best in set(int(x) for x in order[: min(3, len(order))]))
        for left in range(idx.size):
            for right in range(left + 1, idx.size):
                diff = float(group_targets[left] - group_targets[right])
                if abs(diff) <= 1e-6:
                    continue
                pair_total += 1
                pred_diff = float(group_scores[left] - group_scores[right])
                pair_correct += int((diff > 0 and pred_diff > 0) or (diff < 0 and pred_diff < 0))
    return {
        "groups": int(groups),
        "pairwise_accuracy": None if pair_total == 0 else float(pair_correct / pair_total),
        "top1_accuracy": None if groups == 0 else float(top1 / groups),
        "top3_accuracy": None if groups == 0 else float(top3 / groups),
    }


def _build_model(hidden_dim: int):
    from cse2026.ong_solver.models import SlotCNNEncoder, require_torch

    torch = require_torch()
    nn = torch.nn

    class CnnPretrainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = SlotCNNEncoder(out_dim=hidden_dim)
            self.regression_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, len(REGRESSION_LABELS)),
            )
            self.small_gap_head = nn.Linear(hidden_dim, 1)
            self.rank_head = nn.Linear(hidden_dim, 1)

        def forward(self, x, block_bounds):
            embedding = self.encoder(x, block_bounds)
            return {
                "embedding": embedding,
                "regression": self.regression_head(embedding),
                "small_gap_logit": self.small_gap_head(embedding).squeeze(-1),
                "rank_score": self.rank_head(embedding).squeeze(-1),
            }

    return torch, CnnPretrainer()


def _evaluate(
    model: Any,
    data: dict[str, Any],
    batch_size: int,
    device: str,
    torch: Any,
    *,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    preds = []
    gap_logits = []
    rank_scores = []
    batches = _iter_batches(len(data["x"]), batch_size, shuffle=False, rng=np.random.default_rng(0))
    if max_batches > 0:
        batches = batches[:max_batches]
    eval_idx = np.concatenate(batches, axis=0) if batches else np.asarray([], dtype=np.int64)
    with torch.no_grad():
        for idx in batches:
            x = torch.as_tensor(data["x"][idx], dtype=torch.float32, device=device)
            bounds = torch.as_tensor(data["block_bounds"][idx], dtype=torch.float32, device=device)
            out = model(x, bounds)
            preds.append(out["regression"].detach().cpu().numpy())
            gap_logits.append(out["small_gap_logit"].detach().cpu().numpy())
            rank_scores.append(out["rank_score"].detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = data["labels"][eval_idx]
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    metrics: dict[str, Any] = {
        "samples": int(len(y_true)),
        "regression_mae": {name: float(value) for name, value in zip(REGRESSION_LABELS, mae)},
    }
    gap_values = np.concatenate(gap_logits, axis=0)
    metrics["small_gap"] = _classification_metrics(gap_values, data["small_gap"][eval_idx])
    rank_values = np.concatenate(rank_scores, axis=0)
    metrics["ranking"] = _ranking_metrics(rank_values, data["rank_targets"][eval_idx], data["group_ids"][eval_idx])
    return metrics


def run_pretrain_cnn(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("pretrain_cnn requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    torch, model = _build_model(hidden_dim=_raw_int(config, "hidden_dim", 128))
    device = _device(config, torch)
    model.to(device)
    train_split, val_split, test_split = _splits(config)
    train = _load_split(config.dataset_path, train_split)
    val = _load_split(config.dataset_path, val_split)
    test = _load_split(config.dataset_path, test_split)

    epochs = _raw_int(config, "epochs", 12)
    patience = _raw_int(config, "patience", 4)
    lr = _raw_float(config, "learning_rate", 1e-3)
    batch_size = int(config.batch_size)
    max_batches = int(config.max_batches)
    eval_max_batches = _raw_int(config, "eval_max_batches", 0)
    progress_every_batches = _raw_int(config, "progress_every_batches", 0)
    ranking_loss_weight = _raw_float(config, "ranking_loss_weight", 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=_raw_float(config, "weight_decay", 1e-4))
    huber = torch.nn.SmoothL1Loss()
    bce = torch.nn.BCEWithLogitsLoss()
    reg_weights = torch.as_tensor([1.0, 0.5, 0.5, 0.2, 0.3, 0.5], dtype=torch.float32, device=device)
    rng = np.random.default_rng(config.seed)

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    stale = 0
    best_path = run_path / "cnn_encoder_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        batches = _iter_batches(len(train["x"]), batch_size, shuffle=True, rng=rng)
        if max_batches > 0:
            batches = batches[:max_batches]
        losses = []
        for batch_index, idx in enumerate(batches, start=1):
            x = torch.as_tensor(train["x"][idx], dtype=torch.float32, device=device)
            bounds = torch.as_tensor(train["block_bounds"][idx], dtype=torch.float32, device=device)
            labels = torch.as_tensor(train["labels"][idx], dtype=torch.float32, device=device)
            gaps = torch.as_tensor(train["small_gap"][idx], dtype=torch.float32, device=device)
            ranks = torch.as_tensor(train["rank_targets"][idx], dtype=torch.float32, device=device)
            out = model(x, bounds)
            reg_loss = (torch.nn.functional.smooth_l1_loss(out["regression"], labels, reduction="none") * reg_weights).mean()
            gap_loss = bce(out["small_gap_logit"], gaps)
            if ranking_loss_weight > 0.0:
                rank_loss = _ranking_loss(out["rank_score"], train["group_ids"][idx], ranks, torch)
            else:
                rank_loss = out["rank_score"].new_tensor(0.0)
            loss = reg_loss + gap_loss + float(ranking_loss_weight) * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(batches),
                            "train_loss_so_far": float(np.mean(losses)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        val_metrics = _evaluate(model, val, batch_size, device, torch, max_batches=eval_max_batches)
        val_score = (
            val_metrics["regression_mae"]["delta_frag"]
            + val_metrics["regression_mae"]["frag_after"]
            + val_metrics["regression_mae"]["lmax_after_norm"]
            + (1.0 - float(val_metrics["small_gap"]["f1"]))
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_score": float(val_score),
            "val": val_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if val_score < best_val:
            best_val = float(val_score)
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "epoch": epoch,
                    "val_score": best_val,
                    "config": config.resolved,
                    "regression_labels": list(REGRESSION_LABELS),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "stage": "pretrain_cnn",
        "dataset_path": str(config.dataset_path),
        "device": device,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "train_samples": int(len(train["x"])),
        "eval_max_batches": int(eval_max_batches),
        "ranking_loss_weight": float(ranking_loss_weight),
        "val": _evaluate(model, val, batch_size, device, torch, max_batches=eval_max_batches),
        "test": _evaluate(model, test, batch_size, device, torch, max_batches=eval_max_batches),
        "history": history,
    }
    _write_json(run_path / "metrics.json", final_metrics)
    return final_metrics
