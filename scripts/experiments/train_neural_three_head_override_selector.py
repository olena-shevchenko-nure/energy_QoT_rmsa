#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    _make_group_split,
    _resolve_cli_path,
    _resolve_path,
    _write_json,
)


def _load_initial_checkpoint(model: Any, checkpoint_path: Path | None, *, device: str, torch: Any) -> str | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return str(checkpoint_path)


def _build_three_head_model(
    *,
    config: ExperimentConfig,
    data: dict[str, np.ndarray],
    initial_checkpoint: Path | None,
    device: str,
    freeze_encoders: bool,
    freeze_trunk: bool,
    init_from_q_head: bool,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from cse2026.ong_solver.models import CandidateQNetwork

    if CandidateQNetwork is None:
        raise RuntimeError("CandidateQNetwork is unavailable because PyTorch is not installed")

    action_feature_dim = int(np.asarray(data["action_features"]).shape[-1])
    hidden_dim = int(config.resolved.get("hidden_dim", config.raw.get("hidden_dim", 128)))
    base = CandidateQNetwork(action_feature_dim=action_feature_dim, hidden_dim=hidden_dim)
    loaded = _load_initial_checkpoint(base, initial_checkpoint, device=device, torch=torch)

    class ThreeHeadCandidateJudge(torch.nn.Module):
        def __init__(self, source: Any) -> None:
            super().__init__()
            self.gnn = source.gnn
            self.slot_cnn = source.slot_cnn
            self.route_pool = source.route_pool
            self.request_encoder = source.request_encoder
            self.action_encoder = source.action_encoder
            fusion_dim = int(hidden_dim) * 3 + 64 + 64
            q_layers = list(source.q_head.children()) if bool(init_from_q_head) else []
            if len(q_layers) >= 7:
                self.trunk = torch.nn.Sequential(*[copy.deepcopy(layer) for layer in q_layers[:-1]])
                self.delta_head = copy.deepcopy(q_layers[-1])
            else:
                self.trunk = torch.nn.Sequential(
                    torch.nn.Linear(fusion_dim, 256),
                    torch.nn.LayerNorm(256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.10),
                    torch.nn.Linear(256, 128),
                    torch.nn.GELU(),
                )
                self.delta_head = torch.nn.Linear(128, 1)
            self.win_head = torch.nn.Linear(128, 1)
            self.loss_head = torch.nn.Linear(128, 1)

        def _route_embeddings(
            self,
            link_embeddings: Any,
            route_link_mask: Any,
            route_basic_features: Any,
        ) -> Any:
            mask = route_link_mask.unsqueeze(-1)
            denom = mask.sum(dim=2).clamp_min(1.0)
            mean_pool = (link_embeddings[:, None, :, :] * mask).sum(dim=2) / denom
            masked_links = link_embeddings[:, None, :, :].masked_fill(~route_link_mask.unsqueeze(-1).bool(), -1e9)
            max_pool = masked_links.max(dim=2).values
            max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
            return self.route_pool(torch.cat([mean_pool, max_pool, route_basic_features], dim=-1))

        def forward(
            self,
            *,
            node_features: Any,
            link_features: Any,
            global_features: Any,
            edge_index: Any,
            request_features: Any,
            spectrum_tensors: Any,
            action_features: Any,
            route_link_mask: Any,
            route_basic_features: Any,
            block_bounds: Any,
        ) -> dict[str, Any]:
            batch, n_max = action_features.shape[:2]
            h_global, h_links = self.gnn(node_features, link_features, global_features, edge_index)
            h_route = self._route_embeddings(h_links, route_link_mask, route_basic_features)
            h_req = self.request_encoder(request_features)[:, None, :].expand(-1, n_max, -1)
            h_action = self.action_encoder(action_features)
            h_block = self.slot_cnn(
                spectrum_tensors.reshape(batch * n_max, spectrum_tensors.shape[2], spectrum_tensors.shape[3]),
                block_bounds.reshape(batch * n_max, 2),
            ).reshape(batch, n_max, -1)
            h_global_rep = h_global[:, None, :].expand(-1, n_max, -1)
            fused = torch.cat([h_global_rep, h_route, h_block, h_req, h_action], dim=-1)
            hidden = self.trunk(fused)
            return {
                "win_logit": self.win_head(hidden).squeeze(-1),
                "loss_logit": self.loss_head(hidden).squeeze(-1),
                "delta": self.delta_head(hidden).squeeze(-1),
            }

    model = ThreeHeadCandidateJudge(base)
    if freeze_encoders:
        for module in (model.gnn, model.slot_cnn, model.route_pool, model.request_encoder, model.action_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if freeze_trunk:
        for parameter in model.trunk.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    return model, {
        "initial_checkpoint": loaded,
        "hidden_dim": int(hidden_dim),
        "action_feature_dim": int(action_feature_dim),
        "freeze_encoders": bool(freeze_encoders),
        "freeze_trunk": bool(freeze_trunk),
        "init_from_q_head": bool(init_from_q_head),
    }


def _forward_three_head(model: Any, tensors: dict[str, Any], edge_index: Any) -> dict[str, Any]:
    return model(
        node_features=tensors["node_features"],
        link_features=tensors["link_features"],
        global_features=tensors["global_features"],
        edge_index=edge_index,
        request_features=tensors["request_features"],
        spectrum_tensors=tensors["spectrum_tensors"],
        action_features=tensors["action_features"],
        route_link_mask=tensors["route_link_mask"],
        route_basic_features=tensors["route_basic_features"],
        block_bounds=tensors["block_bounds"],
    )


def _masked_bce_with_logits(logits: Any, labels: Any, mask: Any, *, max_pos_weight: float, torch: Any) -> Any:
    labeled_logits = logits.masked_select(mask)
    labeled = labels.to(dtype=logits.dtype).masked_select(mask)
    positives = labeled.sum()
    negatives = torch.clamp(labeled.numel() - positives, min=1.0)
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=float(max_pos_weight))
    return torch.nn.functional.binary_cross_entropy_with_logits(labeled_logits, labeled, pos_weight=pos_weight)


def _loss(
    *,
    outputs: dict[str, Any],
    tensors: dict[str, Any],
    win_bce_weight: float,
    loss_bce_weight: float,
    delta_weight: float,
    ce_weight: float,
    pairwise_weight: float,
    pairwise_margin: float,
    order_epsilon: float,
    target_scale: float,
    max_pos_weight: float,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    label_mask = tensors["label_mask"] & tensors["candidate_mask"]
    usable = label_mask.sum(dim=1) >= 2
    if not bool(usable.any()):
        raise RuntimeError("Batch has no usable labeled groups")

    accepted = tensors["accepted_delta_vs_base"]
    target = tensors["target_delta"]
    win_label = accepted > 0.0
    loss_label = accepted < 0.0
    win_bce = _masked_bce_with_logits(
        outputs["win_logit"],
        win_label,
        label_mask,
        max_pos_weight=float(max_pos_weight),
        torch=torch,
    )
    loss_bce = _masked_bce_with_logits(
        outputs["loss_logit"],
        loss_label,
        label_mask,
        max_pos_weight=float(max_pos_weight),
        torch=torch,
    )

    scaled_target = torch.clamp(target / max(float(target_scale), 1e-6), min=-3.0, max=3.0)
    delta_reg = torch.nn.functional.smooth_l1_loss(
        outputs["delta"].masked_select(label_mask),
        scaled_target.masked_select(label_mask),
    )

    masked_target = target.masked_fill(~label_mask, -1e9)
    best_index = masked_target.argmax(dim=1)
    delta_labeled = outputs["delta"].masked_fill(~label_mask, -1e9)
    ce = torch.nn.functional.cross_entropy(delta_labeled[usable], best_index[usable])
    pred = delta_labeled[usable].argmax(dim=1)
    top1 = (pred == best_index[usable]).to(dtype=torch.float32).mean()

    target_diff = target[:, :, None] - target[:, None, :]
    delta_diff = outputs["delta"][:, :, None] - outputs["delta"][:, None, :]
    pair_mask = label_mask[:, :, None] & label_mask[:, None, :] & (target_diff > float(order_epsilon))
    if bool(pair_mask.any()):
        pairwise = torch.nn.functional.relu(float(pairwise_margin) - delta_diff).masked_select(pair_mask).mean()
    else:
        pairwise = ce * 0.0

    total = (
        float(win_bce_weight) * win_bce
        + float(loss_bce_weight) * loss_bce
        + float(delta_weight) * delta_reg
        + float(ce_weight) * ce
        + float(pairwise_weight) * pairwise
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "win_bce": float(win_bce.detach().cpu()),
        "loss_bce": float(loss_bce.detach().cpu()),
        "delta_regression": float(delta_reg.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "top1_accuracy": float(top1.detach().cpu()),
        "usable_groups": int(usable.sum().detach().cpu()),
    }


def _candidate_score(
    *,
    win_prob: np.ndarray,
    loss_prob: np.ndarray,
    delta_pred: np.ndarray,
    win_score_weight: float,
    loss_score_weight: float,
) -> np.ndarray:
    return (
        np.asarray(delta_pred, dtype=np.float32)
        + float(win_score_weight) * np.asarray(win_prob, dtype=np.float32)
        - float(loss_score_weight) * np.asarray(loss_prob, dtype=np.float32)
    )


def _selection_metrics(
    *,
    predictions: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    win_threshold: float,
    loss_threshold: float,
    delta_margin: float,
    win_score_weight: float,
    loss_score_weight: float,
) -> dict[str, Any]:
    if len(indices) == 0:
        return {"groups": 0}
    candidate_mask = np.asarray(data["candidate_mask"])[indices].astype(bool)
    label_mask = np.asarray(data["label_mask"])[indices].astype(bool) & candidate_mask
    accepted = np.nan_to_num(np.asarray(data["accepted_delta_vs_base"])[indices].astype(np.float32), nan=0.0)
    target = np.nan_to_num(np.asarray(data["target_delta"])[indices].astype(np.float32), nan=0.0)
    base_index = np.asarray(data["base_index"])[indices].astype(np.int64)
    row = np.arange(len(indices))

    win_prob = np.asarray(predictions["win_prob"], dtype=np.float32)
    loss_prob = np.asarray(predictions["loss_prob"], dtype=np.float32)
    delta_pred = np.asarray(predictions["delta_pred"], dtype=np.float32)
    score = _candidate_score(
        win_prob=win_prob,
        loss_prob=loss_prob,
        delta_pred=delta_pred,
        win_score_weight=float(win_score_weight),
        loss_score_weight=float(loss_score_weight),
    )

    score_labeled = np.where(label_mask, score, -1.0e9)
    top1 = score_labeled.argmax(axis=1).astype(np.int64)
    top1_delta = accepted[row, top1]
    oracle = np.where(label_mask, target, -1.0e9).argmax(axis=1).astype(np.int64)
    oracle_delta = accepted[row, oracle]

    nonbase = np.ones_like(label_mask, dtype=bool)
    nonbase[row, np.clip(base_index, 0, label_mask.shape[1] - 1)] = False
    eligible = (
        label_mask
        & nonbase
        & (win_prob >= float(win_threshold))
        & (loss_prob <= float(loss_threshold))
        & (delta_pred >= float(delta_margin))
    )
    eligible_score = np.where(eligible, score, -1.0e9)
    selected = eligible_score.argmax(axis=1).astype(np.int64)
    override = eligible.any(axis=1)
    selected_delta = accepted[row, selected]
    override_delta = np.where(override, selected_delta, 0.0)
    override_count = int(override.sum())
    loss_count = int(((override_delta < 0.0) & override).sum())
    win_count = int(((override_delta > 0.0) & override).sum())
    selected_win_prob = win_prob[row, selected][override]
    selected_loss_prob = loss_prob[row, selected][override]
    selected_delta_pred = delta_pred[row, selected][override]
    selected_score = score[row, selected][override]
    return {
        "groups": int(len(indices)),
        "win_threshold": float(win_threshold),
        "loss_threshold": float(loss_threshold),
        "delta_margin": float(delta_margin),
        "top1_labeled_total_delta": float(top1_delta.sum()),
        "top1_labeled_mean_delta": float(top1_delta.mean()),
        "top1_labeled_win_rate": float((top1_delta > 0.0).mean()),
        "top1_labeled_loss_rate": float((top1_delta < 0.0).mean()),
        "oracle_total_delta": float(np.maximum(oracle_delta, 0.0).sum()),
        "oracle_if_always_best_delta": float(oracle_delta.sum()),
        "oracle_groups_with_win": int((oracle_delta > 0.0).sum()),
        "oracle_top1_accuracy": float((top1 == oracle).mean()),
        "override_count": int(override_count),
        "override_rate": float(override_count / max(len(indices), 1)),
        "selected_total_delta": float(override_delta.sum()),
        "selected_mean_delta": float(override_delta.mean()),
        "selected_win_count": int(win_count),
        "selected_loss_count": int(loss_count),
        "selected_win_rate": None if override_count == 0 else float(win_count / max(override_count, 1)),
        "selected_loss_rate": None if override_count == 0 else float(loss_count / max(override_count, 1)),
        "selected_mean_win_prob": None if override_count == 0 else float(selected_win_prob.mean()),
        "selected_mean_loss_prob": None if override_count == 0 else float(selected_loss_prob.mean()),
        "selected_mean_delta_pred": None if override_count == 0 else float(selected_delta_pred.mean()),
        "selected_mean_score": None if override_count == 0 else float(selected_score.mean()),
        "win_prob_quantiles": [float(np.quantile(win_prob[label_mask], q)) for q in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0)],
        "loss_prob_quantiles": [float(np.quantile(loss_prob[label_mask], q)) for q in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0)],
        "delta_pred_quantiles": [float(np.quantile(delta_pred[label_mask], q)) for q in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0)],
    }


def _predict(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    target_scale: float,
    torch: Any,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {"win_prob": [], "loss_prob": [], "delta_pred": []}
    model.eval()
    with torch.no_grad():
        for batch_indices in _iter_batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            outputs = _forward_three_head(model, tensors, edge_index)
            chunks["win_prob"].append(torch.sigmoid(outputs["win_logit"]).detach().cpu().numpy().astype(np.float32))
            chunks["loss_prob"].append(torch.sigmoid(outputs["loss_logit"]).detach().cpu().numpy().astype(np.float32))
            chunks["delta_pred"].append(
                (outputs["delta"] * float(target_scale)).detach().cpu().numpy().astype(np.float32)
            )
    shape = (0, int(np.asarray(data["candidate_mask"]).shape[1]))
    return {
        key: (np.concatenate(value, axis=0) if value else np.zeros(shape, dtype=np.float32))
        for key, value in chunks.items()
    }


def _grid(values: np.ndarray, fixed: list[float], quantiles: tuple[float, ...], *, lower: float | None = None) -> list[float]:
    result = {float(value) for value in fixed}
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        for q in quantiles:
            result.add(float(np.quantile(finite, q)))
    if lower is not None:
        result = {value for value in result if value >= float(lower)}
    return sorted(result)


def _tune_thresholds(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    target_scale: float,
    win_score_weight: float,
    loss_score_weight: float,
    max_loss_rate: float,
    max_override_rate: float,
    min_total_delta: float,
    min_overrides: int,
    min_delta_floor: float,
    torch: Any,
) -> tuple[dict[str, float], dict[str, Any]]:
    predictions = _predict(
        model=model,
        data=data,
        indices=indices,
        edge_index=edge_index,
        batch_size=batch_size,
        device=device,
        target_scale=float(target_scale),
        torch=torch,
    )
    if predictions["win_prob"].shape[0] == 0:
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
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98],
        (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    )
    loss_grid = _grid(
        loss_values,
        [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
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


def _evaluate_split(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    target_scale: float,
    thresholds: dict[str, float],
    win_score_weight: float,
    loss_score_weight: float,
    torch: Any,
) -> dict[str, Any]:
    predictions = _predict(
        model=model,
        data=data,
        indices=indices,
        edge_index=edge_index,
        batch_size=batch_size,
        device=device,
        target_scale=float(target_scale),
        torch=torch,
    )
    return _selection_metrics(
        predictions=predictions,
        data=data,
        indices=indices,
        win_threshold=float(thresholds["win_threshold"]),
        loss_threshold=float(thresholds["loss_threshold"]),
        delta_margin=float(thresholds["delta_margin"]),
        win_score_weight=float(win_score_weight),
        loss_score_weight=float(loss_score_weight),
    )


def _split_metric(row: dict[str, Any], split_name: str, key: str) -> Any:
    split = row.get("train_eval", {}) if split_name == "train" else row.get(split_name, {})
    return split.get(key)


def _compact_epoch(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": int(row.get("epoch", 0)),
        "thresholds": row.get("thresholds"),
        "train_selected_total_delta": _split_metric(row, "train", "selected_total_delta"),
        "train_selected_loss_rate": _split_metric(row, "train", "selected_loss_rate"),
        "train_override_count": _split_metric(row, "train", "override_count"),
        "calibration_selected_total_delta": _split_metric(row, "calibration", "selected_total_delta"),
        "calibration_selected_loss_rate": _split_metric(row, "calibration", "selected_loss_rate"),
        "calibration_override_count": _split_metric(row, "calibration", "override_count"),
        "calibration_constraints_satisfied": _split_metric(row, "calibration", "constraints_satisfied"),
        "eval_selected_total_delta": _split_metric(row, "eval", "selected_total_delta"),
        "eval_selected_loss_rate": _split_metric(row, "eval", "selected_loss_rate"),
        "eval_override_count": _split_metric(row, "eval", "override_count"),
        "eval_top1_labeled_total_delta": _split_metric(row, "eval", "top1_labeled_total_delta"),
    }


def _best_compact_epoch(
    history: list[dict[str, Any]],
    *,
    split_name: str,
    require_calibration_feasible: bool = False,
    require_zero_loss: bool = False,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in history:
        if require_calibration_feasible and not bool(_split_metric(row, "calibration", "constraints_satisfied")):
            continue
        loss_rate = _split_metric(row, split_name, "selected_loss_rate")
        if require_zero_loss and loss_rate not in (None, 0.0):
            continue
        candidates.append(row)
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            float(_split_metric(row, split_name, "selected_total_delta") or 0.0),
            -float(_split_metric(row, split_name, "selected_loss_rate") or 0.0),
            float(_split_metric(row, split_name, "override_count") or 0.0),
        ),
    )
    return _compact_epoch(best)


def train(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    torch.manual_seed(int(args.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    loaded = _load_dataset(Path(args.input_dir))
    data = loaded["neural"]
    metadata = loaded["metadata"]
    splits = _make_group_split(
        metadata=metadata,
        group_ids=np.asarray(data["group_ids"], dtype=np.int64),
        train_fraction=float(args.train_fraction),
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.seed),
    )
    initial_checkpoint = _resolve_cli_path(args.initial_checkpoint) or _resolve_path(config, "dqn_checkpoint")
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
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -math.inf
    best_epoch = 0
    best_path = output_dir / "neural_three_head_override_selector.pt"
    history: list[dict[str, Any]] = []
    initial_thresholds, initial_calibration = _tune_thresholds(
        model=model,
        data=data,
        indices=splits["calibration"],
        edge_index=edge_index,
        batch_size=int(args.batch_size),
        device=device,
        target_scale=float(args.target_scale),
        win_score_weight=float(args.win_score_weight),
        loss_score_weight=float(args.loss_score_weight),
        max_loss_rate=float(args.max_loss_rate),
        max_override_rate=float(args.max_override_rate),
        min_total_delta=float(args.min_total_delta),
        min_overrides=int(args.min_overrides),
        min_delta_floor=float(args.min_delta_floor),
        torch=torch,
    )
    initial_eval = _evaluate_split(
        model=model,
        data=data,
        indices=splits["eval"],
        edge_index=edge_index,
        batch_size=int(args.batch_size),
        device=device,
        target_scale=float(args.target_scale),
        thresholds=initial_thresholds,
        win_score_weight=float(args.win_score_weight),
        loss_score_weight=float(args.loss_score_weight),
        torch=torch,
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows: list[dict[str, float]] = []
        for batch_indices in _iter_batches(splits["train"], int(args.batch_size), shuffle=True, rng=rng):
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
            epoch_rows.append(parts)

        thresholds, calibration = _tune_thresholds(
            model=model,
            data=data,
            indices=splits["calibration"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            target_scale=float(args.target_scale),
            win_score_weight=float(args.win_score_weight),
            loss_score_weight=float(args.loss_score_weight),
            max_loss_rate=float(args.max_loss_rate),
            max_override_rate=float(args.max_override_rate),
            min_total_delta=float(args.min_total_delta),
            min_overrides=int(args.min_overrides),
            min_delta_floor=float(args.min_delta_floor),
            torch=torch,
        )
        train_eval = _evaluate_split(
            model=model,
            data=data,
            indices=splits["train"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            target_scale=float(args.target_scale),
            thresholds=thresholds,
            win_score_weight=float(args.win_score_weight),
            loss_score_weight=float(args.loss_score_weight),
            torch=torch,
        )
        eval_metrics = _evaluate_split(
            model=model,
            data=data,
            indices=splits["eval"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            target_scale=float(args.target_scale),
            thresholds=thresholds,
            win_score_weight=float(args.win_score_weight),
            loss_score_weight=float(args.loss_score_weight),
            torch=torch,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])) if epoch_rows else None,
            "train_win_bce": float(np.mean([item["win_bce"] for item in epoch_rows])) if epoch_rows else None,
            "train_loss_bce": float(np.mean([item["loss_bce"] for item in epoch_rows])) if epoch_rows else None,
            "train_delta_regression": float(np.mean([item["delta_regression"] for item in epoch_rows])) if epoch_rows else None,
            "train_ce": float(np.mean([item["ce"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise": float(np.mean([item["pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
            "thresholds": thresholds,
            "train_eval": train_eval,
            "calibration": calibration,
            "eval": eval_metrics,
        }
        history.append(row)
        print(json.dumps(_json_safe({"phase": "epoch", **row}), sort_keys=True), flush=True)

        score = float(calibration.get("selected_total_delta") or 0.0)
        if bool(calibration.get("constraints_satisfied")):
            score += 1000.0
        score -= 100.0 * float(calibration.get("selected_loss_rate") or 0.0)
        if score >= best_score:
            best_score = score
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                    "thresholds": thresholds,
                    "config": {
                        "hidden_dim": int(model_info["hidden_dim"]),
                        "n_max": int(np.asarray(data["candidate_mask"]).shape[1]),
                        "target_scale": float(args.target_scale),
                        "win_score_weight": float(args.win_score_weight),
                        "loss_score_weight": float(args.loss_score_weight),
                        "stage": "neural_three_head_override_selector",
                    },
                    "model_info": model_info,
                    "calibration": calibration,
                    "eval": eval_metrics,
                    "history": history,
                },
                best_path,
            )

    summary = {
        "stage": "train_neural_three_head_override_selector",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "model_info": model_info,
        "groups": int(len(data["group_ids"])),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "initial_thresholds": initial_thresholds,
        "initial_calibration": initial_calibration,
        "initial_eval": initial_eval,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_calibration": _best_compact_epoch(history, split_name="calibration"),
        "best_feasible_eval": _best_compact_epoch(
            history,
            split_name="eval",
            require_calibration_feasible=True,
        ),
        "best_zero_loss_eval": _best_compact_epoch(
            history,
            split_name="eval",
            require_zero_loss=True,
        ),
        "history": history,
        "args": vars(args),
    }
    _write_json(output_dir / "neural_three_head_override_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a three-head neural hard-case override selector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--win-bce-weight", type=float, default=2.0)
    parser.add_argument("--loss-bce-weight", type=float, default=3.0)
    parser.add_argument("--delta-weight", type=float, default=0.50)
    parser.add_argument("--ce-weight", type=float, default=0.25)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--order-epsilon", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=4.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--win-score-weight", type=float, default=1.0)
    parser.add_argument("--loss-score-weight", type=float, default=1.0)
    parser.add_argument("--max-loss-rate", type=float, default=0.05)
    parser.add_argument("--max-override-rate", type=float, default=0.20)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--min-overrides", type=int, default=1)
    parser.add_argument("--min-delta-floor", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--freeze-encoders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-trunk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-from-q-head", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = train(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
