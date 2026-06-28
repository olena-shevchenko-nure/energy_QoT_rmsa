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
from cse2026.experiments.eon.train_dqn import _device, _model_forward, _raw_bool


STATE_KEYS = (
    "node_features",
    "link_features",
    "global_features",
    "request_features",
    "spectrum_tensors",
    "action_features",
    "route_link_mask",
    "route_basic_features",
    "block_bounds",
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


def _resolve_path(config: ExperimentConfig, key: str) -> Path | None:
    value = config.resolved.get(key, config.raw.get(key))
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_cli_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def _load_dataset(input_dir: Path) -> dict[str, Any]:
    metadata_path = input_dir / "online_base_topn_examples.csv"
    neural_path = input_dir / "online_base_topn_neural_states.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not neural_path.exists():
        raise FileNotFoundError(f"Missing neural state file: {neural_path}")
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    npz = np.load(neural_path, allow_pickle=True)
    data = {str(key): np.asarray(npz[key]) for key in npz.files}
    group_ids = np.asarray(data["group_ids"], dtype=np.int64)
    if group_ids.ndim != 1 or group_ids.size == 0:
        raise ValueError(f"No neural groups found in {neural_path}")
    for key in STATE_KEYS + ("candidate_mask", "label_mask", "target_delta", "accepted_delta_vs_base"):
        if key not in data:
            raise ValueError(f"Neural state file is missing {key}")
        if int(data[key].shape[0]) != int(group_ids.size):
            raise ValueError(f"{key} first dimension does not match group_ids")
    if "label_weight" in data:
        if int(data["label_weight"].shape[0]) != int(group_ids.size):
            raise ValueError("label_weight first dimension does not match group_ids")
        if tuple(data["label_weight"].shape) != tuple(data["label_mask"].shape):
            raise ValueError("label_weight shape must match label_mask")
    return {"metadata": metadata, "neural": data}


def _make_group_split(
    *,
    metadata: pd.DataFrame,
    group_ids: np.ndarray,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    meta_by_group = {int(group_id): group for group_id, group in metadata.groupby("group_id", sort=False)}
    for position, group_id in enumerate(np.asarray(group_ids, dtype=np.int64)):
        group = meta_by_group.get(int(group_id))
        if group is None:
            bucket = "missing"
        else:
            nonbase = group[~group["is_base"].astype(bool)]
            max_delta = float(nonbase["accepted_delta_vs_base"].max()) if not nonbase.empty else 0.0
            min_delta = float(nonbase["accepted_delta_vs_base"].min()) if not nonbase.empty else 0.0
            bucket = "win_available" if max_delta > 0.0 else ("loss_only" if min_delta < 0.0 else "tie_only")
        rows.append({"position": int(position), "bucket": bucket})
    table = pd.DataFrame(rows)
    rng = np.random.default_rng(int(seed))
    split_by_position: dict[int, str] = {}
    for _, bucket in table.groupby("bucket", sort=False):
        values = bucket["position"].to_numpy(dtype=np.int64, copy=True)
        rng.shuffle(values)
        train_end = int(round(len(values) * float(train_fraction)))
        cal_end = train_end + int(round(len(values) * float(calibration_fraction)))
        for value in values[:train_end]:
            split_by_position[int(value)] = "train"
        for value in values[train_end:cal_end]:
            split_by_position[int(value)] = "calibration"
        for value in values[cal_end:]:
            split_by_position[int(value)] = "eval"
    split = np.asarray([split_by_position.get(int(index), "eval") for index in range(len(group_ids))], dtype=object)
    return {
        "train": np.flatnonzero(split == "train").astype(np.int64),
        "calibration": np.flatnonzero(split == "calibration").astype(np.int64),
        "eval": np.flatnonzero(split == "eval").astype(np.int64),
    }


def _batch_tensors(data: dict[str, np.ndarray], indices: np.ndarray, *, device: str, torch: Any) -> dict[str, Any]:
    tensors = {
        key: torch.as_tensor(np.asarray(data[key])[indices], dtype=torch.float32, device=device)
        for key in STATE_KEYS
    }
    tensors["candidate_mask"] = torch.as_tensor(np.asarray(data["candidate_mask"])[indices], dtype=torch.bool, device=device)
    tensors["label_mask"] = torch.as_tensor(np.asarray(data["label_mask"])[indices], dtype=torch.bool, device=device)
    tensors["target_delta"] = torch.as_tensor(
        np.nan_to_num(np.asarray(data["target_delta"])[indices], nan=0.0),
        dtype=torch.float32,
        device=device,
    )
    tensors["accepted_delta_vs_base"] = torch.as_tensor(
        np.nan_to_num(np.asarray(data["accepted_delta_vs_base"])[indices], nan=0.0),
        dtype=torch.float32,
        device=device,
    )
    label_weight = data.get("label_weight")
    if label_weight is None:
        label_weight_array = np.ones_like(np.asarray(data["label_mask"])[indices], dtype=np.float32)
    else:
        label_weight_array = np.nan_to_num(np.asarray(label_weight)[indices].astype(np.float32), nan=0.0)
    tensors["label_weight"] = torch.as_tensor(label_weight_array, dtype=torch.float32, device=device)
    tensors["base_index"] = torch.as_tensor(np.asarray(data["base_index"])[indices], dtype=torch.long, device=device)
    tensors["group_ids"] = np.asarray(data["group_ids"])[indices].astype(np.int64)
    return tensors


def _iter_batches(indices: np.ndarray, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        rng.shuffle(values)
    return [values[start : start + int(batch_size)] for start in range(0, len(values), int(batch_size))]


def _load_initial_checkpoint(model: Any, checkpoint_path: Path | None, *, device: str, torch: Any) -> str | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    return str(checkpoint_path)


def _build_model(
    *,
    config: ExperimentConfig,
    data: dict[str, np.ndarray],
    initial_checkpoint: Path | None,
    device: str,
    freeze_gnn_slot: bool,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from cse2026.ong_solver.models import CandidateQNetwork

    if CandidateQNetwork is None:
        raise RuntimeError("CandidateQNetwork is unavailable because PyTorch is not installed")
    action_feature_dim = int(np.asarray(data["action_features"]).shape[-1])
    hidden_dim = int(config.resolved.get("hidden_dim", config.raw.get("hidden_dim", 128)))
    model = CandidateQNetwork(action_feature_dim=action_feature_dim, hidden_dim=hidden_dim)
    loaded = _load_initial_checkpoint(model, initial_checkpoint, device=device, torch=torch)
    if freeze_gnn_slot:
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
        for parameter in model.slot_cnn.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    return model, {
        "initial_checkpoint": loaded,
        "hidden_dim": int(hidden_dim),
        "action_feature_dim": int(action_feature_dim),
        "freeze_gnn_slot": bool(freeze_gnn_slot),
    }


def _loss(
    *,
    logits: Any,
    reference_logits: Any | None,
    tensors: dict[str, Any],
    ce_weight: float,
    bce_weight: float,
    pairwise_weight: float,
    regression_weight: float,
    kl_weight: float,
    kl_temperature: float,
    pairwise_margin: float,
    order_epsilon: float,
    target_scale: float,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    label_mask = tensors["label_mask"] & tensors["candidate_mask"]
    target = tensors["target_delta"]
    usable = label_mask.sum(dim=1) >= 2
    if not bool(usable.any()):
        raise RuntimeError("Batch has no usable labeled groups")

    masked_target = target.masked_fill(~label_mask, -1e9)
    best_index = masked_target.argmax(dim=1)
    logits_labeled = logits.masked_fill(~label_mask, -1e9)
    ce = torch.nn.functional.cross_entropy(logits_labeled[usable], best_index[usable])
    pred = logits_labeled[usable].argmax(dim=1)
    top1 = (pred == best_index[usable]).to(dtype=torch.float32).mean()

    target_diff = target[:, :, None] - target[:, None, :]
    logit_diff = logits[:, :, None] - logits[:, None, :]
    pair_mask = label_mask[:, :, None] & label_mask[:, None, :] & (target_diff > float(order_epsilon))
    if bool(pair_mask.any()):
        pair = torch.nn.functional.relu(float(pairwise_margin) - logit_diff).masked_select(pair_mask).mean()
    else:
        pair = ce * 0.0

    scaled_target = torch.clamp(target / max(float(target_scale), 1e-6), min=-3.0, max=3.0)
    reg = torch.nn.functional.smooth_l1_loss(logits.masked_select(label_mask), scaled_target.masked_select(label_mask))
    win_label = (tensors["accepted_delta_vs_base"] > 0.0).to(dtype=logits.dtype)
    labeled_logits = logits.masked_select(label_mask)
    labeled_win = win_label.masked_select(label_mask)
    positives = labeled_win.sum()
    negatives = torch.clamp(labeled_win.numel() - positives, min=1.0)
    pos_weight = torch.clamp(negatives / torch.clamp(positives, min=1.0), min=1.0, max=20.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(labeled_logits, labeled_win, pos_weight=pos_weight)
    kl = ce * 0.0
    if reference_logits is not None and float(kl_weight) > 0.0:
        candidate_mask = tensors["candidate_mask"]
        temperature = max(float(kl_temperature), 1.0e-6)
        student_log_prob = torch.nn.functional.log_softmax(
            logits.masked_fill(~candidate_mask, -1.0e9) / temperature,
            dim=1,
        )
        with torch.no_grad():
            reference_prob = torch.nn.functional.softmax(
                reference_logits.masked_fill(~candidate_mask, -1.0e9) / temperature,
                dim=1,
            )
        kl = torch.nn.functional.kl_div(student_log_prob, reference_prob, reduction="batchmean") * (temperature * temperature)
    total = (
        float(ce_weight) * ce
        + float(bce_weight) * bce
        + float(pairwise_weight) * pair
        + float(regression_weight) * reg
        + float(kl_weight) * kl
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "pairwise": float(pair.detach().cpu()),
        "regression": float(reg.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "top1_accuracy": float(top1.detach().cpu()),
        "usable_groups": int(usable.sum().detach().cpu()),
    }


def _selection_metrics(
    *,
    logits_np: np.ndarray,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    margin: float,
) -> dict[str, Any]:
    candidate_mask = np.asarray(data["candidate_mask"])[indices].astype(bool)
    label_mask = np.asarray(data["label_mask"])[indices].astype(bool) & candidate_mask
    accepted = np.nan_to_num(np.asarray(data["accepted_delta_vs_base"])[indices].astype(np.float32), nan=0.0)
    target = np.nan_to_num(np.asarray(data["target_delta"])[indices].astype(np.float32), nan=0.0)
    base_index = np.asarray(data["base_index"])[indices].astype(np.int64)
    if len(indices) == 0:
        return {"groups": 0}

    labeled_logits = np.where(label_mask, logits_np, -1.0e9)
    selected = labeled_logits.argmax(axis=1).astype(np.int64)
    oracle = np.where(label_mask, target, -1.0e9).argmax(axis=1).astype(np.int64)
    selected_delta = accepted[np.arange(len(indices)), selected]
    oracle_delta = accepted[np.arange(len(indices)), oracle]
    base_logits = logits_np[np.arange(len(indices)), np.clip(base_index, 0, logits_np.shape[1] - 1)]
    selected_margin = logits_np[np.arange(len(indices)), selected] - base_logits
    override = (selected != base_index) & (selected_margin >= float(margin))
    override_delta = np.where(override, selected_delta, 0.0)
    override_count = int(override.sum())
    loss_count = int(((override_delta < 0.0) & override).sum())
    win_count = int(((override_delta > 0.0) & override).sum())
    return {
        "groups": int(len(indices)),
        "margin": float(margin),
        "top1_labeled_total_delta": float(selected_delta.sum()),
        "top1_labeled_mean_delta": float(selected_delta.mean()),
        "top1_labeled_win_rate": float((selected_delta > 0.0).mean()),
        "top1_labeled_loss_rate": float((selected_delta < 0.0).mean()),
        "oracle_total_delta": float(np.maximum(oracle_delta, 0.0).sum()),
        "oracle_if_always_best_delta": float(oracle_delta.sum()),
        "oracle_groups_with_win": int((oracle_delta > 0.0).sum()),
        "oracle_top1_accuracy": float((selected == oracle).mean()),
        "override_count": int(override_count),
        "override_rate": float(override_count / max(len(indices), 1)),
        "selected_total_delta": float(override_delta.sum()),
        "selected_mean_delta": float(override_delta.mean()),
        "selected_win_count": int(win_count),
        "selected_loss_count": int(loss_count),
        "selected_win_rate": None if override_count == 0 else float(win_count / max(override_count, 1)),
        "selected_loss_rate": None if override_count == 0 else float(loss_count / max(override_count, 1)),
        "selected_margin_quantiles": [float(np.quantile(selected_margin, q)) for q in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0)],
    }


def _predict_logits(
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
        for batch_indices in _iter_batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            logits = _model_forward(model, tensors, edge_index)
            chunks.append(logits.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, int(np.asarray(data["candidate_mask"]).shape[1])), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _evaluate_split(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    margin: float,
    batch_size: int,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    logits = _predict_logits(
        model=model,
        data=data,
        indices=indices,
        edge_index=edge_index,
        batch_size=batch_size,
        device=device,
        torch=torch,
    )
    return _selection_metrics(logits_np=logits, data=data, indices=indices, margin=float(margin))


def _tune_margin(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    max_loss_rate: float,
    max_override_rate: float,
    min_total_delta: float,
    torch: Any,
) -> tuple[float, dict[str, Any]]:
    logits = _predict_logits(
        model=model,
        data=data,
        indices=indices,
        edge_index=edge_index,
        batch_size=batch_size,
        device=device,
        torch=torch,
    )
    if logits.shape[0] == 0:
        return 0.0, {"groups": 0, "constraints_satisfied": False}
    candidate_mask = np.asarray(data["candidate_mask"])[indices].astype(bool)
    label_mask = np.asarray(data["label_mask"])[indices].astype(bool) & candidate_mask
    base_index = np.asarray(data["base_index"])[indices].astype(np.int64)
    selected = np.where(label_mask, logits, -1.0e9).argmax(axis=1)
    base_logits = logits[np.arange(len(indices)), np.clip(base_index, 0, logits.shape[1] - 1)]
    margins = logits[np.arange(len(indices)), selected] - base_logits
    grid = {0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.0}
    finite = margins[np.isfinite(margins)]
    if finite.size:
        for q in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
            grid.add(float(np.quantile(finite, q)))
    best_margin = 0.0
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_margin = 0.0
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    for margin in sorted(grid):
        metrics = _selection_metrics(logits_np=logits, data=data, indices=indices, margin=float(margin))
        loss_rate = metrics.get("selected_loss_rate")
        loss_value = 0.0 if loss_rate is None else float(loss_rate)
        total_delta = float(metrics.get("selected_total_delta") or 0.0)
        override_rate = float(metrics.get("override_rate") or 0.0)
        key = (total_delta, -loss_value, float(metrics.get("override_count") or 0))
        if fallback_key is None or key > fallback_key:
            fallback_key = key
            fallback_margin = float(margin)
            fallback_metrics = dict(metrics)
        if total_delta < float(min_total_delta):
            continue
        if loss_value > float(max_loss_rate):
            continue
        if override_rate > float(max_override_rate):
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_margin = float(margin)
            best_metrics = dict(metrics)
    if best_metrics is None:
        assert fallback_metrics is not None
        fallback_metrics["constraints_satisfied"] = False
        fallback_metrics["tune_found_feasible"] = False
        return float(fallback_margin), fallback_metrics
    best_metrics["constraints_satisfied"] = True
    best_metrics["tune_found_feasible"] = True
    return float(best_margin), best_metrics


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
    model, model_info = _build_model(
        config=config,
        data=data,
        initial_checkpoint=initial_checkpoint,
        device=device,
        freeze_gnn_slot=bool(args.freeze_gnn_slot),
        torch=torch,
    )
    reference_checkpoint = _resolve_cli_path(args.reference_checkpoint)
    reference_model = None
    if reference_checkpoint is not None and float(args.kl_weight) > 0.0:
        reference_model, reference_info = _build_model(
            config=config,
            data=data,
            initial_checkpoint=reference_checkpoint,
            device=device,
            freeze_gnn_slot=True,
            torch=torch,
        )
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()
        model_info["reference_checkpoint"] = str(reference_checkpoint)
        model_info["reference_model_info"] = reference_info
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable model parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 0
    best_path = output_dir / "neural_stable_override_selector.pt"

    initial_margin, initial_calibration = _tune_margin(
        model=model,
        data=data,
        indices=splits["calibration"],
        edge_index=edge_index,
        batch_size=int(args.batch_size),
        device=device,
        max_loss_rate=float(args.max_loss_rate),
        max_override_rate=float(args.max_override_rate),
        min_total_delta=float(args.min_total_delta),
        torch=torch,
    )
    initial_eval = _evaluate_split(
        model=model,
        data=data,
        indices=splits["eval"],
        edge_index=edge_index,
        margin=float(initial_margin),
        batch_size=int(args.batch_size),
        device=device,
        torch=torch,
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows: list[dict[str, float]] = []
        for batch_indices in _iter_batches(splits["train"], int(args.batch_size), shuffle=True, rng=rng):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            logits = _model_forward(model, tensors, edge_index)
            reference_logits = None
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits = _model_forward(reference_model, tensors, edge_index)
            loss, parts = _loss(
                logits=logits,
                reference_logits=reference_logits,
                tensors=tensors,
                ce_weight=float(args.ce_weight),
                bce_weight=float(args.bce_weight),
                pairwise_weight=float(args.pairwise_weight),
                regression_weight=float(args.regression_weight),
                kl_weight=float(args.kl_weight),
                kl_temperature=float(args.kl_temperature),
                pairwise_margin=float(args.pairwise_margin),
                order_epsilon=float(args.order_epsilon),
                target_scale=float(args.target_scale),
                torch=torch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
            optimizer.step()
            epoch_rows.append(parts)

        margin, calibration = _tune_margin(
            model=model,
            data=data,
            indices=splits["calibration"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            max_loss_rate=float(args.max_loss_rate),
            max_override_rate=float(args.max_override_rate),
            min_total_delta=float(args.min_total_delta),
            torch=torch,
        )
        train_eval = _evaluate_split(
            model=model,
            data=data,
            indices=splits["train"],
            edge_index=edge_index,
            margin=float(margin),
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        eval_metrics = _evaluate_split(
            model=model,
            data=data,
            indices=splits["eval"],
            edge_index=edge_index,
            margin=float(margin),
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])) if epoch_rows else None,
            "train_ce": float(np.mean([item["ce"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise": float(np.mean([item["pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_bce": float(np.mean([item["bce"] for item in epoch_rows])) if epoch_rows else None,
            "train_regression": float(np.mean([item["regression"] for item in epoch_rows])) if epoch_rows else None,
            "train_kl": float(np.mean([item["kl"] for item in epoch_rows])) if epoch_rows else None,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
            "margin": float(margin),
            "train_eval": train_eval,
            "calibration": calibration,
            "eval": eval_metrics,
        }
        history.append(row)
        print(json.dumps(_json_safe({"phase": "epoch", **row}), sort_keys=True), flush=True)

        score = float(calibration.get("selected_total_delta") or 0.0)
        if bool(calibration.get("constraints_satisfied")):
            score += 1000.0
        if score >= best_score:
            best_score = score
            best_epoch = int(epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                    "config": {
                        "hidden_dim": int(model_info["hidden_dim"]),
                        "n_max": int(np.asarray(data["candidate_mask"]).shape[1]),
                        "q_score_mode": "raw",
                        "residual_scale": 1.0,
                        "residual_delta_clip": 0.0,
                        "override_margin": float(margin),
                        "stage": "neural_stable_override_selector",
                    },
                    "model_info": model_info,
                    "margin": float(margin),
                    "calibration": calibration,
                    "eval": eval_metrics,
                    "history": history,
                },
                best_path,
            )

    summary = {
        "stage": "train_neural_stable_override_selector",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "model_info": model_info,
        "groups": int(len(data["group_ids"])),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "initial_margin": float(initial_margin),
        "initial_calibration": initial_calibration,
        "initial_eval": initial_eval,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "history": history,
        "args": vars(args),
    }
    _write_json(output_dir / "neural_stable_override_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural stable hard-case override selector on counterfactual labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--regression-weight", type=float, default=0.15)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.25)
    parser.add_argument("--order-epsilon", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=4.0)
    parser.add_argument("--max-loss-rate", type=float, default=0.05)
    parser.add_argument("--max-override-rate", type=float, default=0.20)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--freeze-gnn-slot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-checkpoint", default="")
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = train(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
