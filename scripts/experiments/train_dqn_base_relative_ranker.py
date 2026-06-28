from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _target_from_metadata(
    metadata: pd.DataFrame,
    *,
    secondary_scale: float,
    target_clip: float,
) -> np.ndarray:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    secondary = metadata.get("secondary_delta_vs_base", pd.Series(np.zeros((len(metadata),), dtype=np.float32))).to_numpy(
        dtype=np.float32
    )
    target = np.where(accepted != 0.0, accepted, float(secondary_scale) * secondary).astype(np.float32)
    is_base = metadata["candidate_index"].astype(int).to_numpy() == metadata["base_index"].astype(int).to_numpy()
    target[is_base] = 0.0
    return np.clip(target, -float(target_clip), float(target_clip)).astype(np.float32)


def _sample_weights(
    metadata: pd.DataFrame,
    target: np.ndarray,
    *,
    base_weight: float,
    win_weight: float,
    loss_weight: float,
    tie_weight: float,
    secondary_weight: float,
) -> np.ndarray:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    is_base = metadata["candidate_index"].astype(int).to_numpy() == metadata["base_index"].astype(int).to_numpy()
    weights = np.full((len(metadata),), float(tie_weight), dtype=np.float32)
    weights = np.where(accepted > 0.0, float(win_weight), weights).astype(np.float32)
    weights = np.where(accepted < 0.0, float(loss_weight), weights).astype(np.float32)
    weights = np.where((accepted == 0.0) & (np.abs(target) > 1.0e-6), float(secondary_weight), weights).astype(np.float32)
    weights = np.where(is_base, float(base_weight), weights).astype(np.float32)
    return (weights / max(float(np.mean(weights)), 1.0e-6)).astype(np.float32)


def _pair_indices(
    metadata: pd.DataFrame,
    target: np.ndarray,
    *,
    max_pairs_per_group: int,
    pair_target_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    diff_targets: list[float] = []
    weights: list[float] = []
    metadata = metadata.reset_index(drop=True)
    for _, group in metadata.groupby("group_id", sort=False):
        positions = np.asarray(group.index.to_numpy(), dtype=np.int64)
        if positions.size < 2:
            continue
        base_index = int(group["base_index"].iloc[0])
        candidate_indices = group["candidate_index"].astype(int).to_numpy()
        base_locs = np.flatnonzero(candidate_indices == base_index)
        if base_locs.size == 0:
            continue
        base_pos = int(positions[int(base_locs[0])])
        group_targets = target[positions]
        local_order = np.argsort(group_targets, kind="mergesort")
        pair_budget = max(1, int(max_pairs_per_group))

        # Anchor every clearly better/worse candidate against base first.
        anchored = 0
        for local in local_order[::-1]:
            pos = int(positions[int(local)])
            diff = float(target[pos] - target[base_pos])
            if diff <= 1.0e-6:
                continue
            left.append(pos)
            right.append(base_pos)
            diff_targets.append(min(float(pair_target_clip), diff))
            weights.append(2.0 + min(4.0, abs(diff)))
            anchored += 1
            if anchored >= pair_budget:
                break
        anchored = 0
        for local in local_order:
            pos = int(positions[int(local)])
            diff = float(target[base_pos] - target[pos])
            if diff <= 1.0e-6:
                continue
            left.append(base_pos)
            right.append(pos)
            diff_targets.append(min(float(pair_target_clip), diff))
            weights.append(2.0 + min(4.0, abs(diff)))
            anchored += 1
            if anchored >= pair_budget:
                break

        # Add the strongest intra-group contrast to improve ordering beyond base.
        best_pos = int(positions[int(local_order[-1])])
        worst_pos = int(positions[int(local_order[0])])
        diff = float(target[best_pos] - target[worst_pos])
        if diff > 1.0e-6 and best_pos != worst_pos:
            left.append(best_pos)
            right.append(worst_pos)
            diff_targets.append(min(float(pair_target_clip), diff))
            weights.append(1.0 + min(4.0, abs(diff)))

    if not left:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float32)
        return empty_i, empty_i, empty_f, empty_f
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(diff_targets, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def _batch_indices(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(int(size), dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + int(batch_size)] for start in range(0, int(size), int(batch_size))]


def _train_model(
    *,
    torch: Any,
    train_module: Any,
    x: np.ndarray,
    target: np.ndarray,
    row_weight: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_target: np.ndarray,
    pair_weight: np.ndarray,
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
    seed: int,
    device: str,
) -> Any:
    torch.manual_seed(int(seed))
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    model = train_module.TabularDqnPredictor.build(
        torch=torch,
        input_dim=int(x.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
        activation=0,
    )
    model.set_normalizer(
        torch.as_tensor(mean, dtype=torch.float32),
        torch.as_tensor(std, dtype=torch.float32),
    )
    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(requested_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    x_t = torch.as_tensor(x.astype(np.float32), dtype=torch.float32, device=requested_device)
    target_t = torch.as_tensor(target.astype(np.float32), dtype=torch.float32, device=requested_device)
    row_weight_t = torch.as_tensor(row_weight.astype(np.float32), dtype=torch.float32, device=requested_device)
    pair_left_t = torch.as_tensor(pair_left.astype(np.int64), dtype=torch.long, device=requested_device)
    pair_right_t = torch.as_tensor(pair_right.astype(np.int64), dtype=torch.long, device=requested_device)
    pair_target_t = torch.as_tensor(pair_target.astype(np.float32), dtype=torch.float32, device=requested_device)
    pair_weight_t = torch.as_tensor(pair_weight.astype(np.float32), dtype=torch.float32, device=requested_device)
    rng = np.random.default_rng(int(seed))

    for epoch in range(1, int(epochs) + 1):
        model.train()
        row_losses: list[float] = []
        pair_losses: list[float] = []
        rank_losses: list[float] = []
        row_batches = _batch_indices(len(x), batch_size, rng)
        pair_batches = _batch_indices(len(pair_left), pair_batch_size, rng) if len(pair_left) else []
        max_batches = max(len(row_batches), len(pair_batches), 1)
        for step in range(max_batches):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=torch.float32, device=requested_device)
            if row_batches:
                rb = row_batches[step % len(row_batches)]
                ridx = torch.as_tensor(rb, dtype=torch.long, device=requested_device)
                pred = model(x_t.index_select(0, ridx))
                row_loss = torch.nn.functional.smooth_l1_loss(pred, target_t.index_select(0, ridx), reduction="none")
                row_loss = (row_loss * row_weight_t.index_select(0, ridx)).mean()
                loss = loss + row_loss
                row_losses.append(float(row_loss.detach().cpu()))
            if pair_batches:
                pb = pair_batches[step % len(pair_batches)]
                pidx = torch.as_tensor(pb, dtype=torch.long, device=requested_device)
                left_pred = model(x_t.index_select(0, pair_left_t.index_select(0, pidx)))
                right_pred = model(x_t.index_select(0, pair_right_t.index_select(0, pidx)))
                diff = left_pred - right_pred
                target_diff = pair_target_t.index_select(0, pidx)
                pweight = pair_weight_t.index_select(0, pidx)
                pair_loss = torch.nn.functional.smooth_l1_loss(diff, target_diff, reduction="none")
                pair_loss = (pair_loss * pweight).mean()
                rank_loss = torch.nn.functional.softplus(-diff).mul(pweight).mean()
                loss = loss + float(pair_loss_weight) * pair_loss + float(rank_loss_weight) * rank_loss
                pair_losses.append(float(pair_loss.detach().cpu()))
                rank_losses.append(float(rank_loss.detach().cpu()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            print(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "row_loss": float(np.mean(row_losses)) if row_losses else None,
                        "pair_loss": float(np.mean(pair_losses)) if pair_losses else None,
                        "rank_loss": float(np.mean(rank_losses)) if rank_losses else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    model.eval()
    model.to("cpu")
    return model


def _predict(torch: Any, model: Any, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(len(x)), int(batch_size)):
            tensor = torch.as_tensor(x[start : start + int(batch_size)].astype(np.float32), dtype=torch.float32)
            values.append(model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32))
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def _selected_rows_for_margin(module: Any, data: dict[str, Any], scores: np.ndarray, margin: float) -> list[dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = module._safety_mask(data, enabled=True)
    rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        if base_rows.empty:
            rows.append(module._no_override_row(group))
            continue
        base_row_index = int(base_rows.index[0])
        base_score = float(scores[base_row_index])
        selectable = [int(index) for index in group_indices if bool(eligible[int(index)])]
        if not selectable:
            rows.append(module._no_override_row(group))
            continue
        best = int(
            min(
                selectable,
                key=lambda index: (-float(scores[index]), int(metadata.at[index, "candidate_index"])),
            )
        )
        score_margin = float(scores[best] - base_score)
        if score_margin < float(margin):
            rows.append(module._no_override_row(group))
            continue
        row = metadata.loc[best]
        rows.append(
            module._override_row(
                row=row,
                group=group,
                row_index=best,
                win_prob=0.0,
                loss_prob=0.0,
                delta_pred=score_margin,
                selector_score=score_margin,
                ranker_score=float(scores[best]),
            )
        )
    return rows


def _calibrate_margin(
    module: Any,
    data: dict[str, Any],
    scores: np.ndarray,
    *,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = module._safety_mask(data, enabled=True)
    candidates: list[float] = []
    for _, group in metadata.groupby("group_id", sort=False):
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        if base_rows.empty:
            continue
        base_score = float(scores[int(base_rows.index[0])])
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        selectable = [int(index) for index in group_indices if bool(eligible[int(index)])]
        if not selectable:
            continue
        best = max(selectable, key=lambda index: (float(scores[index]), -int(metadata.at[index, "candidate_index"])))
        candidates.append(float(scores[int(best)] - base_score))
    finite = np.asarray([value for value in candidates if np.isfinite(value)], dtype=np.float32)
    if finite.size == 0:
        rows = [module._no_override_row(group) for _, group in metadata.groupby("group_id", sort=False)]
        return float("inf"), module._selection_metrics(rows, metadata), []
    quantiles = np.linspace(0.0, 1.0, 161)
    margins = sorted(set(float(value) for value in np.quantile(finite, quantiles).tolist()))
    margins = [float("-inf")] + margins + [float(np.max(finite) + 1.0)]
    best_margin = float("inf")
    best_metrics: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    for margin in margins:
        rows = _selected_rows_for_margin(module, data, scores, margin)
        metrics = module._selection_metrics(rows, metadata)
        override_count = int(metrics.get("override_count", 0))
        override_rate = float(metrics.get("override_rate", 0.0))
        loss_rate = metrics.get("selected_loss_rate_when_overridden")
        total_delta = float(metrics.get("total_selected_accepted_delta_vs_base", 0.0))
        if override_count < int(min_override_count):
            continue
        if override_rate > float(max_override_rate):
            continue
        if loss_rate is not None and float(loss_rate) > float(max_loss_rate):
            continue
        if total_delta < float(min_total_delta):
            continue
        key = (
            total_delta,
            float(metrics.get("mean_selected_reward_delta_vs_base", 0.0)),
            -float(loss_rate if loss_rate is not None else 1.0),
            float(override_count),
            -float(margin),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_margin = float(margin)
            best_metrics = dict(metrics)
            best_rows = rows
    if best_metrics is None:
        # Fall back to the best total delta even if constraints cannot be met.
        for margin in margins:
            rows = _selected_rows_for_margin(module, data, scores, margin)
            metrics = module._selection_metrics(rows, metadata)
            key = (
                float(metrics.get("total_selected_accepted_delta_vs_base", 0.0)),
                float(metrics.get("mean_selected_reward_delta_vs_base", 0.0)),
                -float(metrics.get("selected_loss_rate_when_overridden") or 1.0),
                -float(metrics.get("override_rate", 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_margin = float(margin)
                best_metrics = dict(metrics)
                best_rows = rows
    assert best_metrics is not None
    best_metrics["selection_margin"] = float(best_margin)
    best_metrics["calibration_constraints_satisfied"] = bool(
        int(best_metrics.get("override_count", 0)) >= int(min_override_count)
        and float(best_metrics.get("override_rate", 0.0)) <= float(max_override_rate)
        and float(best_metrics.get("total_selected_accepted_delta_vs_base", 0.0)) >= float(min_total_delta)
        and (
            best_metrics.get("selected_loss_rate_when_overridden") is None
            or float(best_metrics.get("selected_loss_rate_when_overridden")) <= float(max_loss_rate)
        )
    )
    return float(best_margin), best_metrics, best_rows


def _artifact_meta(
    train_module: Any,
    *,
    feature_names: list[str],
    model_path: Path,
    top_k: int,
    selection_margin: float,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": "torch",
        "model_path": model_path.name,
        "feature_names": list(feature_names),
        "candidate_pool": "energy_topk_hybrid",
        "candidate_pool_top_k": int(top_k),
        "selection_mode": "base_residual",
        "selection_margin": float(selection_margin),
        "base_policy": "energy-aware-ksp-bm-ff",
        "safety_guard": train_module._emergency_safety_guard(),
        "advantage_gate": {"enabled": False},
        "risk_selector": {"enabled": False},
        "training": dict(training_summary),
    }


def train_and_export(
    *,
    run_dir: Path,
    output_dir: Path,
    top_k: int,
    threshold_fraction: float,
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
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    script_dir = Path(__file__).resolve().parent
    quick_module = _load_module("quick_exception_ranker_ab", script_dir / "quick_exception_ranker_ab.py")
    train_module = _load_module("train_distilled_dqn_ranker", script_dir / "train_distilled_dqn_ranker.py")

    original_train = quick_module._load_split(run_dir, "train")
    original_eval = quick_module._load_split(run_dir, "eval")
    train_pool = quick_module._add_runtime_features(quick_module._filter_small_pool(original_train, top_k=int(top_k)))
    eval_pool = quick_module._add_runtime_features(quick_module._filter_small_pool(original_eval, top_k=int(top_k)))
    calibration_npz = run_dir / "calibration_dagger_tree_ranker_examples.npz"
    calibration_csv = run_dir / "calibration_dagger_tree_ranker_examples.csv"
    if calibration_npz.exists() and calibration_csv.exists():
        original_calibration = quick_module._load_split(run_dir, "calibration")
        threshold_val = quick_module._add_runtime_features(
            quick_module._filter_small_pool(original_calibration, top_k=int(top_k))
        )
        train_inner = train_pool
        calibration_source = "calibration_split"
    else:
        train_inner, threshold_val = quick_module._split_train_threshold(
            train_pool,
            threshold_fraction=float(threshold_fraction),
            seed=int(seed),
        )
        calibration_source = "train_threshold_fraction"

    train_meta = train_inner["metadata"].reset_index(drop=True)
    target = _target_from_metadata(train_meta, secondary_scale=secondary_scale, target_clip=target_clip)
    row_weight = _sample_weights(
        train_meta,
        target,
        base_weight=base_weight,
        win_weight=win_weight,
        loss_weight=loss_weight,
        tie_weight=tie_weight,
        secondary_weight=secondary_weight,
    )
    pair_left, pair_right, pair_target, pair_weight = _pair_indices(
        train_meta,
        target,
        max_pairs_per_group=max_pairs_per_group,
        pair_target_clip=target_clip,
    )
    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _train_model(
        torch=torch,
        train_module=train_module,
        x=train_inner["x"].astype(np.float32),
        target=target,
        row_weight=row_weight,
        pair_left=pair_left,
        pair_right=pair_right,
        pair_target=pair_target,
        pair_weight=pair_weight,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        pair_batch_size=pair_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        pair_loss_weight=pair_loss_weight,
        rank_loss_weight=rank_loss_weight,
        seed=seed,
        device=requested_device,
    )

    train_scores = _predict(torch, model, train_inner["x"])
    threshold_scores = _predict(torch, model, threshold_val["x"])
    eval_scores = _predict(torch, model, eval_pool["x"])
    selection_margin, threshold_metrics, _ = _calibrate_margin(
        quick_module,
        threshold_val,
        threshold_scores,
        max_loss_rate=max_loss_rate,
        min_override_count=min_override_count,
        min_total_delta=min_total_delta,
        max_override_rate=max_override_rate,
    )
    train_rows = _selected_rows_for_margin(quick_module, train_inner, train_scores, selection_margin)
    threshold_rows = _selected_rows_for_margin(quick_module, threshold_val, threshold_scores, selection_margin)
    eval_rows = _selected_rows_for_margin(quick_module, eval_pool, eval_scores, selection_margin)
    train_metrics = quick_module._selection_metrics(train_rows, train_inner["metadata"].reset_index(drop=True))
    val_metrics = quick_module._selection_metrics(threshold_rows, threshold_val["metadata"].reset_index(drop=True))
    eval_metrics = quick_module._selection_metrics(eval_rows, eval_pool["metadata"].reset_index(drop=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "torch_dqn_base_relative_ranker.pt"
    train_module._save_scripted(torch, model, model_path)

    training_summary = {
        "run_dir": str(run_dir),
        "top_k": int(top_k),
        "threshold_fraction": float(threshold_fraction),
        "calibration_source": calibration_source,
        "secondary_scale": float(secondary_scale),
        "target_clip": float(target_clip),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "dropout": float(dropout),
        "epochs": int(epochs),
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
        "max_loss_rate": float(max_loss_rate),
        "min_override_count": int(min_override_count),
        "min_total_delta": float(min_total_delta),
        "max_override_rate": float(max_override_rate),
        "seed": int(seed),
        "device": requested_device,
        "train_rows": int(len(train_inner["metadata"])),
        "train_groups": int(train_inner["metadata"]["group_id"].nunique()),
        "pair_count": int(len(pair_left)),
        "target": {
            "mean": float(np.mean(target)),
            "std": float(np.std(target)),
            "min": float(np.min(target)),
            "max": float(np.max(target)),
            "positive_rows": int(np.sum(target > 1.0e-6)),
            "negative_rows": int(np.sum(target < -1.0e-6)),
            "zero_rows": int(np.sum(np.abs(target) <= 1.0e-6)),
        },
    }
    artifact_path = output_dir / "torch_dqn_base_relative_tree_ranker.json"
    _write_json(
        artifact_path,
        _artifact_meta(
            train_module,
            feature_names=list(train_inner["feature_names"]),
            model_path=model_path,
            top_k=top_k,
            selection_margin=selection_margin,
            training_summary=training_summary,
        ),
    )
    summary = {
        "artifact_path": str(artifact_path),
        "selection_margin": float(selection_margin),
        "training": training_summary,
        "threshold_val_calibration": threshold_metrics,
        "train_inner": train_metrics,
        "threshold_val": val_metrics,
        "eval": eval_metrics,
    }
    _write_json(output_dir / "torch_dqn_base_relative_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN ranker directly on base-relative counterfactual targets.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-fraction", type=float, default=0.15)
    parser.add_argument("--secondary-scale", type=float, default=0.25)
    parser.add_argument("--target-clip", type=float, default=4.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=0.35)
    parser.add_argument("--max-pairs-per-group", type=int, default=4)
    parser.add_argument("--base-weight", type=float, default=3.0)
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--loss-weight", type=float, default=8.0)
    parser.add_argument("--tie-weight", type=float, default=0.75)
    parser.add_argument("--secondary-weight", type=float, default=1.5)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_and_export(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        threshold_fraction=float(args.threshold_fraction),
        secondary_scale=float(args.secondary_scale),
        target_clip=float(args.target_clip),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
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
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
