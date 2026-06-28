from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_quick_module() -> Any:
    module_path = Path(__file__).with_name("quick_exception_ranker_ab.py")
    spec = importlib.util.spec_from_file_location("quick_exception_ranker_ab", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _nonbase_mask(data: dict[str, Any]) -> np.ndarray:
    metadata = data["metadata"].reset_index(drop=True)
    return (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()


def _nonbase_view(data: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    mask = _nonbase_mask(data)
    metadata = data["metadata"].reset_index(drop=True)
    return (
        {
            "x": data["x"][mask].astype(np.float32),
            "metadata": metadata.loc[mask].reset_index(drop=True).copy(),
            "feature_names": list(data["feature_names"]),
        },
        np.flatnonzero(mask),
    )


def _load_lightgbm_teacher(module: Any, teacher_artifact: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import lightgbm as lgb

    meta = json.loads(teacher_artifact.read_text(encoding="utf-8"))
    if str(meta.get("backend")) != "lightgbm":
        raise ValueError(f"Teacher artifact must be lightgbm, got {meta.get('backend')}")
    ranker = lgb.Booster(model_file=str(teacher_artifact.parent / str(meta["model_path"])))
    gate = dict(meta.get("advantage_gate") or {})
    heads = {
        "win": lgb.Booster(model_file=str(teacher_artifact.parent / str(gate["win_model_path"]))),
        "loss": lgb.Booster(model_file=str(teacher_artifact.parent / str(gate["loss_model_path"]))),
        "delta": lgb.Booster(model_file=str(teacher_artifact.parent / str(gate["delta_model_path"]))),
    }
    return meta, ranker, heads


def _teacher_preds(module: Any, teacher_ranker: Any, teacher_heads: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    ranker_scores = module._lgb_predict(teacher_ranker, data["x"])
    nonbase_data, positions = _nonbase_view(data)
    head_subset = {
        name: module._lgb_predict(model, nonbase_data["x"])
        for name, model in teacher_heads.items()
    }
    heads = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
    heads["loss"].fill(1.0)
    heads["delta"].fill(-1.0)
    for name, values in head_subset.items():
        heads[name][positions] = np.asarray(values, dtype=np.float32)
    return {"ranker": np.asarray(ranker_scores, dtype=np.float32), "heads": heads}


def _hard_group_ids(module: Any, data: dict[str, Any], preds: dict[str, np.ndarray], thresholds: dict[str, float]) -> set[int]:
    rows, _ = module._three_head_selected_rows(
        data=data,
        preds=preds,
        thresholds=thresholds,
        safety_enabled=True,
        apply_loss_threshold=bool(thresholds.get("max_loss_prob", 1.0) < 1.000001),
    )
    metadata = data["metadata"].reset_index(drop=True)
    hard: set[int] = set()
    for row in rows:
        gid = int(row["group_id"])
        group = metadata[metadata["group_id"].astype(int) == gid]
        has_win = bool((group["accepted_delta_vs_base"].astype(float) > 0.0).any())
        if bool(row.get("override", False)) and float(row["accepted_delta_vs_base"]) < 0.0:
            hard.add(gid)
            continue
        if has_win and (not bool(row.get("override", False)) or float(row["accepted_delta_vs_base"]) <= 0.0):
            hard.add(gid)
    return hard


def _windowed_return_delta(
    metadata: pd.DataFrame,
    *,
    accepted_weight: float,
    block_penalty: float,
    reward_weight: float,
    energy_weight: float,
    energy_norm_w: float,
    fragmentation_weight: float,
    qot_weight: float,
    qot_clip_min: float,
    qot_clip_max: float,
) -> np.ndarray:
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    blocked_delta = -accepted
    reward = metadata.get("future_env_reward_delta_vs_base", pd.Series(np.zeros((len(metadata),), dtype=np.float32))).to_numpy(
        dtype=np.float32
    )
    energy = metadata.get(
        "future_energy_increment_delta_vs_base",
        pd.Series(np.zeros((len(metadata),), dtype=np.float32)),
    ).to_numpy(dtype=np.float32)
    fragmentation = metadata.get(
        "future_fragmentation_after_delta_vs_base",
        pd.Series(np.zeros((len(metadata),), dtype=np.float32)),
    ).to_numpy(dtype=np.float32)
    qot = metadata.get("future_qot_margin_delta_vs_base", pd.Series(np.zeros((len(metadata),), dtype=np.float32))).to_numpy(
        dtype=np.float32
    )
    qot = np.clip(qot, float(qot_clip_min), float(qot_clip_max)).astype(np.float32)
    return (
        float(accepted_weight) * accepted
        - float(block_penalty) * blocked_delta
        + float(reward_weight) * reward
        - float(energy_weight) * (energy / max(float(energy_norm_w), 1.0e-9))
        - float(fragmentation_weight) * fragmentation
        + float(qot_weight) * qot
    ).astype(np.float32)


class TabularDqnPredictor:
    @staticmethod
    def build(*, torch: Any, input_dim: int, hidden_dim: int, depth: int, dropout: float, activation: int) -> Any:
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("mean", torch.zeros(input_dim, dtype=torch.float32))
                self.register_buffer("std", torch.ones(input_dim, dtype=torch.float32))
                layers: list[Any] = []
                width = int(input_dim)
                for _ in range(max(1, int(depth))):
                    layers.append(nn.Linear(width, int(hidden_dim)))
                    layers.append(nn.ReLU())
                    if float(dropout) > 0.0:
                        layers.append(nn.Dropout(float(dropout)))
                    width = int(hidden_dim)
                layers.append(nn.Linear(width, 1))
                self.net = nn.Sequential(*layers)
                self.activation = int(activation)

            def set_normalizer(self, mean: Any, std: Any) -> None:
                self.mean.copy_(mean)
                self.std.copy_(std)

            def forward(self, x: Any) -> Any:
                z = (x - self.mean) / self.std.clamp_min(1.0e-6)
                y = self.net(z).squeeze(-1)
                if self.activation == 1:
                    return torch.sigmoid(y)
                return y

        return _Model()


def _batch_indices(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(int(size), dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + int(batch_size)] for start in range(0, int(size), int(batch_size))]


def _train_predictor(
    *,
    torch: Any,
    x: np.ndarray,
    teacher_y: np.ndarray,
    true_y: np.ndarray,
    sample_weight: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    activation: int,
    objective: str,
    hidden_dim: int,
    depth: int,
    dropout: float,
    distill_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    distill_keep_weight: float,
    seed: int,
    device: str,
) -> Any:
    torch.manual_seed(int(seed))
    model = TabularDqnPredictor.build(
        torch=torch,
        input_dim=int(x.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
        activation=int(activation),
    )
    model.set_normalizer(
        torch.as_tensor(mean.astype(np.float32), dtype=torch.float32),
        torch.as_tensor(std.astype(np.float32), dtype=torch.float32),
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    x_tensor = torch.as_tensor(x.astype(np.float32), dtype=torch.float32, device=device)
    teacher_tensor = torch.as_tensor(teacher_y.astype(np.float32), dtype=torch.float32, device=device)
    true_tensor = torch.as_tensor(true_y.astype(np.float32), dtype=torch.float32, device=device)
    weight_tensor = torch.as_tensor(sample_weight.astype(np.float32), dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(seed))

    def loss_to_target(pred: Any, target: Any, weight: Any | None = None) -> Any:
        if objective == "bce":
            loss = torch.nn.functional.binary_cross_entropy(pred.clamp(1.0e-5, 1.0 - 1.0e-5), target, reduction="none")
        else:
            loss = torch.square(pred - target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    for _ in range(max(0, int(distill_epochs))):
        model.train()
        for batch in _batch_indices(len(x), batch_size, rng):
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_tensor.index_select(0, idx))
            loss = loss_to_target(pred, teacher_tensor.index_select(0, idx))
            loss.backward()
            optimizer.step()

    for _ in range(max(0, int(finetune_epochs))):
        model.train()
        for batch in _batch_indices(len(x), batch_size, rng):
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_tensor.index_select(0, idx))
            true_loss = loss_to_target(
                pred,
                true_tensor.index_select(0, idx),
                weight_tensor.index_select(0, idx),
            )
            keep_loss = loss_to_target(pred, teacher_tensor.index_select(0, idx))
            loss = true_loss + float(distill_keep_weight) * keep_loss
            loss.backward()
            optimizer.step()

    model.eval()
    model.to("cpu")
    return model


def _predict_torch(torch: Any, model: Any, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(len(x)), int(batch_size)):
            tensor = torch.as_tensor(x[start : start + int(batch_size)].astype(np.float32), dtype=torch.float32)
            values.append(model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32))
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def _full_dqn_head_preds(torch: Any, models: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    subset, positions = _nonbase_view(data)
    result = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
    result["loss"].fill(1.0)
    result["delta"].fill(-1.0)
    for name in ("win", "loss", "delta"):
        result[name][positions] = _predict_torch(torch, models[name], subset["x"])
    return result


def _save_scripted(torch: Any, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model = model.cpu()
    example = torch.zeros((1, int(model.mean.numel())), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save(str(path))


def _artifact_meta(
    *,
    feature_names: list[str],
    ranker_path: Path,
    win_path: Path,
    loss_path: Path,
    delta_path: Path,
    thresholds: dict[str, float],
    top_k: int,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "backend": "torch",
        "model_path": ranker_path.name,
        "feature_names": list(feature_names),
        "candidate_pool": "energy_topk_hybrid",
        "candidate_pool_top_k": int(top_k),
        "selection_mode": "positive_advantage",
        "residual_beta": 0.05,
        "selection_margin": 0.0,
        "base_policy": "energy-aware-ksp-bm-ff",
        "safety_guard": _emergency_safety_guard(),
        "advantage_gate": {
            "enabled": True,
            "backend": "torch",
            "feature_source": "ranker_features",
            "feature_names": list(feature_names),
            "win_model_path": win_path.name,
            "loss_model_path": loss_path.name,
            "delta_model_path": delta_path.name,
            "check_loss_prob": True,
            "win_weight": 1.0,
            "loss_weight": 2.0,
            "delta_weight": 1.0,
            "ranker_margin_weight": 0.0,
            **dict(thresholds),
        },
        "risk_selector": {"enabled": False},
        "training": dict(training_summary),
    }


def train_and_export(
    *,
    run_dir: Path,
    teacher_artifact: Path,
    output_dir: Path,
    top_k: int,
    threshold_fraction: float,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    hidden_dim: int,
    depth: int,
    dropout: float,
    distill_epochs: int,
    finetune_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    distill_keep_weight: float,
    hard_group_weight: float,
    accepted_weight: float,
    block_penalty: float,
    reward_weight: float,
    energy_weight: float,
    energy_norm_w: float,
    fragmentation_weight: float,
    qot_weight: float,
    qot_clip_min: float,
    qot_clip_max: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    module = _load_quick_module()
    original_train = module._load_split(run_dir, "train")
    original_eval = module._load_split(run_dir, "eval")
    train_pool = module._add_runtime_features(module._filter_small_pool(original_train, top_k=top_k))
    eval_pool = module._add_runtime_features(module._filter_small_pool(original_eval, top_k=top_k))
    train_inner, threshold_val = module._split_train_threshold(train_pool, threshold_fraction=threshold_fraction, seed=seed)
    teacher_meta, teacher_ranker, teacher_heads = _load_lightgbm_teacher(module, teacher_artifact)
    teacher_thresholds = {
        "min_win_prob": float(teacher_meta["advantage_gate"]["min_win_prob"]),
        "max_loss_prob": float(teacher_meta["advantage_gate"].get("max_loss_prob", 1.0)),
        "min_delta_pred": float(teacher_meta["advantage_gate"]["min_delta_pred"]),
    }
    teacher_train = _teacher_preds(module, teacher_ranker, teacher_heads, train_inner)
    teacher_threshold = _teacher_preds(module, teacher_ranker, teacher_heads, threshold_val)
    teacher_eval = _teacher_preds(module, teacher_ranker, teacher_heads, eval_pool)

    hard_groups = _hard_group_ids(module, train_inner, teacher_train["heads"], teacher_thresholds)
    train_nonbase = module._non_base_dataset(train_inner)
    nonbase_mask = _nonbase_mask(train_inner)
    train_windowed_return = _windowed_return_delta(
        train_inner["metadata"].reset_index(drop=True),
        accepted_weight=accepted_weight,
        block_penalty=block_penalty,
        reward_weight=reward_weight,
        energy_weight=energy_weight,
        energy_norm_w=energy_norm_w,
        fragmentation_weight=fragmentation_weight,
        qot_weight=qot_weight,
        qot_clip_min=qot_clip_min,
        qot_clip_max=qot_clip_max,
    )
    x_head = train_nonbase["x"]
    meta_head = train_nonbase["metadata"].reset_index(drop=True)
    accepted = meta_head["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    nonbase_windowed_return = train_windowed_return[nonbase_mask]
    group_hard = meta_head["group_id"].astype(int).isin(hard_groups).to_numpy()
    return_nonzero = np.abs(nonbase_windowed_return) > 1.0e-6
    hard_weight = np.where(
        accepted > 0.0,
        8.0,
        np.where(accepted < 0.0, 6.0, np.where(return_nonzero, 2.0, 0.75)),
    ).astype(np.float32)
    hard_weight *= np.where(group_hard, float(hard_group_weight), 1.0).astype(np.float32)
    hard_weight = hard_weight / max(float(np.mean(hard_weight)), 1.0e-6)

    x_ranker = train_inner["x"].astype(np.float32)
    ranker_weight = np.ones((len(x_ranker),), dtype=np.float32)
    ranker_meta = train_inner["metadata"].reset_index(drop=True)
    ranker_accepted = ranker_meta["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    ranker_weight *= np.where(ranker_accepted > 0.0, 5.0, np.where(ranker_accepted < 0.0, 4.0, 1.0)).astype(np.float32)
    ranker_weight *= np.where(np.abs(train_windowed_return) > 1.0e-6, 1.5, 1.0).astype(np.float32)
    ranker_weight *= np.where(ranker_meta["group_id"].astype(int).isin(hard_groups).to_numpy(), float(hard_group_weight), 1.0)
    ranker_weight = ranker_weight / max(float(np.mean(ranker_weight)), 1.0e-6)

    mean = x_ranker.mean(axis=0).astype(np.float32)
    std = x_ranker.std(axis=0).astype(np.float32)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    models: dict[str, Any] = {}
    models["ranker"] = _train_predictor(
        torch=torch,
        x=x_ranker,
        teacher_y=teacher_train["ranker"],
        true_y=train_windowed_return,
        sample_weight=ranker_weight,
        mean=mean,
        std=std,
        activation=0,
        objective="mse",
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        distill_epochs=distill_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        distill_keep_weight=distill_keep_weight,
        seed=seed + 1,
        device=requested_device,
    )
    for name, activation, objective, true_y in (
        ("win", 1, "bce", train_nonbase["win_y"]),
        ("loss", 1, "bce", train_nonbase["loss_y"]),
        ("delta", 0, "mse", nonbase_windowed_return),
    ):
        models[name] = _train_predictor(
            torch=torch,
            x=x_head,
            teacher_y=teacher_train["heads"][name][nonbase_mask],
            true_y=np.asarray(true_y, dtype=np.float32),
            sample_weight=hard_weight,
            mean=mean,
            std=std,
            activation=activation,
            objective=objective,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            distill_epochs=distill_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            distill_keep_weight=distill_keep_weight,
            seed=seed + 10 + len(models),
            device=requested_device,
        )

    train_heads = _full_dqn_head_preds(torch, models, train_inner)
    threshold_heads = _full_dqn_head_preds(torch, models, threshold_val)
    eval_heads = _full_dqn_head_preds(torch, models, eval_pool)

    def dqn_selector(*, data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
        preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
        return module._select_three_head(data=data, preds=preds, thresholds=thresholds, safety_enabled=True)

    thresholds, threshold_metrics = module._tune_thresholds(
        selector=dqn_selector,
        data=threshold_val,
        max_loss_rate=max_loss_rate,
        min_override_count=min_override_count,
        min_total_delta=min_total_delta,
    )
    train_metrics = module._select_three_head(data=train_inner, preds=train_heads, thresholds=thresholds, safety_enabled=True)
    eval_metrics = module._select_three_head(data=eval_pool, preds=eval_heads, thresholds=thresholds, safety_enabled=True)
    teacher_train_metrics = module._select_three_head(
        data=train_inner,
        preds=teacher_train["heads"],
        thresholds=teacher_thresholds,
        safety_enabled=True,
    )
    teacher_eval_metrics = module._select_three_head(
        data=eval_pool,
        preds=teacher_eval["heads"],
        thresholds=teacher_thresholds,
        safety_enabled=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ranker_path = output_dir / "torch_dqn_distill_ranker.pt"
    win_path = output_dir / "torch_dqn_distill_advantage_win.pt"
    loss_path = output_dir / "torch_dqn_distill_advantage_loss.pt"
    delta_path = output_dir / "torch_dqn_distill_advantage_delta.pt"
    _save_scripted(torch, models["ranker"], ranker_path)
    _save_scripted(torch, models["win"], win_path)
    _save_scripted(torch, models["loss"], loss_path)
    _save_scripted(torch, models["delta"], delta_path)

    training_summary = {
        "run_dir": str(run_dir),
        "teacher_artifact": str(teacher_artifact),
        "top_k": int(top_k),
        "threshold_fraction": float(threshold_fraction),
        "max_loss_rate": float(max_loss_rate),
        "seed": int(seed),
        "device": requested_device,
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "dropout": float(dropout),
        "distill_epochs": int(distill_epochs),
        "finetune_epochs": int(finetune_epochs),
        "hard_groups": int(len(hard_groups)),
        "windowed_return": {
            "accepted_weight": float(accepted_weight),
            "block_penalty": float(block_penalty),
            "reward_weight": float(reward_weight),
            "energy_weight": float(energy_weight),
            "energy_norm_w": float(energy_norm_w),
            "fragmentation_weight": float(fragmentation_weight),
            "qot_weight": float(qot_weight),
            "qot_clip_min": float(qot_clip_min),
            "qot_clip_max": float(qot_clip_max),
            "train_mean": float(np.mean(train_windowed_return)),
            "train_std": float(np.std(train_windowed_return)),
            "train_min": float(np.min(train_windowed_return)),
            "train_max": float(np.max(train_windowed_return)),
        },
    }
    artifact_path = output_dir / "torch_dqn_distill_old10_tree_ranker.json"
    _write_json(
        artifact_path,
        _artifact_meta(
            feature_names=list(train_inner["feature_names"]),
            ranker_path=ranker_path,
            win_path=win_path,
            loss_path=loss_path,
            delta_path=delta_path,
            thresholds=thresholds,
            top_k=top_k,
            training_summary=training_summary,
        ),
    )
    summary = {
        "artifact_path": str(artifact_path),
        "training": training_summary,
        "thresholds": thresholds,
        "threshold_val": threshold_metrics,
        "train_inner": train_metrics,
        "eval": eval_metrics,
        "teacher_thresholds": teacher_thresholds,
        "teacher_train_inner": teacher_train_metrics,
        "teacher_eval": teacher_eval_metrics,
    }
    _write_json(output_dir / "torch_dqn_distill_old10_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill LightGBM old10 exception policy into a tabular DQN/Q-network.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--teacher-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-fraction", type=float, default=0.1)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--distill-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--distill-keep-weight", type=float, default=0.25)
    parser.add_argument("--hard-group-weight", type=float, default=2.0)
    parser.add_argument("--accepted-weight", type=float, default=2.0)
    parser.add_argument("--block-penalty", type=float, default=1.5)
    parser.add_argument("--reward-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.25)
    parser.add_argument("--energy-norm-w", type=float, default=1200.0)
    parser.add_argument("--fragmentation-weight", type=float, default=0.80)
    parser.add_argument("--qot-weight", type=float, default=0.20)
    parser.add_argument("--qot-clip-min", type=float, default=-1.0)
    parser.add_argument("--qot-clip-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = train_and_export(
        run_dir=Path(args.run_dir),
        teacher_artifact=Path(args.teacher_artifact),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        threshold_fraction=float(args.threshold_fraction),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        distill_epochs=int(args.distill_epochs),
        finetune_epochs=int(args.finetune_epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        distill_keep_weight=float(args.distill_keep_weight),
        hard_group_weight=float(args.hard_group_weight),
        accepted_weight=float(args.accepted_weight),
        block_penalty=float(args.block_penalty),
        reward_weight=float(args.reward_weight),
        energy_weight=float(args.energy_weight),
        energy_norm_w=float(args.energy_norm_w),
        fragmentation_weight=float(args.fragmentation_weight),
        qot_weight=float(args.qot_weight),
        qot_clip_min=float(args.qot_clip_min),
        qot_clip_max=float(args.qot_clip_max),
        seed=int(args.seed),
        device=str(args.device),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
