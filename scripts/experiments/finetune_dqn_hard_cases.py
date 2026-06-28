from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
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


def _nonbase_mask(data: dict[str, Any]) -> np.ndarray:
    metadata = data["metadata"].reset_index(drop=True)
    return (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()


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


def _load_source_models(torch: Any, artifact_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    parent = artifact_path.parent
    gate = dict(meta["advantage_gate"])
    paths = {
        "ranker": parent / str(meta["model_path"]),
        "win": parent / str(gate["win_model_path"]),
        "loss": parent / str(gate["loss_model_path"]),
        "delta": parent / str(gate["delta_model_path"]),
    }
    return {name: torch.jit.load(str(path), map_location="cpu").eval() for name, path in paths.items()}


def _full_head_preds(torch: Any, models: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    mask = _nonbase_mask(data)
    positions = np.flatnonzero(mask)
    x = data["x"][mask].astype(np.float32)
    result = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
    result["loss"].fill(1.0)
    result["delta"].fill(-1.0)
    for name in ("win", "loss", "delta"):
        result[name][positions] = _predict_torch(torch, models[name], x)
    return result


def _script_state_dict(scripted_model: Any) -> dict[str, Any]:
    return {str(key): value.detach().cpu().clone() for key, value in scripted_model.state_dict().items()}


def _clone_predictor(
    *,
    torch: Any,
    distill_module: Any,
    scripted_model: Any,
    input_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    activation: int,
) -> Any:
    model = distill_module.TabularDqnPredictor.build(
        torch=torch,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
        activation=int(activation),
    )
    missing, unexpected = model.load_state_dict(_script_state_dict(scripted_model), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Warm-start state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def _batch_indices(size: int, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(int(size), dtype=np.int64)
    rng.shuffle(indices)
    return [indices[start : start + int(batch_size)] for start in range(0, int(size), int(batch_size))]


def _finetune_predictor(
    *,
    torch: Any,
    model: Any,
    x: np.ndarray,
    teacher_y: np.ndarray,
    true_y: np.ndarray,
    sample_weight: np.ndarray,
    objective: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    keep_weight: float,
    seed: int,
    device: str,
) -> Any:
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

    for _ in range(max(0, int(epochs))):
        model.train()
        for batch in _batch_indices(len(x), batch_size, rng):
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            pred = model(x_tensor.index_select(0, idx))
            true_loss = loss_to_target(pred, true_tensor.index_select(0, idx), weight_tensor.index_select(0, idx))
            keep_loss = loss_to_target(pred, teacher_tensor.index_select(0, idx))
            loss = true_loss + float(keep_weight) * keep_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    model.to("cpu")
    return model


def _selected_by_group(
    *,
    calibrate_module: Any,
    quick_module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    rows = calibrate_module._selected_override_rows_for_thresholds(quick_module, data, preds, thresholds)
    return {int(row["group_id"]): dict(row) for row in rows}


def _hard_case_masks(
    *,
    data: dict[str, Any],
    selected: dict[int, dict[str, Any]],
    weak_scenario_weight: float,
    hard_group_weight: float,
    hard_positive_weight: float,
    hard_negative_weight: float,
    hard_tie_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    hard_groups: set[int] = set()
    hard_positive_rows: set[int] = set()
    hard_negative_rows: set[int] = set()
    hard_tie_rows: set[int] = set()
    weak_scenarios = {
        ("bursty", "high"),
        ("nonuniform", "overload"),
        ("uniform", "high"),
        ("uniform", "overload"),
    }

    for gid, group in metadata.groupby("group_id", sort=False):
        group_id = int(gid)
        nonbase = group[group["candidate_index"].astype(int) != group["base_index"].astype(int)]
        if nonbase.empty:
            continue
        accepted = nonbase["accepted_delta_vs_base"].astype(float)
        reward = nonbase.get("future_env_reward_delta_vs_base", pd.Series(np.zeros((len(nonbase),)))).astype(float)
        best_acc = float(accepted.max())
        selected_row = selected.get(group_id)
        selected_acc = 0.0 if selected_row is None else float(selected_row["accepted_delta_vs_base"])
        selected_reward = 0.0 if selected_row is None else float(selected_row.get("reward_delta_vs_base", 0.0))

        if selected_row is not None and selected_acc < 0.0:
            hard_groups.add(group_id)
            hard_negative_rows.add(int(selected_row["row_index"]))
        if best_acc > selected_acc:
            hard_groups.add(group_id)
            best_rows = nonbase[nonbase["accepted_delta_vs_base"].astype(float) == best_acc]
            for row_index in best_rows.index:
                hard_positive_rows.add(int(row_index))
        if selected_row is not None and selected_acc == 0.0 and selected_reward < 0.0:
            hard_groups.add(group_id)
            hard_tie_rows.add(int(selected_row["row_index"]))
        if selected_row is not None and selected_acc <= 0.0 and best_acc <= 0.0 and float(reward.max()) > selected_reward:
            hard_groups.add(group_id)
            best_reward_rows = nonbase[reward == float(reward.max())]
            for row_index in best_reward_rows.index:
                hard_tie_rows.add(int(row_index))

    weights = np.ones((len(metadata),), dtype=np.float32)
    hard_group_mask = metadata["group_id"].astype(int).isin(hard_groups).to_numpy()
    weights *= np.where(hard_group_mask, float(hard_group_weight), 1.0).astype(np.float32)
    if {"traffic_scenario", "load_name"}.issubset(metadata.columns) and float(weak_scenario_weight) != 1.0:
        weak_mask = np.asarray(
            [
                (str(row.traffic_scenario), str(row.load_name)) in weak_scenarios
                for row in metadata[["traffic_scenario", "load_name"]].itertuples(index=False)
            ],
            dtype=bool,
        )
        weights *= np.where(weak_mask, float(weak_scenario_weight), 1.0).astype(np.float32)
    if hard_positive_rows:
        weights[np.asarray(sorted(hard_positive_rows), dtype=np.int64)] *= float(hard_positive_weight)
    if hard_negative_rows:
        weights[np.asarray(sorted(hard_negative_rows), dtype=np.int64)] *= float(hard_negative_weight)
    if hard_tie_rows:
        weights[np.asarray(sorted(hard_tie_rows), dtype=np.int64)] *= float(hard_tie_weight)

    diagnostics = {
        "hard_groups": int(len(hard_groups)),
        "hard_positive_rows": int(len(hard_positive_rows)),
        "hard_negative_rows": int(len(hard_negative_rows)),
        "hard_tie_rows": int(len(hard_tie_rows)),
        "mean_row_weight": float(np.mean(weights)),
        "max_row_weight": float(np.max(weights)),
    }
    if {"traffic_scenario", "load_name"}.issubset(metadata.columns):
        by_scenario: dict[str, int] = {}
        hard_metadata = metadata[metadata["group_id"].astype(int).isin(hard_groups)]
        for key, count in hard_metadata.groupby(["traffic_scenario", "load_name"])["group_id"].nunique().items():
            by_scenario[f"{key[0]}/{key[1]}"] = int(count)
        diagnostics["hard_groups_by_scenario_load"] = by_scenario
    return weights.astype(np.float32), diagnostics


def _safe_mean_normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return (weights / max(float(np.mean(weights)), 1.0e-6)).astype(np.float32)


def finetune_and_export(
    *,
    run_dir: Path,
    source_artifact: Path,
    output_dir: Path,
    top_k: int,
    threshold_fraction: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    keep_weight: float,
    hard_group_weight: float,
    hard_positive_weight: float,
    hard_negative_weight: float,
    hard_tie_weight: float,
    weak_scenario_weight: float,
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

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    script_dir = Path(__file__).resolve().parent
    distill_module = _load_module("train_distilled_dqn_ranker", script_dir / "train_distilled_dqn_ranker.py")
    calibrate_module = _load_module("calibrate_dqn_override_rate", script_dir / "calibrate_dqn_override_rate.py")
    quick_module = distill_module._load_quick_module()

    source_meta = json.loads(source_artifact.read_text(encoding="utf-8"))
    training = dict(source_meta.get("training") or {})
    hidden_dim = int(training.get("hidden_dim", 256))
    depth = int(training.get("depth", 4))
    dropout = float(training.get("dropout", 0.0))
    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"

    original_train = quick_module._load_split(run_dir, "train")
    original_eval = quick_module._load_split(run_dir, "eval")
    train_pool = quick_module._add_runtime_features(quick_module._filter_small_pool(original_train, top_k=top_k))
    eval_pool = quick_module._add_runtime_features(quick_module._filter_small_pool(original_eval, top_k=top_k))
    train_inner, threshold_val = quick_module._split_train_threshold(train_pool, threshold_fraction=threshold_fraction, seed=seed)

    source_models = _load_source_models(torch, source_artifact, source_meta)
    source_train_heads = _full_head_preds(torch, source_models, train_inner)
    source_threshold_heads = _full_head_preds(torch, source_models, threshold_val)
    source_eval_heads = _full_head_preds(torch, source_models, eval_pool)
    source_train_ranker = _predict_torch(torch, source_models["ranker"], train_inner["x"])
    source_threshold_ranker = _predict_torch(torch, source_models["ranker"], threshold_val["x"])
    source_eval_ranker = _predict_torch(torch, source_models["ranker"], eval_pool["x"])
    source_thresholds = dict(source_meta["advantage_gate"])

    selected = _selected_by_group(
        calibrate_module=calibrate_module,
        quick_module=quick_module,
        data=train_inner,
        preds=source_train_heads,
        thresholds=source_thresholds,
    )
    row_weights, hard_diagnostics = _hard_case_masks(
        data=train_inner,
        selected=selected,
        weak_scenario_weight=weak_scenario_weight,
        hard_group_weight=hard_group_weight,
        hard_positive_weight=hard_positive_weight,
        hard_negative_weight=hard_negative_weight,
        hard_tie_weight=hard_tie_weight,
    )
    metadata = train_inner["metadata"].reset_index(drop=True)
    nonbase_mask = _nonbase_mask(train_inner)
    train_nonbase = quick_module._non_base_dataset(train_inner)
    train_windowed_return = distill_module._windowed_return_delta(
        metadata,
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

    ranker_accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    ranker_weight = row_weights.copy()
    ranker_weight *= np.where(ranker_accepted > 0.0, 4.0, np.where(ranker_accepted < 0.0, 3.0, 1.0)).astype(np.float32)
    ranker_weight *= np.where(np.abs(train_windowed_return) > 1.0e-6, 1.25, 1.0).astype(np.float32)
    ranker_weight = _safe_mean_normalize(ranker_weight)
    head_weight = row_weights[nonbase_mask].copy()
    head_meta = train_nonbase["metadata"].reset_index(drop=True)
    head_accepted = head_meta["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    head_weight *= np.where(head_accepted > 0.0, 5.0, np.where(head_accepted < 0.0, 4.0, 1.0)).astype(np.float32)
    head_weight = _safe_mean_normalize(head_weight)

    input_dim = int(train_inner["x"].shape[1])
    models = {
        "ranker": _clone_predictor(
            torch=torch,
            distill_module=distill_module,
            scripted_model=source_models["ranker"],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            activation=0,
        ),
        "win": _clone_predictor(
            torch=torch,
            distill_module=distill_module,
            scripted_model=source_models["win"],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            activation=1,
        ),
        "loss": _clone_predictor(
            torch=torch,
            distill_module=distill_module,
            scripted_model=source_models["loss"],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            activation=1,
        ),
        "delta": _clone_predictor(
            torch=torch,
            distill_module=distill_module,
            scripted_model=source_models["delta"],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            activation=0,
        ),
    }

    models["ranker"] = _finetune_predictor(
        torch=torch,
        model=models["ranker"],
        x=train_inner["x"],
        teacher_y=source_train_ranker,
        true_y=train_windowed_return,
        sample_weight=ranker_weight,
        objective="mse",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        keep_weight=keep_weight,
        seed=seed + 1,
        device=requested_device,
    )
    for offset, (name, objective, true_y) in enumerate(
        (
            ("win", "bce", train_nonbase["win_y"]),
            ("loss", "bce", train_nonbase["loss_y"]),
            ("delta", "mse", train_windowed_return[nonbase_mask]),
        )
    ):
        models[name] = _finetune_predictor(
            torch=torch,
            model=models[name],
            x=train_nonbase["x"],
            teacher_y=source_train_heads[name][nonbase_mask],
            true_y=np.asarray(true_y, dtype=np.float32),
            sample_weight=head_weight,
            objective=objective,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            keep_weight=keep_weight,
            seed=seed + 10 + offset,
            device=requested_device,
        )

    finetuned_train_heads = _full_head_preds(torch, models, train_inner)
    finetuned_threshold_heads = _full_head_preds(torch, models, threshold_val)
    finetuned_eval_heads = _full_head_preds(torch, models, eval_pool)
    source_metrics = {
        "threshold_val": calibrate_module._metrics_for_thresholds(quick_module, threshold_val, source_threshold_heads, source_thresholds),
        "eval": calibrate_module._metrics_for_thresholds(quick_module, eval_pool, source_eval_heads, source_thresholds),
    }
    finetuned_metrics = {
        "threshold_val": calibrate_module._metrics_for_thresholds(
            quick_module,
            threshold_val,
            finetuned_threshold_heads,
            source_thresholds,
        ),
        "eval": calibrate_module._metrics_for_thresholds(quick_module, eval_pool, finetuned_eval_heads, source_thresholds),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ranker_path = output_dir / "torch_dqn_hard_finetune_ranker.pt"
    win_path = output_dir / "torch_dqn_hard_finetune_advantage_win.pt"
    loss_path = output_dir / "torch_dqn_hard_finetune_advantage_loss.pt"
    delta_path = output_dir / "torch_dqn_hard_finetune_advantage_delta.pt"
    distill_module._save_scripted(torch, models["ranker"], ranker_path)
    distill_module._save_scripted(torch, models["win"], win_path)
    distill_module._save_scripted(torch, models["loss"], loss_path)
    distill_module._save_scripted(torch, models["delta"], delta_path)

    meta = dict(source_meta)
    meta["model_path"] = ranker_path.name
    gate = dict(meta["advantage_gate"])
    gate.update(
        {
            "win_model_path": win_path.name,
            "loss_model_path": loss_path.name,
            "delta_model_path": delta_path.name,
        }
    )
    meta["advantage_gate"] = gate
    meta["training"] = {
        **training,
        "hard_case_finetune": {
            "source_artifact": str(source_artifact),
            "run_dir": str(run_dir),
            "top_k": int(top_k),
            "threshold_fraction": float(threshold_fraction),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "keep_weight": float(keep_weight),
            "hard_group_weight": float(hard_group_weight),
            "hard_positive_weight": float(hard_positive_weight),
            "hard_negative_weight": float(hard_negative_weight),
            "hard_tie_weight": float(hard_tie_weight),
            "weak_scenario_weight": float(weak_scenario_weight),
            "device": requested_device,
            "seed": int(seed),
            "hard_cases": hard_diagnostics,
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
        },
    }
    artifact_path = output_dir / "torch_dqn_hard_finetune_old10_tree_ranker.json"
    _write_json(artifact_path, meta)
    summary = {
        "artifact_path": str(artifact_path),
        "source_artifact": str(source_artifact),
        "hard_cases": hard_diagnostics,
        "source_metrics_with_source_thresholds": source_metrics,
        "finetuned_metrics_with_source_thresholds": finetuned_metrics,
        "source_ranker_score_stats": {
            "train_mean": float(np.mean(source_train_ranker)),
            "threshold_mean": float(np.mean(source_threshold_ranker)),
            "eval_mean": float(np.mean(source_eval_ranker)),
        },
        "training": meta["training"]["hard_case_finetune"],
    }
    _write_json(output_dir / "torch_dqn_hard_finetune_old10_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start fine-tune distilled DQN on hard override cases.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--keep-weight", type=float, default=6.0)
    parser.add_argument("--hard-group-weight", type=float, default=2.0)
    parser.add_argument("--hard-positive-weight", type=float, default=8.0)
    parser.add_argument("--hard-negative-weight", type=float, default=10.0)
    parser.add_argument("--hard-tie-weight", type=float, default=4.0)
    parser.add_argument("--weak-scenario-weight", type=float, default=1.25)
    parser.add_argument("--accepted-weight", type=float, default=2.5)
    parser.add_argument("--block-penalty", type=float, default=2.0)
    parser.add_argument("--reward-weight", type=float, default=0.0)
    parser.add_argument("--energy-weight", type=float, default=0.15)
    parser.add_argument("--energy-norm-w", type=float, default=1200.0)
    parser.add_argument("--fragmentation-weight", type=float, default=0.35)
    parser.add_argument("--qot-weight", type=float, default=0.10)
    parser.add_argument("--qot-clip-min", type=float, default=-1.0)
    parser.add_argument("--qot-clip-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    summary = finetune_and_export(
        run_dir=Path(args.run_dir),
        source_artifact=Path(args.source_artifact),
        output_dir=Path(args.output_dir),
        top_k=int(args.top_k),
        threshold_fraction=float(args.threshold_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        keep_weight=float(args.keep_weight),
        hard_group_weight=float(args.hard_group_weight),
        hard_positive_weight=float(args.hard_positive_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        hard_tie_weight=float(args.hard_tie_weight),
        weak_scenario_weight=float(args.weak_scenario_weight),
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
