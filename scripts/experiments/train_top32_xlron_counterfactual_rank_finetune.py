from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.ong_rollout import _add_ong_source_path, _solver_config
from cse2026.experiments.eon.train_dqn import _device, _raw_int

from train_full_dqn_counterfactual_rank_finetune import (
    _listwise_pairwise_loss,
    _score_checkpoint,
    _selection_metrics,
)
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
from train_top32_xlron_full_dqn_distill import (
    _load_xlron_checkpoint_model,
    _rollout_reference_for_selection,
    _rollout_selection_metric,
    _rollout_validate_for_selection,
    _xlron_forward,
)


ARCHITECTURE_KEYS = (
    "policy",
    "n_max",
    "action_feature_dim",
    "link_feature_dim",
    "global_feature_dim",
    "request_feature_dim",
    "embedding_dim",
    "hidden_dim",
    "transformer_num_layers",
    "transformer_num_heads",
    "dropout",
    "position_dim",
    "architecture",
    "spectrum_channels",
    "route_basic_dim",
    "route_basic_feature_dim",
    "candidate_transformer_layers",
    "candidate_transformer_heads",
    "enable_spectrum_branch",
    "enable_candidate_attention",
    "enable_base_relative_branch",
    "enable_auxiliary_heads",
)


def _solver_config_payload(config: ExperimentConfig, checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_cfg = checkpoint.get("solver_config")
    if isinstance(checkpoint_cfg, dict):
        return checkpoint_cfg
    cfg = _solver_config(config, neural=False)
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(getattr(cfg, "__dict__", {}))


def _freeze_xlron_parameters(model: Any, mode: str) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode == "none":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        if mode == "head_only":
            trainable_markers = ("policy_head", "full_policy_head")
        elif mode == "ranker_light":
            trainable_markers = (
                "policy_head",
                "full_policy_head",
                "full_candidate_fusion",
                "base_relative_encoder",
                "candidate_transformer",
            )
        elif mode == "ranker_action_fusion":
            trainable_markers = (
                "policy_head",
                "full_policy_head",
                "full_candidate_fusion",
                "base_relative_encoder",
                "candidate_transformer",
                "action_encoder",
                "context_encoder",
                "full_route_project",
            )
        else:
            raise ValueError(f"Unsupported freeze mode: {mode}")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(any(marker in name for marker in trainable_markers))

    trainable = 0
    frozen = 0
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        if parameter.requires_grad:
            trainable += count
            if len(trainable_names) < 24:
                trainable_names.append(name)
        else:
            frozen += count
            if len(frozen_names) < 24:
                frozen_names.append(name)
    return {
        "freeze_mode": mode,
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "trainable_parameter_examples": trainable_names,
        "frozen_parameter_examples": frozen_names,
    }


def _build_optimizer(
    *,
    model: Any,
    args: argparse.Namespace,
    torch: Any,
    low_lr_markers: tuple[str, ...] = (),
) -> tuple[Any, list[Any], dict[str, Any]]:
    base_lr = float(args.learning_rate)
    low_lr = float(args.partial_unfreeze_lr)
    regular_params: list[Any] = []
    low_lr_params: list[Any] = []
    regular_names: list[str] = []
    low_lr_names: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if low_lr > 0.0 and any(marker and marker in name for marker in low_lr_markers):
            low_lr_params.append(parameter)
            if len(low_lr_names) < 24:
                low_lr_names.append(name)
        else:
            regular_params.append(parameter)
            if len(regular_names) < 24:
                regular_names.append(name)

    param_groups: list[dict[str, Any]] = []
    if regular_params:
        param_groups.append({"params": regular_params, "lr": base_lr})
    if low_lr_params:
        param_groups.append({"params": low_lr_params, "lr": low_lr})
    if not param_groups:
        raise RuntimeError("No trainable XLRON parameters")

    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=float(args.weight_decay))
    trainable = regular_params + low_lr_params
    return optimizer, trainable, {
        "base_lr": float(base_lr),
        "low_lr": float(low_lr) if low_lr_params else None,
        "regular_parameters": int(sum(int(parameter.numel()) for parameter in regular_params)),
        "low_lr_parameters": int(sum(int(parameter.numel()) for parameter in low_lr_params)),
        "regular_parameter_examples": regular_names,
        "low_lr_parameter_examples": low_lr_names,
        "low_lr_markers": list(low_lr_markers),
    }


def _parse_marker_tuple(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(text or "").split(",") if item.strip())


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
    if len(indices) == 0:
        n_max = int(np.asarray(data["candidate_mask"]).shape[1])
        return np.zeros((0, n_max), dtype=np.float32)
    model.eval()
    logits: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch_indices in _iter_batches(indices, int(batch_size), shuffle=False, rng=rng):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            raw_logits, _value = _xlron_forward(model, tensors, edge_index)
            logits.append(raw_logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits, axis=0)


def _evaluate_split(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
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
    return _selection_metrics(logits_np=logits, data=data, indices=indices)


def _rollout_validate_model(
    *,
    model: Any,
    config: ExperimentConfig,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    torch: Any,
) -> dict[str, Any] | None:
    if int(args.rollout_val_max_episodes) <= 0:
        return None
    return _rollout_validate_for_selection(
        model=model,
        config=config,
        output_dir=output_dir,
        args=args,
        device=device,
        torch=torch,
    )


def _score_and_details(
    *,
    args: argparse.Namespace,
    eval_metrics: dict[str, Any],
    rollout_val_eval: dict[str, Any] | None,
    reference_rollout_val_eval: dict[str, Any] | None,
) -> tuple[float, dict[str, Any]]:
    mode = str(args.checkpoint_selection)
    if mode.startswith("rollout_"):
        return _rollout_selection_metric(
            rollout_val_eval,
            reference_eval=reference_rollout_val_eval,
            args=args,
        )
    score = _score_checkpoint(mode=mode, eval_metrics=eval_metrics, rollout_val_eval=rollout_val_eval)
    return float(score), {"metric": mode, "score": float(score)}


def _checkpoint_payload(
    *,
    model: Any,
    source_checkpoint: dict[str, Any],
    config: ExperimentConfig,
    source_path: Path,
    reference_path: Path | None,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        key: source_checkpoint[key]
        for key in ARCHITECTURE_KEYS
        if key in source_checkpoint
    }
    payload.setdefault("policy", "top32_xlron_stabilized_ppo")
    payload.setdefault("n_max", int(_raw_int(config, "n_max", 32)))
    payload["model_state_dict"] = model.state_dict()
    payload["epoch"] = int(epoch)
    payload["config"] = {
        **dict(config.resolved),
        "initial_xlron_checkpoint": str(source_path),
        "reference_xlron_checkpoint": None if reference_path is None else str(reference_path),
        "checkpoint_selection": str(args.checkpoint_selection),
        "training_mode": "top32_xlron_counterfactual_rank_finetune",
    }
    payload["solver_config"] = _solver_config_payload(config, source_checkpoint)
    payload["metrics"] = metrics
    payload["history"] = history
    payload["args"] = vars(args)
    payload["training_mode"] = "top32_xlron_counterfactual_rank_finetune"
    return payload


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    source_checkpoint: dict[str, Any],
    config: ExperimentConfig,
    source_path: Path,
    reference_path: Path | None,
    epoch: int,
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            model=model,
            source_checkpoint=source_checkpoint,
            config=config,
            source_path=source_path,
            reference_path=reference_path,
            epoch=epoch,
            metrics=metrics,
            history=history,
            args=args,
        ),
        path,
    )


def train(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    from cse2026.ong_solver.models import require_torch

    _add_ong_source_path(config)
    torch = require_torch()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
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

    initial_checkpoint = _resolve_cli_path(args.initial_xlron_checkpoint) or _resolve_path(
        config,
        "top32_xlron_stabilized_ppo_checkpoint",
    )
    if initial_checkpoint is None:
        raise ValueError("--initial-xlron-checkpoint is required")
    model, source_checkpoint = _load_xlron_checkpoint_model(initial_checkpoint, device=device, torch=torch)
    freeze_info = _freeze_xlron_parameters(model, str(args.freeze_mode))

    reference_checkpoint = _resolve_cli_path(args.reference_xlron_checkpoint) or initial_checkpoint
    reference_model = None
    if reference_checkpoint is not None and float(args.reference_kl_weight) > 0.0:
        reference_model, _reference_checkpoint = _load_xlron_checkpoint_model(
            reference_checkpoint,
            device=device,
            torch=torch,
        )
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()

    low_lr_markers = _parse_marker_tuple(str(args.partial_unfreeze_low_lr_markers))
    optimizer, trainable, optimizer_info = _build_optimizer(
        model=model,
        args=args,
        torch=torch,
        low_lr_markers=low_lr_markers,
    )
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    freeze_history: list[dict[str, Any]] = [{"epoch": 0, "freeze": freeze_info, "optimizer": optimizer_info}]
    best_path = output_dir / "top32_xlron_counterfactual_rank_finetune_best.pt"
    best_score = -math.inf
    best_epoch = 0
    best_rollout_val_eval: dict[str, Any] | None = None
    reference_rollout_val_eval: dict[str, Any] | None = None
    epochs_without_improvement = 0

    if str(args.checkpoint_selection) in {"rollout_worst_bucket_score", "rollout_bucket_guard_score"}:
        reference_rollout_val_eval = _rollout_reference_for_selection(
            config=config,
            output_dir=output_dir,
            args=args,
        )
        print(
            json.dumps(
                _json_safe({"phase": "reference_rollout_val_eval", "reference_rollout_val_eval": reference_rollout_val_eval}),
                sort_keys=True,
            ),
            flush=True,
        )

    initial_eval = _evaluate_split(
        model=model,
        data=data,
        indices=splits["eval"],
        edge_index=edge_index,
        batch_size=int(args.batch_size),
        device=device,
        torch=torch,
    )
    initial_rollout_val_eval = None
    if str(args.checkpoint_selection).startswith("rollout_"):
        initial_rollout_val_eval = _rollout_validate_model(
            model=model,
            config=config,
            output_dir=output_dir,
            args=args,
            device=device,
            torch=torch,
        )
    initial_score, initial_score_details = _score_and_details(
        args=args,
        eval_metrics=initial_eval,
        rollout_val_eval=initial_rollout_val_eval,
        reference_rollout_val_eval=reference_rollout_val_eval,
    )
    best_score = float(initial_score)
    best_rollout_val_eval = initial_rollout_val_eval
    _save_checkpoint(
        path=best_path,
        model=model,
        source_checkpoint=source_checkpoint,
        config=config,
        source_path=initial_checkpoint,
        reference_path=reference_checkpoint,
        epoch=0,
        metrics={
            "phase": "initial",
            "score": float(initial_score),
            "selection_metric_details": initial_score_details,
            "eval": initial_eval,
            "rollout_val_eval": initial_rollout_val_eval,
            "reference_rollout_val_eval": reference_rollout_val_eval,
            "freeze": freeze_info,
            "optimizer": optimizer_info,
        },
        history=history,
        args=args,
        torch=torch,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "phase": "initial",
                    "score": float(initial_score),
                    "selection_metric_details": initial_score_details,
                    "eval": initial_eval,
                    "rollout_val_eval": initial_rollout_val_eval,
                    "reference_rollout_val_eval": reference_rollout_val_eval,
                    "freeze": freeze_info,
                    "optimizer": optimizer_info,
                }
            ),
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        if int(args.partial_unfreeze_epoch) > 0 and int(epoch) == int(args.partial_unfreeze_epoch):
            freeze_info = _freeze_xlron_parameters(model, str(args.partial_unfreeze_mode))
            optimizer, trainable, optimizer_info = _build_optimizer(
                model=model,
                args=args,
                torch=torch,
                low_lr_markers=low_lr_markers,
            )
            event = {
                "phase": "partial_unfreeze",
                "epoch": int(epoch),
                "freeze": freeze_info,
                "optimizer": optimizer_info,
            }
            freeze_history.append(event)
            history.append(event)
            print(json.dumps(_json_safe(event), sort_keys=True), flush=True)

        model.train()
        epoch_rows: list[dict[str, float]] = []
        started = time.perf_counter()
        batches = _iter_batches(splits["train"], int(args.batch_size), shuffle=True, rng=rng)
        for batch_index, batch_indices in enumerate(batches, start=1):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            logits, _value = _xlron_forward(model, tensors, edge_index)
            reference_logits = None
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits, _reference_value = _xlron_forward(reference_model, tensors, edge_index)
            loss, parts = _listwise_pairwise_loss(
                logits=logits,
                reference_logits=reference_logits,
                tensors=tensors,
                ce_weight=float(args.ce_weight),
                listwise_weight=float(args.listwise_weight),
                pairwise_weight=float(args.pairwise_weight),
                base_pairwise_weight=float(args.base_pairwise_weight),
                regression_weight=float(args.regression_weight),
                reference_kl_weight=float(args.reference_kl_weight),
                target_temperature=float(args.target_temperature),
                student_temperature=float(args.student_temperature),
                reference_temperature=float(args.reference_temperature),
                pairwise_margin=float(args.pairwise_margin),
                order_epsilon=float(args.order_epsilon),
                target_scale=float(args.target_scale),
                oracle_top1_weight=float(args.oracle_top1_weight),
                oracle_margin_weight=float(args.oracle_margin_weight),
                oracle_accepted_scale=float(args.oracle_accepted_scale),
                oracle_margin=float(args.oracle_margin),
                oracle_positive_only=bool(args.oracle_positive_only),
                torch=torch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable, float(args.grad_clip_norm))
            optimizer.step()
            epoch_rows.append(parts)
            if int(args.progress_every_batches) > 0 and batch_index % int(args.progress_every_batches) == 0:
                print(
                    json.dumps(
                        _json_safe(
                            {
                                "phase": "train_batch",
                                "epoch": int(epoch),
                                "batch": int(batch_index),
                                "batches": int(len(batches)),
                                "loss_mean": float(np.mean([item["loss"] for item in epoch_rows])),
                                "top1_accuracy_mean": float(np.mean([item["top1_accuracy"] for item in epoch_rows])),
                            }
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )

        train_eval = _evaluate_split(
            model=model,
            data=data,
            indices=splits["train"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        eval_metrics = _evaluate_split(
            model=model,
            data=data,
            indices=splits["eval"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        rollout_val_eval = None
        if str(args.checkpoint_selection).startswith("rollout_"):
            rollout_val_eval = _rollout_validate_model(
                model=model,
                config=config,
                output_dir=output_dir,
                args=args,
                device=device,
                torch=torch,
            )
        score, score_details = _score_and_details(
            args=args,
            eval_metrics=eval_metrics,
            rollout_val_eval=rollout_val_eval,
            reference_rollout_val_eval=reference_rollout_val_eval,
        )
        row = {
            "phase": "epoch",
            "epoch": int(epoch),
            "score": float(score),
            "selection_metric_details": score_details,
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])) if epoch_rows else None,
            "train_ce": float(np.mean([item["ce"] for item in epoch_rows])) if epoch_rows else None,
            "train_listwise": float(np.mean([item["listwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise": float(np.mean([item["pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_base_pairwise": float(np.mean([item["base_pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_regression": float(np.mean([item["regression"] for item in epoch_rows])) if epoch_rows else None,
            "train_reference_kl": float(np.mean([item["reference_kl"] for item in epoch_rows])) if epoch_rows else None,
            "train_oracle_top1": float(np.mean([item["oracle_top1"] for item in epoch_rows])) if epoch_rows else None,
            "train_oracle_margin": float(np.mean([item["oracle_margin"] for item in epoch_rows])) if epoch_rows else None,
            "train_oracle_top1_accuracy": float(np.mean([item["oracle_top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
            "train_oracle_usable_groups_mean": float(np.mean([item["oracle_usable_groups"] for item in epoch_rows])) if epoch_rows else None,
            "train_oracle_nonbase_groups_mean": float(np.mean([item["oracle_nonbase_groups"] for item in epoch_rows])) if epoch_rows else None,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
            "train_label_weight_mean": float(np.mean([item["label_weight_mean"] for item in epoch_rows])) if epoch_rows else None,
            "train_best_label_weight_mean": float(np.mean([item["best_label_weight_mean"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise_pairs_mean": float(np.mean([item["pairwise_pairs"] for item in epoch_rows])) if epoch_rows else None,
            "train_base_positive_pairs_mean": float(np.mean([item["base_positive_pairs"] for item in epoch_rows])) if epoch_rows else None,
            "train_eval": train_eval,
            "eval": eval_metrics,
            "rollout_val_eval": rollout_val_eval,
            "freeze": freeze_info,
            "optimizer": optimizer_info,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        history.append(row)
        print(json.dumps(_json_safe(row), sort_keys=True), flush=True)

        if float(score) > float(best_score):
            best_score = float(score)
            best_epoch = int(epoch)
            best_rollout_val_eval = rollout_val_eval
            epochs_without_improvement = 0
            _save_checkpoint(
                path=best_path,
                model=model,
                source_checkpoint=source_checkpoint,
                config=config,
                source_path=initial_checkpoint,
                reference_path=reference_checkpoint,
                epoch=int(epoch),
                metrics={
                    "phase": "best",
                    "score": float(score),
                    "selection_metric_details": score_details,
                    "eval": eval_metrics,
                    "rollout_val_eval": rollout_val_eval,
                    "reference_rollout_val_eval": reference_rollout_val_eval,
                    "freeze": freeze_info,
                    "optimizer": optimizer_info,
                },
                history=history,
                args=args,
                torch=torch,
            )
        else:
            epochs_without_improvement += 1
            if int(args.early_stop_patience) > 0 and epochs_without_improvement >= int(args.early_stop_patience):
                print(
                    json.dumps(
                        _json_safe(
                            {
                                "phase": "early_stop",
                                "epoch": int(epoch),
                                "best_epoch": int(best_epoch),
                                "best_score": float(best_score),
                                "patience": int(args.early_stop_patience),
                            }
                        ),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    summary = {
        "stage": "train_top32_xlron_counterfactual_rank_finetune",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "groups": int(len(data["group_ids"])),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "freeze": freeze_info,
        "initial_checkpoint": str(initial_checkpoint),
        "reference_checkpoint": None if reference_checkpoint is None else str(reference_checkpoint),
        "initial_eval": initial_eval,
        "initial_rollout_val_eval": initial_rollout_val_eval,
        "reference_rollout_val_eval": reference_rollout_val_eval,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_rollout_val_eval": best_rollout_val_eval,
        "freeze_history": freeze_history,
        "history": history,
        "args": vars(args),
    }
    _write_json(output_dir / "top32_xlron_counterfactual_rank_finetune_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Top32 XLRON on stable counterfactual Top-N labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-xlron-checkpoint", required=True)
    parser.add_argument("--reference-xlron-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument(
        "--freeze-mode",
        choices=("none", "head_only", "ranker_light", "ranker_action_fusion"),
        default="ranker_light",
    )
    parser.add_argument("--partial-unfreeze-epoch", type=int, default=0)
    parser.add_argument(
        "--partial-unfreeze-mode",
        choices=("none", "head_only", "ranker_light", "ranker_action_fusion"),
        default="ranker_action_fusion",
    )
    parser.add_argument("--partial-unfreeze-lr", type=float, default=0.0)
    parser.add_argument(
        "--partial-unfreeze-low-lr-markers",
        default="action_encoder,context_encoder,full_route_project,spectrum_encoder",
    )
    parser.add_argument("--ce-weight", type=float, default=0.20)
    parser.add_argument("--listwise-weight", type=float, default=1.00)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--base-pairwise-weight", type=float, default=0.60)
    parser.add_argument("--regression-weight", type=float, default=0.03)
    parser.add_argument("--reference-kl-weight", type=float, default=3.00)
    parser.add_argument("--oracle-top1-weight", type=float, default=0.0)
    parser.add_argument("--oracle-margin-weight", type=float, default=0.0)
    parser.add_argument("--oracle-accepted-scale", type=float, default=8.0)
    parser.add_argument("--oracle-margin", type=float, default=0.35)
    parser.add_argument("--oracle-positive-only", action="store_true")
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--reference-temperature", type=float, default=2.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.12)
    parser.add_argument("--order-epsilon", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=4.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument(
        "--checkpoint-selection",
        choices=(
            "eval_accepted_delta",
            "eval_target_delta",
            "rollout_accepted",
            "rollout_reward",
            "rollout_worst_bucket_score",
            "rollout_bucket_guard_score",
        ),
        default="rollout_accepted",
    )
    parser.add_argument("--rollout-val-split", default="val")
    parser.add_argument("--rollout-val-max-episodes", type=int, default=8)
    parser.add_argument("--rollout-val-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--rollout-val-episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--rollout-val-slices", type=int, default=1)
    parser.add_argument("--rollout-val-slice-stride", type=int, default=1)
    parser.add_argument("--rollout-val-episode-offset", type=int, default=0)
    parser.add_argument("--rollout-val-reference-policy", default="energy-aware-ksp-bm-ff")
    parser.add_argument("--rollout-worst-bucket-penalty", type=float, default=4.0)
    parser.add_argument("--rollout-negative-bucket-penalty", type=float, default=0.0)
    parser.add_argument("--rollout-protected-buckets", default="")
    parser.add_argument("--rollout-protected-bucket-min-delta", type=int, default=0)
    parser.add_argument("--rollout-protected-bucket-penalty", type=float, default=0.0)
    parser.add_argument("--rollout-runtime-guard-buckets", default="")
    parser.add_argument("--rollout-runtime-guard-bucket-margins", default="")
    parser.add_argument("--rollout-runtime-guard-min-margin", type=float, default=0.0)
    parser.add_argument("--rollout-runtime-guard-base-index", type=int, default=0)
    parser.add_argument("--rollout-live-risk-selector-path", default="")
    parser.add_argument("--rollout-live-risk-selector-threshold", type=float, default=-1.0)
    parser.add_argument("--rollout-live-risk-selector-buckets", default="")
    parser.add_argument("--rollout-live-risk-selector-base-index", type=int, default=0)
    parser.add_argument("--rollout-val-run-name-suffix", default="_cf_rank")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = train(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
