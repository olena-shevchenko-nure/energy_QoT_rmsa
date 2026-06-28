#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.train_dqn import (
    _batch_to_arrays,
    _device,
    _model_forward,
    _raw_int,
    _stack_state_arrays,
)
from cse2026.ong_solver import GnnCnnDqnOngSolver

from train_neural_stable_override_selector import (
    _batch_tensors,
    _build_model,
    _iter_batches,
    _json_safe,
    _load_dataset,
    _make_group_split,
    _predict_logits,
    _resolve_cli_path,
    _resolve_path,
    _write_json,
)


def _load_traffic(
    config: ExperimentConfig,
    split: str,
    max_episodes: int,
    episode_selection: str,
):
    import pandas as pd

    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    traffic_path = config.dataset_path / "traffic" / f"{split}.parquet"
    traffic = pd.read_parquet(traffic_path)
    episodes = traffic.drop_duplicates("episode_id").reset_index(drop=True)
    episode_ids = tuple(str(value) for value in episodes["episode_id"].tolist())
    if max_episodes <= 0:
        return traffic, episode_ids

    mode = str(episode_selection).strip().lower().replace("-", "_")
    if mode == "first":
        return traffic, episode_ids[:max_episodes]
    if mode != "stratified":
        raise ValueError(f"Unsupported episode selection mode: {episode_selection}")

    keys = ["traffic_scenario", "load_name"]
    if not all(key in episodes.columns for key in keys):
        return traffic, episode_ids[:max_episodes]
    groups = [
        [str(value) for value in group["episode_id"].tolist()]
        for _key, group in episodes.groupby(keys, sort=False)
    ]
    selected: list[str] = []
    round_index = 0
    while len(selected) < max_episodes:
        added = False
        for group in groups:
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) >= max_episodes:
                    break
        if not added:
            break
        round_index += 1
    return traffic, tuple(selected)


def _model_action_index(
    *,
    model: Any,
    batch: Any,
    cfg: Any,
    device: str,
    torch: Any,
) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1
    arrays = _batch_to_arrays(batch, cfg)
    tensors = _stack_state_arrays([arrays], device, torch)
    edge_index = torch.as_tensor(np.asarray(batch.state.edge_index, dtype=np.int64), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = _model_forward(model, tensors, edge_index).masked_fill(~tensors["candidate_mask"], -1.0e9)
        selected = int(logits.argmax(dim=1).detach().cpu().numpy()[0])
    if selected < 0 or not bool(batch.candidate_mask[selected]):
        return int(valid[0])
    return int(selected)


def _rollout_validate_model(
    *,
    model: Any,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    episode_selection: str,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    traffic, episode_ids = _load_traffic(config, split, max_episodes, episode_selection)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    cfg = solver.config
    run_path = output_dir / f"rollout_validate_{split}"
    model.eval()

    requests = 0
    accepted = 0
    no_candidate = 0
    invalid_selected = 0
    total_reward = 0.0
    selected_indices: list[int] = []
    started = time.perf_counter()

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.iloc[:max_requests_per_episode].reset_index(drop=True)
        traffic_jsonl = _traffic_jsonl_for_episode(run_path, episode_id, episode)
        env = _make_env(
            episode_id=f"cf_rank_val_{episode_id}",
            traffic_path=traffic_jsonl,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))
        while True:
            batch = solver.candidate_batch(env)
            valid = np.flatnonzero(batch.candidate_mask.astype(bool))
            if valid.size == 0:
                action = solver.adapter(env).block_action(env)
                no_candidate += 1
            else:
                selected_index = _model_action_index(
                    model=model,
                    batch=batch,
                    cfg=cfg,
                    device=device,
                    torch=torch,
                )
                if selected_index < 0 or not bool(batch.candidate_mask[selected_index]):
                    invalid_selected += 1
                    selected_index = int(valid[0])
                selected_indices.append(int(selected_index))
                action = batch.topn[selected_index].action

            _observation, reward, terminated, truncated, info = env.step(int(action))
            accepted += int(bool(info.get("accepted", False)))
            total_reward += float(reward)
            requests += 1
            if bool(terminated) or bool(truncated):
                break

    return {
        "split": str(split),
        "episode_selection": str(episode_selection),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": float(total_reward),
        "no_candidate_requests": int(no_candidate),
        "invalid_selected": int(invalid_selected),
        "mean_selected_index": float(np.mean(selected_indices)) if selected_indices else None,
        "elapsed_sec": float(time.perf_counter() - started),
    }


def _listwise_pairwise_loss(
    *,
    logits: Any,
    reference_logits: Any | None,
    tensors: dict[str, Any],
    ce_weight: float,
    listwise_weight: float,
    pairwise_weight: float,
    base_pairwise_weight: float,
    regression_weight: float,
    reference_kl_weight: float,
    target_temperature: float,
    student_temperature: float,
    reference_temperature: float,
    pairwise_margin: float,
    order_epsilon: float,
    target_scale: float,
    oracle_top1_weight: float = 0.0,
    oracle_margin_weight: float = 0.0,
    oracle_accepted_scale: float = 8.0,
    oracle_margin: float = 0.35,
    oracle_positive_only: bool = False,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    label_mask = tensors["label_mask"] & tensors["candidate_mask"]
    candidate_mask = tensors["candidate_mask"]
    target = tensors["target_delta"]
    accepted = tensors.get("accepted_delta_vs_base", target)
    label_weight = tensors.get("label_weight", torch.ones_like(target))
    label_weight = torch.nan_to_num(label_weight.to(dtype=target.dtype), nan=0.0, posinf=0.0, neginf=0.0)
    label_weight = torch.where(label_mask, label_weight.clamp_min(0.0), torch.zeros_like(label_weight))
    usable = label_mask.sum(dim=1) >= 2
    if not bool(usable.any()):
        raise RuntimeError("Batch has no usable labeled groups")

    row_index = torch.arange(logits.shape[0], device=logits.device)
    masked_logits = logits.masked_fill(~label_mask, -1.0e9)
    masked_target = target.masked_fill(~label_mask, -1.0e9)
    best_index = masked_target.argmax(dim=1)
    best_weight = label_weight[row_index, best_index].clamp_min(1.0e-6)
    ce_raw = torch.nn.functional.cross_entropy(masked_logits[usable], best_index[usable], reduction="none")
    ce = (ce_raw * best_weight[usable]).sum() / best_weight[usable].sum().clamp_min(1.0e-6)
    prediction = masked_logits[usable].argmax(dim=1)
    top1 = (prediction == best_index[usable]).to(dtype=torch.float32).mean()

    target_temp = max(float(target_temperature), 1.0e-6)
    student_temp = max(float(student_temperature), 1.0e-6)
    with torch.no_grad():
        target_score = (target / target_temp) + torch.log(label_weight.clamp_min(1.0e-6))
        target_prob = torch.nn.functional.softmax(target_score.masked_fill(~label_mask, -1.0e9), dim=1)
    student_log_prob = torch.nn.functional.log_softmax(masked_logits / student_temp, dim=1)
    listwise_raw = torch.nn.functional.kl_div(
        student_log_prob[usable],
        target_prob[usable],
        reduction="none",
    ).sum(dim=1) * (student_temp * student_temp)
    listwise = (listwise_raw * best_weight[usable]).sum() / best_weight[usable].sum().clamp_min(1.0e-6)

    target_diff = target[:, :, None] - target[:, None, :]
    logit_diff = logits[:, :, None] - logits[:, None, :]
    pair_mask = label_mask[:, :, None] & label_mask[:, None, :] & (target_diff > float(order_epsilon))
    if bool(pair_mask.any()):
        pair_values = torch.nn.functional.relu(float(pairwise_margin) - logit_diff).masked_select(pair_mask)
        pair_weights = torch.sqrt(
            label_weight[:, :, None].clamp_min(1.0e-6) * label_weight[:, None, :].clamp_min(1.0e-6)
        ).masked_select(pair_mask)
        pairwise = (pair_values * pair_weights).sum() / pair_weights.sum().clamp_min(1.0e-6)
    else:
        pairwise = ce * 0.0

    base_index = tensors["base_index"].clamp(0, logits.shape[1] - 1)
    base_target = target[row_index, base_index]
    base_logit = logits[row_index, base_index]
    positive_vs_base = label_mask & (target > (base_target[:, None] + float(order_epsilon)))
    if bool(positive_vs_base.any()):
        base_pair_values = torch.nn.functional.relu(
            float(pairwise_margin) - (logits - base_logit[:, None])
        ).masked_select(positive_vs_base)
        base_pair_weights = label_weight.masked_select(positive_vs_base).clamp_min(1.0e-6)
        base_pairwise = (base_pair_values * base_pair_weights).sum() / base_pair_weights.sum().clamp_min(1.0e-6)
    else:
        base_pairwise = ce * 0.0

    oracle_top1 = ce * 0.0
    oracle_margin_loss = ce * 0.0
    oracle_usable_count = 0
    oracle_nonbase_count = 0
    oracle_top1_accuracy = ce.detach() * 0.0
    if float(oracle_top1_weight) > 0.0 or float(oracle_margin_weight) > 0.0:
        oracle_score = accepted * float(oracle_accepted_scale) + target
        masked_oracle_score = oracle_score.masked_fill(~label_mask, -1.0e9)
        oracle_index = masked_oracle_score.argmax(dim=1)
        oracle_accepted = accepted[row_index, oracle_index]
        oracle_usable = usable
        if bool(oracle_positive_only):
            oracle_usable = oracle_usable & (oracle_accepted > float(order_epsilon))
        if bool(oracle_usable.any()):
            oracle_usable_count = int(oracle_usable.sum().detach().cpu())
            oracle_weight = label_weight[row_index, oracle_index].clamp_min(1.0e-6)
            oracle_top1_raw = torch.nn.functional.cross_entropy(
                masked_logits[oracle_usable],
                oracle_index[oracle_usable],
                reduction="none",
            )
            oracle_top1 = (
                oracle_top1_raw * oracle_weight[oracle_usable]
            ).sum() / oracle_weight[oracle_usable].sum().clamp_min(1.0e-6)
            oracle_prediction = masked_logits[oracle_usable].argmax(dim=1)
            oracle_top1_accuracy = (oracle_prediction == oracle_index[oracle_usable]).to(dtype=torch.float32).mean()
            oracle_nonbase = oracle_index != base_index
            oracle_nonbase_count = int((oracle_usable & oracle_nonbase).sum().detach().cpu())
            margin_rows = oracle_usable & oracle_nonbase
            if bool(margin_rows.any()):
                oracle_logit = logits[row_index, oracle_index]
                oracle_margin_values = torch.nn.functional.relu(
                    float(oracle_margin) - (oracle_logit - base_logit)
                ).masked_select(margin_rows)
                oracle_margin_weights = oracle_weight.masked_select(margin_rows)
                oracle_margin_loss = (
                    oracle_margin_values * oracle_margin_weights
                ).sum() / oracle_margin_weights.sum().clamp_min(1.0e-6)

    scaled_target = torch.clamp(target / max(float(target_scale), 1.0e-6), min=-4.0, max=4.0)
    regression_raw = torch.nn.functional.smooth_l1_loss(
        logits.masked_select(label_mask),
        scaled_target.masked_select(label_mask),
        reduction="none",
    )
    regression_weights = label_weight.masked_select(label_mask).clamp_min(1.0e-6)
    regression = (regression_raw * regression_weights).sum() / regression_weights.sum().clamp_min(1.0e-6)

    reference_kl = ce * 0.0
    if reference_logits is not None and float(reference_kl_weight) > 0.0:
        ref_temp = max(float(reference_temperature), 1.0e-6)
        student_ref_log_prob = torch.nn.functional.log_softmax(
            logits.masked_fill(~candidate_mask, -1.0e9) / ref_temp,
            dim=1,
        )
        with torch.no_grad():
            reference_prob = torch.nn.functional.softmax(
                reference_logits.masked_fill(~candidate_mask, -1.0e9) / ref_temp,
                dim=1,
            )
        reference_kl = torch.nn.functional.kl_div(
            student_ref_log_prob,
            reference_prob,
            reduction="batchmean",
        ) * (ref_temp * ref_temp)

    total = (
        float(ce_weight) * ce
        + float(listwise_weight) * listwise
        + float(pairwise_weight) * pairwise
        + float(base_pairwise_weight) * base_pairwise
        + float(regression_weight) * regression
        + float(reference_kl_weight) * reference_kl
        + float(oracle_top1_weight) * oracle_top1
        + float(oracle_margin_weight) * oracle_margin_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "ce": float(ce.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "base_pairwise": float(base_pairwise.detach().cpu()),
        "regression": float(regression.detach().cpu()),
        "reference_kl": float(reference_kl.detach().cpu()),
        "oracle_top1": float(oracle_top1.detach().cpu()),
        "oracle_margin": float(oracle_margin_loss.detach().cpu()),
        "oracle_top1_accuracy": float(oracle_top1_accuracy.detach().cpu()),
        "top1_accuracy": float(top1.detach().cpu()),
        "usable_groups": int(usable.sum().detach().cpu()),
        "oracle_usable_groups": int(oracle_usable_count),
        "oracle_nonbase_groups": int(oracle_nonbase_count),
        "pairwise_pairs": int(pair_mask.sum().detach().cpu()),
        "base_positive_pairs": int(positive_vs_base.sum().detach().cpu()),
        "label_weight_mean": float(label_weight.masked_select(label_mask).mean().detach().cpu()),
        "best_label_weight_mean": float(best_weight.masked_select(usable).mean().detach().cpu()),
    }


def _selection_metrics(
    *,
    logits_np: np.ndarray,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    if len(indices) == 0:
        return {"groups": 0}
    candidate_mask = np.asarray(data["candidate_mask"])[indices].astype(bool)
    label_mask = np.asarray(data["label_mask"])[indices].astype(bool) & candidate_mask
    target = np.nan_to_num(np.asarray(data["target_delta"])[indices].astype(np.float32), nan=0.0)
    accepted = np.nan_to_num(np.asarray(data["accepted_delta_vs_base"])[indices].astype(np.float32), nan=0.0)
    base_index = np.asarray(data["base_index"])[indices].astype(np.int64)
    row = np.arange(len(indices), dtype=np.int64)

    runtime_logits = np.where(candidate_mask, logits_np, -1.0e9)
    selected = runtime_logits.argmax(axis=1).astype(np.int64)
    labeled_logits = np.where(label_mask, logits_np, -1.0e9)
    labeled_selected = labeled_logits.argmax(axis=1).astype(np.int64)
    oracle_target = np.where(label_mask, target, -1.0e9).argmax(axis=1).astype(np.int64)
    oracle_accepted = np.where(label_mask, accepted, -1.0e9).argmax(axis=1).astype(np.int64)

    selected_labeled = label_mask[row, selected]
    selected_accepted_delta = np.where(selected_labeled, accepted[row, selected], 0.0)
    selected_target_delta = np.where(selected_labeled, target[row, selected], 0.0)
    labeled_selected_accepted_delta = accepted[row, labeled_selected]
    oracle_target_delta = target[row, oracle_target]
    oracle_target_accepted_delta = accepted[row, oracle_target]
    oracle_accepted_delta = accepted[row, oracle_accepted]
    base_logit = logits_np[row, np.clip(base_index, 0, logits_np.shape[1] - 1)]
    selected_margin = logits_np[row, selected] - base_logit
    selected_nonbase = selected != base_index

    def _rate(mask: np.ndarray) -> float:
        return float(mask.mean()) if mask.size else 0.0

    return {
        "groups": int(len(indices)),
        "runtime_selected_total_accepted_delta": float(selected_accepted_delta.sum()),
        "runtime_selected_mean_accepted_delta": float(selected_accepted_delta.mean()),
        "runtime_selected_total_target_delta": float(selected_target_delta.sum()),
        "runtime_selected_mean_target_delta": float(selected_target_delta.mean()),
        "runtime_selected_win_rate": _rate(selected_accepted_delta > 0.0),
        "runtime_selected_loss_rate": _rate(selected_accepted_delta < 0.0),
        "runtime_selected_tie_rate": _rate(selected_accepted_delta == 0.0),
        "runtime_selected_nonbase_rate": _rate(selected_nonbase),
        "runtime_selected_unlabeled_rate": _rate(~selected_labeled),
        "runtime_selected_margin_quantiles": [
            float(np.quantile(selected_margin, q)) for q in (0.0, 0.5, 0.75, 0.9, 0.95, 1.0)
        ],
        "labeled_top1_total_accepted_delta": float(labeled_selected_accepted_delta.sum()),
        "labeled_top1_mean_accepted_delta": float(labeled_selected_accepted_delta.mean()),
        "oracle_target_total_delta": float(np.maximum(oracle_target_delta, 0.0).sum()),
        "oracle_target_if_always_best_delta": float(oracle_target_delta.sum()),
        "oracle_target_accepted_total_delta": float(oracle_target_accepted_delta.sum()),
        "oracle_accepted_total_delta": float(np.maximum(oracle_accepted_delta, 0.0).sum()),
        "oracle_groups_with_accepted_win": int((oracle_accepted_delta > 0.0).sum()),
        "oracle_target_top1_accuracy": float((selected == oracle_target).mean()),
        "base_selected_rate": float((selected == base_index).mean()),
    }


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


def _score_checkpoint(
    *,
    mode: str,
    eval_metrics: dict[str, Any],
    rollout_val_eval: dict[str, Any] | None,
) -> float:
    if mode == "rollout_accepted":
        return float((rollout_val_eval or {}).get("accepted") or 0.0)
    if mode == "rollout_reward":
        return float((rollout_val_eval or {}).get("total_reward") or 0.0)
    if mode == "eval_accepted_delta":
        return float(eval_metrics.get("runtime_selected_total_accepted_delta") or 0.0)
    if mode == "eval_target_delta":
        return float(eval_metrics.get("runtime_selected_total_target_delta") or 0.0)
    raise ValueError(f"Unsupported checkpoint selection: {mode}")


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    epoch: int,
    config: ExperimentConfig,
    model_info: dict[str, Any],
    history: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
    rollout_val_eval: dict[str, Any] | None,
    args: argparse.Namespace,
    torch: Any,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "target_model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "config": {
                "hidden_dim": int(model_info["hidden_dim"]),
                "n_max": int(_raw_int(config, "n_max", 32)),
                "q_score_mode": "raw",
                "residual_scale": 1.0,
                "residual_delta_clip": 0.0,
                "stage": "full_dqn_counterfactual_rank_finetune",
                "freeze_gnn_slot": bool(args.freeze_gnn_slot),
            },
            "model_info": model_info,
            "eval": eval_metrics,
            "rollout_val_eval": rollout_val_eval,
            "history": history,
            "args": vars(args),
        },
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
    if reference_checkpoint is not None and float(args.reference_kl_weight) > 0.0:
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
    best_path = output_dir / "full_dqn_counterfactual_rank_finetune.pt"
    best_score = -math.inf
    best_epoch = -1
    best_rollout_val_eval: dict[str, Any] | None = None

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
            split=str(args.rollout_val_split),
            max_episodes=int(args.rollout_val_max_episodes),
            max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
            episode_selection=str(args.rollout_val_episode_selection),
            device=device,
            torch=torch,
        )
    initial_score = _score_checkpoint(
        mode=str(args.checkpoint_selection),
        eval_metrics=initial_eval,
        rollout_val_eval=initial_rollout_val_eval,
    )
    best_score = float(initial_score)
    best_epoch = 0
    best_rollout_val_eval = initial_rollout_val_eval
    _save_checkpoint(
        path=best_path,
        model=model,
        epoch=0,
        config=config,
        model_info=model_info,
        history=history,
        eval_metrics=initial_eval,
        rollout_val_eval=initial_rollout_val_eval,
        args=args,
        torch=torch,
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "phase": "initial",
                    "score": float(initial_score),
                    "eval": initial_eval,
                    "rollout_val_eval": initial_rollout_val_eval,
                }
            ),
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows: list[dict[str, float]] = []
        started = time.perf_counter()
        batches = _iter_batches(splits["train"], int(args.batch_size), shuffle=True, rng=rng)
        for batch_index, batch_indices in enumerate(batches, start=1):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            logits = _model_forward(model, tensors, edge_index)
            reference_logits = None
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits = _model_forward(reference_model, tensors, edge_index)
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
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection),
                device=device,
                torch=torch,
            )
        score = _score_checkpoint(
            mode=str(args.checkpoint_selection),
            eval_metrics=eval_metrics,
            rollout_val_eval=rollout_val_eval,
        )
        row = {
            "phase": "epoch",
            "epoch": int(epoch),
            "score": float(score),
            "train_loss": float(np.mean([item["loss"] for item in epoch_rows])) if epoch_rows else None,
            "train_ce": float(np.mean([item["ce"] for item in epoch_rows])) if epoch_rows else None,
            "train_listwise": float(np.mean([item["listwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise": float(np.mean([item["pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_base_pairwise": float(np.mean([item["base_pairwise"] for item in epoch_rows])) if epoch_rows else None,
            "train_regression": float(np.mean([item["regression"] for item in epoch_rows])) if epoch_rows else None,
            "train_reference_kl": float(np.mean([item["reference_kl"] for item in epoch_rows])) if epoch_rows else None,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
            "train_label_weight_mean": float(np.mean([item["label_weight_mean"] for item in epoch_rows])) if epoch_rows else None,
            "train_best_label_weight_mean": float(np.mean([item["best_label_weight_mean"] for item in epoch_rows])) if epoch_rows else None,
            "train_pairwise_pairs_mean": float(np.mean([item["pairwise_pairs"] for item in epoch_rows])) if epoch_rows else None,
            "train_base_positive_pairs_mean": float(np.mean([item["base_positive_pairs"] for item in epoch_rows])) if epoch_rows else None,
            "train_eval": train_eval,
            "eval": eval_metrics,
            "rollout_val_eval": rollout_val_eval,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        history.append(row)
        print(json.dumps(_json_safe(row), sort_keys=True), flush=True)

        if float(score) >= float(best_score):
            best_score = float(score)
            best_epoch = int(epoch)
            best_rollout_val_eval = rollout_val_eval
            _save_checkpoint(
                path=best_path,
                model=model,
                epoch=int(epoch),
                config=config,
                model_info=model_info,
                history=history,
                eval_metrics=eval_metrics,
                rollout_val_eval=rollout_val_eval,
                args=args,
                torch=torch,
            )

    summary = {
        "stage": "train_full_dqn_counterfactual_rank_finetune",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "model_info": model_info,
        "groups": int(len(data["group_ids"])),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "initial_eval": initial_eval,
        "initial_rollout_val_eval": initial_rollout_val_eval,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_rollout_val_eval": best_rollout_val_eval,
        "history": history,
        "args": vars(args),
    }
    _write_json(output_dir / "full_dqn_counterfactual_rank_finetune_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune full DQN on windowed counterfactual Top-N labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument("--reference-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.50)
    parser.add_argument("--listwise-weight", type=float, default=1.00)
    parser.add_argument("--pairwise-weight", type=float, default=0.75)
    parser.add_argument("--base-pairwise-weight", type=float, default=0.50)
    parser.add_argument("--regression-weight", type=float, default=0.05)
    parser.add_argument("--reference-kl-weight", type=float, default=0.50)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--student-temperature", type=float, default=1.0)
    parser.add_argument("--reference-temperature", type=float, default=2.0)
    parser.add_argument("--pairwise-margin", type=float, default=0.15)
    parser.add_argument("--order-epsilon", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=4.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--freeze-gnn-slot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("eval_accepted_delta", "eval_target_delta", "rollout_accepted", "rollout_reward"),
        default="rollout_accepted",
    )
    parser.add_argument("--rollout-val-split", default="val")
    parser.add_argument("--rollout-val-max-episodes", type=int, default=4)
    parser.add_argument("--rollout-val-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--rollout-val-episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--progress-every-batches", type=int, default=25)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = train(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
