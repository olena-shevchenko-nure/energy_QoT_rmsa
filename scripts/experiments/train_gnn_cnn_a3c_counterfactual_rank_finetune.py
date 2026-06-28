#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_int,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.train_dqn import _batch_to_arrays, _device, _stack_state_arrays
from cse2026.experiments.eon.train_gnn_cnn_a3c_windowed_online import _model_forward as _a3c_forward
from cse2026.ong_solver import GnnCnnDqnOngSolver

from train_full_dqn_counterfactual_rank_finetune import (
    _batch_tensors,
    _iter_batches,
    _json_safe,
    _listwise_pairwise_loss,
    _load_dataset,
    _make_group_split,
    _score_checkpoint,
    _selection_metrics,
    _write_json,
)
from train_full_dqn_orate60_distill import _parse_weight_rules, _resolve_cli_path
from train_gnn_cnn_a3c_distill import _load_a3c_model, _load_traffic, _rollout_validate_a3c


def _predict_logits_a3c(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    torch: Any,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = np.asarray(indices[start : start + int(batch_size)], dtype=np.int64)
        tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
        with torch.no_grad():
            logits, _value = _a3c_forward(model, tensors, edge_index)
        outputs.append(logits.detach().cpu().numpy().astype(np.float32))
    if not outputs:
        n_max = int(np.asarray(data["candidate_mask"]).shape[1])
        return np.zeros((0, n_max), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def _evaluate_split_a3c(
    *,
    model: Any,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    edge_index: Any,
    batch_size: int,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    logits = _predict_logits_a3c(
        model=model,
        data=data,
        indices=indices,
        edge_index=edge_index,
        batch_size=batch_size,
        device=device,
        torch=torch,
    )
    return _selection_metrics(logits_np=logits, data=data, indices=indices)


def _masked_kl_loss(
    *,
    logits: Any,
    reference_logits: Any,
    candidate_mask: Any,
    sample_weight: Any | None = None,
    temperature: float,
    torch: Any,
) -> Any:
    temp = max(float(temperature), 1e-6)
    student_logits = logits.masked_fill(~candidate_mask, -1e9) / temp
    teacher_logits = reference_logits.masked_fill(~candidate_mask, -1e9) / temp
    student_log_probs = torch.nn.functional.log_softmax(student_logits, dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)
    kl_per = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1) * (temp * temp)
    if sample_weight is None:
        return kl_per.mean()
    weights = sample_weight.to(dtype=kl_per.dtype)
    return (kl_per * weights).sum() / torch.clamp(weights.sum(), min=1e-6)


def _scenario_load_weight(
    *,
    scenario: str,
    load: str,
    rules: dict[tuple[str, str], float],
) -> float:
    normalized_scenario = str(scenario)
    normalized_load = str(load)
    for key in (
        (normalized_scenario, normalized_load),
        (normalized_scenario, "*"),
        ("*", normalized_load),
        ("*", "*"),
    ):
        if key in rules:
            return float(rules[key])
    return 1.0


def _collect_preservation_examples(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    episode_selection: str,
    max_examples: int,
    preservation_weight_rules: dict[tuple[str, str], float],
    behavior_model: Any,
    device: str,
    torch: Any,
    rng: np.random.Generator,
    print_every_episodes: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    traffic, episode_ids = _load_traffic(
        config=config,
        split=str(split),
        max_episodes=int(max_episodes),
        episode_selection=str(episode_selection),
    )
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    cfg = solver.config
    run_path = output_dir / f"preservation_collect_{split}"
    examples: list[dict[str, Any]] = []
    edge_index: np.ndarray | None = None
    requests = 0
    accepted = 0
    no_candidate = 0
    seen_candidate_states = 0
    max_examples_int = max(0, int(max_examples))
    started = time.perf_counter()
    behavior_model.eval()

    for episode_position, episode_id in enumerate(episode_ids):
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if int(max_requests_per_episode) > 0:
            episode = episode.iloc[: int(max_requests_per_episode)].reset_index(drop=True)
        traffic_jsonl = _traffic_jsonl_for_episode(run_path, episode_id, episode)
        traffic_scenario = str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else ""
        load_name = str(episode["load_name"].iloc[0]) if "load_name" in episode else ""
        preservation_weight = _scenario_load_weight(
            scenario=traffic_scenario,
            load=load_name,
            rules=preservation_weight_rules,
        )
        env = _make_env(
            episode_id=episode_id,
            traffic_path=traffic_jsonl,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))

        episode_requests = 0
        episode_accepted = 0
        episode_collected = 0
        while True:
            batch = solver.candidate_batch(env)
            valid = np.flatnonzero(batch.candidate_mask.astype(bool))
            if valid.size == 0:
                action = solver.adapter(env).block_action(env)
                no_candidate += 1
            else:
                arrays = _batch_to_arrays(batch, cfg)
                if edge_index is None:
                    edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
                seen_candidate_states += 1
                example = {
                    "arrays": arrays,
                    "traffic_scenario": traffic_scenario,
                    "load_name": load_name,
                    "sample_weight": float(preservation_weight),
                }
                if max_examples_int <= 0 or len(examples) < max_examples_int:
                    examples.append(example)
                    episode_collected += 1
                else:
                    replacement = int(rng.integers(0, seen_candidate_states))
                    if replacement < max_examples_int:
                        examples[replacement] = example
                        episode_collected += 1

                tensors = _stack_state_arrays([arrays], device, torch)
                edge_index_tensor = torch.as_tensor(
                    np.asarray(batch.state.edge_index, dtype=np.int64),
                    dtype=torch.long,
                    device=device,
                )
                with torch.no_grad():
                    logits, _value = _a3c_forward(behavior_model, tensors, edge_index_tensor)
                    selected = int(logits.masked_fill(~tensors["candidate_mask"], -1e9).argmax(dim=1).detach().cpu().numpy()[0])
                if int(selected) < 0 or not bool(batch.candidate_mask[int(selected)]):
                    selected = int(valid[0])
                action = batch.topn[int(selected)].action

            _observation, _reward, terminated, truncated, info = env.step(int(action))
            accepted_now = int(bool(info.get("accepted", False)))
            accepted += accepted_now
            episode_accepted += accepted_now
            requests += 1
            episode_requests += 1
            if bool(terminated) or bool(truncated):
                break

        if int(print_every_episodes) > 0 and (
            (episode_position + 1) % int(print_every_episodes) == 0 or episode_position + 1 == len(episode_ids)
        ):
            print(
                json.dumps(
                    _json_safe(
                        {
                            "phase": "preservation_collect",
                            "split": str(split),
                            "episodes_done": int(episode_position + 1),
                            "episodes_total": int(len(episode_ids)),
                            "last_episode_id": str(episode_id),
                            "last_episode_requests": int(episode_requests),
                            "last_episode_accepted": int(episode_accepted),
                            "last_episode_collected_or_replaced": int(episode_collected),
                            "last_episode_weight": float(preservation_weight),
                            "stored_examples": int(len(examples)),
                            "seen_candidate_states": int(seen_candidate_states),
                            "requests": int(requests),
                            "accepted": int(accepted),
                            "elapsed_sec": float(time.perf_counter() - started),
                        }
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )

    if edge_index is None or not examples:
        raise RuntimeError("No preservation candidate states collected")
    scenario_load_summary: dict[str, dict[str, float]] = {}
    for item in examples:
        key = f"{item['traffic_scenario']}:{item['load_name']}"
        row = scenario_load_summary.setdefault(key, {"count": 0.0, "weight_sum": 0.0})
        row["count"] += 1.0
        row["weight_sum"] += float(item["sample_weight"])
    for row in scenario_load_summary.values():
        row["mean_weight"] = float(row["weight_sum"] / max(row["count"], 1.0))
    metrics = {
        "split": str(split),
        "episode_selection": str(episode_selection),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "no_candidate_requests": int(no_candidate),
        "seen_candidate_states": int(seen_candidate_states),
        "stored_examples": int(len(examples)),
        "max_examples": int(max_examples_int),
        "weight_rules": {f"{scenario}:{load}": float(weight) for (scenario, load), weight in preservation_weight_rules.items()},
        "scenario_load_summary": scenario_load_summary,
        "mean_sample_weight": float(np.mean([float(item["sample_weight"]) for item in examples])) if examples else None,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    return examples, edge_index, metrics


def _preservation_batch_tensors(
    examples: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    device: str,
    torch: Any,
) -> tuple[dict[str, Any], Any]:
    selected = [examples[int(index)] for index in indices]
    tensors = _stack_state_arrays([item["arrays"] for item in selected], device, torch)
    sample_weight = torch.as_tensor(
        [float(item["sample_weight"]) for item in selected],
        dtype=torch.float32,
        device=device,
    )
    return tensors, sample_weight


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    initial_checkpoint: dict[str, Any],
    config: ExperimentConfig,
    epoch: int,
    model_info: dict[str, Any],
    history: list[dict[str, Any]],
    eval_metrics: dict[str, Any],
    rollout_val_eval: dict[str, Any] | None,
    args: argparse.Namespace,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_config = dict(initial_checkpoint.get("config") or {})
    checkpoint_config.update(
        {
            "stage": "gnn_cnn_a3c_counterfactual_rank_finetune",
            "input_dir": str(args.input_dir),
            "checkpoint_selection": str(args.checkpoint_selection),
            "rollout_policy": str(args.rollout_policy),
            "freeze_gnn_slot": bool(args.freeze_gnn_slot),
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "gnn_cnn_a3c",
            "n_max": int(initial_checkpoint.get("n_max", _raw_int(config, "n_max", 32))),
            "action_feature_dim": int(initial_checkpoint.get("action_feature_dim", model_info["action_feature_dim"])),
            "hidden_dim": int(initial_checkpoint.get("hidden_dim", model_info["hidden_dim"])),
            "epoch": int(epoch),
            "config": checkpoint_config,
            "solver_config": initial_checkpoint.get("solver_config") or asdict(_solver_config(config, neural=False)),
            "metrics": {
                "eval": eval_metrics,
                "rollout_val_eval": rollout_val_eval,
            },
            "model_info": model_info,
            "history": history,
            "args": vars(args),
            "training_mode": "gnn_cnn_a3c_counterfactual_rank_finetune",
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

    initial_checkpoint_path = _resolve_cli_path(str(args.initial_checkpoint))
    if initial_checkpoint_path is None:
        raise ValueError("--initial-checkpoint is required")
    model, initial_checkpoint = _load_a3c_model(initial_checkpoint_path, device=device, torch=torch)
    model_info = {
        "initial_checkpoint": str(initial_checkpoint_path),
        "hidden_dim": int(initial_checkpoint.get("hidden_dim", _raw_int(config, "hidden_dim", 128))),
        "action_feature_dim": int(initial_checkpoint.get("action_feature_dim", np.asarray(data["action_features"]).shape[-1])),
        "freeze_gnn_slot": bool(args.freeze_gnn_slot),
    }
    if bool(args.freeze_gnn_slot):
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
        for parameter in model.slot_cnn.parameters():
            parameter.requires_grad_(False)

    reference_checkpoint_path = _resolve_cli_path(str(args.reference_checkpoint or ""))
    reference_model = None
    if reference_checkpoint_path is not None and (
        float(args.reference_kl_weight) > 0.0 or float(args.preservation_weight) > 0.0
    ):
        reference_model, _reference_checkpoint = _load_a3c_model(reference_checkpoint_path, device=device, torch=torch)
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        reference_model.eval()
        model_info["reference_checkpoint"] = str(reference_checkpoint_path)
    if float(args.preservation_weight) > 0.0 and reference_model is None:
        raise ValueError("--preservation-weight requires --reference-checkpoint")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable model parameters")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preservation_weight_rules = _parse_weight_rules(str(args.preservation_weight_rules or ""))
    preservation_examples: list[dict[str, Any]] = []
    preservation_edge_index = None
    preservation_metrics: dict[str, Any] | None = None
    if float(args.preservation_weight) > 0.0:
        behavior_model = reference_model if reference_model is not None else model
        preservation_examples, preservation_edge_index_np, preservation_metrics = _collect_preservation_examples(
            config=config,
            output_dir=output_dir,
            split=str(args.preservation_split),
            max_episodes=int(args.preservation_max_episodes),
            max_requests_per_episode=int(args.preservation_max_requests_per_episode),
            episode_selection=str(args.preservation_episode_selection),
            max_examples=int(args.preservation_max_examples),
            preservation_weight_rules=preservation_weight_rules,
            behavior_model=behavior_model,
            device=device,
            torch=torch,
            rng=rng,
            print_every_episodes=int(args.preservation_print_every_episodes),
        )
        preservation_edge_index = torch.as_tensor(
            np.asarray(preservation_edge_index_np, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        model_info["preservation_replay"] = preservation_metrics

    history: list[dict[str, Any]] = []
    best_path = output_dir / "gnn_cnn_a3c_counterfactual_rank_finetune.pt"
    best_score = -math.inf
    best_epoch = -1
    best_rollout_val_eval: dict[str, Any] | None = None

    initial_eval = _evaluate_split_a3c(
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
        initial_rollout_val_eval = _rollout_validate_a3c(
            model=model,
            config=config,
            output_dir=output_dir,
            split=str(args.rollout_val_split),
            max_episodes=int(args.rollout_val_max_episodes),
            max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
            episode_selection=str(args.rollout_val_episode_selection),
            rollout_policy=str(args.rollout_policy),
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
        initial_checkpoint=initial_checkpoint,
        config=config,
        epoch=0,
        model_info=model_info,
        history=history,
        eval_metrics=initial_eval,
        rollout_val_eval=initial_rollout_val_eval,
        args=args,
        torch=torch,
    )
    print(json.dumps(_json_safe({"phase": "initial", "score": initial_score, "eval": initial_eval, "rollout_val_eval": initial_rollout_val_eval}), sort_keys=True), flush=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_rows: list[dict[str, float]] = []
        started = time.perf_counter()
        batches = _iter_batches(splits["train"], int(args.batch_size), shuffle=True, rng=rng)
        for batch_index, batch_indices in enumerate(batches, start=1):
            tensors = _batch_tensors(data, batch_indices, device=device, torch=torch)
            logits, _value = _a3c_forward(model, tensors, edge_index)
            reference_logits = None
            if reference_model is not None:
                with torch.no_grad():
                    reference_logits, _reference_value = _a3c_forward(reference_model, tensors, edge_index)
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
            preservation_kl = logits.sum() * 0.0
            if (
                float(args.preservation_weight) > 0.0
                and preservation_examples
                and preservation_edge_index is not None
                and reference_model is not None
            ):
                preservation_batch_size = int(args.preservation_batch_size or args.batch_size)
                preservation_indices = rng.integers(
                    0,
                    len(preservation_examples),
                    size=max(1, preservation_batch_size),
                    endpoint=False,
                )
                preservation_tensors, preservation_sample_weight = _preservation_batch_tensors(
                    preservation_examples,
                    np.asarray(preservation_indices, dtype=np.int64),
                    device=device,
                    torch=torch,
                )
                preservation_logits, _preservation_value = _a3c_forward(model, preservation_tensors, preservation_edge_index)
                with torch.no_grad():
                    preservation_reference_logits, _preservation_reference_value = _a3c_forward(
                        reference_model,
                        preservation_tensors,
                        preservation_edge_index,
                    )
                preservation_kl = _masked_kl_loss(
                    logits=preservation_logits,
                    reference_logits=preservation_reference_logits,
                    candidate_mask=preservation_tensors["candidate_mask"],
                    sample_weight=preservation_sample_weight,
                    temperature=float(args.preservation_temperature),
                    torch=torch,
                )
                loss = loss + float(args.preservation_weight) * preservation_kl
                parts["loss"] = float(loss.detach().cpu())
                parts["preservation_kl"] = float(preservation_kl.detach().cpu())
            else:
                parts["preservation_kl"] = 0.0
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

        train_eval = _evaluate_split_a3c(
            model=model,
            data=data,
            indices=splits["train"],
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        eval_metrics = _evaluate_split_a3c(
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
            rollout_val_eval = _rollout_validate_a3c(
                model=model,
                config=config,
                output_dir=output_dir,
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection),
                rollout_policy=str(args.rollout_policy),
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
            "train_preservation_kl": float(np.mean([item["preservation_kl"] for item in epoch_rows])) if epoch_rows else None,
            "train_batch_top1_accuracy": float(np.mean([item["top1_accuracy"] for item in epoch_rows])) if epoch_rows else None,
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
                initial_checkpoint=initial_checkpoint,
                config=config,
                epoch=int(epoch),
                model_info=model_info,
                history=history,
                eval_metrics=eval_metrics,
                rollout_val_eval=rollout_val_eval,
                args=args,
                torch=torch,
            )

    summary = {
        "stage": "train_gnn_cnn_a3c_counterfactual_rank_finetune",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(best_path),
        "device": str(device),
        "model_info": model_info,
        "preservation_metrics": preservation_metrics,
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
    _write_json(output_dir / "gnn_cnn_a3c_counterfactual_rank_finetune_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GNN+CNN A3C on counterfactual Top-N rollout labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--ce-weight", type=float, default=0.30)
    parser.add_argument("--listwise-weight", type=float, default=0.80)
    parser.add_argument("--pairwise-weight", type=float, default=0.50)
    parser.add_argument("--base-pairwise-weight", type=float, default=0.50)
    parser.add_argument("--regression-weight", type=float, default=0.05)
    parser.add_argument("--reference-kl-weight", type=float, default=1.00)
    parser.add_argument("--preservation-weight", type=float, default=0.0)
    parser.add_argument("--preservation-temperature", type=float, default=2.0)
    parser.add_argument("--preservation-split", default="train")
    parser.add_argument("--preservation-max-episodes", type=int, default=0)
    parser.add_argument("--preservation-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--preservation-episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--preservation-max-examples", type=int, default=2048)
    parser.add_argument("--preservation-batch-size", type=int, default=0)
    parser.add_argument("--preservation-print-every-episodes", type=int, default=4)
    parser.add_argument("--preservation-weight-rules", default="")
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
    parser.add_argument("--rollout-policy", choices=("full", "override"), default="full")
    parser.add_argument("--rollout-val-split", default="val")
    parser.add_argument("--rollout-val-max-episodes", type=int, default=8)
    parser.add_argument("--rollout-val-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--rollout-val-episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--progress-every-batches", type=int, default=25)
    args = parser.parse_args()
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    summary = train(config, args)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
