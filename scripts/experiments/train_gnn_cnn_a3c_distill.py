#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.ong_rollout import (
    _a3c_override_index,
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _raw_str,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.train_dqn import (
    _batch_to_arrays,
    _build_model as _build_dqn_model,
    _device,
    _iter_batches,
    _model_forward as _dqn_forward,
    _stack_state_arrays,
)
from cse2026.experiments.eon.train_gnn_cnn_a3c_windowed_online import _model_forward as _a3c_forward
from cse2026.ong_solver import GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax

from train_full_dqn_orate60_distill import (
    DistillExample,
    _json_safe,
    _load_full_dqn_checkpoint,
    _parse_weight_rules,
    _resolve_cli_path,
    _sample_weight_for_episode,
    _select_episode_ids,
    _teacher_loss_mask,
    _teacher_safety_guard,
    _teacher_score_vector,
    _valid_teacher_index,
    _write_json,
)


def _load_traffic(
    config: ExperimentConfig,
    split: str,
    max_episodes: int,
    episode_selection: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episodes = traffic.drop_duplicates("episode_id").reset_index(drop=True)
    episode_ids = _select_episode_ids(episodes, max_episodes, episode_selection)
    return traffic, episode_ids


def _load_a3c_model(checkpoint_path: Path, *, device: str, torch: Any) -> tuple[Any, dict[str, Any]]:
    from cse2026.ong_solver.models import GnnCnnA3CNetwork

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("GNN+CNN A3C checkpoint must be a dictionary")
    model = GnnCnnA3CNetwork(
        action_feature_dim=int(checkpoint.get("action_feature_dim", 10)),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _a3c_behavior_index(
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
        logits, _value = _a3c_forward(model, tensors, edge_index)
        masked = logits.masked_fill(~tensors["candidate_mask"], -1e9)
        selected = int(masked.argmax(dim=1).detach().cpu().numpy()[0])
    if int(selected) < 0 or not bool(batch.candidate_mask[int(selected)]):
        return int(valid[0])
    return int(selected)


def _xlron_behavior_index(
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
        logits, _value = model(
            link_features=tensors["link_features"],
            edge_index=edge_index,
            global_features=tensors["global_features"],
            request_features=tensors["request_features"],
            action_features=tensors["action_features"],
            route_link_mask=tensors["route_link_mask"],
            spectrum_tensors=tensors["spectrum_tensors"],
            route_basic_features=tensors["route_basic_features"],
            block_bounds=tensors["block_bounds"],
            candidate_mask=tensors["candidate_mask"],
        )
        masked = logits.masked_fill(~tensors["candidate_mask"], -1e9)
        selected = int(masked.argmax(dim=1).detach().cpu().numpy()[0])
    if int(selected) < 0 or not bool(batch.candidate_mask[int(selected)]):
        return int(valid[0])
    return int(selected)


def _full_dqn_teacher_index_scores(
    *,
    teacher_model: Any,
    batch: Any,
    cfg: Any,
    device: str,
    torch: Any,
) -> tuple[int, np.ndarray, int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    scores = np.full((int(cfg.n_max),), -1e9, dtype=np.float32)
    if valid.size == 0:
        return -1, scores, -1
    arrays = _batch_to_arrays(batch, cfg)
    tensors = _stack_state_arrays([arrays], device, torch)
    edge_index = torch.as_tensor(np.asarray(batch.state.edge_index, dtype=np.int64), dtype=torch.long, device=device)
    with torch.no_grad():
        values = _dqn_forward(teacher_model, tensors, edge_index)
        values = values.masked_fill(~tensors["candidate_mask"], -1e9)
    scores = values.detach().cpu().numpy().reshape(-1).astype(np.float32)
    selected = masked_argmax(scores, batch.candidate_mask)
    return int(selected), scores, int(selected)


def _teacher_score_margin(scores: np.ndarray, candidate_mask: np.ndarray) -> float:
    valid_scores = np.asarray(scores, dtype=np.float32)[np.asarray(candidate_mask, dtype=bool)]
    valid_scores = valid_scores[np.isfinite(valid_scores) & (valid_scores > -1e8)]
    if valid_scores.size < 2:
        return 0.0
    top2 = np.partition(valid_scores, -2)[-2:]
    top2.sort()
    return float(top2[-1] - top2[-2])


def _example_score_diagnostics(examples: list[DistillExample]) -> dict[str, Any]:
    margins = np.asarray([float(item.teacher_margin) for item in examples], dtype=np.float64)
    disagreements = np.asarray(
        [int(item.teacher_index) != int(item.behavior_index) for item in examples],
        dtype=bool,
    )
    if margins.size == 0:
        return {
            "teacher_margin_mean": None,
            "teacher_margin_p50": None,
            "teacher_margin_p90": None,
            "teacher_margin_p99": None,
            "teacher_margin_gt_1e-4_rate": None,
            "teacher_margin_gt_1e-3_rate": None,
            "teacher_margin_gt_1e-2_rate": None,
            "disagreement_count": 0,
            "disagreement_teacher_margin_mean": None,
        }
    disagreement_margins = margins[disagreements]
    return {
        "teacher_margin_mean": float(np.mean(margins)),
        "teacher_margin_p50": float(np.percentile(margins, 50)),
        "teacher_margin_p90": float(np.percentile(margins, 90)),
        "teacher_margin_p99": float(np.percentile(margins, 99)),
        "teacher_margin_gt_1e-4_rate": float(np.mean(margins > 1e-4)),
        "teacher_margin_gt_1e-3_rate": float(np.mean(margins > 1e-3)),
        "teacher_margin_gt_1e-2_rate": float(np.mean(margins > 1e-2)),
        "disagreement_count": int(np.sum(disagreements)),
        "disagreement_teacher_margin_mean": (
            float(np.mean(disagreement_margins)) if disagreement_margins.size > 0 else None
        ),
    }


def _collect_examples(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    teacher_kind: str,
    tree_ranker: Any | None,
    dqn_teacher_model: Any | None,
    teacher_base_policy: str,
    behavior_policy: str,
    behavior_a3c_model: Any | None,
    behavior_xlron_model: Any | None,
    episode_selection: str,
    hard_case_weight_rules: dict[tuple[str, str], float],
    teacher_score_selected_boost: float,
    print_every_episodes: int,
    device: str,
    torch: Any,
) -> tuple[list[DistillExample], np.ndarray, dict[str, Any]]:
    traffic, episode_ids = _load_traffic(config, split, max_episodes, episode_selection)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    cfg = solver.config
    run_path = output_dir / f"collect_{split}"
    normalized_teacher = str(teacher_kind).strip().lower().replace("-", "_")
    normalized_behavior = str(behavior_policy).strip().lower().replace("-", "_")
    if normalized_teacher == "tree" and tree_ranker is None:
        raise ValueError("tree teacher requires tree_ranker")
    if normalized_teacher == "full_dqn" and dqn_teacher_model is None:
        raise ValueError("full_dqn teacher requires dqn_teacher_model")
    if normalized_behavior == "student_a3c" and behavior_a3c_model is None:
        raise ValueError("student_a3c behavior requires behavior_a3c_model")
    if normalized_behavior == "student_xlron" and behavior_xlron_model is None:
        raise ValueError("student_xlron behavior requires behavior_xlron_model")
    if normalized_behavior not in {"teacher", "student_a3c", "student_xlron"}:
        raise ValueError(f"Unsupported behavior_policy: {behavior_policy}")

    examples: list[DistillExample] = []
    edge_index: np.ndarray | None = None
    requests = 0
    accepted = 0
    no_candidate = 0
    fallback_teacher = 0
    behavior_teacher_agree = 0
    behavior_teacher_total = 0
    behavior_indices: list[int] = []
    started = time.perf_counter()

    for episode_position, episode_id in enumerate(episode_ids):
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.iloc[:max_requests_per_episode].reset_index(drop=True)
        traffic_scenario = str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else ""
        load_name = str(episode["load_name"].iloc[0]) if "load_name" in episode else ""
        sample_weight = _sample_weight_for_episode(episode, hard_case_weight_rules)
        traffic_jsonl = _traffic_jsonl_for_episode(run_path, episode_id, episode)
        env = _make_env(
            episode_id=episode_id,
            traffic_path=traffic_jsonl,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))

        episode_requests = 0
        episode_examples = 0
        episode_accepted = 0
        while True:
            batch = solver.candidate_batch(env)
            valid = np.flatnonzero(batch.candidate_mask.astype(bool))
            if valid.size == 0:
                action = solver.adapter(env).block_action(env)
                no_candidate += 1
            else:
                if normalized_teacher == "tree":
                    assert tree_ranker is not None
                    teacher_index, teacher_margin, did_fallback = _valid_teacher_index(
                        batch=batch,
                        ranker=tree_ranker,
                        n_max=cfg.n_max,
                        fallback_policy=teacher_base_policy,
                    )
                    teacher_scores, teacher_ranker_argmax = _teacher_score_vector(
                        batch=batch,
                        ranker=tree_ranker,
                        n_max=cfg.n_max,
                        selected_index=int(teacher_index),
                        selected_boost=float(teacher_score_selected_boost),
                    )
                    fallback_teacher += int(bool(did_fallback))
                else:
                    assert dqn_teacher_model is not None
                    teacher_index, teacher_scores, teacher_ranker_argmax = _full_dqn_teacher_index_scores(
                        teacher_model=dqn_teacher_model,
                        batch=batch,
                        cfg=cfg,
                        device=device,
                        torch=torch,
                    )
                    teacher_margin = _teacher_score_margin(teacher_scores, batch.candidate_mask)

                arrays = _batch_to_arrays(batch, cfg)
                if edge_index is None:
                    edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)

                if normalized_behavior == "teacher":
                    behavior_index = int(teacher_index)
                elif normalized_behavior == "student_a3c":
                    assert behavior_a3c_model is not None
                    behavior_index = _a3c_behavior_index(
                        model=behavior_a3c_model,
                        batch=batch,
                        cfg=cfg,
                        device=device,
                        torch=torch,
                    )
                else:
                    assert behavior_xlron_model is not None
                    behavior_index = _xlron_behavior_index(
                        model=behavior_xlron_model,
                        batch=batch,
                        cfg=cfg,
                        device=device,
                        torch=torch,
                    )
                if int(behavior_index) < 0 or not bool(batch.candidate_mask[int(behavior_index)]):
                    behavior_index = int(valid[0])

                examples.append(
                    DistillExample(
                        arrays=arrays,
                        teacher_index=int(teacher_index),
                        teacher_scores=np.asarray(teacher_scores, dtype=np.float32),
                        teacher_margin=float(teacher_margin),
                        behavior_index=int(behavior_index),
                        valid_count=int(valid.size),
                        sample_weight=float(sample_weight),
                        episode_id=str(episode_id),
                        request_id=int(episode_requests),
                        traffic_scenario=traffic_scenario,
                        load_name=load_name,
                        teacher_ranker_argmax=int(teacher_ranker_argmax),
                    )
                )
                episode_examples += 1
                behavior_indices.append(int(behavior_index))
                behavior_teacher_agree += int(int(behavior_index) == int(teacher_index))
                behavior_teacher_total += 1
                action = batch.topn[int(behavior_index)].action

            _observation, _reward, terminated, truncated, info = env.step(int(action))
            accepted_now = int(bool(info.get("accepted", False)))
            accepted += accepted_now
            episode_accepted += accepted_now
            requests += 1
            episode_requests += 1
            if bool(terminated) or bool(truncated):
                break

        if print_every_episodes > 0 and (
            (episode_position + 1) % print_every_episodes == 0 or episode_position + 1 == len(episode_ids)
        ):
            print(
                json.dumps(
                    {
                        "phase": "collect",
                        "split": split,
                        "episodes_done": int(episode_position + 1),
                        "episodes_total": int(len(episode_ids)),
                        "last_episode_id": str(episode_id),
                        "last_episode_requests": int(episode_requests),
                        "last_episode_examples": int(episode_examples),
                        "last_episode_accepted": int(episode_accepted),
                        "last_episode_weight": float(sample_weight),
                        "examples": int(len(examples)),
                        "requests": int(requests),
                        "accepted": int(accepted),
                        "behavior_policy": str(normalized_behavior),
                        "behavior_teacher_agreement_rate": float(behavior_teacher_agree / max(behavior_teacher_total, 1)),
                        "elapsed_sec": float(time.perf_counter() - started),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if edge_index is None:
        raise RuntimeError(f"No non-empty candidate states collected for split={split}")
    metrics = {
        "split": split,
        "teacher_kind": str(normalized_teacher),
        "episode_selection": str(episode_selection),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "examples": int(len(examples)),
        "no_candidate_requests": int(no_candidate),
        "fallback_teacher": int(fallback_teacher),
        "behavior_policy": str(normalized_behavior),
        "behavior_rollout_blocking_rate": float((requests - accepted) / max(requests, 1)),
        "behavior_teacher_agreement_rate": float(behavior_teacher_agree / max(behavior_teacher_total, 1)),
        "behavior_teacher_disagreement_rate": float(1.0 - behavior_teacher_agree / max(behavior_teacher_total, 1)),
        "mean_valid_candidates": float(np.mean([item.valid_count for item in examples])) if examples else None,
        "mean_teacher_index": float(np.mean([item.teacher_index for item in examples])) if examples else None,
        "mean_behavior_index": float(np.mean(behavior_indices)) if behavior_indices else None,
        "mean_teacher_margin": float(np.mean([item.teacher_margin for item in examples])) if examples else None,
        "mean_sample_weight": float(np.mean([item.sample_weight for item in examples])) if examples else None,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    metrics.update(_example_score_diagnostics(examples))
    return examples, edge_index, metrics


def _batch_examples(
    examples: list[DistillExample],
    indices: np.ndarray,
    *,
    device: str,
    torch: Any,
) -> tuple[dict[str, Any], Any, Any, Any, Any, Any]:
    selected = [examples[int(index)] for index in indices]
    tensors = _stack_state_arrays([item.arrays for item in selected], device, torch)
    teacher_index = torch.as_tensor([item.teacher_index for item in selected], dtype=torch.long, device=device)
    teacher_scores = torch.as_tensor(np.stack([item.teacher_scores for item in selected], axis=0), dtype=torch.float32, device=device)
    sample_weight = torch.as_tensor([item.sample_weight for item in selected], dtype=torch.float32, device=device)
    behavior_index = torch.as_tensor([item.behavior_index for item in selected], dtype=torch.long, device=device)
    teacher_margin = torch.as_tensor([item.teacher_margin for item in selected], dtype=torch.float32, device=device)
    return tensors, teacher_index, teacher_scores, sample_weight, behavior_index, teacher_margin


def _distillation_loss(
    *,
    logits: Any,
    candidate_mask: Any,
    teacher_index: Any,
    teacher_scores: Any,
    sample_weight: Any,
    behavior_index: Any,
    teacher_margin: Any,
    ce_weight: float,
    listwise_kl_weight: float,
    temperature: float,
    pairwise_rank_weight: float,
    pairwise_temperature: float,
    pairwise_teacher_gap: float,
    teacher_loss_filter: str,
    min_teacher_margin: float,
    entropy_weight: float,
    torch: Any,
) -> tuple[Any, dict[str, float]]:
    ce_per = torch.nn.functional.cross_entropy(logits, teacher_index, reduction="none")
    teacher_mask = _teacher_loss_mask(
        teacher_index=teacher_index,
        behavior_index=behavior_index,
        teacher_margin=teacher_margin,
        mode=teacher_loss_filter,
        min_teacher_margin=float(min_teacher_margin),
        torch=torch,
    )
    temp = max(float(temperature), 1e-6)
    if float(listwise_kl_weight) > 0.0:
        teacher_logits = teacher_scores.masked_fill(~candidate_mask, -1e9) / temp
        student_log_probs = torch.nn.functional.log_softmax(logits / temp, dim=1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=1)
        kl_per = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1) * (temp * temp)
    else:
        kl_per = torch.zeros_like(ce_per)

    if float(pairwise_rank_weight) > 0.0:
        pair_temp = max(float(pairwise_temperature), 1e-6)
        valid_pair = candidate_mask.unsqueeze(2) & candidate_mask.unsqueeze(1)
        teacher_delta = teacher_scores.unsqueeze(2) - teacher_scores.unsqueeze(1)
        ordered_pair = valid_pair & (teacher_delta > float(pairwise_teacher_gap))
        student_delta = (logits.unsqueeze(2) - logits.unsqueeze(1)) / pair_temp
        pair_loss = torch.nn.functional.softplus(-student_delta).masked_fill(~ordered_pair, 0.0)
        pair_count = ordered_pair.to(dtype=logits.dtype).sum(dim=(1, 2))
        pairwise_per = pair_loss.sum(dim=(1, 2)) / torch.clamp(pair_count, min=1.0)
        pairwise_per = torch.where(pair_count > 0, pairwise_per, torch.zeros_like(pairwise_per))
    else:
        pairwise_per = torch.zeros_like(ce_per)

    per_example = (
        float(ce_weight) * ce_per
        + float(listwise_kl_weight) * kl_per
        + float(pairwise_rank_weight) * pairwise_per
    )
    weights = sample_weight * teacher_mask.to(dtype=sample_weight.dtype)
    if bool(teacher_mask.any()):
        supervised_loss = (per_example * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
    else:
        supervised_loss = per_example.sum() * 0.0

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    probs = torch.nn.functional.softmax(logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    loss = supervised_loss - float(entropy_weight) * entropy
    return loss, {
        "ce_loss": float(ce_per.detach().mean().cpu()),
        "kl_loss": float(kl_per.detach().mean().cpu()),
        "pairwise_rank_loss": float(pairwise_per.detach().mean().cpu()),
        "supervised_loss": float(supervised_loss.detach().cpu()),
        "teacher_loss_examples": int(teacher_mask.detach().sum().cpu()),
        "teacher_loss_fraction": float(teacher_mask.detach().float().mean().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "weighted_loss": float(loss.detach().cpu()),
    }


def _evaluate_examples(
    *,
    model: Any,
    examples: list[DistillExample],
    edge_index: Any,
    batch_size: int,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    losses: list[float] = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch_indices in _iter_batches(len(examples), batch_size, shuffle=False, rng=rng):
            tensors, teacher_index, _teacher_scores, _sample_weight, _behavior_index, _teacher_margin = _batch_examples(
                examples,
                batch_indices,
                device=device,
                torch=torch,
            )
            raw_logits, _values = _a3c_forward(model, tensors, edge_index)
            logits = raw_logits.masked_fill(~tensors["candidate_mask"], -1e9)
            losses.append(float(torch.nn.functional.cross_entropy(logits, teacher_index).detach().cpu()))
            prediction = logits.argmax(dim=1)
            correct += int((prediction == teacher_index).sum().detach().cpu())
            total += int(len(batch_indices))
    return {
        "examples": int(total),
        "loss": float(np.mean(losses)) if losses else None,
        "teacher_top1_accuracy": float(correct / max(total, 1)),
        "disagreement_rate": float(1.0 - correct / max(total, 1)),
    }


def _rollout_validate_a3c(
    *,
    model: Any,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    episode_selection: str,
    rollout_policy: str,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    traffic, episode_ids = _load_traffic(config, split, max_episodes, episode_selection)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    cfg = solver.config
    run_path = output_dir / f"rollout_validate_{split}_{rollout_policy}"
    model.eval()
    requests = 0
    accepted = 0
    no_candidate = 0
    override_requests = 0
    override_applied = 0
    total_reward = 0.0
    selected_indices: list[int] = []
    started = time.perf_counter()
    normalized_policy = str(rollout_policy).strip().lower().replace("-", "_")

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.iloc[:max_requests_per_episode].reset_index(drop=True)
        traffic_jsonl = _traffic_jsonl_for_episode(run_path, episode_id, episode)
        env = _make_env(
            episode_id=episode_id,
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
                arrays = _batch_to_arrays(batch, cfg)
                tensors = _stack_state_arrays([arrays], device, torch)
                edge_index = torch.as_tensor(np.asarray(batch.state.edge_index, dtype=np.int64), dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, _value = _a3c_forward(model, tensors, edge_index)
                    scores = logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
                if normalized_policy == "override":
                    selected_index, did_override, _margin = _a3c_override_index(
                        batch=batch,
                        logits=scores,
                        config=config,
                        n_max=cfg.n_max,
                    )
                    override_requests += 1
                    override_applied += int(bool(did_override))
                else:
                    selected_index = masked_argmax(scores, batch.candidate_mask)
                if int(selected_index) < 0 or not bool(batch.candidate_mask[int(selected_index)]):
                    selected_index = int(valid[0])
                selected_indices.append(int(selected_index))
                action = batch.topn[int(selected_index)].action

            _observation, reward, terminated, truncated, info = env.step(int(action))
            accepted += int(bool(info.get("accepted", False)))
            total_reward += float(reward)
            requests += 1
            if bool(terminated) or bool(truncated):
                break
    return {
        "split": str(split),
        "rollout_policy": str(normalized_policy),
        "episode_selection": str(episode_selection),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": float(total_reward),
        "no_candidate_requests": int(no_candidate),
        "override_requests": int(override_requests),
        "override_applied": int(override_applied),
        "override_rate": float(override_applied / max(override_requests, 1)),
        "mean_selected_index": float(np.mean(selected_indices)) if selected_indices else None,
        "elapsed_sec": float(time.perf_counter() - started),
    }


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    initial_checkpoint: dict[str, Any],
    config: ExperimentConfig,
    step: int,
    epoch: int,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_config = dict(initial_checkpoint.get("config") or {})
    checkpoint_config.update(
        {
            "teacher_kind": str(args.teacher_kind),
            "teacher_artifact": str(args.teacher_artifact),
            "teacher_dqn_checkpoint": str(args.teacher_dqn_checkpoint),
            "behavior_policy": str(args.behavior_policy),
            "checkpoint_selection": str(args.checkpoint_selection),
            "rollout_policy": str(args.rollout_policy),
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "gnn_cnn_a3c",
            "n_max": int(initial_checkpoint.get("n_max", _raw_int(config, "n_max", 32))),
            "action_feature_dim": int(initial_checkpoint.get("action_feature_dim", 10)),
            "hidden_dim": int(initial_checkpoint.get("hidden_dim", _raw_int(config, "hidden_dim", 128))),
            "step": int(step),
            "epoch": int(epoch),
            "config": checkpoint_config,
            "solver_config": initial_checkpoint.get("solver_config") or asdict(_solver_config(config, neural=False)),
            "metrics": metrics,
            "training_mode": "gnn_cnn_a3c_supervised_distill",
        },
        path,
    )


def _train(
    *,
    config: ExperimentConfig,
    train_examples: list[DistillExample],
    val_examples: list[DistillExample],
    edge_index_np: np.ndarray,
    output_dir: Path,
    initial_checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    torch.manual_seed(int(config.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))
    model, initial_checkpoint = _load_a3c_model(initial_checkpoint_path, device=device, torch=torch)
    if bool(args.freeze_encoders):
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
        for parameter in model.slot_cnn.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    rng = np.random.default_rng(int(config.seed))
    best_metric = -1.0
    best_epoch = 0
    best_rollout_val_eval: dict[str, Any] | None = None
    initial_rollout_val_eval: dict[str, Any] | None = None
    best_path = output_dir / "gnn_cnn_a3c_distill_best.pt"
    history: list[dict[str, Any]] = []
    global_step = 0

    if str(args.checkpoint_selection) != "teacher_top1":
        initial_rollout_val_eval = _rollout_validate_a3c(
            model=model,
            config=config,
            output_dir=output_dir,
            split=str(args.rollout_val_split),
            max_episodes=int(args.rollout_val_max_episodes),
            max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
            episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
            rollout_policy=str(args.rollout_policy),
            device=device,
            torch=torch,
        )
        metric_key = "accepted" if str(args.checkpoint_selection) == "rollout_accepted" else "total_reward"
        best_metric = float(initial_rollout_val_eval.get(metric_key) or 0.0)
        best_rollout_val_eval = initial_rollout_val_eval
        _save_checkpoint(
            path=best_path,
            model=model,
            initial_checkpoint=initial_checkpoint,
            config=config,
            step=0,
            epoch=0,
            metrics={"initial_rollout_val_eval": initial_rollout_val_eval},
            args=args,
            torch=torch,
        )
        print(
            json.dumps(
                {
                    "phase": "initial_checkpoint_candidate",
                    "checkpoint_selection": str(args.checkpoint_selection),
                    "selection_metric": float(best_metric),
                    "rollout_val_eval": initial_rollout_val_eval,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        batches = _iter_batches(len(train_examples), int(args.batch_size), shuffle=True, rng=rng)
        epoch_losses: list[float] = []
        epoch_ce_losses: list[float] = []
        epoch_kl_losses: list[float] = []
        epoch_pairwise_losses: list[float] = []
        epoch_correct = 0
        epoch_total = 0
        started = time.perf_counter()
        for batch_number, batch_indices in enumerate(batches, start=1):
            tensors, teacher_index, teacher_scores, sample_weight, behavior_index, teacher_margin = _batch_examples(
                train_examples,
                batch_indices,
                device=device,
                torch=torch,
            )
            raw_logits, _values = _a3c_forward(model, tensors, edge_index)
            logits = raw_logits.masked_fill(~tensors["candidate_mask"], -1e9)
            loss, loss_parts = _distillation_loss(
                logits=logits,
                candidate_mask=tensors["candidate_mask"],
                teacher_index=teacher_index,
                teacher_scores=teacher_scores,
                sample_weight=sample_weight,
                behavior_index=behavior_index,
                teacher_margin=teacher_margin,
                ce_weight=float(args.ce_weight),
                listwise_kl_weight=float(args.listwise_kl_weight),
                temperature=float(args.listwise_temperature),
                pairwise_rank_weight=float(args.pairwise_rank_weight),
                pairwise_temperature=float(args.pairwise_temperature),
                pairwise_teacher_gap=float(args.pairwise_teacher_gap),
                teacher_loss_filter=str(args.teacher_loss_filter),
                min_teacher_margin=float(args.min_teacher_margin),
                entropy_weight=float(args.entropy_weight),
                torch=torch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(args.grad_clip_norm))
            optimizer.step()

            prediction = logits.detach().argmax(dim=1)
            epoch_correct += int((prediction == teacher_index).sum().detach().cpu())
            epoch_total += int(len(batch_indices))
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_ce_losses.append(float(loss_parts["ce_loss"]))
            epoch_kl_losses.append(float(loss_parts["kl_loss"]))
            epoch_pairwise_losses.append(float(loss_parts["pairwise_rank_loss"]))
            global_step += int(len(batch_indices))
            if int(args.progress_every_batches) > 0 and batch_number % int(args.progress_every_batches) == 0:
                print(
                    json.dumps(
                        {
                            "phase": "train_batch",
                            "epoch": int(epoch),
                            "batch": int(batch_number),
                            "batches": int(len(batches)),
                            "train_loss_mean": float(np.mean(epoch_losses)),
                            "train_ce_loss_mean": float(np.mean(epoch_ce_losses)),
                            "train_kl_loss_mean": float(np.mean(epoch_kl_losses)),
                            "train_pairwise_rank_loss_mean": float(np.mean(epoch_pairwise_losses)),
                            "train_teacher_top1_accuracy": float(epoch_correct / max(epoch_total, 1)),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        train_eval = _evaluate_examples(
            model=model,
            examples=train_examples,
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        val_eval = _evaluate_examples(
            model=model,
            examples=val_examples,
            edge_index=edge_index,
            batch_size=int(args.batch_size),
            device=device,
            torch=torch,
        )
        rollout_val_eval = None
        if str(args.checkpoint_selection) != "teacher_top1":
            rollout_val_eval = _rollout_validate_a3c(
                model=model,
                config=config,
                output_dir=output_dir,
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
                rollout_policy=str(args.rollout_policy),
                device=device,
                torch=torch,
            )
        row = {
            "phase": "epoch",
            "epoch": int(epoch),
            "train_loss_online": float(np.mean(epoch_losses)) if epoch_losses else None,
            "train_ce_loss_online": float(np.mean(epoch_ce_losses)) if epoch_ce_losses else None,
            "train_kl_loss_online": float(np.mean(epoch_kl_losses)) if epoch_kl_losses else None,
            "train_pairwise_rank_loss_online": float(np.mean(epoch_pairwise_losses)) if epoch_pairwise_losses else None,
            "train_teacher_top1_accuracy_online": float(epoch_correct / max(epoch_total, 1)),
            "train_eval": train_eval,
            "val_eval": val_eval,
            "rollout_val_eval": rollout_val_eval,
            "elapsed_sec": float(time.perf_counter() - started),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if str(args.checkpoint_selection) == "teacher_top1":
            metric = float(val_eval.get("teacher_top1_accuracy") or 0.0)
        elif str(args.checkpoint_selection) == "rollout_accepted":
            metric = float((rollout_val_eval or {}).get("accepted") or 0.0)
        elif str(args.checkpoint_selection) == "rollout_reward":
            metric = float((rollout_val_eval or {}).get("total_reward") or 0.0)
        else:
            raise ValueError(f"Unsupported checkpoint_selection: {args.checkpoint_selection}")
        if metric >= best_metric:
            best_metric = metric
            best_epoch = int(epoch)
            best_rollout_val_eval = rollout_val_eval
            _save_checkpoint(
                path=best_path,
                model=model,
                initial_checkpoint=initial_checkpoint,
                config=config,
                step=global_step,
                epoch=epoch,
                metrics={"val_eval": val_eval, "rollout_val_eval": rollout_val_eval},
                args=args,
                torch=torch,
            )

    summary = {
        "stage": "train_gnn_cnn_a3c_distill",
        "checkpoint_path": str(best_path),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "best_rollout_val_eval": best_rollout_val_eval,
        "initial_rollout_val_eval": initial_rollout_val_eval,
        "checkpoint_selection": str(args.checkpoint_selection),
        "rollout_policy": str(args.rollout_policy),
        "device": device,
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "freeze_encoders": bool(args.freeze_encoders),
        "ce_weight": float(args.ce_weight),
        "listwise_kl_weight": float(args.listwise_kl_weight),
        "listwise_temperature": float(args.listwise_temperature),
        "pairwise_rank_weight": float(args.pairwise_rank_weight),
        "pairwise_temperature": float(args.pairwise_temperature),
        "pairwise_teacher_gap": float(args.pairwise_teacher_gap),
        "teacher_loss_filter": str(args.teacher_loss_filter),
        "min_teacher_margin": float(args.min_teacher_margin),
        "teacher_score_selected_boost": float(args.teacher_score_selected_boost),
        "hard_case_weight_rules": str(args.hard_case_weight_rules),
        "initial_a3c_checkpoint": str(initial_checkpoint_path),
        "teacher_kind": str(args.teacher_kind),
        "teacher_artifact": str(args.teacher_artifact),
        "teacher_dqn_checkpoint": str(args.teacher_dqn_checkpoint),
        "history": history,
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill teachers into the full GNN+CNN A3C actor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-a3c-checkpoint", required=True)
    parser.add_argument("--teacher-kind", choices=("tree", "full_dqn"), required=True)
    parser.add_argument("--teacher-artifact", default="")
    parser.add_argument("--teacher-dqn-checkpoint", default="")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-max-episodes", type=int, default=16)
    parser.add_argument("--val-max-episodes", type=int, default=4)
    parser.add_argument("--episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--max-requests-per-episode", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--freeze-encoders", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--listwise-kl-weight", type=float, default=0.5)
    parser.add_argument("--listwise-temperature", type=float, default=2.0)
    parser.add_argument("--pairwise-rank-weight", type=float, default=0.0)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-teacher-gap", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-loss-filter",
        choices=("all", "disagreement", "high_confidence_disagreement"),
        default="all",
    )
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--teacher-score-selected-boost", type=float, default=0.0)
    parser.add_argument("--hard-case-weight-rules", default="")
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--print-every-episodes", type=int, default=1)
    parser.add_argument("--behavior-policy", choices=("teacher", "student_a3c"), default="student_a3c")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("teacher_top1", "rollout_accepted", "rollout_reward"),
        default="rollout_accepted",
    )
    parser.add_argument("--rollout-policy", choices=("full", "override"), default="full")
    parser.add_argument("--rollout-val-split", default="val")
    parser.add_argument("--rollout-val-max-episodes", type=int, default=4)
    parser.add_argument("--rollout-val-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--rollout-val-episode-selection", default="")
    parser.add_argument("--teacher-selection-mode", default="positive_advantage")
    parser.add_argument("--teacher-base-policy", default="energy-aware-ksp-bm-ff")
    parser.add_argument("--teacher-safety-guard", choices=("emergency", "none"), default="emergency")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_a3c_checkpoint = _resolve_cli_path(str(args.initial_a3c_checkpoint))
    if initial_a3c_checkpoint is None:
        raise ValueError("--initial-a3c-checkpoint is required")

    from cse2026.ong_solver.models import require_torch
    from cse2026.experiments.eon.tree_ranker_runtime import TreeCandidateRanker

    torch = require_torch()
    device = _device(config, torch)
    ong_source = _add_ong_source_path(config)
    hard_case_weight_rules = _parse_weight_rules(str(args.hard_case_weight_rules))
    tree_ranker = None
    dqn_teacher_model = None
    teacher_artifact = _resolve_cli_path(str(args.teacher_artifact or ""))
    teacher_dqn_checkpoint = _resolve_cli_path(str(args.teacher_dqn_checkpoint or ""))
    if str(args.teacher_kind) == "tree":
        if teacher_artifact is None:
            raise ValueError("tree teacher requires --teacher-artifact")
        tree_ranker = TreeCandidateRanker.load(
            teacher_artifact,
            selection_mode=str(args.teacher_selection_mode),
            base_policy=str(args.teacher_base_policy),
            safety_guard=_teacher_safety_guard(args),
        )
    else:
        if teacher_dqn_checkpoint is None:
            raise ValueError("full_dqn teacher requires --teacher-dqn-checkpoint")
        dqn_teacher_model, _pretrained = _build_dqn_model(config, device, torch)
        _load_full_dqn_checkpoint(dqn_teacher_model, teacher_dqn_checkpoint, device=device, torch=torch)
        dqn_teacher_model.eval()
        for parameter in dqn_teacher_model.parameters():
            parameter.requires_grad_(False)

    behavior_a3c_model = None
    if str(args.behavior_policy) == "student_a3c":
        behavior_a3c_model, _initial = _load_a3c_model(initial_a3c_checkpoint, device=device, torch=torch)
        behavior_a3c_model.eval()
        for parameter in behavior_a3c_model.parameters():
            parameter.requires_grad_(False)

    print(
        json.dumps(
            {
                "phase": "start",
                "dataset_path": str(config.dataset_path),
                "output_dir": str(output_dir),
                "ong_source_path": ong_source,
                "initial_a3c_checkpoint": str(initial_a3c_checkpoint),
                "teacher_kind": str(args.teacher_kind),
                "teacher_artifact": None if teacher_artifact is None else str(teacher_artifact),
                "teacher_dqn_checkpoint": None if teacher_dqn_checkpoint is None else str(teacher_dqn_checkpoint),
                "behavior_policy": str(args.behavior_policy),
                "checkpoint_selection": str(args.checkpoint_selection),
                "rollout_policy": str(args.rollout_policy),
                "train_max_episodes": int(args.train_max_episodes),
                "val_max_episodes": int(args.val_max_episodes),
                "episode_selection": str(args.episode_selection),
                "hard_case_weight_rules": {
                    f"{scenario}:{load}": weight for (scenario, load), weight in hard_case_weight_rules.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    train_examples, edge_index, train_collect = _collect_examples(
        config=config,
        output_dir=output_dir,
        split=str(args.train_split),
        max_episodes=int(args.train_max_episodes),
        max_requests_per_episode=int(args.max_requests_per_episode),
        teacher_kind=str(args.teacher_kind),
        tree_ranker=tree_ranker,
        dqn_teacher_model=dqn_teacher_model,
        teacher_base_policy=str(args.teacher_base_policy),
        behavior_policy=str(args.behavior_policy),
        behavior_a3c_model=behavior_a3c_model,
        behavior_xlron_model=None,
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=float(args.teacher_score_selected_boost),
        print_every_episodes=int(args.print_every_episodes),
        device=device,
        torch=torch,
    )
    val_examples, _val_edge_index, val_collect = _collect_examples(
        config=config,
        output_dir=output_dir,
        split=str(args.val_split),
        max_episodes=int(args.val_max_episodes),
        max_requests_per_episode=int(args.max_requests_per_episode),
        teacher_kind=str(args.teacher_kind),
        tree_ranker=tree_ranker,
        dqn_teacher_model=dqn_teacher_model,
        teacher_base_policy=str(args.teacher_base_policy),
        behavior_policy=str(args.behavior_policy),
        behavior_a3c_model=behavior_a3c_model,
        behavior_xlron_model=None,
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=float(args.teacher_score_selected_boost),
        print_every_episodes=int(args.print_every_episodes),
        device=device,
        torch=torch,
    )
    collect_summary = {"train_collect": train_collect, "val_collect": val_collect}
    _write_json(output_dir / "collection_summary.json", collect_summary)
    print(json.dumps({"phase": "collection_summary", **collect_summary}, sort_keys=True), flush=True)

    summary = _train(
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        edge_index_np=edge_index,
        output_dir=output_dir,
        initial_checkpoint_path=initial_a3c_checkpoint,
        args=args,
    )
    print(json.dumps({"phase": "done", **_json_safe(summary)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
