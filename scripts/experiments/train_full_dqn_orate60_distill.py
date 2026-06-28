#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_int,
    _solver_config,
    _traffic_jsonl_for_episode,
)
from cse2026.experiments.eon.lookahead_override_features import candidate_feature_matrix
from cse2026.experiments.eon.train_dqn import (
    _batch_to_arrays,
    _build_model,
    _device,
    _iter_batches,
    _model_forward,
    _raw_bool,
    _raw_float,
    _raw_str,
    _stack_state_arrays,
)
from cse2026.experiments.eon.tree_ranker_runtime import (
    TreeCandidateRanker,
    _append_runtime_features,
    select_tree_base_index,
)
from cse2026.ong_solver import GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax


@dataclass(frozen=True)
class DistillExample:
    arrays: dict[str, np.ndarray]
    teacher_index: int
    teacher_scores: np.ndarray
    teacher_margin: float
    behavior_index: int
    valid_count: int
    sample_weight: float
    episode_id: str
    request_id: int
    traffic_scenario: str
    load_name: str
    teacher_ranker_argmax: int


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


def _teacher_safety_guard(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.teacher_safety_guard == "none":
        return None
    if args.teacher_safety_guard != "emergency":
        raise ValueError(f"Unsupported teacher safety guard: {args.teacher_safety_guard}")
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


def _valid_teacher_index(
    *,
    batch: Any,
    ranker: TreeCandidateRanker,
    n_max: int,
    fallback_policy: str,
) -> tuple[int, float, bool]:
    selected_index, margin = ranker.select_index(batch, n_max)
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if int(selected_index) in {int(index) for index in valid}:
        return int(selected_index), float(margin), False

    base_index = select_tree_base_index(batch, n_max, fallback_policy)
    if int(base_index) in {int(index) for index in valid}:
        return int(base_index), 0.0, True
    return int(valid[0]), 0.0, True


def _teacher_score_vector(
    *,
    batch: Any,
    ranker: TreeCandidateRanker,
    n_max: int,
    selected_index: int,
    selected_boost: float,
) -> tuple[np.ndarray, int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    scores_vector = np.full((int(n_max),), -1e9, dtype=np.float32)
    if valid.size == 0:
        return scores_vector, -1

    base_index = select_tree_base_index(batch, n_max, ranker.base_policy)
    if int(base_index) < 0:
        base_index = int(valid[0])
    candidate_indices = ranker._candidate_indices(batch, int(base_index))
    features, kept_indices = candidate_feature_matrix(
        batch=batch,
        candidate_indices=candidate_indices,
        n_max=n_max,
        reference_index=int(base_index),
    )
    ranker_argmax = -1
    if kept_indices:
        ranker_features = _append_runtime_features(
            features=features,
            kept_indices=kept_indices,
            base_index=int(base_index),
            feature_names=ranker.feature_names,
        )
        scores = np.asarray(ranker.scores(ranker_features), dtype=np.float32)
        if scores.size == len(kept_indices):
            for candidate_index, score in zip(kept_indices, scores):
                if 0 <= int(candidate_index) < int(n_max) and np.isfinite(float(score)):
                    scores_vector[int(candidate_index)] = float(score)
            ranker_argmax = int(kept_indices[int(np.argmax(scores))])

    finite = np.isfinite(scores_vector) & (scores_vector > -1e8)
    if not bool(finite.any()):
        scores_vector[valid] = 0.0
    else:
        min_score = float(np.min(scores_vector[finite]))
        for index in valid:
            if not (np.isfinite(scores_vector[int(index)]) and scores_vector[int(index)] > -1e8):
                scores_vector[int(index)] = min_score - 1.0

    if 0 <= int(selected_index) < int(n_max) and bool(batch.candidate_mask[int(selected_index)]):
        if float(selected_boost) > 0.0:
            valid_scores = scores_vector[valid]
            scores_vector[int(selected_index)] = max(
                float(scores_vector[int(selected_index)]),
                float(np.max(valid_scores)) + float(selected_boost),
            )
    return scores_vector.astype(np.float32), int(ranker_argmax)


def _parse_weight_rules(text: str) -> dict[tuple[str, str], float]:
    rules: dict[tuple[str, str], float] = {}
    for item in str(text or "").split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token or ":" not in token:
            raise ValueError(
                "Hard-case weight rules must use traffic_scenario:load_name=weight, "
                f"got {token!r}"
            )
        key_text, weight_text = token.split("=", 1)
        scenario, load = key_text.split(":", 1)
        rules[(scenario.strip(), load.strip())] = float(weight_text)
    return rules


def _sample_weight_for_episode(episode: pd.DataFrame, rules: dict[tuple[str, str], float]) -> float:
    scenario = str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else ""
    load = str(episode["load_name"].iloc[0]) if "load_name" in episode else ""
    return float(rules.get((scenario, load), 1.0))


def _resolve_cli_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def _student_solver(
    *,
    config: ExperimentConfig,
    checkpoint_path: Path,
    epsilon: float,
) -> GnnCnnDqnOngSolver:
    solver_cfg = replace(
        _solver_config(config, neural=False),
        use_neural=True,
        checkpoint_path=str(checkpoint_path),
        epsilon=float(epsilon),
    )
    return GnnCnnDqnOngSolver(solver_cfg)


def _student_behavior_index(batch: Any, solver: GnnCnnDqnOngSolver) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1
    q_values = solver.q_values(batch)
    selected = masked_argmax(q_values, batch.candidate_mask)
    if int(selected) < 0:
        return int(valid[0])
    return int(selected)


def _select_episode_ids(episodes: pd.DataFrame, max_episodes: int, mode: str) -> tuple[str, ...]:
    episode_ids = tuple(str(value) for value in episodes["episode_id"].tolist())
    if max_episodes <= 0:
        return episode_ids
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "first":
        return episode_ids[:max_episodes]
    if normalized != "stratified":
        raise ValueError(f"Unsupported episode selection mode: {mode}")

    keys = ["traffic_scenario", "load_name"]
    if not all(key in episodes.columns for key in keys):
        return episode_ids[:max_episodes]
    groups = [
        [str(value) for value in group["episode_id"].tolist()]
        for _group_key, group in episodes.groupby(keys, sort=False)
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
    return tuple(selected)


def _load_traffic(
    config: ExperimentConfig,
    split: str,
    max_episodes: int,
    episode_selection: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    traffic_path = config.dataset_path / "traffic" / f"{split}.parquet"
    traffic = pd.read_parquet(traffic_path)
    episodes = traffic.drop_duplicates("episode_id").reset_index(drop=True)
    episode_ids = _select_episode_ids(episodes, max_episodes, episode_selection)
    return traffic, episode_ids


def _collect_examples(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    ranker: TreeCandidateRanker,
    teacher_base_policy: str,
    behavior_policy: str,
    student_checkpoint: Path | None,
    student_epsilon: float,
    episode_selection: str,
    hard_case_weight_rules: dict[tuple[str, str], float],
    teacher_score_selected_boost: float,
    print_every_episodes: int,
) -> tuple[list[DistillExample], np.ndarray, dict[str, Any]]:
    traffic, episode_ids = _load_traffic(config, split, max_episodes, episode_selection)
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    behavior_solver: GnnCnnDqnOngSolver | None = None
    normalized_behavior = str(behavior_policy).strip().lower().replace("-", "_")
    if normalized_behavior == "student_dqn":
        if student_checkpoint is None:
            raise ValueError("student_dqn behavior requires --student-checkpoint")
        behavior_solver = _student_solver(
            config=config,
            checkpoint_path=student_checkpoint,
            epsilon=float(student_epsilon),
        )
    elif normalized_behavior != "teacher":
        raise ValueError(f"Unsupported behavior policy: {behavior_policy}")
    cfg = solver.config
    run_path = output_dir / f"collect_{split}"

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
                teacher_index, teacher_margin, did_fallback = _valid_teacher_index(
                    batch=batch,
                    ranker=ranker,
                    n_max=cfg.n_max,
                    fallback_policy=teacher_base_policy,
                )
                fallback_teacher += int(bool(did_fallback))
                teacher_scores, teacher_ranker_argmax = _teacher_score_vector(
                    batch=batch,
                    ranker=ranker,
                    n_max=cfg.n_max,
                    selected_index=int(teacher_index),
                    selected_boost=float(teacher_score_selected_boost),
                )
                arrays = _batch_to_arrays(batch, cfg)
                examples.append(
                    DistillExample(
                        arrays=arrays,
                        teacher_index=int(teacher_index),
                        teacher_scores=teacher_scores,
                        teacher_margin=float(teacher_margin),
                        behavior_index=-1,
                        valid_count=int(valid.size),
                        sample_weight=float(sample_weight),
                        episode_id=str(episode_id),
                        request_id=int(episode_requests),
                        traffic_scenario=traffic_scenario,
                        load_name=load_name,
                        teacher_ranker_argmax=int(teacher_ranker_argmax),
                    )
                )
                if edge_index is None:
                    edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
                episode_examples += 1
                if normalized_behavior == "teacher":
                    behavior_index = int(teacher_index)
                else:
                    assert behavior_solver is not None
                    behavior_index = _student_behavior_index(batch, behavior_solver)
                examples[-1] = replace(examples[-1], behavior_index=int(behavior_index))
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
                        "fallback_teacher": int(fallback_teacher),
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


def _teacher_loss_mask(
    *,
    teacher_index: Any,
    behavior_index: Any,
    teacher_margin: Any,
    mode: str,
    min_teacher_margin: float,
    torch: Any,
) -> Any:
    normalized = str(mode or "all").strip().lower().replace("-", "_")
    mask = torch.ones_like(teacher_margin, dtype=torch.bool)
    if normalized in {"disagreement", "high_confidence_disagreement"}:
        mask = mask & (teacher_index != behavior_index)
    if normalized == "high_confidence_disagreement":
        mask = mask & (teacher_margin >= float(min_teacher_margin))
    elif normalized not in {"all", "disagreement"}:
        raise ValueError(f"Unsupported teacher loss filter: {mode}")
    return mask


def _distillation_loss(
    *,
    logits: Any,
    candidate_mask: Any,
    teacher_index: Any,
    teacher_scores: Any,
    sample_weight: Any,
    behavior_index: Any,
    teacher_margin: Any,
    reference_logits: Any | None,
    ce_weight: float,
    listwise_kl_weight: float,
    temperature: float,
    teacher_loss_filter: str,
    min_teacher_margin: float,
    reference_kl_weight: float,
    reference_temperature: float,
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

    teacher_per = float(ce_weight) * ce_per + float(listwise_kl_weight) * kl_per
    teacher_weight = sample_weight * teacher_mask.to(dtype=sample_weight.dtype)
    if bool(teacher_mask.any()):
        teacher_loss = (teacher_per * teacher_weight).sum() / torch.clamp(teacher_weight.sum(), min=1e-6)
    else:
        teacher_loss = teacher_per.sum() * 0.0

    if reference_logits is not None and float(reference_kl_weight) > 0.0:
        ref_temp = max(float(reference_temperature), 1e-6)
        reference_logits = reference_logits.masked_fill(~candidate_mask, -1e9)
        reference_probs = torch.nn.functional.softmax(reference_logits / ref_temp, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(logits / ref_temp, dim=1)
        reference_kl_per = (
            torch.nn.functional.kl_div(student_log_probs, reference_probs, reduction="none").sum(dim=1)
            * (ref_temp * ref_temp)
        )
        reference_weight = sample_weight / torch.clamp(sample_weight.mean(), min=1e-6)
        reference_kl_loss = (reference_kl_per * reference_weight).mean()
    else:
        reference_kl_per = torch.zeros_like(ce_per)
        reference_kl_loss = ce_per.sum() * 0.0

    loss = teacher_loss + float(reference_kl_weight) * reference_kl_loss
    return loss, {
        "ce_loss": float(ce_per.detach().mean().cpu()),
        "kl_loss": float(kl_per.detach().mean().cpu()),
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "teacher_loss_examples": int(teacher_mask.detach().sum().cpu()),
        "teacher_loss_fraction": float(teacher_mask.detach().float().mean().cpu()),
        "reference_kl_loss": float(reference_kl_loss.detach().cpu()),
        "reference_kl_per": float(reference_kl_per.detach().mean().cpu()),
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
    margins: list[float] = []
    student_indices: list[int] = []
    teacher_indices: list[int] = []
    valid_counts: list[int] = []
    sample_weights: list[float] = []
    context_rows: dict[tuple[str, str], dict[str, Any]] = {}
    rng = np.random.default_rng(0)
    batches = _iter_batches(len(examples), batch_size, shuffle=False, rng=rng)
    with torch.no_grad():
        for batch_indices in batches:
            tensors, teacher_index, _teacher_scores, _sample_weight, _behavior_index, _teacher_margin = _batch_examples(
                examples,
                batch_indices,
                device=device,
                torch=torch,
            )
            raw_values = _model_forward(model, tensors, edge_index)
            logits = raw_values.masked_fill(~tensors["candidate_mask"], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, teacher_index)
            prediction = logits.argmax(dim=1)
            teacher_q = logits.gather(1, teacher_index[:, None]).squeeze(1)
            masked = logits.clone()
            masked.scatter_(1, teacher_index[:, None], -1e9)
            other_q = masked.max(dim=1).values
            losses.append(float(loss.detach().cpu()))
            margins.extend(float(value) for value in (teacher_q - other_q).detach().cpu().numpy())
            correct += int((prediction == teacher_index).sum().detach().cpu())
            total += int(len(batch_indices))
            student_indices.extend(int(value) for value in prediction.detach().cpu().numpy())
            teacher_indices.extend(int(value) for value in teacher_index.detach().cpu().numpy())
            valid_counts.extend(int(examples[int(index)].valid_count) for index in batch_indices)
            sample_weights.extend(float(examples[int(index)].sample_weight) for index in batch_indices)
            prediction_np = prediction.detach().cpu().numpy()
            teacher_np = teacher_index.detach().cpu().numpy()
            for local_position, example_index in enumerate(batch_indices):
                item = examples[int(example_index)]
                key = (str(item.traffic_scenario), str(item.load_name))
                row = context_rows.setdefault(
                    key,
                    {
                        "traffic_scenario": str(item.traffic_scenario),
                        "load_name": str(item.load_name),
                        "examples": 0,
                        "correct": 0,
                        "student_index_sum": 0.0,
                        "teacher_index_sum": 0.0,
                        "sample_weight_sum": 0.0,
                    },
                )
                row["examples"] += 1
                row["correct"] += int(int(prediction_np[int(local_position)]) == int(teacher_np[int(local_position)]))
                row["student_index_sum"] += float(prediction_np[int(local_position)])
                row["teacher_index_sum"] += float(teacher_np[int(local_position)])
                row["sample_weight_sum"] += float(item.sample_weight)
    context_summary = []
    for row in context_rows.values():
        count = max(int(row["examples"]), 1)
        context_summary.append(
            {
                "traffic_scenario": row["traffic_scenario"],
                "load_name": row["load_name"],
                "examples": int(row["examples"]),
                "teacher_top1_accuracy": float(row["correct"] / count),
                "disagreement_rate": float(1.0 - row["correct"] / count),
                "mean_student_index": float(row["student_index_sum"] / count),
                "mean_teacher_index": float(row["teacher_index_sum"] / count),
                "mean_sample_weight": float(row["sample_weight_sum"] / count),
            }
        )
    context_summary.sort(key=lambda row: (-float(row["disagreement_rate"]), str(row["traffic_scenario"]), str(row["load_name"])))
    return {
        "examples": int(total),
        "loss": float(np.mean(losses)) if losses else None,
        "teacher_top1_accuracy": float(correct / max(total, 1)),
        "disagreement_rate": float(1.0 - correct / max(total, 1)),
        "mean_teacher_margin_logit": float(np.mean(margins)) if margins else None,
        "mean_student_index": float(np.mean(student_indices)) if student_indices else None,
        "mean_teacher_index": float(np.mean(teacher_indices)) if teacher_indices else None,
        "mean_valid_candidates": float(np.mean(valid_counts)) if valid_counts else None,
        "mean_sample_weight": float(np.mean(sample_weights)) if sample_weights else None,
        "context_audit": context_summary,
    }


def _load_full_dqn_checkpoint(model: Any, checkpoint_path: Path, *, device: str, torch: Any) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)


def _model_behavior_index(
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
        logits = _model_forward(model, tensors, edge_index).masked_fill(~tensors["candidate_mask"], -1e9)
        selected = int(logits.argmax(dim=1).detach().cpu().numpy()[0])
    if int(selected) < 0 or not bool(batch.candidate_mask[int(selected)]):
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
                selected_index = _model_behavior_index(
                    model=model,
                    batch=batch,
                    cfg=cfg,
                    device=device,
                    torch=torch,
                )
                if int(selected_index) < 0 or not bool(batch.candidate_mask[int(selected_index)]):
                    invalid_selected += 1
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


def _train(
    *,
    config: ExperimentConfig,
    train_examples: list[DistillExample],
    val_examples: list[DistillExample],
    edge_index_np: np.ndarray,
    output_dir: Path,
    teacher_artifact: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    torch.manual_seed(int(config.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))

    model, pretrained = _build_model(config, device, torch)
    initial_checkpoint = _resolve_cli_path(str(args.initial_checkpoint or ""))
    if initial_checkpoint is not None:
        _load_full_dqn_checkpoint(model, initial_checkpoint, device=device, torch=torch)
        pretrained = dict(pretrained)
        pretrained["initial_dqn_checkpoint_override"] = str(initial_checkpoint)
    reference_model = None
    reference_checkpoint = _resolve_cli_path(str(args.reference_checkpoint or ""))
    if reference_checkpoint is not None:
        reference_model, _reference_pretrained = _build_model(config, device, torch)
        _load_full_dqn_checkpoint(reference_model, reference_checkpoint, device=device, torch=torch)
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad_(False)
        pretrained = dict(pretrained)
        pretrained["reference_dqn_checkpoint"] = str(reference_checkpoint)
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
    best_path = output_dir / "full_dqn_orate60_distill_frozen.pt"
    history: list[dict[str, Any]] = []

    if str(args.checkpoint_selection) != "teacher_top1":
        initial_rollout_val_eval = _rollout_validate_model(
            model=model,
            config=config,
            output_dir=output_dir,
            split=str(args.rollout_val_split),
            max_episodes=int(args.rollout_val_max_episodes),
            max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
            episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
            device=device,
            torch=torch,
        )
        if str(args.checkpoint_selection) == "rollout_accepted":
            best_metric = float(initial_rollout_val_eval.get("accepted") or 0.0)
        elif str(args.checkpoint_selection) == "rollout_reward":
            best_metric = float(initial_rollout_val_eval.get("total_reward") or 0.0)
        best_rollout_val_eval = initial_rollout_val_eval
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "target_model_state_dict": model.state_dict(),
                "epoch": 0,
                "config": {
                    "hidden_dim": int(_raw_int(config, "hidden_dim", 128)),
                    "n_max": int(_raw_int(config, "n_max", 32)),
                    "q_score_mode": "raw",
                    "residual_scale": 1.0,
                    "residual_delta_clip": 0.0,
                    "freeze_encoders": bool(_raw_bool(config, "freeze_encoders", True)),
                    "teacher_artifact": str(teacher_artifact),
                    "teacher_selection_mode": str(args.teacher_selection_mode),
                    "teacher_safety_guard": str(args.teacher_safety_guard),
                    "behavior_policy": str(args.behavior_policy),
                    "checkpoint_selection": str(args.checkpoint_selection),
                    "ce_weight": float(args.ce_weight),
                    "listwise_kl_weight": float(args.listwise_kl_weight),
                    "listwise_temperature": float(args.listwise_temperature),
                    "teacher_loss_filter": str(args.teacher_loss_filter),
                    "min_teacher_margin": float(args.min_teacher_margin),
                    "reference_kl_weight": float(args.reference_kl_weight),
                    "reference_temperature": float(args.reference_temperature),
                    "hard_case_weight_rules": str(args.hard_case_weight_rules),
                },
                "pretrained": pretrained,
                "history": history,
                "initial_rollout_val_eval": initial_rollout_val_eval,
            },
            best_path,
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
        epoch_teacher_losses: list[float] = []
        epoch_teacher_loss_fractions: list[float] = []
        epoch_reference_kl_losses: list[float] = []
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
            raw_values = _model_forward(model, tensors, edge_index)
            logits = raw_values.masked_fill(~tensors["candidate_mask"], -1e9)
            reference_logits = None
            if reference_model is not None and float(args.reference_kl_weight) > 0.0:
                with torch.no_grad():
                    reference_logits = _model_forward(reference_model, tensors, edge_index)
            loss, loss_parts = _distillation_loss(
                logits=logits,
                candidate_mask=tensors["candidate_mask"],
                teacher_index=teacher_index,
                teacher_scores=teacher_scores,
                sample_weight=sample_weight,
                behavior_index=behavior_index,
                teacher_margin=teacher_margin,
                reference_logits=reference_logits,
                ce_weight=float(args.ce_weight),
                listwise_kl_weight=float(args.listwise_kl_weight),
                temperature=float(args.listwise_temperature),
                teacher_loss_filter=str(args.teacher_loss_filter),
                min_teacher_margin=float(args.min_teacher_margin),
                reference_kl_weight=float(args.reference_kl_weight),
                reference_temperature=float(args.reference_temperature),
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
            epoch_teacher_losses.append(float(loss_parts["teacher_loss"]))
            epoch_teacher_loss_fractions.append(float(loss_parts["teacher_loss_fraction"]))
            epoch_reference_kl_losses.append(float(loss_parts["reference_kl_loss"]))
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
                            "train_teacher_loss_mean": float(np.mean(epoch_teacher_losses)),
                            "train_teacher_loss_fraction_mean": float(np.mean(epoch_teacher_loss_fractions)),
                            "train_reference_kl_loss_mean": float(np.mean(epoch_reference_kl_losses)),
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
            rollout_val_eval = _rollout_validate_model(
                model=model,
                config=config,
                output_dir=output_dir,
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
                device=device,
                torch=torch,
            )
        row = {
            "phase": "epoch",
            "epoch": int(epoch),
            "train_loss_online": float(np.mean(epoch_losses)) if epoch_losses else None,
            "train_ce_loss_online": float(np.mean(epoch_ce_losses)) if epoch_ce_losses else None,
            "train_kl_loss_online": float(np.mean(epoch_kl_losses)) if epoch_kl_losses else None,
            "train_teacher_loss_online": float(np.mean(epoch_teacher_losses)) if epoch_teacher_losses else None,
            "train_teacher_loss_fraction_online": float(np.mean(epoch_teacher_loss_fractions)) if epoch_teacher_loss_fractions else None,
            "train_reference_kl_loss_online": float(np.mean(epoch_reference_kl_losses)) if epoch_reference_kl_losses else None,
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
            raise ValueError(f"Unsupported checkpoint selection: {args.checkpoint_selection}")
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
            best_rollout_val_eval = rollout_val_eval
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": model.state_dict(),
                    "epoch": int(epoch),
                    "config": {
                        "hidden_dim": int(_raw_int(config, "hidden_dim", 128)),
                        "n_max": int(_raw_int(config, "n_max", 32)),
                        "q_score_mode": "raw",
                        "residual_scale": 1.0,
                        "residual_delta_clip": 0.0,
                        "freeze_encoders": bool(_raw_bool(config, "freeze_encoders", True)),
                        "teacher_artifact": str(teacher_artifact),
                        "teacher_selection_mode": str(args.teacher_selection_mode),
                        "teacher_safety_guard": str(args.teacher_safety_guard),
                        "behavior_policy": str(args.behavior_policy),
                        "checkpoint_selection": str(args.checkpoint_selection),
                        "ce_weight": float(args.ce_weight),
                        "listwise_kl_weight": float(args.listwise_kl_weight),
                        "listwise_temperature": float(args.listwise_temperature),
                        "teacher_loss_filter": str(args.teacher_loss_filter),
                        "min_teacher_margin": float(args.min_teacher_margin),
                        "reference_kl_weight": float(args.reference_kl_weight),
                        "reference_temperature": float(args.reference_temperature),
                        "hard_case_weight_rules": str(args.hard_case_weight_rules),
                    },
                    "pretrained": pretrained,
                    "history": history,
                },
                best_path,
            )

    summary = {
        "stage": "train_full_dqn_orate60_distill",
        "checkpoint_path": str(best_path),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "best_val_teacher_top1_accuracy": float(
            (history[int(best_epoch) - 1]["val_eval"].get("teacher_top1_accuracy") if best_epoch > 0 else 0.0) or 0.0
        ),
        "best_rollout_val_eval": best_rollout_val_eval,
        "initial_rollout_val_eval": initial_rollout_val_eval,
        "checkpoint_selection": str(args.checkpoint_selection),
        "device": device,
        "pretrained": pretrained,
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "ce_weight": float(args.ce_weight),
        "listwise_kl_weight": float(args.listwise_kl_weight),
        "listwise_temperature": float(args.listwise_temperature),
        "teacher_loss_filter": str(args.teacher_loss_filter),
        "min_teacher_margin": float(args.min_teacher_margin),
        "reference_kl_weight": float(args.reference_kl_weight),
        "reference_temperature": float(args.reference_temperature),
        "teacher_score_selected_boost": float(args.teacher_score_selected_boost),
        "hard_case_weight_rules": str(args.hard_case_weight_rules),
        "teacher_artifact": str(teacher_artifact),
        "history": history,
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill the orate60 tree-ranker into the full GNN+CNN DQN scorer.")
    parser.add_argument("--config", required=True, help="Experiment YAML with dataset/model parameters.")
    parser.add_argument("--teacher-artifact", required=True, help="TreeCandidateRanker JSON artifact used as teacher.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and summaries.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-max-episodes", type=int, default=8)
    parser.add_argument("--val-max-episodes", type=int, default=4)
    parser.add_argument("--episode-selection", choices=("first", "stratified"), default="first")
    parser.add_argument("--max-requests-per-episode", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--listwise-kl-weight", type=float, default=0.0)
    parser.add_argument("--listwise-temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher-loss-filter",
        choices=("all", "disagreement", "high_confidence_disagreement"),
        default="all",
    )
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--reference-checkpoint", default="", help="Optional frozen DQN checkpoint for behavior KL regularization.")
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--reference-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-score-selected-boost", type=float, default=0.0)
    parser.add_argument(
        "--hard-case-weight-rules",
        default="",
        help="Comma-separated traffic_scenario:load_name=weight rules, e.g. hotspot:high=1.7,nonuniform:medium=1.7",
    )
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--print-every-episodes", type=int, default=1)
    parser.add_argument("--behavior-policy", choices=("teacher", "student_dqn"), default="teacher")
    parser.add_argument("--student-checkpoint", default="", help="Checkpoint used by student_dqn behavior collection.")
    parser.add_argument("--student-epsilon", type=float, default=0.0)
    parser.add_argument("--initial-checkpoint", default="", help="Optional full DQN checkpoint used to warm-start training.")
    parser.add_argument(
        "--checkpoint-selection",
        choices=("teacher_top1", "rollout_accepted", "rollout_reward"),
        default="teacher_top1",
    )
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
    teacher_artifact = Path(args.teacher_artifact)
    if not teacher_artifact.is_absolute():
        teacher_artifact = ROOT / teacher_artifact
    student_checkpoint = _resolve_cli_path(str(args.student_checkpoint or ""))
    initial_checkpoint = _resolve_cli_path(str(args.initial_checkpoint or ""))
    reference_checkpoint = _resolve_cli_path(str(args.reference_checkpoint or ""))

    ong_source = _add_ong_source_path(config)
    ranker = TreeCandidateRanker.load(
        teacher_artifact,
        selection_mode=str(args.teacher_selection_mode),
        base_policy=str(args.teacher_base_policy),
        safety_guard=_teacher_safety_guard(args),
    )
    hard_case_weight_rules = _parse_weight_rules(str(args.hard_case_weight_rules))
    print(
        json.dumps(
            {
                "phase": "start",
                "dataset_path": str(config.dataset_path),
                "output_dir": str(output_dir),
                "ong_source_path": ong_source,
                "teacher_artifact": str(teacher_artifact),
                "teacher_selection_mode": str(args.teacher_selection_mode),
                "teacher_safety_guard": str(args.teacher_safety_guard),
                "behavior_policy": str(args.behavior_policy),
                "student_checkpoint": None if student_checkpoint is None else str(student_checkpoint),
                "student_epsilon": float(args.student_epsilon),
                "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
                "reference_checkpoint": None if reference_checkpoint is None else str(reference_checkpoint),
                "checkpoint_selection": str(args.checkpoint_selection),
                "rollout_val_split": str(args.rollout_val_split),
                "rollout_val_max_episodes": int(args.rollout_val_max_episodes),
                "episode_selection": str(args.episode_selection),
                "train_max_episodes": int(args.train_max_episodes),
                "val_max_episodes": int(args.val_max_episodes),
                "ce_weight": float(args.ce_weight),
                "listwise_kl_weight": float(args.listwise_kl_weight),
                "listwise_temperature": float(args.listwise_temperature),
                "teacher_loss_filter": str(args.teacher_loss_filter),
                "min_teacher_margin": float(args.min_teacher_margin),
                "reference_kl_weight": float(args.reference_kl_weight),
                "reference_temperature": float(args.reference_temperature),
                "teacher_score_selected_boost": float(args.teacher_score_selected_boost),
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
        ranker=ranker,
        teacher_base_policy=str(args.teacher_base_policy),
        behavior_policy=str(args.behavior_policy),
        student_checkpoint=student_checkpoint,
        student_epsilon=float(args.student_epsilon),
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=float(args.teacher_score_selected_boost),
        print_every_episodes=int(args.print_every_episodes),
    )
    val_examples, _val_edge_index, val_collect = _collect_examples(
        config=config,
        output_dir=output_dir,
        split=str(args.val_split),
        max_episodes=int(args.val_max_episodes),
        max_requests_per_episode=int(args.max_requests_per_episode),
        ranker=ranker,
        teacher_base_policy=str(args.teacher_base_policy),
        behavior_policy=str(args.behavior_policy),
        student_checkpoint=student_checkpoint,
        student_epsilon=float(args.student_epsilon),
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=float(args.teacher_score_selected_boost),
        print_every_episodes=int(args.print_every_episodes),
    )
    collect_summary = {
        "train_collect": train_collect,
        "val_collect": val_collect,
    }
    _write_json(output_dir / "collection_summary.json", collect_summary)
    print(json.dumps({"phase": "collection_summary", **collect_summary}, sort_keys=True), flush=True)

    summary = _train(
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        edge_index_np=edge_index,
        output_dir=output_dir,
        teacher_artifact=teacher_artifact,
        args=args,
    )
    print(json.dumps({"phase": "done", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
