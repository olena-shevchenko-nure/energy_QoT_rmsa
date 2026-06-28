#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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
    _add_ong_source_path,
    _make_env,
    _select_candidate,
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
from cse2026.ong_solver import GnnCnnDqnOngSolver
from cse2026.ong_solver.common import masked_argmax

from train_full_dqn_orate60_distill import (
    DistillExample,
    _json_safe,
    _load_full_dqn_checkpoint,
    _parse_weight_rules,
    _resolve_cli_path,
    _select_episode_ids,
    _write_json,
)
from train_gnn_cnn_a3c_distill import (
    _batch_examples,
    _collect_examples,
    _distillation_loss,
    _load_traffic,
    _teacher_score_margin,
)
from top32_xlron_live_risk_features import (
    live_risk_feature_vector,
    load_live_risk_artifact,
    parse_bucket_set as _parse_live_risk_bucket_set,
    predict_live_risk,
)


def _raw_bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _xlron_architecture_kwargs(args: argparse.Namespace, model_shapes: dict[str, int]) -> dict[str, Any]:
    architecture = str(args.xlron_architecture).strip().lower()
    full_default = architecture == "full"
    return {
        "architecture": architecture,
        "spectrum_channels": int(model_shapes.get("spectrum_channels", 6)),
        "route_basic_dim": int(model_shapes.get("route_basic_feature_dim", 2)),
        "candidate_transformer_layers": int(args.xlron_candidate_transformer_layers),
        "candidate_transformer_heads": int(args.xlron_candidate_transformer_heads),
        "enable_spectrum_branch": _raw_bool_value(args.xlron_enable_spectrum_branch, full_default),
        "enable_candidate_attention": _raw_bool_value(args.xlron_enable_candidate_attention, full_default),
        "enable_base_relative_branch": _raw_bool_value(args.xlron_enable_base_relative_branch, full_default),
        "enable_auxiliary_heads": _raw_bool_value(args.xlron_enable_auxiliary_heads, False),
    }


def _model_shapes_from_examples(examples: list[Any]) -> dict[str, int]:
    if not examples:
        raise ValueError("Cannot infer XLRON shapes from an empty example set")
    arrays = examples[0].arrays
    return {
        "action_feature_dim": int(arrays["action_features"].shape[1]),
        "link_feature_dim": int(arrays["link_features"].shape[1]),
        "global_feature_dim": int(arrays["global_features"].shape[0]),
        "request_feature_dim": int(arrays["request_features"].shape[0]),
        "spectrum_channels": int(arrays["spectrum_tensors"].shape[1]),
        "route_basic_feature_dim": int(arrays["route_basic_features"].shape[1]),
    }


def _load_xlron_checkpoint_model(checkpoint_path: Path, *, device: str, torch: Any) -> tuple[Any, dict[str, Any]]:
    from cse2026.ong_solver.models import XlronGraphTransformerPpoNetwork

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("XLRON checkpoint must be a dictionary")
    architecture = str(checkpoint.get("architecture", "link_transformer"))
    full_default = architecture.strip().lower() == "full"
    model = XlronGraphTransformerPpoNetwork(
        action_feature_dim=int(checkpoint.get("action_feature_dim", 10)),
        link_feature_dim=int(checkpoint.get("link_feature_dim", 8)),
        global_feature_dim=int(checkpoint.get("global_feature_dim", 8)),
        request_feature_dim=int(checkpoint.get("request_feature_dim", 3)),
        embedding_dim=int(checkpoint.get("embedding_dim", checkpoint.get("hidden_dim", 128))),
        num_layers=int(checkpoint.get("transformer_num_layers", 2)),
        num_heads=int(checkpoint.get("transformer_num_heads", 8)),
        dropout=float(checkpoint.get("dropout", 0.0)),
        position_dim=int(checkpoint.get("position_dim", 8)),
        architecture=architecture,
        spectrum_channels=int(checkpoint.get("spectrum_channels", 6)),
        route_basic_dim=int(checkpoint.get("route_basic_dim", checkpoint.get("route_basic_feature_dim", 2))),
        candidate_transformer_layers=int(checkpoint.get("candidate_transformer_layers", 1 if full_default else 0)),
        candidate_transformer_heads=int(checkpoint.get("candidate_transformer_heads", 4)),
        enable_spectrum_branch=bool(checkpoint.get("enable_spectrum_branch", full_default)),
        enable_candidate_attention=bool(checkpoint.get("enable_candidate_attention", full_default)),
        enable_base_relative_branch=bool(checkpoint.get("enable_base_relative_branch", full_default)),
        enable_auxiliary_heads=bool(checkpoint.get("enable_auxiliary_heads", False)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _scale_example_weights(examples: list[Any], factor: float) -> list[Any]:
    return [replace(item, sample_weight=float(item.sample_weight) * float(factor)) for item in examples]


def _parse_bucket_set(text: str) -> set[str]:
    buckets: set[str] = set()
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Hard DAgger bucket must be scenario:load, got {item!r}")
        buckets.add(item)
    return buckets


def _parse_bucket_float_map(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bucket float rule must be scenario:load=value, got {item!r}")
        bucket, value = item.split("=", 1)
        bucket = bucket.strip()
        if ":" not in bucket:
            raise ValueError(f"Bucket float rule must be scenario:load=value, got {item!r}")
        values[bucket] = float(value)
    return values


def _example_margin_stats(examples: list[Any]) -> dict[str, Any]:
    margins = np.asarray([float(item.teacher_margin) for item in examples], dtype=np.float64)
    if margins.size == 0:
        return {
            "teacher_margin_mean": None,
            "teacher_margin_p50": None,
            "teacher_margin_p90": None,
            "teacher_margin_gt_1e-3_rate": None,
            "teacher_margin_gt_1e-2_rate": None,
        }
    return {
        "teacher_margin_mean": float(np.mean(margins)),
        "teacher_margin_p50": float(np.percentile(margins, 50)),
        "teacher_margin_p90": float(np.percentile(margins, 90)),
        "teacher_margin_gt_1e-3_rate": float(np.mean(margins > 1e-3)),
        "teacher_margin_gt_1e-2_rate": float(np.mean(margins > 1e-2)),
    }


def _hard_dagger_transform_examples(
    examples: list[Any],
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[Any], dict[str, Any]]:
    hard_buckets = _parse_bucket_set(str(args.hard_dagger_loss_buckets))
    disagreement_weight = max(0.0, float(args.hard_dagger_disagreement_weight))
    loss_bucket_weight = max(0.0, float(args.hard_dagger_loss_bucket_weight))
    agreement_weight = max(0.0, float(args.hard_dagger_agreement_weight))
    agreement_keep_frac = min(1.0, max(0.0, float(args.hard_dagger_agreement_keep_frac)))
    max_weight = max(0.0, float(args.hard_dagger_max_weight))

    transformed: list[Any] = []
    before_weights: list[float] = []
    after_weights: list[float] = []
    input_disagreements = 0
    kept_disagreements = 0
    kept_agreements = 0
    kept_hard_bucket = 0
    input_by_bucket: dict[str, int] = {}
    kept_by_bucket: dict[str, int] = {}
    disagreement_by_bucket: dict[str, int] = {}

    for item in examples:
        bucket = f"{item.traffic_scenario}:{item.load_name}"
        disagreement = int(item.teacher_index) != int(item.behavior_index)
        is_hard_bucket = bucket in hard_buckets
        input_by_bucket[bucket] = int(input_by_bucket.get(bucket, 0)) + 1
        before_weights.append(float(item.sample_weight))
        if disagreement:
            input_disagreements += 1
            disagreement_by_bucket[bucket] = int(disagreement_by_bucket.get(bucket, 0)) + 1

        keep_probability = 1.0
        if not disagreement and not is_hard_bucket:
            keep_probability = agreement_keep_frac
        if keep_probability < 1.0 and float(rng.random()) > keep_probability:
            continue

        weight = float(item.sample_weight)
        weight *= disagreement_weight if disagreement else agreement_weight
        if is_hard_bucket:
            weight *= loss_bucket_weight
        if max_weight > 0.0:
            weight = min(weight, max_weight)
        transformed.append(replace(item, sample_weight=float(weight)))
        after_weights.append(float(weight))
        kept_by_bucket[bucket] = int(kept_by_bucket.get(bucket, 0)) + 1
        kept_disagreements += int(disagreement)
        kept_agreements += int(not disagreement)
        kept_hard_bucket += int(is_hard_bucket)

    if not transformed:
        raise RuntimeError("Hard DAgger filtering removed all training examples")
    summary = {
        "enabled": bool(
            hard_buckets
            or disagreement_weight != 1.0
            or loss_bucket_weight != 1.0
            or agreement_weight != 1.0
            or agreement_keep_frac != 1.0
            or max_weight > 0.0
        ),
        "input_examples": int(len(examples)),
        "kept_examples": int(len(transformed)),
        "input_disagreements": int(input_disagreements),
        "kept_disagreements": int(kept_disagreements),
        "kept_agreements": int(kept_agreements),
        "kept_hard_bucket_examples": int(kept_hard_bucket),
        "input_disagreement_rate": float(input_disagreements / max(len(examples), 1)),
        "kept_disagreement_rate": float(kept_disagreements / max(len(transformed), 1)),
        "agreement_keep_frac": float(agreement_keep_frac),
        "disagreement_weight": float(disagreement_weight),
        "agreement_weight": float(agreement_weight),
        "loss_bucket_weight": float(loss_bucket_weight),
        "max_weight": float(max_weight),
        "loss_buckets": sorted(hard_buckets),
        "input_by_bucket": dict(sorted(input_by_bucket.items())),
        "kept_by_bucket": dict(sorted(kept_by_bucket.items())),
        "disagreement_by_bucket": dict(sorted(disagreement_by_bucket.items())),
        "mean_weight_before": float(np.mean(before_weights)) if before_weights else None,
        "mean_weight_after": float(np.mean(after_weights)) if after_weights else None,
        "margin_stats_before": _example_margin_stats(examples),
        "margin_stats_after": _example_margin_stats(transformed),
    }
    return transformed, summary


COUNTERFACTUAL_STATE_KEYS = (
    "node_features",
    "link_features",
    "global_features",
    "request_features",
    "spectrum_tensors",
    "action_features",
    "route_link_mask",
    "route_basic_features",
    "block_bounds",
    "candidate_mask",
)


def _dqn_teacher_scores_from_arrays(
    *,
    teacher_model: Any,
    arrays: dict[str, np.ndarray],
    edge_index_np: np.ndarray,
    device: str,
    torch: Any,
) -> np.ndarray:
    tensors = _stack_state_arrays([arrays], device, torch)
    edge_index = torch.as_tensor(np.asarray(edge_index_np, dtype=np.int64), dtype=torch.long, device=device)
    with torch.no_grad():
        values = _dqn_forward(teacher_model, tensors, edge_index)
        values = values.masked_fill(~tensors["candidate_mask"], -1.0e9)
    return values.detach().cpu().numpy().reshape(-1).astype(np.float32)


def _load_counterfactual_aux_examples(
    input_dir: Path | None,
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    dqn_teacher_model: Any | None = None,
    edge_index_np: np.ndarray | None = None,
    device: str = "cpu",
    torch: Any | None = None,
) -> tuple[list[DistillExample], dict[str, Any] | None]:
    if input_dir is None:
        return [], None
    mode = str(args.counterfactual_aux_mode).strip().lower()
    if mode not in {"hard_masked", "soft_blend"}:
        raise ValueError(f"Unsupported counterfactual aux mode: {args.counterfactual_aux_mode}")
    if mode == "soft_blend" and (dqn_teacher_model is None or edge_index_np is None or torch is None):
        raise ValueError("counterfactual soft_blend requires a DQN teacher model, edge_index, and torch")
    metadata_path = input_dir / "online_base_topn_examples.csv"
    neural_path = input_dir / "online_base_topn_neural_states.npz"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing counterfactual metadata: {metadata_path}")
    if not neural_path.exists():
        raise FileNotFoundError(f"Missing counterfactual neural states: {neural_path}")

    metadata = pd.read_csv(metadata_path).reset_index(drop=True)
    npz = np.load(neural_path, allow_pickle=True)
    data = {str(key): np.asarray(npz[key]) for key in npz.files}
    for key in COUNTERFACTUAL_STATE_KEYS + (
        "group_ids",
        "base_index",
        "accepted_delta_vs_base",
        "target_delta",
        "label_mask",
    ):
        if key not in data:
            raise ValueError(f"Counterfactual neural state file is missing {key}")

    group_ids = np.asarray(data["group_ids"], dtype=np.int64)
    n_max = int(np.asarray(data["candidate_mask"]).shape[1])
    indices = np.arange(len(group_ids), dtype=np.int64)
    if int(args.counterfactual_aux_max_examples) > 0 and len(indices) > int(args.counterfactual_aux_max_examples):
        rng.shuffle(indices)
        indices = np.sort(indices[: int(args.counterfactual_aux_max_examples)])

    meta_by_group = {int(group_id): group.reset_index(drop=True) for group_id, group in metadata.groupby("group_id", sort=False)}
    examples: list[DistillExample] = []
    skipped_tie_only = 0
    skipped_unusable = 0
    win_groups = 0
    loss_groups = 0
    tie_groups = 0
    weights: list[float] = []
    selected_indices: list[int] = []
    max_abs_deltas: list[float] = []
    dqn_selected_indices: list[int] = []
    changed_by_delta = 0

    score_boost = float(args.counterfactual_aux_score_boost)
    target_scale = max(float(args.counterfactual_aux_target_scale), 1.0e-6)
    score_clip = max(float(args.counterfactual_aux_score_clip), 0.0)
    magnitude_cap = max(float(args.counterfactual_aux_magnitude_cap), 0.0)
    magnitude_weight = max(float(args.counterfactual_aux_magnitude_weight), 0.0)

    for position in indices:
        raw_candidate_mask = np.asarray(data["candidate_mask"][position], dtype=bool)
        label_mask = np.asarray(data["label_mask"][position], dtype=bool) & raw_candidate_mask
        if int(label_mask.sum()) < 2:
            skipped_unusable += 1
            continue

        accepted_delta = np.nan_to_num(
            np.asarray(data["accepted_delta_vs_base"][position], dtype=np.float32),
            nan=0.0,
        )
        target_delta = np.nan_to_num(
            np.asarray(data["target_delta"][position], dtype=np.float32),
            nan=0.0,
        )
        nonbase_mask = label_mask.copy()
        base_index = int(np.asarray(data["base_index"], dtype=np.int64)[position])
        if 0 <= base_index < n_max:
            nonbase_mask[base_index] = False

        nonbase_accepted = accepted_delta[nonbase_mask]
        max_accepted = float(np.max(nonbase_accepted)) if nonbase_accepted.size else 0.0
        min_accepted = float(np.min(nonbase_accepted)) if nonbase_accepted.size else 0.0
        has_win = max_accepted > 0.0
        has_loss = min_accepted < 0.0
        if not bool(args.counterfactual_aux_include_tie_only) and not has_win and not has_loss:
            skipped_tie_only += 1
            continue

        arrays = {key: np.asarray(data[key][position]).copy() for key in COUNTERFACTUAL_STATE_KEYS}
        normalized_target = target_delta / target_scale
        if score_clip > 0.0:
            normalized_target = np.clip(normalized_target, -score_clip, score_clip)
        if mode == "hard_masked":
            train_mask = label_mask.astype(np.bool_)
            blended_scores = np.full((n_max,), -1.0e9, dtype=np.float32)
            blended_scores[train_mask] = (score_boost * normalized_target[train_mask]).astype(np.float32)
            dqn_index = -1
        else:
            train_mask = raw_candidate_mask.astype(np.bool_)
            arrays["candidate_mask"] = train_mask
            blended_scores = _dqn_teacher_scores_from_arrays(
                teacher_model=dqn_teacher_model,
                arrays=arrays,
                edge_index_np=np.asarray(edge_index_np, dtype=np.int64),
                device=str(device),
                torch=torch,
            )
            blended_scores[~train_mask] = -1.0e9
            dqn_index = int(masked_argmax(blended_scores, train_mask))
            blended_scores[label_mask] = (
                blended_scores[label_mask] + score_boost * normalized_target[label_mask]
            ).astype(np.float32)
        teacher_index = int(masked_argmax(blended_scores, train_mask))
        if teacher_index < 0 or not bool(train_mask[teacher_index]):
            skipped_unusable += 1
            continue

        arrays["candidate_mask"] = train_mask.astype(np.bool_)
        weight = float(args.counterfactual_aux_weight)
        if has_win:
            weight *= float(args.counterfactual_aux_win_weight)
            win_groups += 1
        elif has_loss:
            weight *= float(args.counterfactual_aux_loss_weight)
            loss_groups += 1
        else:
            weight *= float(args.counterfactual_aux_tie_weight)
            tie_groups += 1
        max_abs_delta = float(np.max(np.abs(accepted_delta[nonbase_mask]))) if nonbase_accepted.size else 0.0
        if magnitude_weight > 0.0 and max_abs_delta > 0.0:
            weight *= 1.0 + magnitude_weight * min(max_abs_delta, magnitude_cap if magnitude_cap > 0.0 else max_abs_delta)

        group_id = int(group_ids[position])
        group_meta = meta_by_group.get(group_id)
        if group_meta is None or group_meta.empty:
            episode_id = f"cf_group_{group_id}"
            request_id = int(group_id)
            traffic_scenario = "counterfactual"
            load_name = "unknown"
        else:
            row = group_meta.iloc[0]
            episode_id = str(row.get("episode_id", f"cf_group_{group_id}"))
            request_id = int(row.get("request_id", group_id))
            traffic_scenario = str(row.get("traffic_scenario", ""))
            load_name = str(row.get("load_name", ""))

        teacher_margin = _teacher_score_margin(blended_scores, train_mask)
        examples.append(
            DistillExample(
                arrays=arrays,
                teacher_index=int(teacher_index),
                teacher_scores=np.asarray(blended_scores, dtype=np.float32),
                teacher_margin=float(teacher_margin),
                behavior_index=int(base_index if 0 <= base_index < n_max and train_mask[base_index] else teacher_index),
                valid_count=int(train_mask.sum()),
                sample_weight=float(weight),
                episode_id=str(episode_id),
                request_id=int(request_id),
                traffic_scenario=str(traffic_scenario),
                load_name=str(load_name),
                teacher_ranker_argmax=int(teacher_index),
            )
        )
        weights.append(float(weight))
        selected_indices.append(int(teacher_index))
        max_abs_deltas.append(float(max_abs_delta))
        if dqn_index >= 0:
            dqn_selected_indices.append(int(dqn_index))
            changed_by_delta += int(teacher_index != dqn_index)

    summary = {
        "enabled": True,
        "mode": str(mode),
        "input_dir": str(input_dir),
        "metadata_rows": int(len(metadata)),
        "input_groups": int(len(group_ids)),
        "used_examples": int(len(examples)),
        "skipped_unusable": int(skipped_unusable),
        "skipped_tie_only": int(skipped_tie_only),
        "win_groups": int(win_groups),
        "loss_groups": int(loss_groups),
        "tie_groups": int(tie_groups),
        "mean_weight": float(np.mean(weights)) if weights else None,
        "max_weight": float(np.max(weights)) if weights else None,
        "mean_teacher_index": float(np.mean(selected_indices)) if selected_indices else None,
        "mean_dqn_teacher_index": float(np.mean(dqn_selected_indices)) if dqn_selected_indices else None,
        "changed_by_delta_count": int(changed_by_delta),
        "changed_by_delta_rate": float(changed_by_delta / max(len(dqn_selected_indices), 1)) if dqn_selected_indices else None,
        "mean_max_abs_accepted_delta": float(np.mean(max_abs_deltas)) if max_abs_deltas else None,
        "score_boost": float(score_boost),
        "target_scale": float(target_scale),
        "score_clip": float(score_clip),
        "include_tie_only": bool(args.counterfactual_aux_include_tie_only),
    }
    return examples, summary


def _select_episode_ids_with_offset(episodes: Any, max_episodes: int, mode: str, episode_offset: int) -> tuple[str, ...]:
    offset = max(0, int(episode_offset))
    if offset <= 0:
        return _select_episode_ids(episodes, max_episodes, mode)

    episode_ids = tuple(str(value) for value in episodes["episode_id"].tolist())
    normalized = str(mode).strip().lower().replace("-", "_")
    if max_episodes <= 0:
        max_episodes = len(episode_ids)
    if normalized == "first":
        rotated = episode_ids[offset % max(len(episode_ids), 1) :] + episode_ids[: offset % max(len(episode_ids), 1)]
        return tuple(rotated[:max_episodes])
    if normalized != "stratified":
        raise ValueError(f"Unsupported episode selection mode: {mode}")

    keys = ["traffic_scenario", "load_name"]
    if not all(key in episodes.columns for key in keys):
        rotated = episode_ids[offset % max(len(episode_ids), 1) :] + episode_ids[: offset % max(len(episode_ids), 1)]
        return tuple(rotated[:max_episodes])

    groups: list[list[str]] = []
    for _group_key, group in episodes.groupby(keys, sort=False):
        values = [str(value) for value in group["episode_id"].tolist()]
        if values:
            shift = offset % len(values)
            groups.append(values[shift:] + values[:shift])

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


def _load_rollout_traffic(
    config: ExperimentConfig,
    split: str,
    max_episodes: int,
    episode_selection: str,
    episode_offset: int,
) -> tuple[Any, tuple[str, ...]]:
    if int(episode_offset) <= 0:
        return _load_traffic(config, split, max_episodes, episode_selection)
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episodes = traffic.drop_duplicates("episode_id").reset_index(drop=True)
    episode_ids = _select_episode_ids_with_offset(episodes, max_episodes, episode_selection, int(episode_offset))
    return traffic, episode_ids


def _bucket_label(episode: Any) -> str:
    scenario = str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else ""
    load_name = str(episode["load_name"].iloc[0]) if "load_name" in episode else ""
    return f"{scenario}:{load_name}"


def _empty_bucket_stats() -> dict[str, Any]:
    return {"episodes": 0, "requests": 0, "accepted": 0, "blocked": 0, "total_reward": 0.0}


def _add_bucket_stats(
    buckets: dict[str, dict[str, Any]],
    bucket: str,
    *,
    requests: int,
    accepted: int,
    total_reward: float,
    episode_done: bool,
) -> None:
    stats = buckets.setdefault(bucket, _empty_bucket_stats())
    stats["requests"] = int(stats["requests"]) + int(requests)
    stats["accepted"] = int(stats["accepted"]) + int(accepted)
    stats["blocked"] = int(stats["requests"]) - int(stats["accepted"])
    stats["total_reward"] = float(stats["total_reward"]) + float(total_reward)
    if bool(episode_done):
        stats["episodes"] = int(stats["episodes"]) + 1


def _finalize_bucket_stats(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for bucket, stats in sorted(buckets.items()):
        requests = int(stats.get("requests", 0))
        accepted = int(stats.get("accepted", 0))
        finalized[bucket] = {
            "episodes": int(stats.get("episodes", 0)),
            "requests": requests,
            "accepted": accepted,
            "blocked": int(requests - accepted),
            "blocking_rate": float((requests - accepted) / max(requests, 1)),
            "total_reward": float(stats.get("total_reward", 0.0)),
        }
    return finalized


def _guarded_selected_index(
    *,
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    selected_index: int,
    bucket: str,
    runtime_guard_buckets: set[str],
    runtime_guard_bucket_margins: dict[str, float] | None,
    runtime_guard_min_margin: float,
    runtime_guard_base_index: int,
) -> tuple[int, dict[str, Any]]:
    valid = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    if valid.size == 0 or str(bucket) not in runtime_guard_buckets:
        return int(selected_index), {"active": False}

    base_index = int(runtime_guard_base_index)
    if base_index < 0 or base_index >= len(candidate_mask) or not bool(candidate_mask[base_index]):
        base_index = int(valid[0])
    selected_index = int(selected_index)
    if selected_index == base_index:
        return int(selected_index), {"active": True, "nonbase": False, "fallback": False, "base_index": int(base_index)}

    margin = float(np.asarray(scores, dtype=np.float32)[selected_index] - np.asarray(scores, dtype=np.float32)[base_index])
    min_margin = float((runtime_guard_bucket_margins or {}).get(str(bucket), float(runtime_guard_min_margin)))
    fallback = bool(margin < min_margin)
    return (
        int(base_index if fallback else selected_index),
        {
            "active": True,
            "nonbase": True,
            "fallback": fallback,
            "base_index": int(base_index),
            "raw_selected_index": int(selected_index),
            "margin": margin,
            "min_margin": min_margin,
        },
    )


def _load_runtime_live_risk_selector(path_text: str | None, threshold_override: float | None, bucket_text: str | None) -> dict[str, Any] | None:
    if not path_text or not str(path_text).strip():
        return None
    artifact_path = _resolve_cli_path(str(path_text))
    if artifact_path is None:
        return None
    artifact = load_live_risk_artifact(artifact_path)
    if threshold_override is not None and math.isfinite(float(threshold_override)) and float(threshold_override) >= 0.0:
        artifact["threshold"] = float(threshold_override)
    buckets = _parse_live_risk_bucket_set(str(bucket_text or ""))
    if buckets:
        artifact["apply_buckets"] = sorted(buckets)
    return artifact


def _live_risk_selected_index(
    *,
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    selected_index: int,
    bucket: str,
    selector: dict[str, Any] | None,
    default_base_index: int,
) -> tuple[int, dict[str, Any]]:
    if selector is None:
        return int(selected_index), {"active": False}
    apply_buckets = set(str(item) for item in selector.get("apply_buckets", []))
    if apply_buckets and str(bucket) not in apply_buckets:
        return int(selected_index), {"active": False}
    valid = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    if valid.size == 0:
        return int(selected_index), {"active": False}
    base_index = int(default_base_index)
    if base_index < 0 or base_index >= len(candidate_mask) or not bool(candidate_mask[base_index]):
        base_index = int(valid[0])
    selected_index = int(selected_index)
    if selected_index == base_index:
        return int(selected_index), {"active": True, "nonbase": False, "fallback": False, "base_index": int(base_index)}
    features = live_risk_feature_vector(
        arrays=arrays,
        scores=scores,
        candidate_mask=candidate_mask,
        selected_index=selected_index,
        base_index=base_index,
        bucket=str(bucket),
        bucket_vocab=list(selector.get("bucket_vocab", [])),
    )
    risk = predict_live_risk(selector, features)
    threshold = float(selector.get("threshold", 1.0))
    fallback = bool(risk >= threshold)
    return (
        int(base_index if fallback else selected_index),
        {
            "active": True,
            "nonbase": True,
            "fallback": fallback,
            "base_index": int(base_index),
            "raw_selected_index": int(selected_index),
            "risk": float(risk),
            "threshold": threshold,
        },
    )


def _xlron_forward(model: Any, tensors: dict[str, Any], edge_index: Any) -> tuple[Any, Any]:
    return model(
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


def _evaluate_examples(
    *,
    model: Any,
    examples: list[Any],
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
            raw_logits, _value = _xlron_forward(model, tensors, edge_index)
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


def _rollout_validate(
    *,
    model: Any,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    episode_selection: str,
    episode_offset: int = 0,
    run_name_suffix: str = "",
    runtime_guard_buckets: set[str] | None = None,
    runtime_guard_bucket_margins: dict[str, float] | None = None,
    runtime_guard_min_margin: float = 0.0,
    runtime_guard_base_index: int = 0,
    runtime_live_risk_selector: dict[str, Any] | None = None,
    runtime_live_risk_base_index: int = 0,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    traffic, episode_ids = _load_rollout_traffic(config, split, max_episodes, episode_selection, int(episode_offset))
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    cfg = solver.config
    run_path = output_dir / f"rollout_validate_{split}{run_name_suffix}"
    model.eval()
    requests = 0
    accepted = 0
    no_candidate = 0
    invalid_selected = 0
    total_reward = 0.0
    selected_indices: list[int] = []
    buckets: dict[str, dict[str, Any]] = {}
    guard_buckets = set(runtime_guard_buckets or set())
    guard_bucket_margins = dict(runtime_guard_bucket_margins or {})
    guard_buckets.update(str(bucket) for bucket in guard_bucket_margins)
    guard_requests = 0
    guard_nonbase = 0
    guard_fallbacks = 0
    guard_margins: list[float] = []
    guard_by_bucket: dict[str, dict[str, int]] = {}
    risk_requests = 0
    risk_nonbase = 0
    risk_fallbacks = 0
    risk_values: list[float] = []
    risk_by_bucket: dict[str, dict[str, int]] = {}
    started = time.perf_counter()

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.iloc[:max_requests_per_episode].reset_index(drop=True)
        bucket = _bucket_label(episode)
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
        episode_accepted = 0
        episode_reward = 0.0
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
                    raw_logits, _value = _xlron_forward(model, tensors, edge_index)
                    scores = raw_logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
                try:
                    selected_index = masked_argmax(scores, batch.candidate_mask)
                except ValueError:
                    selected_index = int(valid[0])
                if int(selected_index) < 0 or not bool(batch.candidate_mask[int(selected_index)]):
                    invalid_selected += 1
                    selected_index = int(valid[0])
                selected_index, risk_details = _live_risk_selected_index(
                    arrays=arrays,
                    scores=scores,
                    candidate_mask=batch.candidate_mask,
                    selected_index=int(selected_index),
                    bucket=bucket,
                    selector=runtime_live_risk_selector,
                    default_base_index=int(runtime_live_risk_base_index),
                )
                if bool(risk_details.get("active", False)):
                    risk_requests += 1
                    stats = risk_by_bucket.setdefault(str(bucket), {"requests": 0, "nonbase": 0, "fallbacks": 0})
                    stats["requests"] = int(stats["requests"]) + 1
                    if bool(risk_details.get("nonbase", False)):
                        risk_nonbase += 1
                        stats["nonbase"] = int(stats["nonbase"]) + 1
                        risk_values.append(float(risk_details.get("risk", 0.0)))
                    if bool(risk_details.get("fallback", False)):
                        risk_fallbacks += 1
                        stats["fallbacks"] = int(stats["fallbacks"]) + 1
                selected_index, guard_details = _guarded_selected_index(
                    scores=scores,
                    candidate_mask=batch.candidate_mask,
                    selected_index=int(selected_index),
                    bucket=bucket,
                    runtime_guard_buckets=guard_buckets,
                    runtime_guard_bucket_margins=guard_bucket_margins,
                    runtime_guard_min_margin=float(runtime_guard_min_margin),
                    runtime_guard_base_index=int(runtime_guard_base_index),
                )
                if bool(guard_details.get("active", False)):
                    guard_requests += 1
                    stats = guard_by_bucket.setdefault(str(bucket), {"requests": 0, "nonbase": 0, "fallbacks": 0})
                    stats["requests"] = int(stats["requests"]) + 1
                    if bool(guard_details.get("nonbase", False)):
                        guard_nonbase += 1
                        stats["nonbase"] = int(stats["nonbase"]) + 1
                        guard_margins.append(float(guard_details.get("margin", 0.0)))
                    if bool(guard_details.get("fallback", False)):
                        guard_fallbacks += 1
                        stats["fallbacks"] = int(stats["fallbacks"]) + 1
                selected_indices.append(int(selected_index))
                action = batch.topn[int(selected_index)].action

            _observation, reward, terminated, truncated, info = env.step(int(action))
            accepted += int(bool(info.get("accepted", False)))
            total_reward += float(reward)
            requests += 1
            episode_requests += 1
            episode_accepted += int(bool(info.get("accepted", False)))
            episode_reward += float(reward)
            if bool(terminated) or bool(truncated):
                break
        _add_bucket_stats(
            buckets,
            bucket,
            requests=episode_requests,
            accepted=episode_accepted,
            total_reward=episode_reward,
            episode_done=True,
        )
    return {
        "split": str(split),
        "episode_selection": str(episode_selection),
        "episode_offset": int(episode_offset),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": float(total_reward),
        "no_candidate_requests": int(no_candidate),
        "invalid_selected": int(invalid_selected),
        "mean_selected_index": float(np.mean(selected_indices)) if selected_indices else None,
        "by_bucket": _finalize_bucket_stats(buckets),
        "runtime_guard": {
            "buckets": sorted(guard_buckets),
            "min_margin": float(runtime_guard_min_margin),
            "bucket_margins": guard_bucket_margins,
            "base_index": int(runtime_guard_base_index),
            "requests": int(guard_requests),
            "nonbase_candidates": int(guard_nonbase),
            "fallbacks": int(guard_fallbacks),
            "fallback_rate": float(guard_fallbacks / max(guard_nonbase, 1)),
            "margin_quantiles": [float(np.quantile(np.asarray(guard_margins, dtype=np.float32), q)) for q in (0.0, 0.5, 0.9, 1.0)]
            if guard_margins
            else [],
            "by_bucket": guard_by_bucket,
        },
        "live_risk_selector": {
            "enabled": runtime_live_risk_selector is not None,
            "artifact_path": None if runtime_live_risk_selector is None else str(runtime_live_risk_selector.get("artifact_path", "")),
            "threshold": None if runtime_live_risk_selector is None else float(runtime_live_risk_selector.get("threshold", 0.0)),
            "apply_buckets": [] if runtime_live_risk_selector is None else sorted(str(item) for item in runtime_live_risk_selector.get("apply_buckets", [])),
            "base_index": int(runtime_live_risk_base_index),
            "requests": int(risk_requests),
            "nonbase_candidates": int(risk_nonbase),
            "fallbacks": int(risk_fallbacks),
            "fallback_rate": float(risk_fallbacks / max(risk_nonbase, 1)),
            "risk_quantiles": [float(np.quantile(np.asarray(risk_values, dtype=np.float32), q)) for q in (0.0, 0.5, 0.9, 1.0)]
            if risk_values
            else [],
            "by_bucket": risk_by_bucket,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }


def _rollout_validate_reference_policy(
    *,
    policy: str,
    config: ExperimentConfig,
    output_dir: Path,
    split: str,
    max_episodes: int,
    max_requests_per_episode: int,
    episode_selection: str,
    episode_offset: int = 0,
    run_name_suffix: str = "",
) -> dict[str, Any]:
    traffic, episode_ids = _load_rollout_traffic(config, split, max_episodes, episode_selection, int(episode_offset))
    solver = GnnCnnDqnOngSolver(_solver_config(config, neural=False))
    rng = np.random.default_rng(int(config.seed))
    run_path = output_dir / f"rollout_validate_{split}_reference{run_name_suffix}"
    requests = 0
    accepted = 0
    no_candidate = 0
    total_reward = 0.0
    selected_indices: list[int] = []
    buckets: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == str(episode_id)].sort_values("request_id").reset_index(drop=True)
        if max_requests_per_episode > 0:
            episode = episode.iloc[:max_requests_per_episode].reset_index(drop=True)
        bucket = _bucket_label(episode)
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
        episode_accepted = 0
        episode_reward = 0.0
        while True:
            action, _candidate, valid_count, selected_index, _override_applied, _override_probability = _select_candidate(
                policy=str(policy),
                solver=solver,
                env=env,
                rng=rng,
                config=config,
            )
            if int(valid_count) <= 0:
                no_candidate += 1
            else:
                selected_indices.append(int(selected_index))
            _observation, reward, terminated, truncated, info = env.step(int(action))
            accepted += int(bool(info.get("accepted", False)))
            total_reward += float(reward)
            requests += 1
            episode_requests += 1
            episode_accepted += int(bool(info.get("accepted", False)))
            episode_reward += float(reward)
            if bool(terminated) or bool(truncated):
                break
        _add_bucket_stats(
            buckets,
            bucket,
            requests=episode_requests,
            accepted=episode_accepted,
            total_reward=episode_reward,
            episode_done=True,
        )
    return {
        "policy": str(policy),
        "split": str(split),
        "episode_selection": str(episode_selection),
        "episode_offset": int(episode_offset),
        "episodes": int(len(episode_ids)),
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": float(total_reward),
        "no_candidate_requests": int(no_candidate),
        "mean_selected_index": float(np.mean(selected_indices)) if selected_indices else None,
        "by_bucket": _finalize_bucket_stats(buckets),
        "elapsed_sec": float(time.perf_counter() - started),
    }


def _aggregate_rollout_evals(evals: list[dict[str, Any]], *, label: str = "") -> dict[str, Any]:
    if not evals:
        return {}
    requests = int(sum(int(row.get("requests", 0)) for row in evals))
    accepted = int(sum(int(row.get("accepted", 0)) for row in evals))
    no_candidate = int(sum(int(row.get("no_candidate_requests", 0)) for row in evals))
    invalid = int(sum(int(row.get("invalid_selected", 0)) for row in evals))
    total_reward = float(sum(float(row.get("total_reward", 0.0)) for row in evals))
    guard_requests = int(sum(int((row.get("runtime_guard") or {}).get("requests", 0)) for row in evals))
    guard_nonbase = int(sum(int((row.get("runtime_guard") or {}).get("nonbase_candidates", 0)) for row in evals))
    guard_fallbacks = int(sum(int((row.get("runtime_guard") or {}).get("fallbacks", 0)) for row in evals))
    risk_requests = int(sum(int((row.get("live_risk_selector") or {}).get("requests", 0)) for row in evals))
    risk_nonbase = int(sum(int((row.get("live_risk_selector") or {}).get("nonbase_candidates", 0)) for row in evals))
    risk_fallbacks = int(sum(int((row.get("live_risk_selector") or {}).get("fallbacks", 0)) for row in evals))
    guard_margins: list[float] = []
    guard_by_bucket: dict[str, dict[str, int]] = {}
    guard_bucket_set: set[str] = set()
    guard_bucket_margin_map: dict[str, float] = {}
    risk_values: list[float] = []
    risk_by_bucket: dict[str, dict[str, int]] = {}
    risk_apply_buckets: set[str] = set()
    risk_enabled = False
    risk_artifact_path = ""
    risk_threshold: float | None = None
    for row in evals:
        guard = row.get("runtime_guard") or {}
        guard_bucket_set.update(str(item) for item in guard.get("buckets", []))
        for bucket, value in (guard.get("bucket_margins") or {}).items():
            guard_bucket_margin_map[str(bucket)] = float(value)
        guard_margins.extend(float(value) for value in guard.get("margin_quantiles", []) if value is not None)
        for bucket, stats in (guard.get("by_bucket") or {}).items():
            target = guard_by_bucket.setdefault(str(bucket), {"requests": 0, "nonbase": 0, "fallbacks": 0})
            target["requests"] = int(target["requests"]) + int(stats.get("requests", 0))
            target["nonbase"] = int(target["nonbase"]) + int(stats.get("nonbase", 0))
            target["fallbacks"] = int(target["fallbacks"]) + int(stats.get("fallbacks", 0))
        risk = row.get("live_risk_selector") or {}
        risk_enabled = bool(risk_enabled or risk.get("enabled", False))
        if risk.get("artifact_path"):
            risk_artifact_path = str(risk.get("artifact_path"))
        if risk.get("threshold") is not None:
            risk_threshold = float(risk.get("threshold"))
        risk_apply_buckets.update(str(item) for item in risk.get("apply_buckets", []))
        risk_values.extend(float(value) for value in risk.get("risk_quantiles", []) if value is not None)
        for bucket, stats in (risk.get("by_bucket") or {}).items():
            target = risk_by_bucket.setdefault(str(bucket), {"requests": 0, "nonbase": 0, "fallbacks": 0})
            target["requests"] = int(target["requests"]) + int(stats.get("requests", 0))
            target["nonbase"] = int(target["nonbase"]) + int(stats.get("nonbase", 0))
            target["fallbacks"] = int(target["fallbacks"]) + int(stats.get("fallbacks", 0))
    selected_indices = [
        float(row["mean_selected_index"])
        for row in evals
        if row.get("mean_selected_index") is not None
    ]
    buckets: dict[str, dict[str, Any]] = {}
    for row in evals:
        for bucket, stats in (row.get("by_bucket") or {}).items():
            _add_bucket_stats(
                buckets,
                str(bucket),
                requests=int(stats.get("requests", 0)),
                accepted=int(stats.get("accepted", 0)),
                total_reward=float(stats.get("total_reward", 0.0)),
                episode_done=False,
            )
            buckets[str(bucket)]["episodes"] = int(buckets[str(bucket)]["episodes"]) + int(stats.get("episodes", 0))
    return {
        "label": str(label),
        "slices": int(len(evals)),
        "slice_offsets": [int(row.get("episode_offset", 0)) for row in evals],
        "split": str(evals[0].get("split", "")),
        "episode_selection": str(evals[0].get("episode_selection", "")),
        "episodes": int(sum(int(row.get("episodes", 0)) for row in evals)),
        "requests": requests,
        "accepted": accepted,
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": total_reward,
        "no_candidate_requests": no_candidate,
        "invalid_selected": invalid,
        "mean_selected_index": float(np.mean(selected_indices)) if selected_indices else None,
        "by_bucket": _finalize_bucket_stats(buckets),
        "runtime_guard": {
            "buckets": sorted(guard_bucket_set),
            "bucket_margins": guard_bucket_margin_map,
            "requests": int(guard_requests),
            "nonbase_candidates": int(guard_nonbase),
            "fallbacks": int(guard_fallbacks),
            "fallback_rate": float(guard_fallbacks / max(guard_nonbase, 1)),
            "margin_quantiles": [float(np.quantile(np.asarray(guard_margins, dtype=np.float32), q)) for q in (0.0, 0.5, 0.9, 1.0)]
            if guard_margins
            else [],
            "by_bucket": guard_by_bucket,
        },
        "live_risk_selector": {
            "enabled": bool(risk_enabled),
            "artifact_path": risk_artifact_path,
            "threshold": risk_threshold,
            "apply_buckets": sorted(risk_apply_buckets),
            "requests": int(risk_requests),
            "nonbase_candidates": int(risk_nonbase),
            "fallbacks": int(risk_fallbacks),
            "fallback_rate": float(risk_fallbacks / max(risk_nonbase, 1)),
            "risk_quantiles": [float(np.quantile(np.asarray(risk_values, dtype=np.float32), q)) for q in (0.0, 0.5, 0.9, 1.0)]
            if risk_values
            else [],
            "by_bucket": risk_by_bucket,
        },
        "elapsed_sec": float(sum(float(row.get("elapsed_sec", 0.0)) for row in evals)),
        "slice_evals": evals,
    }


def _rollout_validate_for_selection(
    *,
    model: Any,
    config: ExperimentConfig,
    output_dir: Path,
    args: argparse.Namespace,
    device: str,
    torch: Any,
) -> dict[str, Any]:
    evals: list[dict[str, Any]] = []
    slices = max(1, int(args.rollout_val_slices))
    stride = max(1, int(args.rollout_val_slice_stride))
    runtime_guard_buckets = _parse_bucket_set(str(getattr(args, "rollout_runtime_guard_buckets", "")))
    runtime_guard_bucket_margins = _parse_bucket_float_map(
        str(getattr(args, "rollout_runtime_guard_bucket_margins", ""))
    )
    runtime_guard_buckets.update(runtime_guard_bucket_margins.keys())
    risk_threshold = float(getattr(args, "rollout_live_risk_selector_threshold", -1.0))
    runtime_live_risk_selector = _load_runtime_live_risk_selector(
        str(getattr(args, "rollout_live_risk_selector_path", "")),
        risk_threshold if risk_threshold >= 0.0 else None,
        str(getattr(args, "rollout_live_risk_selector_buckets", "")),
    )
    for slice_index in range(slices):
        offset = int(args.rollout_val_episode_offset) + slice_index * stride
        suffix = "" if slices == 1 and offset == 0 else f"_slice{slice_index}_offset{offset}"
        evals.append(
            _rollout_validate(
                model=model,
                config=config,
                output_dir=output_dir,
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
                episode_offset=offset,
                run_name_suffix=suffix,
                runtime_guard_buckets=runtime_guard_buckets,
                runtime_guard_bucket_margins=runtime_guard_bucket_margins,
                runtime_guard_min_margin=float(getattr(args, "rollout_runtime_guard_min_margin", 0.0)),
                runtime_guard_base_index=int(getattr(args, "rollout_runtime_guard_base_index", 0)),
                runtime_live_risk_selector=runtime_live_risk_selector,
                runtime_live_risk_base_index=int(getattr(args, "rollout_live_risk_selector_base_index", 0)),
                device=device,
                torch=torch,
            )
        )
    if len(evals) == 1:
        return evals[0]
    return _aggregate_rollout_evals(evals, label="student_multi_slice")


def _rollout_reference_for_selection(
    *,
    config: ExperimentConfig,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    policy = str(args.rollout_val_reference_policy).strip()
    if not policy:
        return None
    evals: list[dict[str, Any]] = []
    slices = max(1, int(args.rollout_val_slices))
    stride = max(1, int(args.rollout_val_slice_stride))
    for slice_index in range(slices):
        offset = int(args.rollout_val_episode_offset) + slice_index * stride
        suffix = "" if slices == 1 and offset == 0 else f"_slice{slice_index}_offset{offset}"
        evals.append(
            _rollout_validate_reference_policy(
                policy=policy,
                config=config,
                output_dir=output_dir,
                split=str(args.rollout_val_split),
                max_episodes=int(args.rollout_val_max_episodes),
                max_requests_per_episode=int(args.rollout_val_max_requests_per_episode),
                episode_selection=str(args.rollout_val_episode_selection or args.episode_selection),
                episode_offset=offset,
                run_name_suffix=suffix,
            )
        )
    if len(evals) == 1:
        return evals[0]
    return _aggregate_rollout_evals(evals, label=f"reference_{policy}_multi_slice")


def _rollout_selection_metric(
    rollout_eval: dict[str, Any] | None,
    *,
    reference_eval: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    if rollout_eval is None:
        return 0.0, {"metric": str(args.checkpoint_selection), "score": 0.0}
    mode = str(args.checkpoint_selection)
    if mode == "rollout_accepted":
        score = float(rollout_eval.get("accepted") or 0.0)
        return score, {"metric": mode, "score": score}
    if mode == "rollout_reward":
        score = float(rollout_eval.get("total_reward") or 0.0)
        return score, {"metric": mode, "score": score}
    if mode not in {"rollout_worst_bucket_score", "rollout_bucket_guard_score"}:
        raise ValueError(f"Unsupported checkpoint_selection: {args.checkpoint_selection}")

    accepted_score = float(rollout_eval.get("accepted") or 0.0)
    by_bucket = rollout_eval.get("by_bucket") or {}
    reference_by_bucket = (reference_eval or {}).get("by_bucket") or {}
    bucket_deltas: dict[str, int] = {}
    for bucket, ref_stats in reference_by_bucket.items():
        student_stats = by_bucket.get(bucket) or {}
        bucket_deltas[str(bucket)] = int(student_stats.get("accepted", 0)) - int(ref_stats.get("accepted", 0))
    for bucket, student_stats in by_bucket.items():
        if bucket not in bucket_deltas:
            bucket_deltas[str(bucket)] = int(student_stats.get("accepted", 0))
    worst_bucket_delta = min(bucket_deltas.values(), default=0)
    negative_bucket_sum = int(sum(delta for delta in bucket_deltas.values() if delta < 0))
    protected_buckets = sorted(_parse_bucket_set(str(getattr(args, "rollout_protected_buckets", ""))))
    protected_min_delta = int(getattr(args, "rollout_protected_bucket_min_delta", 0))
    protected_bucket_shortfalls: dict[str, int] = {}
    for bucket in protected_buckets:
        delta = int(bucket_deltas.get(bucket, 0))
        shortfall = min(0, delta - protected_min_delta)
        if shortfall < 0:
            protected_bucket_shortfalls[bucket] = int(shortfall)
    protected_bucket_penalty_sum = int(sum(protected_bucket_shortfalls.values()))
    score = (
        accepted_score
        + float(args.rollout_worst_bucket_penalty) * float(min(0, worst_bucket_delta))
        + float(args.rollout_negative_bucket_penalty) * float(negative_bucket_sum)
    )
    if mode == "rollout_bucket_guard_score":
        score += float(args.rollout_protected_bucket_penalty) * float(protected_bucket_penalty_sum)
    details = {
        "metric": mode,
        "score": float(score),
        "accepted_score": accepted_score,
        "reference_policy": str(args.rollout_val_reference_policy),
        "worst_bucket_delta": int(worst_bucket_delta),
        "negative_bucket_sum": int(negative_bucket_sum),
        "worst_bucket_penalty": float(args.rollout_worst_bucket_penalty),
        "negative_bucket_penalty": float(args.rollout_negative_bucket_penalty),
        "protected_buckets": protected_buckets,
        "protected_bucket_min_delta": int(protected_min_delta),
        "protected_bucket_shortfalls": protected_bucket_shortfalls,
        "protected_bucket_penalty_sum": int(protected_bucket_penalty_sum),
        "protected_bucket_penalty": float(args.rollout_protected_bucket_penalty),
        "bucket_deltas": bucket_deltas,
    }
    return float(score), details


def _save_checkpoint(
    *,
    path: Path,
    model: Any,
    config: ExperimentConfig,
    cfg: Any,
    model_shapes: dict[str, int],
    args: argparse.Namespace,
    step: int,
    epoch: int,
    metrics: dict[str, Any],
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    architecture_kwargs = _xlron_architecture_kwargs(args, model_shapes)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy": "top32_xlron_stabilized_ppo",
            "n_max": int(cfg.n_max),
            "action_feature_dim": int(model_shapes["action_feature_dim"]),
            "link_feature_dim": int(model_shapes["link_feature_dim"]),
            "global_feature_dim": int(model_shapes["global_feature_dim"]),
            "request_feature_dim": int(model_shapes["request_feature_dim"]),
            "embedding_dim": int(args.transformer_embedding_size),
            "hidden_dim": int(args.transformer_embedding_size),
            "transformer_num_layers": int(args.transformer_num_layers),
            "transformer_num_heads": int(args.transformer_num_heads),
            "dropout": float(args.dropout),
            "position_dim": int(args.transformer_position_dim),
            **architecture_kwargs,
            "step": int(step),
            "epoch": int(epoch),
            "config": {
                **dict(config.resolved),
                "teacher_dqn_checkpoint": str(args.teacher_dqn_checkpoint),
                "initial_xlron_checkpoint": str(args.initial_xlron_checkpoint),
                "behavior_policy": str(args.behavior_policy),
                "behavior_xlron_checkpoint": str(args.behavior_xlron_checkpoint),
                "checkpoint_selection": str(args.checkpoint_selection),
                "training_mode": "top32_xlron_full_dqn_distill",
            },
            "solver_config": asdict(cfg),
            "metrics": metrics,
            "training_mode": "top32_xlron_full_dqn_distill",
        },
        path,
    )


def _train(
    *,
    config: ExperimentConfig,
    train_examples: list[Any],
    val_examples: list[Any],
    edge_index_np: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from cse2026.ong_solver.models import XlronGraphTransformerPpoNetwork, require_torch

    torch = require_torch()
    device = _device(config, torch)
    torch.manual_seed(int(config.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))

    model_shapes = _model_shapes_from_examples(train_examples)
    cfg = _solver_config(config, neural=False)
    model = XlronGraphTransformerPpoNetwork(
        action_feature_dim=int(model_shapes["action_feature_dim"]),
        link_feature_dim=int(model_shapes["link_feature_dim"]),
        global_feature_dim=int(model_shapes["global_feature_dim"]),
        request_feature_dim=int(model_shapes["request_feature_dim"]),
        embedding_dim=int(args.transformer_embedding_size),
        num_layers=int(args.transformer_num_layers),
        num_heads=int(args.transformer_num_heads),
        dropout=float(args.dropout),
        position_dim=int(args.transformer_position_dim),
        **_xlron_architecture_kwargs(args, model_shapes),
    ).to(device)
    initial_xlron_checkpoint = _resolve_cli_path(str(args.initial_xlron_checkpoint or ""))
    initial_checkpoint_loaded = False
    if initial_xlron_checkpoint is not None:
        checkpoint = torch.load(initial_xlron_checkpoint, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("--initial-xlron-checkpoint must point to a dictionary checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        initial_checkpoint_loaded = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    edge_index = torch.as_tensor(edge_index_np, dtype=torch.long, device=device)
    rng = np.random.default_rng(int(config.seed))
    best_metric = -1.0
    best_epoch = 0
    best_rollout_val_eval: dict[str, Any] | None = None
    initial_rollout_val_eval: dict[str, Any] | None = None
    reference_rollout_val_eval: dict[str, Any] | None = None
    best_selection_metric_details: dict[str, Any] | None = None
    best_path = output_dir / "top32_xlron_full_dqn_distill_best.pt"
    history: list[dict[str, Any]] = []
    global_step = 0

    if str(args.checkpoint_selection) != "teacher_top1":
        if str(args.checkpoint_selection) in {"rollout_worst_bucket_score", "rollout_bucket_guard_score"}:
            reference_rollout_val_eval = _rollout_reference_for_selection(
                config=config,
                output_dir=output_dir,
                args=args,
            )
            print(
                json.dumps(
                    {"phase": "reference_rollout_val_eval", "reference_rollout_val_eval": _json_safe(reference_rollout_val_eval)},
                    sort_keys=True,
                ),
                flush=True,
            )
        initial_rollout_val_eval = _rollout_validate_for_selection(
            model=model,
            config=config,
            output_dir=output_dir,
            args=args,
            device=device,
            torch=torch,
        )
        best_metric, best_selection_metric_details = _rollout_selection_metric(
            initial_rollout_val_eval,
            reference_eval=reference_rollout_val_eval,
            args=args,
        )
        best_rollout_val_eval = initial_rollout_val_eval
        _save_checkpoint(
            path=best_path,
            model=model,
            config=config,
            cfg=cfg,
            model_shapes=model_shapes,
            args=args,
            step=0,
            epoch=0,
            metrics={
                "initial_rollout_val_eval": initial_rollout_val_eval,
                "reference_rollout_val_eval": reference_rollout_val_eval,
                "selection_metric": best_selection_metric_details,
            },
            torch=torch,
        )
        print(
            json.dumps(
                {
                    "phase": "initial_checkpoint_candidate",
                    "checkpoint_selection": str(args.checkpoint_selection),
                    "selection_metric": float(best_metric),
                    "selection_metric_details": best_selection_metric_details,
                    "reference_rollout_val_eval": reference_rollout_val_eval,
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
            raw_logits, _value = _xlron_forward(model, tensors, edge_index)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
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
            rollout_val_eval = _rollout_validate_for_selection(
                model=model,
                config=config,
                output_dir=output_dir,
                args=args,
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
        if str(args.checkpoint_selection) == "teacher_top1":
            metric = float(val_eval.get("teacher_top1_accuracy") or 0.0)
            selection_metric_details = {"metric": "teacher_top1", "score": float(metric)}
        elif str(args.checkpoint_selection) == "rollout_accepted":
            metric, selection_metric_details = _rollout_selection_metric(
                rollout_val_eval,
                reference_eval=reference_rollout_val_eval,
                args=args,
            )
        elif str(args.checkpoint_selection) == "rollout_reward":
            metric, selection_metric_details = _rollout_selection_metric(
                rollout_val_eval,
                reference_eval=reference_rollout_val_eval,
                args=args,
            )
        elif str(args.checkpoint_selection) in {"rollout_worst_bucket_score", "rollout_bucket_guard_score"}:
            metric, selection_metric_details = _rollout_selection_metric(
                rollout_val_eval,
                reference_eval=reference_rollout_val_eval,
                args=args,
            )
        else:
            raise ValueError(f"Unsupported checkpoint_selection: {args.checkpoint_selection}")
        row["selection_metric"] = float(metric)
        row["selection_metric_details"] = selection_metric_details
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if metric >= best_metric:
            best_metric = metric
            best_epoch = int(epoch)
            best_rollout_val_eval = rollout_val_eval
            best_selection_metric_details = selection_metric_details
            _save_checkpoint(
                path=best_path,
                model=model,
                config=config,
                cfg=cfg,
                model_shapes=model_shapes,
                args=args,
                step=global_step,
                epoch=epoch,
                metrics={
                    "train_eval": train_eval,
                    "val_eval": val_eval,
                    "rollout_val_eval": rollout_val_eval,
                    "reference_rollout_val_eval": reference_rollout_val_eval,
                    "selection_metric": selection_metric_details,
                },
                torch=torch,
            )

    summary = {
        "stage": "train_top32_xlron_full_dqn_distill",
        "checkpoint_path": str(best_path),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "best_selection_metric_details": best_selection_metric_details,
        "best_rollout_val_eval": best_rollout_val_eval,
        "initial_rollout_val_eval": initial_rollout_val_eval,
        "reference_rollout_val_eval": reference_rollout_val_eval,
        "checkpoint_selection": str(args.checkpoint_selection),
        "device": device,
        "model_shapes": dict(model_shapes),
        "architecture": _xlron_architecture_kwargs(args, model_shapes),
        "initial_xlron_checkpoint": None if initial_xlron_checkpoint is None else str(initial_xlron_checkpoint),
        "initial_checkpoint_loaded": bool(initial_checkpoint_loaded),
        "behavior_policy": str(args.behavior_policy),
        "behavior_xlron_checkpoint": str(args.behavior_xlron_checkpoint),
        "preserve_teacher_max_episodes": int(args.preserve_teacher_max_episodes),
        "preserve_weight": float(args.preserve_weight),
        "hard_dagger_loss_buckets": str(args.hard_dagger_loss_buckets),
        "hard_dagger_disagreement_weight": float(args.hard_dagger_disagreement_weight),
        "hard_dagger_loss_bucket_weight": float(args.hard_dagger_loss_bucket_weight),
        "hard_dagger_agreement_weight": float(args.hard_dagger_agreement_weight),
        "hard_dagger_agreement_keep_frac": float(args.hard_dagger_agreement_keep_frac),
        "hard_dagger_max_weight": float(args.hard_dagger_max_weight),
        "counterfactual_aux_dir": str(args.counterfactual_aux_dir),
        "counterfactual_aux_weight": float(args.counterfactual_aux_weight),
        "counterfactual_aux_win_weight": float(args.counterfactual_aux_win_weight),
        "counterfactual_aux_loss_weight": float(args.counterfactual_aux_loss_weight),
        "counterfactual_aux_tie_weight": float(args.counterfactual_aux_tie_weight),
        "counterfactual_aux_score_boost": float(args.counterfactual_aux_score_boost),
        "counterfactual_aux_target_scale": float(args.counterfactual_aux_target_scale),
        "counterfactual_aux_score_clip": float(args.counterfactual_aux_score_clip),
        "counterfactual_aux_magnitude_weight": float(args.counterfactual_aux_magnitude_weight),
        "counterfactual_aux_magnitude_cap": float(args.counterfactual_aux_magnitude_cap),
        "counterfactual_aux_max_examples": int(args.counterfactual_aux_max_examples),
        "counterfactual_aux_include_tie_only": bool(args.counterfactual_aux_include_tie_only),
        "counterfactual_aux_mode": str(args.counterfactual_aux_mode),
        "train_examples": int(len(train_examples)),
        "val_examples": int(len(val_examples)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "ce_weight": float(args.ce_weight),
        "listwise_kl_weight": float(args.listwise_kl_weight),
        "listwise_temperature": float(args.listwise_temperature),
        "pairwise_rank_weight": float(args.pairwise_rank_weight),
        "pairwise_temperature": float(args.pairwise_temperature),
        "pairwise_teacher_gap": float(args.pairwise_teacher_gap),
        "teacher_loss_filter": str(args.teacher_loss_filter),
        "min_teacher_margin": float(args.min_teacher_margin),
        "hard_case_weight_rules": str(args.hard_case_weight_rules),
        "rollout_val_slices": int(args.rollout_val_slices),
        "rollout_val_slice_stride": int(args.rollout_val_slice_stride),
        "rollout_val_episode_offset": int(args.rollout_val_episode_offset),
        "rollout_val_reference_policy": str(args.rollout_val_reference_policy),
        "rollout_worst_bucket_penalty": float(args.rollout_worst_bucket_penalty),
        "rollout_negative_bucket_penalty": float(args.rollout_negative_bucket_penalty),
        "rollout_protected_buckets": str(args.rollout_protected_buckets),
        "rollout_protected_bucket_min_delta": int(args.rollout_protected_bucket_min_delta),
        "rollout_protected_bucket_penalty": float(args.rollout_protected_bucket_penalty),
        "teacher_dqn_checkpoint": str(args.teacher_dqn_checkpoint),
        "history": history,
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill full-DQN stratified32_e5 into full Top32 XLRON.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-dqn-checkpoint", required=True)
    parser.add_argument("--initial-xlron-checkpoint", default="")
    parser.add_argument("--behavior-policy", choices=("teacher", "student_xlron"), default="teacher")
    parser.add_argument("--behavior-xlron-checkpoint", default="")
    parser.add_argument("--preserve-teacher-max-episodes", type=int, default=0)
    parser.add_argument("--preserve-weight", type=float, default=0.5)
    parser.add_argument("--hard-dagger-loss-buckets", default="")
    parser.add_argument("--hard-dagger-disagreement-weight", type=float, default=1.0)
    parser.add_argument("--hard-dagger-loss-bucket-weight", type=float, default=1.0)
    parser.add_argument("--hard-dagger-agreement-weight", type=float, default=1.0)
    parser.add_argument("--hard-dagger-agreement-keep-frac", type=float, default=1.0)
    parser.add_argument("--hard-dagger-max-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-aux-dir", default="")
    parser.add_argument("--counterfactual-aux-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-aux-win-weight", type=float, default=2.0)
    parser.add_argument("--counterfactual-aux-loss-weight", type=float, default=1.25)
    parser.add_argument("--counterfactual-aux-tie-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-aux-score-boost", type=float, default=2.0)
    parser.add_argument("--counterfactual-aux-target-scale", type=float, default=4.0)
    parser.add_argument("--counterfactual-aux-score-clip", type=float, default=4.0)
    parser.add_argument("--counterfactual-aux-magnitude-weight", type=float, default=0.15)
    parser.add_argument("--counterfactual-aux-magnitude-cap", type=float, default=4.0)
    parser.add_argument("--counterfactual-aux-max-examples", type=int, default=0)
    parser.add_argument("--counterfactual-aux-include-tie-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--counterfactual-aux-mode",
        choices=("hard_masked", "soft_blend"),
        default="hard_masked",
        help="hard_masked trains only on labeled CF candidates; soft_blend keeps TopN mask and adds CF delta to DQN scores.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--train-max-episodes", type=int, default=32)
    parser.add_argument("--val-max-episodes", type=int, default=8)
    parser.add_argument("--episode-selection", choices=("first", "stratified"), default="stratified")
    parser.add_argument("--max-requests-per-episode", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--listwise-kl-weight", type=float, default=1.0)
    parser.add_argument("--listwise-temperature", type=float, default=3.0)
    parser.add_argument("--pairwise-rank-weight", type=float, default=0.2)
    parser.add_argument("--pairwise-temperature", type=float, default=1.0)
    parser.add_argument("--pairwise-teacher-gap", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument(
        "--teacher-loss-filter",
        choices=("all", "disagreement", "high_confidence_disagreement"),
        default="all",
    )
    parser.add_argument("--min-teacher-margin", type=float, default=0.0)
    parser.add_argument("--hard-case-weight-rules", default="")
    parser.add_argument("--progress-every-batches", type=int, default=25)
    parser.add_argument("--print-every-episodes", type=int, default=4)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("teacher_top1", "rollout_accepted", "rollout_reward", "rollout_worst_bucket_score", "rollout_bucket_guard_score"),
        default="rollout_accepted",
    )
    parser.add_argument("--rollout-val-split", default="val")
    parser.add_argument("--rollout-val-max-episodes", type=int, default=8)
    parser.add_argument("--rollout-val-max-requests-per-episode", type=int, default=0)
    parser.add_argument("--rollout-val-episode-selection", default="")
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
    parser.add_argument("--transformer-embedding-size", type=int, default=128)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--transformer-num-heads", type=int, default=8)
    parser.add_argument("--transformer-position-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--xlron-architecture", choices=("link_transformer", "full"), default="full")
    parser.add_argument("--xlron-enable-spectrum-branch", default="true")
    parser.add_argument("--xlron-enable-base-relative-branch", default="true")
    parser.add_argument("--xlron-enable-candidate-attention", default="true")
    parser.add_argument("--xlron-candidate-transformer-layers", type=int, default=2)
    parser.add_argument("--xlron-candidate-transformer-heads", type=int, default=4)
    parser.add_argument("--xlron-enable-auxiliary-heads", default="false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config, root=ROOT)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_dqn_checkpoint = _resolve_cli_path(str(args.teacher_dqn_checkpoint))
    if teacher_dqn_checkpoint is None:
        raise ValueError("--teacher-dqn-checkpoint is required")
    initial_xlron_checkpoint = _resolve_cli_path(str(args.initial_xlron_checkpoint or ""))
    behavior_xlron_checkpoint = _resolve_cli_path(str(args.behavior_xlron_checkpoint or "")) or initial_xlron_checkpoint
    counterfactual_aux_dir = _resolve_cli_path(str(args.counterfactual_aux_dir or ""))
    if initial_xlron_checkpoint is not None:
        args.initial_xlron_checkpoint = str(initial_xlron_checkpoint)
    if behavior_xlron_checkpoint is not None:
        args.behavior_xlron_checkpoint = str(behavior_xlron_checkpoint)
    if counterfactual_aux_dir is not None:
        args.counterfactual_aux_dir = str(counterfactual_aux_dir)

    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    ong_source = _add_ong_source_path(config)
    hard_case_weight_rules = _parse_weight_rules(str(args.hard_case_weight_rules))
    dqn_teacher_model, _pretrained = _build_dqn_model(config, device, torch)
    _load_full_dqn_checkpoint(dqn_teacher_model, teacher_dqn_checkpoint, device=device, torch=torch)
    dqn_teacher_model.eval()
    for parameter in dqn_teacher_model.parameters():
        parameter.requires_grad_(False)
    behavior_xlron_model = None
    if str(args.behavior_policy) == "student_xlron":
        if behavior_xlron_checkpoint is None:
            raise ValueError("student_xlron behavior requires --behavior-xlron-checkpoint or --initial-xlron-checkpoint")
        behavior_xlron_model, _behavior_checkpoint = _load_xlron_checkpoint_model(
            behavior_xlron_checkpoint,
            device=device,
            torch=torch,
        )
        behavior_xlron_model.eval()
        for parameter in behavior_xlron_model.parameters():
            parameter.requires_grad_(False)

    print(
        json.dumps(
            {
                "phase": "start",
                "dataset_path": str(config.dataset_path),
                "output_dir": str(output_dir),
                "ong_source_path": ong_source,
                "teacher_dqn_checkpoint": str(teacher_dqn_checkpoint),
                "initial_xlron_checkpoint": None if initial_xlron_checkpoint is None else str(initial_xlron_checkpoint),
                "behavior_policy": str(args.behavior_policy),
                "behavior_xlron_checkpoint": None if behavior_xlron_checkpoint is None else str(behavior_xlron_checkpoint),
                "checkpoint_selection": str(args.checkpoint_selection),
                "train_max_episodes": int(args.train_max_episodes),
                "val_max_episodes": int(args.val_max_episodes),
                "episode_selection": str(args.episode_selection),
                "preserve_teacher_max_episodes": int(args.preserve_teacher_max_episodes),
                "preserve_weight": float(args.preserve_weight),
                "hard_dagger_loss_buckets": str(args.hard_dagger_loss_buckets),
                "hard_dagger_disagreement_weight": float(args.hard_dagger_disagreement_weight),
                "hard_dagger_loss_bucket_weight": float(args.hard_dagger_loss_bucket_weight),
                "hard_dagger_agreement_weight": float(args.hard_dagger_agreement_weight),
                "hard_dagger_agreement_keep_frac": float(args.hard_dagger_agreement_keep_frac),
                "hard_dagger_max_weight": float(args.hard_dagger_max_weight),
                "hard_case_weight_rules": {
                    f"{scenario}:{load}": weight for (scenario, load), weight in hard_case_weight_rules.items()
                },
                "counterfactual_aux_dir": None if counterfactual_aux_dir is None else str(counterfactual_aux_dir),
                "counterfactual_aux_mode": str(args.counterfactual_aux_mode),
                "counterfactual_aux_weight": float(args.counterfactual_aux_weight),
                "counterfactual_aux_score_boost": float(args.counterfactual_aux_score_boost),
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
        teacher_kind="full_dqn",
        tree_ranker=None,
        dqn_teacher_model=dqn_teacher_model,
        teacher_base_policy="energy-aware-ksp-bm-ff",
        behavior_policy=str(args.behavior_policy),
        behavior_a3c_model=None,
        behavior_xlron_model=behavior_xlron_model,
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=0.0,
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
        teacher_kind="full_dqn",
        tree_ranker=None,
        dqn_teacher_model=dqn_teacher_model,
        teacher_base_policy="energy-aware-ksp-bm-ff",
        behavior_policy=str(args.behavior_policy),
        behavior_a3c_model=None,
        behavior_xlron_model=behavior_xlron_model,
        episode_selection=str(args.episode_selection),
        hard_case_weight_rules=hard_case_weight_rules,
        teacher_score_selected_boost=0.0,
        print_every_episodes=int(args.print_every_episodes),
        device=device,
        torch=torch,
    )
    hard_dagger_summary = None
    train_examples, hard_dagger_summary = _hard_dagger_transform_examples(
        train_examples,
        args=args,
        rng=np.random.default_rng(int(config.seed) + 9173),
    )
    print(json.dumps({"phase": "hard_dagger_summary", **_json_safe(hard_dagger_summary)}, sort_keys=True), flush=True)

    preserve_collect = None
    preserve_examples: list[Any] = []
    if int(args.preserve_teacher_max_episodes) > 0:
        preserve_examples, _preserve_edge_index, preserve_collect = _collect_examples(
            config=config,
            output_dir=output_dir,
            split=str(args.train_split),
            max_episodes=int(args.preserve_teacher_max_episodes),
            max_requests_per_episode=int(args.max_requests_per_episode),
            teacher_kind="full_dqn",
            tree_ranker=None,
            dqn_teacher_model=dqn_teacher_model,
            teacher_base_policy="energy-aware-ksp-bm-ff",
            behavior_policy="teacher",
            behavior_a3c_model=None,
            behavior_xlron_model=None,
            episode_selection=str(args.episode_selection),
            hard_case_weight_rules=hard_case_weight_rules,
            teacher_score_selected_boost=0.0,
            print_every_episodes=int(args.print_every_episodes),
            device=device,
            torch=torch,
        )
        preserve_examples = _scale_example_weights(preserve_examples, float(args.preserve_weight))
        train_examples = list(train_examples) + preserve_examples

    counterfactual_aux_summary = None
    counterfactual_aux_examples: list[DistillExample] = []
    if counterfactual_aux_dir is not None:
        counterfactual_aux_examples, counterfactual_aux_summary = _load_counterfactual_aux_examples(
            counterfactual_aux_dir,
            args=args,
            rng=np.random.default_rng(int(config.seed) + 37217),
            dqn_teacher_model=dqn_teacher_model,
            edge_index_np=edge_index,
            device=device,
            torch=torch,
        )
        train_examples = list(train_examples) + counterfactual_aux_examples
        print(
            json.dumps(
                {"phase": "counterfactual_aux_summary", **_json_safe(counterfactual_aux_summary)},
                sort_keys=True,
            ),
            flush=True,
        )

    collect_summary = {
        "train_collect": train_collect,
        "val_collect": val_collect,
        "hard_dagger_train": hard_dagger_summary,
        "preserve_collect": preserve_collect,
        "preserve_examples": int(len(preserve_examples)),
        "preserve_weight": float(args.preserve_weight),
        "counterfactual_aux": counterfactual_aux_summary,
        "counterfactual_aux_examples": int(len(counterfactual_aux_examples)),
        "total_train_examples": int(len(train_examples)),
    }
    _write_json(output_dir / "collection_summary.json", collect_summary)
    print(json.dumps({"phase": "collection_summary", **_json_safe(collect_summary)}, sort_keys=True), flush=True)

    summary = _train(
        config=config,
        train_examples=train_examples,
        val_examples=val_examples,
        edge_index_np=edge_index,
        output_dir=output_dir,
        args=args,
    )
    print(json.dumps({"phase": "done", **_json_safe(summary)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
