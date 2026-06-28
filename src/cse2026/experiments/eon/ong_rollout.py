from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import (
    Candidate,
    DeepRmsaA3COngSolver,
    GnnCnnA3COngSolver,
    GnnCnnDqnOngSolver,
    SolverConfig,
    XlronGraphTransformerPpoOngSolver,
)
from cse2026.ong_solver.common import masked_argmax, pad_q_scores

from ..config import ExperimentConfig
from .lookahead_override_features import OverrideClassifier, select_q_head_index
from .neural_three_head_runtime import NeuralThreeHeadOverridePolicy
from .tree_ranker_runtime import (
    TREE_RANKER_POLICIES,
    TreeCandidateRanker,
    _passes_safety_guard,
    select_tree_base_index,
)


NEURAL_THREE_HEAD_POLICY = "neural_three_head_override"
GNN_CNN_A3C_OVERRIDE_POLICY = "gnn_cnn_a3c_override"

POLICIES = (
    "random_feasible",
    "ksp_ff",
    "ksp-ff",
    "ksp_bm_ff",
    "ksp-bm-ff",
    "energy_aware_ksp_bm_ff",
    "energy-aware-ksp-bm-ff",
    "j_total_heuristic",
    "q_head_heuristic",
    "gnn_cnn_dqn",
    "gnn_cnn_dqn_safe",
    "gnn_cnn_a3c",
    GNN_CNN_A3C_OVERRIDE_POLICY,
    "deeprmsa_a3c",
    "xlron_graph_transformer_ppo",
    "top32_xlron_stabilized_ppo",
    "q_head_override_classifier",
    NEURAL_THREE_HEAD_POLICY,
    *TREE_RANKER_POLICIES,
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _raw_int(config: ExperimentConfig, key: str, default: int) -> int:
    return int(config.resolved.get(key, config.raw.get(key, default)))


def _raw_float(config: ExperimentConfig, key: str, default: float) -> float:
    return float(config.resolved.get(key, config.raw.get(key, default)))


def _raw_bool(config: ExperimentConfig, key: str, default: bool) -> bool:
    value = config.resolved.get(key, config.raw.get(key, default))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _raw_str(config: ExperimentConfig, key: str, default: str) -> str:
    return str(config.resolved.get(key, config.raw.get(key, default)))


def _raw_list(config: ExperimentConfig, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = config.resolved.get(key, config.raw.get(key, default))
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return tuple(default)


def _resolve_optional_path(config: ExperimentConfig, key: str) -> Path | None:
    value = config.resolved.get(key, config.raw.get(key))
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _tree_ranker_guard_defaults(mode: str) -> dict[str, float | int | bool | str]:
    normalized = str(mode or "strict").strip().lower().replace("_", "-")
    if normalized in {"emergency", "emergency-risk", "extreme", "extreme-risk"}:
        return {
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
    return {
        "mode": "strict",
        "check_fragmentation": True,
        "check_small_gap": True,
        "check_lmax": True,
        "check_qot_margin": True,
        "check_energy": True,
        "check_delay": True,
        "fragmentation_slack": 0.02,
        "small_gap_slack": 0.02,
        "lmax_slack_slots": 4,
        "qot_margin_slack": 0.08,
        "energy_slack_w": 80.0,
        "delay_slack_ms": 1.0,
    }


def _has_config_key(config: ExperimentConfig, key: str) -> bool:
    return key in config.resolved or key in config.raw


def _tree_advantage_gate_overrides(config: ExperimentConfig) -> dict[str, Any] | None:
    keys = (
        "advantage_gate_enabled",
        "advantage_gate_min_win_prob",
        "advantage_gate_max_loss_prob",
        "advantage_gate_min_delta_pred",
        "advantage_gate_win_weight",
        "advantage_gate_loss_weight",
        "advantage_gate_delta_weight",
        "advantage_gate_ranker_margin_weight",
    )
    if not any(_has_config_key(config, key) for key in keys):
        return None
    overrides: dict[str, Any] = {
        "enabled": _raw_bool(config, "advantage_gate_enabled", True),
    }
    float_keys = {
        "advantage_gate_min_win_prob": "min_win_prob",
        "advantage_gate_max_loss_prob": "max_loss_prob",
        "advantage_gate_min_delta_pred": "min_delta_pred",
        "advantage_gate_win_weight": "win_weight",
        "advantage_gate_loss_weight": "loss_weight",
        "advantage_gate_delta_weight": "delta_weight",
        "advantage_gate_ranker_margin_weight": "ranker_margin_weight",
    }
    defaults = {
        "advantage_gate_min_win_prob": 0.5,
        "advantage_gate_max_loss_prob": 0.05,
        "advantage_gate_min_delta_pred": 0.0,
        "advantage_gate_win_weight": 1.0,
        "advantage_gate_loss_weight": 2.0,
        "advantage_gate_delta_weight": 1.0,
        "advantage_gate_ranker_margin_weight": 0.0,
    }
    for config_key, gate_key in float_keys.items():
        if _has_config_key(config, config_key):
            overrides[gate_key] = _raw_float(config, config_key, defaults[config_key])
    return overrides


def _tree_ranker_path_keys(policy: str) -> tuple[str, ...]:
    if policy == "xgboost_candidate_ranker":
        return ("xgboost_ranker_path", "tree_ranker_path")
    if policy == "lightgbm_candidate_ranker":
        return ("lightgbm_ranker_path", "tree_ranker_path")
    return (f"{policy}_path", f"{policy}_ranker_path")


def _resolve_tree_ranker_path(config: ExperimentConfig, policy: str) -> Path | None:
    for key in _tree_ranker_path_keys(policy):
        path = _resolve_optional_path(config, key)
        if path is not None:
            return path
    return None


def _add_ong_source_path(config: ExperimentConfig) -> str | None:
    source = _resolve_optional_path(config, "ong_source_path")
    if source is None:
        return None
    src = source / "src" if (source / "src").exists() else source
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src_text


def _device(config: ExperimentConfig) -> str:
    requested = str(config.resolved.get("device", config.device))
    if requested != "auto":
        return requested
    try:
        from cse2026.ong_solver.models import require_torch

        torch = require_torch()
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _sum(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.sum(np.asarray(finite, dtype=np.float64))) if finite else 0.0


def _topn_count_key(index: int) -> str:
    return f"selected_topn_count_{int(index)}"


def _selected_topn_counts(values: list[Any], n_max: int) -> dict[str, int]:
    counts = {_topn_count_key(index): 0 for index in range(max(int(n_max), 0))}
    for value in values:
        if value is None:
            continue
        topn_index = int(value)
        if 0 <= topn_index < int(n_max):
            counts[_topn_count_key(topn_index)] += 1
    return counts


def _p95_selected_topn_from_rows(rows: list[dict[str, Any]]) -> float | None:
    counts_by_index: dict[int, int] = {}
    for row in rows:
        for key, value in row.items():
            if not str(key).startswith("selected_topn_count_"):
                continue
            try:
                index = int(str(key).rsplit("_", 1)[1])
            except ValueError:
                continue
            counts_by_index[index] = counts_by_index.get(index, 0) + int(value or 0)
    total = sum(counts_by_index.values())
    if total <= 0:
        return None
    threshold = 0.95 * float(total - 1)
    cumulative = 0
    for index in sorted(counts_by_index):
        cumulative += counts_by_index[index]
        if threshold < cumulative:
            return float(index)
    return float(max(counts_by_index))


def _traffic_jsonl_for_episode(run_path: Path, episode_id: str, episode: pd.DataFrame) -> Path:
    table_id = str(episode_id)
    output = run_path / "artifacts" / "traffic_tables" / f"{table_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = episode.sort_values("request_id").reset_index(drop=True)
    manifest = {
        "record_type": "traffic_table",
        "traffic_table_version": "v1",
        "table_id": table_id,
        "scenario_id": table_id,
        "topology_id": "nsfnet_chen",
        "traffic_mode_source": "cse2026_smoke_static_replay",
        "request_count": int(len(ordered)),
        "time_unit": "simulation_time",
        "bit_rate_unit": "Gbps",
        "seed": int(ordered["seed"].iloc[0]) if "seed" in ordered else None,
    }
    lines = [json.dumps(manifest, separators=(",", ":"))]
    for index, row in ordered.iterrows():
        record = {
            "record_type": "traffic_record",
            "request_index": int(index),
            "service_id": int(index),
            "source_id": int(row["src"]) - 1,
            "destination_id": int(row["dst"]) - 1,
            "bit_rate": int(row["bit_rate_gbps"]),
            "arrival_time": float(row["arrival_time"]),
            "holding_time": float(row["holding_time"]),
            "table_id": table_id,
            "row_index": int(index),
            "source_label": str(int(row["src"])),
            "destination_label": str(int(row["dst"])),
            "bit_rate_class": str(int(row["bit_rate_gbps"])),
            "metadata": {
                "original_request_id": int(row["request_id"]),
                "traffic_scenario": str(row.get("traffic_scenario", "")),
                "load_name": str(row.get("load_name", "")),
            },
        }
        lines.append(json.dumps(record, separators=(",", ":")))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _make_env(*, episode_id: str, traffic_path: Path, request_count: int, seed: int, config: ExperimentConfig):
    from optical_networking_gym_v2 import make_env
    from optical_networking_gym_v2.config.scenario import ScenarioConfig
    from optical_networking_gym_v2.contracts.enums import MaskMode, RewardProfile, TrafficMode
    from optical_networking_gym_v2.contracts.modulation import Modulation

    modulations = (
        Modulation("BPSK", 4000.0, 1, minimum_osnr=6.5),
        Modulation("QPSK", 2000.0, 2, minimum_osnr=9.5),
        Modulation("8QAM", 1000.0, 3, minimum_osnr=12.5),
        Modulation("16QAM", 500.0, 4, minimum_osnr=15.5),
    )
    scenario = ScenarioConfig(
        scenario_id=f"cse2026_smoke_{episode_id}",
        topology_id=str(config.resolved.get("ong_topology_id", config.raw.get("ong_topology_id", "nsfnet_chen"))),
        topology_dir=_resolve_optional_path(config, "ong_topology_dir"),
        k_paths=_raw_int(config, "k_routes", 5),
        num_spectrum_resources=_raw_int(config, "slots", 100),
        episode_length=int(request_count),
        max_span_length_km=_raw_float(config, "span_length_km", 80.0),
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"path": str(traffic_path)},
        mask_mode=MaskMode.RESOURCE_AND_QOT,
        reward_profile=RewardProfile.BALANCED,
        qot_constraint=str(config.resolved.get("qot_constraint", config.raw.get("qot_constraint", "DIST"))),
        channel_width=_raw_float(config, "slot_capacity_gbps_at_1bpshz", 12.5),
        frequency_slot_bandwidth=_raw_float(config, "frequency_slot_bandwidth", 12.5e9),
        modulations=modulations,
        modulations_to_consider=len(modulations),
        enable_observation=_raw_bool(config, "enable_observation", False),
        enable_action_mask=True,
        include_mask_in_info=True,
        seed=int(seed),
    )
    return make_env(config=scenario)


def _solver_config(config: ExperimentConfig, *, neural: bool, checkpoint_key: str = "dqn_checkpoint") -> SolverConfig:
    checkpoint_path = _resolve_optional_path(config, checkpoint_key) if neural else None
    return SolverConfig(
        n_max=_raw_int(config, "n_max", 32),
        random_starts_per_route=_raw_int(config, "random_starts_per_route", 2),
        rng_seed=int(config.seed),
        use_neural=bool(neural),
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        q_score_mode=str(config.resolved.get("q_score_mode", config.raw.get("q_score_mode", "raw"))),
        residual_scale=_raw_float(config, "residual_scale", 1.0),
        residual_delta_clip=_raw_float(config, "residual_delta_clip", 0.0),
        deeprmsa_prior_score=str(config.resolved.get("deeprmsa_prior_score", config.raw.get("deeprmsa_prior_score", "q_head_score"))),
        device=_device(config),
        hidden_dim=_raw_int(config, "hidden_dim", 128),
        slot_capacity_gbps_at_1bpshz=_raw_float(config, "slot_capacity_gbps_at_1bpshz", 12.5),
        guard_band_slots=_raw_int(config, "guard_band_slots", 1),
        mean_holding_time=_raw_float(config, "mean_holding_time", 14.0),
        min_useful_slot_width=_raw_int(config, "min_useful_slot_width", 4),
        delay_bound_ms=_raw_float(config, "delay_bound_ms", 50.0),
        span_length_km=_raw_float(config, "span_length_km", 80.0),
        amplifier_power_w=_raw_float(config, "amplifier_power_w", 20.0),
        roadm_switch_power_w=_raw_float(config, "roadm_switch_power_w", 5.0),
        transponder_power_w=_raw_float(config, "transponder_power_w", 100.0),
        energy_norm_w=_raw_float(config, "energy_norm_w", 1200.0),
        max_route_length_norm_km=_raw_float(config, "max_route_length_norm_km", 6000.0),
    )


def _safe_dqn_index(
    *,
    batch: Any,
    q_values: np.ndarray,
    config: ExperimentConfig,
    n_max: int,
) -> tuple[int, bool, float]:
    base_index = select_q_head_index(batch, n_max)
    proposed_index = masked_argmax(q_values, batch.candidate_mask)
    if base_index < 0 or proposed_index < 0:
        return int(proposed_index), False, 0.0
    margin = float(q_values[int(proposed_index)] - q_values[int(base_index)])
    if int(proposed_index) == int(base_index):
        return int(base_index), False, margin
    if margin < _raw_float(config, "dqn_gate_margin", 0.005):
        return int(base_index), False, margin

    base = batch.topn[int(base_index)]
    proposed = batch.topn[int(proposed_index)]
    if proposed.fragmentation_after > base.fragmentation_after + _raw_float(config, "dqn_gate_fragmentation_slack", 0.02):
        return int(base_index), False, margin
    if proposed.small_gap_penalty > base.small_gap_penalty + _raw_float(config, "dqn_gate_small_gap_slack", 0.02):
        return int(base_index), False, margin
    if proposed.largest_free_block_after < base.largest_free_block_after - _raw_int(config, "dqn_gate_lmax_slack_slots", 4):
        return int(base_index), False, margin
    if proposed.qot_margin_norm < base.qot_margin_norm - _raw_float(config, "dqn_gate_qot_margin_slack", 0.08):
        return int(base_index), False, margin
    if proposed.energy_increment > base.energy_increment + _raw_float(config, "dqn_gate_energy_slack_w", 80.0):
        return int(base_index), False, margin
    if proposed.delay_ms > base.delay_ms + _raw_float(config, "dqn_gate_delay_slack_ms", 1.0):
        return int(base_index), False, margin
    return int(proposed_index), True, margin


def _a3c_override_candidate_indices(
    *,
    batch: Any,
    base_index: int,
    candidate_pool: str,
    candidate_pool_top_k: int,
) -> list[int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return []
    pool = str(candidate_pool or "energy_topk_hybrid").strip().lower().replace("-", "_")
    if pool in {"all", "all_topn"}:
        selected = [int(index) for index in valid]
    elif pool in {"energy_topk_hybrid", "quick_topk_hybrid", "quick_top8_hybrid"}:
        top_k = max(1, int(candidate_pool_top_k))
        selected_set: set[int] = set()
        if int(base_index) >= 0:
            selected_set.add(int(base_index))
        energy_order = sorted((int(index) for index in valid), key=lambda index: (float(batch.topn[index].energy_increment_norm), index))
        selected_set.update(energy_order[: max(1, min(top_k, len(energy_order)))])
        selected_set.add(int(valid[0]))
        selected_set.add(min((int(index) for index in valid), key=lambda index: (float(batch.topn[index].fragmentation_after), index)))
        selected_set.add(max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].largest_free_block_after), -index)))
        selected_set.add(max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].qot_margin_norm), -index)))
        selected = [int(index) for index in valid if int(index) in selected_set]
    else:
        raise ValueError(f"Unsupported A3C override candidate_pool: {candidate_pool}")
    if int(base_index) >= 0 and int(base_index) not in selected:
        selected.append(int(base_index))
    return selected


def _a3c_override_safety_guard(config: ExperimentConfig) -> dict[str, float | int | bool | str]:
    mode = _raw_str(config, "a3c_override_safety_guard_mode", _raw_str(config, "tree_ranker_safety_guard_mode", "emergency"))
    defaults = _tree_ranker_guard_defaults(mode)
    return {
        "enabled": _raw_bool(config, "a3c_override_safety_guard", _raw_bool(config, "tree_ranker_safety_guard", True)),
        "mode": str(defaults["mode"]),
        "check_fragmentation": _raw_bool(
            config,
            "a3c_override_guard_check_fragmentation",
            _raw_bool(config, "tree_ranker_guard_check_fragmentation", bool(defaults["check_fragmentation"])),
        ),
        "check_small_gap": _raw_bool(
            config,
            "a3c_override_guard_check_small_gap",
            _raw_bool(config, "tree_ranker_guard_check_small_gap", bool(defaults["check_small_gap"])),
        ),
        "check_lmax": _raw_bool(
            config,
            "a3c_override_guard_check_lmax",
            _raw_bool(config, "tree_ranker_guard_check_lmax", bool(defaults["check_lmax"])),
        ),
        "check_qot_margin": _raw_bool(
            config,
            "a3c_override_guard_check_qot_margin",
            _raw_bool(config, "tree_ranker_guard_check_qot_margin", bool(defaults["check_qot_margin"])),
        ),
        "check_energy": _raw_bool(
            config,
            "a3c_override_guard_check_energy",
            _raw_bool(config, "tree_ranker_guard_check_energy", bool(defaults["check_energy"])),
        ),
        "check_delay": _raw_bool(
            config,
            "a3c_override_guard_check_delay",
            _raw_bool(config, "tree_ranker_guard_check_delay", bool(defaults["check_delay"])),
        ),
        "fragmentation_slack": _raw_float(
            config,
            "a3c_override_guard_fragmentation_slack",
            _raw_float(config, "tree_ranker_guard_fragmentation_slack", float(defaults["fragmentation_slack"])),
        ),
        "small_gap_slack": _raw_float(
            config,
            "a3c_override_guard_small_gap_slack",
            _raw_float(config, "tree_ranker_guard_small_gap_slack", float(defaults["small_gap_slack"])),
        ),
        "lmax_slack_slots": _raw_int(
            config,
            "a3c_override_guard_lmax_slack_slots",
            _raw_int(config, "tree_ranker_guard_lmax_slack_slots", int(defaults["lmax_slack_slots"])),
        ),
        "qot_margin_slack": _raw_float(
            config,
            "a3c_override_guard_qot_margin_slack",
            _raw_float(config, "tree_ranker_guard_qot_margin_slack", float(defaults["qot_margin_slack"])),
        ),
        "energy_slack_w": _raw_float(
            config,
            "a3c_override_guard_energy_slack_w",
            _raw_float(config, "tree_ranker_guard_energy_slack_w", float(defaults["energy_slack_w"])),
        ),
        "delay_slack_ms": _raw_float(
            config,
            "a3c_override_guard_delay_slack_ms",
            _raw_float(config, "tree_ranker_guard_delay_slack_ms", float(defaults["delay_slack_ms"])),
        ),
    }


def _a3c_override_index(
    *,
    batch: Any,
    logits: np.ndarray,
    config: ExperimentConfig,
    n_max: int,
) -> tuple[int, bool, float]:
    base_policy = _raw_str(config, "a3c_override_base_policy", _raw_str(config, "tree_ranker_base_policy", "energy-aware-ksp-bm-ff"))
    base_index = select_tree_base_index(batch, n_max, base_policy)
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1, False, 0.0
    if int(base_index) < 0 or not bool(batch.candidate_mask[int(base_index)]):
        base_index = int(valid[0])
    candidate_indices = _a3c_override_candidate_indices(
        batch=batch,
        base_index=int(base_index),
        candidate_pool=_raw_str(config, "a3c_override_candidate_pool", "energy_topk_hybrid"),
        candidate_pool_top_k=_raw_int(config, "a3c_override_candidate_pool_top_k", 8),
    )
    if not candidate_indices:
        return int(base_index), False, 0.0
    best_index = int(max(candidate_indices, key=lambda index: (float(logits[int(index)]), -int(index))))
    margin = float(logits[int(best_index)] - logits[int(base_index)])
    if int(best_index) == int(base_index):
        return int(base_index), False, margin
    if margin < _raw_float(config, "a3c_override_selection_margin", 0.0):
        return int(base_index), False, margin
    guard = _a3c_override_safety_guard(config)
    if not _passes_safety_guard(batch.topn[int(best_index)], batch.topn[int(base_index)], guard):
        return int(base_index), False, margin
    return int(best_index), True, margin


def _best_index_by_key(batch: Any, key: Any) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1
    return int(min((int(index) for index in valid), key=lambda index: key(batch.topn[index], index)))


def _select_ksp_ff_index(batch: Any) -> int:
    return _best_index_by_key(
        batch,
        lambda candidate, index: (
            int(candidate.route_id),
            int(candidate.b_start),
            int(candidate.modulation_offset),
            int(candidate.w),
            float(candidate.energy_increment),
            int(index),
        ),
    )


def _select_ksp_bm_ff_index(batch: Any) -> int:
    return _best_index_by_key(
        batch,
        lambda candidate, index: (
            int(candidate.route_id),
            int(candidate.b_start),
            int(candidate.w),
            -float(candidate.spectral_efficiency),
            float(candidate.energy_increment),
            int(index),
        ),
    )


def _select_energy_aware_ksp_bm_ff_index(batch: Any) -> int:
    return _best_index_by_key(
        batch,
        lambda candidate, index: (
            float(candidate.energy_increment),
            int(candidate.route_id),
            int(candidate.b_start),
            int(candidate.w),
            -float(candidate.spectral_efficiency),
            int(index),
        ),
    )


def _select_candidate(
    *,
    policy: str,
    solver: Any,
    env: Any,
    rng: np.random.Generator,
    config: ExperimentConfig,
    override_classifier: OverrideClassifier | None = None,
    tree_ranker: TreeCandidateRanker | None = None,
    neural_three_head: NeuralThreeHeadOverridePolicy | None = None,
) -> tuple[int, Candidate | None, int, int, bool, float]:
    batch = solver.candidate_batch(env)
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return int(solver.adapter(env).block_action(env)), None, 0, 0, False, 0.0

    override_applied = False
    override_probability = 0.0
    if policy == "random_feasible":
        selected_index = int(rng.choice(valid))
    elif policy in {"ksp_ff", "ksp-ff"}:
        selected_index = _select_ksp_ff_index(batch)
    elif policy in {"ksp_bm_ff", "ksp-bm-ff"}:
        selected_index = _select_ksp_bm_ff_index(batch)
    elif policy in {"energy_aware_ksp_bm_ff", "energy-aware-ksp-bm-ff"}:
        selected_index = _select_energy_aware_ksp_bm_ff_index(batch)
    elif policy == "j_total_heuristic":
        selected_index = int(valid[0])
    elif policy == "q_head_heuristic":
        scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
        selected_index = masked_argmax(pad_q_scores(scores, solver.config.n_max), batch.candidate_mask)
    elif policy == "q_head_override_classifier":
        if override_classifier is None:
            raise ValueError("q_head_override_classifier requires override_classifier_path")
        selected_index = select_q_head_index(batch, solver.config.n_max)
        override_index, override_probability = override_classifier.select_index(batch, solver.config.n_max)
        if override_index >= 0:
            selected_index = int(override_index)
            override_applied = True
    elif policy in TREE_RANKER_POLICIES:
        if tree_ranker is None:
            raise ValueError(f"{policy} requires a tree ranker path")
        selected_index = select_tree_base_index(batch, solver.config.n_max, tree_ranker.base_policy)
        if selected_index < 0:
            selected_index = int(valid[0])
        ranker_index, override_probability = tree_ranker.select_index(batch, solver.config.n_max)
        if ranker_index >= 0:
            override_applied = int(ranker_index) != int(selected_index)
            selected_index = int(ranker_index)
    elif policy == "gnn_cnn_dqn":
        q_values = solver.q_values(batch)
        selected_index = masked_argmax(q_values, batch.candidate_mask)
    elif policy == "gnn_cnn_dqn_safe":
        q_values = solver.q_values(batch)
        selected_index, override_applied, override_probability = _safe_dqn_index(
            batch=batch,
            q_values=q_values,
            config=config,
            n_max=solver.config.n_max,
        )
    elif policy == GNN_CNN_A3C_OVERRIDE_POLICY:
        q_values = solver.q_values(batch)
        selected_index, override_applied, override_probability = _a3c_override_index(
            batch=batch,
            logits=q_values,
            config=config,
            n_max=solver.config.n_max,
        )
    elif policy == NEURAL_THREE_HEAD_POLICY:
        if neural_three_head is None:
            raise ValueError(f"{NEURAL_THREE_HEAD_POLICY} requires neural_three_head_override_path")
        selected_index, override_applied, override_probability = neural_three_head.select_index(batch, solver.config)
        if selected_index < 0:
            selected_index = int(valid[0])
    elif policy in {"deeprmsa_a3c", "xlron_graph_transformer_ppo", "top32_xlron_stabilized_ppo", "gnn_cnn_a3c"}:
        q_values = solver.q_values(batch)
        selected_index = masked_argmax(q_values, batch.candidate_mask)
    else:
        raise ValueError(f"Unsupported rollout policy: {policy}")

    selected = batch.topn[selected_index]
    return (
        int(selected.action),
        selected,
        int(valid.size),
        int(selected_index),
        bool(override_applied),
        float(override_probability),
    )


def _run_policy_episode(
    *,
    policy: str,
    episode_id: str,
    episode: pd.DataFrame,
    traffic_path: Path,
    config: ExperimentConfig,
    solver: Any,
    rng: np.random.Generator,
    override_classifier: OverrideClassifier | None = None,
    tree_ranker: TreeCandidateRanker | None = None,
    neural_three_head: NeuralThreeHeadOverridePolicy | None = None,
) -> dict[str, Any]:
    env = _make_env(
        episode_id=episode_id,
        traffic_path=traffic_path,
        request_count=len(episode),
        seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
        config=config,
    )
    env.reset(seed=int(config.seed))
    requests = 0
    accepted = 0
    invalid_actions = 0
    no_candidate_requests = 0
    total_reward: list[float] = []
    selected_energy: list[float] = []
    selected_fragmentation: list[float] = []
    selected_delay: list[float] = []
    selected_qot_margin_norm: list[float] = []
    selected_candidate_count: list[float] = []
    selected_topn_index: list[float] = []
    decision_latency_ms: list[float] = []
    override_requests = 0
    override_applied = 0
    override_probabilities: list[float] = []
    final_info: dict[str, Any] = {}

    while True:
        decision_started = time.perf_counter()
        (
            action,
            candidate,
            valid_count,
            topn_index,
            did_override,
            override_probability,
        ) = _select_candidate(
            policy=policy,
            solver=solver,
            env=env,
            rng=rng,
            config=config,
            override_classifier=override_classifier,
            tree_ranker=tree_ranker,
            neural_three_head=neural_three_head,
        )
        decision_latency_ms.append(float((time.perf_counter() - decision_started) * 1000.0))
        if policy == "q_head_override_classifier" and candidate is not None:
            override_requests += 1
            override_applied += int(bool(did_override))
            override_probabilities.append(float(override_probability))
        if policy in TREE_RANKER_POLICIES and candidate is not None:
            override_requests += 1
            override_applied += int(bool(did_override))
            override_probabilities.append(float(override_probability))
        if policy == "gnn_cnn_dqn_safe" and candidate is not None:
            override_requests += 1
            override_applied += int(bool(did_override))
            override_probabilities.append(float(override_probability))
        if policy == GNN_CNN_A3C_OVERRIDE_POLICY and candidate is not None:
            override_requests += 1
            override_applied += int(bool(did_override))
            override_probabilities.append(float(override_probability))
        if policy == NEURAL_THREE_HEAD_POLICY and candidate is not None:
            override_requests += 1
            override_applied += int(bool(did_override))
            override_probabilities.append(float(override_probability))
        mask = env.action_masks()
        if mask is not None and 0 <= int(action) < len(mask) and not bool(mask[int(action)]):
            invalid_actions += 1
        if candidate is None:
            no_candidate_requests += 1
        else:
            selected_energy.append(float(candidate.energy_increment))
            selected_fragmentation.append(float(candidate.fragmentation_after))
            selected_delay.append(float(candidate.delay_ms))
            selected_qot_margin_norm.append(float(candidate.qot_margin_norm))
            selected_candidate_count.append(float(valid_count))
            selected_topn_index.append(float(topn_index))

        _observation, reward, terminated, truncated, info = env.step(int(action))
        final_info = dict(info)
        total_reward.append(float(reward))
        requests += 1
        accepted += int(bool(info.get("accepted", False)))
        if bool(terminated) or bool(truncated):
            break

    selected_topn_counts = _selected_topn_counts(selected_topn_index, int(solver.config.n_max))
    return {
        "policy": policy,
        "episode_id": episode_id,
        "requests": int(requests),
        "accepted": int(accepted),
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": _sum(total_reward),
        "mean_reward": _mean(total_reward),
        "invalid_actions": int(invalid_actions),
        "no_candidate_requests": int(no_candidate_requests),
        "mean_selected_energy_increment": _mean(selected_energy),
        "mean_selected_fragmentation_after": _mean(selected_fragmentation),
        "mean_selected_delay_ms": _mean(selected_delay),
        "mean_selected_qot_margin_norm": _mean(selected_qot_margin_norm),
        "mean_valid_topn_candidates": _mean(selected_candidate_count),
        "mean_selected_topn_index": _mean(selected_topn_index),
        "p95_selected_topn_index": _p95_selected_topn_from_rows([selected_topn_counts]),
        "mean_decision_latency_ms": _mean(decision_latency_ms),
        "max_decision_latency_ms": None if not decision_latency_ms else float(max(decision_latency_ms)),
        "p95_decision_latency_ms": None
        if not decision_latency_ms
        else float(np.percentile(np.asarray(decision_latency_ms, dtype=np.float64), 95)),
        "override_requests": int(override_requests),
        "override_applied": int(override_applied),
        "override_rate": float(override_applied / max(override_requests, 1)),
        "mean_override_probability": _mean(override_probabilities),
        "ong_episode_service_blocking_rate": final_info.get("episode_service_blocking_rate"),
        "ong_episode_bit_rate_blocking_rate": final_info.get("episode_bit_rate_blocking_rate"),
        "traffic_scenario": str(episode["traffic_scenario"].iloc[0]) if "traffic_scenario" in episode else "",
        "load_name": str(episode["load_name"].iloc[0]) if "load_name" in episode else "",
        "seed": int(episode["seed"].iloc[0]) if "seed" in episode else None,
        **selected_topn_counts,
    }


def _aggregate_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requests = int(sum(int(row["requests"]) for row in rows))
    accepted = int(sum(int(row["accepted"]) for row in rows))
    rewards = [float(row["total_reward"]) for row in rows]
    aggregate = {
        "policy": rows[0]["policy"] if rows else "",
        "episodes": int(len(rows)),
        "requests": requests,
        "accepted": accepted,
        "blocked": int(requests - accepted),
        "blocking_rate": float((requests - accepted) / max(requests, 1)),
        "total_reward": _sum(rewards),
        "mean_episode_reward": _mean(rewards),
        "mean_reward_per_request": float(_sum(rewards) / max(requests, 1)),
        "invalid_actions": int(sum(int(row["invalid_actions"]) for row in rows)),
        "no_candidate_requests": int(sum(int(row["no_candidate_requests"]) for row in rows)),
        "mean_selected_energy_increment": _mean([row["mean_selected_energy_increment"] for row in rows]),
        "mean_selected_fragmentation_after": _mean([row["mean_selected_fragmentation_after"] for row in rows]),
        "mean_selected_delay_ms": _mean([row["mean_selected_delay_ms"] for row in rows]),
        "mean_selected_qot_margin_norm": _mean([row["mean_selected_qot_margin_norm"] for row in rows]),
        "mean_valid_topn_candidates": _mean([row["mean_valid_topn_candidates"] for row in rows]),
        "mean_selected_topn_index": _mean([row["mean_selected_topn_index"] for row in rows]),
        "p95_selected_topn_index": _p95_selected_topn_from_rows(rows),
        "mean_decision_latency_ms": _mean([row["mean_decision_latency_ms"] for row in rows]),
        "max_decision_latency_ms": max(
            [float(row["max_decision_latency_ms"]) for row in rows if row["max_decision_latency_ms"] is not None],
            default=None,
        ),
        "p95_decision_latency_ms": _mean([row["p95_decision_latency_ms"] for row in rows]),
        "override_requests": int(sum(int(row.get("override_requests", 0)) for row in rows)),
        "override_applied": int(sum(int(row.get("override_applied", 0)) for row in rows)),
        "override_rate": float(
            sum(int(row.get("override_applied", 0)) for row in rows)
            / max(sum(int(row.get("override_requests", 0)) for row in rows), 1)
        ),
        "mean_override_probability": _mean([row.get("mean_override_probability") for row in rows]),
    }
    count_keys = sorted(
        {key for row in rows for key in row if str(key).startswith("selected_topn_count_")},
        key=lambda key: int(str(key).rsplit("_", 1)[1]),
    )
    aggregate.update({key: int(sum(int(row.get(key, 0) or 0) for row in rows)) for key in count_keys})
    return aggregate


def run_ong_rollout(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("evaluate_policy requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)

    policies = _raw_list(config, "policies", POLICIES)
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    split = str(config.resolved.get("rollout_split", config.raw.get("rollout_split", "test")))
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    episode_ids = tuple(str(value) for value in traffic["episode_id"].drop_duplicates().tolist())
    max_episodes = _raw_int(config, "max_episodes", 0)
    if max_episodes > 0:
        episode_ids = episode_ids[:max_episodes]
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)

    traffic_paths: dict[str, Path] = {}
    episodes: dict[str, pd.DataFrame] = {}
    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"] == episode_id].sort_values("request_id")
        if max_requests_per_episode > 0:
            episode = episode.head(max_requests_per_episode)
        episodes[episode_id] = episode
        traffic_paths[episode_id] = _traffic_jsonl_for_episode(run_path, episode_id, episode)

    solvers: dict[str, Any] = {}
    override_classifiers: dict[str, OverrideClassifier | None] = {}
    tree_rankers: dict[str, TreeCandidateRanker | None] = {}
    neural_three_heads: dict[str, NeuralThreeHeadOverridePolicy | None] = {}
    tree_selection_mode = str(config.resolved.get("tree_ranker_selection_mode", config.raw.get("tree_ranker_selection_mode", "")))
    tree_selection_mode = tree_selection_mode or None
    tree_residual_beta = config.resolved.get("tree_ranker_residual_beta", config.raw.get("tree_ranker_residual_beta"))
    tree_selection_margin = config.resolved.get("tree_ranker_selection_margin", config.raw.get("tree_ranker_selection_margin"))
    tree_base_policy = config.resolved.get("tree_ranker_base_policy", config.raw.get("tree_ranker_base_policy"))
    tree_advantage_gate = _tree_advantage_gate_overrides(config)
    tree_guard_mode = _raw_str(config, "tree_ranker_safety_guard_mode", "strict")
    tree_guard_defaults = _tree_ranker_guard_defaults(tree_guard_mode)
    tree_safety_guard = {
        "enabled": _raw_bool(config, "tree_ranker_safety_guard", False),
        "mode": str(tree_guard_defaults["mode"]),
        "check_fragmentation": _raw_bool(
            config,
            "tree_ranker_guard_check_fragmentation",
            bool(tree_guard_defaults["check_fragmentation"]),
        ),
        "check_small_gap": _raw_bool(
            config,
            "tree_ranker_guard_check_small_gap",
            bool(tree_guard_defaults["check_small_gap"]),
        ),
        "check_lmax": _raw_bool(config, "tree_ranker_guard_check_lmax", bool(tree_guard_defaults["check_lmax"])),
        "check_qot_margin": _raw_bool(
            config,
            "tree_ranker_guard_check_qot_margin",
            bool(tree_guard_defaults["check_qot_margin"]),
        ),
        "check_energy": _raw_bool(config, "tree_ranker_guard_check_energy", bool(tree_guard_defaults["check_energy"])),
        "check_delay": _raw_bool(config, "tree_ranker_guard_check_delay", bool(tree_guard_defaults["check_delay"])),
        "fragmentation_slack": _raw_float(
            config,
            "tree_ranker_guard_fragmentation_slack",
            float(tree_guard_defaults["fragmentation_slack"]),
        ),
        "small_gap_slack": _raw_float(
            config,
            "tree_ranker_guard_small_gap_slack",
            float(tree_guard_defaults["small_gap_slack"]),
        ),
        "lmax_slack_slots": _raw_int(
            config,
            "tree_ranker_guard_lmax_slack_slots",
            int(tree_guard_defaults["lmax_slack_slots"]),
        ),
        "qot_margin_slack": _raw_float(
            config,
            "tree_ranker_guard_qot_margin_slack",
            float(tree_guard_defaults["qot_margin_slack"]),
        ),
        "energy_slack_w": _raw_float(
            config,
            "tree_ranker_guard_energy_slack_w",
            float(tree_guard_defaults["energy_slack_w"]),
        ),
        "delay_slack_ms": _raw_float(
            config,
            "tree_ranker_guard_delay_slack_ms",
            float(tree_guard_defaults["delay_slack_ms"]),
        ),
    }
    for policy in policies:
        if policy == "deeprmsa_a3c":
            solvers[policy] = DeepRmsaA3COngSolver(
                _solver_config(config, neural=True, checkpoint_key="deeprmsa_checkpoint")
            )
        elif policy == "gnn_cnn_a3c":
            solvers[policy] = GnnCnnA3COngSolver(
                _solver_config(config, neural=True, checkpoint_key="gnn_cnn_a3c_checkpoint")
            )
        elif policy == GNN_CNN_A3C_OVERRIDE_POLICY:
            solvers[policy] = GnnCnnA3COngSolver(
                _solver_config(config, neural=True, checkpoint_key="gnn_cnn_a3c_override_checkpoint")
            )
        elif policy == "xlron_graph_transformer_ppo":
            solvers[policy] = XlronGraphTransformerPpoOngSolver(
                _solver_config(config, neural=True, checkpoint_key="xlron_transformer_checkpoint")
            )
        elif policy == "top32_xlron_stabilized_ppo":
            solvers[policy] = XlronGraphTransformerPpoOngSolver(
                _solver_config(config, neural=True, checkpoint_key="top32_xlron_stabilized_ppo_checkpoint")
            )
        else:
            solvers[policy] = GnnCnnDqnOngSolver(
                _solver_config(config, neural=(policy in {"gnn_cnn_dqn", "gnn_cnn_dqn_safe"}))
            )
        if policy == "q_head_override_classifier":
            classifier_path = _resolve_optional_path(config, "override_classifier_path")
            if classifier_path is None:
                raise ValueError("q_head_override_classifier requires override_classifier_path")
            override_classifiers[policy] = OverrideClassifier.load(classifier_path)
        else:
            override_classifiers[policy] = None
        if policy in TREE_RANKER_POLICIES:
            ranker_path = _resolve_tree_ranker_path(config, policy)
            if ranker_path is None:
                raise ValueError(f"{policy} requires one of {', '.join(_tree_ranker_path_keys(policy))}")
            tree_rankers[policy] = TreeCandidateRanker.load(
                ranker_path,
                selection_mode=tree_selection_mode,
                residual_beta=None if tree_residual_beta is None else float(tree_residual_beta),
                selection_margin=None if tree_selection_margin is None else float(tree_selection_margin),
                base_policy=None if tree_base_policy is None else str(tree_base_policy),
                safety_guard=tree_safety_guard if bool(tree_safety_guard["enabled"]) else None,
                advantage_gate=tree_advantage_gate,
            )
        else:
            tree_rankers[policy] = None
        if policy == NEURAL_THREE_HEAD_POLICY:
            three_head_path = _resolve_optional_path(config, "neural_three_head_override_path")
            if three_head_path is None:
                raise ValueError(f"{NEURAL_THREE_HEAD_POLICY} requires neural_three_head_override_path")
            neural_three_heads[policy] = NeuralThreeHeadOverridePolicy.load(
                three_head_path,
                device=_device(config),
                base_policy=_raw_str(config, "neural_three_head_base_policy", "energy-aware-ksp-bm-ff"),
                live_gate_path=_resolve_optional_path(config, "neural_three_head_live_gate_path"),
            )
        else:
            neural_three_heads[policy] = None

    rows: list[dict[str, Any]] = []
    for policy in policies:
        rng = np.random.default_rng(int(config.seed))
        print(f"Starting ONG rollout policy={policy} episodes={len(episode_ids)} split={split}")
        for episode_id in episode_ids:
            row = _run_policy_episode(
                policy=policy,
                episode_id=episode_id,
                episode=episodes[episode_id],
                traffic_path=traffic_paths[episode_id],
                config=config,
                solver=solvers[policy],
                rng=rng,
                override_classifier=override_classifiers.get(policy),
                tree_ranker=tree_rankers.get(policy),
                neural_three_head=neural_three_heads.get(policy),
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    per_policy = [_aggregate_policy([row for row in rows if row["policy"] == policy]) for policy in policies]
    per_episode = pd.DataFrame(rows)
    summary = pd.DataFrame(per_policy)
    per_episode.to_csv(run_path / "policy_episode_metrics.csv", index=False)
    summary.to_csv(run_path / "policy_summary.csv", index=False)
    metrics = {
        "stage": "evaluate_policy",
        "rollout": "optical_networking_gym_v2_static_replay",
        "ong_source_path": ong_source,
        "dataset_path": str(config.dataset_path),
        "split": split,
        "policies": list(policies),
        "episodes": list(episode_ids),
        "solver_config": asdict(_solver_config(config, neural=False)),
        "dqn_checkpoint": str(_resolve_optional_path(config, "dqn_checkpoint")),
        "deeprmsa_checkpoint": str(_resolve_optional_path(config, "deeprmsa_checkpoint")),
        "gnn_cnn_a3c_checkpoint": str(_resolve_optional_path(config, "gnn_cnn_a3c_checkpoint")),
        "gnn_cnn_a3c_override_checkpoint": str(_resolve_optional_path(config, "gnn_cnn_a3c_override_checkpoint")),
        "xlron_transformer_checkpoint": str(_resolve_optional_path(config, "xlron_transformer_checkpoint")),
        "top32_xlron_stabilized_ppo_checkpoint": str(_resolve_optional_path(config, "top32_xlron_stabilized_ppo_checkpoint")),
        "override_classifier_path": str(_resolve_optional_path(config, "override_classifier_path")),
        "xgboost_ranker_path": str(_resolve_optional_path(config, "xgboost_ranker_path")),
        "lightgbm_ranker_path": str(_resolve_optional_path(config, "lightgbm_ranker_path")),
        "tree_ranker_paths": {policy: str(_resolve_tree_ranker_path(config, policy)) for policy in policies if policy in TREE_RANKER_POLICIES},
        "dqn_safe_gate": {
            "margin": _raw_float(config, "dqn_gate_margin", 0.005),
            "fragmentation_slack": _raw_float(config, "dqn_gate_fragmentation_slack", 0.02),
            "small_gap_slack": _raw_float(config, "dqn_gate_small_gap_slack", 0.02),
            "lmax_slack_slots": _raw_int(config, "dqn_gate_lmax_slack_slots", 4),
            "qot_margin_slack": _raw_float(config, "dqn_gate_qot_margin_slack", 0.08),
            "energy_slack_w": _raw_float(config, "dqn_gate_energy_slack_w", 80.0),
            "delay_slack_ms": _raw_float(config, "dqn_gate_delay_slack_ms", 1.0),
        },
        "per_policy": per_policy,
        "per_episode": rows,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
