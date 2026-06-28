from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import (
    build_checksums,
    ensure_dir,
    project_root,
    read_json,
    save_npz_deterministic,
    stringify,
    utc_timestamp,
    write_checksums,
    write_json,
    write_parquet,
)
from cse2026.data_generation.spectrum import allocate_on_mask, free_count, largest_free_block, route_availability, segment_count
from cse2026.ong_solver import Candidate, CandidateBatch, GnnCnnDqnOngSolver, SolverConfig

from ..config import ExperimentConfig
from .ong_rollout import _add_ong_source_path, _make_env, _raw_bool, _raw_float, _raw_int, _traffic_jsonl_for_episode


def _resolve_output_dataset_path(config: ExperimentConfig) -> Path:
    value = config.resolved.get("output_dataset_path", config.raw.get("output_dataset_path"))
    if not value:
        raise ValueError("collect_ong_expert_dataset requires output_dataset_path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _splits(config: ExperimentConfig) -> list[str]:
    if config.splits:
        return list(config.splits)
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    manifest_path = config.dataset_path / "manifest.json"
    if manifest_path.exists():
        return list(read_json(manifest_path)["splits"].keys())
    return ["train", "val", "test"]


def _solver_config(config: ExperimentConfig) -> SolverConfig:
    return SolverConfig(
        n_max=_raw_int(config, "n_max", 32),
        random_starts_per_route=_raw_int(config, "random_starts_per_route", 2),
        rng_seed=int(config.seed),
        use_neural=False,
        device=str(config.resolved.get("device", config.device)),
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


def _config_string(config: ExperimentConfig, key: str, default: str) -> str:
    return str(config.resolved.get(key, config.raw.get(key, default)))


def _canonical_behavior_policy(value: str | None) -> str:
    normalized = str(value or "q_head_heuristic").strip().lower().replace("_", "-")
    if normalized in {"q-head", "q-head-score", "q-head-heuristic", "hybrid-q-head", "hybrid-qhead"}:
        return "q_head_heuristic"
    if normalized in {"energy-aware-ksp-bm-ff", "energy-aware-ksp-bm-ff-heuristic"}:
        return "energy-aware-ksp-bm-ff"
    raise ValueError(f"Unsupported collection behavior policy: {value}")


def _behavior_branch_name(policy: str, *, softmax: bool = False) -> str:
    if policy == "q_head_heuristic":
        return "softmax_q_head" if softmax else "hybrid_q_head"
    label = policy.replace("-", "_")
    return f"softmax_{label}" if softmax else label


def _energy_aware_ksp_bm_ff_key(candidate: Candidate, index: int) -> tuple[float, int, int, int, float, int]:
    return (
        float(candidate.energy_increment),
        int(candidate.route_id),
        int(candidate.b_start),
        int(candidate.w),
        -float(candidate.spectral_efficiency),
        int(index),
    )


def _ordered_valid_indices(batch: CandidateBatch, policy: str) -> list[int]:
    valid = [int(index) for index in np.flatnonzero(batch.candidate_mask.astype(bool))]
    if not valid:
        return []
    if policy == "q_head_heuristic":
        return sorted(valid, key=lambda index: (-float(batch.topn[index].q_head_score), int(index)))
    if policy == "energy-aware-ksp-bm-ff":
        return sorted(valid, key=lambda index: _energy_aware_ksp_bm_ff_key(batch.topn[index], int(index)))
    raise ValueError(f"Unsupported collection behavior policy: {policy}")


def _select_index(
    *,
    batch: CandidateBatch,
    rng: np.random.Generator,
    expert_probability: float,
    softmax_probability: float,
    softmax_tau: float,
    softmax_top_k: int,
    expert_policy: str,
    softmax_policy: str,
) -> tuple[int, str]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1, "block"
    draw = float(rng.random())
    if draw < expert_probability:
        ordered = _ordered_valid_indices(batch, expert_policy)
        return int(ordered[0]), _behavior_branch_name(expert_policy)
    if draw < expert_probability + softmax_probability:
        ordered = _ordered_valid_indices(batch, softmax_policy)
        top_k = max(1, min(int(softmax_top_k), int(valid.size)))
        top_valid = ordered[:top_k]
        tau = max(float(softmax_tau), 1e-9)
        if softmax_policy == "q_head_heuristic":
            logits = np.asarray([batch.topn[index].q_head_score for index in top_valid], dtype=np.float64) / tau
        else:
            # Tuple-key policies have no natural scalar score; sample by rank under the same ordering.
            logits = -np.arange(len(top_valid), dtype=np.float64) / tau
        logits -= float(np.max(logits))
        probs = np.exp(logits)
        probs /= float(probs.sum())
        return int(top_valid[int(rng.choice(len(top_valid), p=probs))]), _behavior_branch_name(softmax_policy, softmax=True)
    return int(rng.choice(valid)), "random_feasible"


def _best_index(batch: CandidateBatch, policy: str) -> int:
    ordered = _ordered_valid_indices(batch, policy)
    return -1 if not ordered else int(ordered[0])


def _j_total_index(batch: CandidateBatch) -> int:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return -1
    order = sorted(
        (batch.topn[int(index)] for index in valid),
        key=lambda candidate: (candidate.j_total, candidate.energy_increment, candidate.route_id, candidate.b_start),
    )
    return int(order[0].topn_index)


def _hard_case_metadata(batch: CandidateBatch, *, small_margin: float, min_candidates: int) -> dict[str, Any]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return {
            "hard_case": False,
            "hard_case_reason": "no_candidate",
            "q_head_best_candidate_index": -1,
            "j_total_candidate_index": -1,
            "q_head_margin": None,
            "q_head_j_total_disagree": False,
        }
    scores = np.asarray([batch.topn[int(index)].q_head_score for index in valid], dtype=np.float64)
    order = np.argsort(-scores)
    q_index = int(valid[int(order[0])])
    j_index = _j_total_index(batch)
    margin = float(scores[int(order[0])] - scores[int(order[1])]) if len(order) > 1 else math.inf
    reasons: list[str] = []
    if q_index != j_index:
        reasons.append("q_head_differs_j_total")
    if valid.size >= int(min_candidates) and margin <= float(small_margin):
        reasons.append("small_q_head_margin")
    return {
        "hard_case": bool(reasons),
        "hard_case_reason": ",".join(reasons),
        "q_head_best_candidate_index": int(q_index),
        "j_total_candidate_index": int(j_index),
        "q_head_margin": None if not math.isfinite(margin) else float(margin),
        "q_head_j_total_disagree": bool(q_index != j_index),
    }


def _candidate_scores(batch: CandidateBatch, n_max: int) -> list[float | None]:
    scores: list[float | None] = []
    for index in range(n_max):
        if index < len(batch.topn) and bool(batch.candidate_mask[index]):
            scores.append(float(batch.topn[index].q_head_score))
        else:
            scores.append(None)
    return scores


def _candidate_description(candidate: Candidate | None) -> dict[str, Any]:
    if candidate is None:
        return {"route_node_ids": [], "route_directed_link_ids": [], "modulation_id": -1, "b_start": -1, "w": 0}
    return {
        "route_node_ids": list(candidate.route_node_ids),
        "route_directed_link_ids": list(candidate.route_link_ids),
        "modulation_id": int(candidate.modulation_index),
        "b_start": int(candidate.b_start),
        "w": int(candidate.w),
    }


def _candidate_row(
    *,
    candidate: Candidate,
    batch: CandidateBatch,
    candidate_id: int,
    topn_index: int,
    in_topn: bool,
    request: pd.Series,
    n_max: int,
) -> dict[str, Any]:
    availability = route_availability(batch.state.occupancy, candidate.route_link_ids)
    after = allocate_on_mask(availability, candidate.b_start, candidate.w)
    return {
        "candidate_id": int(candidate_id),
        "episode_id": str(request["episode_id"]),
        "request_id": int(request["request_id"]),
        "route_id": int(candidate.route_id),
        "route_node_ids": stringify(list(candidate.route_node_ids)),
        "route_directed_link_ids": stringify(list(candidate.route_link_ids)),
        "modulation_id": int(candidate.modulation_index),
        "b_start": int(candidate.b_start),
        "w": int(candidate.w),
        "required_slots": int(candidate.w),
        "route_length_km": float(candidate.route_length_km),
        "hop_count": int(candidate.hop_count),
        "delay_ms": float(candidate.delay_ms),
        "energy_increment": float(candidate.energy_increment),
        "qot_margin": float(candidate.qot_margin_norm),
        "qot_margin_norm": float(candidate.qot_margin_norm),
        "qot_risk": float(candidate.qot_risk),
        "fragmentation_before": float(candidate.fragmentation_before),
        "fragmentation_after": float(candidate.fragmentation_after),
        "delta_fragmentation": float(candidate.delta_fragmentation),
        "largest_free_block_before": int(max(largest_free_block(availability), 0)),
        "largest_free_block_after": int(candidate.largest_free_block_after),
        "n_free_before": int(free_count(availability)),
        "n_free_after": int(free_count(after)),
        "n_segments_before": int(segment_count(availability)),
        "n_segments_after": int(segment_count(after)),
        "left_gap_after": int(candidate.left_gap_after),
        "right_gap_after": int(candidate.right_gap_after),
        "small_gap_penalty": float(candidate.small_gap_penalty),
        "compactness": float(candidate.compactness),
        "energy_increment_norm": float(candidate.energy_increment_norm),
        "delay_norm": float(candidate.action_features[5]),
        "w_norm": float(candidate.action_features[3]),
        "residual_fit_norm": float(max(candidate.left_gap_after + candidate.right_gap_after, 0) / max(batch.state.slot_count, 1)),
        "j_frag": float(candidate.j_frag),
        "j_tie": float(candidate.j_tie),
        "j_total": float(candidate.j_total),
        "is_feasible": True,
        "in_topn": bool(in_topn),
        "candidate_mask": int(in_topn),
        "q_head_score": float(candidate.q_head_score),
        "topn_index": int(topn_index),
        "n_max": int(n_max),
        "split": str(request["split"]),
        "seed": int(request["seed"]),
        "traffic_scenario": str(request["traffic_scenario"]),
        "load_name": str(request["load_name"]),
    }


def _cnn_index_row(*, sample_id: int, row: dict[str, Any], candidate: Candidate) -> dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "episode_id": row["episode_id"],
        "request_id": int(row["request_id"]),
        "route_id": int(row["route_id"]),
        "modulation_id": int(row["modulation_id"]),
        "b_start": int(row["b_start"]),
        "w": int(row["w"]),
        "route_directed_link_ids": row["route_directed_link_ids"],
        "delta_frag": float(candidate.delta_fragmentation),
        "frag_after": float(candidate.fragmentation_after),
        "lmax_after_norm": float(candidate.action_features[8]),
        "nseg_after_norm": 0.0,
        "created_small_gap": int(candidate.small_gap_penalty > 0.0),
        "compactness": float(candidate.compactness),
        "placement_score": float(-candidate.j_total),
        "J_total": float(candidate.j_total),
        "group_id": f"{row['episode_id']}:{row['request_id']}:r{row['route_id']}:m{row['modulation_id']}",
        "split": row["split"],
        "seed": int(row["seed"]),
        "traffic_scenario": row["traffic_scenario"],
        "load_name": row["load_name"],
    }


def _transition_row(
    *,
    transition_id: int,
    request: pd.Series,
    selected_index: int,
    best_index: int,
    selected: Candidate | None,
    batch: CandidateBatch,
    reward: float,
    ong_reward: float,
    reward_components: dict[str, float],
    done: bool,
    next_state_id: str,
    accepted: bool,
    invalid_action: bool,
    branch: str,
    best_policy: str,
    hard_case_metadata: dict[str, Any],
) -> dict[str, Any]:
    selected_for_learning = selected if accepted and not invalid_action else None
    selected_candidate_index = -1 if selected_for_learning is None else int(selected_index)
    return {
        "transition_id": int(transition_id),
        "episode_id": str(request["episode_id"]),
        "request_id": int(request["request_id"]),
        "state_id": f"{request['episode_id']}:{int(request['request_id'])}",
        "next_state_id": next_state_id,
        "request_features": stringify(
            {
                "src": int(request["src"]),
                "dst": int(request["dst"]),
                "bit_rate_gbps": float(request["bit_rate_gbps"]),
                "holding_time": float(request["holding_time"]),
            }
        ),
        "selected_candidate_index": selected_candidate_index,
        "best_candidate_index": int(best_index),
        "best_candidate_policy": best_policy,
        "q_head_scores": stringify(_candidate_scores(batch, int(batch.candidate_mask.shape[0]))),
        "selected_action_description": stringify(_candidate_description(selected_for_learning)),
        "reward": float(reward),
        "ong_reward": float(ong_reward),
        "reward_components": stringify(reward_components),
        "done": bool(done),
        "blocked": bool(selected_for_learning is None),
        "blocking_reason": "ong_rejected_or_no_candidate" if selected_for_learning is None else "",
        "delta_energy": 0.0 if selected_for_learning is None else float(selected_for_learning.energy_increment),
        "fragmentation_after": 1.0 if selected_for_learning is None else float(selected_for_learning.fragmentation_after),
        "qot_margin": 0.0 if selected_for_learning is None else float(selected_for_learning.qot_margin_norm),
        "qot_risk": 1.0 if selected_for_learning is None else float(selected_for_learning.qot_risk),
        "delay_ms": 0.0 if selected_for_learning is None else float(selected_for_learning.delay_ms),
        "num_feasible_before_topn": int(len(batch.candidates)),
        "num_candidates_after_topn": int(batch.candidate_mask.sum()),
        "candidate_mask_valid": bool(len(batch.topn) <= int(batch.candidate_mask.shape[0])),
        "invalid_action_selected": bool(invalid_action),
        "padding_action_selected": bool(selected_index >= int(batch.candidate_mask.shape[0])),
        "collection_policy_branch": branch,
        "hard_case": bool(hard_case_metadata.get("hard_case", False)),
        "hard_case_reason": str(hard_case_metadata.get("hard_case_reason", "")),
        "q_head_best_candidate_index": int(hard_case_metadata.get("q_head_best_candidate_index", -1)),
        "j_total_candidate_index": int(hard_case_metadata.get("j_total_candidate_index", -1)),
        "q_head_margin": hard_case_metadata.get("q_head_margin"),
        "q_head_j_total_disagree": bool(hard_case_metadata.get("q_head_j_total_disagree", False)),
        "ong_accepted": bool(accepted),
        "split": str(request["split"]),
        "seed": int(request["seed"]),
        "traffic_scenario": str(request["traffic_scenario"]),
        "load_name": str(request["load_name"]),
}


def _problem_shaped_reward(candidate: Candidate | None, accepted: bool, config: ExperimentConfig) -> tuple[float, dict[str, float]]:
    accepted_term = _raw_float(config, "accepted_service_reward", 1.0) if accepted else 0.0
    block_term = _raw_float(config, "block_penalty", -1.0) if not accepted else 0.0
    if candidate is None or not accepted:
        components = {
            "accepted_service": float(accepted_term),
            "block_penalty": float(block_term),
            "energy_penalty": 0.0,
            "fragmentation_penalty": 0.0,
            "qot_margin_bonus": 0.0,
            "delay_penalty": 0.0,
        }
        return float(sum(components.values())), components
    components = {
        "accepted_service": float(accepted_term),
        "block_penalty": 0.0,
        "energy_penalty": -_raw_float(config, "reward_energy_weight", 0.30) * float(candidate.energy_increment_norm),
        "fragmentation_penalty": -_raw_float(config, "reward_fragmentation_weight", 0.35) * float(candidate.fragmentation_after),
        "qot_margin_bonus": _raw_float(config, "reward_qot_margin_weight", 0.15) * float(candidate.qot_margin_norm),
        "delay_penalty": -_raw_float(config, "reward_delay_weight", 0.10) * float(candidate.action_features[5]),
    }
    return float(sum(components.values())), components


def _training_reward(
    *,
    candidate: Candidate | None,
    accepted: bool,
    ong_reward: float,
    config: ExperimentConfig,
) -> tuple[float, dict[str, float], str]:
    mode = str(config.resolved.get("reward_mode", config.raw.get("reward_mode", "actual_ong"))).strip().lower()
    if mode == "problem_shaped":
        reward, components = _problem_shaped_reward(candidate, accepted, config)
        return reward, components, mode
    return float(ong_reward), {"ong_reward": float(ong_reward)}, "actual_ong"


def _write_topology_lengths(output: Path, batch: CandidateBatch | None) -> None:
    if batch is None:
        return
    topology_dir = output / "topology"
    ensure_dir(topology_dir)
    edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
    rows = []
    for link_id, length in enumerate(batch.state.edge_lengths_km):
        src = int(edge_index[0, link_id]) if edge_index.shape[1] > link_id else -1
        dst = int(edge_index[1, link_id]) if edge_index.shape[1] > link_id else -1
        rows.append({"directed_link_id": int(link_id), "src": src, "dst": dst, "length_km": float(length)})
    pd.DataFrame(rows).to_csv(topology_dir / "directed_links.csv", index=False)


def _copy_source_topology(source: Path, output: Path) -> None:
    source_topology = source / "topology"
    output_topology = output / "topology"
    if source_topology.exists() and not output_topology.exists():
        shutil.copytree(source_topology, output_topology)


def _split_counts(output: Path, splits: list[str]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split in splits:
        row: dict[str, Any] = {}
        for key, path in (
            ("traffic_rows", output / "traffic" / f"{split}.parquet"),
            ("candidate_topn_rows", output / "candidates" / f"{split}.parquet"),
            ("candidate_full_rows", output / "candidates" / f"{split}_full.parquet"),
            ("cnn_index_rows", output / "cnn" / f"{split}_index.parquet"),
            ("dqn_transition_rows", output / "dqn" / f"{split}_transitions.parquet"),
        ):
            if path.exists():
                row[key] = int(len(pd.read_parquet(path)))
        graph_path = output / "gnn" / f"{split}_graphs.npz"
        if graph_path.exists():
            row["gnn_graph_shape"] = list(np.load(graph_path)["node_features"].shape)
        cnn_path = output / "cnn" / f"{split}_tensors.npz"
        if cnn_path.exists():
            row["cnn_tensor_shape"] = list(np.load(cnn_path)["X_spec"].shape)
        counts[split] = row
    return counts


def _write_split(
    *,
    output: Path,
    split: str,
    traffic_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    full_candidate_rows: list[dict[str, Any]],
    node_features: list[np.ndarray],
    link_features: list[np.ndarray],
    global_features: list[np.ndarray],
    request_features: list[np.ndarray],
    edge_index: np.ndarray | None,
    sample_ids: list[str],
    cnn_tensors: list[np.ndarray],
    cnn_index_rows: list[dict[str, Any]],
    dqn_rows: list[dict[str, Any]],
) -> None:
    write_parquet(output / "traffic" / f"{split}.parquet", traffic_rows)
    write_parquet(output / "candidates" / f"{split}.parquet", candidate_rows)
    write_parquet(output / "candidates" / f"{split}_full.parquet", full_candidate_rows)
    write_parquet(output / "gnn" / f"{split}_routes.parquet", [])
    write_parquet(output / "cnn" / f"{split}_index.parquet", cnn_index_rows)
    write_parquet(output / "dqn" / f"{split}_transitions.parquet", dqn_rows)
    if node_features:
        nodes = np.stack(node_features, axis=0).astype(np.float32)
        links = np.stack(link_features, axis=0).astype(np.float32)
        globals_ = np.stack(global_features, axis=0).astype(np.float32)
        requests = np.stack(request_features, axis=0).astype(np.float32)
    else:
        nodes = np.zeros((0, 0, 4), dtype=np.float32)
        links = np.zeros((0, 0, 8), dtype=np.float32)
        globals_ = np.zeros((0, 8), dtype=np.float32)
        requests = np.zeros((0, 3), dtype=np.float32)
    save_npz_deterministic(
        output / "gnn" / f"{split}_graphs.npz",
        node_features=nodes,
        link_features=links,
        global_features=globals_,
        request_features=requests,
        edge_index=np.zeros((2, 0), dtype=np.int64) if edge_index is None else np.asarray(edge_index, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype="U128"),
    )
    if cnn_tensors:
        x_spec = np.stack(cnn_tensors, axis=0).astype(np.float16)
    else:
        slots = int(nodes.shape[0] and 100)
        x_spec = np.zeros((0, 6, slots), dtype=np.float16)
    save_npz_deterministic(output / "cnn" / f"{split}_tensors.npz", X_spec=x_spec)


def run_collect_ong_expert_dataset(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("collect_ong_expert_dataset requires source dataset_path")
    source = config.dataset_path
    output = _resolve_output_dataset_path(config)
    if output.exists():
        shutil.rmtree(output)
    for subdir in ("topology", "traffic", "candidates", "gnn", "cnn", "dqn", "reports"):
        ensure_dir(output / subdir)
    _copy_source_topology(source, output)
    _add_ong_source_path(config)

    cfg = _solver_config(config)
    solver = GnnCnnDqnOngSolver(cfg)
    rng = np.random.default_rng(int(config.seed))
    splits = _splits(config)
    expert_probability = _raw_float(config, "expert_probability", 0.70)
    softmax_probability = _raw_float(config, "softmax_probability", 0.20)
    softmax_tau = _raw_float(config, "softmax_tau", 0.35)
    softmax_top_k = _raw_int(config, "softmax_top_k", 8)
    expert_policy = _canonical_behavior_policy(_config_string(config, "expert_policy", "q_head_heuristic"))
    softmax_policy = _canonical_behavior_policy(_config_string(config, "softmax_policy", expert_policy))
    hard_case_collection = _raw_bool(config, "hard_case_collection_enabled", False)
    hard_case_expert_probability = _raw_float(config, "hard_case_expert_probability", expert_probability)
    hard_case_softmax_probability = _raw_float(config, "hard_case_softmax_probability", softmax_probability)
    hard_case_softmax_tau = _raw_float(config, "hard_case_softmax_tau", softmax_tau)
    hard_case_softmax_top_k = _raw_int(config, "hard_case_softmax_top_k", softmax_top_k)
    hard_case_expert_policy = _canonical_behavior_policy(_config_string(config, "hard_case_expert_policy", expert_policy))
    hard_case_softmax_policy = _canonical_behavior_policy(_config_string(config, "hard_case_softmax_policy", softmax_policy))
    hard_case_small_margin = _raw_float(config, "hard_case_small_margin", 0.03)
    hard_case_min_candidates = _raw_int(config, "hard_case_min_candidates", 4)
    max_episodes = _raw_int(config, "max_episodes", 0)
    max_requests_per_episode = _raw_int(config, "max_requests_per_episode", 0)

    run_path = Path(run_dir)
    split_summaries: list[dict[str, Any]] = []
    first_batch: CandidateBatch | None = None

    for split in splits:
        source_traffic = pd.read_parquet(source / "traffic" / f"{split}.parquet")
        episode_ids = tuple(str(value) for value in source_traffic["episode_id"].drop_duplicates().tolist())
        if max_episodes > 0:
            episode_ids = episode_ids[:max_episodes]

        traffic_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        full_candidate_rows: list[dict[str, Any]] = []
        node_features: list[np.ndarray] = []
        link_features: list[np.ndarray] = []
        global_features: list[np.ndarray] = []
        request_features: list[np.ndarray] = []
        sample_ids: list[str] = []
        cnn_tensors: list[np.ndarray] = []
        cnn_index_rows: list[dict[str, Any]] = []
        dqn_rows: list[dict[str, Any]] = []
        edge_index: np.ndarray | None = None
        branch_counts: dict[str, int] = {"hybrid_q_head": 0, "softmax_q_head": 0, "random_feasible": 0, "block": 0}
        hard_case_count = 0
        accepted_count = 0
        invalid_count = 0
        transition_id = 0

        for episode_id in episode_ids:
            episode = source_traffic[source_traffic["episode_id"] == episode_id].sort_values("request_id").reset_index(drop=True)
            if max_requests_per_episode > 0:
                episode = episode.head(max_requests_per_episode).reset_index(drop=True)
            traffic_path = _traffic_jsonl_for_episode(run_path, episode_id, episode)
            env = _make_env(
                episode_id=episode_id,
                traffic_path=traffic_path,
                request_count=len(episode),
                seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
                config=config,
            )
            env.reset(seed=int(config.seed))

            for position, request in episode.iterrows():
                batch = solver.candidate_batch(env)
                if first_batch is None:
                    first_batch = batch
                edge_index = np.asarray(batch.state.edge_index, dtype=np.int64)
                hard_case_metadata = _hard_case_metadata(
                    batch,
                    small_margin=hard_case_small_margin,
                    min_candidates=hard_case_min_candidates,
                )
                hard_case = bool(hard_case_metadata.get("hard_case", False))
                hard_case_count += int(hard_case)
                use_hard_case_branch = bool(hard_case_collection and hard_case)
                selected_index, branch = _select_index(
                    batch=batch,
                    rng=rng,
                    expert_probability=hard_case_expert_probability if use_hard_case_branch else expert_probability,
                    softmax_probability=hard_case_softmax_probability if use_hard_case_branch else softmax_probability,
                    softmax_tau=hard_case_softmax_tau if use_hard_case_branch else softmax_tau,
                    softmax_top_k=hard_case_softmax_top_k if use_hard_case_branch else softmax_top_k,
                    expert_policy=hard_case_expert_policy if use_hard_case_branch else expert_policy,
                    softmax_policy=hard_case_softmax_policy if use_hard_case_branch else softmax_policy,
                )
                if use_hard_case_branch:
                    branch = f"hard_{branch}"
                branch_counts[branch] = branch_counts.get(branch, 0) + 1
                selected = None if selected_index < 0 else batch.topn[selected_index]
                action = solver.adapter(env).block_action(env) if selected is None else selected.action
                mask = env.action_masks()
                invalid = bool(mask is not None and 0 <= int(action) < len(mask) and not bool(mask[int(action)]))

                for full_index, candidate in enumerate(batch.candidates):
                    in_topn = full_index < len(batch.topn)
                    row = _candidate_row(
                        candidate=batch.topn[full_index] if in_topn else candidate,
                        batch=batch,
                        candidate_id=full_index,
                        topn_index=full_index if in_topn else -1,
                        in_topn=in_topn,
                        request=request,
                        n_max=int(cfg.n_max),
                    )
                    full_candidate_rows.append(row)
                    if in_topn:
                        candidate_rows.append(row)
                        cnn_index_rows.append(_cnn_index_row(sample_id=len(cnn_tensors), row=row, candidate=batch.topn[full_index]))
                        cnn_tensors.append(batch.spectrum_tensors[full_index])

                node_features.append(np.asarray(batch.node_features, dtype=np.float32))
                link_features.append(np.asarray(batch.link_features, dtype=np.float32))
                global_features.append(np.asarray(batch.global_features, dtype=np.float32))
                request_features.append(np.asarray(batch.request_features, dtype=np.float32))
                sample_ids.append(f"{episode_id}:{int(request['request_id'])}")

                observation, reward, terminated, truncated, info = env.step(int(action))
                del observation
                done = bool(terminated) or bool(truncated)
                accepted = bool(info.get("accepted", False))
                accepted_count += int(accepted)
                invalid_count += int(invalid)
                training_reward, reward_components, reward_mode = _training_reward(
                    candidate=selected,
                    accepted=accepted and not invalid,
                    ong_reward=float(reward),
                    config=config,
                )
                next_state_id = ""
                if not done and position + 1 < len(episode):
                    next_request = episode.iloc[position + 1]
                    next_state_id = f"{episode_id}:{int(next_request['request_id'])}"
                best_policy = hard_case_expert_policy if use_hard_case_branch else expert_policy
                best_index = _best_index(batch, best_policy)
                dqn_rows.append(
                    _transition_row(
                        transition_id=transition_id,
                        request=request,
                        selected_index=selected_index,
                        best_index=best_index,
                        selected=selected,
                        batch=batch,
                        reward=float(training_reward),
                        ong_reward=float(reward),
                        reward_components=reward_components,
                        done=done,
                        next_state_id=next_state_id,
                        accepted=accepted,
                        invalid_action=invalid,
                        branch=branch,
                        best_policy=best_policy,
                        hard_case_metadata=hard_case_metadata,
                    )
                )
                traffic_row = dict(request)
                traffic_row["num_feasible"] = int(len(batch.candidates))
                traffic_row["num_topn_real"] = int(batch.candidate_mask.sum())
                traffic_row["blocked_by_feasibility"] = bool(not batch.has_real_candidates)
                traffic_rows.append(traffic_row)
                transition_id += 1
                if done:
                    break

        _write_split(
            output=output,
            split=split,
            traffic_rows=traffic_rows,
            candidate_rows=candidate_rows,
            full_candidate_rows=full_candidate_rows,
            node_features=node_features,
            link_features=link_features,
            global_features=global_features,
            request_features=request_features,
            edge_index=edge_index,
            sample_ids=sample_ids,
            cnn_tensors=cnn_tensors,
            cnn_index_rows=cnn_index_rows,
            dqn_rows=dqn_rows,
        )
        split_summaries.append(
            {
                "split": split,
                "episodes": int(len(episode_ids)),
                "requests": int(len(traffic_rows)),
                "accepted": int(accepted_count),
                "blocking_rate": float(1.0 - accepted_count / max(len(traffic_rows), 1)),
                "invalid_actions": int(invalid_count),
                "hard_case_requests": int(hard_case_count),
                "branch_counts": branch_counts,
                "candidate_topn_rows": int(len(candidate_rows)),
                "cnn_tensors": int(len(cnn_tensors)),
            }
        )
        print(json.dumps(split_summaries[-1], sort_keys=True))

    _write_topology_lengths(output, first_batch)
    manifest = {
        "dataset_name": output.name,
        "source_dataset_path": str(source),
        "generation_timestamp": utc_timestamp(),
        "topology": "optical_networking_gym_v2",
        "slot_total": _raw_int(config, "slots", 100),
        "k_routes": _raw_int(config, "k_routes", 5),
        "n_max": int(cfg.n_max),
        "collection_policy": {
            "expert": _behavior_branch_name(expert_policy),
            "expert_policy": expert_policy,
            "softmax_policy": softmax_policy,
            "expert_probability": expert_probability,
            "softmax_probability": softmax_probability,
            "random_probability": max(0.0, 1.0 - expert_probability - softmax_probability),
            "softmax_tau": softmax_tau,
            "softmax_top_k": int(softmax_top_k),
            "reward_source": str(config.resolved.get("reward_mode", config.raw.get("reward_mode", "actual_ong"))),
            "hard_case_collection_enabled": bool(hard_case_collection),
            "hard_case_expert_policy": hard_case_expert_policy,
            "hard_case_softmax_policy": hard_case_softmax_policy,
            "hard_case_expert_probability": float(hard_case_expert_probability),
            "hard_case_softmax_probability": float(hard_case_softmax_probability),
            "hard_case_random_probability": max(0.0, 1.0 - hard_case_expert_probability - hard_case_softmax_probability),
            "hard_case_softmax_tau": float(hard_case_softmax_tau),
            "hard_case_softmax_top_k": int(hard_case_softmax_top_k),
            "hard_case_small_margin": float(hard_case_small_margin),
            "hard_case_min_candidates": int(hard_case_min_candidates),
        },
        "splits": _split_counts(output, splits),
        "parameters": dict(config.resolved),
    }
    checksums = build_checksums(output)
    manifest["checksums"] = checksums
    write_json(output / "manifest.json", manifest)
    write_checksums(output / "checksums.sha256", checksums)
    return {
        "stage": "collect_ong_expert_dataset",
        "dataset_path": str(output),
        "source_dataset_path": str(source),
        "splits": split_summaries,
        "manifest": str(output / "manifest.json"),
    }
