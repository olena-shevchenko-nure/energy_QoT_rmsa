from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cse2026.data_generation.spectrum import (
    allocate_on_mask,
    contiguous_segments,
    fragmentation,
    largest_free_block,
    route_occupancy_fraction,
)


@dataclass(frozen=True)
class SolverConfig:
    """Runtime configuration for the ONG GNN+CNN+DQN solver."""

    n_max: int = 32
    random_starts_per_route: int = 2
    rng_seed: int = 0
    epsilon: float = 0.0
    use_neural: bool = False
    checkpoint_path: str | None = None
    q_score_mode: str = "raw"
    residual_scale: float = 1.0
    residual_delta_clip: float = 0.0
    deeprmsa_prior_score: str = "q_head_score"
    device: str = "cpu"
    hidden_dim: int = 128
    slot_capacity_gbps_at_1bpshz: float = 12.5
    guard_band_slots: int = 1
    mean_holding_time: float = 14.0
    min_useful_slot_width: int = 4
    delay_bound_ms: float = 50.0
    span_length_km: float = 80.0
    amplifier_power_w: float = 20.0
    roadm_switch_power_w: float = 5.0
    transponder_power_w: float = 100.0
    energy_norm_w: float = 1200.0
    max_route_length_norm_km: float = 6000.0
    q_head_weights: dict[str, float] = field(
        default_factory=lambda: {
            "energy": 0.30,
            "fragmentation_after": 0.25,
            "qot_risk": 0.10,
            "delay": 0.08,
            "qot_margin": 0.12,
            "largest_free_block": 0.35,
            "delta_fragmentation": 0.30,
            "small_gap": 0.35,
            "compactness": 0.05,
        }
    )
    scoring_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lambda_frag": 1.0,
            "lambda_gap": 0.6,
            "lambda_lmax": 0.4,
            "lambda_fit": 0.3,
            "mu_energy": 0.25,
            "mu_delay": 0.2,
            "mu_qot": 0.2,
            "mu_width": 0.1,
        }
    )


def normalize_q_score_mode(value: str | None) -> str:
    mode = str(value or "raw").strip().lower()
    if mode in {"raw", "neural", "q", "dqn"}:
        return "raw"
    if mode in {"q_head_residual", "qhead_residual", "residual", "q_head_plus_delta"}:
        return "q_head_residual"
    raise ValueError(f"Unsupported q_score_mode: {value}")


@dataclass(frozen=True)
class StateView:
    node_names: tuple[Any, ...]
    edge_index: np.ndarray
    edge_lengths_km: np.ndarray
    occupancy: np.ndarray
    release_times: np.ndarray
    active_link_counts: np.ndarray
    active_node_counts: np.ndarray
    src: Any
    dst: Any
    bit_rate_gbps: float
    holding_time: float
    current_time: float
    topology_name: str = ""

    @property
    def slot_count(self) -> int:
        return int(self.occupancy.shape[1])

    @property
    def link_count(self) -> int:
        return int(self.occupancy.shape[0])


@dataclass(frozen=True)
class Candidate:
    action: Any
    route_id: int
    modulation_index: int
    modulation_offset: int
    b_start: int
    w: int
    route_node_ids: tuple[Any, ...]
    route_link_ids: tuple[int, ...]
    route_length_km: float
    hop_count: int
    delay_ms: float
    modulation_name: str
    spectral_efficiency: float
    qot_margin_norm: float
    qot_risk: float
    energy_increment: float
    energy_increment_norm: float
    fragmentation_before: float
    fragmentation_after: float
    delta_fragmentation: float
    largest_free_block_after: int
    left_gap_after: int
    right_gap_after: int
    small_gap_penalty: float
    compactness: float
    j_frag: float
    j_tie: float
    j_total: float
    q_head_score: float
    action_features: tuple[float, ...]
    topn_index: int = -1

    def with_topn_index(self, topn_index: int) -> "Candidate":
        return Candidate(**{**self.__dict__, "topn_index": int(topn_index)})


@dataclass(frozen=True)
class CandidateBatch:
    state: StateView
    candidates: tuple[Candidate, ...]
    topn: tuple[Candidate, ...]
    candidate_mask: np.ndarray
    node_features: np.ndarray
    link_features: np.ndarray
    global_features: np.ndarray
    request_features: np.ndarray
    spectrum_tensors: np.ndarray
    action_features: np.ndarray

    @property
    def has_real_candidates(self) -> bool:
        return bool(np.any(self.candidate_mask))


def candidate_starts(mask: np.ndarray, width: int, random_count: int, rng: np.random.Generator) -> list[int]:
    segments = [(start, end) for start, end in contiguous_segments(mask) if end - start >= width]
    if not segments:
        return []
    starts: set[int] = {segments[0][0], segments[-1][1] - width}
    best_start, _ = min(segments, key=lambda item: (item[1] - item[0] - width, item[0]))
    starts.add(best_start)

    all_starts: list[int] = []
    for start, end in segments:
        all_starts.extend(range(start, end - width + 1))
    if random_count > 0 and all_starts:
        sample_size = min(random_count, len(all_starts))
        sampled = rng.choice(np.asarray(all_starts, dtype=np.int64), size=sample_size, replace=False)
        starts.update(int(value) for value in np.atleast_1d(sampled))
    return sorted(starts)


def route_availability(occupancy: np.ndarray, route_link_ids: tuple[int, ...] | list[int]) -> np.ndarray:
    if not route_link_ids:
        return np.ones(occupancy.shape[1], dtype=np.uint8)
    return (occupancy[np.asarray(route_link_ids, dtype=np.int64), :].sum(axis=0) == 0).astype(np.uint8)


def route_release_times(release_times: np.ndarray, route_link_ids: tuple[int, ...] | list[int], now: float) -> np.ndarray:
    if not route_link_ids:
        return np.zeros(release_times.shape[1], dtype=np.float32)
    route_release = release_times[np.asarray(route_link_ids, dtype=np.int64), :]
    residual = np.maximum(route_release - float(now), 0.0)
    return residual.max(axis=0).astype(np.float32)


def selected_block_distance(slots: int, b_start: int, width: int) -> np.ndarray:
    distance = np.zeros(slots, dtype=np.float32)
    block_end = b_start + width - 1
    for slot in range(slots):
        if b_start <= slot <= block_end:
            distance[slot] = 0.0
        elif slot < b_start:
            distance[slot] = float(b_start - slot) / float(slots)
        else:
            distance[slot] = float(slot - block_end) / float(slots)
    return distance


def route_slot_tensor(state: StateView, candidate: Candidate, cfg: SolverConfig) -> np.ndarray:
    slots = state.slot_count
    availability = route_availability(state.occupancy, candidate.route_link_ids)
    selected = np.zeros(slots, dtype=np.float32)
    selected[candidate.b_start : candidate.b_start + candidate.w] = 1.0
    future = route_release_times(state.release_times, candidate.route_link_ids, state.current_time)
    future_scale = max(cfg.mean_holding_time * 3.0, 1e-9)
    future_norm = np.clip(future / future_scale, 0.0, 1.0)
    occ_frac = route_occupancy_fraction(state.occupancy, list(candidate.route_link_ids))
    distance = selected_block_distance(slots, candidate.b_start, candidate.w)
    local_frag = np.zeros(slots, dtype=np.float32)
    for start, end in contiguous_segments(availability):
        local_frag[start:end] = 1.0 - float(end - start) / float(slots)
    return np.stack(
        [
            availability.astype(np.float32),
            selected,
            local_frag,
            future_norm.astype(np.float32),
            occ_frac.astype(np.float32),
            distance,
        ],
        axis=0,
    ).astype(np.float32)


def _small_gap_penalty(left_gap: int, right_gap: int, w_min: int) -> float:
    penalty = 0.0
    if 0 < left_gap < w_min:
        penalty += (w_min - left_gap) / float(w_min)
    if 0 < right_gap < w_min:
        penalty += (w_min - right_gap) / float(w_min)
    return float(penalty)


def _segment_for_start(mask: np.ndarray, start: int, width: int) -> tuple[int, int]:
    for seg_start, seg_end in contiguous_segments(mask):
        if seg_start <= start and start + width <= seg_end:
            return int(seg_start), int(seg_end)
    raise ValueError("candidate start is not inside a feasible segment")


def score_candidate(
    *,
    action: Any,
    route_id: int,
    modulation_index: int,
    modulation_offset: int,
    b_start: int,
    width: int,
    route_node_ids: tuple[Any, ...],
    route_link_ids: tuple[int, ...],
    route_length_km: float,
    hop_count: int,
    spectral_efficiency: float,
    modulation_name: str,
    modulation_reach_km: float,
    transponder_power_w: float,
    state: StateView,
    cfg: SolverConfig,
) -> Candidate:
    slots = state.slot_count
    availability = route_availability(state.occupancy, route_link_ids)
    after = allocate_on_mask(availability, b_start, width)
    frag_before = fragmentation(availability)
    frag_after = fragmentation(after)
    seg_start, seg_end = _segment_for_start(availability, b_start, width)
    left_gap = b_start - seg_start
    right_gap = seg_end - (b_start + width)
    left_neighbor_occupied = 1.0 if b_start == 0 or after[b_start - 1] == 0 else 0.0
    right_neighbor_occupied = 1.0 if b_start + width == slots or after[b_start + width] == 0 else 0.0
    compactness = 0.5 * left_neighbor_occupied + 0.5 * right_neighbor_occupied
    small_gap = _small_gap_penalty(left_gap, right_gap, cfg.min_useful_slot_width)
    lmax_after = largest_free_block(after)

    link_lengths = state.edge_lengths_km[np.asarray(route_link_ids, dtype=np.int64)] if route_link_ids else np.asarray([], dtype=np.float32)
    amplifier_count = np.ceil(link_lengths / max(cfg.span_length_km, 1e-9)).sum() if link_lengths.size else 0.0
    energy_increment = float(transponder_power_w + amplifier_count * cfg.amplifier_power_w + hop_count * cfg.roadm_switch_power_w)
    energy_norm = float(energy_increment / max(cfg.energy_norm_w, 1e-9))
    delay_ms = float(route_length_km / 200.0)
    delay_norm = float(delay_ms / max(cfg.delay_bound_ms, 1e-9))
    qot_margin = max(float(modulation_reach_km - route_length_km), 0.0)
    qot_margin_norm = float(qot_margin / max(modulation_reach_km, 1e-9))
    qot_risk = float(max(0.0, route_length_km / max(modulation_reach_km, 1e-9)))
    residual_fit_norm = float(left_gap + right_gap) / float(slots)
    width_norm = float(width) / float(slots)
    lmax_after_norm = float(lmax_after) / float(slots)

    scoring = cfg.scoring_weights
    j_frag = float(
        scoring["lambda_frag"] * (frag_after - frag_before)
        + scoring["lambda_gap"] * small_gap
        - scoring["lambda_lmax"] * lmax_after_norm
        + scoring["lambda_fit"] * residual_fit_norm
    )
    j_tie = float(
        scoring["mu_energy"] * energy_norm
        + scoring["mu_delay"] * delay_norm
        - scoring["mu_qot"] * qot_margin_norm
        + scoring["mu_width"] * width_norm
    )
    j_total = float(j_frag + 0.10 * j_tie)
    q_weights = cfg.q_head_weights
    q_head_score = float(
        -float(q_weights.get("energy", q_weights.get("c1", 0.25))) * energy_norm
        - float(q_weights.get("fragmentation_after", q_weights.get("c2", 0.35))) * frag_after
        - float(q_weights.get("qot_risk", q_weights.get("c3", 0.1))) * qot_risk
        - float(q_weights.get("delay", q_weights.get("c4", 0.1))) * delay_norm
        + float(q_weights.get("qot_margin", q_weights.get("c5", 0.1))) * qot_margin_norm
        + float(q_weights.get("largest_free_block", q_weights.get("c6", 0.1))) * lmax_after_norm
        - float(q_weights.get("delta_fragmentation", 0.0)) * (frag_after - frag_before)
        - float(q_weights.get("small_gap", 0.0)) * small_gap
        + float(q_weights.get("compactness", 0.0)) * compactness
    )
    action_features = (
        float(route_length_km / max(cfg.max_route_length_norm_km, 1e-9)),
        float(hop_count / 8.0),
        float(b_start / max(slots - 1, 1)),
        width_norm,
        qot_margin_norm,
        delay_norm,
        energy_norm,
        float(frag_after),
        float(lmax_after_norm),
        float(small_gap),
    )
    return Candidate(
        action=action,
        route_id=int(route_id),
        modulation_index=int(modulation_index),
        modulation_offset=int(modulation_offset),
        b_start=int(b_start),
        w=int(width),
        route_node_ids=tuple(route_node_ids),
        route_link_ids=tuple(int(link_id) for link_id in route_link_ids),
        route_length_km=float(route_length_km),
        hop_count=int(hop_count),
        delay_ms=delay_ms,
        modulation_name=str(modulation_name),
        spectral_efficiency=float(spectral_efficiency),
        qot_margin_norm=qot_margin_norm,
        qot_risk=qot_risk,
        energy_increment=energy_increment,
        energy_increment_norm=energy_norm,
        fragmentation_before=float(frag_before),
        fragmentation_after=float(frag_after),
        delta_fragmentation=float(frag_after - frag_before),
        largest_free_block_after=int(lmax_after),
        left_gap_after=int(left_gap),
        right_gap_after=int(right_gap),
        small_gap_penalty=float(small_gap),
        compactness=float(compactness),
        j_frag=j_frag,
        j_tie=j_tie,
        j_total=j_total,
        q_head_score=q_head_score,
        action_features=action_features,
    )


def build_state_features(state: StateView, cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slots = state.slot_count
    node_count = len(state.node_names)
    node_index = {node: index for index, node in enumerate(state.node_names)}
    degrees = np.zeros(node_count, dtype=np.float32)
    for src_idx, dst_idx in state.edge_index.T:
        degrees[int(src_idx)] += 1.0
        degrees[int(dst_idx)] += 1.0
    max_degree = max(float(degrees.max()), 1.0)
    node_features = np.zeros((node_count, 4), dtype=np.float32)
    src_idx = node_index.get(state.src)
    dst_idx = node_index.get(state.dst)
    if src_idx is not None:
        node_features[src_idx, 0] = 1.0
    if dst_idx is not None:
        node_features[dst_idx, 1] = 1.0
    node_features[:, 2] = degrees / max_degree
    node_features[:, 3] = np.clip(state.active_node_counts.astype(np.float32) / float(slots), 0.0, 1.0)

    max_length = max(float(state.edge_lengths_km.max(initial=1.0)), 1.0)
    delays = state.edge_lengths_km.astype(np.float32) / 200.0
    max_delay = max(float(delays.max(initial=1.0)), 1.0)
    energy_indicator = np.ceil(state.edge_lengths_km / max(cfg.span_length_km, 1e-9)) * cfg.amplifier_power_w
    max_energy = max(float(energy_indicator.max(initial=1.0)), 1.0)
    link_features = np.zeros((state.link_count, 8), dtype=np.float32)
    link_frags: list[float] = []
    link_lmax: list[float] = []
    for link_id in range(state.link_count):
        free_mask = (state.occupancy[link_id] == 0).astype(np.uint8)
        lmax = largest_free_block(free_mask)
        frag = fragmentation(free_mask)
        link_lmax.append(float(lmax) / float(slots))
        link_frags.append(frag)
        link_features[link_id, 0] = float(state.edge_lengths_km[link_id]) / max_length
        link_features[link_id, 1] = float(delays[link_id]) / max_delay
        link_features[link_id, 2] = float(state.occupancy[link_id].mean())
        link_features[link_id, 3] = float(lmax) / float(slots)
        link_features[link_id, 4] = frag
        link_features[link_id, 5] = min(float(state.active_link_counts[link_id]) / float(slots), 1.0)
        link_features[link_id, 6] = float(energy_indicator[link_id]) / max_energy
        link_features[link_id, 7] = min(float(state.edge_lengths_km[link_id]) / 4000.0, 1.0)

    occupancy_ratio = state.occupancy.mean(axis=1) if state.link_count else np.zeros(1, dtype=np.float32)
    global_features = np.asarray(
        [
            float(occupancy_ratio.mean()),
            float(occupancy_ratio.max(initial=0.0)),
            float(np.mean(link_frags) if link_frags else 0.0),
            float(np.max(link_frags) if link_frags else 0.0),
            float(np.mean(link_lmax) if link_lmax else 1.0),
            float(np.min(link_lmax) if link_lmax else 1.0),
            min(float(state.active_link_counts.sum()) / float(max(state.link_count * slots, 1)), 1.0),
            float((state.occupancy * energy_indicator[:, None]).sum() / max(state.link_count * slots * max_energy, 1.0)),
        ],
        dtype=np.float32,
    )
    best_efficiency = 4.0
    min_required = int(np.ceil(float(state.bit_rate_gbps) / (cfg.slot_capacity_gbps_at_1bpshz * best_efficiency))) + cfg.guard_band_slots
    max_bit_rate = max(float(state.bit_rate_gbps), 400.0)
    request_features = np.asarray(
        [
            float(state.bit_rate_gbps) / max_bit_rate,
            min(float(state.holding_time) / max(cfg.mean_holding_time * 3.0, 1e-9), 1.0),
            min(float(min_required) / float(slots), 1.0),
        ],
        dtype=np.float32,
    )
    return node_features, link_features, global_features, request_features


def build_candidate_batch(state: StateView, candidates: list[Candidate], cfg: SolverConfig) -> CandidateBatch:
    ordered = sorted(candidates, key=lambda row: (float(row.j_total), float(row.energy_increment), int(row.route_id), int(row.b_start)))
    topn = tuple(candidate.with_topn_index(index) for index, candidate in enumerate(ordered[: cfg.n_max]))
    mask = np.zeros(cfg.n_max, dtype=np.float32)
    mask[: len(topn)] = 1.0
    node_features, link_features, global_features, request_features = build_state_features(state, cfg)
    spectrum_tensors = np.zeros((cfg.n_max, 6, state.slot_count), dtype=np.float32)
    action_feature_dim = len(topn[0].action_features) if topn else 10
    action_features = np.zeros((cfg.n_max, action_feature_dim), dtype=np.float32)
    for index, candidate in enumerate(topn):
        spectrum_tensors[index] = route_slot_tensor(state, candidate, cfg)
        action_features[index] = np.asarray(candidate.action_features, dtype=np.float32)
    return CandidateBatch(
        state=state,
        candidates=tuple(ordered),
        topn=topn,
        candidate_mask=mask,
        node_features=node_features,
        link_features=link_features,
        global_features=global_features,
        request_features=request_features,
        spectrum_tensors=spectrum_tensors,
        action_features=action_features,
    )


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    valid = np.asarray(mask, dtype=bool)
    if not valid.any():
        raise ValueError("masked_argmax requires at least one valid item")
    masked = np.asarray(values, dtype=np.float64).copy()
    masked[~valid] = -np.inf
    return int(np.argmax(masked))


def pad_q_scores(scores: np.ndarray, n_max: int) -> np.ndarray:
    out = np.full(n_max, -np.inf, dtype=np.float32)
    score_values = np.asarray(scores, dtype=np.float32)
    out[: min(n_max, len(score_values))] = score_values[:n_max]
    return out
