from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .io_utils import stringify
from .modulation import Modulation, required_slots
from .routing import Route
from .safety_filter import candidate_is_feasible
from .spectrum import (
    allocate_on_mask,
    contiguous_segments,
    fragmentation,
    free_count,
    largest_free_block,
    route_availability,
    segment_count,
)
from .topology import Topology


def _segment_for_start(mask: np.ndarray, start: int, width: int) -> tuple[int, int]:
    for seg_start, seg_end in contiguous_segments(mask):
        if seg_start <= start and start + width <= seg_end:
            return seg_start, seg_end
    raise ValueError("candidate start is not inside a feasible segment")


def _candidate_starts(mask: np.ndarray, width: int, random_count: int, rng: np.random.Generator) -> list[int]:
    segments = [(start, end) for start, end in contiguous_segments(mask) if end - start >= width]
    if not segments:
        return []
    starts: set[int] = set()
    starts.add(segments[0][0])
    best_start, best_end = min(segments, key=lambda item: (item[1] - item[0] - width, item[0]))
    starts.add(best_start)
    starts.add(segments[-1][1] - width)

    all_starts: list[int] = []
    for start, end in segments:
        all_starts.extend(range(start, end - width + 1))
    if all_starts and random_count > 0:
        replace = len(all_starts) < random_count
        sampled = rng.choice(np.asarray(all_starts, dtype=np.int64), size=min(random_count, len(all_starts)) if not replace else random_count, replace=replace)
        starts.update(int(x) for x in np.atleast_1d(sampled))
    return sorted(starts)


def _small_gap_penalty(left_gap: int, right_gap: int, w_min: int) -> float:
    penalty = 0.0
    if 0 < left_gap < w_min:
        penalty += (w_min - left_gap) / float(w_min)
    if 0 < right_gap < w_min:
        penalty += (w_min - right_gap) / float(w_min)
    return float(penalty)


def _q_head_score(candidate: dict[str, Any], cfg: dict[str, Any]) -> float:
    weights = cfg.get("q_head", {})
    energy_norm = float(candidate["energy_increment_norm"])
    frag_after = float(candidate["fragmentation_after"])
    qot_risk = float(candidate["qot_risk"])
    delay_norm = float(candidate["delay_norm"])
    qot_margin_norm = float(candidate["qot_margin_norm"])
    lmax_after_norm = float(candidate["largest_free_block_after"]) / float(cfg.get("slots", 100))
    delta_fragmentation = float(candidate["delta_fragmentation"])
    small_gap = float(candidate["small_gap_penalty"])
    compactness = float(candidate["compactness"])
    return float(
        -float(weights.get("energy", weights.get("c1", 0.25))) * energy_norm
        - float(weights.get("fragmentation_after", weights.get("c2", 0.35))) * frag_after
        - float(weights.get("qot_risk", weights.get("c3", 0.1))) * qot_risk
        - float(weights.get("delay", weights.get("c4", 0.1))) * delay_norm
        + float(weights.get("qot_margin", weights.get("c5", 0.1))) * qot_margin_norm
        + float(weights.get("largest_free_block", weights.get("c6", 0.1))) * lmax_after_norm
        - float(weights.get("delta_fragmentation", 0.0)) * delta_fragmentation
        - float(weights.get("small_gap", 0.0)) * small_gap
        + float(weights.get("compactness", 0.0)) * compactness
    )


def _candidate_from_start(
    *,
    candidate_id: int,
    request: dict[str, Any],
    route: Route,
    modulation: Modulation,
    b_start: int,
    width: int,
    availability: np.ndarray,
    cfg: dict[str, Any],
    topology: Topology,
) -> dict[str, Any]:
    slots = int(cfg.get("slots", 100))
    scoring = cfg.get("scoring", {})
    w_min = int(cfg.get("min_useful_slot_width", 4))
    amplifier_power = float(cfg.get("amplifier_power_w", 20.0))
    roadm_power = float(cfg.get("roadm_switch_power_w", 5.0))
    delay_bound = float(cfg.get("delay_bound_ms", 50.0))
    energy_norm_w = float(scoring.get("energy_norm_w", 1200.0))

    before = availability.astype(np.uint8)
    after = allocate_on_mask(before, b_start, width)
    frag_before = fragmentation(before)
    frag_after = fragmentation(after)
    seg_start, seg_end = _segment_for_start(before, b_start, width)
    left_gap = b_start - seg_start
    right_gap = seg_end - (b_start + width)
    residual_fit_norm = float(left_gap + right_gap) / float(slots)
    small_gap = _small_gap_penalty(left_gap, right_gap, w_min)
    left_neighbor_occupied = 1.0 if b_start == 0 or after[b_start - 1] == 0 else 0.0
    right_neighbor_occupied = 1.0 if b_start + width == slots or after[b_start + width] == 0 else 0.0
    compactness = 0.5 * left_neighbor_occupied + 0.5 * right_neighbor_occupied

    directed = topology.directed_links.set_index("directed_link_id")
    link_energy = sum(float(directed.loc[link_id, "amplifier_count"]) * amplifier_power for link_id in route.directed_link_ids)
    energy_increment = float(modulation.transponder_power_w + link_energy + route.hop_count * roadm_power)
    qot_margin = float(modulation.reach_km - route.length_km)
    qot_risk = float(max(0.0, route.length_km / modulation.reach_km))
    qot_margin_norm = float(max(qot_margin, 0.0) / modulation.reach_km)
    delay_norm = float(route.delay_ms / max(delay_bound, 1e-9))
    width_norm = float(width) / float(slots)
    energy_norm = float(energy_increment / max(energy_norm_w, 1e-9))
    lmax_after_norm = float(largest_free_block(after)) / float(slots)
    delta_fragmentation = float(frag_after - frag_before)

    j_frag = float(
        float(scoring.get("lambda_frag", 1.0)) * delta_fragmentation
        + float(scoring.get("lambda_gap", 0.6)) * small_gap
        - float(scoring.get("lambda_lmax", 0.4)) * lmax_after_norm
        + float(scoring.get("lambda_fit", 0.3)) * residual_fit_norm
    )
    j_tie = float(
        float(scoring.get("mu_energy", 0.25)) * energy_norm
        + float(scoring.get("mu_delay", 0.2)) * delay_norm
        - float(scoring.get("mu_qot", 0.2)) * qot_margin_norm
        + float(scoring.get("mu_width", 0.1)) * width_norm
    )

    row: dict[str, Any] = {
        "candidate_id": int(candidate_id),
        "episode_id": request["episode_id"],
        "request_id": int(request["request_id"]),
        "route_id": int(route.route_id),
        "route_node_ids": route.node_ids,
        "route_directed_link_ids": route.directed_link_ids,
        "modulation_id": int(modulation.modulation_id),
        "b_start": int(b_start),
        "w": int(width),
        "required_slots": int(width),
        "route_length_km": float(route.length_km),
        "hop_count": int(route.hop_count),
        "delay_ms": float(route.delay_ms),
        "energy_increment": energy_increment,
        "qot_margin": qot_margin,
        "qot_margin_norm": qot_margin_norm,
        "qot_risk": qot_risk,
        "fragmentation_before": float(frag_before),
        "fragmentation_after": float(frag_after),
        "delta_fragmentation": delta_fragmentation,
        "largest_free_block_before": int(largest_free_block(before)),
        "largest_free_block_after": int(largest_free_block(after)),
        "n_free_before": int(free_count(before)),
        "n_free_after": int(free_count(after)),
        "n_segments_before": int(segment_count(before)),
        "n_segments_after": int(segment_count(after)),
        "left_gap_after": int(left_gap),
        "right_gap_after": int(right_gap),
        "small_gap_penalty": small_gap,
        "compactness": float(compactness),
        "energy_increment_norm": energy_norm,
        "delay_norm": delay_norm,
        "w_norm": width_norm,
        "residual_fit_norm": residual_fit_norm,
        "j_frag": j_frag,
        "j_tie": j_tie,
        "j_total": float(j_frag + 0.10 * j_tie),
        "is_feasible": True,
        "in_topn": False,
        "candidate_mask": 0,
    }
    row["q_head_score"] = _q_head_score(row, cfg)
    return row


def generate_candidates_for_request(
    *,
    request: dict[str, Any],
    routes: list[Route],
    topology: Topology,
    modulations: list[Modulation],
    occupancy: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    slot_capacity = float(cfg.get("slot_capacity_gbps_at_1bpshz", 12.5))
    guard = int(cfg.get("guard_band_slots", 1))
    random_starts = int(cfg.get("random_starts_per_route_mod", 2))
    delay_bound = float(cfg.get("delay_bound_ms", 50.0))
    candidates: list[dict[str, Any]] = []
    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    candidate_id = 0

    for route in routes:
        availability = route_availability(occupancy, route.directed_link_ids)
        for modulation in modulations:
            width = required_slots(float(request["bit_rate_gbps"]), modulation, slot_capacity, guard)
            key = (route.route_id, modulation.modulation_id)
            summary = {
                "route_id": int(route.route_id),
                "modulation_id": int(modulation.modulation_id),
                "required_slots": int(width),
                "feasible_label": 0,
                "heuristic_route_score": 999.0,
                "num_placements": 0,
            }
            if route.length_km > modulation.reach_km or route.delay_ms > delay_bound or width > int(cfg.get("slots", 100)):
                summaries[key] = summary
                continue

            starts = _candidate_starts(availability, width, random_starts, rng)
            for start in starts:
                row = _candidate_from_start(
                    candidate_id=candidate_id,
                    request=request,
                    route=route,
                    modulation=modulation,
                    b_start=int(start),
                    width=int(width),
                    availability=availability,
                    cfg=cfg,
                    topology=topology,
                )
                if candidate_is_feasible(row, occupancy, modulation, delay_bound):
                    candidates.append(row)
                    candidate_id += 1
            route_mod_candidates = [row for row in candidates if row["route_id"] == route.route_id and row["modulation_id"] == modulation.modulation_id]
            if route_mod_candidates:
                summary["feasible_label"] = 1
                summary["heuristic_route_score"] = float(min(row["j_total"] for row in route_mod_candidates))
                summary["num_placements"] = len(route_mod_candidates)
            summaries[key] = summary

    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[dict[str, Any]] = []
    for row in candidates:
        key = (int(row["route_id"]), int(row["modulation_id"]), int(row["b_start"]), int(row["w"]))
        if key in seen:
            continue
        row["candidate_id"] = len(deduped)
        seen.add(key)
        deduped.append(row)

    for key, summary in summaries.items():
        route_mod_candidates = [row for row in deduped if (row["route_id"], row["modulation_id"]) == key]
        if route_mod_candidates:
            summary["feasible_label"] = 1
            summary["heuristic_route_score"] = float(min(row["j_total"] for row in route_mod_candidates))
            summary["num_placements"] = len(route_mod_candidates)
    return deduped, summaries


def sorted_feasible(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda row: (float(row["j_total"]), float(row["energy_increment"]), int(row["route_id"]), int(row["b_start"])))


def serialize_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if not isinstance(out.get("route_node_ids"), str):
        out["route_node_ids"] = stringify(out.get("route_node_ids", []))
    if not isinstance(out.get("route_directed_link_ids"), str):
        out["route_directed_link_ids"] = stringify(out.get("route_directed_link_ids", []))
    return out


def group_candidates_by_route_mod(candidates: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(int(row["route_id"]), int(row["modulation_id"]))].append(row)
    return groups
