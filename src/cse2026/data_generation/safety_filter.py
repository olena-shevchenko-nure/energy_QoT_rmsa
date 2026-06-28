from __future__ import annotations

from typing import Any

import numpy as np

from .modulation import Modulation


def is_spectrum_contiguous(candidate: dict[str, Any]) -> bool:
    return int(candidate["w"]) > 0 and int(candidate["b_start"]) >= 0


def is_spectrum_continuous(occupancy: np.ndarray, route_link_ids: list[int], b_start: int, width: int) -> bool:
    if not route_link_ids:
        return True
    block = occupancy[np.asarray(route_link_ids, dtype=np.int64), b_start : b_start + width]
    return bool(block.size > 0 and int(block.sum()) == 0)


def candidate_is_feasible(
    candidate: dict[str, Any],
    occupancy: np.ndarray,
    modulation: Modulation,
    delay_bound_ms: float,
) -> bool:
    if not is_spectrum_contiguous(candidate):
        return False
    link_ids = [int(x) for x in candidate["route_directed_link_ids"]]
    b_start = int(candidate["b_start"])
    width = int(candidate["w"])
    if b_start + width > occupancy.shape[1]:
        return False
    if not is_spectrum_continuous(occupancy, link_ids, b_start, width):
        return False
    if float(candidate["route_length_km"]) > modulation.reach_km:
        return False
    if float(candidate["qot_margin"]) < 0.0:
        return False
    return float(candidate["delay_ms"]) <= float(delay_bound_ms)


def make_padding_candidate(
    *,
    episode_id: str,
    request_id: int,
    topn_index: int,
    n_max: int,
    split: str,
    seed: int,
    traffic_scenario: str,
    load_name: str,
) -> dict[str, Any]:
    return {
        "candidate_id": -1,
        "episode_id": episode_id,
        "request_id": int(request_id),
        "topn_index": int(topn_index),
        "route_id": -1,
        "route_node_ids": "[]",
        "route_directed_link_ids": "[]",
        "modulation_id": -1,
        "b_start": -1,
        "w": 0,
        "required_slots": 0,
        "route_length_km": 0.0,
        "hop_count": 0,
        "delay_ms": 0.0,
        "energy_increment": 0.0,
        "qot_margin": 0.0,
        "qot_risk": 0.0,
        "fragmentation_before": 0.0,
        "fragmentation_after": 0.0,
        "delta_fragmentation": 0.0,
        "largest_free_block_before": 0,
        "largest_free_block_after": 0,
        "n_free_before": 0,
        "n_free_after": 0,
        "n_segments_before": 0,
        "n_segments_after": 0,
        "small_gap_penalty": 0.0,
        "compactness": 0.0,
        "j_frag": 0.0,
        "j_tie": 0.0,
        "j_total": 0.0,
        "q_head_score": 0.0,
        "is_feasible": False,
        "in_topn": False,
        "candidate_mask": 0,
        "n_max": int(n_max),
        "split": split,
        "seed": int(seed),
        "traffic_scenario": traffic_scenario,
        "load_name": load_name,
    }

