from __future__ import annotations

from typing import Any

import numpy as np

from .candidates import group_candidates_by_route_mod
from .io_utils import stringify
from .spectrum import future_release_for_route, local_fragmentation_context, route_availability, route_occupancy_fraction


def tensor_for_candidate(
    *,
    candidate: dict[str, Any],
    occupancy: np.ndarray,
    release_times: np.ndarray,
    now: float,
    cfg: dict[str, Any],
) -> np.ndarray:
    slots = int(cfg.get("slots", 100))
    route_link_ids = [int(x) for x in candidate["route_directed_link_ids"]]
    availability = route_availability(occupancy, route_link_ids)
    selected = np.zeros(slots, dtype=np.float32)
    b_start = int(candidate["b_start"])
    width = int(candidate["w"])
    selected[b_start : b_start + width] = 1.0
    future = future_release_for_route(release_times, route_link_ids, now)
    future_norm = np.clip(future / max(float(cfg.get("mean_holding_time", 14.0)) * 3.0, 1e-9), 0.0, 1.0)
    occ_frac = route_occupancy_fraction(occupancy, route_link_ids)
    distance = np.zeros(slots, dtype=np.float32)
    for idx in range(slots):
        if b_start <= idx < b_start + width:
            distance[idx] = 0.0
        elif idx < b_start:
            distance[idx] = float(b_start - idx) / float(slots)
        else:
            distance[idx] = float(idx - (b_start + width - 1)) / float(slots)
    return np.stack(
        [
            availability.astype(np.float32),
            selected,
            local_fragmentation_context(availability),
            future_norm.astype(np.float32),
            occ_frac.astype(np.float32),
            distance,
        ],
        axis=0,
    ).astype(np.float32)


def select_cnn_candidates(
    candidates: list[dict[str, Any]],
    rng: np.random.Generator,
    max_samples_per_request: int,
) -> list[dict[str, Any]]:
    groups = group_candidates_by_route_mod(candidates)
    ordered_groups = sorted(groups.values(), key=lambda rows: min(float(row["j_total"]) for row in rows))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for rows in ordered_groups:
        ordered = sorted(rows, key=lambda row: float(row["j_total"]))
        choices = [ordered[0]]
        if len(ordered) > 1:
            choices.append(ordered[int(rng.integers(0, len(ordered)))])
            choices.append(max(ordered, key=lambda row: float(row["delta_fragmentation"])))
        for row in choices:
            key = (int(row["route_id"]), int(row["modulation_id"]), int(row["b_start"]), int(row["w"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= max_samples_per_request:
                return selected
    return selected


def cnn_index_row(
    *,
    sample_id: int,
    candidate: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    created_small_gap = int(float(candidate["small_gap_penalty"]) > 0.0)
    group_id = f"{candidate['episode_id']}:{candidate['request_id']}:r{candidate['route_id']}:m{candidate['modulation_id']}"
    return {
        "sample_id": int(sample_id),
        "episode_id": candidate["episode_id"],
        "request_id": int(candidate["request_id"]),
        "route_id": int(candidate["route_id"]),
        "modulation_id": int(candidate["modulation_id"]),
        "b_start": int(candidate["b_start"]),
        "w": int(candidate["w"]),
        "route_directed_link_ids": stringify(candidate["route_directed_link_ids"]),
        "delta_frag": float(candidate["delta_fragmentation"]),
        "frag_after": float(candidate["fragmentation_after"]),
        "lmax_after_norm": float(candidate["largest_free_block_after"]) / 100.0,
        "nseg_after_norm": float(candidate["n_segments_after"]) / 100.0,
        "created_small_gap": created_small_gap,
        "compactness": float(candidate["compactness"]),
        "placement_score": -float(candidate["j_total"]),
        "J_total": float(candidate["j_total"]),
        "group_id": group_id,
        "split": request["split"],
        "seed": int(request["seed"]),
        "traffic_scenario": request["traffic_scenario"],
        "load_name": request["load_name"],
    }

