from __future__ import annotations

from typing import Any

import numpy as np

from .modulation import required_slots
from .spectrum import fragmentation, largest_free_block, route_availability, route_occupancy_fraction
from .topology import Topology


def degree_by_node(topology: Topology) -> dict[int, int]:
    degree = {int(node): 0 for node in topology.nodes["node_id"].tolist()}
    for row in topology.undirected_links.itertuples(index=False):
        degree[int(row.u)] += 1
        degree[int(row.v)] += 1
    return degree


def graph_features(
    *,
    topology: Topology,
    occupancy: np.ndarray,
    active_link_counts: np.ndarray,
    active_node_counts: np.ndarray,
    request: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slots = int(cfg.get("slots", 100))
    degrees = degree_by_node(topology)
    max_degree = max(degrees.values())
    node_features = np.zeros((topology.node_count, 4), dtype=np.float32)
    for row in topology.nodes.itertuples(index=False):
        node_id = int(row.node_id)
        idx = node_id - 1
        node_features[idx, 0] = 1.0 if node_id == int(request["src"]) else 0.0
        node_features[idx, 1] = 1.0 if node_id == int(request["dst"]) else 0.0
        node_features[idx, 2] = degrees[node_id] / float(max_degree)
        node_features[idx, 3] = min(float(active_node_counts[idx]) / float(slots), 1.0)

    directed = topology.directed_links
    max_length = float(directed["length_km"].max())
    max_delay = float(directed["delay_ms"].max())
    max_energy = float(directed["base_energy_cost"].max())
    link_features = np.zeros((topology.directed_link_count, 8), dtype=np.float32)
    link_lmax: list[float] = []
    link_frags: list[float] = []
    for row in directed.itertuples(index=False):
        link_id = int(row.directed_link_id)
        availability = (occupancy[link_id] == 0).astype(np.uint8)
        lmax = largest_free_block(availability)
        frag = fragmentation(availability)
        link_lmax.append(float(lmax) / float(slots))
        link_frags.append(frag)
        link_features[link_id, 0] = float(row.length_km) / max_length
        link_features[link_id, 1] = float(row.delay_ms) / max_delay
        link_features[link_id, 2] = float(occupancy[link_id].mean())
        link_features[link_id, 3] = float(lmax) / float(slots)
        link_features[link_id, 4] = frag
        link_features[link_id, 5] = min(float(active_link_counts[link_id]) / float(slots), 1.0)
        link_features[link_id, 6] = float(row.base_energy_cost) / max_energy
        link_features[link_id, 7] = float(row.base_qot_risk)

    occupancy_ratio = occupancy.mean(axis=1)
    total_active = float(active_link_counts.sum())
    global_features = np.asarray(
        [
            float(occupancy_ratio.mean()),
            float(occupancy_ratio.max()),
            float(np.mean(link_frags)),
            float(np.max(link_frags)),
            float(np.mean(link_lmax)),
            float(np.min(link_lmax)),
            min(total_active / float(topology.directed_link_count * slots), 1.0),
            float((occupancy * directed["base_energy_cost"].to_numpy()[:, None]).sum() / max(1.0, topology.directed_link_count * slots * max_energy)),
        ],
        dtype=np.float32,
    )

    max_bit_rate = max(float(x) for x in cfg.get("bit_rates", {400: 1.0}).keys())
    best_eff = 4.0
    min_required = int(np.ceil(float(request["bit_rate_gbps"]) / (float(cfg.get("slot_capacity_gbps_at_1bpshz", 12.5)) * best_eff))) + int(
        cfg.get("guard_band_slots", 1)
    )
    request_features = np.asarray(
        [
            float(request["bit_rate_gbps"]) / max_bit_rate,
            min(float(request["holding_time"]) / max(float(cfg.get("mean_holding_time", 14.0)) * 3.0, 1e-9), 1.0),
            min(float(min_required) / float(slots), 1.0),
        ],
        dtype=np.float32,
    )
    return node_features, link_features, global_features, request_features


def route_label_row(
    *,
    sample_id: str,
    request: dict[str, Any],
    route,
    modulation,
    occupancy: np.ndarray,
    summary: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    availability = route_availability(occupancy, route.directed_link_ids)
    occ_frac = route_occupancy_fraction(occupancy, route.directed_link_ids)
    slot_capacity = float(cfg.get("slot_capacity_gbps_at_1bpshz", 12.5))
    guard = int(cfg.get("guard_band_slots", 1))
    width = required_slots(float(request["bit_rate_gbps"]), modulation, slot_capacity, guard)
    qot_margin = float(modulation.reach_km - route.length_km)
    return {
        "sample_id": sample_id,
        "episode_id": request["episode_id"],
        "request_id": int(request["request_id"]),
        "route_candidate_id": f"{sample_id}:r{route.route_id}:m{modulation.modulation_id}",
        "route_id": int(route.route_id),
        "route_node_ids": route.node_ids,
        "route_directed_link_ids": route.directed_link_ids,
        "modulation_id": int(modulation.modulation_id),
        "route_length_norm": float(route.length_km) / 6000.0,
        "hop_count_norm": float(route.hop_count) / 8.0,
        "required_slots_norm": float(width) / float(cfg.get("slots", 100)),
        "delay_norm": float(route.delay_ms) / max(float(cfg.get("delay_bound_ms", 50.0)), 1e-9),
        "energy_norm": float(route.hop_count) / 8.0,
        "mean_occupancy": float(occ_frac.mean()),
        "max_occupancy": float(occ_frac.max()),
        "route_fragmentation": fragmentation(availability),
        "c_route_max_norm": float(largest_free_block(availability)) / float(cfg.get("slots", 100)),
        "qot_margin_norm": max(qot_margin, 0.0) / modulation.reach_km,
        "qot_risk": max(0.0, float(route.length_km) / modulation.reach_km),
        "feasible_label": int(summary.get("feasible_label", 0)),
        "heuristic_route_score": float(summary.get("heuristic_route_score", 999.0)),
        "block_now": 0,
        "num_feasible": 0,
        "num_feasible_norm": 0.0,
        "global_fragmentation": 0.0,
        "split": request["split"],
        "seed": int(request["seed"]),
        "traffic_scenario": request["traffic_scenario"],
        "load_name": request["load_name"],
    }

