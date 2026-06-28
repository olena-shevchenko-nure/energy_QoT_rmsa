from __future__ import annotations

from typing import Any

import numpy as np

from .io_utils import stable_seed


def normalize_probs(weights: list[float]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probability weights must sum to a positive value")
    return values / total


def ordered_pairs(node_count: int = 14) -> list[tuple[int, int]]:
    return [(src, dst) for src in range(1, node_count + 1) for dst in range(1, node_count + 1) if src != dst]


def od_distribution(scenario: str, cfg: dict[str, Any], node_count: int = 14) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs = ordered_pairs(node_count)
    if scenario == "uniform" or scenario == "bursty":
        return pairs, normalize_probs([1.0] * len(pairs))

    if scenario == "hotspot":
        hotspots = set(int(node) for node in cfg.get("hotspot_nodes", [8, 9, 12, 14]))
        weights = []
        for src, dst in pairs:
            weight = 1.0
            if src in hotspots:
                weight += 3.0
            if dst in hotspots:
                weight += 3.0
            if src in hotspots and dst in hotspots:
                weight += 2.0
            weights.append(weight)
        return pairs, normalize_probs(weights)

    if scenario == "nonuniform":
        high = {(1, 14), (14, 1), (2, 9), (9, 2), (8, 12), (12, 8), (3, 11), (11, 3), (5, 13), (13, 5)}
        medium = {(1, 8), (8, 1), (6, 14), (14, 6), (9, 13), (13, 9)}
        weights = [8.0 if pair in high else 4.0 if pair in medium else 1.0 for pair in pairs]
        return pairs, normalize_probs(weights)

    raise ValueError(f"unknown traffic scenario: {scenario}")


def bit_rate_distribution(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw = cfg.get("bit_rates", {100: 0.45, 200: 0.35, 400: 0.20})
    items = sorted((float(rate), float(weight)) for rate, weight in raw.items() if float(weight) > 0)
    rates = np.asarray([rate for rate, _ in items], dtype=np.float64)
    probs = normalize_probs([weight for _, weight in items])
    return rates, probs


def arrival_rate_for(load_name: str, cfg: dict[str, Any]) -> float:
    profiles = cfg.get("load_profiles", {"low": 3.0, "medium": 5.0, "high": 8.0, "overload": 12.0})
    return float(profiles[load_name])


def burst_multiplier(request_idx: int) -> float:
    phase = (request_idx // 75) % 4
    return [0.55, 1.8, 0.75, 2.25][phase]


def generate_requests_for_episode(
    *,
    split: str,
    seed: int,
    scenario: str,
    load_name: str,
    requests_per_seed: int,
    cfg: dict[str, Any],
    episode_id: str,
    node_count: int = 14,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(stable_seed("traffic", split, seed, scenario, load_name))
    pairs, pair_probs = od_distribution(scenario, cfg, node_count=node_count)
    rates, rate_probs = bit_rate_distribution(cfg)
    mean_holding = float(cfg.get("mean_holding_time", 14.0))
    base_lambda = arrival_rate_for(load_name, cfg)
    now = 0.0
    rows: list[dict[str, Any]] = []
    for request_id in range(int(requests_per_seed)):
        lam = base_lambda * burst_multiplier(request_id) if scenario == "bursty" else base_lambda
        interarrival = float(rng.exponential(1.0 / lam))
        holding = float(rng.exponential(mean_holding))
        now += interarrival
        pair_idx = int(rng.choice(len(pairs), p=pair_probs))
        bit_rate = float(rng.choice(rates, p=rate_probs))
        src, dst = pairs[pair_idx]
        rows.append(
            {
                "episode_id": episode_id,
                "request_id": request_id,
                "seed": int(seed),
                "split": split,
                "traffic_scenario": scenario,
                "load_name": load_name,
                "arrival_time": now,
                "departure_time": now + holding,
                "src": int(src),
                "dst": int(dst),
                "bit_rate_gbps": bit_rate,
                "holding_time": holding,
                "interarrival_time": interarrival,
            }
        )
    return rows

