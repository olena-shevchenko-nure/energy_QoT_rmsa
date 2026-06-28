from __future__ import annotations

from typing import Any

import numpy as np

from .io_utils import stringify


def choose_collection_action(
    topn_real_candidates: list[dict[str, Any]],
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not topn_real_candidates:
        return None
    policy = cfg.get("collection_policy", {})

    def expert_choice() -> dict[str, Any]:
        expert_metric = str(policy.get("expert_metric", "fragmentation_tuple"))
        if expert_metric == "q_head_score":
            return max(topn_real_candidates, key=lambda row: float(row["q_head_score"]))
        if expert_metric == "j_total":
            return min(topn_real_candidates, key=lambda row: float(row["j_total"]))
        return min(
            topn_real_candidates,
            key=lambda row: (
                float(row["fragmentation_after"]),
                float(row["small_gap_penalty"]),
                float(row["energy_increment"]),
                float(row["j_total"]),
            ),
        )

    draw = float(rng.random())
    expert_probability = float(policy.get("expert_probability", 0.70))
    softmax_probability = float(policy.get("softmax_probability", 0.20))
    if draw < expert_probability:
        return expert_choice()
    if draw < expert_probability + softmax_probability:
        top_k = int(policy.get("softmax_top_k", 8))
        tau = max(float(policy.get("softmax_tau", 0.35)), 1e-9)
        metric = str(policy.get("softmax_metric", "j_total"))
        maximize = bool(policy.get("softmax_maximize", False))
        pool = sorted(topn_real_candidates, key=lambda row: float(row[metric]), reverse=maximize)[:top_k]
        raw_scores = np.asarray([float(row[metric]) for row in pool], dtype=np.float64)
        scores = (raw_scores if maximize else -raw_scores) / tau
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum()
        return pool[int(rng.choice(len(pool), p=probs))]
    return topn_real_candidates[int(rng.integers(0, len(topn_real_candidates)))]


def reward_for(candidate: dict[str, Any] | None, cfg: dict[str, Any]) -> float:
    if candidate is None:
        return -5.0
    weights = cfg.get("reward", {})
    return float(
        1.0
        - float(weights.get("alpha_energy", 0.2)) * float(candidate["energy_increment_norm"])
        - float(weights.get("beta_fragmentation", 0.3)) * float(candidate["fragmentation_after"])
        - float(weights.get("kappa_qot", 0.1)) * float(candidate["qot_risk"])
        - float(weights.get("mu_delay", 0.1)) * float(candidate["delay_norm"])
    )


def dqn_transition_row(
    *,
    transition_id: int,
    request: dict[str, Any],
    state_id: str,
    next_state_id: str,
    selected: dict[str, Any] | None,
    topn_real_candidates: list[dict[str, Any]],
    full_candidate_count: int,
    n_max: int,
    done: bool,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    blocked = selected is None
    scores = [float(row["q_head_score"]) for row in topn_real_candidates]
    scores.extend([None] * (n_max - len(scores)))
    best_candidate_index = -1
    if topn_real_candidates:
        best = max(topn_real_candidates, key=lambda row: float(row["q_head_score"]))
        best_candidate_index = int(best["topn_index"])
    selected_index = -1 if selected is None else int(selected["topn_index"])
    return {
        "transition_id": int(transition_id),
        "episode_id": request["episode_id"],
        "request_id": int(request["request_id"]),
        "state_id": state_id,
        "next_state_id": next_state_id,
        "request_features": stringify(
            {
                "src": int(request["src"]),
                "dst": int(request["dst"]),
                "bit_rate_gbps": float(request["bit_rate_gbps"]),
                "holding_time": float(request["holding_time"]),
            }
        ),
        "selected_candidate_index": selected_index,
        "best_candidate_index": best_candidate_index,
        "q_head_scores": stringify(scores),
        "selected_action_description": stringify(
            {
                "route_node_ids": [] if selected is None else selected["route_node_ids"],
                "route_directed_link_ids": [] if selected is None else selected["route_directed_link_ids"],
                "modulation_id": -1 if selected is None else int(selected["modulation_id"]),
                "b_start": -1 if selected is None else int(selected["b_start"]),
                "w": 0 if selected is None else int(selected["w"]),
            }
        ),
        "reward": reward_for(selected, cfg),
        "done": bool(done),
        "blocked": bool(blocked),
        "blocking_reason": "no_feasible_candidate" if blocked else "",
        "delta_energy": 0.0 if selected is None else float(selected["energy_increment"]),
        "fragmentation_after": 1.0 if selected is None else float(selected["fragmentation_after"]),
        "qot_margin": 0.0 if selected is None else float(selected["qot_margin"]),
        "qot_risk": 1.0 if selected is None else float(selected["qot_risk"]),
        "delay_ms": 0.0 if selected is None else float(selected["delay_ms"]),
        "num_feasible_before_topn": int(full_candidate_count),
        "num_candidates_after_topn": int(len(topn_real_candidates)),
        "candidate_mask_valid": len(topn_real_candidates) <= n_max,
        "invalid_action_selected": False,
        "padding_action_selected": False,
        "split": request["split"],
        "seed": int(request["seed"]),
        "traffic_scenario": request["traffic_scenario"],
        "load_name": request["load_name"],
    }
