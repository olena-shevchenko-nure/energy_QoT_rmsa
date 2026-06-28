"""Label names emitted by the v1 EON data-generation pipeline."""

CNN_LABELS = [
    "delta_frag",
    "frag_after",
    "lmax_after_norm",
    "nseg_after_norm",
    "created_small_gap",
    "compactness",
    "placement_score",
    "J_total",
]

GNN_ROUTE_LABELS = [
    "feasible_label",
    "heuristic_route_score",
    "block_now",
    "num_feasible",
    "num_feasible_norm",
    "global_fragmentation",
]

DQN_LABELS = [
    "reward",
    "blocked",
    "best_candidate_index",
    "q_head_scores",
]

