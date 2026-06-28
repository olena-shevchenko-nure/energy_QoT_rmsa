from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cse2026.ong_solver.common import Candidate, CandidateBatch, masked_argmax, pad_q_scores


OVERRIDE_FEATURE_NAMES = [
    "candidate_index_norm",
    "valid_candidates_norm",
    "is_j_total",
    "energy_norm",
    "energy_delta",
    "fragmentation_after",
    "fragmentation_delta",
    "delta_fragmentation",
    "delta_fragmentation_delta",
    "largest_free_block_norm",
    "largest_free_block_delta_norm",
    "small_gap_penalty",
    "small_gap_delta",
    "delay_norm",
    "delay_delta_norm",
    "qot_margin",
    "qot_margin_delta",
    "qot_risk",
    "qot_risk_delta",
    "compactness",
    "compactness_delta",
    "route_length_norm",
    "route_length_delta_norm",
    "hop_count_norm",
    "hop_count_delta_norm",
    "width_norm",
    "width_delta_norm",
    "j_total",
    "j_total_delta",
    "request_bit_rate_norm",
    "request_holding_norm",
    "global_load",
    "global_fragmentation",
]


def select_q_head_index(batch: CandidateBatch, n_max: int) -> int:
    if not np.asarray(batch.candidate_mask, dtype=bool).any():
        return -1
    scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
    return masked_argmax(pad_q_scores(scores, n_max), batch.candidate_mask)


def candidate_indices_for_override(batch: CandidateBatch, n_max: int, top_k: int, include_j_total: bool = True) -> list[int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    if valid.size == 0:
        return []
    scores = np.asarray([batch.topn[index].q_head_score for index in valid], dtype=np.float64)
    order = np.argsort(-scores)
    selected = [int(valid[index]) for index in order[: max(1, min(int(top_k), int(valid.size)))]]
    if include_j_total:
        j_total = int(valid[0])
        if j_total not in selected:
            selected.append(j_total)
    return selected


def candidate_indices_for_topn(batch: CandidateBatch) -> list[int]:
    valid = np.flatnonzero(batch.candidate_mask.astype(bool))
    return [int(index) for index in valid]


def _safe(value: float, scale: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(value) / max(float(scale), 1e-9)


def _global_value(batch: CandidateBatch, index: int, default: float = 0.0) -> float:
    values = np.asarray(batch.global_features, dtype=np.float32).reshape(-1)
    if 0 <= int(index) < values.size:
        return float(values[int(index)])
    return float(default)


def _request_value(batch: CandidateBatch, index: int, default: float = 0.0) -> float:
    values = np.asarray(batch.request_features, dtype=np.float32).reshape(-1)
    if 0 <= int(index) < values.size:
        return float(values[int(index)])
    return float(default)


def _feature_vector(
    *,
    batch: CandidateBatch,
    candidate: Candidate,
    reference: Candidate,
    candidate_index: int,
    valid_candidates: int,
    n_max: int,
) -> list[float]:
    slots = max(int(batch.state.slot_count), 1)
    return [
        _safe(candidate_index, max(n_max - 1, 1)),
        _safe(valid_candidates, n_max),
        1.0 if int(candidate_index) == 0 else 0.0,
        float(candidate.energy_increment_norm),
        float(candidate.energy_increment_norm - reference.energy_increment_norm),
        float(candidate.fragmentation_after),
        float(candidate.fragmentation_after - reference.fragmentation_after),
        float(candidate.delta_fragmentation),
        float(candidate.delta_fragmentation - reference.delta_fragmentation),
        _safe(candidate.largest_free_block_after, slots),
        _safe(candidate.largest_free_block_after - reference.largest_free_block_after, slots),
        float(candidate.small_gap_penalty),
        float(candidate.small_gap_penalty - reference.small_gap_penalty),
        _safe(candidate.delay_ms, 50.0),
        _safe(candidate.delay_ms - reference.delay_ms, 50.0),
        float(candidate.qot_margin_norm),
        float(candidate.qot_margin_norm - reference.qot_margin_norm),
        float(candidate.qot_risk),
        float(candidate.qot_risk - reference.qot_risk),
        float(candidate.compactness),
        float(candidate.compactness - reference.compactness),
        _safe(candidate.route_length_km, 6000.0),
        _safe(candidate.route_length_km - reference.route_length_km, 6000.0),
        _safe(candidate.hop_count, 8.0),
        _safe(candidate.hop_count - reference.hop_count, 8.0),
        _safe(candidate.w, slots),
        _safe(candidate.w - reference.w, slots),
        float(candidate.j_total),
        float(candidate.j_total - reference.j_total),
        _request_value(batch, 0),
        _request_value(batch, 2),
        _global_value(batch, 0),
        _global_value(batch, 2),
    ]


def override_feature_matrix(
    *,
    batch: CandidateBatch,
    candidate_indices: list[int],
    n_max: int,
) -> tuple[np.ndarray, list[int]]:
    q_head_index = select_q_head_index(batch, n_max)
    if q_head_index < 0:
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    q_head = batch.topn[q_head_index]
    valid_candidates = int(np.flatnonzero(batch.candidate_mask.astype(bool)).size)
    rows: list[list[float]] = []
    kept_indices: list[int] = []
    for candidate_index in candidate_indices:
        if int(candidate_index) == int(q_head_index):
            continue
        if not (0 <= int(candidate_index) < len(batch.topn)):
            continue
        if not bool(batch.candidate_mask[int(candidate_index)]):
            continue
        candidate = batch.topn[int(candidate_index)]
        rows.append(
            _feature_vector(
                batch=batch,
                candidate=candidate,
                reference=q_head,
                candidate_index=int(candidate_index),
                valid_candidates=valid_candidates,
                n_max=n_max,
            )
        )
        kept_indices.append(int(candidate_index))
    if not rows:
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), kept_indices


def candidate_feature_matrix(
    *,
    batch: CandidateBatch,
    candidate_indices: list[int],
    n_max: int,
    reference_index: int,
) -> tuple[np.ndarray, list[int]]:
    if int(reference_index) < 0:
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    if not (0 <= int(reference_index) < len(batch.topn)):
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    if not bool(batch.candidate_mask[int(reference_index)]):
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    reference = batch.topn[int(reference_index)]
    valid_candidates = int(np.flatnonzero(batch.candidate_mask.astype(bool)).size)
    rows: list[list[float]] = []
    kept_indices: list[int] = []
    for candidate_index in candidate_indices:
        if not (0 <= int(candidate_index) < len(batch.topn)):
            continue
        if not bool(batch.candidate_mask[int(candidate_index)]):
            continue
        candidate = batch.topn[int(candidate_index)]
        rows.append(
            _feature_vector(
                batch=batch,
                candidate=candidate,
                reference=reference,
                candidate_index=int(candidate_index),
                valid_candidates=valid_candidates,
                n_max=n_max,
            )
        )
        kept_indices.append(int(candidate_index))
    if not rows:
        return np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), kept_indices


@dataclass
class OverrideClassifier:
    model_type: str
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    top_k: int
    include_j_total: bool
    hidden_weights: np.ndarray | None = None
    hidden_bias: np.ndarray | None = None
    output_weights: np.ndarray | None = None
    output_bias: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> "OverrideClassifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_type=str(data.get("model_type", "logistic_override_classifier")),
            feature_names=[str(value) for value in data["feature_names"]],
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
            weights=np.asarray(data["weights"], dtype=np.float32),
            bias=float(data["bias"]),
            threshold=float(data["threshold"]),
            top_k=int(data.get("top_k", 4)),
            include_j_total=bool(data.get("include_j_total", True)),
            hidden_weights=(
                np.asarray(data["hidden_weights"], dtype=np.float32)
                if "hidden_weights" in data
                else None
            ),
            hidden_bias=np.asarray(data["hidden_bias"], dtype=np.float32) if "hidden_bias" in data else None,
            output_weights=(
                np.asarray(data["output_weights"], dtype=np.float32)
                if "output_weights" in data
                else None
            ),
            output_bias=float(data["output_bias"]) if "output_bias" in data else None,
        )

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x = (features.astype(np.float32) - self.mean) / np.maximum(self.scale, 1e-6)
        if self.model_type == "mlp_override_classifier":
            if self.hidden_weights is None or self.hidden_bias is None or self.output_weights is None or self.output_bias is None:
                raise ValueError("MLP override classifier checkpoint is missing hidden/output weights")
            hidden = np.maximum(x @ self.hidden_weights + self.hidden_bias, 0.0)
            logits = hidden @ self.output_weights.reshape(-1) + float(self.output_bias)
        else:
            logits = x @ self.weights.astype(np.float32) + float(self.bias)
        logits = np.clip(logits, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def select_index(self, batch: CandidateBatch, n_max: int) -> tuple[int, float]:
        candidates = candidate_indices_for_override(batch, n_max, self.top_k, self.include_j_total)
        features, indices = override_feature_matrix(batch=batch, candidate_indices=candidates, n_max=n_max)
        if not indices:
            return -1, 0.0
        probabilities = self.probabilities(features)
        best_position = int(np.argmax(probabilities))
        best_probability = float(probabilities[best_position])
        if best_probability < float(self.threshold):
            return -1, best_probability
        return int(indices[best_position]), best_probability
