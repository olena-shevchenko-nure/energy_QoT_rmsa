from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .common import CandidateBatch, SolverConfig, masked_argmax, pad_q_scores
from .solver import GnnCnnDqnOngSolver


DEEPRMSA_ACTION_FEATURE_COLUMNS = (
    "route_length_norm",
    "hop_count_norm",
    "b_start_norm",
    "w_norm",
    "qot_margin_norm",
    "delay_norm",
    "energy_increment_norm",
    "fragmentation_after",
    "largest_free_block_after_norm",
    "small_gap_penalty",
)

DEEPRMSA_CANDIDATE_FEATURE_COLUMNS = (
    *DEEPRMSA_ACTION_FEATURE_COLUMNS,
    "prior_score",
    "topn_rank_norm",
    "candidate_mask",
)


def normalize_deeprmsa_prior_score(value: str | None) -> str:
    mode = str(value or "q_head_score").strip().lower().replace("-", "_")
    if mode in {"q_head", "qhead", "q_head_score"}:
        return "q_head_score"
    if mode in {"energy_aware", "energy_aware_rank", "energy_aware_ksp_bm_ff", "energy_aware_ksp_bm_ff_rank"}:
        return "energy_aware_rank"
    raise ValueError(f"Unsupported DeepRMSA prior score: {value}")


def deeprmsa_candidate_feature_columns(prior_score_mode: str | None = "q_head_score") -> tuple[str, ...]:
    prior = normalize_deeprmsa_prior_score(prior_score_mode)
    prior_name = "q_head_score" if prior == "q_head_score" else "energy_aware_rank_score"
    return (
        *DEEPRMSA_ACTION_FEATURE_COLUMNS,
        prior_name,
        "topn_rank_norm",
        "candidate_mask",
    )


def _energy_aware_ksp_bm_ff_key(candidate: Any, index: int) -> tuple[float, int, int, int, float, int]:
    return (
        float(candidate.energy_increment),
        int(candidate.route_id),
        int(candidate.b_start),
        int(candidate.w),
        -float(candidate.spectral_efficiency),
        int(index),
    )


def deeprmsa_prior_scores_from_batch(batch: CandidateBatch, n_max: int, prior_score_mode: str | None) -> np.ndarray:
    prior = normalize_deeprmsa_prior_score(prior_score_mode)
    if prior == "q_head_score":
        return pad_q_scores(np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32), n_max)

    scores = np.zeros(int(n_max), dtype=np.float32)
    valid = [int(index) for index in np.flatnonzero(batch.candidate_mask.astype(bool))]
    if not valid:
        return scores
    ordered = sorted(valid, key=lambda index: _energy_aware_ksp_bm_ff_key(batch.topn[index], int(index)))
    denom = float(max(len(ordered) - 1, 1))
    for rank, index in enumerate(ordered):
        scores[int(index)] = float(1.0 - float(rank) / denom)
    return scores


def deeprmsa_features_from_arrays(
    *,
    node_features: np.ndarray,
    global_features: np.ndarray,
    request_features: np.ndarray,
    action_features: np.ndarray,
    candidate_mask: np.ndarray,
    q_head_scores: np.ndarray | None = None,
    prior_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_max = int(action_features.shape[0])
    if prior_scores is None:
        if q_head_scores is None:
            raise ValueError("deeprmsa_features_from_arrays requires prior_scores or q_head_scores")
        prior_scores = q_head_scores
    prior = np.asarray(prior_scores, dtype=np.float32).reshape(n_max, 1)
    prior = np.where(np.isfinite(prior), prior, 0.0).astype(np.float32)
    rank_norm = (np.arange(n_max, dtype=np.float32) / float(max(n_max - 1, 1))).reshape(n_max, 1)
    mask_feature = np.asarray(candidate_mask, dtype=np.float32).reshape(n_max, 1)
    candidate_features = np.concatenate(
        [
            np.asarray(action_features, dtype=np.float32),
            prior,
            rank_norm,
            mask_feature,
        ],
        axis=1,
    ).astype(np.float32)
    context_features = np.concatenate(
        [
            np.asarray(node_features, dtype=np.float32).reshape(-1),
            np.asarray(global_features, dtype=np.float32).reshape(-1),
            np.asarray(request_features, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)
    return candidate_features, context_features


def deeprmsa_features_from_batch(
    batch: CandidateBatch,
    n_max: int,
    prior_score_mode: str | None = "q_head_score",
) -> tuple[np.ndarray, np.ndarray]:
    prior_scores = deeprmsa_prior_scores_from_batch(batch, n_max, prior_score_mode)
    return deeprmsa_features_from_arrays(
        node_features=batch.node_features,
        global_features=batch.global_features,
        request_features=batch.request_features,
        action_features=batch.action_features,
        prior_scores=prior_scores,
        candidate_mask=batch.candidate_mask,
    )


class DeepRmsaA3COngSolver:
    """DeepRMSA actor-critic policy adapted to the CSE2026 Top-N candidate API."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        batch_config = replace(self.config, use_neural=False, checkpoint_path=None)
        self._candidate_solver = GnnCnnDqnOngSolver(batch_config)
        self._model: Any = None
        self._torch: Any = None
        self.last_batch: CandidateBatch | None = None
        self.last_q_values: np.ndarray | None = None
        self.prior_score_mode = normalize_deeprmsa_prior_score(self.config.deeprmsa_prior_score)
        if self.config.checkpoint_path:
            self._load_model()

    def __call__(self, env: Any, observation: Any | None = None) -> Any:
        return self.act(env, observation=observation)

    def act(self, env: Any, observation: Any | None = None) -> Any:
        del observation
        batch = self.candidate_batch(env)
        self.last_batch = batch
        if not batch.has_real_candidates:
            self.last_q_values = pad_q_scores(np.asarray([], dtype=np.float32), self.config.n_max)
            return self.adapter(env).block_action(env)
        q_values = self.q_values(batch)
        self.last_q_values = q_values
        selected_index = masked_argmax(q_values, batch.candidate_mask)
        return batch.topn[selected_index].action

    def candidate_batch(self, env: Any) -> CandidateBatch:
        return self._candidate_solver.candidate_batch(env)

    def adapter(self, env: Any):
        return self._candidate_solver.adapter(env)

    def _load_model(self) -> None:
        from .models import DeepRmsaA3CNetwork, require_torch

        torch = require_torch()
        self._torch = torch
        if DeepRmsaA3CNetwork is None:
            raise RuntimeError("DeepRmsaA3CNetwork is unavailable because PyTorch is not installed")
        checkpoint = torch.load(Path(str(self.config.checkpoint_path)), map_location=self.config.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("DeepRMSA-A3C checkpoint must be a dictionary")
        self._model = DeepRmsaA3CNetwork(
            n_max=int(checkpoint.get("n_max", self.config.n_max)),
            candidate_feature_dim=int(checkpoint["candidate_feature_dim"]),
            context_feature_dim=int(checkpoint["context_feature_dim"]),
            hidden_dim=int(checkpoint.get("hidden_dim", self.config.hidden_dim)),
            layers=int(checkpoint.get("layers", 5)),
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.config.device)
        self._model.eval()
        checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if isinstance(checkpoint_config, dict):
            self.prior_score_mode = normalize_deeprmsa_prior_score(
                checkpoint_config.get("deeprmsa_prior_score", self.prior_score_mode)
            )

    def q_values(self, batch: CandidateBatch) -> np.ndarray:
        if self._model is None:
            scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
            return pad_q_scores(scores, self.config.n_max)
        torch = self._torch
        if torch is None:
            raise RuntimeError("DeepRMSA-A3C Q-values requested before model initialization")
        candidate_features, context_features = deeprmsa_features_from_batch(
            batch,
            self.config.n_max,
            self.prior_score_mode,
        )
        with torch.no_grad():
            logits, _value = self._model(
                torch.as_tensor(candidate_features[None, ...], dtype=torch.float32, device=self.config.device),
                torch.as_tensor(context_features[None, ...], dtype=torch.float32, device=self.config.device),
            )
        return logits.detach().cpu().numpy().reshape(-1).astype(np.float32)

