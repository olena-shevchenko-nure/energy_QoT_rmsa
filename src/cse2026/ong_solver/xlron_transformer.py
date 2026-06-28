from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .common import CandidateBatch, SolverConfig, masked_argmax, pad_q_scores
from .solver import GnnCnnDqnOngSolver


def xlron_route_link_mask(batch: CandidateBatch, n_max: int) -> np.ndarray:
    """Build the Top-N route/link incidence mask consumed by the transformer."""

    mask = np.zeros((int(n_max), int(batch.state.link_count)), dtype=np.float32)
    for index, candidate in enumerate(batch.topn[: int(n_max)]):
        for link_id in candidate.route_link_ids:
            link_index = int(link_id)
            if 0 <= link_index < batch.state.link_count:
                mask[index, link_index] = 1.0
    return mask


def xlron_route_basic_features(batch: CandidateBatch, cfg: SolverConfig, n_max: int) -> np.ndarray:
    features = np.zeros((int(n_max), 2), dtype=np.float32)
    for index, candidate in enumerate(batch.topn[: int(n_max)]):
        features[index, 0] = float(candidate.route_length_km / max(cfg.max_route_length_norm_km, 1e-9))
        features[index, 1] = float(candidate.hop_count / 8.0)
    return features


def xlron_block_bounds(batch: CandidateBatch, n_max: int) -> np.ndarray:
    bounds = np.zeros((int(n_max), 2), dtype=np.float32)
    for index, candidate in enumerate(batch.topn[: int(n_max)]):
        bounds[index, 0] = float(candidate.b_start)
        bounds[index, 1] = float(candidate.w)
    return bounds


class XlronGraphTransformerPpoOngSolver:
    """XLRON Graph Transformer PPO candidate policy adapted to CSE2026 ONG rollout.

    XLRON trains a link-token actor-critic transformer for RMSA. This adapter keeps
    the actor-critic scoring model but constrains actions to this repository's
    deterministic Top-N feasible candidate surface so it is directly comparable
    with DQN, DeepRMSA-A3C, and tree-ranker policies.
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        batch_config = replace(self.config, use_neural=False, checkpoint_path=None)
        self._candidate_solver = GnnCnnDqnOngSolver(batch_config)
        self._model: Any = None
        self._torch: Any = None
        self.last_batch: CandidateBatch | None = None
        self.last_q_values: np.ndarray | None = None
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
        from .models import XlronGraphTransformerPpoNetwork, require_torch

        torch = require_torch()
        self._torch = torch
        if XlronGraphTransformerPpoNetwork is None:
            raise RuntimeError("XlronGraphTransformerPpoNetwork is unavailable because PyTorch is not installed")
        checkpoint = torch.load(Path(str(self.config.checkpoint_path)), map_location=self.config.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("XLRON Graph Transformer PPO checkpoint must be a dictionary")

        architecture = str(checkpoint.get("architecture", "link_transformer"))
        full_default = architecture.strip().lower() == "full"
        self._model = XlronGraphTransformerPpoNetwork(
            action_feature_dim=int(checkpoint.get("action_feature_dim", 10)),
            link_feature_dim=int(checkpoint.get("link_feature_dim", 8)),
            global_feature_dim=int(checkpoint.get("global_feature_dim", 8)),
            request_feature_dim=int(checkpoint.get("request_feature_dim", 3)),
            embedding_dim=int(checkpoint.get("embedding_dim", checkpoint.get("hidden_dim", self.config.hidden_dim))),
            num_layers=int(checkpoint.get("transformer_num_layers", 2)),
            num_heads=int(checkpoint.get("transformer_num_heads", 8)),
            dropout=float(checkpoint.get("dropout", 0.0)),
            position_dim=int(checkpoint.get("position_dim", 8)),
            architecture=architecture,
            spectrum_channels=int(checkpoint.get("spectrum_channels", 6)),
            route_basic_dim=int(checkpoint.get("route_basic_dim", checkpoint.get("route_basic_feature_dim", 2))),
            candidate_transformer_layers=int(checkpoint.get("candidate_transformer_layers", 1 if full_default else 0)),
            candidate_transformer_heads=int(checkpoint.get("candidate_transformer_heads", 4)),
            enable_spectrum_branch=bool(checkpoint.get("enable_spectrum_branch", full_default)),
            enable_candidate_attention=bool(checkpoint.get("enable_candidate_attention", full_default)),
            enable_base_relative_branch=bool(checkpoint.get("enable_base_relative_branch", full_default)),
            enable_auxiliary_heads=bool(checkpoint.get("enable_auxiliary_heads", False)),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.config.device)
        self._model.eval()

    def q_values(self, batch: CandidateBatch) -> np.ndarray:
        if self._model is None:
            scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
            return pad_q_scores(scores, self.config.n_max)
        torch = self._torch
        if torch is None:
            raise RuntimeError("XLRON Graph Transformer PPO logits requested before model initialization")

        route_link_mask = xlron_route_link_mask(batch, self.config.n_max)
        route_basic_features = xlron_route_basic_features(batch, self.config, self.config.n_max)
        block_bounds = xlron_block_bounds(batch, self.config.n_max)
        with torch.no_grad():
            logits, _value = self._model(
                link_features=torch.as_tensor(batch.link_features[None, ...], dtype=torch.float32, device=self.config.device),
                edge_index=torch.as_tensor(batch.state.edge_index, dtype=torch.long, device=self.config.device),
                global_features=torch.as_tensor(batch.global_features[None, ...], dtype=torch.float32, device=self.config.device),
                request_features=torch.as_tensor(batch.request_features[None, ...], dtype=torch.float32, device=self.config.device),
                action_features=torch.as_tensor(batch.action_features[None, ...], dtype=torch.float32, device=self.config.device),
                route_link_mask=torch.as_tensor(route_link_mask[None, ...], dtype=torch.float32, device=self.config.device),
                spectrum_tensors=torch.as_tensor(batch.spectrum_tensors[None, ...], dtype=torch.float32, device=self.config.device),
                route_basic_features=torch.as_tensor(route_basic_features[None, ...], dtype=torch.float32, device=self.config.device),
                block_bounds=torch.as_tensor(block_bounds[None, ...], dtype=torch.float32, device=self.config.device),
                candidate_mask=torch.as_tensor(batch.candidate_mask[None, :] > 0.0, dtype=torch.bool, device=self.config.device),
            )
        return logits.detach().cpu().numpy().reshape(-1).astype(np.float32)

