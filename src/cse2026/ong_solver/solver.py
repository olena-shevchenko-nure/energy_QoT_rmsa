from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .adapters import OngAdapter, select_adapter
from .common import CandidateBatch, SolverConfig, masked_argmax, normalize_q_score_mode, pad_q_scores


class GnnCnnDqnOngSolver:
    """GNN+CNN+DQN candidate scorer for Optical Networking Gym environments.

    The solver always enforces deterministic candidate feasibility before scoring.
    If no trained checkpoint is provided, it uses the same energy/QoT/fragmentation
    candidate score as a deterministic fallback while preserving the final API.
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()
        self.rng = np.random.default_rng(self.config.rng_seed)
        self._adapter: OngAdapter | None = None
        self._model: Any = None
        self._torch: Any = None
        self.q_score_mode = normalize_q_score_mode(self.config.q_score_mode)
        self.residual_scale = float(self.config.residual_scale)
        self.residual_delta_clip = float(self.config.residual_delta_clip)
        self.last_batch: CandidateBatch | None = None
        self.last_q_values: np.ndarray | None = None
        if self.config.use_neural or self.config.checkpoint_path:
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

        valid_indices = np.flatnonzero(batch.candidate_mask.astype(bool))
        if self.config.epsilon > 0.0 and float(self.rng.random()) < self.config.epsilon:
            selected_index = int(self.rng.choice(valid_indices))
            self.last_q_values = self.q_values(batch)
            return batch.topn[selected_index].action

        q_values = self.q_values(batch)
        self.last_q_values = q_values
        selected_index = masked_argmax(q_values, batch.candidate_mask)
        return batch.topn[selected_index].action

    def candidate_batch(self, env: Any) -> CandidateBatch:
        return self.adapter(env).candidate_batch(env, self.config, self.rng)

    def adapter(self, env: Any) -> OngAdapter:
        if self._adapter is None:
            self._adapter = select_adapter(env)
        return self._adapter

    def q_values(self, batch: CandidateBatch) -> np.ndarray:
        if self._model is None:
            scores = np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32)
            return pad_q_scores(scores, self.config.n_max)
        return self._neural_q_values(batch)

    def _load_model(self) -> None:
        from .models import CandidateQNetwork, require_torch

        torch = require_torch()
        self._torch = torch
        if CandidateQNetwork is None:
            raise RuntimeError("CandidateQNetwork is unavailable because PyTorch is not installed")

        action_feature_dim = 10
        self._model = CandidateQNetwork(action_feature_dim=action_feature_dim, hidden_dim=self.config.hidden_dim)
        if self.config.checkpoint_path:
            checkpoint = torch.load(Path(self.config.checkpoint_path), map_location=self.config.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self._model.load_state_dict(state_dict)
            if isinstance(checkpoint, dict):
                checkpoint_config = checkpoint.get("config") or {}
                if isinstance(checkpoint_config, dict):
                    self.q_score_mode = normalize_q_score_mode(checkpoint_config.get("q_score_mode", self.q_score_mode))
                    self.residual_scale = float(checkpoint_config.get("residual_scale", self.residual_scale))
                    self.residual_delta_clip = float(checkpoint_config.get("residual_delta_clip", self.residual_delta_clip))
        self._model.to(self.config.device)
        self._model.eval()

    def _neural_q_values(self, batch: CandidateBatch) -> np.ndarray:
        torch = self._torch
        if torch is None or self._model is None:
            raise RuntimeError("neural Q-values requested before model initialization")

        edge_count = batch.state.link_count
        n_max = self.config.n_max
        route_link_mask = np.zeros((n_max, edge_count), dtype=np.float32)
        route_basic = np.zeros((n_max, 2), dtype=np.float32)
        block_bounds = np.zeros((n_max, 2), dtype=np.float32)
        for index, candidate in enumerate(batch.topn):
            for link_id in candidate.route_link_ids:
                if 0 <= int(link_id) < edge_count:
                    route_link_mask[index, int(link_id)] = 1.0
            route_basic[index, 0] = float(candidate.route_length_km / max(self.config.max_route_length_norm_km, 1e-9))
            route_basic[index, 1] = float(candidate.hop_count / 8.0)
            block_bounds[index, 0] = float(candidate.b_start)
            block_bounds[index, 1] = float(candidate.w)

        device = self.config.device
        with torch.no_grad():
            q_values = self._model(
                node_features=torch.as_tensor(batch.node_features[None, ...], dtype=torch.float32, device=device),
                link_features=torch.as_tensor(batch.link_features[None, ...], dtype=torch.float32, device=device),
                global_features=torch.as_tensor(batch.global_features[None, ...], dtype=torch.float32, device=device),
                edge_index=torch.as_tensor(batch.state.edge_index, dtype=torch.long, device=device),
                request_features=torch.as_tensor(batch.request_features[None, ...], dtype=torch.float32, device=device),
                spectrum_tensors=torch.as_tensor(batch.spectrum_tensors[None, ...], dtype=torch.float32, device=device),
                action_features=torch.as_tensor(batch.action_features[None, ...], dtype=torch.float32, device=device),
                route_link_mask=torch.as_tensor(route_link_mask[None, ...], dtype=torch.float32, device=device),
                route_basic_features=torch.as_tensor(route_basic[None, ...], dtype=torch.float32, device=device),
                block_bounds=torch.as_tensor(block_bounds[None, ...], dtype=torch.float32, device=device),
            )
        values = q_values.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if self.q_score_mode == "q_head_residual":
            baseline = pad_q_scores(np.asarray([candidate.q_head_score for candidate in batch.topn], dtype=np.float32), n_max)
            if self.residual_delta_clip > 0.0:
                values = np.clip(values, -self.residual_delta_clip, self.residual_delta_clip)
            values = baseline + float(self.residual_scale) * values
        return values.astype(np.float32)


def gnn_cnn_dqn_policy(env: Any) -> Any:
    """One-line policy function compatible with Gym evaluation loops."""

    return GnnCnnDqnOngSolver().act(env)
