from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .common import CandidateBatch, SolverConfig, masked_argmax, pad_q_scores
from .solver import GnnCnnDqnOngSolver


class GnnCnnA3COngSolver:
    """Full GNN+CNN actor-critic policy over the shared Top-N RMSA candidate surface."""

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
        from .models import GnnCnnA3CNetwork, require_torch

        torch = require_torch()
        self._torch = torch
        if GnnCnnA3CNetwork is None:
            raise RuntimeError("GnnCnnA3CNetwork is unavailable because PyTorch is not installed")
        checkpoint = torch.load(Path(str(self.config.checkpoint_path)), map_location=self.config.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("GNN+CNN A3C checkpoint must be a dictionary")
        self._model = GnnCnnA3CNetwork(
            action_feature_dim=int(checkpoint.get("action_feature_dim", 10)),
            hidden_dim=int(checkpoint.get("hidden_dim", self.config.hidden_dim)),
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
            raise RuntimeError("GNN+CNN A3C logits requested before model initialization")

        n_max = int(self.config.n_max)
        edge_count = int(batch.state.link_count)
        route_link_mask = np.zeros((n_max, edge_count), dtype=np.float32)
        route_basic = np.zeros((n_max, 2), dtype=np.float32)
        block_bounds = np.zeros((n_max, 2), dtype=np.float32)
        for index, candidate in enumerate(batch.topn[:n_max]):
            for link_id in candidate.route_link_ids:
                link_index = int(link_id)
                if 0 <= link_index < edge_count:
                    route_link_mask[index, link_index] = 1.0
            route_basic[index, 0] = float(candidate.route_length_km / max(self.config.max_route_length_norm_km, 1e-9))
            route_basic[index, 1] = float(candidate.hop_count / 8.0)
            block_bounds[index, 0] = float(candidate.b_start)
            block_bounds[index, 1] = float(candidate.w)

        device = self.config.device
        with torch.no_grad():
            logits, _value = self._model(
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
                candidate_mask=torch.as_tensor(batch.candidate_mask[None, :] > 0.0, dtype=torch.bool, device=device),
            )
        return logits.detach().cpu().numpy().reshape(-1).astype(np.float32)
