from __future__ import annotations

import copy
import math
import json
from pathlib import Path
from typing import Any

import numpy as np

from cse2026.experiments.eon.train_dqn import _batch_to_arrays
from cse2026.experiments.eon.tree_ranker_runtime import select_tree_base_index
from cse2026.ong_solver.common import SolverConfig


LIVE_OVERRIDE_GATE_FEATURE_NAMES = (
    "valid_candidates_norm",
    "eligible_count_norm",
    "selected_index_norm",
    "base_index_norm",
    "selected_score_rank_norm",
    "base_score_rank_norm",
    "selected_win_prob",
    "selected_loss_prob",
    "selected_delta_pred",
    "selected_score",
    "base_win_prob",
    "base_loss_prob",
    "base_delta_pred",
    "base_score",
    "win_prob_delta",
    "loss_prob_delta",
    "delta_pred_delta",
    "score_delta",
    "selected_energy_increment_norm",
    "base_energy_increment_norm",
    "energy_increment_norm_delta",
    "selected_fragmentation_after",
    "base_fragmentation_after",
    "fragmentation_after_delta",
    "selected_delta_fragmentation",
    "base_delta_fragmentation",
    "delta_fragmentation_delta",
    "selected_largest_free_block_norm",
    "base_largest_free_block_norm",
    "largest_free_block_delta_norm",
    "selected_small_gap_penalty",
    "base_small_gap_penalty",
    "small_gap_delta",
    "selected_qot_margin_norm",
    "base_qot_margin_norm",
    "qot_margin_delta",
    "selected_qot_risk",
    "base_qot_risk",
    "qot_risk_delta",
    "selected_delay_norm",
    "base_delay_norm",
    "delay_delta_norm",
    "selected_j_total",
    "base_j_total",
    "j_total_delta",
    "selected_width_norm",
    "base_width_norm",
    "width_delta_norm",
    "selected_route_id_norm",
    "base_route_id_norm",
    "route_id_delta_norm",
    "same_route",
    "same_modulation",
    "selected_is_j_total",
    "base_is_j_total",
)


def _safe(value: float, scale: float = 1.0) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(value) / max(float(scale), 1.0e-9)


def _rank_desc(values: np.ndarray, valid_indices: np.ndarray, selected_index: int) -> int | None:
    if selected_index < 0 or valid_indices.size == 0:
        return None
    ordered = sorted((int(index) for index in valid_indices), key=lambda index: (-float(values[int(index)]), index))
    try:
        return int(ordered.index(int(selected_index)) + 1)
    except ValueError:
        return None


def _candidate_gate_features(
    *,
    batch: Any,
    solver_config: SolverConfig,
    details: dict[str, Any],
) -> dict[str, float]:
    pred = details["prediction"]
    selected_index = int(details["selected_index"])
    base_index = int(details["base_index"])
    n_max = int(solver_config.n_max)
    valid = np.asarray(details["valid_indices"], dtype=np.int64)
    valid_count = max(int(valid.size), 1)
    selected = batch.topn[int(selected_index)]
    base = batch.topn[int(base_index)]
    slots = max(int(batch.state.slot_count), 1)
    delay_bound = max(float(getattr(solver_config, "delay_bound_ms", 50.0)), 1.0e-9)
    route_scale = 8.0
    selected_rank = _rank_desc(np.asarray(pred["score"], dtype=np.float32), valid, selected_index)
    base_rank = _rank_desc(np.asarray(pred["score"], dtype=np.float32), valid, base_index)
    values = {
        "valid_candidates_norm": _safe(valid_count, n_max),
        "eligible_count_norm": _safe(int(details["eligible_count"]), n_max),
        "selected_index_norm": _safe(selected_index, max(n_max - 1, 1)),
        "base_index_norm": _safe(base_index, max(n_max - 1, 1)),
        "selected_score_rank_norm": _safe(float(selected_rank or valid_count), valid_count),
        "base_score_rank_norm": _safe(float(base_rank or valid_count), valid_count),
        "selected_win_prob": float(pred["win_prob"][selected_index]),
        "selected_loss_prob": float(pred["loss_prob"][selected_index]),
        "selected_delta_pred": float(pred["delta_pred"][selected_index]),
        "selected_score": float(pred["score"][selected_index]),
        "base_win_prob": float(pred["win_prob"][base_index]),
        "base_loss_prob": float(pred["loss_prob"][base_index]),
        "base_delta_pred": float(pred["delta_pred"][base_index]),
        "base_score": float(pred["score"][base_index]),
        "selected_energy_increment_norm": float(selected.energy_increment_norm),
        "base_energy_increment_norm": float(base.energy_increment_norm),
        "selected_fragmentation_after": float(selected.fragmentation_after),
        "base_fragmentation_after": float(base.fragmentation_after),
        "selected_delta_fragmentation": float(selected.delta_fragmentation),
        "base_delta_fragmentation": float(base.delta_fragmentation),
        "selected_largest_free_block_norm": _safe(selected.largest_free_block_after, slots),
        "base_largest_free_block_norm": _safe(base.largest_free_block_after, slots),
        "selected_small_gap_penalty": float(selected.small_gap_penalty),
        "base_small_gap_penalty": float(base.small_gap_penalty),
        "selected_qot_margin_norm": float(selected.qot_margin_norm),
        "base_qot_margin_norm": float(base.qot_margin_norm),
        "selected_qot_risk": float(selected.qot_risk),
        "base_qot_risk": float(base.qot_risk),
        "selected_delay_norm": _safe(selected.delay_ms, delay_bound),
        "base_delay_norm": _safe(base.delay_ms, delay_bound),
        "selected_j_total": float(selected.j_total),
        "base_j_total": float(base.j_total),
        "selected_width_norm": _safe(selected.w, slots),
        "base_width_norm": _safe(base.w, slots),
        "selected_route_id_norm": _safe(selected.route_id, route_scale),
        "base_route_id_norm": _safe(base.route_id, route_scale),
        "same_route": 1.0 if int(selected.route_id) == int(base.route_id) else 0.0,
        "same_modulation": 1.0 if int(selected.modulation_index) == int(base.modulation_index) else 0.0,
        "selected_is_j_total": 1.0 if int(selected_index) == 0 else 0.0,
        "base_is_j_total": 1.0 if int(base_index) == 0 else 0.0,
    }
    values.update(
        {
            "win_prob_delta": values["selected_win_prob"] - values["base_win_prob"],
            "loss_prob_delta": values["selected_loss_prob"] - values["base_loss_prob"],
            "delta_pred_delta": values["selected_delta_pred"] - values["base_delta_pred"],
            "score_delta": values["selected_score"] - values["base_score"],
            "energy_increment_norm_delta": values["selected_energy_increment_norm"] - values["base_energy_increment_norm"],
            "fragmentation_after_delta": values["selected_fragmentation_after"] - values["base_fragmentation_after"],
            "delta_fragmentation_delta": values["selected_delta_fragmentation"] - values["base_delta_fragmentation"],
            "largest_free_block_delta_norm": values["selected_largest_free_block_norm"] - values["base_largest_free_block_norm"],
            "small_gap_delta": values["selected_small_gap_penalty"] - values["base_small_gap_penalty"],
            "qot_margin_delta": values["selected_qot_margin_norm"] - values["base_qot_margin_norm"],
            "qot_risk_delta": values["selected_qot_risk"] - values["base_qot_risk"],
            "delay_delta_norm": values["selected_delay_norm"] - values["base_delay_norm"],
            "j_total_delta": values["selected_j_total"] - values["base_j_total"],
            "width_delta_norm": values["selected_width_norm"] - values["base_width_norm"],
            "route_id_delta_norm": values["selected_route_id_norm"] - values["base_route_id_norm"],
        }
    )
    return values


class LiveOverrideGate:
    def __init__(
        self,
        *,
        models: dict[str, Any],
        feature_names: list[str],
        thresholds: dict[str, float],
        backend: str,
        score_weights: dict[str, float],
        model_module: Any,
    ) -> None:
        self.models = dict(models)
        self.feature_names = [str(name) for name in feature_names]
        self.thresholds = dict(thresholds)
        self.backend = str(backend)
        self.score_weights = dict(score_weights)
        self.model_module = model_module

    @classmethod
    def load(cls, path: str | Path) -> "LiveOverrideGate":
        artifact_path = Path(path)
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
        backend = str(data.get("backend", "xgboost"))
        if backend != "xgboost":
            raise ValueError(f"Unsupported live override gate backend: {backend}")
        import xgboost as xgb

        models: dict[str, Any] = {}
        for name, model_path_value in dict(data.get("model_paths") or {}).items():
            model_path = Path(str(model_path_value))
            if not model_path.is_absolute():
                model_path = artifact_path.parent / model_path
            model = xgb.Booster()
            model.load_model(str(model_path))
            models[str(name)] = model
        return cls(
            models=models,
            feature_names=[str(name) for name in data.get("feature_names", LIVE_OVERRIDE_GATE_FEATURE_NAMES)],
            thresholds=dict(data.get("thresholds") or {}),
            backend=backend,
            score_weights=dict(data.get("score_weights") or {}),
            model_module=xgb,
        )

    def _predict(self, name: str, features: np.ndarray) -> float:
        model = self.models.get(name)
        if model is None:
            return 0.0
        matrix = self.model_module.DMatrix(features.astype(np.float32), feature_names=self.feature_names)
        return float(np.asarray(model.predict(matrix), dtype=np.float32).reshape(-1)[0])

    def decision(self, batch: Any, solver_config: SolverConfig, details: dict[str, Any]) -> dict[str, Any]:
        feature_values = _candidate_gate_features(batch=batch, solver_config=solver_config, details=details)
        features = np.asarray([[float(feature_values.get(name, 0.0)) for name in self.feature_names]], dtype=np.float32)
        win_score = self._predict("win", features)
        loss_score = self._predict("loss", features)
        delta_score = self._predict("delta", features)
        combined_score = (
            float(self.score_weights.get("win", 1.0)) * win_score
            - float(self.score_weights.get("loss", 1.0)) * loss_score
            + float(self.score_weights.get("delta", 1.0)) * delta_score
        )
        allow = (
            win_score >= float(self.thresholds.get("win_threshold", -math.inf))
            and loss_score <= float(self.thresholds.get("loss_threshold", math.inf))
            and delta_score >= float(self.thresholds.get("delta_threshold", -math.inf))
            and combined_score >= float(self.thresholds.get("combined_threshold", -math.inf))
        )
        return {
            "allow": bool(allow),
            "win_score": float(win_score),
            "loss_score": float(loss_score),
            "delta_score": float(delta_score),
            "combined_score": float(combined_score),
        }


class ThreeHeadCandidateJudge:
    """Runtime three-head candidate judge matching the experiment trainer architecture."""

    @staticmethod
    def build(*, action_feature_dim: int, hidden_dim: int, init_from_q_head: bool, torch: Any) -> Any:
        from cse2026.ong_solver.models import CandidateQNetwork

        if CandidateQNetwork is None:
            raise RuntimeError("CandidateQNetwork is unavailable because PyTorch is not installed")
        source = CandidateQNetwork(action_feature_dim=int(action_feature_dim), hidden_dim=int(hidden_dim))

        class _ThreeHeadCandidateJudge(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gnn = source.gnn
                self.slot_cnn = source.slot_cnn
                self.route_pool = source.route_pool
                self.request_encoder = source.request_encoder
                self.action_encoder = source.action_encoder
                fusion_dim = int(hidden_dim) * 3 + 64 + 64
                q_layers = list(source.q_head.children()) if bool(init_from_q_head) else []
                if len(q_layers) >= 7:
                    self.trunk = torch.nn.Sequential(*[copy.deepcopy(layer) for layer in q_layers[:-1]])
                    self.delta_head = copy.deepcopy(q_layers[-1])
                else:
                    self.trunk = torch.nn.Sequential(
                        torch.nn.Linear(fusion_dim, 256),
                        torch.nn.LayerNorm(256),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.10),
                        torch.nn.Linear(256, 128),
                        torch.nn.GELU(),
                    )
                    self.delta_head = torch.nn.Linear(128, 1)
                self.win_head = torch.nn.Linear(128, 1)
                self.loss_head = torch.nn.Linear(128, 1)

            def _route_embeddings(self, link_embeddings: Any, route_link_mask: Any, route_basic_features: Any) -> Any:
                mask = route_link_mask.unsqueeze(-1)
                denom = mask.sum(dim=2).clamp_min(1.0)
                mean_pool = (link_embeddings[:, None, :, :] * mask).sum(dim=2) / denom
                masked_links = link_embeddings[:, None, :, :].masked_fill(~route_link_mask.unsqueeze(-1).bool(), -1e9)
                max_pool = masked_links.max(dim=2).values
                max_pool = torch.where(max_pool < -1e8, torch.zeros_like(max_pool), max_pool)
                return self.route_pool(torch.cat([mean_pool, max_pool, route_basic_features], dim=-1))

            def forward(
                self,
                *,
                node_features: Any,
                link_features: Any,
                global_features: Any,
                edge_index: Any,
                request_features: Any,
                spectrum_tensors: Any,
                action_features: Any,
                route_link_mask: Any,
                route_basic_features: Any,
                block_bounds: Any,
            ) -> dict[str, Any]:
                batch, n_max = action_features.shape[:2]
                h_global, h_links = self.gnn(node_features, link_features, global_features, edge_index)
                h_route = self._route_embeddings(h_links, route_link_mask, route_basic_features)
                h_req = self.request_encoder(request_features)[:, None, :].expand(-1, n_max, -1)
                h_action = self.action_encoder(action_features)
                h_block = self.slot_cnn(
                    spectrum_tensors.reshape(batch * n_max, spectrum_tensors.shape[2], spectrum_tensors.shape[3]),
                    block_bounds.reshape(batch * n_max, 2),
                ).reshape(batch, n_max, -1)
                h_global_rep = h_global[:, None, :].expand(-1, n_max, -1)
                fused = torch.cat([h_global_rep, h_route, h_block, h_req, h_action], dim=-1)
                hidden = self.trunk(fused)
                return {
                    "win_logit": self.win_head(hidden).squeeze(-1),
                    "loss_logit": self.loss_head(hidden).squeeze(-1),
                    "delta": self.delta_head(hidden).squeeze(-1),
                }

        return _ThreeHeadCandidateJudge()


class NeuralThreeHeadOverridePolicy:
    """High-precision neural override policy calibrated by OOF counterfactual labels."""

    def __init__(
        self,
        *,
        model: Any,
        torch: Any,
        device: str,
        base_policy: str,
        thresholds: dict[str, float],
        target_scale: float,
        win_score_weight: float,
        loss_score_weight: float,
        live_gate: LiveOverrideGate | None = None,
    ) -> None:
        self.model = model
        self.torch = torch
        self.device = str(device)
        self.base_policy = str(base_policy)
        self.thresholds = dict(thresholds)
        self.target_scale = float(target_scale)
        self.win_score_weight = float(win_score_weight)
        self.loss_score_weight = float(loss_score_weight)
        self.live_gate = live_gate

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str,
        base_policy: str = "energy-aware-ksp-bm-ff",
        live_gate_path: str | Path | None = None,
    ) -> "NeuralThreeHeadOverridePolicy":
        from cse2026.ong_solver.models import require_torch

        torch = require_torch()
        checkpoint = torch.load(Path(path), map_location=str(device), weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Three-head override checkpoint must be a dictionary")
        checkpoint_config = checkpoint.get("config") or {}
        model_info = checkpoint.get("model_info") or {}
        hidden_dim = int(checkpoint_config.get("hidden_dim", model_info.get("hidden_dim", 128)))
        action_feature_dim = int(model_info.get("action_feature_dim", 10))
        init_from_q_head = bool(model_info.get("init_from_q_head", True))
        model = ThreeHeadCandidateJudge.build(
            action_feature_dim=action_feature_dim,
            hidden_dim=hidden_dim,
            init_from_q_head=init_from_q_head,
            torch=torch,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(str(device))
        model.eval()
        thresholds = checkpoint.get("thresholds") or {}
        return cls(
            model=model,
            torch=torch,
            device=str(device),
            base_policy=str(base_policy),
            thresholds={
                "win_threshold": float(thresholds.get("win_threshold", 1.0)),
                "loss_threshold": float(thresholds.get("loss_threshold", 0.0)),
                "delta_margin": float(thresholds.get("delta_margin", 0.0)),
            },
            target_scale=float(checkpoint_config.get("target_scale", 4.0)),
            win_score_weight=float(checkpoint_config.get("win_score_weight", 1.0)),
            loss_score_weight=float(checkpoint_config.get("loss_score_weight", 3.0)),
            live_gate=None if live_gate_path is None else LiveOverrideGate.load(live_gate_path),
        )

    def _predict(self, batch: Any, solver_config: SolverConfig) -> dict[str, np.ndarray]:
        arrays = _batch_to_arrays(batch, solver_config)
        torch = self.torch
        with torch.no_grad():
            outputs = self.model(
                node_features=torch.as_tensor(arrays["node_features"][None, ...], dtype=torch.float32, device=self.device),
                link_features=torch.as_tensor(arrays["link_features"][None, ...], dtype=torch.float32, device=self.device),
                global_features=torch.as_tensor(arrays["global_features"][None, ...], dtype=torch.float32, device=self.device),
                edge_index=torch.as_tensor(batch.state.edge_index, dtype=torch.long, device=self.device),
                request_features=torch.as_tensor(arrays["request_features"][None, ...], dtype=torch.float32, device=self.device),
                spectrum_tensors=torch.as_tensor(arrays["spectrum_tensors"][None, ...], dtype=torch.float32, device=self.device),
                action_features=torch.as_tensor(arrays["action_features"][None, ...], dtype=torch.float32, device=self.device),
                route_link_mask=torch.as_tensor(arrays["route_link_mask"][None, ...], dtype=torch.float32, device=self.device),
                route_basic_features=torch.as_tensor(
                    arrays["route_basic_features"][None, ...],
                    dtype=torch.float32,
                    device=self.device,
                ),
                block_bounds=torch.as_tensor(arrays["block_bounds"][None, ...], dtype=torch.float32, device=self.device),
            )
        win_prob = torch.sigmoid(outputs["win_logit"]).detach().cpu().numpy().reshape(-1).astype(np.float32)
        loss_prob = torch.sigmoid(outputs["loss_logit"]).detach().cpu().numpy().reshape(-1).astype(np.float32)
        delta_pred = (
            outputs["delta"].detach().cpu().numpy().reshape(-1).astype(np.float32) * float(self.target_scale)
        )
        score = delta_pred + float(self.win_score_weight) * win_prob - float(self.loss_score_weight) * loss_prob
        return {
            "win_prob": win_prob,
            "loss_prob": loss_prob,
            "delta_pred": delta_pred.astype(np.float32),
            "score": score.astype(np.float32),
            "candidate_mask": np.asarray(arrays["candidate_mask"], dtype=bool),
        }

    def decision_details(self, batch: Any, solver_config: SolverConfig) -> dict[str, Any]:
        n_max = int(solver_config.n_max)
        valid = np.flatnonzero(np.asarray(batch.candidate_mask[:n_max], dtype=bool))
        if valid.size == 0:
            return {
                "base_index": -1,
                "selected_index": -1,
                "override_applied": False,
                "override_probability": 0.0,
                "valid_indices": valid,
                "eligible": np.zeros((n_max,), dtype=bool),
                "eligible_count": 0,
                "prediction": {
                    "win_prob": np.zeros((n_max,), dtype=np.float32),
                    "loss_prob": np.ones((n_max,), dtype=np.float32),
                    "delta_pred": np.zeros((n_max,), dtype=np.float32),
                    "score": np.zeros((n_max,), dtype=np.float32),
                    "candidate_mask": np.zeros((n_max,), dtype=bool),
                },
            }
        base_index = int(select_tree_base_index(batch, n_max, self.base_policy))
        if base_index < 0 or base_index >= n_max or not bool(batch.candidate_mask[base_index]):
            base_index = int(valid[0])

        pred = self._predict(batch, solver_config)
        candidate_mask = np.asarray(pred["candidate_mask"][:n_max], dtype=bool)
        eligible = (
            candidate_mask
            & (np.asarray(pred["win_prob"][:n_max]) >= float(self.thresholds["win_threshold"]))
            & (np.asarray(pred["loss_prob"][:n_max]) <= float(self.thresholds["loss_threshold"]))
            & (np.asarray(pred["delta_pred"][:n_max]) >= float(self.thresholds["delta_margin"]))
        ).copy()
        eligible[base_index] = False
        if not bool(eligible.any()):
            selected_index = int(base_index)
        else:
            scores = np.where(eligible, np.asarray(pred["score"][:n_max], dtype=np.float32), -1.0e9)
            selected_index = int(np.argmax(scores))
        return {
            "base_index": int(base_index),
            "selected_index": int(selected_index),
            "override_applied": bool(selected_index != int(base_index)),
            "override_probability": float(pred["win_prob"][selected_index]) if selected_index >= 0 else 0.0,
            "valid_indices": valid,
            "eligible": eligible,
            "eligible_count": int(eligible.sum()),
            "prediction": pred,
        }

    def select_index(self, batch: Any, solver_config: SolverConfig) -> tuple[int, bool, float]:
        details = self.decision_details(batch, solver_config)
        selected_index = int(details["selected_index"])
        if selected_index < 0:
            return -1, False, 0.0
        if bool(details["override_applied"]) and self.live_gate is not None:
            gate = self.live_gate.decision(batch, solver_config, details)
            if not bool(gate["allow"]):
                return int(details["base_index"]), False, float(gate["loss_score"])
        return (
            selected_index,
            bool(details["override_applied"]),
            float(details["override_probability"]),
        )
