from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .lookahead_override_features import (
    OVERRIDE_FEATURE_NAMES,
    candidate_feature_matrix,
    candidate_indices_for_topn,
    select_q_head_index,
)


TREE_RANKER_POLICIES = (
    "xgboost_candidate_ranker",
    "lightgbm_candidate_ranker",
    "xgboost_candidate_ranker_old10",
    "xgboost_candidate_ranker_risk5",
    "lightgbm_candidate_ranker_old5",
    "lightgbm_candidate_ranker_old10",
    "torch_dqn_candidate_ranker_distill_old10",
)
RUNTIME_FEATURE_NAMES = ("is_base_runtime", "energy_rank_norm", "energy_rank_delta", "pool_size_norm")
BASE_RAW_FEATURE_NAMES = (
    "candidate_index_norm",
    "valid_candidates_norm",
    "is_j_total",
    "energy_norm",
    "fragmentation_after",
    "delta_fragmentation",
    "largest_free_block_norm",
    "small_gap_penalty",
    "delay_norm",
    "qot_margin",
    "qot_risk",
    "compactness",
    "route_length_norm",
    "hop_count_norm",
    "width_norm",
    "j_total",
    "request_bit_rate_norm",
    "request_holding_norm",
    "global_load",
    "global_fragmentation",
)
ADVANTAGE_BASE_RAW_FEATURE_INDICES = tuple(OVERRIDE_FEATURE_NAMES.index(name) for name in BASE_RAW_FEATURE_NAMES)
ADVANTAGE_FEATURE_NAMES = (
    [f"candidate_{name}" for name in OVERRIDE_FEATURE_NAMES]
    + [f"base_{name}" for name in BASE_RAW_FEATURE_NAMES]
    + ["ranker_score", "ranker_margin"]
)
RISK_SELECTOR_EXTRA_FEATURE_NAMES = (
    "risk_head_win_prob",
    "risk_head_loss_prob",
    "risk_head_loss_percentile",
    "risk_head_delta_pred",
    "risk_ranker_score",
    "risk_ranker_delta_vs_base",
    "risk_ranker_margin_to_next",
)
DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES = (
    "risk_dqn_score",
    "risk_dqn_delta_vs_base",
    "risk_dqn_margin_to_next",
    "risk_dqn_margin_over_threshold",
)


def _valid_indices(batch: Any) -> np.ndarray:
    return np.flatnonzero(batch.candidate_mask.astype(bool))


def _best_index_by_key(batch: Any, key: Any) -> int:
    valid = _valid_indices(batch)
    if valid.size == 0:
        return -1
    return int(min((int(index) for index in valid), key=lambda index: key(batch.topn[index], index)))


def select_tree_base_index(batch: Any, n_max: int, base_policy: str) -> int:
    policy = str(base_policy or "energy-aware-ksp-bm-ff").strip().lower().replace("_", "-")
    valid = _valid_indices(batch)
    if valid.size == 0:
        return -1
    if policy in {"q-head-heuristic", "q-head", "qhead"}:
        return select_q_head_index(batch, n_max)
    if policy in {"j-total-heuristic", "j-total", "jtotal"}:
        return int(valid[0])
    if policy == "ksp-ff":
        return _best_index_by_key(
            batch,
            lambda candidate, index: (
                int(candidate.route_id),
                int(candidate.b_start),
                int(candidate.modulation_offset),
                int(candidate.w),
                float(candidate.energy_increment),
                int(index),
            ),
        )
    if policy == "ksp-bm-ff":
        return _best_index_by_key(
            batch,
            lambda candidate, index: (
                int(candidate.route_id),
                int(candidate.b_start),
                int(candidate.w),
                -float(candidate.spectral_efficiency),
                float(candidate.energy_increment),
                int(index),
            ),
        )
    if policy == "energy-aware-ksp-bm-ff":
        return _best_index_by_key(
            batch,
            lambda candidate, index: (
                float(candidate.energy_increment),
                int(candidate.route_id),
                int(candidate.b_start),
                int(candidate.w),
                -float(candidate.spectral_efficiency),
                int(index),
            ),
        )
    raise ValueError(f"Unsupported tree ranker base_policy: {base_policy}")


def _passes_safety_guard(candidate: Any, base: Any, guard: dict[str, float | int | bool]) -> bool:
    if not bool(guard.get("enabled", False)):
        return True
    if bool(guard.get("check_fragmentation", True)) and candidate.fragmentation_after > base.fragmentation_after + float(
        guard.get("fragmentation_slack", 0.02)
    ):
        return False
    if bool(guard.get("check_small_gap", True)) and candidate.small_gap_penalty > base.small_gap_penalty + float(
        guard.get("small_gap_slack", 0.02)
    ):
        return False
    if bool(guard.get("check_lmax", True)) and candidate.largest_free_block_after < base.largest_free_block_after - int(
        guard.get("lmax_slack_slots", 4)
    ):
        return False
    if bool(guard.get("check_qot_margin", True)) and candidate.qot_margin_norm < base.qot_margin_norm - float(
        guard.get("qot_margin_slack", 0.08)
    ):
        return False
    if bool(guard.get("check_energy", True)) and candidate.energy_increment > base.energy_increment + float(
        guard.get("energy_slack_w", 80.0)
    ):
        return False
    if bool(guard.get("check_delay", True)) and candidate.delay_ms > base.delay_ms + float(guard.get("delay_slack_ms", 1.0)):
        return False
    return True


def _predict(
    backend: str,
    model: Any,
    features: np.ndarray,
    *,
    feature_names: list[str] | tuple[str, ...] = OVERRIDE_FEATURE_NAMES,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if backend == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(features, feature_names=list(feature_names))
        return np.asarray(model.predict(matrix), dtype=np.float32)
    if backend == "torch":
        import torch

        with torch.inference_mode():
            tensor = torch.as_tensor(features.astype(np.float32), dtype=torch.float32)
            values = model(tensor)
        return np.asarray(values.detach().cpu().reshape(-1), dtype=np.float32)
    return np.asarray(model.predict(features), dtype=np.float32)


def _load_backend_model(backend: str, path: Path) -> Any:
    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(path))
        return model
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    if backend == "torch":
        import torch

        torch.set_num_threads(1)
        model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model
    raise ValueError(f"Unsupported tree ranker backend: {backend}")


def _resolve_model_path(meta_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return meta_path.parent / path


def _uses_runtime_features(feature_names: list[str] | tuple[str, ...]) -> bool:
    names = tuple(str(name) for name in feature_names)
    return all(name in names for name in RUNTIME_FEATURE_NAMES)


def _runtime_feature_matrix(
    *,
    features: np.ndarray,
    kept_indices: list[int],
    base_index: int,
) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, len(RUNTIME_FEATURE_NAMES)), dtype=np.float32)
    energy_values = np.asarray(features[:, OVERRIDE_FEATURE_NAMES.index("energy_norm")], dtype=np.float32)
    order = np.argsort(energy_values, kind="mergesort")
    ranks = np.empty((len(kept_indices),), dtype=np.float32)
    ranks[order] = np.arange(len(kept_indices), dtype=np.float32)
    denom = max(float(len(kept_indices) - 1), 1.0)
    base_position = kept_indices.index(int(base_index)) if int(base_index) in kept_indices else 0
    base_rank = float(ranks[int(base_position)])
    extra = np.zeros((len(kept_indices), len(RUNTIME_FEATURE_NAMES)), dtype=np.float32)
    extra[:, 0] = np.asarray([1.0 if int(index) == int(base_index) else 0.0 for index in kept_indices], dtype=np.float32)
    extra[:, 1] = ranks / denom
    extra[:, 2] = (ranks - base_rank) / denom
    extra[:, 3] = float(len(kept_indices)) / 16.0
    return extra


def _append_runtime_features(
    *,
    features: np.ndarray,
    kept_indices: list[int],
    base_index: int,
    feature_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    if not _uses_runtime_features(feature_names):
        return features.astype(np.float32)
    expected = len(OVERRIDE_FEATURE_NAMES) + len(RUNTIME_FEATURE_NAMES)
    if len(feature_names) != expected:
        raise ValueError(f"Unsupported runtime feature layout: {len(feature_names)} names, expected {expected}")
    return np.concatenate(
        [
            features.astype(np.float32),
            _runtime_feature_matrix(features=features, kept_indices=kept_indices, base_index=base_index),
        ],
        axis=1,
    ).astype(np.float32)


def _percentile_from_reference(value: float, reference: np.ndarray) -> float:
    if reference.size == 0:
        return 1.0
    return float(np.searchsorted(reference, float(value), side="right") / float(reference.size))


def _feature_condition_mask(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    condition: dict[str, Any],
) -> np.ndarray:
    index = {str(name): int(position) for position, name in enumerate(feature_names)}
    feature = str(condition["feature"])
    if feature not in index:
        raise ValueError(f"Unknown context gate feature: {feature}")
    values = np.asarray(features[:, index[feature]], dtype=np.float32)
    threshold = float(condition["value"])
    op = str(condition.get("op", "ge")).strip().lower()
    if op in {"ge", ">="}:
        return values >= threshold
    if op in {"gt", ">"}:
        return values > threshold
    if op in {"le", "<="}:
        return values <= threshold
    if op in {"lt", "<"}:
        return values < threshold
    if op in {"eq", "=="}:
        return np.isclose(values, threshold)
    raise ValueError(f"Unsupported context gate condition op: {op}")


def _context_required_gate_score(
    *,
    advantage_gate: dict[str, Any],
    gate_features: np.ndarray,
    gate_feature_names: list[str] | tuple[str, ...],
    default_min_gate_score: float,
) -> np.ndarray:
    required = np.full((gate_features.shape[0],), float(default_min_gate_score), dtype=np.float32)
    for rule in advantage_gate.get("context_gate_rules") or ():
        conditions = list(rule.get("conditions") or [])
        if not conditions:
            continue
        mask = np.ones((gate_features.shape[0],), dtype=bool)
        for condition in conditions:
            mask &= _feature_condition_mask(gate_features, gate_feature_names, dict(condition))
        if mask.any():
            required[mask] = np.maximum(required[mask], float(rule["min_gate_score"]))
    return required


def _advantage_feature_matrix(
    *,
    features: np.ndarray,
    scores: np.ndarray,
    base_position: int,
    candidate_positions: list[int],
) -> np.ndarray:
    if not candidate_positions:
        return np.zeros((0, len(ADVANTAGE_FEATURE_NAMES)), dtype=np.float32)
    base_features = np.asarray(features[int(base_position)], dtype=np.float32)
    base_raw_features = base_features[list(ADVANTAGE_BASE_RAW_FEATURE_INDICES)]
    base_score = float(scores[int(base_position)])
    rows: list[np.ndarray] = []
    for position in candidate_positions:
        candidate_features = np.asarray(features[int(position)], dtype=np.float32)
        rows.append(
            np.concatenate(
                [
                    candidate_features,
                    base_raw_features,
                    np.asarray(
                        [float(scores[int(position)]), float(scores[int(position)] - base_score)],
                        dtype=np.float32,
                    ),
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)


class TreeCandidateRanker:
    def __init__(
        self,
        *,
        backend: str,
        model: Any,
        selection_mode: str,
        residual_beta: float,
        selection_margin: float,
        base_policy: str = "energy-aware-ksp-bm-ff",
        safety_guard: dict[str, float | int | bool] | None = None,
        advantage_gate: dict[str, Any] | None = None,
        advantage_models: dict[str, Any] | None = None,
        feature_names: list[str] | tuple[str, ...] = OVERRIDE_FEATURE_NAMES,
        candidate_pool: str = "all_topn",
        candidate_pool_top_k: int = 8,
        risk_selector: dict[str, Any] | None = None,
        risk_selector_model: Any | None = None,
        risk_rescue_model: Any | None = None,
        loss_percentile_reference: np.ndarray | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.feature_names = tuple(str(name) for name in feature_names)
        self.selection_mode = str(selection_mode)
        self.residual_beta = float(residual_beta)
        self.selection_margin = float(selection_margin)
        self.base_policy = str(base_policy)
        self.candidate_pool = str(candidate_pool)
        self.candidate_pool_top_k = int(candidate_pool_top_k)
        self.safety_guard = dict(safety_guard or {"enabled": False})
        self.advantage_gate = dict(advantage_gate or {"enabled": False})
        self.advantage_models = dict(advantage_models or {})
        self.risk_selector = dict(risk_selector or {"enabled": False})
        self.risk_selector_model = risk_selector_model
        self.risk_rescue_model = risk_rescue_model
        reference = np.asarray(loss_percentile_reference if loss_percentile_reference is not None else [], dtype=np.float32)
        self.loss_percentile_reference = np.sort(reference[np.isfinite(reference)]).astype(np.float32)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        selection_mode: str | None = None,
        residual_beta: float | None = None,
        selection_margin: float | None = None,
        base_policy: str | None = None,
        safety_guard: dict[str, float | int | bool] | None = None,
        advantage_gate: dict[str, Any] | None = None,
    ) -> "TreeCandidateRanker":
        meta_path = Path(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = str(meta["backend"])
        model_path = Path(str(meta["model_path"]))
        if not model_path.is_absolute():
            model_path = meta_path.parent / model_path
        model = _load_backend_model(backend, model_path)
        feature_names = [str(value) for value in meta.get("feature_names", OVERRIDE_FEATURE_NAMES)]
        loaded_advantage_gate = dict(meta.get("advantage_gate") or {"enabled": False})
        if advantage_gate is not None:
            loaded_advantage_gate.update(advantage_gate)
        advantage_backend = str(loaded_advantage_gate.get("backend", backend))
        advantage_models: dict[str, Any] = {}
        if bool(loaded_advantage_gate.get("enabled", False)):
            for name in ("win", "loss", "delta"):
                path_key = f"{name}_model_path"
                gate_model_path = _resolve_model_path(meta_path, loaded_advantage_gate.get(path_key))
                if gate_model_path is None:
                    raise ValueError(f"advantage_gate requires {path_key}")
                advantage_models[name] = _load_backend_model(advantage_backend, gate_model_path)
        risk_selector = dict(meta.get("risk_selector") or {"enabled": False})
        risk_selector_model = None
        risk_rescue_model = None
        loss_percentile_reference: np.ndarray | None = None
        if bool(risk_selector.get("enabled", False)):
            risk_model_path = _resolve_model_path(meta_path, risk_selector.get("model_path"))
            if risk_model_path is None:
                raise ValueError("risk_selector requires model_path")
            risk_backend = str(risk_selector.get("backend", backend))
            risk_selector_model = _load_backend_model(risk_backend, risk_model_path)
            rescue_model_path = _resolve_model_path(meta_path, risk_selector.get("rescue_model_path"))
            if rescue_model_path is not None:
                rescue_backend = str(risk_selector.get("rescue_backend", risk_backend))
                risk_rescue_model = _load_backend_model(rescue_backend, rescue_model_path)
            reference_path = _resolve_model_path(meta_path, risk_selector.get("loss_percentile_reference_path"))
            if reference_path is not None:
                loss_percentile_reference = np.asarray(np.load(reference_path), dtype=np.float32)
        return cls(
            backend=backend,
            model=model,
            feature_names=feature_names,
            selection_mode=str(selection_mode or meta.get("selection_mode", "pure")),
            residual_beta=float(residual_beta if residual_beta is not None else meta.get("residual_beta", 0.05)),
            selection_margin=float(selection_margin if selection_margin is not None else meta.get("selection_margin", 0.0)),
            base_policy=str(base_policy or meta.get("base_policy", "energy-aware-ksp-bm-ff")),
            candidate_pool=str(meta.get("candidate_pool", "all_topn")),
            candidate_pool_top_k=int(meta.get("candidate_pool_top_k", 8)),
            safety_guard=safety_guard if safety_guard is not None else meta.get("safety_guard", {"enabled": False}),
            advantage_gate=loaded_advantage_gate,
            advantage_models=advantage_models,
            risk_selector=risk_selector,
            risk_selector_model=risk_selector_model,
            risk_rescue_model=risk_rescue_model,
            loss_percentile_reference=loss_percentile_reference,
        )

    def scores(self, features: np.ndarray) -> np.ndarray:
        return _predict(self.backend, self.model, features, feature_names=self.feature_names)

    def _candidate_indices(self, batch: Any, base_index: int) -> list[int]:
        valid = _valid_indices(batch)
        if valid.size == 0:
            return []
        pool = str(self.candidate_pool or "all_topn").strip().lower().replace("-", "_")
        if pool in {"all", "all_topn"}:
            selected = [int(index) for index in valid]
        elif pool in {"energy_topk_hybrid", "quick_topk_hybrid", "quick_top8_hybrid"}:
            top_k = max(1, int(self.candidate_pool_top_k))
            selected_set: set[int] = set()
            if int(base_index) >= 0:
                selected_set.add(int(base_index))
            energy_order = sorted((int(index) for index in valid), key=lambda index: (float(batch.topn[index].energy_increment_norm), index))
            selected_set.update(energy_order[: max(1, min(top_k, len(energy_order)))])
            selected_set.add(int(valid[0]))
            selected_set.add(min((int(index) for index in valid), key=lambda index: (float(batch.topn[index].fragmentation_after), index)))
            selected_set.add(max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].largest_free_block_after), -index)))
            selected_set.add(max((int(index) for index in valid), key=lambda index: (float(batch.topn[index].qot_margin_norm), -index)))
            selected = [int(index) for index in valid if int(index) in selected_set]
        else:
            raise ValueError(f"Unsupported tree ranker candidate_pool: {self.candidate_pool}")
        if int(base_index) >= 0 and int(base_index) not in selected:
            selected.append(int(base_index))
        return selected

    def _select_positive_advantage(
        self,
        *,
        batch: Any,
        kept_indices: list[int],
        features: np.ndarray,
        ranker_features: np.ndarray,
        scores: np.ndarray,
        base_position: int,
        base_index: int,
    ) -> tuple[int, float]:
        if not bool(self.advantage_gate.get("enabled", False)):
            raise ValueError("positive_advantage selection requires enabled advantage_gate metadata")
        missing = sorted(set(("win", "loss", "delta")) - set(self.advantage_models))
        if missing:
            raise ValueError(f"positive_advantage selection is missing models: {missing}")
        if bool(self.advantage_gate.get("fallback_no_override", False)):
            return int(base_index), 0.0

        base_candidate = batch.topn[int(base_index)]
        candidate_positions: list[int] = []
        ranker_margins: list[float] = []
        base_score = float(scores[int(base_position)])
        for position, candidate_index in enumerate(kept_indices):
            if int(position) == int(base_position):
                continue
            margin = float(scores[int(position)] - base_score)
            candidate = batch.topn[int(candidate_index)]
            if not _passes_safety_guard(candidate, base_candidate, self.safety_guard):
                continue
            candidate_positions.append(int(position))
            ranker_margins.append(float(margin))
        if not candidate_positions:
            return int(base_index), 0.0

        feature_source = str(self.advantage_gate.get("feature_source", "advantage_features")).strip().lower()
        gate_feature_names = tuple(str(name) for name in self.advantage_gate.get("feature_names", ADVANTAGE_FEATURE_NAMES))
        if feature_source in {"ranker_features", "candidate_features"}:
            gate_features = ranker_features[candidate_positions].astype(np.float32)
        else:
            gate_features = _advantage_feature_matrix(
                features=features,
                scores=scores,
                base_position=base_position,
                candidate_positions=candidate_positions,
            )
            gate_feature_names = tuple(ADVANTAGE_FEATURE_NAMES)
        advantage_backend = str(self.advantage_gate.get("backend", self.backend))
        win_prob = _predict(
            advantage_backend,
            self.advantage_models["win"],
            gate_features,
            feature_names=gate_feature_names,
        )
        loss_prob = _predict(
            advantage_backend,
            self.advantage_models["loss"],
            gate_features,
            feature_names=gate_feature_names,
        )
        delta_pred = _predict(
            advantage_backend,
            self.advantage_models["delta"],
            gate_features,
            feature_names=gate_feature_names,
        )
        ranker_margins_array = np.asarray(ranker_margins, dtype=np.float32)
        gate_score = (
            float(self.advantage_gate.get("delta_weight", 1.0)) * delta_pred
            + float(self.advantage_gate.get("win_weight", 1.0)) * win_prob
            - float(self.advantage_gate.get("loss_weight", 2.0)) * loss_prob
            + float(self.advantage_gate.get("ranker_margin_weight", 0.0)) * ranker_margins_array
        )
        passed = (win_prob >= float(self.advantage_gate.get("min_win_prob", 0.5))) & (
            delta_pred >= float(self.advantage_gate.get("min_delta_pred", 0.0))
        )
        if bool(self.advantage_gate.get("check_loss_prob", True)):
            passed &= loss_prob <= float(self.advantage_gate.get("max_loss_prob", 0.05))
        if "min_gate_score" in self.advantage_gate:
            required_gate_score = _context_required_gate_score(
                advantage_gate=self.advantage_gate,
                gate_features=gate_features,
                gate_feature_names=gate_feature_names,
                default_min_gate_score=float(self.advantage_gate.get("min_gate_score", -np.inf)),
            )
            passed &= gate_score >= required_gate_score
        passed_positions = np.flatnonzero(passed)
        if passed_positions.size == 0:
            return int(base_index), 0.0
        best_local = int(
            min(
                (int(position) for position in passed_positions),
                key=lambda position: (-float(gate_score[position]), int(kept_indices[candidate_positions[position]])),
            )
        )
        selected_position = int(candidate_positions[best_local])
        if bool(self.risk_selector.get("enabled", False)):
            if self.risk_selector_model is None:
                raise ValueError("risk_selector is enabled but its model was not loaded")
            selected_loss_percentile = _percentile_from_reference(
                float(loss_prob[best_local]),
                self.loss_percentile_reference,
            )
            order = np.argsort(scores, kind="mergesort")
            best_position = int(order[-1])
            second_score = float(scores[int(order[-2])]) if len(order) > 1 else float(scores[best_position])
            best_score = float(scores[best_position])
            selected_score = float(scores[selected_position])
            other_best = second_score if selected_position == best_position else best_score
            risk_features = np.concatenate(
                [
                    ranker_features[selected_position].astype(np.float32),
                    np.asarray(
                        [
                            float(win_prob[best_local]),
                            float(loss_prob[best_local]),
                            float(selected_loss_percentile),
                            float(delta_pred[best_local]),
                            float(selected_score),
                            float(selected_score - base_score),
                            float(selected_score - other_best),
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).reshape(1, -1)
            risk_backend = str(self.risk_selector.get("backend", self.backend))
            risk_feature_names = tuple(str(name) for name in self.risk_selector.get("feature_names", ()))
            if not risk_feature_names:
                risk_feature_names = tuple(self.feature_names) + RISK_SELECTOR_EXTRA_FEATURE_NAMES
            risk_score = float(
                _predict(
                    risk_backend,
                    self.risk_selector_model,
                    risk_features.astype(np.float32),
                    feature_names=risk_feature_names,
                )[0]
            )
            if risk_score > float(self.risk_selector.get("score_cutoff", 0.0)):
                return int(base_index), risk_score
        return int(kept_indices[selected_position]), float(win_prob[best_local])

    def _base_residual_risk_score(
        self,
        *,
        ranker_features: np.ndarray,
        scores: np.ndarray,
        selected_position: int,
        base_position: int,
    ) -> float:
        if not bool(self.risk_selector.get("enabled", False)):
            return -np.inf
        if self.risk_selector_model is None:
            raise ValueError("risk_selector is enabled but its model was not loaded")

        order = np.argsort(scores, kind="mergesort")
        best_position = int(order[-1])
        second_score = float(scores[int(order[-2])]) if len(order) > 1 else float(scores[best_position])
        best_score = float(scores[best_position])
        selected_score = float(scores[int(selected_position)])
        base_score = float(scores[int(base_position)])
        other_best = second_score if int(selected_position) == best_position else best_score
        margin = float(selected_score - base_score)
        risk_features = np.concatenate(
            [
                ranker_features[int(selected_position)].astype(np.float32),
                np.asarray(
                    [
                        selected_score,
                        margin,
                        float(selected_score - other_best),
                        float(margin - self.selection_margin),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).reshape(1, -1)
        risk_backend = str(self.risk_selector.get("backend", self.backend))
        risk_feature_names = tuple(str(name) for name in self.risk_selector.get("feature_names", ()))
        if not risk_feature_names:
            risk_feature_names = tuple(self.feature_names) + DQN_RISK_SELECTOR_EXTRA_FEATURE_NAMES
        risk_score = float(
            _predict(
                risk_backend,
                self.risk_selector_model,
                risk_features.astype(np.float32),
                feature_names=risk_feature_names,
            )[0]
        )
        score_cutoff = float(self.risk_selector.get("score_cutoff", 0.0))
        if risk_score <= score_cutoff or self.risk_rescue_model is None:
            return risk_score

        rescue_backend = str(self.risk_selector.get("rescue_backend", risk_backend))
        rescue_feature_names = tuple(str(name) for name in self.risk_selector.get("rescue_feature_names", risk_feature_names))
        rescue_score = float(
            _predict(
                rescue_backend,
                self.risk_rescue_model,
                risk_features.astype(np.float32),
                feature_names=rescue_feature_names,
            )[0]
        )
        if (
            rescue_score >= float(self.risk_selector.get("rescue_score_cutoff", math.inf))
            and margin - float(self.selection_margin)
            >= float(self.risk_selector.get("rescue_min_margin_over_threshold", -math.inf))
            and risk_score <= float(self.risk_selector.get("rescue_max_risk_score", math.inf))
        ):
            return score_cutoff
        return risk_score

    def select_index(self, batch: Any, n_max: int) -> tuple[int, float]:
        base_index = select_tree_base_index(batch, n_max, self.base_policy)
        candidate_indices = self._candidate_indices(batch, base_index)
        features, kept_indices = candidate_feature_matrix(
            batch=batch,
            candidate_indices=candidate_indices,
            n_max=n_max,
            reference_index=base_index,
        )
        if not kept_indices:
            return -1, 0.0
        if base_index not in kept_indices:
            return -1, 0.0
        base_position = kept_indices.index(base_index)
        ranker_features = _append_runtime_features(
            features=features,
            kept_indices=kept_indices,
            base_index=base_index,
            feature_names=self.feature_names,
        )
        scores = self.scores(ranker_features)

        if self.selection_mode == "pure":
            best_position = int(np.argmax(scores))
            return int(kept_indices[best_position]), float(scores[best_position] - scores[base_position])

        if self.selection_mode in {"residual", "guarded_residual"}:
            raise ValueError(
                "residual tree-ranker selection modes are disabled for the energy-aware base pipeline; "
                "use pure, guarded/base_residual, or positive_advantage"
            )

        if self.selection_mode in {"guarded", "base_residual", "base_residual_margin", "dqn_base_residual"}:
            best_position = int(np.argmax(scores))
            margin = float(scores[best_position] - scores[base_position])
            if margin < float(self.selection_margin):
                return int(base_index), margin
            if not _passes_safety_guard(batch.topn[int(kept_indices[best_position])], batch.topn[int(base_index)], self.safety_guard):
                return int(base_index), margin
            risk_score = self._base_residual_risk_score(
                ranker_features=ranker_features,
                scores=scores,
                selected_position=best_position,
                base_position=base_position,
            )
            if risk_score > float(self.risk_selector.get("score_cutoff", 0.0)):
                return int(base_index), risk_score
            return int(kept_indices[best_position]), margin

        if self.selection_mode in {"advantage", "positive_advantage"}:
            return self._select_positive_advantage(
                batch=batch,
                kept_indices=kept_indices,
                features=features,
                ranker_features=ranker_features,
                scores=scores,
                base_position=base_position,
                base_index=base_index,
            )

        raise ValueError(f"Unsupported tree ranker selection_mode: {self.selection_mode}")
