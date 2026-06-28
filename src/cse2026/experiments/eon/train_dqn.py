from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import CandidateBatch, SolverConfig
from cse2026.ong_solver.common import normalize_q_score_mode

from ..config import ExperimentConfig
from .ong_solver_eval import (
    ACTION_FEATURE_COLUMNS,
    _build_batch,
    _cnn_lookup,
    _edge_lengths_from_dataset,
    _finite_float,
    _offline_state_views,
)


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits(config: ExperimentConfig) -> tuple[str, str, str]:
    values = tuple(config.splits or ("train", "val", "test"))
    train = values[0] if len(values) >= 1 else "train"
    val = values[1] if len(values) >= 2 else "val"
    test = values[2] if len(values) >= 3 else "test"
    return train, val, test


def _raw_bool(config: ExperimentConfig, key: str, default: bool) -> bool:
    value = config.resolved.get(key, config.raw.get(key, default))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _raw_int(config: ExperimentConfig, key: str, default: int) -> int:
    return int(config.resolved.get(key, config.raw.get(key, default)))


def _raw_float(config: ExperimentConfig, key: str, default: float) -> float:
    return float(config.resolved.get(key, config.raw.get(key, default)))


def _raw_str(config: ExperimentConfig, key: str, default: str) -> str:
    return str(config.resolved.get(key, config.raw.get(key, default)))


def _raw_csv_set(config: ExperimentConfig, key: str, default: tuple[str, ...]) -> set[str]:
    value = config.resolved.get(key, config.raw.get(key, default))
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(item) for item in default}


def _device(config: ExperimentConfig, torch: Any) -> str:
    requested = str(config.resolved.get("device", config.device))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _resolve_path(config: ExperimentConfig, key: str) -> Path | None:
    value = config.resolved.get(key, config.raw.get(key))
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _parse_state_id(value: Any) -> tuple[str, int] | None:
    if value is None:
        return None
    text = str(value)
    if ":" not in text:
        return None
    episode_id, request_text = text.rsplit(":", 1)
    try:
        return episode_id, int(request_text)
    except ValueError:
        return None


def _iter_batches(size: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def _batch_to_arrays(batch: CandidateBatch, cfg: SolverConfig) -> dict[str, np.ndarray]:
    n_max = int(cfg.n_max)
    edge_count = int(batch.link_features.shape[0])
    route_link_mask = np.zeros((n_max, edge_count), dtype=np.float32)
    route_basic = np.zeros((n_max, 2), dtype=np.float32)
    block_bounds = np.zeros((n_max, 2), dtype=np.float32)
    for index, candidate in enumerate(batch.topn[:n_max]):
        for link_id in candidate.route_link_ids:
            link_index = int(link_id)
            if 0 <= link_index < edge_count:
                route_link_mask[index, link_index] = 1.0
        route_basic[index, 0] = float(candidate.route_length_km / max(cfg.max_route_length_norm_km, 1e-9))
        route_basic[index, 1] = float(candidate.hop_count / 8.0)
        block_bounds[index, 0] = float(candidate.b_start)
        block_bounds[index, 1] = float(candidate.w)
    return {
        "node_features": np.asarray(batch.node_features, dtype=np.float32),
        "link_features": np.asarray(batch.link_features, dtype=np.float32),
        "global_features": np.asarray(batch.global_features, dtype=np.float32),
        "request_features": np.asarray(batch.request_features, dtype=np.float32),
        "spectrum_tensors": np.asarray(batch.spectrum_tensors, dtype=np.float32),
        "action_features": np.asarray(batch.action_features, dtype=np.float32),
        "route_link_mask": route_link_mask,
        "route_basic_features": route_basic,
        "block_bounds": block_bounds,
        "candidate_mask": np.asarray(batch.candidate_mask > 0.0, dtype=np.bool_),
    }


def _zero_arrays(
    *,
    node_count: int,
    edge_count: int,
    slots: int,
    channels: int,
    n_max: int,
    action_feature_dim: int,
) -> dict[str, np.ndarray]:
    return {
        "node_features": np.zeros((node_count, 4), dtype=np.float32),
        "link_features": np.zeros((edge_count, 8), dtype=np.float32),
        "global_features": np.zeros((8,), dtype=np.float32),
        "request_features": np.zeros((3,), dtype=np.float32),
        "spectrum_tensors": np.zeros((n_max, channels, slots), dtype=np.float32),
        "action_features": np.zeros((n_max, action_feature_dim), dtype=np.float32),
        "route_link_mask": np.zeros((n_max, edge_count), dtype=np.float32),
        "route_basic_features": np.zeros((n_max, 2), dtype=np.float32),
        "block_bounds": np.zeros((n_max, 2), dtype=np.float32),
        "candidate_mask": np.zeros((n_max,), dtype=np.bool_),
    }


@dataclass
class DqnOfflineSplit:
    dataset_path: Path
    split: str
    cfg: SolverConfig
    graphs: np.lib.npyio.NpzFile
    x_spec: np.ndarray
    candidates: pd.DataFrame
    dqn: pd.DataFrame
    candidate_groups: dict[tuple[str, int], pd.DataFrame]
    sample_index_by_key: dict[tuple[str, int], int]
    row_position_by_key: dict[tuple[str, int], int]
    states: dict[tuple[str, int], Any]
    cnn_rows_by_candidate: dict[tuple[Any, ...], int]
    valid_indices: np.ndarray
    skipped_forced_block: int
    skipped_invalid_action: int
    sampling_stats: dict[str, Any] = field(default_factory=dict)
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = field(default_factory=dict)
    cnn_compare: dict[str, int] = field(default_factory=lambda: {"compared": 0, "matched": 0})

    @property
    def edge_index(self) -> np.ndarray:
        return np.asarray(self.graphs["edge_index"], dtype=np.int64)

    @property
    def empty_arrays(self) -> dict[str, np.ndarray]:
        return _zero_arrays(
            node_count=int(self.graphs["node_features"].shape[1]),
            edge_count=int(self.graphs["link_features"].shape[1]),
            slots=int(self.x_spec.shape[2]),
            channels=int(self.x_spec.shape[1]),
            n_max=int(self.cfg.n_max),
            action_feature_dim=len(ACTION_FEATURE_COLUMNS),
        )

    def arrays_for_key(self, key: tuple[str, int]) -> dict[str, np.ndarray]:
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        group = self.candidate_groups.get(key)
        state = self.states.get(key)
        sample_index = self.sample_index_by_key.get(key)
        if group is None or state is None or sample_index is None:
            raise KeyError(f"{self.split}: missing offline state/candidates for {key[0]}:{key[1]}")
        batch = _build_batch(
            graphs=self.graphs,
            x_spec=self.x_spec,
            cnn_rows_by_candidate=self.cnn_rows_by_candidate,
            group=group,
            sample_index=sample_index,
            state=state,
            cfg=self.cfg,
            cnn_compare=self.cnn_compare,
        )
        arrays = _batch_to_arrays(batch, self.cfg)
        self.cache[key] = arrays
        return arrays

    def candidate_row(self, key: tuple[str, int], topn_index: int) -> pd.Series | None:
        group = self.candidate_groups.get(key)
        if group is None:
            return None
        rows = group[group["topn_index"] == int(topn_index)]
        if rows.empty:
            return None
        return rows.iloc[0]

    def candidate_metric_vector(self, key: tuple[str, int], metric: str) -> np.ndarray:
        values = np.full((int(self.cfg.n_max),), -np.inf, dtype=np.float32)
        group = self.candidate_groups.get(key)
        if group is None or metric not in group.columns:
            return values
        for row in group.itertuples(index=False):
            topn_index = int(getattr(row, "topn_index"))
            if 0 <= topn_index < int(self.cfg.n_max):
                value = _finite_float(getattr(row, metric), default=math.nan)
                if math.isfinite(value):
                    values[topn_index] = float(value)
        return values

    def n_step_target_view(
        self,
        row_position: int,
        *,
        gamma: float,
        n_step_return: int,
    ) -> tuple[float, float, bool, tuple[str, int] | None]:
        reward_sum = 0.0
        discount = 1.0
        current_position = int(row_position)
        bootstrap_key: tuple[str, int] | None = None
        terminal = False
        steps = max(1, int(n_step_return))

        for step in range(steps):
            row = self.dqn.iloc[current_position]
            reward_sum += discount * float(row.reward)
            terminal = bool(row.done)
            next_key = _parse_state_id(row.next_state_id)
            discount *= float(gamma)
            bootstrap_key = None if terminal else next_key
            if terminal or next_key is None:
                break
            if step == steps - 1:
                break
            next_position = self.row_position_by_key.get(next_key)
            if next_position is None:
                break
            current_position = int(next_position)

        return float(reward_sum), float(discount), bool(terminal), bootstrap_key


def _transition_limit(config: ExperimentConfig, split: str, default: int = 0) -> int:
    return _raw_int(config, f"{split}_transition_limit", _raw_int(config, "transition_limit", default))


def _candidate_hard_case_mask(
    dqn: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    min_candidates: int,
    small_margin: float,
) -> np.ndarray:
    hard_by_key: dict[tuple[str, int], bool] = {}
    if candidates.empty:
        return np.zeros((len(dqn),), dtype=np.bool_)
    for (episode_id, request_id), group in candidates.groupby(["episode_id", "request_id"], sort=False):
        real = group[group["candidate_mask"].astype(bool)].copy()
        if real.empty:
            hard_by_key[(str(episode_id), int(request_id))] = False
            continue
        q_order = real.sort_values(["q_head_score", "j_total"], ascending=[False, True])
        j_order = real.sort_values(["j_total", "energy_increment", "route_id", "b_start"], ascending=[True, True, True, True])
        q_index = int(q_order.iloc[0]["topn_index"])
        j_index = int(j_order.iloc[0]["topn_index"])
        q_scores = q_order["q_head_score"].to_numpy(dtype=np.float64)
        margin = float(q_scores[0] - q_scores[1]) if len(q_scores) > 1 else math.inf
        hard_by_key[(str(episode_id), int(request_id))] = bool(
            q_index != j_index or (len(real) >= int(min_candidates) and margin <= float(small_margin))
        )
    return np.asarray(
        [hard_by_key.get((str(row.episode_id), int(row.request_id)), False) for row in dqn.itertuples(index=False)],
        dtype=np.bool_,
    )


def _sample_transitions(
    dqn: pd.DataFrame,
    *,
    candidates: pd.DataFrame,
    limit: int,
    seed: int,
    sampling_mode: str = "uniform",
    hard_case_fraction: float = 0.0,
    hard_case_min_candidates: int = 4,
    hard_case_small_margin: float = 0.03,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mode = str(sampling_mode or "uniform").strip().lower()
    hard_mask = _candidate_hard_case_mask(
        dqn,
        candidates,
        min_candidates=int(hard_case_min_candidates),
        small_margin=float(hard_case_small_margin),
    )
    stats: dict[str, Any] = {
        "mode": mode,
        "source_rows": int(len(dqn)),
        "hard_case_rows": int(hard_mask.sum()),
        "hard_case_fraction_requested": float(hard_case_fraction),
        "hard_case_min_candidates": int(hard_case_min_candidates),
        "hard_case_small_margin": float(hard_case_small_margin),
    }
    if limit <= 0 or len(dqn) <= limit:
        sampled = dqn.reset_index(drop=True)
        stats.update({"sampled_rows": int(len(sampled)), "sampled_hard_case_rows": int(hard_mask.sum())})
        return sampled, stats
    if mode not in {"uniform", "hard_case"}:
        raise ValueError(f"Unsupported transition_sampling_mode: {sampling_mode}")
    if mode == "uniform" or hard_case_fraction <= 0.0 or not bool(hard_mask.any()):
        sampled = dqn.sample(n=int(limit), random_state=int(seed)).sort_values(["episode_id", "request_id"]).reset_index(drop=True)
        sampled_keys = {(str(row.episode_id), int(row.request_id)) for row in sampled.itertuples(index=False)}
        sampled_hard = int(
            sum(
                1
                for row, is_hard in zip(dqn.itertuples(index=False), hard_mask)
                if is_hard and (str(row.episode_id), int(row.request_id)) in sampled_keys
            )
        )
        stats.update({"sampled_rows": int(len(sampled)), "sampled_hard_case_rows": sampled_hard})
        return sampled, stats

    rng = np.random.default_rng(int(seed))
    all_positions = np.arange(len(dqn), dtype=np.int64)
    hard_positions = np.flatnonzero(hard_mask)
    hard_take = min(int(hard_positions.size), int(round(int(limit) * float(hard_case_fraction))))
    chosen: list[int] = []
    if hard_take > 0:
        chosen.extend(int(value) for value in rng.choice(hard_positions, size=hard_take, replace=False))
    chosen_set = set(chosen)
    remaining = np.asarray([int(value) for value in all_positions if int(value) not in chosen_set], dtype=np.int64)
    rest_take = int(limit) - len(chosen)
    if rest_take > 0 and remaining.size:
        chosen.extend(int(value) for value in rng.choice(remaining, size=min(rest_take, int(remaining.size)), replace=False))
    sampled = dqn.iloc[np.asarray(chosen, dtype=np.int64)].sort_values(["episode_id", "request_id"]).reset_index(drop=True)
    sampled_keys = {(str(row.episode_id), int(row.request_id)) for row in sampled.itertuples(index=False)}
    sampled_hard = int(
        sum(
            1
            for row, is_hard in zip(dqn.itertuples(index=False), hard_mask)
            if is_hard and (str(row.episode_id), int(row.request_id)) in sampled_keys
        )
    )
    stats.update({"sampled_rows": int(len(sampled)), "sampled_hard_case_rows": sampled_hard})
    return sampled, stats


def _filter_state_rows(frame: pd.DataFrame, target_keys: set[tuple[str, int]]) -> pd.DataFrame:
    if frame.empty or not target_keys:
        return frame.iloc[0:0].copy()
    keys = pd.MultiIndex.from_arrays([frame["episode_id"].astype(str), frame["request_id"].astype(int)])
    allowed = pd.MultiIndex.from_tuples(tuple(target_keys))
    return frame.loc[keys.isin(allowed)].reset_index(drop=True)


def _load_split(
    dataset_path: Path,
    split: str,
    cfg: SolverConfig,
    *,
    transition_limit: int = 0,
    seed: int = 0,
    sampling_mode: str = "uniform",
    hard_case_fraction: float = 0.0,
    hard_case_min_candidates: int = 4,
    hard_case_small_margin: float = 0.03,
) -> DqnOfflineSplit:
    graphs = np.load(dataset_path / "gnn" / f"{split}_graphs.npz")
    x_spec = np.load(dataset_path / "cnn" / f"{split}_tensors.npz")["X_spec"].astype(np.float32)
    cnn_index = pd.read_parquet(dataset_path / "cnn" / f"{split}_index.parquet")
    candidates = pd.read_parquet(dataset_path / "candidates" / f"{split}.parquet")
    traffic = pd.read_parquet(dataset_path / "traffic" / f"{split}.parquet")
    dqn_all = pd.read_parquet(dataset_path / "dqn" / f"{split}_transitions.parquet")
    dqn, sampling_stats = _sample_transitions(
        dqn_all,
        candidates=candidates,
        limit=int(transition_limit),
        seed=int(seed),
        sampling_mode=sampling_mode,
        hard_case_fraction=float(hard_case_fraction),
        hard_case_min_candidates=int(hard_case_min_candidates),
        hard_case_small_margin=float(hard_case_small_margin),
    )

    dqn_by_request = {(str(row.episode_id), int(row.request_id)): row for row in dqn_all.itertuples(index=False)}
    sample_ids = [str(value) for value in graphs["sample_ids"]]
    sample_index_by_key: dict[tuple[str, int], int] = {}
    target_keys: set[tuple[str, int]] = set()
    for sample_index, sample_id in enumerate(sample_ids):
        key = _parse_state_id(sample_id)
        if key is not None:
            sample_index_by_key[key] = sample_index
    for row in dqn.itertuples(index=False):
        current_key = (str(row.episode_id), int(row.request_id))
        target_keys.add(current_key)
        next_key = _parse_state_id(row.next_state_id)
        if next_key is not None:
            target_keys.add(next_key)

    edge_lengths = _edge_lengths_from_dataset(dataset_path, candidates, int(graphs["link_features"].shape[1]))
    candidates = _filter_state_rows(candidates, target_keys)
    cnn_index = _filter_state_rows(cnn_index, target_keys)
    states = _offline_state_views(
        traffic=traffic,
        dqn_by_request=dqn_by_request,
        target_keys=target_keys,
        graphs=graphs,
        slots=int(x_spec.shape[2]),
        edge_lengths_km=edge_lengths,
    )
    candidate_groups = {
        (str(episode_id), int(request_id)): group
        for (episode_id, request_id), group in candidates.groupby(["episode_id", "request_id"], sort=False)
    }
    row_position_by_key = {
        (str(row.episode_id), int(row.request_id)): int(position)
        for position, row in enumerate(dqn.itertuples(index=False))
    }

    forced_block = dqn["selected_candidate_index"].to_numpy(dtype=np.int64) < 0
    invalid_action = (
        ~dqn["candidate_mask_valid"].astype(bool)
        | dqn["invalid_action_selected"].astype(bool)
        | dqn["padding_action_selected"].astype(bool)
    ).to_numpy()
    usable: list[int] = []
    for position, row in enumerate(dqn.itertuples(index=False)):
        if forced_block[position] or invalid_action[position]:
            continue
        key = (str(row.episode_id), int(row.request_id))
        selected = int(row.selected_candidate_index)
        if key not in sample_index_by_key or key not in candidate_groups or key not in states:
            continue
        if selected < 0 or selected >= int(cfg.n_max):
            continue
        selected_row = candidate_groups[key][candidate_groups[key]["topn_index"] == selected]
        if selected_row.empty or not bool(int(selected_row.iloc[0].get("candidate_mask", 0))):
            continue
        usable.append(position)

    return DqnOfflineSplit(
        dataset_path=dataset_path,
        split=split,
        cfg=cfg,
        graphs=graphs,
        x_spec=x_spec,
        candidates=candidates,
        dqn=dqn,
        candidate_groups=candidate_groups,
        sample_index_by_key=sample_index_by_key,
        row_position_by_key=row_position_by_key,
        states=states,
        cnn_rows_by_candidate=_cnn_lookup(cnn_index),
        valid_indices=np.asarray(usable, dtype=np.int64),
        skipped_forced_block=int(forced_block.sum()),
        skipped_invalid_action=int(invalid_action.sum()),
        sampling_stats=sampling_stats,
    )


def _sampling_mode_for_split(config: ExperimentConfig, split: str) -> str:
    mode = _raw_str(config, "transition_sampling_mode", "uniform").strip().lower()
    if mode != "hard_case":
        return mode
    allowed = _raw_csv_set(config, "hard_case_sampling_splits", ("train",))
    return "hard_case" if str(split) in allowed else "uniform"


def _apply_reward_override(split: DqnOfflineSplit, config: ExperimentConfig) -> dict[str, Any] | None:
    mode = str(config.resolved.get("reward_override_mode", config.raw.get("reward_override_mode", "stored"))).strip().lower()
    if mode in {"", "stored", "none"}:
        return None
    if mode != "problem_shaped":
        raise ValueError(f"Unsupported reward_override_mode: {mode}")

    dqn = split.dqn.copy()
    accepted = ~dqn["blocked"].astype(bool).to_numpy()
    energy_norm = max(_raw_float(config, "reward_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)), 1e-9)
    delay_norm = max(_raw_float(config, "reward_delay_norm_ms", _raw_float(config, "delay_norm_ms", 50.0)), 1e-9)
    reward = np.where(
        accepted,
        _raw_float(config, "accepted_service_reward", 1.0),
        _raw_float(config, "block_penalty", -1.0),
    ).astype(np.float64)
    if accepted.any():
        reward[accepted] += (
            -_raw_float(config, "reward_energy_weight", 0.30)
            * dqn.loc[accepted, "delta_energy"].to_numpy(dtype=np.float64)
            / energy_norm
        )
        reward[accepted] += (
            -_raw_float(config, "reward_fragmentation_weight", 0.35)
            * dqn.loc[accepted, "fragmentation_after"].to_numpy(dtype=np.float64)
        )
        reward[accepted] += (
            _raw_float(config, "reward_qot_margin_weight", 0.15)
            * dqn.loc[accepted, "qot_margin"].to_numpy(dtype=np.float64)
        )
        reward[accepted] += (
            -_raw_float(config, "reward_delay_weight", 0.10)
            * dqn.loc[accepted, "delay_ms"].to_numpy(dtype=np.float64)
            / delay_norm
        )
    original_reward = dqn["reward"].to_numpy(dtype=np.float64)
    dqn["reward"] = reward.astype(np.float64)
    split.dqn = dqn
    return {
        "mode": mode,
        "split": split.split,
        "mean_original_reward": float(np.mean(original_reward)) if original_reward.size else None,
        "mean_override_reward": float(np.mean(reward)) if reward.size else None,
        "min_override_reward": float(np.min(reward)) if reward.size else None,
        "max_override_reward": float(np.max(reward)) if reward.size else None,
        "accepted_rows": int(accepted.sum()),
        "blocked_rows": int((~accepted).sum()),
    }


def _target_params(config: ExperimentConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "topn_rank_weight": _raw_float(config, "target_topn_rank_weight", 0.0),
        "fragmentation_weight": _raw_float(config, "target_fragmentation_weight", 0.0),
        "lmax_weight": _raw_float(config, "target_lmax_weight", 0.0),
        "small_gap_weight": _raw_float(config, "target_small_gap_weight", 0.0),
        "delta_fragmentation_weight": _raw_float(config, "target_delta_fragmentation_weight", 0.0),
    }
    external_path = _raw_str(config, "lookahead_target_path", "")
    if external_path:
        path = Path(external_path)
        if not path.is_absolute():
            path = project_root() / path
        table = pd.read_csv(path)
        params["external_targets"] = {
            (str(row.episode_id), int(row.request_id)): int(row.oracle_index)
            for row in table.itertuples(index=False)
            if int(row.oracle_index) >= 0
        }
        params["external_target_path"] = str(path)
        params["external_target_count"] = int(len(params["external_targets"]))
    return params


def _target_params_for_metrics(params: dict[str, Any]) -> dict[str, Any]:
    public = dict(params)
    external_targets = public.pop("external_targets", None)
    if external_targets is not None and "external_target_count" not in public:
        public["external_target_count"] = int(len(external_targets))
    return public


def _target_index_for_key(
    split: DqnOfflineSplit,
    key: tuple[str, int],
    *,
    stored_best: int,
    selected: int,
    mode: str,
    params: dict[str, Any],
) -> int:
    target_mode = str(mode or "best").strip().lower()
    if target_mode in {"", "best", "stored_best", "q_head", "qhead"}:
        return int(stored_best)
    if target_mode in {"selected", "recorded", "expert"}:
        return int(selected)
    if target_mode in {"lookahead_oracle", "lookahead_oracle_strict", "external", "external_strict"}:
        target = params.get("external_targets", {}).get(key)
        if target is not None:
            return int(target)
        if target_mode in {"lookahead_oracle_strict", "external_strict"}:
            return -1
        return int(stored_best)

    group = split.candidate_groups.get(key)
    if group is None or group.empty:
        return int(stored_best)
    real = group[group["candidate_mask"].astype(bool)].copy()
    if real.empty:
        return int(stored_best)

    if target_mode in {"j_total", "jtotal", "topn", "j_total_heuristic"}:
        order = real.sort_values(["j_total", "energy_increment", "route_id", "b_start"], ascending=[True, True, True, True])
        return int(order.iloc[0]["topn_index"])

    if target_mode in {"blocking_sensitive", "blocking_sensitive_hybrid", "hybrid_blocking"}:
        slots = int(split.x_spec.shape[2])
        topn_norm = real["topn_index"].to_numpy(dtype=np.float64) / float(max(int(split.cfg.n_max) - 1, 1))
        q_head = real["q_head_score"].to_numpy(dtype=np.float64)
        fragmentation = real["fragmentation_after"].to_numpy(dtype=np.float64)
        lmax = real["largest_free_block_after"].to_numpy(dtype=np.float64) / float(max(slots, 1))
        small_gap = real["small_gap_penalty"].to_numpy(dtype=np.float64)
        delta_frag = real["delta_fragmentation"].to_numpy(dtype=np.float64)
        score = (
            q_head
            - float(params.get("topn_rank_weight", 0.0)) * topn_norm
            - float(params.get("fragmentation_weight", 0.0)) * fragmentation
            + float(params.get("lmax_weight", 0.0)) * lmax
            - float(params.get("small_gap_weight", 0.0)) * small_gap
            - float(params.get("delta_fragmentation_weight", 0.0)) * delta_frag
        )
        if not np.isfinite(score).any():
            return int(stored_best)
        return int(real.iloc[int(np.nanargmax(score))]["topn_index"])

    raise ValueError(f"Unsupported learning target mode: {mode}")


def _stack_state_arrays(arrays: list[dict[str, np.ndarray]], device: str, torch: Any) -> dict[str, Any]:
    keys = (
        "node_features",
        "link_features",
        "global_features",
        "request_features",
        "spectrum_tensors",
        "action_features",
        "route_link_mask",
        "route_basic_features",
        "block_bounds",
    )
    stacked = {
        key: torch.as_tensor(np.stack([item[key] for item in arrays], axis=0), dtype=torch.float32, device=device)
        for key in keys
    }
    stacked["candidate_mask"] = torch.as_tensor(
        np.stack([item["candidate_mask"] for item in arrays], axis=0), dtype=torch.bool, device=device
    )
    return stacked


def _batch_tensors(
    split: DqnOfflineSplit,
    row_indices: np.ndarray,
    device: str,
    torch: Any,
    *,
    gamma: float = 0.95,
    n_step_return: int = 1,
    target_mode: str = "best",
    target_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_arrays: list[dict[str, np.ndarray]] = []
    next_arrays: list[dict[str, np.ndarray]] = []
    selected_indices: list[int] = []
    best_indices: list[int] = []
    q_head_scores: list[np.ndarray] = []
    next_q_head_scores: list[np.ndarray] = []
    rewards: list[float] = []
    discounts: list[float] = []
    done_flags: list[bool] = []
    next_available: list[bool] = []
    empty = split.empty_arrays
    empty_q_head_scores = np.full((int(split.cfg.n_max),), -np.inf, dtype=np.float32)

    for row_position, row in zip(row_indices, split.dqn.iloc[row_indices].itertuples(index=False)):
        current_key = (str(row.episode_id), int(row.request_id))
        current_arrays.append(split.arrays_for_key(current_key))
        selected_indices.append(int(row.selected_candidate_index))
        best_indices.append(
            _target_index_for_key(
                split,
                current_key,
                stored_best=int(row.best_candidate_index),
                selected=int(row.selected_candidate_index),
                mode=target_mode,
                params=target_params or {},
            )
        )
        q_head_scores.append(split.candidate_metric_vector(current_key, "q_head_score"))
        reward, discount, done, next_key = split.n_step_target_view(
            int(row_position),
            gamma=gamma,
            n_step_return=n_step_return,
        )
        rewards.append(float(reward))
        discounts.append(float(discount))
        done_flags.append(done)
        if done or next_key is None or next_key not in split.sample_index_by_key or next_key not in split.candidate_groups or next_key not in split.states:
            next_arrays.append(empty)
            next_q_head_scores.append(empty_q_head_scores)
            next_available.append(False)
            continue
        next_item = split.arrays_for_key(next_key)
        next_arrays.append(next_item)
        next_q_head_scores.append(split.candidate_metric_vector(next_key, "q_head_score"))
        next_available.append(bool(next_item["candidate_mask"].any()))

    current_q_head_tensor = torch.as_tensor(np.stack(q_head_scores, axis=0), dtype=torch.float32, device=device)
    batch = {
        "current": _stack_state_arrays(current_arrays, device, torch),
        "next": _stack_state_arrays(next_arrays, device, torch),
        "selected_index": torch.as_tensor(selected_indices, dtype=torch.long, device=device),
        "best_index": torch.as_tensor(best_indices, dtype=torch.long, device=device),
        "q_head_scores": current_q_head_tensor,
        "current_q_head_scores": current_q_head_tensor,
        "next_q_head_scores": torch.as_tensor(np.stack(next_q_head_scores, axis=0), dtype=torch.float32, device=device),
        "reward": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "discount": torch.as_tensor(discounts, dtype=torch.float32, device=device),
        "done": torch.as_tensor(done_flags, dtype=torch.bool, device=device),
        "next_available": torch.as_tensor(next_available, dtype=torch.bool, device=device),
    }
    return batch


def _model_forward(model: Any, tensors: dict[str, Any], edge_index: Any) -> Any:
    return model(
        node_features=tensors["node_features"],
        link_features=tensors["link_features"],
        global_features=tensors["global_features"],
        edge_index=edge_index,
        request_features=tensors["request_features"],
        spectrum_tensors=tensors["spectrum_tensors"],
        action_features=tensors["action_features"],
        route_link_mask=tensors["route_link_mask"],
        route_basic_features=tensors["route_basic_features"],
        block_bounds=tensors["block_bounds"],
    )


def _score_values(
    raw_values: Any,
    q_head_scores: Any,
    *,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    torch: Any,
) -> Any:
    if score_mode == "raw":
        return raw_values
    if residual_delta_clip > 0.0:
        raw_values = raw_values.clamp(min=-float(residual_delta_clip), max=float(residual_delta_clip))
    baseline = torch.where(torch.isfinite(q_head_scores), q_head_scores, torch.zeros_like(q_head_scores))
    return baseline + float(residual_scale) * raw_values


def _masked_delta_l2(raw_values: Any, candidate_mask: Any, torch: Any) -> Any | None:
    if not bool(candidate_mask.any()):
        return None
    return raw_values.masked_select(candidate_mask).square().mean()


def _load_pretrained(model: Any, config: ExperimentConfig, device: str, torch: Any) -> dict[str, str | None]:
    cnn_path = _resolve_path(config, "pretrained_cnn_checkpoint")
    gnn_path = _resolve_path(config, "pretrained_gnn_checkpoint")
    dqn_path = _resolve_path(config, "initial_dqn_checkpoint")
    if cnn_path is not None:
        checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
        state = checkpoint.get("encoder_state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.slot_cnn.load_state_dict(state)
    if gnn_path is not None:
        checkpoint = torch.load(gnn_path, map_location=device, weights_only=False)
        state = checkpoint.get("gnn_state_dict", checkpoint.get("model_state_dict", checkpoint))
        model.gnn.load_state_dict(state)
    if dqn_path is not None:
        checkpoint = torch.load(dqn_path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
    return {
        "pretrained_cnn_checkpoint": None if cnn_path is None else str(cnn_path),
        "pretrained_gnn_checkpoint": None if gnn_path is None else str(gnn_path),
        "initial_dqn_checkpoint": None if dqn_path is None else str(dqn_path),
    }


def _build_model(config: ExperimentConfig, device: str, torch: Any):
    from cse2026.ong_solver.models import CandidateQNetwork

    if CandidateQNetwork is None:
        raise RuntimeError("CandidateQNetwork is unavailable because PyTorch is not installed")
    model = CandidateQNetwork(
        action_feature_dim=len(ACTION_FEATURE_COLUMNS),
        hidden_dim=_raw_int(config, "hidden_dim", 128),
    )
    pretrained = _load_pretrained(model, config, device, torch)
    if _raw_bool(config, "freeze_encoders", True):
        for parameter in model.gnn.parameters():
            parameter.requires_grad_(False)
        for parameter in model.slot_cnn.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    return model, pretrained


def _masked_double_dqn_target(
    *,
    online: Any,
    target: Any,
    batch: dict[str, Any],
    edge_index: Any,
    gamma: float,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    torch: Any,
) -> Any:
    with torch.no_grad():
        next_mask = batch["next"]["candidate_mask"] & batch["next_available"][:, None] & (~batch["done"])[:, None]
        next_online_raw = _model_forward(online, batch["next"], edge_index)
        next_online = _score_values(
            next_online_raw,
            batch["next_q_head_scores"],
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        ).masked_fill(~next_mask, -1e9)
        next_action = next_online.argmax(dim=1)
        next_target_raw = _model_forward(target, batch["next"], edge_index)
        next_target = _score_values(
            next_target_raw,
            batch["next_q_head_scores"],
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        )
        next_value = next_target.gather(1, next_action[:, None]).squeeze(1)
        has_next = next_mask.any(dim=1)
        next_value = torch.where(has_next, next_value, torch.zeros_like(next_value))
        discount = batch.get("discount")
        if discount is None:
            discount = torch.full_like(batch["reward"], float(gamma))
        expected = batch["reward"] + discount * next_value
    if not bool(torch.isfinite(expected).all()):
        raise RuntimeError("Non-finite Double DQN targets detected")
    return expected.detach()


def _supervised_ce_loss(q_values: Any, candidate_mask: Any, best_index: Any, torch: Any) -> tuple[Any | None, int, int]:
    usable = (best_index >= 0) & (best_index < candidate_mask.shape[1])
    safe_best = best_index.clamp(0, candidate_mask.shape[1] - 1)
    usable = usable & candidate_mask.gather(1, safe_best[:, None]).squeeze(1)
    if not bool(usable.any()):
        return None, 0, 0
    logits = q_values.masked_fill(~candidate_mask, -1e9)
    loss = torch.nn.functional.cross_entropy(logits[usable], best_index[usable])
    prediction = logits[usable].argmax(dim=1)
    correct = int((prediction == best_index[usable]).sum().detach().cpu())
    total = int(usable.sum().detach().cpu())
    return loss, correct, total


def _best_pairwise_ranking_loss(q_values: Any, candidate_mask: Any, best_index: Any, margin: float, torch: Any) -> Any | None:
    usable = (best_index >= 0) & (best_index < candidate_mask.shape[1])
    safe_best = best_index.clamp(0, candidate_mask.shape[1] - 1)
    usable = usable & candidate_mask.gather(1, safe_best[:, None]).squeeze(1)
    if not bool(usable.any()):
        return None
    q_usable = q_values[usable]
    mask_usable = candidate_mask[usable].clone()
    best_usable = best_index[usable]
    mask_usable.scatter_(1, best_usable[:, None], False)
    if not bool(mask_usable.any()):
        return None
    best_q = q_usable.gather(1, best_usable[:, None])
    violations = torch.nn.functional.relu(float(margin) - (best_q - q_usable))
    return violations.masked_select(mask_usable).mean()


def _score_metric(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _validation_score(metrics: dict[str, Any], config: ExperimentConfig) -> float:
    mode = str(config.resolved.get("checkpoint_score_mode", config.raw.get("checkpoint_score_mode", "td_plus_agreement")))
    td_loss = _score_metric(metrics.get("td_huber_loss"), math.inf)
    best_agreement = _score_metric(metrics.get("greedy_matches_best_candidate_index"), 0.0)
    if mode == "td":
        return td_loss
    if mode == "agreement":
        return 1.0 - best_agreement
    if mode == "action_quality":
        energy = _score_metric(metrics.get("mean_greedy_energy_increment"), 1200.0)
        fragmentation = _score_metric(metrics.get("mean_greedy_fragmentation_after"), 1.0)
        delay = _score_metric(metrics.get("mean_greedy_delay_ms"), 50.0)
        qot_margin_norm = _score_metric(metrics.get("mean_greedy_qot_margin_norm"), 0.0)
        energy_norm = max(_raw_float(config, "validation_energy_norm_w", 1200.0), 1e-9)
        delay_norm = max(_raw_float(config, "validation_delay_norm_ms", 50.0), 1e-9)
        return float(
            _raw_float(config, "validation_td_weight", 0.0) * td_loss
            + _raw_float(config, "validation_best_miss_weight", 1.0) * (1.0 - best_agreement)
            + _raw_float(config, "validation_energy_weight", 0.35) * (energy / energy_norm)
            + _raw_float(config, "validation_fragmentation_weight", 1.25) * fragmentation
            + _raw_float(config, "validation_delay_weight", 0.05) * (delay / delay_norm)
            - _raw_float(config, "validation_qot_margin_weight", 0.35) * qot_margin_norm
        )
    return td_loss + (1.0 - best_agreement)


def _evaluate(
    *,
    online: Any,
    target: Any,
    split: DqnOfflineSplit,
    batch_size: int,
    max_batches: int,
    device: str,
    gamma: float,
    n_step_return: int,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    target_mode: str,
    target_params: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    online.eval()
    target.eval()
    rng = np.random.default_rng(0)
    indices = split.valid_indices
    batches = _iter_batches(len(indices), batch_size, shuffle=False, rng=rng)
    if max_batches > 0:
        batches = batches[:max_batches]

    total = 0
    best_agree = 0
    selected_agree = 0
    td_losses: list[float] = []
    td_errors: list[float] = []
    q_pred_values: list[float] = []
    target_values: list[float] = []
    raw_model_values: list[float] = []
    abs_raw_model_values: list[float] = []
    greedy_energy: list[float] = []
    greedy_fragmentation: list[float] = []
    greedy_qot_margin: list[float] = []
    greedy_qot_margin_norm: list[float] = []
    greedy_delay: list[float] = []
    mask_violations = 0
    empty_current = 0
    edge_index = torch.as_tensor(split.edge_index, dtype=torch.long, device=device)
    huber = torch.nn.SmoothL1Loss(reduction="none")

    with torch.no_grad():
        for positions in batches:
            row_indices = indices[positions]
            batch = _batch_tensors(
                split,
                row_indices,
                device,
                torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
            )
            raw_values = _model_forward(online, batch["current"], edge_index)
            q_values = _score_values(
                raw_values,
                batch["current_q_head_scores"],
                score_mode=score_mode,
                residual_scale=residual_scale,
                residual_delta_clip=residual_delta_clip,
                torch=torch,
            )
            current_mask = batch["current"]["candidate_mask"]
            raw_valid = raw_values.masked_select(current_mask)
            raw_model_values.extend(float(value) for value in raw_valid.detach().cpu().numpy())
            abs_raw_model_values.extend(float(value) for value in torch.abs(raw_valid).detach().cpu().numpy())
            selected = batch["selected_index"]
            q_pred = q_values.gather(1, selected[:, None]).squeeze(1)
            expected = _masked_double_dqn_target(
                online=online,
                target=target,
                batch=batch,
                edge_index=edge_index,
                gamma=gamma,
                score_mode=score_mode,
                residual_scale=residual_scale,
                residual_delta_clip=residual_delta_clip,
                torch=torch,
            )
            losses = huber(q_pred, expected)
            td_losses.extend(float(value) for value in losses.detach().cpu().numpy())
            td_errors.extend(float(value) for value in torch.abs(q_pred - expected).detach().cpu().numpy())
            q_pred_values.extend(float(value) for value in q_pred.detach().cpu().numpy())
            target_values.extend(float(value) for value in expected.detach().cpu().numpy())

            masked_q = q_values.masked_fill(~current_mask, -1e9)
            greedy = masked_q.argmax(dim=1)
            valid_counts = current_mask.sum(dim=1)
            empty_current += int((valid_counts == 0).sum().detach().cpu())
            invalid = ~current_mask.gather(1, greedy[:, None]).squeeze(1)
            mask_violations += int(invalid.sum().detach().cpu())
            best = batch["best_index"]
            best_valid = (best >= 0) & (best < current_mask.shape[1])
            selected_valid = selected >= 0
            best_agree += int(((greedy == best) & best_valid).sum().detach().cpu())
            selected_agree += int(((greedy == selected) & selected_valid).sum().detach().cpu())
            total += int(len(row_indices))

            greedy_np = greedy.detach().cpu().numpy()
            for row_position, action_index in zip(row_indices, greedy_np):
                row = split.dqn.iloc[int(row_position)]
                key = (str(row.episode_id), int(row.request_id))
                candidate_row = split.candidate_row(key, int(action_index))
                if candidate_row is None:
                    continue
                greedy_energy.append(_finite_float(candidate_row.get("energy_increment"), default=math.nan))
                greedy_fragmentation.append(_finite_float(candidate_row.get("fragmentation_after"), default=math.nan))
                greedy_qot_margin.append(_finite_float(candidate_row.get("qot_margin"), default=math.nan))
                greedy_qot_margin_norm.append(_finite_float(candidate_row.get("qot_margin_norm"), default=math.nan))
                greedy_delay.append(_finite_float(candidate_row.get("delay_ms"), default=math.nan))

    def mean(values: list[float]) -> float | None:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return None
        return float(np.mean(np.asarray(finite, dtype=np.float64)))

    return {
        "samples": int(total),
        "skipped_forced_block_transitions": int(split.skipped_forced_block),
        "skipped_invalid_action_transitions": int(split.skipped_invalid_action),
        "td_huber_loss": mean(td_losses),
        "td_abs_error": mean(td_errors),
        "mean_recorded_action_q": mean(q_pred_values),
        "mean_td_target": mean(target_values),
        "mean_model_output": mean(raw_model_values),
        "mean_abs_model_output": mean(abs_raw_model_values),
        "greedy_matches_best_candidate_index": None if total == 0 else float(best_agree / total),
        "greedy_matches_recorded_selected_candidate_index": None if total == 0 else float(selected_agree / total),
        "mask_violations": int(mask_violations),
        "empty_current_states": int(empty_current),
        "mean_greedy_energy_increment": mean(greedy_energy),
        "mean_greedy_fragmentation_after": mean(greedy_fragmentation),
        "mean_greedy_qot_margin": mean(greedy_qot_margin),
        "mean_greedy_qot_margin_norm": mean(greedy_qot_margin_norm),
        "mean_greedy_delay_ms": mean(greedy_delay),
        "generated_state_cache_size": int(len(split.cache)),
        "cnn_reference_tensor_match_rate": (
            float(split.cnn_compare["matched"] / split.cnn_compare["compared"])
            if split.cnn_compare["compared"]
            else None
        ),
        "cnn_reference_tensor_compared": int(split.cnn_compare["compared"]),
    }


def _train_supervised_warmup(
    *,
    online: Any,
    train: DqnOfflineSplit,
    optimizer: Any,
    batch_size: int,
    max_batches: int,
    device: str,
    torch: Any,
    rng: np.random.Generator,
    score_mode: str,
    residual_scale: float,
    residual_delta_clip: float,
    residual_l2_weight: float,
    target_mode: str,
    target_params: dict[str, Any],
    progress_every_batches: int = 0,
    epoch: int = 0,
) -> dict[str, Any]:
    online.train()
    edge_index = torch.as_tensor(train.edge_index, dtype=torch.long, device=device)
    batches = _iter_batches(len(train.valid_indices), batch_size, shuffle=True, rng=rng)
    if max_batches > 0:
        batches = batches[:max_batches]
    losses: list[float] = []
    residual_l2_losses: list[float] = []
    accuracy_count = 0
    accuracy_total = 0
    for batch_index, positions in enumerate(batches, start=1):
        row_indices = train.valid_indices[positions]
        batch = _batch_tensors(train, row_indices, device, torch, target_mode=target_mode, target_params=target_params)
        raw_values = _model_forward(online, batch["current"], edge_index)
        q_values = _score_values(
            raw_values,
            batch["current_q_head_scores"],
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            torch=torch,
        )
        current_mask = batch["current"]["candidate_mask"]
        best = batch["best_index"]
        usable = (best >= 0) & (best < current_mask.shape[1])
        safe_best = best.clamp(0, current_mask.shape[1] - 1)
        usable = usable & current_mask.gather(1, safe_best[:, None]).squeeze(1)
        if not bool(usable.any()):
            continue
        logits = q_values.masked_fill(~current_mask, -1e9)
        loss = torch.nn.functional.cross_entropy(logits[usable], best[usable])
        residual_l2 = _masked_delta_l2(raw_values, current_mask, torch)
        if residual_l2 is not None:
            residual_l2_losses.append(float(residual_l2.detach().cpu()))
            if residual_l2_weight > 0.0:
                loss = loss + float(residual_l2_weight) * residual_l2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in online.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        prediction = logits[usable].argmax(dim=1)
        accuracy_count += int((prediction == best[usable]).sum().detach().cpu())
        accuracy_total += int(usable.sum().detach().cpu())
        if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
            print(
                json.dumps(
                    {
                        "event": "warmup_progress",
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches": len(batches),
                        "loss_so_far": float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else None,
                        "accuracy_so_far": None if accuracy_total == 0 else float(accuracy_count / accuracy_total),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "loss": None if not losses else float(np.mean(np.asarray(losses, dtype=np.float64))),
        "residual_l2_loss": None if not residual_l2_losses else float(np.mean(np.asarray(residual_l2_losses, dtype=np.float64))),
        "accuracy": None if accuracy_total == 0 else float(accuracy_count / accuracy_total),
        "batches": int(len(losses)),
    }


def run_train_dqn(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_dqn requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    device = _device(config, torch)
    cfg = SolverConfig(
        n_max=_raw_int(config, "n_max", 32),
        rng_seed=int(config.seed),
        device=device,
        hidden_dim=_raw_int(config, "hidden_dim", 128),
    )
    train_split, val_split, test_split = _splits(config)
    transition_limits = {
        "train": _transition_limit(config, train_split),
        "val": _transition_limit(config, val_split),
        "test": _transition_limit(config, test_split),
    }
    hard_case_fraction = _raw_float(config, "hard_case_fraction", 0.0)
    hard_case_min_candidates = _raw_int(config, "hard_case_min_candidates", 4)
    hard_case_small_margin = _raw_float(config, "hard_case_small_margin", 0.03)
    print(
        f"Loading DQN offline splits: train={train_split} val={val_split} test={test_split} "
        f"transition_limits={transition_limits}",
        flush=True,
    )
    train = _load_split(
        config.dataset_path,
        train_split,
        cfg,
        transition_limit=transition_limits["train"],
        seed=config.seed,
        sampling_mode=_sampling_mode_for_split(config, train_split),
        hard_case_fraction=hard_case_fraction,
        hard_case_min_candidates=hard_case_min_candidates,
        hard_case_small_margin=hard_case_small_margin,
    )
    val = _load_split(
        config.dataset_path,
        val_split,
        cfg,
        transition_limit=transition_limits["val"],
        seed=config.seed + 1,
        sampling_mode=_sampling_mode_for_split(config, val_split),
        hard_case_fraction=hard_case_fraction,
        hard_case_min_candidates=hard_case_min_candidates,
        hard_case_small_margin=hard_case_small_margin,
    )
    test = _load_split(
        config.dataset_path,
        test_split,
        cfg,
        transition_limit=transition_limits["test"],
        seed=config.seed + 2,
        sampling_mode=_sampling_mode_for_split(config, test_split),
        hard_case_fraction=hard_case_fraction,
        hard_case_min_candidates=hard_case_min_candidates,
        hard_case_small_margin=hard_case_small_margin,
    )
    reward_override_stats = {
        "train": _apply_reward_override(train, config),
        "val": _apply_reward_override(val, config),
        "test": _apply_reward_override(test, config),
    }

    online, pretrained = _build_model(config, device, torch)
    target = copy.deepcopy(online).to(device)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = [parameter for parameter in online.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=_raw_float(config, "learning_rate", 3e-4),
        weight_decay=_raw_float(config, "weight_decay", 1e-4),
    )
    batch_size = int(config.batch_size)
    max_batches = int(config.max_batches)
    progress_every_batches = _raw_int(config, "progress_every_batches", 0)
    epochs = _raw_int(config, "epochs", 4)
    warmup_epochs = _raw_int(config, "supervised_warmup_epochs", 1)
    patience = _raw_int(config, "patience", 3)
    gamma = _raw_float(config, "gamma", 0.95)
    n_step_return = max(1, _raw_int(config, "n_step_return", 1))
    score_mode = normalize_q_score_mode(_raw_str(config, "q_score_mode", "raw"))
    residual_scale = _raw_float(config, "residual_scale", 1.0)
    residual_delta_clip = _raw_float(config, "residual_delta_clip", 0.0)
    residual_l2_weight = _raw_float(config, "residual_l2_weight", 0.0)
    target_mode = _raw_str(config, "learning_target", _raw_str(config, "imitation_target", "best"))
    target_params = _target_params(config)
    target_update_interval = _raw_int(config, "target_update_interval", 200)
    td_loss_weight = _raw_float(config, "td_loss_weight", 1.0)
    imitation_loss_weight = _raw_float(config, "imitation_loss_weight", 0.0)
    ranking_loss_weight = _raw_float(config, "ranking_loss_weight", 0.0)
    ranking_margin = _raw_float(config, "ranking_margin", 0.10)
    save_epoch_checkpoints = _raw_bool(config, "save_epoch_checkpoints", False)
    rng = np.random.default_rng(config.seed)
    huber = torch.nn.SmoothL1Loss()
    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = -1
    stale = 0
    global_step = 0
    best_path = run_path / "dqn_best.pt"
    warmup_path = run_path / "dqn_warmup.pt"
    edge_index = torch.as_tensor(train.edge_index, dtype=torch.long, device=device)

    for warmup_epoch in range(1, warmup_epochs + 1):
        warmup_metrics = _train_supervised_warmup(
            online=online,
            train=train,
            optimizer=optimizer,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            torch=torch,
            rng=rng,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            residual_l2_weight=residual_l2_weight,
            target_mode=target_mode,
            target_params=target_params,
            progress_every_batches=progress_every_batches,
            epoch=warmup_epoch,
        )
        target.load_state_dict(online.state_dict())
        row = {"phase": "supervised_warmup", "epoch": warmup_epoch, **warmup_metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    if warmup_epochs > 0:
        torch.save(
            {
                "model_state_dict": online.state_dict(),
                "target_model_state_dict": target.state_dict(),
                "epoch": int(warmup_epochs),
                "phase": "supervised_warmup",
                "config": config.resolved,
                "solver_config": {
                    "n_max": int(cfg.n_max),
                    "hidden_dim": int(cfg.hidden_dim),
                    "device": device,
                    "q_score_mode": score_mode,
                    "residual_scale": float(residual_scale),
                    "residual_delta_clip": float(residual_delta_clip),
                },
                "action_feature_columns": list(ACTION_FEATURE_COLUMNS),
                **pretrained,
            },
            warmup_path,
        )

    for epoch in range(1, epochs + 1):
        online.train()
        batches = _iter_batches(len(train.valid_indices), batch_size, shuffle=True, rng=rng)
        if max_batches > 0:
            batches = batches[:max_batches]
        losses: list[float] = []
        td_losses: list[float] = []
        imitation_losses: list[float] = []
        ranking_losses: list[float] = []
        residual_l2_losses: list[float] = []
        td_errors: list[float] = []
        q_pred_values: list[float] = []
        target_values: list[float] = []
        imitation_correct = 0
        imitation_total = 0
        selected_mask_violations = 0
        for batch_index, positions in enumerate(batches, start=1):
            row_indices = train.valid_indices[positions]
            batch = _batch_tensors(
                train,
                row_indices,
                device,
                torch,
                gamma=gamma,
                n_step_return=n_step_return,
                target_mode=target_mode,
                target_params=target_params,
            )
            raw_values = _model_forward(online, batch["current"], edge_index)
            q_values = _score_values(
                raw_values,
                batch["current_q_head_scores"],
                score_mode=score_mode,
                residual_scale=residual_scale,
                residual_delta_clip=residual_delta_clip,
                torch=torch,
            )
            selected = batch["selected_index"]
            current_mask = batch["current"]["candidate_mask"]
            selected_valid = current_mask.gather(1, selected[:, None]).squeeze(1)
            selected_mask_violations += int((~selected_valid).sum().detach().cpu())
            if not bool(selected_valid.all()):
                q_values = q_values[selected_valid]
                raw_values = raw_values[selected_valid]
                selected = selected[selected_valid]
                reduced_batch = {
                    "current": {key: value[selected_valid] for key, value in batch["current"].items()},
                    "next": {key: value[selected_valid] for key, value in batch["next"].items()},
                    "selected_index": selected,
                    "best_index": batch["best_index"][selected_valid],
                    "q_head_scores": batch["q_head_scores"][selected_valid],
                    "current_q_head_scores": batch["current_q_head_scores"][selected_valid],
                    "next_q_head_scores": batch["next_q_head_scores"][selected_valid],
                    "reward": batch["reward"][selected_valid],
                    "discount": batch["discount"][selected_valid],
                    "done": batch["done"][selected_valid],
                    "next_available": batch["next_available"][selected_valid],
                }
                batch = reduced_batch
                if selected.numel() == 0:
                    continue
            current_mask = batch["current"]["candidate_mask"]
            best = batch["best_index"]
            q_pred = q_values.gather(1, selected[:, None]).squeeze(1)
            expected = _masked_double_dqn_target(
                online=online,
                target=target,
                batch=batch,
                edge_index=edge_index,
                gamma=gamma,
                score_mode=score_mode,
                residual_scale=residual_scale,
                residual_delta_clip=residual_delta_clip,
                torch=torch,
            )
            td_loss = huber(q_pred, expected)
            loss = float(td_loss_weight) * td_loss
            residual_l2 = _masked_delta_l2(raw_values, current_mask, torch)
            if residual_l2 is not None:
                residual_l2_losses.append(float(residual_l2.detach().cpu()))
                if residual_l2_weight > 0.0:
                    loss = loss + float(residual_l2_weight) * residual_l2
            ce_loss, ce_correct, ce_total = _supervised_ce_loss(q_values, current_mask, best, torch)
            if ce_loss is not None and imitation_loss_weight > 0.0:
                loss = loss + float(imitation_loss_weight) * ce_loss
            if ce_loss is not None:
                imitation_losses.append(float(ce_loss.detach().cpu()))
                imitation_correct += ce_correct
                imitation_total += ce_total
            ranking_loss = _best_pairwise_ranking_loss(q_values, current_mask, best, ranking_margin, torch)
            if ranking_loss is not None and ranking_loss_weight > 0.0:
                loss = loss + float(ranking_loss_weight) * ranking_loss
            if ranking_loss is not None:
                ranking_losses.append(float(ranking_loss.detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()

            global_step += 1
            if target_update_interval > 0 and global_step % target_update_interval == 0:
                target.load_state_dict(online.state_dict())

            losses.append(float(loss.detach().cpu()))
            td_losses.append(float(td_loss.detach().cpu()))
            td_errors.extend(float(value) for value in torch.abs(q_pred - expected).detach().cpu().numpy())
            q_pred_values.extend(float(value) for value in q_pred.detach().cpu().numpy())
            target_values.extend(float(value) for value in expected.detach().cpu().numpy())
            if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
                print(
                    json.dumps(
                        {
                            "event": "dqn_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(batches),
                            "train_loss_so_far": float(np.mean(np.asarray(losses, dtype=np.float64))) if losses else None,
                            "td_loss_so_far": float(np.mean(np.asarray(td_losses, dtype=np.float64))) if td_losses else None,
                            "imitation_accuracy_so_far": None
                            if imitation_total == 0
                            else float(imitation_correct / imitation_total),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        target.load_state_dict(online.state_dict())
        val_metrics = _evaluate(
            online=online,
            target=target,
            split=val,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            gamma=gamma,
            n_step_return=n_step_return,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            target_mode=target_mode,
            target_params=target_params,
            torch=torch,
        )
        val_score = _validation_score(val_metrics, config)
        epoch_row = {
            "phase": "dqn",
            "epoch": epoch,
            "train_loss": None if not losses else float(np.mean(np.asarray(losses, dtype=np.float64))),
            "train_td_loss": None if not td_losses else float(np.mean(np.asarray(td_losses, dtype=np.float64))),
            "train_imitation_loss": None if not imitation_losses else float(np.mean(np.asarray(imitation_losses, dtype=np.float64))),
            "train_ranking_loss": None if not ranking_losses else float(np.mean(np.asarray(ranking_losses, dtype=np.float64))),
            "train_residual_l2_loss": None if not residual_l2_losses else float(np.mean(np.asarray(residual_l2_losses, dtype=np.float64))),
            "train_imitation_accuracy": None if imitation_total == 0 else float(imitation_correct / imitation_total),
            "train_td_abs_error": None if not td_errors else float(np.mean(np.asarray(td_errors, dtype=np.float64))),
            "train_mean_recorded_action_q": None if not q_pred_values else float(np.mean(np.asarray(q_pred_values, dtype=np.float64))),
            "train_mean_td_target": None if not target_values else float(np.mean(np.asarray(target_values, dtype=np.float64))),
            "selected_mask_violations": int(selected_mask_violations),
            "val_score": val_score,
            "val": val_metrics,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, sort_keys=True), flush=True)
        if save_epoch_checkpoints:
            torch.save(
                {
                    "model_state_dict": online.state_dict(),
                    "target_model_state_dict": target.state_dict(),
                    "epoch": epoch,
                    "val_score": val_score,
                    "config": config.resolved,
                    "solver_config": {
                        "n_max": int(cfg.n_max),
                        "hidden_dim": int(cfg.hidden_dim),
                        "device": device,
                        "q_score_mode": score_mode,
                        "residual_scale": float(residual_scale),
                        "residual_delta_clip": float(residual_delta_clip),
                    },
                    "action_feature_columns": list(ACTION_FEATURE_COLUMNS),
                    **pretrained,
                },
                run_path / f"dqn_epoch_{epoch}.pt",
            )
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": online.state_dict(),
                    "target_model_state_dict": target.state_dict(),
                    "epoch": epoch,
                    "val_score": best_val,
                    "config": config.resolved,
                    "solver_config": {
                        "n_max": int(cfg.n_max),
                        "hidden_dim": int(cfg.hidden_dim),
                        "device": device,
                        "q_score_mode": score_mode,
                        "residual_scale": float(residual_scale),
                        "residual_delta_clip": float(residual_delta_clip),
                    },
                    "action_feature_columns": list(ACTION_FEATURE_COLUMNS),
                    **pretrained,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    if best_epoch < 0:
        torch.save(
            {
                "model_state_dict": online.state_dict(),
                "target_model_state_dict": target.state_dict(),
                "epoch": 0,
                "val_score": None,
                "config": config.resolved,
                "solver_config": {
                    "n_max": int(cfg.n_max),
                    "hidden_dim": int(cfg.hidden_dim),
                    "device": device,
                    "q_score_mode": score_mode,
                    "residual_scale": float(residual_scale),
                    "residual_delta_clip": float(residual_delta_clip),
                },
                "action_feature_columns": list(ACTION_FEATURE_COLUMNS),
                **pretrained,
            },
            best_path,
        )
    else:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        online.load_state_dict(checkpoint["model_state_dict"])
        target.load_state_dict(checkpoint["target_model_state_dict"])

    final_metrics = {
        "stage": "train_dqn",
        "dataset_path": str(config.dataset_path),
        "device": device,
        "best_epoch": int(best_epoch),
        "best_checkpoint": str(best_path),
        "warmup_checkpoint": str(warmup_path) if warmup_epochs > 0 else None,
        "train_valid_transitions": int(len(train.valid_indices)),
        "val_valid_transitions": int(len(val.valid_indices)),
        "test_valid_transitions": int(len(test.valid_indices)),
        "transition_limits": transition_limits,
        "transition_sampling": {
            "train": train.sampling_stats,
            "val": val.sampling_stats,
            "test": test.sampling_stats,
        },
        "skipped_transitions": {
            "train_forced_block": int(train.skipped_forced_block),
            "val_forced_block": int(val.skipped_forced_block),
            "test_forced_block": int(test.skipped_forced_block),
            "train_invalid_action": int(train.skipped_invalid_action),
            "val_invalid_action": int(val.skipped_invalid_action),
            "test_invalid_action": int(test.skipped_invalid_action),
        },
        "pretrained": pretrained,
        "freeze_encoders": _raw_bool(config, "freeze_encoders", True),
        "save_epoch_checkpoints": bool(save_epoch_checkpoints),
        "reward_override": reward_override_stats,
        "gamma": float(gamma),
        "n_step_return": int(n_step_return),
        "q_score_mode": score_mode,
        "residual_scale": float(residual_scale),
        "residual_delta_clip": float(residual_delta_clip),
        "residual_l2_weight": float(residual_l2_weight),
        "learning_target": str(target_mode),
        "target_params": _target_params_for_metrics(target_params),
        "td_loss_weight": float(td_loss_weight),
        "imitation_loss_weight": float(imitation_loss_weight),
        "ranking_loss_weight": float(ranking_loss_weight),
        "ranking_margin": float(ranking_margin),
        "checkpoint_score_mode": str(config.resolved.get("checkpoint_score_mode", config.raw.get("checkpoint_score_mode", "td_plus_agreement"))),
        "history": history,
        "train": _evaluate(
            online=online,
            target=target,
            split=train,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            gamma=gamma,
            n_step_return=n_step_return,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            target_mode=target_mode,
            target_params=target_params,
            torch=torch,
        ),
        "val": _evaluate(
            online=online,
            target=target,
            split=val,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            gamma=gamma,
            n_step_return=n_step_return,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            target_mode=target_mode,
            target_params=target_params,
            torch=torch,
        ),
        "test": _evaluate(
            online=online,
            target=target,
            split=test,
            batch_size=batch_size,
            max_batches=max_batches,
            device=device,
            gamma=gamma,
            n_step_return=n_step_return,
            score_mode=score_mode,
            residual_scale=residual_scale,
            residual_delta_clip=residual_delta_clip,
            target_mode=target_mode,
            target_params=target_params,
            torch=torch,
        ),
    }
    _write_json(run_path / "metrics.json", final_metrics)
    return final_metrics
