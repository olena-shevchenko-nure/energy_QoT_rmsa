from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root, read_json
from cse2026.ong_solver import Candidate, CandidateBatch, GnnCnnDqnOngSolver, SolverConfig, StateView
from cse2026.ong_solver.common import masked_argmax, pad_q_scores, route_slot_tensor

from ..config import ExperimentConfig


ACTION_FEATURE_COLUMNS = (
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


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _splits(config: ExperimentConfig) -> list[str]:
    if config.splits:
        return list(config.splits)
    if config.dataset_path is None:
        raise ValueError("dataset_path is required")
    manifest = read_json(config.dataset_path / "manifest.json")
    return list(manifest["splits"].keys())


def _json_list(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ()
        if isinstance(parsed, list):
            return tuple(parsed)
    return ()


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _candidate_action_features(row: pd.Series, cfg: SolverConfig, slots: int) -> tuple[float, ...]:
    b_start = max(int(row.get("b_start", 0)), 0)
    width = max(int(row.get("w", 0)), 0)
    lmax = max(_finite_float(row.get("largest_free_block_after")), 0.0)
    return (
        _finite_float(row.get("route_length_km")) / max(cfg.max_route_length_norm_km, 1e-9),
        _finite_float(row.get("hop_count")) / 8.0,
        float(b_start) / float(max(slots - 1, 1)),
        float(width) / float(max(slots, 1)),
        _finite_float(row.get("qot_margin_norm")),
        _finite_float(row.get("delay_norm")),
        _finite_float(row.get("energy_increment_norm")),
        _finite_float(row.get("fragmentation_after")),
        float(lmax) / float(max(slots, 1)),
        _finite_float(row.get("small_gap_penalty")),
    )


def _candidate_from_row(row: pd.Series, cfg: SolverConfig, slots: int) -> Candidate:
    route_links = tuple(int(link_id) for link_id in _json_list(row.get("route_directed_link_ids")))
    route_nodes = tuple(_json_list(row.get("route_node_ids")))
    width = max(int(row.get("w", 0)), 0)
    b_start = max(int(row.get("b_start", 0)), 0)
    q_head_score = _finite_float(row.get("q_head_score"), default=-math.inf)
    return Candidate(
        action=int(row.get("topn_index", -1)),
        route_id=int(row.get("route_id", -1)),
        modulation_index=int(row.get("modulation_id", -1)),
        modulation_offset=int(row.get("modulation_id", -1)),
        b_start=b_start,
        w=width,
        route_node_ids=route_nodes,
        route_link_ids=route_links,
        route_length_km=_finite_float(row.get("route_length_km")),
        hop_count=int(max(_finite_float(row.get("hop_count")), 0.0)),
        delay_ms=_finite_float(row.get("delay_ms")),
        modulation_name=str(row.get("modulation_id", "")),
        spectral_efficiency=0.0,
        qot_margin_norm=_finite_float(row.get("qot_margin_norm")),
        qot_risk=_finite_float(row.get("qot_risk")),
        energy_increment=_finite_float(row.get("energy_increment")),
        energy_increment_norm=_finite_float(row.get("energy_increment_norm")),
        fragmentation_before=_finite_float(row.get("fragmentation_before")),
        fragmentation_after=_finite_float(row.get("fragmentation_after")),
        delta_fragmentation=_finite_float(row.get("delta_fragmentation")),
        largest_free_block_after=int(max(_finite_float(row.get("largest_free_block_after")), 0.0)),
        left_gap_after=int(max(_finite_float(row.get("left_gap_after")), 0.0)),
        right_gap_after=int(max(_finite_float(row.get("right_gap_after")), 0.0)),
        small_gap_penalty=_finite_float(row.get("small_gap_penalty")),
        compactness=_finite_float(row.get("compactness")),
        j_frag=_finite_float(row.get("j_frag")),
        j_tie=_finite_float(row.get("j_tie")),
        j_total=_finite_float(row.get("j_total")),
        q_head_score=q_head_score,
        action_features=_candidate_action_features(row, cfg, slots),
        topn_index=int(row.get("topn_index", -1)),
    )


def _cnn_lookup(cnn_index: pd.DataFrame) -> dict[tuple[Any, ...], int]:
    lookup: dict[tuple[Any, ...], int] = {}
    for row in cnn_index.itertuples(index=False):
        key = (
            row.episode_id,
            int(row.request_id),
            int(row.route_id),
            int(row.modulation_id),
            int(row.b_start),
            int(row.w),
        )
        lookup[key] = int(row.sample_id)
    return lookup


def _parse_selected_action(row: Any | None) -> dict[str, Any] | None:
    if row is None or int(row.selected_candidate_index) < 0:
        return None
    try:
        selected = json.loads(row.selected_action_description)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(selected, dict) or int(selected.get("w", 0)) <= 0:
        return None
    return selected


def _release_expired(
    *,
    active_lightpaths: list[dict[str, Any]],
    occupancy: np.ndarray,
    release_times: np.ndarray,
    active_link_counts: np.ndarray,
    active_node_counts: np.ndarray,
    now: float,
) -> list[dict[str, Any]]:
    remaining: list[dict[str, Any]] = []
    for lightpath in active_lightpaths:
        if float(lightpath["departure_time"]) > now:
            remaining.append(lightpath)
            continue
        start = int(lightpath["b_start"])
        width = int(lightpath["w"])
        for link_id in lightpath["route_directed_link_ids"]:
            occupancy[int(link_id), start : start + width] = 0
            release_times[int(link_id), start : start + width] = 0.0
            active_link_counts[int(link_id)] = max(active_link_counts[int(link_id)] - 1.0, 0.0)
        for node_id in lightpath["route_node_ids"]:
            node_index = int(node_id) - 1
            if 0 <= node_index < active_node_counts.shape[0]:
                active_node_counts[node_index] = max(active_node_counts[node_index] - 1.0, 0.0)
    return remaining


def _apply_selected_action(
    *,
    selected: dict[str, Any],
    request: Any,
    active_lightpaths: list[dict[str, Any]],
    occupancy: np.ndarray,
    release_times: np.ndarray,
    active_link_counts: np.ndarray,
    active_node_counts: np.ndarray,
) -> None:
    start = int(selected["b_start"])
    width = int(selected["w"])
    links = [int(link_id) for link_id in selected.get("route_directed_link_ids", [])]
    nodes = [int(node_id) for node_id in selected.get("route_node_ids", [])]
    departure = float(request.departure_time)
    for link_id in links:
        if 0 <= link_id < occupancy.shape[0]:
            occupancy[link_id, start : start + width] = 1
            release_times[link_id, start : start + width] = departure
            active_link_counts[link_id] += 1.0
    for node_id in nodes:
        node_index = int(node_id) - 1
        if 0 <= node_index < active_node_counts.shape[0]:
            active_node_counts[node_index] += 1.0
    active_lightpaths.append(
        {
            "departure_time": departure,
            "route_directed_link_ids": links,
            "route_node_ids": nodes,
            "b_start": start,
            "w": width,
        }
    )


def _offline_state_views(
    *,
    traffic: pd.DataFrame,
    dqn_by_request: dict[tuple[str, int], Any],
    target_keys: set[tuple[str, int]],
    graphs: np.lib.npyio.NpzFile,
    slots: int,
    edge_lengths_km: np.ndarray,
) -> dict[tuple[str, int], StateView]:
    edge_index = np.asarray(graphs["edge_index"], dtype=np.int64)
    link_count = int(graphs["link_features"].shape[1])
    node_count = int(graphs["node_features"].shape[1])
    states: dict[tuple[str, int], StateView] = {}
    for episode_id, episode in traffic.groupby("episode_id", sort=False):
        occupancy = np.zeros((link_count, slots), dtype=np.uint8)
        release_times = np.zeros((link_count, slots), dtype=np.float32)
        active_link_counts = np.zeros(link_count, dtype=np.float32)
        active_node_counts = np.zeros(node_count, dtype=np.float32)
        active_lightpaths: list[dict[str, Any]] = []
        ordered = episode.sort_values("request_id")
        for request in ordered.itertuples(index=False):
            key = (str(episode_id), int(request.request_id))
            now = float(request.arrival_time)
            active_lightpaths = _release_expired(
                active_lightpaths=active_lightpaths,
                occupancy=occupancy,
                release_times=release_times,
                active_link_counts=active_link_counts,
                active_node_counts=active_node_counts,
                now=now,
            )
            if key in target_keys:
                states[key] = StateView(
                    node_names=tuple(range(node_count)),
                    edge_index=edge_index,
                    edge_lengths_km=edge_lengths_km.astype(np.float32),
                    occupancy=occupancy.copy(),
                    release_times=release_times.copy(),
                    active_link_counts=active_link_counts.copy(),
                    active_node_counts=active_node_counts.copy(),
                    src=int(request.src),
                    dst=int(request.dst),
                    bit_rate_gbps=float(request.bit_rate_gbps),
                    holding_time=float(request.holding_time),
                    current_time=now,
                    topology_name="offline-eon-dataset",
                )
            selected = _parse_selected_action(dqn_by_request.get(key))
            if selected is not None:
                _apply_selected_action(
                    selected=selected,
                    request=request,
                    active_lightpaths=active_lightpaths,
                    occupancy=occupancy,
                    release_times=release_times,
                    active_link_counts=active_link_counts,
                    active_node_counts=active_node_counts,
                )
        if len(states) >= len(target_keys):
            break
    return states


def _edge_lengths_from_dataset(dataset_path: Path, candidates: pd.DataFrame, link_count: int) -> np.ndarray:
    directed_links = dataset_path / "topology" / "directed_links.csv"
    if directed_links.exists():
        links = pd.read_csv(directed_links)
        lengths = np.ones(link_count, dtype=np.float32)
        for row in links.itertuples(index=False):
            link_id = int(row.directed_link_id)
            if 0 <= link_id < link_count:
                lengths[link_id] = float(row.length_km)
        return lengths

    sums = np.zeros(link_count, dtype=np.float64)
    counts = np.zeros(link_count, dtype=np.float64)
    real = candidates[candidates["candidate_mask"] == 1]
    for row in real.itertuples(index=False):
        links = tuple(int(link_id) for link_id in _json_list(row.route_directed_link_ids))
        if not links:
            continue
        per_link = _finite_float(row.route_length_km) / float(len(links))
        for link_id in links:
            if 0 <= link_id < link_count:
                sums[link_id] += per_link
                counts[link_id] += 1.0
    lengths = np.divide(sums, counts, out=np.ones(link_count, dtype=np.float64), where=counts > 0)
    return lengths.astype(np.float32)


def _build_batch(
    *,
    graphs: np.lib.npyio.NpzFile,
    x_spec: np.ndarray,
    cnn_rows_by_candidate: dict[tuple[Any, ...], int],
    group: pd.DataFrame,
    sample_index: int,
    state: StateView,
    cfg: SolverConfig,
    cnn_compare: dict[str, int],
) -> CandidateBatch:
    slots = int(x_spec.shape[2])
    ordered = group.sort_values("topn_index")
    candidates = tuple(_candidate_from_row(row, cfg, slots) for _, row in ordered.iterrows())
    n_max = int(cfg.n_max)
    topn = tuple(candidates[:n_max])
    mask = np.zeros(n_max, dtype=np.float32)
    action_features = np.zeros((n_max, len(ACTION_FEATURE_COLUMNS)), dtype=np.float32)
    spectrum_tensors = np.zeros((n_max, x_spec.shape[1], slots), dtype=np.float32)

    for index, (candidate, (_, row)) in enumerate(zip(topn, ordered.iterrows())):
        mask[index] = _finite_float(row.get("candidate_mask"))
        action_features[index] = np.asarray(candidate.action_features, dtype=np.float32)
        if not mask[index]:
            continue
        key = (
            row["episode_id"],
            int(row["request_id"]),
            int(row["route_id"]),
            int(row["modulation_id"]),
            int(row["b_start"]),
            int(row["w"]),
        )
        cnn_index = cnn_rows_by_candidate.get(key)
        generated = route_slot_tensor(state, candidate, cfg)
        spectrum_tensors[index] = generated
        if cnn_index is not None and 0 <= cnn_index < len(x_spec):
            cnn_compare["compared"] += 1
            cnn_compare["matched"] += int(np.allclose(generated, x_spec[cnn_index], atol=1e-3))

    return CandidateBatch(
        state=state,
        candidates=candidates,
        topn=topn,
        candidate_mask=mask,
        node_features=np.asarray(graphs["node_features"][sample_index], dtype=np.float32),
        link_features=np.asarray(graphs["link_features"][sample_index], dtype=np.float32),
        global_features=np.asarray(graphs["global_features"][sample_index], dtype=np.float32),
        request_features=np.asarray(graphs["request_features"][sample_index], dtype=np.float32),
        spectrum_tensors=spectrum_tensors,
        action_features=action_features,
    )


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _q_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _solver_config(config: ExperimentConfig) -> SolverConfig:
    raw = config.resolved
    checkpoint_path = raw.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.is_absolute():
            checkpoint = project_root() / checkpoint
        checkpoint_path = str(checkpoint)
    use_neural = raw.get("use_neural", False)
    if isinstance(use_neural, str):
        use_neural = use_neural.strip().lower() in {"1", "true", "yes", "y", "on"}
    return SolverConfig(
        n_max=int(raw.get("n_max", 32)),
        rng_seed=int(raw.get("seed", config.seed)),
        use_neural=bool(use_neural),
        checkpoint_path=checkpoint_path,
        q_score_mode=str(raw.get("q_score_mode", "raw")),
        residual_scale=float(raw.get("residual_scale", 1.0)),
        residual_delta_clip=float(raw.get("residual_delta_clip", 0.0)),
        deeprmsa_prior_score=str(raw.get("deeprmsa_prior_score", "q_head_score")),
        device=str(raw.get("device", config.device)),
        hidden_dim=int(raw.get("hidden_dim", 128)),
    )


def run_ong_solver_eval(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("ong_solver_eval requires dataset_path")

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    cfg = _solver_config(config)
    solver = GnnCnnDqnOngSolver(cfg)
    split_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for split in _splits(config):
        graphs = np.load(config.dataset_path / "gnn" / f"{split}_graphs.npz")
        x_spec = np.load(config.dataset_path / "cnn" / f"{split}_tensors.npz")["X_spec"]
        cnn_index = pd.read_parquet(config.dataset_path / "cnn" / f"{split}_index.parquet")
        candidates = pd.read_parquet(config.dataset_path / "candidates" / f"{split}.parquet")
        traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
        dqn_path = config.dataset_path / "dqn" / f"{split}_transitions.parquet"
        dqn = pd.read_parquet(dqn_path) if dqn_path.exists() else pd.DataFrame()

        dqn_by_request = {
            (row.episode_id, int(row.request_id)): row
            for row in dqn.itertuples(index=False)
        }
        candidate_groups = {
            (episode_id, int(request_id)): group
            for (episode_id, request_id), group in candidates.groupby(["episode_id", "request_id"], sort=False)
        }
        sample_ids = [str(value) for value in graphs["sample_ids"]]
        if config.max_batches <= 0:
            max_samples = len(sample_ids)
        else:
            max_samples = min(len(sample_ids), int(config.batch_size) * int(config.max_batches))
        cnn_rows_by_candidate = _cnn_lookup(cnn_index)
        edge_lengths = _edge_lengths_from_dataset(config.dataset_path, candidates, int(graphs["link_features"].shape[1]))
        target_keys = {
            (sample_id.rsplit(":", 1)[0], int(sample_id.rsplit(":", 1)[1]))
            for sample_id in sample_ids[:max_samples]
        }
        offline_states = _offline_state_views(
            traffic=traffic,
            dqn_by_request=dqn_by_request,
            target_keys=target_keys,
            graphs=graphs,
            slots=int(x_spec.shape[2]),
            edge_lengths_km=edge_lengths,
        )

        selected_energy: list[float] = []
        selected_fragmentation: list[float] = []
        selected_qot_margin: list[float] = []
        selected_delay: list[float] = []
        selected_q: list[float] = []
        valid_q_values: list[float] = []
        valid_candidate_counts: list[float] = []
        predictions: list[dict[str, Any]] = []
        blocked = 0
        missing_groups = 0
        best_agree = 0
        selected_agree = 0
        comparable_best = 0
        comparable_selected = 0
        mask_violations = 0
        tensor_rows_generated = 0
        tensor_rows_expected = 0
        cnn_compare = {"compared": 0, "matched": 0}

        for sample_index, sample_id in enumerate(sample_ids[:max_samples]):
            episode_id, request_id_text = sample_id.rsplit(":", 1)
            key = (episode_id, int(request_id_text))
            group = candidate_groups.get(key)
            if group is None:
                missing_groups += 1
                continue
            state = offline_states.get(key)
            if state is None:
                missing_groups += 1
                continue

            batch = _build_batch(
                graphs=graphs,
                x_spec=x_spec,
                cnn_rows_by_candidate=cnn_rows_by_candidate,
                group=group,
                sample_index=sample_index,
                state=state,
                cfg=cfg,
                cnn_compare=cnn_compare,
            )
            tensor_rows_expected += int(batch.candidate_mask.sum())
            if batch.spectrum_tensors[batch.candidate_mask.astype(bool)].size:
                tensor_rows_generated += int(
                    np.any(batch.spectrum_tensors[batch.candidate_mask.astype(bool)] != 0.0, axis=(1, 2)).sum()
                )

            q_values = solver.q_values(batch)
            if q_values.shape[0] != cfg.n_max:
                q_values = pad_q_scores(q_values, cfg.n_max)
            valid_mask = batch.candidate_mask.astype(bool)
            valid_candidate_counts.append(float(valid_mask.sum()))
            valid_q_values.extend(float(value) for value in q_values[valid_mask])

            if not valid_mask.any():
                blocked += 1
                predictions.append(
                    {
                        "split": split,
                        "episode_id": episode_id,
                        "request_id": key[1],
                        "solver_index": None,
                        "solver_q_value": None,
                        "best_candidate_index": None,
                        "selected_candidate_index": None,
                        "blocked": True,
                    }
                )
                continue

            solver_index = masked_argmax(q_values, batch.candidate_mask)
            if not valid_mask[solver_index]:
                mask_violations += 1
            selected_row = group[group["topn_index"] == solver_index].iloc[0]
            q_value = float(q_values[solver_index])
            selected_q.append(q_value)
            selected_energy.append(_finite_float(selected_row.get("energy_increment"), default=math.nan))
            selected_fragmentation.append(_finite_float(selected_row.get("fragmentation_after"), default=math.nan))
            selected_qot_margin.append(_finite_float(selected_row.get("qot_margin"), default=math.nan))
            selected_delay.append(_finite_float(selected_row.get("delay_ms"), default=math.nan))

            dqn_row = dqn_by_request.get(key)
            best_index = None
            selected_index = None
            if dqn_row is not None:
                best_index = int(dqn_row.best_candidate_index)
                selected_index = int(dqn_row.selected_candidate_index)
                if best_index >= 0:
                    comparable_best += 1
                    best_agree += int(solver_index == best_index)
                if selected_index >= 0:
                    comparable_selected += 1
                    selected_agree += int(solver_index == selected_index)

            predictions.append(
                {
                    "split": split,
                    "episode_id": episode_id,
                    "request_id": key[1],
                    "solver_index": int(solver_index),
                    "solver_q_value": q_value,
                    "best_candidate_index": best_index,
                    "selected_candidate_index": selected_index,
                    "blocked": False,
                    "candidate_mask_sum": float(valid_mask.sum()),
                    "energy_increment": _finite_float(selected_row.get("energy_increment"), default=math.nan),
                    "fragmentation_after": _finite_float(selected_row.get("fragmentation_after"), default=math.nan),
                    "qot_margin": _finite_float(selected_row.get("qot_margin"), default=math.nan),
                    "delay_ms": _finite_float(selected_row.get("delay_ms"), default=math.nan),
                }
            )

        split_frame = pd.DataFrame(predictions)
        if not split_frame.empty:
            prediction_frames.append(split_frame)

        evaluated = len(predictions)
        split_metrics.append(
            {
                "split": split,
                "samples_evaluated": int(evaluated),
                "missing_candidate_groups": int(missing_groups),
                "candidate_group_size": int(candidates["n_max"].mode().iloc[0]) if "n_max" in candidates else cfg.n_max,
                "mean_valid_candidates": _mean(valid_candidate_counts),
                "blocking_rate": float(blocked / evaluated) if evaluated else None,
                "agreement_with_best_candidate_index": float(best_agree / comparable_best) if comparable_best else None,
                "agreement_with_recorded_selected_candidate_index": float(selected_agree / comparable_selected)
                if comparable_selected
                else None,
                "comparable_best_count": int(comparable_best),
                "comparable_selected_count": int(comparable_selected),
                "mean_selected_energy_increment": _mean(selected_energy),
                "mean_selected_fragmentation_after": _mean(selected_fragmentation),
                "mean_selected_qot_margin": _mean(selected_qot_margin),
                "mean_selected_delay_ms": _mean(selected_delay),
                "mean_selected_q_value": _mean(selected_q),
                "valid_q_summary": _q_summary(valid_q_values),
                "mask_violations": int(mask_violations),
                "generated_spectrum_tensor_rate": float(tensor_rows_generated / tensor_rows_expected)
                if tensor_rows_expected
                else None,
                "cnn_reference_tensor_match_rate": float(cnn_compare["matched"] / cnn_compare["compared"])
                if cnn_compare["compared"]
                else None,
                "cnn_reference_tensor_compared": int(cnn_compare["compared"]),
            }
        )

    if prediction_frames:
        predictions_path = run_path / "predictions.parquet"
        pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)

    metrics = {
        "stage": "ong_solver_eval",
        "dataset_path": str(config.dataset_path),
        "solver": {
            "package": "cse2026.ong_solver",
            "class": "GnnCnnDqnOngSolver",
            "n_max": int(cfg.n_max),
            "use_neural": bool(cfg.use_neural),
            "checkpoint_path": cfg.checkpoint_path,
            "trained_checkpoint_loaded": bool(cfg.checkpoint_path),
            "policy": "CandidateQNetwork" if cfg.use_neural or cfg.checkpoint_path else "heuristic_q_head_fallback",
            "diagnostic_only": bool((cfg.use_neural or cfg.checkpoint_path) and not cfg.checkpoint_path),
        },
        "action_feature_columns": list(ACTION_FEATURE_COLUMNS),
        "splits": split_metrics,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
