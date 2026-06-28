#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.train_dqn import (
    _build_model as _build_dqn_model,
    _device,
    _model_forward as _dqn_forward,
)
from train_full_dqn_orate60_distill import _json_safe, _load_full_dqn_checkpoint, _resolve_cli_path, _write_json


STATE_KEYS = (
    "node_features",
    "link_features",
    "global_features",
    "request_features",
    "spectrum_tensors",
    "action_features",
    "route_link_mask",
    "route_basic_features",
    "block_bounds",
    "candidate_mask",
)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    denominator = float(denominator)
    if denominator <= 0.0:
        return None
    return float(float(numerator) / denominator)


def _distribution(values: np.ndarray | pd.Series) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p05": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None}
    q = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "mean": float(np.mean(arr)),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
    }


def _load_metadata(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "online_base_topn_examples.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    metadata = pd.read_csv(path).reset_index(drop=True)
    if metadata.empty:
        return metadata
    metadata["is_base"] = metadata["is_base"].astype(bool)
    metadata["accepted_delta_vs_base"] = metadata["accepted_delta_vs_base"].astype(float)
    metadata["secondary_delta_vs_base"] = metadata["secondary_delta_vs_base"].astype(float)
    return metadata


def _load_neural_states(input_dir: Path) -> dict[str, np.ndarray]:
    path = input_dir / "online_base_topn_neural_states.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing neural state file: {path}")
    npz = np.load(path, allow_pickle=True)
    data = {str(key): np.asarray(npz[key]) for key in npz.files}
    for key in ("group_ids", "base_index", "accepted_delta_vs_base", "secondary_delta_vs_base", "label_mask", "edge_index") + STATE_KEYS:
        if key not in data:
            raise ValueError(f"Neural state file is missing {key}")
    return data


def _tensors_for_slice(data: dict[str, np.ndarray], start: int, end: int, *, device: str, torch: Any) -> dict[str, Any]:
    return {
        key: torch.as_tensor(np.asarray(data[key][start:end]), dtype=torch.bool if key == "candidate_mask" else torch.float32, device=device)
        for key in STATE_KEYS
    }


def _teacher_scores(
    *,
    config: ExperimentConfig,
    data: dict[str, np.ndarray],
    checkpoint_path: Path,
    batch_size: int,
) -> np.ndarray:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = _device(config, torch)
    model, _pretrained = _build_dqn_model(config, device, torch)
    _load_full_dqn_checkpoint(model, checkpoint_path, device=device, torch=torch)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    n_states = int(np.asarray(data["candidate_mask"]).shape[0])
    n_max = int(np.asarray(data["candidate_mask"]).shape[1])
    result = np.full((n_states, n_max), -1.0e9, dtype=np.float32)
    edge_index = torch.as_tensor(np.asarray(data["edge_index"], dtype=np.int64), dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, n_states, max(1, int(batch_size))):
            end = min(n_states, start + max(1, int(batch_size)))
            tensors = _tensors_for_slice(data, start, end, device=device, torch=torch)
            values = _dqn_forward(model, tensors, edge_index)
            values = values.masked_fill(~tensors["candidate_mask"], -1.0e9)
            result[start:end] = values.detach().cpu().numpy().astype(np.float32)
    return result


def _best_labeled_index(delta: np.ndarray, secondary: np.ndarray, label_mask: np.ndarray) -> int:
    valid = np.flatnonzero(np.asarray(label_mask, dtype=bool))
    if valid.size == 0:
        return -1
    order = sorted((int(index) for index in valid), key=lambda index: (float(delta[index]), float(secondary[index]), -index), reverse=True)
    return int(order[0])


def _masked_argmax_np(scores: np.ndarray, mask: np.ndarray) -> int:
    valid = np.flatnonzero(np.asarray(mask, dtype=bool))
    if valid.size == 0:
        return -1
    local = int(valid[int(np.argmax(np.asarray(scores, dtype=np.float32)[valid]))])
    return local


def _context_summary(groups: pd.DataFrame) -> list[dict[str, Any]]:
    if groups.empty or not {"traffic_scenario", "load_name"}.issubset(groups.columns):
        return []
    rows: list[dict[str, Any]] = []
    for (scenario, load), group in groups.groupby(["traffic_scenario", "load_name"], sort=True):
        rows.append(
            {
                "traffic_scenario": str(scenario),
                "load_name": str(load),
                "groups": int(len(group)),
                "groups_with_win": int(group["has_win"].sum()),
                "state_positive_rate": _safe_rate(int(group["has_win"].sum()), len(group)),
                "oracle_gain_accepted_sum": float(group["oracle_best_delta"].sum()),
                "teacher_delta_sum": float(group["teacher_delta"].sum()),
                "teacher_win_capture_groups": int((group["has_win"] & (group["teacher_delta"] > 0.0)).sum()),
                "teacher_win_capture_rate": _safe_rate(int((group["has_win"] & (group["teacher_delta"] > 0.0)).sum()), int(group["has_win"].sum())),
                "teacher_missed_win_groups": int((group["has_win"] & (group["teacher_delta"] <= 0.0)).sum()),
                "teacher_loss_on_win_groups": int((group["has_win"] & (group["teacher_delta"] < 0.0)).sum()),
                "residual_oracle_gain_after_teacher_sum": float(np.maximum(group["oracle_best_delta"] - group["teacher_delta"], 0.0).sum()),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["residual_oracle_gain_after_teacher_sum"]),
            -float(row["oracle_gain_accepted_sum"]),
            str(row["traffic_scenario"]),
            str(row["load_name"]),
        )
    )
    return rows


def audit_signal(
    *,
    config: ExperimentConfig,
    input_dir: Path,
    output_dir: Path,
    teacher_dqn_checkpoint: Path | None,
    batch_size: int,
    label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata(input_dir)
    data = _load_neural_states(input_dir)

    group_ids = np.asarray(data["group_ids"], dtype=np.int64)
    base_index = np.asarray(data["base_index"], dtype=np.int64)
    accepted_delta = np.asarray(data["accepted_delta_vs_base"], dtype=np.float32)
    secondary_delta = np.asarray(data["secondary_delta_vs_base"], dtype=np.float32)
    label_mask = np.asarray(data["label_mask"], dtype=bool)
    candidate_mask = np.asarray(data["candidate_mask"], dtype=bool)
    eval_mask = label_mask & candidate_mask

    teacher_score_matrix: np.ndarray | None = None
    if teacher_dqn_checkpoint is not None:
        teacher_score_matrix = _teacher_scores(
            config=config,
            data=data,
            checkpoint_path=teacher_dqn_checkpoint,
            batch_size=int(batch_size),
        )

    group_meta = metadata.drop_duplicates("group_id").set_index("group_id", drop=False)
    rows: list[dict[str, Any]] = []
    for position, group_id_raw in enumerate(group_ids):
        group_id = int(group_id_raw)
        mask = eval_mask[position]
        delta = accepted_delta[position]
        secondary = secondary_delta[position]
        nonbase_mask = mask.copy()
        bidx = int(base_index[position])
        if 0 <= bidx < nonbase_mask.shape[0]:
            nonbase_mask[bidx] = False
        win_mask = nonbase_mask & (delta > 0.0)
        loss_mask = nonbase_mask & (delta < 0.0)
        oracle_index = _best_labeled_index(delta, secondary, mask)
        oracle_delta = float(delta[oracle_index]) if oracle_index >= 0 else 0.0
        oracle_secondary = float(secondary[oracle_index]) if oracle_index >= 0 else 0.0

        teacher_index = None
        teacher_delta = None
        teacher_secondary = None
        teacher_score_margin = None
        teacher_labeled = None
        if teacher_score_matrix is not None:
            teacher_index_int = _masked_argmax_np(teacher_score_matrix[position], mask)
            teacher_index = int(teacher_index_int)
            teacher_labeled = bool(0 <= teacher_index_int < mask.shape[0] and mask[teacher_index_int])
            if teacher_labeled:
                teacher_delta = float(delta[teacher_index_int])
                teacher_secondary = float(secondary[teacher_index_int])
            valid_scores = np.asarray(teacher_score_matrix[position], dtype=np.float32)[mask]
            valid_scores = valid_scores[np.isfinite(valid_scores) & (valid_scores > -1.0e8)]
            if valid_scores.size >= 2:
                top2 = np.partition(valid_scores, -2)[-2:]
                teacher_score_margin = float(np.max(top2) - np.min(top2))
            else:
                teacher_score_margin = 0.0

        meta = group_meta.loc[group_id] if group_id in group_meta.index else None
        rows.append(
            {
                "label": str(label),
                "group_id": group_id,
                "episode_id": "" if meta is None else str(meta.get("episode_id", "")),
                "request_id": None if meta is None else int(meta.get("request_id", -1)),
                "position": None if meta is None else int(meta.get("position", -1)),
                "traffic_scenario": "" if meta is None else str(meta.get("traffic_scenario", "")),
                "load_name": "" if meta is None else str(meta.get("load_name", "")),
                "base_index": bidx,
                "labeled_candidates": int(mask.sum()),
                "nonbase_candidates": int(nonbase_mask.sum()),
                "win_candidates": int(win_mask.sum()),
                "loss_candidates": int(loss_mask.sum()),
                "tie_candidates": int((nonbase_mask & (delta == 0.0)).sum()),
                "has_win": bool(win_mask.any()),
                "has_loss": bool(loss_mask.any()),
                "oracle_index": int(oracle_index),
                "oracle_is_base": bool(int(oracle_index) == bidx),
                "oracle_best_delta": float(oracle_delta),
                "oracle_best_secondary_delta": float(oracle_secondary),
                "teacher_index": teacher_index,
                "teacher_is_base": None if teacher_index is None else bool(int(teacher_index) == bidx),
                "teacher_labeled": teacher_labeled,
                "teacher_delta": teacher_delta,
                "teacher_secondary_delta": teacher_secondary,
                "teacher_score_margin": teacher_score_margin,
                "teacher_matches_oracle": None if teacher_index is None else bool(int(teacher_index) == int(oracle_index)),
                "teacher_captures_win": None
                if teacher_delta is None
                else bool(win_mask.any() and float(teacher_delta) > 0.0),
                "teacher_misses_win": None
                if teacher_delta is None
                else bool(win_mask.any() and float(teacher_delta) <= 0.0),
                "teacher_loss_on_win": None
                if teacher_delta is None
                else bool(win_mask.any() and float(teacher_delta) < 0.0),
                "teacher_regret_to_oracle": None if teacher_delta is None else float(oracle_delta - float(teacher_delta)),
            }
        )

    groups = pd.DataFrame(rows)
    groups.to_csv(output_dir / f"{label}_group_signal.csv", index=False)

    nonbase = metadata[~metadata["is_base"].astype(bool)]
    nonbase_delta = nonbase["accepted_delta_vs_base"].to_numpy(dtype=np.float32) if not nonbase.empty else np.asarray([])
    summary: dict[str, Any] = {
        "label": str(label),
        "input_dir": str(input_dir),
        "rows": int(len(metadata)),
        "groups": int(len(groups)),
        "non_base_rows": int(len(nonbase)),
        "candidate_pool_positive_rows": int((nonbase_delta > 0.0).sum()),
        "candidate_pool_loss_rows": int((nonbase_delta < 0.0).sum()),
        "candidate_pool_tie_rows": int((nonbase_delta == 0.0).sum()),
        "candidate_pool_positive_rate": _safe_rate(int((nonbase_delta > 0.0).sum()), len(nonbase_delta)),
        "candidate_pool_loss_rate": _safe_rate(int((nonbase_delta < 0.0).sum()), len(nonbase_delta)),
        "candidate_pool_delta_sum": float(np.sum(nonbase_delta)) if nonbase_delta.size else 0.0,
        "groups_with_win": int(groups["has_win"].sum()),
        "state_positive_rate": _safe_rate(int(groups["has_win"].sum()), len(groups)),
        "groups_with_loss": int(groups["has_loss"].sum()),
        "state_loss_available_rate": _safe_rate(int(groups["has_loss"].sum()), len(groups)),
        "oracle_gain_accepted_sum": float(groups["oracle_best_delta"].sum()) if not groups.empty else 0.0,
        "oracle_best_delta_distribution": _distribution(groups["oracle_best_delta"] if not groups.empty else np.asarray([])),
        "context": _context_summary(groups),
    }
    if teacher_score_matrix is not None:
        teacher_delta_arr = groups["teacher_delta"].fillna(0.0).to_numpy(dtype=np.float32)
        has_win = groups["has_win"].astype(bool).to_numpy()
        teacher_win = teacher_delta_arr > 0.0
        teacher_loss = teacher_delta_arr < 0.0
        teacher_regret = groups["teacher_regret_to_oracle"].fillna(0.0).to_numpy(dtype=np.float32)
        residual_gain = np.maximum(groups["oracle_best_delta"].to_numpy(dtype=np.float32) - teacher_delta_arr, 0.0)
        summary["teacher"] = {
            "checkpoint": str(teacher_dqn_checkpoint),
            "teacher_nonbase_groups": int((~groups["teacher_is_base"].fillna(True).astype(bool)).sum()),
            "teacher_nonbase_rate": _safe_rate(int((~groups["teacher_is_base"].fillna(True).astype(bool)).sum()), len(groups)),
            "teacher_win_groups": int(teacher_win.sum()),
            "teacher_loss_groups": int(teacher_loss.sum()),
            "teacher_delta_sum": float(np.sum(teacher_delta_arr)),
            "teacher_delta_distribution": _distribution(teacher_delta_arr),
            "teacher_oracle_match_groups": int(groups["teacher_matches_oracle"].fillna(False).astype(bool).sum()),
            "teacher_oracle_match_rate": _safe_rate(
                int(groups["teacher_matches_oracle"].fillna(False).astype(bool).sum()), len(groups)
            ),
            "teacher_win_capture_groups": int((has_win & teacher_win).sum()),
            "teacher_win_capture_rate": _safe_rate(int((has_win & teacher_win).sum()), int(has_win.sum())),
            "teacher_missed_win_groups": int((has_win & ~teacher_win).sum()),
            "teacher_loss_on_win_groups": int((has_win & teacher_loss).sum()),
            "teacher_regret_to_oracle_sum": float(np.sum(teacher_regret)),
            "residual_oracle_gain_after_teacher_sum": float(np.sum(residual_gain)),
            "residual_oracle_gain_after_teacher_distribution": _distribution(residual_gain),
            "teacher_score_margin_distribution": _distribution(groups["teacher_score_margin"].fillna(0.0).to_numpy(dtype=np.float32)),
        }

    _write_json(output_dir / f"{label}_signal_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit whether Top32 counterfactual labels contain usable signal beyond teacher imitation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-dqn-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--label", type=str, default="signal")
    args = parser.parse_args(argv)

    config = ExperimentConfig.from_file(args.config, root=ROOT)
    input_dir = _resolve_cli_path(str(args.input_dir))
    if input_dir is None:
        input_dir = args.input_dir
    output_dir = _resolve_cli_path(str(args.output_dir))
    if output_dir is None:
        output_dir = args.output_dir
    teacher_checkpoint = _resolve_cli_path(str(args.teacher_dqn_checkpoint or ""))
    summary = audit_signal(
        config=config,
        input_dir=input_dir,
        output_dir=output_dir,
        teacher_dqn_checkpoint=teacher_checkpoint,
        batch_size=int(args.batch_size),
        label=str(args.label),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
