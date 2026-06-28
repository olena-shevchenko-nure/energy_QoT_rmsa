from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.dagger_tree_ranker import (
    _advantage_gate_metrics,
    _raw_bool,
    _raw_float,
    _raw_int,
    _raw_str,
    _ranker_metrics,
    _safety_guard_from_config,
    _tune_advantage_thresholds,
    _predict_advantage,
    _write_json,
)
from cse2026.experiments.eon.lookahead_override_features import OVERRIDE_FEATURE_NAMES
from cse2026.experiments.eon.lookahead_tree_ranker import _predict as _predict_ranker
from cse2026.experiments.eon.tree_ranker_runtime import ADVANTAGE_FEATURE_NAMES, _load_backend_model


def _load_ranker_dataset(run_path: Path, split: str) -> dict[str, Any] | None:
    npz_path = run_path / f"{split}_dagger_tree_ranker_examples.npz"
    csv_path = run_path / f"{split}_dagger_tree_ranker_examples.csv"
    if not npz_path.exists() or not csv_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "x": np.asarray(data["features"], dtype=np.float32),
        "y": np.asarray(data["targets"], dtype=np.float32),
        "group_sizes": np.asarray(data["group_sizes"], dtype=np.int32),
        "metadata": pd.read_csv(csv_path),
    }


def _load_advantage_dataset(run_path: Path, split: str) -> dict[str, Any] | None:
    npz_path = run_path / f"{split}_advantage_gate_examples.npz"
    csv_path = run_path / f"{split}_advantage_gate_examples.csv"
    if not npz_path.exists() or not csv_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return {
        "x": np.asarray(data["features"], dtype=np.float32),
        "win_y": np.asarray(data["win_targets"], dtype=np.float32),
        "loss_y": np.asarray(data["loss_targets"], dtype=np.float32),
        "delta_y": np.asarray(data["delta_targets"], dtype=np.float32),
        "metadata": pd.read_csv(csv_path),
    }


def _load_history(run_path: Path) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    stdout_path = run_path / "stdout.log"
    if not stdout_path.exists():
        return history
    for line in stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{") or "dagger_tree_ranker_iteration" not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event.pop("event", None)
        history.append(event)
    return history


def finalize_run(run_path: Path) -> dict[str, Any]:
    config = ExperimentConfig.from_file(run_path / "config.yaml", root=Path.cwd())
    backend = _raw_str(config, "tree_ranker_backend", "lightgbm").strip().lower()
    model_suffix = "json" if backend == "xgboost" else "txt"
    model_path = run_path / f"{backend}_dagger_tree_ranker.{model_suffix}"
    ranker_path = run_path / "tree_ranker.json"
    ranker_model = _load_backend_model(backend, model_path)

    train_data = _load_ranker_dataset(run_path, "train")
    if train_data is None:
        raise ValueError(f"Missing train ranker dataset in {run_path}")
    eval_data = _load_ranker_dataset(run_path, "eval")
    train_scores = _predict_ranker(backend, ranker_model, train_data["x"])
    eval_scores = _predict_ranker(backend, ranker_model, eval_data["x"]) if eval_data is not None else None
    final_train_metrics = _ranker_metrics(train_data["metadata"], train_scores)
    eval_metrics = _ranker_metrics(eval_data["metadata"], eval_scores) if eval_data is not None and eval_scores is not None else None

    train_advantage = _load_advantage_dataset(run_path, "train")
    if train_advantage is None:
        raise ValueError(f"Missing train advantage dataset in {run_path}")
    eval_advantage = _load_advantage_dataset(run_path, "eval")
    win_model_path = run_path / f"{backend}_advantage_win.{model_suffix}"
    loss_model_path = run_path / f"{backend}_advantage_loss.{model_suffix}"
    delta_model_path = run_path / f"{backend}_advantage_delta.{model_suffix}"
    win_model = _load_backend_model(backend, win_model_path)
    loss_model = _load_backend_model(backend, loss_model_path)
    delta_model = _load_backend_model(backend, delta_model_path)

    train_win_prob = _predict_advantage(backend, win_model, train_advantage["x"])
    train_loss_prob = _predict_advantage(backend, loss_model, train_advantage["x"])
    train_delta_pred = _predict_advantage(backend, delta_model, train_advantage["x"])
    eval_win_prob = (
        _predict_advantage(backend, win_model, eval_advantage["x"])
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0
        else None
    )
    eval_loss_prob = (
        _predict_advantage(backend, loss_model, eval_advantage["x"])
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0
        else None
    )
    eval_delta_pred = (
        _predict_advantage(backend, delta_model, eval_advantage["x"])
        if eval_advantage is not None and eval_advantage["x"].shape[0] > 0
        else None
    )
    tuned_thresholds = _tune_advantage_thresholds(
        dataset=eval_advantage if eval_advantage is not None and eval_advantage["x"].shape[0] > 0 else train_advantage,
        win_prob=eval_win_prob if eval_win_prob is not None else train_win_prob,
        loss_prob=eval_loss_prob if eval_loss_prob is not None else train_loss_prob,
        delta_pred=eval_delta_pred if eval_delta_pred is not None else train_delta_pred,
        config=config,
    )
    train_advantage_metrics = _advantage_gate_metrics(
        dataset=train_advantage,
        win_prob=train_win_prob,
        loss_prob=train_loss_prob,
        delta_pred=train_delta_pred,
        config=config,
        thresholds=tuned_thresholds,
    )
    eval_advantage_metrics = (
        _advantage_gate_metrics(
            dataset=eval_advantage,
            win_prob=eval_win_prob,
            loss_prob=eval_loss_prob,
            delta_pred=eval_delta_pred,
            config=config,
            thresholds=tuned_thresholds,
        )
        if eval_advantage is not None
        and eval_advantage["x"].shape[0] > 0
        and eval_win_prob is not None
        and eval_loss_prob is not None
        and eval_delta_pred is not None
        else None
    )

    base_policy = _raw_str(config, "dagger_base_policy", "energy-aware-ksp-bm-ff")
    selection_mode = _raw_str(config, "tree_ranker_selection_mode", "guarded")
    dagger_follow_selection_mode = _raw_str(
        config,
        "dagger_follow_selection_mode",
        "guarded" if str(selection_mode).strip().lower() in {"advantage", "positive_advantage"} else selection_mode,
    )
    residual_beta = _raw_float(config, "tree_ranker_residual_beta", 0.05)
    selection_margin = _raw_float(config, "tree_ranker_selection_margin", 0.005)
    safety_guard = _safety_guard_from_config(config)
    history = _load_history(run_path)
    train_split = _raw_str(config, "dagger_train_split", "train")
    eval_split = _raw_str(config, "dagger_eval_split", "val")
    iterations = max(1, _raw_int(config, "dagger_iterations", 1))
    follow_trained = _raw_bool(config, "dagger_follow_trained_ranker", True)
    advantage_gate_meta = {
        "enabled": True,
        "backend": backend,
        "feature_names": list(ADVANTAGE_FEATURE_NAMES),
        "win_model_path": str(win_model_path),
        "loss_model_path": str(loss_model_path),
        "delta_model_path": str(delta_model_path),
        "win_min_accepted_delta": _raw_int(config, "advantage_gate_win_min_accepted_delta", 1),
        "loss_min_accepted_delta": _raw_int(config, "advantage_gate_loss_min_accepted_delta", 1),
        "delta_reward_weight": _raw_float(config, "advantage_gate_delta_reward_weight", 0.0),
        "auto_tuned_thresholds": _raw_bool(config, "advantage_gate_auto_tune_thresholds", True),
        **tuned_thresholds,
    }
    ranker_meta = {
        "backend": backend,
        "model_path": str(model_path),
        "feature_names": list(OVERRIDE_FEATURE_NAMES),
        "candidate_pool": "all_topn",
        "selection_mode": selection_mode,
        "residual_beta": float(residual_beta),
        "selection_margin": float(selection_margin),
        "base_policy": base_policy,
        "safety_guard": safety_guard,
        "advantage_gate": advantage_gate_meta,
        "dagger": {
            "iterations": int(iterations),
            "follow_trained_ranker": bool(follow_trained),
            "follow_selection_mode": dagger_follow_selection_mode,
            "train_split": train_split,
            "eval_split": eval_split,
            "lookahead_horizon": _raw_int(config, "dagger_lookahead_horizon", 12),
            "lookahead_rollout_policy": _raw_str(config, "dagger_lookahead_rollout_policy", base_policy),
            "rank_target_mode": _raw_str(config, "dagger_rank_target_mode", "shifted_utility"),
        },
        "utility": {
            "accepted_weight": _raw_float(config, "dagger_utility_accepted_weight", 2.0),
            "block_penalty": _raw_float(config, "dagger_utility_block_penalty", 1.5),
            "energy_weight": _raw_float(config, "dagger_utility_energy_weight", 0.25),
            "fragmentation_weight": _raw_float(config, "dagger_utility_fragmentation_weight", 0.80),
            "qot_weight": _raw_float(config, "dagger_utility_qot_weight", 0.20),
            "energy_norm_w": _raw_float(config, "dagger_utility_energy_norm_w", _raw_float(config, "energy_norm_w", 1200.0)),
        },
    }
    _write_json(ranker_path, ranker_meta)
    metrics = {
        "stage": "train_dagger_tree_ranker",
        "finalized_from_saved_artifacts": True,
        "dataset_path": str(config.dataset_path),
        "backend": backend,
        "model_path": str(model_path),
        "ranker_path": str(ranker_path),
        "base_policy": base_policy,
        "selection_mode": selection_mode,
        "safety_guard": safety_guard,
        "history": history,
        "train": final_train_metrics,
        "eval": eval_metrics,
        "train_groups": int(train_data["group_sizes"].size),
        "train_rows": int(train_data["x"].shape[0]),
        "eval_groups": None if eval_data is None else int(eval_data["group_sizes"].size),
        "eval_rows": None if eval_data is None else int(eval_data["x"].shape[0]),
        "advantage_gate": {
            "train": train_advantage_metrics,
            "eval": eval_advantage_metrics,
            "train_rows": int(train_advantage["x"].shape[0]),
            "eval_rows": None if eval_advantage is None else int(eval_advantage["x"].shape[0]),
        },
        "ranker": ranker_meta,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a DAgger advantage run from saved artifacts.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing saved DAgger and advantage artifacts.")
    args = parser.parse_args()
    metrics = finalize_run(Path(args.run_dir))
    print(
        json.dumps(
            {
                "ranker_path": metrics["ranker_path"],
                "train": metrics["train"],
                "eval": metrics["eval"],
                "advantage_gate": metrics["advantage_gate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
