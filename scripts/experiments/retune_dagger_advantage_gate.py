from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.experiments.config import ExperimentConfig
from cse2026.experiments.eon.dagger_tree_ranker import (
    _advantage_gate_metrics,
    _predict_advantage,
    _raw_bool,
    _tune_advantage_thresholds,
    _write_json,
)
from cse2026.experiments.eon.tree_ranker_runtime import _load_backend_model


def _load_advantage_dataset(run_path: Path, split: str) -> dict[str, Any]:
    npz_path = run_path / f"{split}_advantage_gate_examples.npz"
    csv_path = run_path / f"{split}_advantage_gate_examples.csv"
    if not npz_path.exists() or not csv_path.exists():
        raise FileNotFoundError(f"Missing advantage dataset for split={split} in {run_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {
        "x": np.asarray(data["features"], dtype=np.float32),
        "win_y": np.asarray(data["win_targets"], dtype=np.float32),
        "loss_y": np.asarray(data["loss_targets"], dtype=np.float32),
        "delta_y": np.asarray(data["delta_targets"], dtype=np.float32),
        "metadata": pd.read_csv(csv_path),
    }


def _with_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    raw = dict(config.raw)
    resolved = dict(config.resolved)
    raw.update(overrides)
    resolved.update(overrides)
    return replace(config, raw=raw, resolved=resolved)


def _model_suffix(backend: str) -> str:
    return "json" if backend == "xgboost" else "txt"


def retune_run(
    *,
    run_path: Path,
    output_path: Path,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    min_win_prob_grid: str,
    max_loss_prob_grid: str,
    min_delta_pred_grid: str,
) -> dict[str, Any]:
    config = ExperimentConfig.from_file(run_path / "config.yaml", root=Path.cwd())
    backend = str(config.resolved.get("tree_ranker_backend", config.raw.get("tree_ranker_backend", "lightgbm"))).strip().lower()
    suffix = _model_suffix(backend)
    overrides = {
        "advantage_gate_auto_tune_thresholds": True,
        "advantage_gate_tune_max_loss_rate": float(max_loss_rate),
        "advantage_gate_tune_min_override_count": int(min_override_count),
        "advantage_gate_tune_min_total_delta": float(min_total_delta),
        "advantage_gate_tune_min_win_prob_grid": min_win_prob_grid,
        "advantage_gate_tune_max_loss_prob_grid": max_loss_prob_grid,
        "advantage_gate_tune_min_delta_pred_grid": min_delta_pred_grid,
    }
    config = _with_overrides(config, overrides)

    train = _load_advantage_dataset(run_path, "train")
    eval_data = _load_advantage_dataset(run_path, "eval")
    win_model = _load_backend_model(backend, run_path / f"{backend}_advantage_win.{suffix}")
    loss_model = _load_backend_model(backend, run_path / f"{backend}_advantage_loss.{suffix}")
    delta_model = _load_backend_model(backend, run_path / f"{backend}_advantage_delta.{suffix}")

    train_win = _predict_advantage(backend, win_model, train["x"])
    train_loss = _predict_advantage(backend, loss_model, train["x"])
    train_delta = _predict_advantage(backend, delta_model, train["x"])
    eval_win = _predict_advantage(backend, win_model, eval_data["x"])
    eval_loss = _predict_advantage(backend, loss_model, eval_data["x"])
    eval_delta = _predict_advantage(backend, delta_model, eval_data["x"])

    thresholds = _tune_advantage_thresholds(
        dataset=eval_data,
        win_prob=eval_win,
        loss_prob=eval_loss,
        delta_pred=eval_delta,
        config=config,
    )
    train_metrics = _advantage_gate_metrics(
        dataset=train,
        win_prob=train_win,
        loss_prob=train_loss,
        delta_pred=train_delta,
        config=config,
        thresholds=thresholds,
    )
    eval_metrics = _advantage_gate_metrics(
        dataset=eval_data,
        win_prob=eval_win,
        loss_prob=eval_loss,
        delta_pred=eval_delta,
        config=config,
        thresholds=thresholds,
    )

    source_ranker = json.loads((run_path / "tree_ranker.json").read_text(encoding="utf-8"))
    advantage_gate = dict(source_ranker.get("advantage_gate") or {})
    advantage_gate.update(thresholds)
    advantage_gate["auto_tuned_thresholds"] = _raw_bool(config, "advantage_gate_auto_tune_thresholds", True)
    advantage_gate["retuned_from"] = str(run_path / "tree_ranker.json")
    advantage_gate["retune_constraints"] = overrides
    source_ranker["advantage_gate"] = advantage_gate
    source_ranker["retuned"] = True
    source_ranker["retuned_from"] = str(run_path / "tree_ranker.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, source_ranker)

    metrics = {
        "run_path": str(run_path),
        "output_ranker_path": str(output_path),
        "backend": backend,
        "thresholds": thresholds,
        "constraints": overrides,
        "train": train_metrics,
        "eval": eval_metrics,
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".metrics.json"), metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Retune DAgger advantage gate thresholds from saved examples.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-loss-rate", type=float, default=0.005)
    parser.add_argument("--min-override-count", type=int, default=50)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--min-win-prob-grid", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--max-loss-prob-grid", default="0.005,0.01,0.02,0.03,0.05,0.08")
    parser.add_argument("--min-delta-pred-grid", default="-0.05,0.0,0.02,0.05,0.10,0.20,0.40")
    args = parser.parse_args()

    metrics = retune_run(
        run_path=Path(args.run_dir),
        output_path=Path(args.output),
        max_loss_rate=args.max_loss_rate,
        min_override_count=args.min_override_count,
        min_total_delta=args.min_total_delta,
        min_win_prob_grid=args.min_win_prob_grid,
        max_loss_prob_grid=args.max_loss_prob_grid,
        min_delta_pred_grid=args.min_delta_pred_grid,
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
