from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import GnnCnnDqnOngSolver

from ..config import ExperimentConfig
from .lookahead_override_features import (
    OVERRIDE_FEATURE_NAMES,
    candidate_indices_for_override,
    override_feature_matrix,
)
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _traffic_jsonl_for_episode,
)
from .lookahead_oracle import _select_rollout_index, _solver_config


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_path(config: ExperimentConfig, key: str) -> Path | None:
    value = config.resolved.get(key, config.raw.get(key))
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root() / path


def _label_path(config: ExperimentConfig, split: str) -> Path | None:
    return _resolve_path(config, f"{split}_lookahead_label_path") or _resolve_path(config, "lookahead_label_path")


def _positive_override(row: Any, *, accepted_min: int, reward_min: float) -> bool:
    accepted_delta = int(getattr(row, "oracle_accepted_delta_vs_q_head", 0))
    reward_delta = float(getattr(row, "oracle_reward_delta_vs_q_head", 0.0))
    return bool(accepted_delta >= int(accepted_min) or reward_delta >= float(reward_min))


def _load_label_table(path: Path, *, accepted_min: int, reward_min: float) -> pd.DataFrame:
    labels = pd.read_csv(path)
    labels = labels[labels["oracle_index"].astype(int) >= 0].copy()
    labels["positive_override"] = [
        _positive_override(row, accepted_min=accepted_min, reward_min=reward_min)
        for row in labels.itertuples(index=False)
    ]
    return labels


def _build_split_examples(
    *,
    config: ExperimentConfig,
    split: str,
    labels: pd.DataFrame,
    solver: GnnCnnDqnOngSolver,
    run_path: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    label_by_key = {
        (str(row.episode_id), int(row.request_id)): row
        for row in labels.itertuples(index=False)
    }
    episode_ids = tuple(str(value) for value in labels["episode_id"].drop_duplicates().tolist())
    top_k = _raw_int(config, "override_top_k", _raw_int(config, "lookahead_top_k", 4))
    include_j_total = _raw_bool(config, "override_include_j_total", True)
    state_policy = str(config.resolved.get("lookahead_state_policy", config.raw.get("lookahead_state_policy", "q_head_heuristic")))
    rows: list[np.ndarray] = []
    targets: list[int] = []
    metadata: list[dict[str, Any]] = []

    for episode_id in episode_ids:
        episode = traffic[traffic["episode_id"].astype(str) == episode_id].sort_values("request_id").reset_index(drop=True)
        if episode.empty:
            continue
        max_request = int(labels[labels["episode_id"].astype(str) == episode_id]["request_id"].max())
        episode = episode[episode["request_id"].astype(int) <= max_request].reset_index(drop=True)
        traffic_path = _traffic_jsonl_for_episode(run_path, f"{split}_{episode_id}", episode)
        env = _make_env(
            episode_id=episode_id,
            traffic_path=traffic_path,
            request_count=len(episode),
            seed=int(episode["seed"].iloc[0]) if "seed" in episode else int(config.seed),
            config=config,
        )
        env.reset(seed=int(config.seed))

        for _, request in episode.iterrows():
            key = (episode_id, int(request["request_id"]))
            batch = solver.candidate_batch(env)
            label = label_by_key.get(key)
            if label is not None and np.asarray(batch.candidate_mask, dtype=bool).any():
                candidate_indices = candidate_indices_for_override(
                    batch,
                    solver.config.n_max,
                    top_k=top_k,
                    include_j_total=include_j_total,
                )
                features, kept_indices = override_feature_matrix(
                    batch=batch,
                    candidate_indices=candidate_indices,
                    n_max=solver.config.n_max,
                )
                oracle_index = int(label.oracle_index)
                positive_state = bool(label.positive_override) and oracle_index in kept_indices
                for feature_row, candidate_index in zip(features, kept_indices):
                    target = int(positive_state and int(candidate_index) == oracle_index)
                    rows.append(feature_row)
                    targets.append(target)
                    metadata.append(
                        {
                            "split": split,
                            "episode_id": episode_id,
                            "request_id": int(request["request_id"]),
                            "candidate_index": int(candidate_index),
                            "oracle_index": oracle_index,
                            "positive_state": bool(positive_state),
                            "target": int(target),
                            "accepted_delta": int(getattr(label, "oracle_accepted_delta_vs_q_head", 0)),
                            "reward_delta": float(getattr(label, "oracle_reward_delta_vs_q_head", 0.0)),
                        }
                    )

            state_index = _select_rollout_index(batch, state_policy, solver.config.n_max)
            if state_index < 0:
                action = int(solver.adapter(env).block_action(env))
            else:
                action = int(batch.topn[state_index].action)
            observation, reward, terminated, truncated, info = env.step(action)
            del observation, reward, info
            if bool(terminated) or bool(truncated):
                break

    if not rows:
        return (
            np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            pd.DataFrame(metadata),
        )
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(metadata)


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if train.size == 0:
        return np.zeros((len(OVERRIDE_FEATURE_NAMES),), dtype=np.float32), np.ones((len(OVERRIDE_FEATURE_NAMES),), dtype=np.float32)
    mean = train.mean(axis=0).astype(np.float32)
    scale = train.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _train_logistic(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: ExperimentConfig,
) -> tuple[np.ndarray, float, list[dict[str, Any]]]:
    rng = np.random.default_rng(int(config.seed))
    x = (x_train - mean) / np.maximum(scale, 1e-6)
    y = y_train.astype(np.float32)
    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = 0.0
    epochs = _raw_int(config, "override_epochs", 200)
    batch_size = max(1, _raw_int(config, "override_batch_size", 256))
    lr = _raw_float(config, "override_learning_rate", 0.05)
    l2 = _raw_float(config, "override_l2", 0.001)
    positive_weight = _raw_float(config, "override_positive_weight", 0.0)
    if positive_weight <= 0.0:
        positives = max(float(y.sum()), 1.0)
        negatives = max(float(y.size - y.sum()), 1.0)
        positive_weight = min(negatives / positives, _raw_float(config, "override_max_positive_weight", 50.0))
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        order = np.arange(y.size)
        rng.shuffle(order)
        for start in range(0, y.size, batch_size):
            batch_indices = order[start : start + batch_size]
            xb = x[batch_indices]
            yb = y[batch_indices]
            logits = xb @ weights + bias
            probabilities = _sigmoid(logits)
            sample_weight = np.where(yb > 0.5, positive_weight, 1.0).astype(np.float32)
            error = (probabilities - yb) * sample_weight
            normalizer = max(float(sample_weight.sum()), 1.0)
            grad_w = (xb.T @ error) / normalizer + float(l2) * weights
            grad_b = float(error.sum() / normalizer)
            weights -= float(lr) * grad_w.astype(np.float32)
            bias -= float(lr) * grad_b
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            metrics = _binary_metrics(_sigmoid(x @ weights + bias), y, threshold=0.5)
            metrics.update({"epoch": int(epoch), "phase": "train"})
            history.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
    return weights.astype(np.float32), float(bias), history


def _train_mlp(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, np.ndarray | float], list[dict[str, Any]]]:
    from cse2026.ong_solver.models import require_torch

    torch = require_torch()
    device = str(config.resolved.get("device", config.device))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(int(config.seed))
    torch.manual_seed(int(config.seed))
    x = ((x_train - mean) / np.maximum(scale, 1e-6)).astype(np.float32)
    y = y_train.astype(np.float32)
    hidden_dim = _raw_int(config, "override_hidden_dim", 64)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    ).to(device)
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(y.size - y.sum()), 1.0)
    positive_weight = _raw_float(config, "override_positive_weight", 0.0)
    if positive_weight <= 0.0:
        positive_weight = min(negatives / positives, _raw_float(config, "override_max_positive_weight", 50.0))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_raw_float(config, "override_learning_rate", 0.001),
        weight_decay=_raw_float(config, "override_l2", 0.001),
    )
    epochs = _raw_int(config, "override_epochs", 300)
    batch_size = max(1, _raw_int(config, "override_batch_size", 256))
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y[:, None], dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = np.arange(y.size)
        rng.shuffle(order)
        model.train()
        for start in range(0, y.size, batch_size):
            batch_indices = torch.as_tensor(order[start : start + batch_size], dtype=torch.long, device=device)
            logits = model(x_tensor[batch_indices])
            loss = criterion(logits, y_tensor[batch_indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                probabilities = torch.sigmoid(model(x_tensor)).detach().cpu().numpy().reshape(-1)
            metrics = _binary_metrics(probabilities, y, threshold=0.5)
            metrics.update({"epoch": int(epoch), "phase": "train_mlp"})
            history.append(metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

    first = model[0]
    second = model[2]
    return (
        {
            "hidden_weights": first.weight.detach().cpu().numpy().T.astype(np.float32),
            "hidden_bias": first.bias.detach().cpu().numpy().astype(np.float32),
            "output_weights": second.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
            "output_bias": float(second.bias.detach().cpu().numpy().reshape(-1)[0]),
        },
        history,
    )


def _binary_metrics(probabilities: np.ndarray, targets: np.ndarray, *, threshold: float) -> dict[str, Any]:
    pred = probabilities >= float(threshold)
    truth = targets > 0.5
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    recall = None if tp + fn == 0 else float(tp / (tp + fn))
    return {
        "samples": int(targets.size),
        "positives": int(truth.sum()),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "positive_rate": float(truth.mean()) if truth.size else None,
    }


def _tune_threshold(probabilities: np.ndarray, targets: np.ndarray, config: ExperimentConfig) -> tuple[float, dict[str, Any]]:
    if probabilities.size == 0:
        threshold = _raw_float(config, "override_default_threshold", 0.9)
        return threshold, {"threshold": threshold, "reason": "empty_validation"}
    min_precision = _raw_float(config, "override_min_precision", 0.75)
    fp_penalty = _raw_float(config, "override_false_positive_penalty", 2.0)
    candidates = np.linspace(0.50, 0.99, 50)
    best_score = -math.inf
    best_threshold = _raw_float(config, "override_default_threshold", 0.9)
    best_metrics: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = _binary_metrics(probabilities, targets, threshold=float(threshold))
        precision = metrics["precision"]
        if precision is not None and precision < min_precision:
            continue
        score = float(metrics["tp"] - fp_penalty * metrics["fp"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {**metrics, "utility": score}
    if best_metrics is None:
        best_metrics = _binary_metrics(probabilities, targets, threshold=best_threshold)
        best_metrics["reason"] = "min_precision_not_met"
    return float(best_threshold), best_metrics


def _threshold_sweep(probabilities: np.ndarray, targets: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
        output.append(_binary_metrics(probabilities, targets, threshold=float(threshold)))
    return output


def _predict(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    standardized = (x - mean) / np.maximum(scale, 1e-6)
    return _sigmoid(standardized @ weights + float(bias)).astype(np.float32)


def _predict_mlp(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, params: dict[str, np.ndarray | float]) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    standardized = (x - mean) / np.maximum(scale, 1e-6)
    hidden = np.maximum(standardized @ np.asarray(params["hidden_weights"], dtype=np.float32) + np.asarray(params["hidden_bias"], dtype=np.float32), 0.0)
    logits = hidden @ np.asarray(params["output_weights"], dtype=np.float32).reshape(-1) + float(params["output_bias"])
    return _sigmoid(logits).astype(np.float32)


def run_train_lookahead_override(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_lookahead_override requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)
    accepted_min = _raw_int(config, "override_positive_accepted_delta", 1)
    reward_min = _raw_float(config, "override_positive_reward_delta", 0.0)
    solver = GnnCnnDqnOngSolver(_solver_config(config))

    split_data: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = _label_path(config, split)
        if path is None:
            continue
        labels = _load_label_table(path, accepted_min=accepted_min, reward_min=reward_min)
        x, y, metadata = _build_split_examples(config=config, split=split, labels=labels, solver=solver, run_path=run_path)
        metadata.to_csv(run_path / f"{split}_override_examples.csv", index=False)
        split_data[split] = {
            "path": str(path),
            "labels": labels,
            "x": x,
            "y": y,
            "metadata": metadata,
        }

    if "train" not in split_data or split_data["train"]["x"].shape[0] == 0:
        raise ValueError("No train override examples were generated")

    x_train = split_data["train"]["x"]
    y_train = split_data["train"]["y"]
    mean, scale = _standardize(x_train)
    model_type = str(config.resolved.get("override_model_type", config.raw.get("override_model_type", "logistic"))).strip().lower()
    mlp_params: dict[str, np.ndarray | float] | None = None
    if model_type == "mlp":
        mlp_params, history = _train_mlp(x_train=x_train, y_train=y_train, mean=mean, scale=scale, config=config)
        weights = np.zeros((x_train.shape[1],), dtype=np.float32)
        bias = 0.0
    else:
        weights, bias, history = _train_logistic(x_train=x_train, y_train=y_train, mean=mean, scale=scale, config=config)

    val_split = split_data.get("val", split_data["train"])
    val_probs = (
        _predict_mlp(val_split["x"], mean, scale, mlp_params)
        if mlp_params is not None
        else _predict(val_split["x"], mean, scale, weights, bias)
    )
    threshold, threshold_metrics = _tune_threshold(val_probs, val_split["y"], config)

    classifier = {
        "model_type": "mlp_override_classifier" if mlp_params is not None else "logistic_override_classifier",
        "feature_names": OVERRIDE_FEATURE_NAMES,
        "mean": mean.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "threshold": float(threshold),
        "top_k": _raw_int(config, "override_top_k", _raw_int(config, "lookahead_top_k", 4)),
        "include_j_total": _raw_bool(config, "override_include_j_total", True),
    }
    if mlp_params is not None:
        classifier.update(
            {
                "hidden_weights": np.asarray(mlp_params["hidden_weights"], dtype=np.float32).astype(float).tolist(),
                "hidden_bias": np.asarray(mlp_params["hidden_bias"], dtype=np.float32).astype(float).tolist(),
                "output_weights": np.asarray(mlp_params["output_weights"], dtype=np.float32).astype(float).tolist(),
                "output_bias": float(mlp_params["output_bias"]),
            }
        )
    classifier_path = run_path / "override_classifier.json"
    _write_json(classifier_path, classifier)

    split_metrics: dict[str, Any] = {}
    for split, data in split_data.items():
        probabilities = (
            _predict_mlp(data["x"], mean, scale, mlp_params)
            if mlp_params is not None
            else _predict(data["x"], mean, scale, weights, bias)
        )
        scored = data["metadata"].copy()
        scored["probability"] = probabilities
        scored.to_csv(run_path / f"{split}_override_examples_scored.csv", index=False)
        np.savez_compressed(
            run_path / f"{split}_override_examples.npz",
            features=data["x"].astype(np.float32),
            targets=data["y"].astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            feature_names=np.asarray(OVERRIDE_FEATURE_NAMES, dtype=object),
        )
        split_metrics[split] = {
            "label_path": data["path"],
            "candidate_examples": int(data["x"].shape[0]),
            "positive_candidate_examples": int(data["y"].sum()),
            "positive_candidate_rate": float(data["y"].mean()) if data["y"].size else None,
            "positive_states": int(data["metadata"].drop_duplicates(["episode_id", "request_id"])["positive_state"].sum())
            if not data["metadata"].empty
            else 0,
            "threshold_metrics": _binary_metrics(probabilities, data["y"], threshold=threshold),
            "threshold_sweep": _threshold_sweep(probabilities, data["y"]),
        }

    metrics = {
        "stage": "train_lookahead_override",
        "dataset_path": str(config.dataset_path),
        "ong_source_path": ong_source,
        "classifier_path": str(classifier_path),
        "feature_names": OVERRIDE_FEATURE_NAMES,
        "model_type": classifier["model_type"],
        "positive_rule": {
            "accepted_delta_min": int(accepted_min),
            "reward_delta_min": float(reward_min),
        },
        "threshold": float(threshold),
        "threshold_tuning": threshold_metrics,
        "history": history,
        "splits": split_metrics,
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
