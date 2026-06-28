from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BUCKETS = (
    "bursty:high",
    "bursty:low",
    "bursty:medium",
    "bursty:overload",
    "hotspot:high",
    "hotspot:low",
    "hotspot:medium",
    "hotspot:overload",
    "nonuniform:high",
    "nonuniform:low",
    "nonuniform:medium",
    "nonuniform:overload",
    "uniform:high",
    "uniform:low",
    "uniform:medium",
    "uniform:overload",
)


def parse_bucket_set(text: str) -> set[str]:
    buckets: set[str] = set()
    for raw_item in str(text or "").split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bucket must be scenario:load, got {item!r}")
        buckets.add(item)
    return buckets


def _valid_scores(scores: np.ndarray, candidate_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(candidate_mask, dtype=bool)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return valid, np.zeros(0, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32)[valid]
    return valid.astype(np.int64), values.astype(np.float32)


def selected_rank_norm(scores: np.ndarray, candidate_mask: np.ndarray, selected_index: int) -> float:
    valid, values = _valid_scores(scores, candidate_mask)
    if valid.size <= 1:
        return 0.0
    order = np.argsort(-values, kind="mergesort")
    ranked_valid = valid[order]
    matches = np.flatnonzero(ranked_valid == int(selected_index))
    if matches.size == 0:
        return 1.0
    return float(matches[0]) / float(max(valid.size - 1, 1))


def top1_gap(scores: np.ndarray, candidate_mask: np.ndarray) -> float:
    _valid, values = _valid_scores(scores, candidate_mask)
    if values.size < 2:
        return 0.0
    top2 = np.partition(values, -2)[-2:]
    return float(np.max(top2) - np.min(top2))


def _take_candidate(values: np.ndarray, index: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2 or int(index) < 0 or int(index) >= int(arr.shape[0]):
        width = int(arr.shape[-1]) if arr.ndim >= 1 else 0
        return np.zeros(width, dtype=np.float32)
    return np.nan_to_num(arr[int(index)].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def live_risk_feature_names(
    *,
    action_dim: int,
    route_basic_dim: int,
    global_dim: int,
    request_dim: int,
    bucket_vocab: list[str] | tuple[str, ...] = DEFAULT_BUCKETS,
) -> list[str]:
    names = [
        "selected_index_norm",
        "base_index_norm",
        "valid_candidates_norm",
        "selected_score",
        "base_score",
        "score_margin_vs_base",
        "selected_rank_norm",
        "base_rank_norm",
        "top1_gap",
    ]
    names.extend(f"bucket={bucket}" for bucket in bucket_vocab)
    names.extend(f"selected_action_{index}" for index in range(int(action_dim)))
    names.extend(f"base_action_{index}" for index in range(int(action_dim)))
    names.extend(f"delta_action_{index}" for index in range(int(action_dim)))
    names.extend(f"selected_route_basic_{index}" for index in range(int(route_basic_dim)))
    names.extend(f"base_route_basic_{index}" for index in range(int(route_basic_dim)))
    names.extend(f"delta_route_basic_{index}" for index in range(int(route_basic_dim)))
    names.extend(f"global_{index}" for index in range(int(global_dim)))
    names.extend(f"request_{index}" for index in range(int(request_dim)))
    return names


def live_risk_feature_vector(
    *,
    arrays: dict[str, np.ndarray],
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    selected_index: int,
    base_index: int,
    bucket: str,
    bucket_vocab: list[str] | tuple[str, ...] = DEFAULT_BUCKETS,
) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=bool)
    n_max = max(int(mask.shape[0]), 1)
    valid_count = int(mask.sum())
    selected_index = int(selected_index)
    base_index = int(base_index)
    if base_index < 0 or base_index >= n_max or not bool(mask[base_index]):
        valid = np.flatnonzero(mask)
        base_index = int(valid[0]) if valid.size else 0

    score_values = np.asarray(scores, dtype=np.float32).reshape(-1)
    selected_score = float(score_values[selected_index]) if 0 <= selected_index < score_values.size else 0.0
    base_score = float(score_values[base_index]) if 0 <= base_index < score_values.size else 0.0
    selected_action = _take_candidate(np.asarray(arrays["action_features"], dtype=np.float32), selected_index)
    base_action = _take_candidate(np.asarray(arrays["action_features"], dtype=np.float32), base_index)
    route_basic = np.asarray(arrays.get("route_basic_features", np.zeros((n_max, 0), dtype=np.float32)), dtype=np.float32)
    selected_route = _take_candidate(route_basic, selected_index)
    base_route = _take_candidate(route_basic, base_index)
    global_features = np.nan_to_num(np.asarray(arrays["global_features"], dtype=np.float32).reshape(-1), nan=0.0)
    request_features = np.nan_to_num(np.asarray(arrays["request_features"], dtype=np.float32).reshape(-1), nan=0.0)

    values: list[float] = [
        float(selected_index) / float(max(n_max - 1, 1)),
        float(base_index) / float(max(n_max - 1, 1)),
        float(valid_count) / float(n_max),
        selected_score,
        base_score,
        float(selected_score - base_score),
        selected_rank_norm(score_values, mask, selected_index),
        selected_rank_norm(score_values, mask, base_index),
        top1_gap(score_values, mask),
    ]
    values.extend(1.0 if str(bucket) == str(item) else 0.0 for item in bucket_vocab)
    values.extend(float(x) for x in selected_action)
    values.extend(float(x) for x in base_action)
    values.extend(float(x) for x in (selected_action - base_action))
    values.extend(float(x) for x in selected_route)
    values.extend(float(x) for x in base_route)
    values.extend(float(x) for x in (selected_route - base_route))
    values.extend(float(x) for x in global_features)
    values.extend(float(x) for x in request_features)
    return np.asarray(values, dtype=np.float32)


def artifact_model_path(artifact_path: str | Path, model_path: str) -> Path:
    path = Path(str(model_path))
    if path.is_absolute():
        return path
    return Path(artifact_path).resolve().parent / path


def load_live_risk_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_path = artifact_model_path(artifact_path, str(data["model_path"]))
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    data["artifact_path"] = str(artifact_path)
    data["resolved_model_path"] = str(model_path)
    data["booster"] = booster
    return data


def predict_live_risk(artifact: dict[str, Any], features: np.ndarray) -> float:
    import xgboost as xgb

    row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    matrix = xgb.DMatrix(row, feature_names=list(artifact.get("feature_names", [])))
    booster = artifact["booster"]
    return float(booster.predict(matrix)[0])
