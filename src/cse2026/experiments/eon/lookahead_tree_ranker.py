from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cse2026.data_generation.io_utils import project_root
from cse2026.ong_solver import GnnCnnDqnOngSolver

from ..config import ExperimentConfig
from .lookahead_oracle import _select_rollout_index, _solver_config
from .lookahead_override_features import (
    OVERRIDE_FEATURE_NAMES,
    candidate_feature_matrix,
    candidate_indices_for_topn,
    select_q_head_index,
)
from .tree_ranker_runtime import select_tree_base_index
from .ong_rollout import (
    _add_ong_source_path,
    _make_env,
    _raw_bool,
    _raw_float,
    _raw_int,
    _traffic_jsonl_for_episode,
)


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


def _load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    return labels[labels["oracle_index"].astype(int) >= 0].copy()


def _label_accepted_delta_vs_base(label: Any) -> int:
    return int(getattr(label, "oracle_accepted_delta_vs_base", getattr(label, "oracle_accepted_delta_vs_q_head", 0)))


def _label_reward_delta_vs_base(label: Any) -> float:
    return float(getattr(label, "oracle_reward_delta_vs_base", getattr(label, "oracle_reward_delta_vs_q_head", 0.0)))


def _oracle_relevance(label: Any, mode: str) -> float:
    accepted_delta = _label_accepted_delta_vs_base(label)
    reward_delta = _label_reward_delta_vs_base(label)
    if mode == "accepted_grade":
        return 2.0 if accepted_delta > 0 else 1.0
    if mode == "hybrid_grade":
        return float(1.0 + max(accepted_delta, 0) + min(max(reward_delta, 0.0), 1.0))
    return 1.0


def _build_split_examples(
    *,
    config: ExperimentConfig,
    split: str,
    labels: pd.DataFrame,
    solver: GnnCnnDqnOngSolver,
    run_path: Path,
) -> dict[str, Any]:
    traffic = pd.read_parquet(config.dataset_path / "traffic" / f"{split}.parquet")
    label_by_key = {
        (str(row.episode_id), int(row.request_id)): row
        for row in labels.itertuples(index=False)
    }
    episode_ids = tuple(str(value) for value in labels["episode_id"].drop_duplicates().tolist())
    base_policy = str(config.resolved.get("tree_ranker_base_policy", config.raw.get("tree_ranker_base_policy", "energy-aware-ksp-bm-ff")))
    state_policy = str(config.resolved.get("lookahead_state_policy", config.raw.get("lookahead_state_policy", base_policy)))
    relevance_mode = str(config.resolved.get("tree_ranker_relevance_mode", config.raw.get("tree_ranker_relevance_mode", "accepted_grade")))

    rows: list[np.ndarray] = []
    targets: list[float] = []
    group_sizes: list[int] = []
    metadata: list[dict[str, Any]] = []
    skipped_groups = 0
    group_id = 0

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
                candidate_indices = candidate_indices_for_topn(batch)
                base_index = select_tree_base_index(batch, solver.config.n_max, base_policy)
                features, kept_indices = candidate_feature_matrix(
                    batch=batch,
                    candidate_indices=candidate_indices,
                    n_max=solver.config.n_max,
                    reference_index=base_index,
                )
                oracle_index = int(label.oracle_index)
                if features.shape[0] >= 2 and oracle_index in kept_indices:
                    relevance = _oracle_relevance(label, relevance_mode)
                    group_sizes.append(int(features.shape[0]))
                    q_head_index = select_q_head_index(batch, solver.config.n_max)
                    for feature_row, candidate_index in zip(features, kept_indices):
                        target = relevance if int(candidate_index) == oracle_index else 0.0
                        rows.append(feature_row)
                        targets.append(float(target))
                        metadata.append(
                            {
                                "split": split,
                                "group_id": int(group_id),
                                "episode_id": episode_id,
                                "request_id": int(request["request_id"]),
                                "candidate_index": int(candidate_index),
                                "base_policy": base_policy,
                                "base_index": int(base_index),
                                "q_head_index": int(q_head_index),
                                "j_total_index": int(getattr(label, "j_total_index", 0)),
                                "oracle_index": oracle_index,
                                "target": float(target),
                                "oracle_accepted_delta_vs_base": _label_accepted_delta_vs_base(label),
                                "oracle_reward_delta_vs_base": _label_reward_delta_vs_base(label),
                            }
                        )
                    group_id += 1
                else:
                    skipped_groups += 1

            state_index = _select_rollout_index(batch, state_policy, solver.config.n_max)
            if state_index < 0:
                action = int(solver.adapter(env).block_action(env))
            else:
                action = int(batch.topn[state_index].action)
            observation, reward, terminated, truncated, info = env.step(action)
            del observation, reward, info
            if bool(terminated) or bool(truncated):
                break

    if rows:
        x = np.asarray(rows, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
    else:
        x = np.zeros((0, len(OVERRIDE_FEATURE_NAMES)), dtype=np.float32)
        y = np.zeros((0,), dtype=np.float32)
    return {
        "x": x,
        "y": y,
        "group_sizes": np.asarray(group_sizes, dtype=np.int32),
        "metadata": pd.DataFrame(metadata),
        "skipped_groups": int(skipped_groups),
    }


def _group_metrics(metadata: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    if metadata.empty:
        return {
            "groups": 0,
            "rows": 0,
            "oracle_top1_accuracy": None,
            "base_oracle_accuracy": None,
            "override_rate_vs_base": None,
            "positive_accepted_capture_rate": None,
        }
    scored = metadata.copy()
    scored["score"] = np.asarray(scores, dtype=np.float32)
    groups = []
    for _, group in scored.groupby("group_id", sort=False):
        best = group.sort_values(["score", "candidate_index"], ascending=[False, True]).iloc[0]
        oracle_index = int(best["oracle_index"])
        base_index = int(best["base_index"]) if "base_index" in best else int(best["q_head_index"])
        selected_index = int(best["candidate_index"])
        accepted_delta = int(best["oracle_accepted_delta_vs_base"]) if "oracle_accepted_delta_vs_base" in best else int(best["oracle_accepted_delta_vs_q_head"])
        groups.append(
            {
                "selected_oracle": selected_index == oracle_index,
                "base_oracle": base_index == oracle_index,
                "selected_differs_base": selected_index != base_index,
                "positive_accepted": accepted_delta > 0,
                "captured_positive_accepted": accepted_delta > 0 and selected_index == oracle_index,
            }
        )
    table = pd.DataFrame(groups)
    positive = table[table["positive_accepted"]]
    return {
        "groups": int(len(table)),
        "rows": int(len(scored)),
        "oracle_top1_accuracy": float(table["selected_oracle"].mean()),
        "base_oracle_accuracy": float(table["base_oracle"].mean()),
        "override_rate_vs_base": float(table["selected_differs_base"].mean()),
        "positive_accepted_groups": int(len(positive)),
        "positive_accepted_capture_rate": None if positive.empty else float(positive["captured_positive_accepted"].mean()),
    }


def _train_xgboost(
    *,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required for tree_ranker_backend=xgboost") from exc

    dtrain = xgb.DMatrix(train["x"], label=train["y"], feature_names=OVERRIDE_FEATURE_NAMES)
    dtrain.set_group(train["group_sizes"].astype(np.uint32))
    evals = [(dtrain, "train")]
    if val is not None and val["x"].shape[0] > 0:
        dval = xgb.DMatrix(val["x"], label=val["y"], feature_names=OVERRIDE_FEATURE_NAMES)
        dval.set_group(val["group_sizes"].astype(np.uint32))
        evals.append((dval, "val"))
    tree_method = str(config.resolved.get("tree_ranker_tree_method", config.raw.get("tree_ranker_tree_method", "hist")))
    device = str(config.resolved.get("tree_ranker_device", config.raw.get("tree_ranker_device", ""))).strip()
    version_parts = tuple(int(part) for part in str(getattr(xgb, "__version__", "0")).split(".")[:2] if part.isdigit())
    if device and device.lower().startswith(("cuda", "gpu")) and version_parts and version_parts < (2, 0):
        tree_method = str(config.resolved.get("tree_ranker_gpu_tree_method", config.raw.get("tree_ranker_gpu_tree_method", "gpu_hist")))

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@1",
        "ndcg_exp_gain": False,
        "tree_method": tree_method,
        "eta": _raw_float(config, "tree_ranker_learning_rate", 0.05),
        "max_depth": _raw_int(config, "tree_ranker_max_depth", 4),
        "min_child_weight": _raw_float(config, "tree_ranker_min_child_weight", 1.0),
        "subsample": _raw_float(config, "tree_ranker_subsample", 0.9),
        "colsample_bytree": _raw_float(config, "tree_ranker_colsample_bytree", 0.9),
        "seed": int(config.seed),
        "nthread": _raw_int(config, "tree_ranker_nthread", 4),
    }
    if device and (not version_parts or version_parts >= (2, 0)):
        params["device"] = device
    predictor = str(config.resolved.get("tree_ranker_predictor", config.raw.get("tree_ranker_predictor", ""))).strip()
    if predictor:
        params["predictor"] = predictor
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=_raw_int(config, "tree_ranker_num_boost_round", 160),
        evals=evals,
        verbose_eval=False,
    )
    booster.save_model(str(model_path))
    return booster


def _train_lightgbm(
    *,
    train: dict[str, Any],
    val: dict[str, Any] | None,
    config: ExperimentConfig,
    model_path: Path,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is required for tree_ranker_backend=lightgbm") from exc

    dtrain = lgb.Dataset(
        train["x"],
        label=train["y"],
        group=train["group_sizes"].astype(np.int32),
        feature_name=OVERRIDE_FEATURE_NAMES,
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    if val is not None and val["x"].shape[0] > 0:
        dval = lgb.Dataset(
            val["x"],
            label=val["y"],
            group=val["group_sizes"].astype(np.int32),
            feature_name=OVERRIDE_FEATURE_NAMES,
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets.append(dval)
        valid_names.append("val")
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1],
        "learning_rate": _raw_float(config, "tree_ranker_learning_rate", 0.05),
        "num_leaves": _raw_int(config, "tree_ranker_num_leaves", 31),
        "max_depth": _raw_int(config, "tree_ranker_max_depth", -1),
        "min_data_in_leaf": _raw_int(config, "tree_ranker_min_data_in_leaf", 10),
        "feature_fraction": _raw_float(config, "tree_ranker_feature_fraction", 0.9),
        "bagging_fraction": _raw_float(config, "tree_ranker_bagging_fraction", 0.9),
        "bagging_freq": _raw_int(config, "tree_ranker_bagging_freq", 1),
        "seed": int(config.seed),
        "num_threads": _raw_int(config, "tree_ranker_nthread", 4),
        "verbosity": -1,
    }
    max_label = int(np.ceil(float(np.max(train["y"])))) if train["y"].size else 0
    if val is not None and val["x"].shape[0] > 0 and val["y"].size:
        max_label = max(max_label, int(np.ceil(float(np.max(val["y"])))))
    params["label_gain"] = list(range(max_label + 1))
    device_type = str(config.resolved.get("tree_ranker_device_type", config.raw.get("tree_ranker_device_type", ""))).strip()
    if device_type:
        params["device_type"] = device_type
    max_bin = config.resolved.get("tree_ranker_max_bin", config.raw.get("tree_ranker_max_bin"))
    if max_bin is not None:
        params["max_bin"] = int(max_bin)
    gpu_platform_id = config.resolved.get("tree_ranker_gpu_platform_id", config.raw.get("tree_ranker_gpu_platform_id"))
    if gpu_platform_id is not None:
        params["gpu_platform_id"] = int(gpu_platform_id)
    gpu_device_id = config.resolved.get("tree_ranker_gpu_device_id", config.raw.get("tree_ranker_gpu_device_id"))
    if gpu_device_id is not None:
        params["gpu_device_id"] = int(gpu_device_id)
    gpu_use_dp = config.resolved.get("tree_ranker_gpu_use_dp", config.raw.get("tree_ranker_gpu_use_dp"))
    if gpu_use_dp is not None:
        params["gpu_use_dp"] = _raw_bool(config, "tree_ranker_gpu_use_dp", False)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=_raw_int(config, "tree_ranker_num_boost_round", 160),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    booster.save_model(str(model_path))
    return booster


def _predict(backend: str, booster: Any, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if backend == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(x, feature_names=OVERRIDE_FEATURE_NAMES)
        return np.asarray(booster.predict(matrix), dtype=np.float32)
    return np.asarray(booster.predict(x), dtype=np.float32)


def run_train_tree_ranker(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("train_tree_ranker requires dataset_path")
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ong_source = _add_ong_source_path(config)
    solver = GnnCnnDqnOngSolver(_solver_config(config))

    split_data: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = _label_path(config, split)
        if path is None:
            continue
        labels = _load_labels(path)
        data = _build_split_examples(config=config, split=split, labels=labels, solver=solver, run_path=run_path)
        data["label_path"] = str(path)
        data["metadata"].to_csv(run_path / f"{split}_tree_ranker_examples.csv", index=False)
        np.savez_compressed(
            run_path / f"{split}_tree_ranker_examples.npz",
            features=data["x"].astype(np.float32),
            targets=data["y"].astype(np.float32),
            group_sizes=data["group_sizes"].astype(np.int32),
            feature_names=np.asarray(OVERRIDE_FEATURE_NAMES, dtype=object),
        )
        split_data[split] = data

    if "train" not in split_data or split_data["train"]["x"].shape[0] == 0:
        raise ValueError("No train tree-ranker examples were generated")

    backend = str(config.resolved.get("tree_ranker_backend", config.raw.get("tree_ranker_backend", "xgboost"))).strip().lower()
    base_policy = str(config.resolved.get("tree_ranker_base_policy", config.raw.get("tree_ranker_base_policy", "energy-aware-ksp-bm-ff")))
    model_suffix = "json" if backend == "xgboost" else "txt"
    model_path = run_path / f"{backend}_tree_ranker.{model_suffix}"
    if backend == "xgboost":
        booster = _train_xgboost(train=split_data["train"], val=split_data.get("val"), config=config, model_path=model_path)
    elif backend == "lightgbm":
        booster = _train_lightgbm(train=split_data["train"], val=split_data.get("val"), config=config, model_path=model_path)
    else:
        raise ValueError(f"Unsupported tree_ranker_backend: {backend}")

    ranker = {
        "backend": backend,
        "model_path": str(model_path),
        "feature_names": OVERRIDE_FEATURE_NAMES,
        "candidate_pool": "all_topn",
        "selection_mode": str(config.resolved.get("tree_ranker_selection_mode", config.raw.get("tree_ranker_selection_mode", "pure"))),
        "residual_beta": _raw_float(config, "tree_ranker_residual_beta", 0.05),
        "selection_margin": _raw_float(config, "tree_ranker_selection_margin", 0.0),
        "base_policy": base_policy,
    }
    ranker_path = run_path / "tree_ranker.json"
    _write_json(ranker_path, ranker)

    split_metrics: dict[str, Any] = {}
    for split, data in split_data.items():
        scores = _predict(backend, booster, data["x"])
        scored = data["metadata"].copy()
        scored["score"] = scores
        scored.to_csv(run_path / f"{split}_tree_ranker_examples_scored.csv", index=False)
        split_metrics[split] = {
            "label_path": data["label_path"],
            "groups": int(data["group_sizes"].size),
            "rows": int(data["x"].shape[0]),
            "skipped_groups": int(data["skipped_groups"]),
            "mean_group_size": float(data["group_sizes"].mean()) if data["group_sizes"].size else None,
            "target_positive_rows": int((data["y"] > 0).sum()),
            "target_positive_rate": float((data["y"] > 0).mean()) if data["y"].size else None,
            "rank_metrics": _group_metrics(data["metadata"], scores),
        }

    metrics = {
        "stage": "train_tree_ranker",
        "dataset_path": str(config.dataset_path),
        "ong_source_path": ong_source,
        "backend": backend,
        "ranker_path": str(ranker_path),
        "model_path": str(model_path),
        "feature_names": OVERRIDE_FEATURE_NAMES,
        "splits": split_metrics,
        "parameters": {
            "candidate_pool": ranker["candidate_pool"],
            "base_policy": ranker["base_policy"],
            "relevance_mode": str(
                config.resolved.get("tree_ranker_relevance_mode", config.raw.get("tree_ranker_relevance_mode", "accepted_grade"))
            ),
            "selection_mode": ranker["selection_mode"],
            "residual_beta": ranker["residual_beta"],
            "selection_margin": ranker["selection_margin"],
        },
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
