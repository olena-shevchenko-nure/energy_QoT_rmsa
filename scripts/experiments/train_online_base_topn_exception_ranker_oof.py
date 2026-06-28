from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_online_base_topn_exception_ranker as base
from cse2026.experiments.eon.lookahead_override_features import OVERRIDE_FEATURE_NAMES
from cse2026.experiments.eon.tree_ranker_runtime import ADVANTAGE_FEATURE_NAMES


def _json_safe(value: Any) -> Any:
    return base._json_safe(value)


def _write_json(path: Path, data: Any) -> None:
    base._write_json(path, data)


def _safe_no_override_thresholds(thresholds: dict[str, float] | None = None) -> dict[str, float]:
    result = {
        "delta_weight": 1.0,
        "win_weight": 1.0,
        "loss_weight": 2.0,
        "ranker_margin_weight": 0.0,
        "min_win_prob": 2.0,
        "max_loss_prob": -1.0,
        "min_delta_pred": 1.0e9,
        "fallback_no_override": 1.0,
        "tune_found_feasible": 0.0,
    }
    if thresholds:
        for key in ("delta_weight", "win_weight", "loss_weight", "ranker_margin_weight"):
            if key in thresholds:
                result[key] = float(thresholds[key])
    return result


def _group_bucket(metadata: pd.DataFrame, group_id: int) -> str:
    group = metadata[metadata["group_id"].astype(int) == int(group_id)]
    non_base = group[~group["is_base"].astype(bool)]
    max_delta = float(non_base["accepted_delta_vs_base"].max()) if not non_base.empty else 0.0
    min_delta = float(non_base["accepted_delta_vs_base"].min()) if not non_base.empty else 0.0
    outcome = "win_available" if max_delta > 0.0 else ("loss_only" if min_delta < 0.0 else "tie_only")
    scenario = str(group["traffic_scenario"].iloc[0]) if "traffic_scenario" in group and not group.empty else ""
    load_name = str(group["load_name"].iloc[0]) if "load_name" in group and not group.empty else ""
    return f"{scenario}|{load_name}|{outcome}"


def _make_group_folds(metadata: pd.DataFrame, *, folds: int, seed: int) -> list[set[int]]:
    groups = sorted(int(value) for value in metadata["group_id"].drop_duplicates().tolist())
    if not groups:
        raise ValueError("No groups available for OOF folds")
    fold_count = max(2, min(int(folds), len(groups)))
    rng = np.random.default_rng(int(seed))
    buckets: dict[str, list[int]] = {}
    for group_id in groups:
        buckets.setdefault(_group_bucket(metadata, group_id), []).append(int(group_id))
    fold_sets: list[set[int]] = [set() for _ in range(fold_count)]
    for bucket_groups in buckets.values():
        values = np.asarray(bucket_groups, dtype=np.int64)
        rng.shuffle(values)
        for offset, group_id in enumerate(values.tolist()):
            fold_sets[int(offset % fold_count)].add(int(group_id))
    return fold_sets


def _row_indices_for_groups(metadata: pd.DataFrame, groups: set[int]) -> np.ndarray:
    mask = metadata["group_id"].astype(int).isin(set(int(value) for value in groups)).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def _train_fold(
    *,
    fold_id: int,
    output_dir: Path,
    metadata: pd.DataFrame,
    x: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_boost_round: int,
    seed: int,
) -> dict[str, Any]:
    fold_dir = output_dir / "fold_models" / f"fold_{int(fold_id):02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_meta = metadata.iloc[train_idx].reset_index(drop=True)
    val_meta = metadata.iloc[val_idx].reset_index(drop=True)
    ranker = base._train_xgboost_regressor(
        x=x[train_idx],
        y=target[train_idx],
        weights=base._row_weights(train_meta, base_weight=0.40, win_weight=8.0, loss_weight=6.0, tie_weight=0.50),
        feature_names=list(OVERRIDE_FEATURE_NAMES),
        model_path=fold_dir / "xgboost_online_base_topn_ranker.json",
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 1000 + int(fold_id),
    )
    train_scores = base._predict_xgboost(ranker, x[train_idx], list(OVERRIDE_FEATURE_NAMES))
    val_scores = base._predict_xgboost(ranker, x[val_idx], list(OVERRIDE_FEATURE_NAMES))
    train_adv = base._build_advantage_dataset(train_meta, x[train_idx], train_scores, target[train_idx])
    val_adv = base._build_advantage_dataset(val_meta, x[val_idx], val_scores, target[val_idx])
    if train_adv["x"].shape[0] == 0 or val_adv["x"].shape[0] == 0:
        raise ValueError(f"Fold {fold_id} produced empty advantage data")

    win_model = base._train_xgboost_binary(
        x=train_adv["x"],
        y=train_adv["win_y"],
        weights=base._adv_weights(train_adv["metadata"], "win"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=fold_dir / "xgboost_online_base_topn_win.json",
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 2000 + int(fold_id),
    )
    loss_model = base._train_xgboost_binary(
        x=train_adv["x"],
        y=train_adv["loss_y"],
        weights=base._adv_weights(train_adv["metadata"], "loss"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=fold_dir / "xgboost_online_base_topn_loss.json",
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 3000 + int(fold_id),
    )
    delta_model = base._train_xgboost_regressor(
        x=train_adv["x"],
        y=train_adv["delta_y"],
        weights=base._adv_weights(train_adv["metadata"], "delta"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=fold_dir / "xgboost_online_base_topn_delta.json",
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 4000 + int(fold_id),
    )
    val_pred = {
        "win": base._predict_xgboost(win_model, val_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "loss": base._predict_xgboost(loss_model, val_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "delta": base._predict_xgboost(delta_model, val_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
    }
    return {
        "metadata": val_adv["metadata"].reset_index(drop=True),
        "x": val_adv["x"].astype(np.float32),
        "win_y": val_adv["win_y"].astype(np.float32),
        "loss_y": val_adv["loss_y"].astype(np.float32),
        "delta_y": val_adv["delta_y"].astype(np.float32),
        "pred": val_pred,
        "summary": {
            "fold": int(fold_id),
            "train_groups": int(train_meta["group_id"].nunique()),
            "train_rows": int(len(train_meta)),
            "train_adv_rows": int(len(train_adv["metadata"])),
            "val_groups": int(val_meta["group_id"].nunique()),
            "val_rows": int(len(val_meta)),
            "val_adv_rows": int(len(val_adv["metadata"])),
            "val_wins": int((val_adv["metadata"]["accepted_delta_vs_base"].astype(float) > 0.0).sum()),
            "val_losses": int((val_adv["metadata"]["accepted_delta_vs_base"].astype(float) < 0.0).sum()),
            "val_total_delta": int(round(float(val_adv["metadata"]["accepted_delta_vs_base"].astype(float).sum()))),
        },
    }


def _concat_fold_outputs(fold_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = pd.concat([item["metadata"] for item in fold_outputs], ignore_index=True)
    x = np.concatenate([item["x"] for item in fold_outputs], axis=0).astype(np.float32)
    win_y = np.concatenate([item["win_y"] for item in fold_outputs], axis=0).astype(np.float32)
    loss_y = np.concatenate([item["loss_y"] for item in fold_outputs], axis=0).astype(np.float32)
    delta_y = np.concatenate([item["delta_y"] for item in fold_outputs], axis=0).astype(np.float32)
    pred = {
        key: np.concatenate([item["pred"][key] for item in fold_outputs], axis=0).astype(np.float32)
        for key in ("win", "loss", "delta")
    }
    order = np.lexsort(
        (
            metadata["candidate_index"].astype(int).to_numpy(),
            metadata["group_id"].astype(int).to_numpy(),
        )
    )
    metadata = metadata.iloc[order].reset_index(drop=True)
    return {
        "metadata": metadata,
        "x": x[order],
        "win_y": win_y[order],
        "loss_y": loss_y[order],
        "delta_y": delta_y[order],
        "pred": {key: value[order] for key, value in pred.items()},
    }


def train_oof_exception_ranker(
    *,
    input_dir: Path,
    output_dir: Path,
    candidate_pool: str,
    top_k: int,
    folds: int,
    secondary_scale: float,
    num_boost_round: int,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    seed: int,
) -> dict[str, Any]:
    data = base._load_dataset(input_dir)
    metadata = data["metadata"].reset_index(drop=True)
    x = data["x"].astype(np.float32)
    raw_target = data["target"].astype(np.float32)
    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    secondary_delta = metadata["secondary_delta_vs_base"].to_numpy(dtype=np.float32)
    target = np.where(np.abs(accepted_delta) > 0.0, accepted_delta, float(secondary_scale) * secondary_delta).astype(np.float32)
    if metadata.empty:
        raise ValueError("No online base Top-N examples found")

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_groups = _make_group_folds(metadata, folds=int(folds), seed=int(seed))
    all_groups = set(int(value) for value in metadata["group_id"].drop_duplicates().tolist())
    fold_outputs: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_id, val_groups in enumerate(fold_groups):
        train_groups = all_groups.difference(val_groups)
        train_idx = _row_indices_for_groups(metadata, train_groups)
        val_idx = _row_indices_for_groups(metadata, val_groups)
        output = _train_fold(
            fold_id=int(fold_id),
            output_dir=output_dir,
            metadata=metadata,
            x=x,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            num_boost_round=int(num_boost_round),
            seed=int(seed),
        )
        fold_outputs.append(output)
        fold_summaries.append(output["summary"])

    oof_adv = _concat_fold_outputs(fold_outputs)
    thresholds, oof_metrics = base._tune_thresholds(
        oof_adv["metadata"],
        win_prob=oof_adv["pred"]["win"],
        loss_prob=oof_adv["pred"]["loss"],
        delta_pred=oof_adv["pred"]["delta"],
        max_loss_rate=float(max_loss_rate),
        min_override_count=int(min_override_count),
        min_total_delta=float(min_total_delta),
        max_override_rate=float(max_override_rate),
    )
    if not bool(oof_metrics.get("constraints_satisfied", False)):
        thresholds = _safe_no_override_thresholds(thresholds)
        oof_metrics = base._selection_metrics(
            oof_adv["metadata"],
            win_prob=oof_adv["pred"]["win"],
            loss_prob=oof_adv["pred"]["loss"],
            delta_pred=oof_adv["pred"]["delta"],
            thresholds=thresholds,
        )
        oof_metrics["constraints_satisfied"] = False

    oof_predictions = oof_adv["metadata"].copy()
    oof_predictions["oof_win_prob"] = oof_adv["pred"]["win"].astype(np.float32)
    oof_predictions["oof_loss_prob"] = oof_adv["pred"]["loss"].astype(np.float32)
    oof_predictions["oof_delta_pred"] = oof_adv["pred"]["delta"].astype(np.float32)
    oof_predictions.to_csv(output_dir / "online_base_topn_exception_oof_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "online_base_topn_exception_oof_predictions.npz",
        x=oof_adv["x"].astype(np.float32),
        win_y=oof_adv["win_y"].astype(np.float32),
        loss_y=oof_adv["loss_y"].astype(np.float32),
        delta_y=oof_adv["delta_y"].astype(np.float32),
        oof_win=oof_adv["pred"]["win"].astype(np.float32),
        oof_loss=oof_adv["pred"]["loss"].astype(np.float32),
        oof_delta=oof_adv["pred"]["delta"].astype(np.float32),
        feature_names=np.asarray(ADVANTAGE_FEATURE_NAMES, dtype=object),
    )

    ranker_model_path = output_dir / "xgboost_online_base_topn_ranker.json"
    final_ranker = base._train_xgboost_regressor(
        x=x,
        y=target,
        weights=base._row_weights(metadata, base_weight=0.40, win_weight=8.0, loss_weight=6.0, tie_weight=0.50),
        feature_names=list(OVERRIDE_FEATURE_NAMES),
        model_path=ranker_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )
    final_scores = base._predict_xgboost(final_ranker, x, list(OVERRIDE_FEATURE_NAMES))
    full_adv = base._build_advantage_dataset(metadata, x, final_scores, target)
    win_model_path = output_dir / "xgboost_online_base_topn_win.json"
    loss_model_path = output_dir / "xgboost_online_base_topn_loss.json"
    delta_model_path = output_dir / "xgboost_online_base_topn_delta.json"
    final_win = base._train_xgboost_binary(
        x=full_adv["x"],
        y=full_adv["win_y"],
        weights=base._adv_weights(full_adv["metadata"], "win"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=win_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 11,
    )
    final_loss = base._train_xgboost_binary(
        x=full_adv["x"],
        y=full_adv["loss_y"],
        weights=base._adv_weights(full_adv["metadata"], "loss"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=loss_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 12,
    )
    final_delta = base._train_xgboost_regressor(
        x=full_adv["x"],
        y=full_adv["delta_y"],
        weights=base._adv_weights(full_adv["metadata"], "delta"),
        feature_names=list(ADVANTAGE_FEATURE_NAMES),
        model_path=delta_model_path,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 13,
    )
    final_pred = {
        "win": base._predict_xgboost(final_win, full_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "loss": base._predict_xgboost(final_loss, full_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
        "delta": base._predict_xgboost(final_delta, full_adv["x"], list(ADVANTAGE_FEATURE_NAMES)),
    }
    final_train_metrics = base._selection_metrics(
        full_adv["metadata"],
        win_prob=final_pred["win"],
        loss_prob=final_pred["loss"],
        delta_pred=final_pred["delta"],
        thresholds=thresholds,
    )
    full_adv["metadata"].to_csv(output_dir / "online_base_topn_exception_full_advantage_examples.csv", index=False)
    np.savez_compressed(
        output_dir / "online_base_topn_exception_full_advantage.npz",
        x=full_adv["x"].astype(np.float32),
        win_y=full_adv["win_y"].astype(np.float32),
        loss_y=full_adv["loss_y"].astype(np.float32),
        delta_y=full_adv["delta_y"].astype(np.float32),
        feature_names=np.asarray(ADVANTAGE_FEATURE_NAMES, dtype=object),
    )

    training_summary = {
        "input_dir": str(input_dir),
        "candidate_pool": str(candidate_pool),
        "candidate_pool_top_k": int(top_k),
        "folds": int(len(fold_groups)),
        "secondary_scale": float(secondary_scale),
        "num_boost_round": int(num_boost_round),
        "max_loss_rate": float(max_loss_rate),
        "min_override_count": int(min_override_count),
        "min_total_delta": float(min_total_delta),
        "max_override_rate": float(max_override_rate),
        "seed": int(seed),
        "target": {
            "mean": float(np.mean(target)),
            "std": float(np.std(target)),
            "positive_rows": int(np.sum(target > 1.0e-6)),
            "negative_rows": int(np.sum(target < -1.0e-6)),
            "zero_rows": int(np.sum(np.abs(target) <= 1.0e-6)),
            "raw_target_mean": float(np.mean(raw_target)),
        },
    }
    artifact_path = base._export_artifact(
        output_dir=output_dir,
        ranker_model_path=ranker_model_path,
        win_model_path=win_model_path,
        loss_model_path=loss_model_path,
        delta_model_path=delta_model_path,
        thresholds=thresholds,
        candidate_pool=str(candidate_pool),
        top_k=int(top_k),
        training_summary=training_summary,
    )
    summary = {
        "artifact_path": str(artifact_path),
        "ranker_model_path": str(ranker_model_path),
        "win_model_path": str(win_model_path),
        "loss_model_path": str(loss_model_path),
        "delta_model_path": str(delta_model_path),
        "thresholds": dict(thresholds),
        "training": training_summary,
        "label_counts": {
            "groups": int(metadata["group_id"].nunique()),
            "rows": int(len(metadata)),
            "non_base_rows": int((~metadata["is_base"].astype(bool)).sum()),
            "win_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) > 0).sum()),
            "loss_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) < 0).sum()),
            "tie_rows": int((metadata.loc[~metadata["is_base"].astype(bool), "accepted_delta_vs_base"].astype(float) == 0).sum()),
        },
        "folds": fold_summaries,
        "gate": {
            "oof": oof_metrics,
            "final_train": final_train_metrics,
        },
    }
    _write_json(output_dir / "online_base_topn_exception_oof_summary.json", summary)
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOF-calibrated online base Top-N exception ranker.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-pool", default="all")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--secondary-scale", type=float, default=0.25)
    parser.add_argument("--num-boost-round", type=int, default=60)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260607)
    args = parser.parse_args()
    train_oof_exception_ranker(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        candidate_pool=str(args.candidate_pool),
        top_k=int(args.top_k),
        folds=int(args.folds),
        secondary_scale=float(args.secondary_scale),
        num_boost_round=int(args.num_boost_round),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
