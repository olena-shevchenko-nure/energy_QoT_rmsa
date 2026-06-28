from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_dataset(input_dir: Path) -> dict[str, Any]:
    metadata = pd.read_csv(input_dir / "online_override_examples.csv").reset_index(drop=True)
    npz = np.load(input_dir / "online_override_examples.npz", allow_pickle=True)
    features = np.asarray(npz["features"], dtype=np.float32)
    feature_names = [str(value) for value in npz["feature_names"].tolist()]
    if len(metadata) != int(features.shape[0]):
        raise ValueError(f"metadata rows ({len(metadata)}) != feature rows ({features.shape[0]})")
    return {"metadata": metadata, "x": features, "feature_names": feature_names}


def _make_group_split(
    metadata: pd.DataFrame,
    *,
    train_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    groups = metadata[["group_id", "accepted_delta_vs_base"]].copy()
    groups["bucket"] = np.where(
        groups["accepted_delta_vs_base"].astype(float) > 0,
        "win",
        np.where(groups["accepted_delta_vs_base"].astype(float) < 0, "loss", "tie"),
    )
    group_table = groups.drop_duplicates("group_id").reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    split_for_group: dict[int, str] = {}
    train_fraction = float(train_fraction)
    calibration_fraction = float(calibration_fraction)
    for _, bucket in group_table.groupby("bucket", sort=False):
        values = bucket["group_id"].astype(int).to_numpy(copy=True)
        rng.shuffle(values)
        train_end = int(round(len(values) * train_fraction))
        cal_end = train_end + int(round(len(values) * calibration_fraction))
        for value in values[:train_end]:
            split_for_group[int(value)] = "train"
        for value in values[train_end:cal_end]:
            split_for_group[int(value)] = "calibration"
        for value in values[cal_end:]:
            split_for_group[int(value)] = "eval"
    split = metadata["group_id"].astype(int).map(split_for_group).fillna("eval").to_numpy()
    return {
        "train": np.flatnonzero(split == "train").astype(np.int64),
        "calibration": np.flatnonzero(split == "calibration").astype(np.int64),
        "eval": np.flatnonzero(split == "eval").astype(np.int64),
    }


def _pos_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y > 0.5))
    negatives = float(max(int(y.size) - int(positives), 0))
    return float(negatives / max(positives, 1.0))


def _train_xgboost(
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    sample_weight: np.ndarray,
    num_boost_round: int,
    seed: int,
) -> Any:
    import xgboost as xgb

    matrix = xgb.DMatrix(x, label=y, weight=sample_weight, feature_names=feature_names)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "scale_pos_weight": _pos_weight(y),
        "seed": int(seed),
        "verbosity": 0,
    }
    return xgb.train(params, matrix, num_boost_round=int(num_boost_round), verbose_eval=False)


def _predict_xgboost(model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    import xgboost as xgb

    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(model.predict(xgb.DMatrix(x, feature_names=feature_names)), dtype=np.float32)


def _metrics_for_cutoff(metadata: pd.DataFrame, risk_score: np.ndarray, cutoff: float) -> dict[str, Any]:
    accepted_delta = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    reward_delta = metadata["future_env_reward_delta_vs_base"].to_numpy(dtype=np.float32)
    allowed = np.asarray(risk_score, dtype=np.float32) <= float(cutoff)
    selected_delta = np.where(allowed, accepted_delta, 0.0)
    selected_reward = np.where(allowed, reward_delta, 0.0)
    override_count = int(allowed.sum())
    override = metadata.loc[allowed]
    loss_rate = None if override_count == 0 else float((override["accepted_delta_vs_base"].astype(float) < 0).mean())
    win_rate = None if override_count == 0 else float((override["accepted_delta_vs_base"].astype(float) > 0).mean())
    tie_rate = None if override_count == 0 else float((override["accepted_delta_vs_base"].astype(float) == 0).mean())
    return {
        "groups": int(len(metadata)),
        "raw_override_count": int(len(metadata)),
        "override_count": override_count,
        "override_rate": float(override_count / max(len(metadata), 1)),
        "vetoed_override_count": int(len(metadata) - override_count),
        "total_selected_accepted_delta_vs_base": int(round(float(np.sum(selected_delta)))),
        "mean_selected_accepted_delta_vs_base": float(np.mean(selected_delta)) if len(metadata) else 0.0,
        "mean_selected_reward_delta_vs_base": float(np.mean(selected_reward)) if len(metadata) else 0.0,
        "selected_loss_rate_when_overridden": loss_rate,
        "selected_win_rate_when_overridden": win_rate,
        "selected_tie_rate_when_overridden": tie_rate,
        "score_cutoff": float(cutoff),
    }


def _raw_metrics(metadata: pd.DataFrame) -> dict[str, Any]:
    return _metrics_for_cutoff(
        metadata,
        np.zeros((len(metadata),), dtype=np.float32),
        cutoff=1.0,
    )


def _tune_cutoff(
    metadata: pd.DataFrame,
    risk_score: np.ndarray,
    *,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
) -> tuple[float, dict[str, Any]]:
    finite = np.asarray([float(value) for value in risk_score if math.isfinite(float(value))], dtype=np.float32)
    if finite.size == 0:
        metrics = _metrics_for_cutoff(metadata, risk_score, cutoff=-math.inf)
        metrics["constraints_satisfied"] = False
        return -math.inf, metrics
    cutoffs = sorted(set(float(value) for value in finite.tolist()))
    cutoffs = [float(np.min(finite) - 1.0e-6)] + cutoffs + [float(np.max(finite) + 1.0e-6)]
    best_cutoff: float | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_cutoff: float | None = None
    fallback_metrics: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    for cutoff in cutoffs:
        metrics = _metrics_for_cutoff(metadata, risk_score, cutoff)
        loss_rate = metrics["selected_loss_rate_when_overridden"]
        loss_value = float(loss_rate if loss_rate is not None else 0.0)
        total_delta = float(metrics["total_selected_accepted_delta_vs_base"])
        key = (
            total_delta,
            float(metrics["mean_selected_reward_delta_vs_base"]),
            -loss_value,
            float(metrics["override_count"]),
        )
        if fallback_key is None or key > fallback_key:
            fallback_key = key
            fallback_cutoff = float(cutoff)
            fallback_metrics = dict(metrics)
        if int(metrics["override_count"]) < int(min_override_count):
            continue
        if total_delta < float(min_total_delta):
            continue
        if loss_rate is not None and float(loss_rate) > float(max_loss_rate):
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_cutoff = float(cutoff)
            best_metrics = dict(metrics)
    if best_cutoff is None or best_metrics is None:
        assert fallback_cutoff is not None and fallback_metrics is not None
        fallback_metrics["constraints_satisfied"] = False
        return float(fallback_cutoff), fallback_metrics
    best_metrics["constraints_satisfied"] = True
    return float(best_cutoff), best_metrics


def _resolve_model_path(meta_path: Path, value: str | None) -> Path:
    if not value:
        raise ValueError(f"{meta_path} does not define model_path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return meta_path.parent / path


def _export_runtime_artifact(
    *,
    source_artifact: Path,
    output_dir: Path,
    risk_model_path: Path,
    feature_names: list[str],
    score_cutoff: float,
) -> Path:
    source_meta = json.loads(source_artifact.read_text(encoding="utf-8"))
    source_model_path = _resolve_model_path(source_artifact, str(source_meta["model_path"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_model = output_dir / source_model_path.name
    copied_risk = output_dir / risk_model_path.name
    if source_model_path.resolve() != copied_model.resolve():
        shutil.copy2(source_model_path, copied_model)
    if risk_model_path.resolve() != copied_risk.resolve():
        shutil.copy2(risk_model_path, copied_risk)
    meta = dict(source_meta)
    meta["model_path"] = copied_model.name
    meta["selection_mode"] = "base_residual"
    meta["risk_selector"] = {
        "enabled": True,
        "backend": "xgboost",
        "model_path": copied_risk.name,
        "feature_kind": "dqn_base_residual_online_override_gate",
        "feature_names": list(feature_names),
        "score_cutoff": float(score_cutoff),
        "label": "online_counterfactual_accepted_delta_vs_base < 0",
    }
    meta["online_override_gate"] = {
        "source_artifact": str(source_artifact),
        "score_cutoff": float(score_cutoff),
    }
    artifact_path = output_dir / "torch_dqn_base_relative_online_override_gate_tree_ranker.json"
    _write_json(artifact_path, meta)
    return artifact_path


def train_gate(
    *,
    input_dir: Path,
    source_artifact: Path,
    output_dir: Path,
    train_fraction: float,
    calibration_fraction: float,
    num_boost_round: int,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    loss_weight: float,
    win_weight: float,
    tie_weight: float,
    seed: int,
) -> dict[str, Any]:
    data = _load_dataset(input_dir)
    metadata = data["metadata"].reset_index(drop=True)
    x = data["x"].astype(np.float32)
    feature_names = list(data["feature_names"])
    if metadata.empty:
        raise ValueError("No online override examples found")
    splits = _make_group_split(
        metadata,
        train_fraction=float(train_fraction),
        calibration_fraction=float(calibration_fraction),
        seed=int(seed),
    )
    train_idx = splits["train"]
    cal_idx = splits["calibration"]
    eval_idx = splits["eval"]
    accepted = metadata["accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    y = (accepted < 0.0).astype(np.float32)
    sample_weight = np.where(
        y > 0.5,
        float(loss_weight),
        np.where(accepted > 0.0, float(win_weight), float(tie_weight)),
    ).astype(np.float32)
    model = _train_xgboost(
        x=x[train_idx],
        y=y[train_idx],
        feature_names=feature_names,
        sample_weight=sample_weight[train_idx],
        num_boost_round=int(num_boost_round),
        seed=int(seed),
    )
    train_score = _predict_xgboost(model, x[train_idx], feature_names)
    cal_score = _predict_xgboost(model, x[cal_idx], feature_names)
    eval_score = _predict_xgboost(model, x[eval_idx], feature_names)
    score_cutoff, cal_metrics = _tune_cutoff(
        metadata.iloc[cal_idx].reset_index(drop=True),
        cal_score,
        max_loss_rate=float(max_loss_rate),
        min_override_count=int(min_override_count),
        min_total_delta=float(min_total_delta),
    )
    train_metrics = _metrics_for_cutoff(metadata.iloc[train_idx].reset_index(drop=True), train_score, score_cutoff)
    eval_metrics = _metrics_for_cutoff(metadata.iloc[eval_idx].reset_index(drop=True), eval_score, score_cutoff)

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_model_path = output_dir / "xgboost_online_override_loss_gate.json"
    model.save_model(str(risk_model_path))
    artifact_path = _export_runtime_artifact(
        source_artifact=source_artifact,
        output_dir=output_dir,
        risk_model_path=risk_model_path,
        feature_names=feature_names,
        score_cutoff=float(score_cutoff),
    )
    summary = {
        "artifact_path": str(artifact_path),
        "risk_model_path": str(risk_model_path),
        "source_artifact": str(source_artifact),
        "input_dir": str(input_dir),
        "score_cutoff": float(score_cutoff),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "label_counts": {
            "examples": int(len(metadata)),
            "wins": int((accepted > 0.0).sum()),
            "losses": int((accepted < 0.0).sum()),
            "ties": int((accepted == 0.0).sum()),
            "raw_total_delta": int(round(float(accepted.sum()))),
            "raw_loss_rate": float((accepted < 0.0).mean()),
            "raw_win_rate": float((accepted > 0.0).mean()),
        },
        "raw": {
            "train": _raw_metrics(metadata.iloc[train_idx].reset_index(drop=True)),
            "calibration": _raw_metrics(metadata.iloc[cal_idx].reset_index(drop=True)),
            "eval": _raw_metrics(metadata.iloc[eval_idx].reset_index(drop=True)),
        },
        "gate": {
            "train": train_metrics,
            "calibration": cal_metrics,
            "eval": eval_metrics,
        },
        "training": {
            "num_boost_round": int(num_boost_round),
            "max_loss_rate": float(max_loss_rate),
            "min_override_count": int(min_override_count),
            "min_total_delta": float(min_total_delta),
            "loss_weight": float(loss_weight),
            "win_weight": float(win_weight),
            "tie_weight": float(tie_weight),
            "seed": int(seed),
        },
    }
    _write_json(output_dir / "online_override_gate_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a veto gate on online override hard-case examples.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--num-boost-round", type=int, default=120)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=5)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--loss-weight", type=float, default=10.0)
    parser.add_argument("--win-weight", type=float, default=3.0)
    parser.add_argument("--tie-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260606)
    args = parser.parse_args()
    summary = train_gate(
        input_dir=Path(args.input_dir),
        source_artifact=Path(args.source_artifact),
        output_dir=Path(args.output_dir),
        train_fraction=float(args.train_fraction),
        calibration_fraction=float(args.calibration_fraction),
        num_boost_round=int(args.num_boost_round),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        loss_weight=float(args.loss_weight),
        win_weight=float(args.win_weight),
        tie_weight=float(args.tie_weight),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
