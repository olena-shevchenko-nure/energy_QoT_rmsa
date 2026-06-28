from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _save_model(backend: str, model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "xgboost":
        model.save_model(str(path))
        return
    if backend == "lightgbm":
        model.save_model(str(path))
        return
    raise ValueError(f"Unsupported backend: {backend}")


def _model_suffix(backend: str) -> str:
    if backend == "xgboost":
        return "json"
    if backend == "lightgbm":
        return "txt"
    raise ValueError(f"Unsupported backend: {backend}")


def _train_win_model(
    quick_module: Any,
    *,
    backend: str,
    x: np.ndarray,
    y: np.ndarray,
    accepted_delta: np.ndarray,
    feature_names: list[str],
    num_boost_round: int,
    seed: int,
    win_sample_weight: float,
    loss_sample_weight: float,
    tie_sample_weight: float,
) -> Any:
    sample_weight = np.where(
        y > 0.5,
        float(win_sample_weight),
        np.where(accepted_delta < 0.0, float(loss_sample_weight), float(tie_sample_weight)),
    ).astype(np.float32)
    if backend == "xgboost":
        return quick_module._train_xgboost_model(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            feature_names=feature_names,
            objective="binary:logistic",
            num_boost_round=int(num_boost_round),
            seed=int(seed),
            sample_weight=sample_weight,
        )
    if backend == "lightgbm":
        return quick_module._train_lightgbm_model(
            x=x.astype(np.float32),
            y=y.astype(np.float32),
            objective="binary",
            num_boost_round=int(num_boost_round),
            seed=int(seed),
            sample_weight=sample_weight,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _predict_backend(quick_module: Any, backend: str, model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if backend == "xgboost":
        return quick_module._xgb_predict(model, x.astype(np.float32), feature_names)
    return quick_module._lgb_predict(model, x.astype(np.float32))


def _split_proposals_with_rescue(
    sweep_module: Any,
    quick_module: Any,
    *,
    data: dict[str, Any],
    dqn_scores: np.ndarray,
    risk_scores: np.ndarray,
    rescue_scores: np.ndarray,
    selection_margin: float,
) -> pd.DataFrame:
    proposals = sweep_module._split_proposals(
        quick_module,
        data=data,
        dqn_scores=dqn_scores,
        risk_scores=risk_scores,
        selection_margin=float(selection_margin),
    )
    rescue = []
    for _, row in proposals.iterrows():
        row_index = row.get("row_index")
        rescue.append(-math.inf if row_index is None or pd.isna(row_index) else float(rescue_scores[int(row_index)]))
    proposals["rescue_score"] = rescue
    proposals["margin_over_threshold"] = proposals["dqn_margin"].astype(float) - float(selection_margin)
    return proposals


def _combined_metrics(
    proposals: pd.DataFrame,
    *,
    risk_cutoff: float,
    rescue_cutoff: float,
    rescue_min_margin_over_threshold: float,
    rescue_max_risk_score: float,
    win_groups: int,
) -> dict[str, Any]:
    groups = int(len(proposals))
    raw = proposals["raw_override"].astype(bool).to_numpy()
    risk = proposals["risk_score"].astype(float).to_numpy()
    rescue = proposals["rescue_score"].astype(float).to_numpy()
    margin_over = proposals["margin_over_threshold"].astype(float).to_numpy()
    allow = raw & (
        (risk <= float(risk_cutoff))
        | (
            (rescue >= float(rescue_cutoff))
            & (margin_over >= float(rescue_min_margin_over_threshold))
            & (risk <= float(rescue_max_risk_score))
        )
    )
    override = proposals.loc[allow]
    accepted = np.where(allow, proposals["accepted_delta"].to_numpy(dtype=np.float32), 0.0)
    reward = np.where(allow, proposals["reward_delta"].to_numpy(dtype=np.float32), 0.0)
    utility = np.where(allow, proposals["utility_delta"].to_numpy(dtype=np.float32), 0.0)
    override_count = int(allow.sum())
    raw_override_count = int(raw.sum())
    rescued_count = int((raw & (risk > float(risk_cutoff)) & allow).sum())
    loss_rate = None if override_count == 0 else float((override["accepted_delta"].astype(float) < 0.0).mean())
    win_rate = None if override_count == 0 else float((override["accepted_delta"].astype(float) > 0.0).mean())
    tie_rate = None if override_count == 0 else float((override["accepted_delta"].astype(float) == 0.0).mean())
    captured_win_groups = int(override[override["accepted_delta"].astype(float) > 0.0]["group_id"].nunique())
    return {
        "groups": groups,
        "override_count": override_count,
        "override_rate": float(override_count / max(groups, 1)),
        "raw_override_count": raw_override_count,
        "vetoed_override_count": int(raw_override_count - override_count),
        "rescued_override_count": rescued_count,
        "selected_win_rate_when_overridden": win_rate,
        "selected_loss_rate_when_overridden": loss_rate,
        "selected_tie_rate_when_overridden": tie_rate,
        "total_selected_accepted_delta_vs_base": int(round(float(np.sum(accepted)))),
        "mean_selected_accepted_delta_vs_base": float(np.mean(accepted)) if groups else 0.0,
        "mean_selected_reward_delta_vs_base": float(np.mean(reward)) if groups else 0.0,
        "mean_selected_utility_delta_vs_base": float(np.mean(utility)) if groups else 0.0,
        "win_groups": int(win_groups),
        "win_group_rate": float(win_groups / max(groups, 1)),
        "captured_win_groups": captured_win_groups,
        "captured_win_group_rate": float(captured_win_groups / max(win_groups, 1)),
        "risk_selector_score_cutoff": float(risk_cutoff),
        "rescue_score_cutoff": float(rescue_cutoff),
        "rescue_min_margin_over_threshold": float(rescue_min_margin_over_threshold),
        "rescue_max_risk_score": float(rescue_max_risk_score),
    }


def _flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_delta": metrics["total_selected_accepted_delta_vs_base"],
        f"{prefix}_reward": metrics["mean_selected_reward_delta_vs_base"],
        f"{prefix}_loss": metrics["selected_loss_rate_when_overridden"],
        f"{prefix}_win": metrics["selected_win_rate_when_overridden"],
        f"{prefix}_tie": metrics["selected_tie_rate_when_overridden"],
        f"{prefix}_override_count": metrics["override_count"],
        f"{prefix}_override_rate": metrics["override_rate"],
        f"{prefix}_vetoed": metrics["vetoed_override_count"],
        f"{prefix}_rescued": metrics["rescued_override_count"],
    }


def _best_under(
    rows: list[dict[str, Any]],
    *,
    split_prefix: str,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
) -> dict[str, Any] | None:
    feasible = []
    for row in rows:
        loss = row.get(f"{split_prefix}_loss")
        if loss is None:
            continue
        if int(row[f"{split_prefix}_override_count"]) < int(min_override_count):
            continue
        if float(row[f"{split_prefix}_override_rate"]) > float(max_override_rate):
            continue
        if float(row[f"{split_prefix}_delta"]) < float(min_total_delta):
            continue
        if float(loss) > float(max_loss_rate):
            continue
        feasible.append(row)
    if not feasible:
        return None
    return dict(
        max(
            feasible,
            key=lambda row: (
                float(row[f"{split_prefix}_delta"]),
                float(row[f"{split_prefix}_reward"]),
                float(row[f"{split_prefix}_rescued"]),
                -float(row[f"{split_prefix}_loss"]),
                float(row[f"{split_prefix}_override_count"]),
            ),
        )
    )


def _candidate_thresholds(values: np.ndarray, *, max_count: int, include: list[float] | None = None) -> list[float]:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float32)
    candidates: list[float] = list(include or [])
    if finite.size:
        if finite.size <= int(max_count):
            candidates.extend(float(v) for v in np.unique(finite))
        else:
            qs = np.linspace(0.0, 1.0, int(max_count))
            candidates.extend(float(v) for v in np.quantile(finite, qs))
        candidates.append(float(np.max(finite) + 1.0e-6))
        candidates.append(float(np.min(finite) - 1.0e-6))
    return sorted(set(float(v) for v in candidates if math.isfinite(float(v))))


def _export_artifact(
    sweep_module: Any,
    *,
    source_artifact: Path,
    output_dir: Path,
    rescue_model_path: Path,
    rescue_backend: str,
    rescue_feature_names: list[str],
    selected: dict[str, Any],
) -> Path:
    meta = json.loads(source_artifact.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dqn_model_path = sweep_module._resolve_model_path(source_artifact, str(meta["model_path"]))
    risk_selector = dict(meta["risk_selector"])
    risk_model_path = sweep_module._resolve_model_path(source_artifact, str(risk_selector["model_path"]))
    copied_dqn = output_dir / dqn_model_path.name
    copied_risk = output_dir / risk_model_path.name
    copied_rescue = output_dir / rescue_model_path.name
    if dqn_model_path.resolve() != copied_dqn.resolve():
        shutil.copy2(dqn_model_path, copied_dqn)
    if risk_model_path.resolve() != copied_risk.resolve():
        shutil.copy2(risk_model_path, copied_risk)
    if rescue_model_path.resolve() != copied_rescue.resolve():
        shutil.copy2(rescue_model_path, copied_rescue)
    meta["model_path"] = copied_dqn.name
    risk_selector["model_path"] = copied_risk.name
    risk_selector["score_cutoff"] = float(selected["risk_cutoff"])
    risk_selector["rescue_backend"] = str(rescue_backend)
    risk_selector["rescue_model_path"] = copied_rescue.name
    risk_selector["rescue_feature_names"] = list(rescue_feature_names)
    risk_selector["rescue_score_cutoff"] = float(selected["rescue_cutoff"])
    risk_selector["rescue_min_margin_over_threshold"] = float(selected["rescue_min_margin_over_threshold"])
    risk_selector["rescue_max_risk_score"] = float(selected["rescue_max_risk_score"])
    meta["risk_selector"] = risk_selector
    meta["tuned_from_artifact"] = str(source_artifact)
    meta["rescue_tuning"] = dict(selected)
    artifact_path = output_dir / "torch_dqn_base_relative_oof_risk_rescue_tree_ranker.json"
    _write_json(artifact_path, meta)
    return artifact_path


def train_and_sweep(
    *,
    run_dir: Path,
    artifact: Path,
    output_dir: Path,
    backend: str,
    oof_folds: int,
    oof_epochs: int,
    train_margin_slack: float,
    min_risk_examples: int,
    min_win_examples: int,
    num_boost_round: int,
    max_loss_rates: list[float],
    export_max_loss_rate: float | None,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
    rescue_max_risk_score: float,
    seed: int,
    device: str,
    dqn_params: dict[str, Any],
    win_sample_weight: float,
    loss_sample_weight: float,
    tie_sample_weight: float,
) -> dict[str, Any]:
    import torch

    script_dir = Path(__file__).resolve().parent
    quick = _load_module("quick_exception_ranker_ab", script_dir / "quick_exception_ranker_ab.py")
    dqn = _load_module("train_dqn_base_relative_ranker", script_dir / "train_dqn_base_relative_ranker.py")
    oof = _load_module("train_dqn_base_relative_oof_risk_selector", script_dir / "train_dqn_base_relative_oof_risk_selector.py")
    sweep = _load_module("sweep_dqn_oof_risk_selector", script_dir / "sweep_dqn_oof_risk_selector.py")

    meta, dqn_model, _ = oof._load_torch_artifact_model(torch, artifact)
    selection_margin = float(meta["selection_margin"])
    top_k = int(meta.get("candidate_pool_top_k", 8))
    risk_selector = dict(meta["risk_selector"])
    risk_backend = str(risk_selector["backend"])
    risk_feature_names = [str(name) for name in risk_selector["feature_names"]]
    risk_model = sweep._load_backend_model(risk_backend, sweep._resolve_model_path(artifact, str(risk_selector["model_path"])))

    train_pool = oof._load_split_pool(quick, run_dir, "train", top_k)
    calibration_pool = oof._load_split_pool(quick, run_dir, "calibration", top_k)
    eval_pool = oof._load_split_pool(quick, run_dir, "eval", top_k)

    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    oof_scores, fold_summaries = oof._oof_dqn_scores(
        torch=torch,
        dqn_module=dqn,
        train_module=_load_module("train_distilled_dqn_ranker", script_dir / "train_distilled_dqn_ranker.py"),
        train_pool=train_pool,
        folds=int(oof_folds),
        seed=int(seed),
        device=requested_device,
        dqn_params=dqn_params,
    )
    oof_x, rescue_feature_names = oof._risk_feature_matrix(
        data=train_pool,
        scores=oof_scores,
        selection_margin=selection_margin,
    )
    train_meta = train_pool["metadata"].reset_index(drop=True)
    proposal_indices = np.zeros((0,), dtype=np.int64)
    proposal_summary: dict[str, Any] = {}
    for slack in [float(train_margin_slack), max(float(train_margin_slack), 1.5), math.inf]:
        proposal_indices = oof._candidate_proposal_indices(
            quick,
            train_pool,
            oof_scores,
            selection_margin=selection_margin,
            margin_slack=float(slack),
        )
        accepted = train_meta.loc[proposal_indices, "accepted_delta_vs_base"].to_numpy(dtype=np.float32)
        win_count = int(np.sum(accepted > 0.0))
        loss_count = int(np.sum(accepted < 0.0))
        proposal_summary = {
            "margin_slack": None if not math.isfinite(float(slack)) else float(slack),
            "examples": int(len(proposal_indices)),
            "win_examples": win_count,
            "loss_examples": loss_count,
            "tie_examples": int(len(proposal_indices) - win_count - loss_count),
        }
        if len(proposal_indices) >= int(min_risk_examples) and win_count >= int(min_win_examples):
            break
    accepted_delta = train_meta.loc[proposal_indices, "accepted_delta_vs_base"].to_numpy(dtype=np.float32)
    y = (accepted_delta > 0.0).astype(np.float32)
    rescue_model = _train_win_model(
        quick,
        backend=str(backend),
        x=oof_x[proposal_indices],
        y=y,
        accepted_delta=accepted_delta,
        feature_names=rescue_feature_names,
        num_boost_round=int(num_boost_round),
        seed=int(seed) + 7000,
        win_sample_weight=float(win_sample_weight),
        loss_sample_weight=float(loss_sample_weight),
        tie_sample_weight=float(tie_sample_weight),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rescue_model_path = output_dir / f"{backend}_dqn_base_relative_oof_rescue_selector.{_model_suffix(str(backend))}"
    _save_model(str(backend), rescue_model, rescue_model_path)

    split_data = {"train": train_pool, "cal": calibration_pool, "eval": eval_pool}
    proposals: dict[str, pd.DataFrame] = {}
    win_groups: dict[str, int] = {}
    for split, data in split_data.items():
        dqn_scores = dqn._predict(torch, dqn_model, data["x"])
        risk_x, _ = oof._risk_feature_matrix(data=data, scores=dqn_scores, selection_margin=selection_margin)
        risk_scores = _predict_backend(quick, risk_backend, risk_model, risk_x, risk_feature_names)
        rescue_scores = _predict_backend(quick, str(backend), rescue_model, risk_x, rescue_feature_names)
        proposals[split] = _split_proposals_with_rescue(
            sweep,
            quick,
            data=data,
            dqn_scores=dqn_scores,
            risk_scores=risk_scores,
            rescue_scores=rescue_scores,
            selection_margin=selection_margin,
        )
        win_groups[split] = sweep._win_group_count(data["metadata"].reset_index(drop=True))

    cal_raw = proposals["cal"][proposals["cal"]["raw_override"].astype(bool)]
    risk_cutoffs = _candidate_thresholds(
        cal_raw["risk_score"].to_numpy(dtype=np.float32),
        max_count=80,
        include=[float(risk_selector["score_cutoff"])],
    )
    rescue_cutoffs = _candidate_thresholds(cal_raw["rescue_score"].to_numpy(dtype=np.float32), max_count=80)
    margin_thresholds = [-math.inf, 0.0]
    rows: list[dict[str, Any]] = []
    for risk_cutoff in risk_cutoffs:
        for rescue_cutoff in rescue_cutoffs:
            for margin_threshold in margin_thresholds:
                train_metrics = _combined_metrics(
                    proposals["train"],
                    risk_cutoff=risk_cutoff,
                    rescue_cutoff=rescue_cutoff,
                    rescue_min_margin_over_threshold=margin_threshold,
                    rescue_max_risk_score=float(rescue_max_risk_score),
                    win_groups=win_groups["train"],
                )
                cal_metrics = _combined_metrics(
                    proposals["cal"],
                    risk_cutoff=risk_cutoff,
                    rescue_cutoff=rescue_cutoff,
                    rescue_min_margin_over_threshold=margin_threshold,
                    rescue_max_risk_score=float(rescue_max_risk_score),
                    win_groups=win_groups["cal"],
                )
                eval_metrics = _combined_metrics(
                    proposals["eval"],
                    risk_cutoff=risk_cutoff,
                    rescue_cutoff=rescue_cutoff,
                    rescue_min_margin_over_threshold=margin_threshold,
                    rescue_max_risk_score=float(rescue_max_risk_score),
                    win_groups=win_groups["eval"],
                )
                rows.append(
                    {
                        "risk_cutoff": float(risk_cutoff),
                        "rescue_cutoff": float(rescue_cutoff),
                        "rescue_min_margin_over_threshold": float(margin_threshold),
                        "rescue_max_risk_score": float(rescue_max_risk_score),
                        **_flatten("train", train_metrics),
                        **_flatten("cal", cal_metrics),
                        **_flatten("eval", eval_metrics),
                    }
                )

    cal_best = {
        str(max_loss): _best_under(
            rows,
            split_prefix="cal",
            max_loss_rate=float(max_loss),
            min_override_count=int(min_override_count),
            min_total_delta=float(min_total_delta),
            max_override_rate=float(max_override_rate),
        )
        for max_loss in max_loss_rates
    }
    eval_oracle = {
        str(max_loss): _best_under(
            rows,
            split_prefix="eval",
            max_loss_rate=float(max_loss),
            min_override_count=int(min_override_count),
            min_total_delta=float(min_total_delta),
            max_override_rate=float(max_override_rate),
        )
        for max_loss in max_loss_rates
    }
    exported_artifact = None
    export_metrics = None
    if export_max_loss_rate is not None:
        export_metrics = cal_best.get(str(float(export_max_loss_rate))) or cal_best.get(str(export_max_loss_rate))
        if export_metrics is not None:
            exported_artifact = _export_artifact(
                sweep,
                source_artifact=artifact,
                output_dir=output_dir,
                rescue_model_path=rescue_model_path,
                rescue_backend=str(backend),
                rescue_feature_names=rescue_feature_names,
                selected=export_metrics,
            )

    summary = {
        "artifact": str(artifact),
        "exported_artifact": None if exported_artifact is None else str(exported_artifact),
        "export_metrics": export_metrics,
        "selection_margin": float(selection_margin),
        "rescue_model_path": str(rescue_model_path),
        "rescue_train_label_rate": {
            "examples": int(len(proposal_indices)),
            "win_examples": int(np.sum(y > 0.5)),
            "win_rate": float(np.mean(y)) if y.size else None,
            "loss_examples": int(np.sum(accepted_delta < 0.0)),
            "tie_examples": int(np.sum(accepted_delta == 0.0)),
        },
        "oof_fold_summaries": fold_summaries,
        "oof_proposal_summary": proposal_summary,
        "calibration_best": cal_best,
        "eval_oracle": eval_oracle,
        "grid_size": int(len(rows)),
        "rows": rows,
    }
    _write_json(output_dir / "rescue_selector_sweep_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OOF win/rescue head on top of a DQN OOF risk selector.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--oof-epochs", type=int, default=30)
    parser.add_argument("--train-margin-slack", type=float, default=1.0)
    parser.add_argument("--min-risk-examples", type=int, default=250)
    parser.add_argument("--min-win-examples", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=160)
    parser.add_argument("--max-loss-rate", type=float, action="append", default=None)
    parser.add_argument("--export-max-loss-rate", type=float, default=None)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.35)
    parser.add_argument("--rescue-max-risk-score", type=float, default=0.90)
    parser.add_argument("--secondary-scale", type=float, default=0.20)
    parser.add_argument("--target-clip", type=float, default=4.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--pair-loss-weight", type=float, default=0.7)
    parser.add_argument("--rank-loss-weight", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-group", type=int, default=4)
    parser.add_argument("--base-weight", type=float, default=4.0)
    parser.add_argument("--win-weight", type=float, default=8.0)
    parser.add_argument("--loss-weight", type=float, default=10.0)
    parser.add_argument("--tie-weight", type=float, default=0.5)
    parser.add_argument("--secondary-weight", type=float, default=1.0)
    parser.add_argument("--rescue-win-sample-weight", type=float, default=8.0)
    parser.add_argument("--rescue-loss-sample-weight", type=float, default=8.0)
    parser.add_argument("--rescue-tie-sample-weight", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    max_loss_rates = list(args.max_loss_rate or [0.08, 0.10, 0.11, 0.12, 0.121, 0.125, 0.13])
    dqn_params = {
        "secondary_scale": float(args.secondary_scale),
        "target_clip": float(args.target_clip),
        "hidden_dim": int(args.hidden_dim),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "epochs": int(args.oof_epochs),
        "batch_size": int(args.batch_size),
        "pair_batch_size": int(args.pair_batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "pair_loss_weight": float(args.pair_loss_weight),
        "rank_loss_weight": float(args.rank_loss_weight),
        "max_pairs_per_group": int(args.max_pairs_per_group),
        "base_weight": float(args.base_weight),
        "win_weight": float(args.win_weight),
        "loss_weight": float(args.loss_weight),
        "tie_weight": float(args.tie_weight),
        "secondary_weight": float(args.secondary_weight),
    }
    summary = train_and_sweep(
        run_dir=Path(args.run_dir),
        artifact=Path(args.artifact),
        output_dir=Path(args.output_dir),
        backend=str(args.backend),
        oof_folds=int(args.oof_folds),
        oof_epochs=int(args.oof_epochs),
        train_margin_slack=float(args.train_margin_slack),
        min_risk_examples=int(args.min_risk_examples),
        min_win_examples=int(args.min_win_examples),
        num_boost_round=int(args.num_boost_round),
        max_loss_rates=max_loss_rates,
        export_max_loss_rate=args.export_max_loss_rate,
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
        rescue_max_risk_score=float(args.rescue_max_risk_score),
        seed=int(args.seed),
        device=str(args.device),
        dqn_params=dqn_params,
        win_sample_weight=float(args.rescue_win_sample_weight),
        loss_sample_weight=float(args.rescue_loss_sample_weight),
        tie_sample_weight=float(args.rescue_tie_sample_weight),
    )
    print(json.dumps(_json_safe({k: v for k, v in summary.items() if k != "rows"}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
