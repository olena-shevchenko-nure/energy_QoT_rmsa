from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_quick_module() -> Any:
    module_path = Path(__file__).with_name("quick_exception_ranker_ab.py")
    spec = importlib.util.spec_from_file_location("quick_exception_ranker_ab", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    return "json" if backend == "xgboost" else "txt"


def _emergency_safety_guard() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "emergency",
        "check_fragmentation": False,
        "check_small_gap": False,
        "check_lmax": False,
        "check_qot_margin": True,
        "check_energy": True,
        "check_delay": True,
        "fragmentation_slack": 0.50,
        "small_gap_slack": 1.0,
        "lmax_slack_slots": 40,
        "qot_margin_slack": 0.25,
        "energy_slack_w": 480.0,
        "delay_slack_ms": 10.0,
    }


def _nonbase_view(data: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    metadata = data["metadata"].reset_index(drop=True)
    mask = (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()
    subset = {
        "x": data["x"][mask].astype(np.float32),
        "metadata": metadata.loc[mask].reset_index(drop=True).copy(),
        "feature_names": list(data["feature_names"]),
    }
    return subset, np.flatnonzero(mask)


def _full_head_preds(module: Any, backend: str, heads: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    subset, positions = _nonbase_view(data)
    preds_subset = module._predict_heads(backend, heads, subset)
    result = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
    result["loss"].fill(1.0)
    result["delta"].fill(-1.0)
    for name, values in preds_subset.items():
        result[name][positions] = values
    return result


def _artifact_meta(
    *,
    backend: str,
    model_path: Path,
    feature_names: list[str],
    win_model_path: Path,
    loss_model_path: Path,
    delta_model_path: Path,
    thresholds: dict[str, float],
    top_k: int,
    risk_selector: dict[str, Any] | None = None,
) -> dict[str, Any]:
    check_loss_prob = not bool(risk_selector and risk_selector.get("enabled", False))
    advantage_thresholds = dict(thresholds)
    if not check_loss_prob:
        advantage_thresholds["max_loss_prob"] = 1.0
    return {
        "backend": backend,
        "model_path": model_path.name,
        "feature_names": list(feature_names),
        "candidate_pool": "energy_topk_hybrid",
        "candidate_pool_top_k": int(top_k),
        "selection_mode": "positive_advantage",
        "residual_beta": 0.05,
        "selection_margin": 0.0,
        "base_policy": "energy-aware-ksp-bm-ff",
        "safety_guard": _emergency_safety_guard(),
        "advantage_gate": {
            "enabled": True,
            "backend": backend,
            "feature_source": "ranker_features",
            "feature_names": list(feature_names),
            "win_model_path": win_model_path.name,
            "loss_model_path": loss_model_path.name,
            "delta_model_path": delta_model_path.name,
            "check_loss_prob": bool(check_loss_prob),
            "win_weight": 1.0,
            "loss_weight": 2.0,
            "delta_weight": 1.0,
            "ranker_margin_weight": 0.0,
            **advantage_thresholds,
        },
        "risk_selector": dict(risk_selector or {"enabled": False}),
    }


def export_backend(
    *,
    run_dir: Path,
    output_dir: Path,
    backend: str,
    variants: list[str],
    top_k: int,
    num_boost_round: int,
    threshold_fraction: float,
    seed: int,
    min_override_count: int,
    min_total_delta: float,
) -> dict[str, Any]:
    module = _load_quick_module()
    original_train = module._load_split(run_dir, "train")
    original_eval = module._load_split(run_dir, "eval")
    train_pool = module._add_runtime_features(module._filter_small_pool(original_train, top_k=top_k))
    eval_pool = module._add_runtime_features(module._filter_small_pool(original_eval, top_k=top_k))
    train_inner, threshold_val = module._split_train_threshold(train_pool, threshold_fraction=threshold_fraction, seed=seed)
    train_nonbase = module._non_base_dataset(train_inner)

    heads = module._train_backend_heads(backend=backend, train=train_nonbase, num_boost_round=num_boost_round, seed=seed)
    ranker = module._train_ranker(backend=backend, train=train_inner, num_boost_round=num_boost_round, seed=seed + 10)

    threshold_heads = _full_head_preds(module, backend, heads, threshold_val)
    eval_heads = _full_head_preds(module, backend, heads, eval_pool)
    train_heads = _full_head_preds(module, backend, heads, train_inner)
    threshold_ranker_scores = module._predict_ranker(backend, ranker, threshold_val)
    eval_ranker_scores = module._predict_ranker(backend, ranker, eval_pool)
    train_ranker_scores = module._predict_ranker(backend, ranker, train_inner)

    suffix = _model_suffix(backend)
    prefix = output_dir / backend
    ranker_model_path = prefix.with_name(f"{backend}_quick_ranker.{suffix}")
    win_model_path = prefix.with_name(f"{backend}_quick_advantage_win.{suffix}")
    loss_model_path = prefix.with_name(f"{backend}_quick_advantage_loss.{suffix}")
    delta_model_path = prefix.with_name(f"{backend}_quick_advantage_delta.{suffix}")
    _save_model(backend, ranker, ranker_model_path)
    _save_model(backend, heads["win"], win_model_path)
    _save_model(backend, heads["loss"], loss_model_path)
    _save_model(backend, heads["delta"], delta_model_path)

    def three_selector(*, data: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
        preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
        return module._select_three_head(data=data, preds=preds, thresholds=thresholds, safety_enabled=True)

    def three_row_selector(
        *,
        data: dict[str, Any],
        thresholds: dict[str, float],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        preds = threshold_heads if data is threshold_val else eval_heads if data is eval_pool else train_heads
        return module._three_head_selected_rows(
            data=data,
            preds=preds,
            thresholds=thresholds,
            safety_enabled=True,
            apply_loss_threshold=False,
        )

    risk_selector_model = None
    risk_scores: dict[str, np.ndarray] | None = None
    loss_reference_path: Path | None = None
    if any("risk" in variant for variant in variants):
        reference_loss_scores = train_heads["loss"][module._nonbase_mask(train_inner)]
        train_loss_percentile = module._percentile_from_reference(train_heads["loss"], reference_loss_scores)
        threshold_loss_percentile = module._percentile_from_reference(threshold_heads["loss"], reference_loss_scores)
        eval_loss_percentile = module._percentile_from_reference(eval_heads["loss"], reference_loss_scores)
        train_risk_data = module._risk_selector_dataset(
            data=train_inner,
            heads=train_heads,
            ranker_scores=train_ranker_scores,
            loss_percentile=train_loss_percentile,
        )
        threshold_risk_data = module._risk_selector_dataset(
            data=threshold_val,
            heads=threshold_heads,
            ranker_scores=threshold_ranker_scores,
            loss_percentile=threshold_loss_percentile,
        )
        eval_risk_data = module._risk_selector_dataset(
            data=eval_pool,
            heads=eval_heads,
            ranker_scores=eval_ranker_scores,
            loss_percentile=eval_loss_percentile,
        )
        risk_selector_model = module._train_risk_selector(
            backend=backend,
            risk_data=train_risk_data,
            num_boost_round=num_boost_round,
            seed=seed + 20,
        )
        risk_model_path = prefix.with_name(f"{backend}_quick_risk_selector.{suffix}")
        _save_model(backend, risk_selector_model, risk_model_path)
        loss_reference_path = prefix.with_name(f"{backend}_quick_loss_percentile_reference.npy")
        np.save(loss_reference_path, np.sort(np.asarray(reference_loss_scores, dtype=np.float32)))
        risk_scores = {
            "train": module._predict_risk_selector(backend, risk_selector_model, train_risk_data),
            "threshold": module._predict_risk_selector(backend, risk_selector_model, threshold_risk_data),
            "eval": module._predict_risk_selector(backend, risk_selector_model, eval_risk_data),
        }
        risk_feature_names = list(train_risk_data["feature_names"])
    else:
        risk_model_path = None
        risk_feature_names = []

    exported: dict[str, Any] = {}
    for variant in variants:
        normalized = str(variant).strip().lower()
        if normalized.endswith("old5"):
            max_loss_rate = 0.05
            thresholds, threshold_metrics = module._tune_thresholds(
                selector=three_selector,
                data=threshold_val,
                max_loss_rate=max_loss_rate,
                min_override_count=min_override_count,
                min_total_delta=min_total_delta,
            )
            train_metrics = module._select_three_head(data=train_inner, preds=train_heads, thresholds=thresholds, safety_enabled=True)
            eval_metrics = module._select_three_head(data=eval_pool, preds=eval_heads, thresholds=thresholds, safety_enabled=True)
            risk_meta = {"enabled": False}
        elif normalized.endswith("old10"):
            max_loss_rate = 0.10
            thresholds, threshold_metrics = module._tune_thresholds(
                selector=three_selector,
                data=threshold_val,
                max_loss_rate=max_loss_rate,
                min_override_count=min_override_count,
                min_total_delta=min_total_delta,
            )
            train_metrics = module._select_three_head(data=train_inner, preds=train_heads, thresholds=thresholds, safety_enabled=True)
            eval_metrics = module._select_three_head(data=eval_pool, preds=eval_heads, thresholds=thresholds, safety_enabled=True)
            risk_meta = {"enabled": False}
        elif normalized.endswith("risk5"):
            if risk_scores is None or risk_model_path is None or loss_reference_path is None:
                raise RuntimeError("risk5 variant requested without a trained risk selector")
            max_loss_rate = 0.05
            thresholds, threshold_metrics = module._tune_score_veto(
                row_selector=three_row_selector,
                data=threshold_val,
                scores=risk_scores["threshold"],
                score_field="learned_risk_selector_score",
                max_loss_rate=max_loss_rate,
                min_override_count=min_override_count,
                min_total_delta=min_total_delta,
            )
            train_metrics = module._select_with_score_veto(
                row_selector=three_row_selector,
                data=train_inner,
                scores=risk_scores["train"],
                thresholds=thresholds,
                score_field="learned_risk_selector_score",
            )
            eval_metrics = module._select_with_score_veto(
                row_selector=three_row_selector,
                data=eval_pool,
                scores=risk_scores["eval"],
                thresholds=thresholds,
                score_field="learned_risk_selector_score",
            )
            risk_meta = {
                "enabled": True,
                "backend": backend,
                "model_path": risk_model_path.name,
                "feature_names": risk_feature_names,
                "loss_percentile_reference_path": loss_reference_path.name,
                "score_cutoff": float(thresholds["score_veto_cutoff"]),
                "score_field": "learned_risk_selector_score",
            }
        else:
            raise ValueError(f"Unsupported variant: {variant}")

        artifact_name = f"{backend}_{normalized}_tree_ranker.json"
        artifact_path = output_dir / artifact_name
        meta = _artifact_meta(
            backend=backend,
            model_path=ranker_model_path,
            feature_names=list(train_inner["feature_names"]),
            win_model_path=win_model_path,
            loss_model_path=loss_model_path,
            delta_model_path=delta_model_path,
            thresholds=thresholds,
            top_k=top_k,
            risk_selector=risk_meta,
        )
        _write_json(artifact_path, meta)
        exported[normalized] = {
            "artifact_path": str(artifact_path),
            "thresholds": thresholds,
            "threshold_val": threshold_metrics,
            "train_inner": train_metrics,
            "eval": eval_metrics,
        }
    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "backend": backend,
        "variants": exported,
        "top_k": int(top_k),
        "num_boost_round": int(num_boost_round),
        "threshold_fraction": float(threshold_fraction),
        "seed": int(seed),
    }
    _write_json(output_dir / f"{backend}_quick_runtime_export_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quick A/B tree-ranker runtime artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--variants", required=True, help="Comma-separated variants, e.g. xgboost_old10,xgboost_risk5")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-boost-round", type=int, default=60)
    parser.add_argument("--threshold-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    args = parser.parse_args()
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    summary = export_backend(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir),
        backend=str(args.backend),
        variants=variants,
        top_k=int(args.top_k),
        num_boost_round=int(args.num_boost_round),
        threshold_fraction=float(args.threshold_fraction),
        seed=int(args.seed),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
