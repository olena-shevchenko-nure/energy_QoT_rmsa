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


def _load_backend_model(backend: str, path: Path) -> Any:
    if backend == "xgboost":
        import xgboost as xgb

        model = xgb.Booster()
        model.load_model(str(path))
        return model
    if backend == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))
    raise ValueError(f"Unsupported backend: {backend}")


def _resolve_model_path(meta_path: Path, value: str | None) -> Path:
    if not value:
        raise ValueError(f"Missing model path in {meta_path}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return meta_path.parent / path


def _predict_risk(quick_module: Any, backend: str, model: Any, x: np.ndarray, feature_names: list[str]) -> np.ndarray:
    if backend == "xgboost":
        return quick_module._xgb_predict(model, x.astype(np.float32), feature_names)
    return quick_module._lgb_predict(model, x.astype(np.float32))


def _load_pool(quick_module: Any, oof_module: Any, run_dir: Path, split: str, top_k: int) -> dict[str, Any]:
    return oof_module._load_split_pool(quick_module, run_dir, split, int(top_k))


def _score_split(
    *,
    torch: Any,
    quick_module: Any,
    dqn_module: Any,
    oof_module: Any,
    data: dict[str, Any],
    dqn_model: Any,
    risk_model: Any,
    risk_backend: str,
    risk_feature_names: list[str],
    selection_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    dqn_scores = dqn_module._predict(torch, dqn_model, data["x"])
    risk_x, _ = oof_module._risk_feature_matrix(
        data=data,
        scores=dqn_scores,
        selection_margin=float(selection_margin),
    )
    risk_scores = _predict_risk(quick_module, risk_backend, risk_model, risk_x, risk_feature_names)
    return dqn_scores.astype(np.float32), risk_scores.astype(np.float32)


def _split_proposals(
    quick_module: Any,
    *,
    data: dict[str, Any],
    dqn_scores: np.ndarray,
    risk_scores: np.ndarray,
    selection_margin: float,
) -> pd.DataFrame:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = quick_module._safety_mask(data, enabled=True)
    rows: list[dict[str, Any]] = []
    for group_id, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        base_index = int(group["base_index"].iloc[0])
        base_rows = group[group["candidate_index"].astype(int) == base_index]
        base_row_index = int(base_rows.index[0]) if not base_rows.empty else int(group_indices[0])
        base_score = float(dqn_scores[base_row_index])
        base_utility = float(metadata.at[base_row_index, "utility"])
        selectable = [int(index) for index in group_indices if bool(eligible[int(index)])]
        if selectable:
            best = int(min(selectable, key=lambda index: (-float(dqn_scores[index]), int(metadata.at[index, "candidate_index"]))))
            margin = float(dqn_scores[best] - base_score)
        else:
            best = -1
            margin = -math.inf
        raw_override = bool(best >= 0 and margin >= float(selection_margin))
        if raw_override:
            row = metadata.loc[best]
            accepted_delta = float(row["accepted_delta_vs_base"])
            reward_delta = float(row.get("future_env_reward_delta_vs_base", 0.0))
            utility_delta = float(row["utility"]) - base_utility
            risk_score = float(risk_scores[best])
            candidate_index = int(row["candidate_index"])
        else:
            accepted_delta = 0.0
            reward_delta = 0.0
            utility_delta = 0.0
            risk_score = math.inf
            candidate_index = -1
        rows.append(
            {
                "group_id": int(group_id),
                "raw_override": raw_override,
                "row_index": int(best) if best >= 0 else None,
                "candidate_index": candidate_index,
                "risk_score": risk_score,
                "dqn_margin": margin,
                "accepted_delta": accepted_delta,
                "reward_delta": reward_delta,
                "utility_delta": utility_delta,
            }
        )
    return pd.DataFrame(rows)


def _win_group_count(metadata: pd.DataFrame) -> int:
    nonbase_win = metadata[
        (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int))
        & (metadata["accepted_delta_vs_base"].astype(float) > 0.0)
    ]
    return int(nonbase_win["group_id"].nunique()) if not nonbase_win.empty else 0


def _metrics_from_proposals(proposals: pd.DataFrame, *, cutoff: float, win_groups: int) -> dict[str, Any]:
    groups = int(len(proposals))
    finite_cutoff = float(cutoff)
    override_mask = proposals["raw_override"].astype(bool).to_numpy() & (
        proposals["risk_score"].astype(float).to_numpy() <= finite_cutoff
    )
    override = proposals.loc[override_mask]
    accepted = np.where(override_mask, proposals["accepted_delta"].to_numpy(dtype=np.float32), 0.0)
    reward = np.where(override_mask, proposals["reward_delta"].to_numpy(dtype=np.float32), 0.0)
    utility = np.where(override_mask, proposals["utility_delta"].to_numpy(dtype=np.float32), 0.0)
    override_count = int(override_mask.sum())
    raw_override_count = int(proposals["raw_override"].astype(bool).sum())
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
        "risk_selector_score_cutoff": finite_cutoff,
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
    feasible: list[dict[str, Any]] = []
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
                -float(row[f"{split_prefix}_loss"]),
                float(row[f"{split_prefix}_override_count"]),
                float(row["cutoff"]),
            ),
        )
    )


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_delta": metrics["total_selected_accepted_delta_vs_base"],
        f"{prefix}_reward": metrics["mean_selected_reward_delta_vs_base"],
        f"{prefix}_loss": metrics["selected_loss_rate_when_overridden"],
        f"{prefix}_win": metrics["selected_win_rate_when_overridden"],
        f"{prefix}_tie": metrics["selected_tie_rate_when_overridden"],
        f"{prefix}_override_count": metrics["override_count"],
        f"{prefix}_override_rate": metrics["override_rate"],
        f"{prefix}_vetoed": metrics["vetoed_override_count"],
    }


def _export_artifact(
    *,
    source_artifact: Path,
    output_dir: Path,
    cutoff: float,
    suffix: str,
) -> Path:
    meta = json.loads(source_artifact.read_text(encoding="utf-8"))
    source_dir = source_artifact.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_model_path(source_artifact, str(meta["model_path"]))
    risk_selector = dict(meta["risk_selector"])
    risk_path = _resolve_model_path(source_artifact, str(risk_selector["model_path"]))
    copied_model = output_dir / model_path.name
    copied_risk = output_dir / risk_path.name
    if model_path.resolve() != copied_model.resolve():
        shutil.copy2(model_path, copied_model)
    if risk_path.resolve() != copied_risk.resolve():
        shutil.copy2(risk_path, copied_risk)
    meta["model_path"] = copied_model.name
    risk_selector["model_path"] = copied_risk.name
    risk_selector["score_cutoff"] = float(cutoff)
    meta["risk_selector"] = risk_selector
    meta["tuned_from_artifact"] = str(source_artifact)
    meta["tuned_score_cutoff"] = float(cutoff)
    output_path = output_dir / f"torch_dqn_base_relative_oof_risk_{suffix}_tree_ranker.json"
    _write_json(output_path, meta)
    return output_path


def run_sweep(
    *,
    run_dir: Path,
    artifact: Path,
    output_dir: Path,
    max_loss_rates: list[float],
    export_max_loss_rate: float | None,
    min_override_count: int,
    min_total_delta: float,
    max_override_rate: float,
) -> dict[str, Any]:
    import torch

    script_dir = Path(__file__).resolve().parent
    quick_module = _load_module("quick_exception_ranker_ab", script_dir / "quick_exception_ranker_ab.py")
    dqn_module = _load_module("train_dqn_base_relative_ranker", script_dir / "train_dqn_base_relative_ranker.py")
    oof_module = _load_module("train_dqn_base_relative_oof_risk_selector", script_dir / "train_dqn_base_relative_oof_risk_selector.py")

    meta, dqn_model, _ = oof_module._load_torch_artifact_model(torch, artifact)
    selection_margin = float(meta["selection_margin"])
    top_k = int(meta.get("candidate_pool_top_k", 8))
    risk_selector = dict(meta["risk_selector"])
    risk_backend = str(risk_selector["backend"])
    risk_feature_names = [str(name) for name in risk_selector["feature_names"]]
    risk_model = _load_backend_model(risk_backend, _resolve_model_path(artifact, str(risk_selector["model_path"])))

    split_data = {
        "train": _load_pool(quick_module, oof_module, run_dir, "train", top_k),
        "calibration": _load_pool(quick_module, oof_module, run_dir, "calibration", top_k),
        "eval": _load_pool(quick_module, oof_module, run_dir, "eval", top_k),
    }
    proposals: dict[str, pd.DataFrame] = {}
    win_groups: dict[str, int] = {}
    for split, data in split_data.items():
        dqn_scores, risk_scores = _score_split(
            torch=torch,
            quick_module=quick_module,
            dqn_module=dqn_module,
            oof_module=oof_module,
            data=data,
            dqn_model=dqn_model,
            risk_model=risk_model,
            risk_backend=risk_backend,
            risk_feature_names=risk_feature_names,
            selection_margin=selection_margin,
        )
        proposals[split] = _split_proposals(
            quick_module,
            data=data,
            dqn_scores=dqn_scores,
            risk_scores=risk_scores,
            selection_margin=selection_margin,
        )
        win_groups[split] = _win_group_count(data["metadata"].reset_index(drop=True))

    calibration_scores = proposals["calibration"].loc[
        proposals["calibration"]["raw_override"].astype(bool), "risk_score"
    ].astype(float)
    finite_scores = sorted(set(float(value) for value in calibration_scores.to_numpy() if math.isfinite(float(value))))
    if finite_scores:
        cutoffs = [float(min(finite_scores) - 1.0e-6)] + finite_scores + [float(max(finite_scores) + 1.0e-6)]
    else:
        cutoffs = [-math.inf, math.inf]

    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        train_metrics = _metrics_from_proposals(proposals["train"], cutoff=float(cutoff), win_groups=win_groups["train"])
        cal_metrics = _metrics_from_proposals(
            proposals["calibration"],
            cutoff=float(cutoff),
            win_groups=win_groups["calibration"],
        )
        eval_metrics = _metrics_from_proposals(proposals["eval"], cutoff=float(cutoff), win_groups=win_groups["eval"])
        rows.append(
            {
                "cutoff": float(cutoff),
                **_flatten_metrics("train", train_metrics),
                **_flatten_metrics("cal", cal_metrics),
                **_flatten_metrics("eval", eval_metrics),
            }
        )

    cal_best = {
        str(max_loss): _best_under(
            rows,
            split_prefix="cal",
            max_loss_rate=float(max_loss),
            min_override_count=min_override_count,
            min_total_delta=min_total_delta,
            max_override_rate=max_override_rate,
        )
        for max_loss in max_loss_rates
    }
    eval_oracle = {
        str(max_loss): _best_under(
            rows,
            split_prefix="eval",
            max_loss_rate=float(max_loss),
            min_override_count=min_override_count,
            min_total_delta=min_total_delta,
            max_override_rate=max_override_rate,
        )
        for max_loss in max_loss_rates
    }

    exported_artifact = None
    export_metrics = None
    if export_max_loss_rate is not None:
        selected = cal_best.get(str(float(export_max_loss_rate))) or cal_best.get(str(export_max_loss_rate))
        if selected is not None:
            exported_artifact = _export_artifact(
                source_artifact=artifact,
                output_dir=output_dir,
                cutoff=float(selected["cutoff"]),
                suffix=f"cutoff{str(export_max_loss_rate).replace('.', 'p')}",
            )
            export_metrics = dict(selected)

    summary = {
        "artifact": str(artifact),
        "selection_margin": selection_margin,
        "current_cutoff": float(risk_selector["score_cutoff"]),
        "cutoff_count": int(len(cutoffs)),
        "max_loss_rates": [float(value) for value in max_loss_rates],
        "calibration_best": cal_best,
        "eval_oracle": eval_oracle,
        "proposal_counts": {
            split: {
                "groups": int(len(frame)),
                "raw_override_count": int(frame["raw_override"].astype(bool).sum()),
            }
            for split, frame in proposals.items()
        },
        "rows": rows,
        "exported_artifact": None if exported_artifact is None else str(exported_artifact),
        "export_metrics": export_metrics,
    }
    _write_json(output_dir / "risk_cutoff_sweep_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized cutoff sweep for DQN OOF risk selector artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-loss-rate", type=float, action="append", default=None)
    parser.add_argument("--export-max-loss-rate", type=float, default=None)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--max-override-rate", type=float, default=0.35)
    args = parser.parse_args()
    max_loss_rates = list(args.max_loss_rate or [0.08, 0.10, 0.11, 0.12, 0.121, 0.125, 0.13])
    summary = run_sweep(
        run_dir=Path(args.run_dir),
        artifact=Path(args.artifact),
        output_dir=Path(args.output_dir),
        max_loss_rates=max_loss_rates,
        export_max_loss_rate=args.export_max_loss_rate,
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        max_override_rate=float(args.max_override_rate),
    )
    print(json.dumps(_json_safe({k: v for k, v in summary.items() if k != "rows"}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
