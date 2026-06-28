from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_quick_module() -> Any:
    module_path = Path(__file__).with_name("quick_exception_ranker_ab.py")
    spec = importlib.util.spec_from_file_location("quick_exception_ranker_ab", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
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


def _nonbase_mask(data: dict[str, Any]) -> np.ndarray:
    metadata = data["metadata"].reset_index(drop=True)
    return (metadata["candidate_index"].astype(int) != metadata["base_index"].astype(int)).to_numpy()


def _predict_torch_model(torch: Any, model: Any, x: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(len(x)), int(batch_size)):
            tensor = torch.as_tensor(x[start : start + int(batch_size)].astype(np.float32), dtype=torch.float32)
            values.append(model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32))
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def _load_torch_models(torch: Any, artifact_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    gate = dict(meta["advantage_gate"])
    parent = artifact_path.parent
    paths = {
        "ranker": parent / str(meta["model_path"]),
        "win": parent / str(gate["win_model_path"]),
        "loss": parent / str(gate["loss_model_path"]),
        "delta": parent / str(gate["delta_model_path"]),
    }
    return {name: torch.jit.load(str(path), map_location="cpu").eval() for name, path in paths.items()}


def _full_head_preds(torch: Any, models: dict[str, Any], data: dict[str, Any]) -> dict[str, np.ndarray]:
    mask = _nonbase_mask(data)
    positions = np.flatnonzero(mask)
    result = {name: np.zeros((len(data["metadata"]),), dtype=np.float32) for name in ("win", "loss", "delta")}
    result["loss"].fill(1.0)
    result["delta"].fill(-1.0)
    x = data["x"][mask].astype(np.float32)
    for name in ("win", "loss", "delta"):
        result[name][positions] = _predict_torch_model(torch, models[name], x)
    return result


def _feature_condition_mask(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    condition: dict[str, Any],
) -> np.ndarray:
    index = {str(name): int(position) for position, name in enumerate(feature_names)}
    feature = str(condition["feature"])
    if feature not in index:
        raise ValueError(f"Unknown context gate feature: {feature}")
    values = np.asarray(features[:, index[feature]], dtype=np.float32)
    threshold = float(condition["value"])
    op = str(condition.get("op", "ge")).strip().lower()
    if op in {"ge", ">="}:
        return values >= threshold
    if op in {"gt", ">"}:
        return values > threshold
    if op in {"le", "<="}:
        return values <= threshold
    if op in {"lt", "<"}:
        return values < threshold
    if op in {"eq", "=="}:
        return np.isclose(values, threshold)
    raise ValueError(f"Unsupported context gate condition op: {op}")


def _context_rule_mask(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    rule: dict[str, Any],
) -> np.ndarray:
    conditions = list(rule.get("conditions") or [])
    if not conditions:
        return np.zeros((features.shape[0],), dtype=bool)
    mask = np.ones((features.shape[0],), dtype=bool)
    for condition in conditions:
        mask &= _feature_condition_mask(features, feature_names, dict(condition))
    return mask


def _context_required_gate_score(
    features: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    thresholds: dict[str, Any],
) -> np.ndarray:
    default = float(thresholds.get("min_gate_score", -np.inf))
    required = np.full((features.shape[0],), default, dtype=np.float32)
    for rule in thresholds.get("context_gate_rules") or ():
        mask = _context_rule_mask(features, feature_names, dict(rule))
        if mask.any():
            required[mask] = np.maximum(required[mask], float(rule["min_gate_score"]))
    return required


def _top_rows_for_thresholds(
    module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    eligible = module._safety_mask(data, enabled=True)
    win_prob = np.asarray(preds["win"], dtype=np.float32)
    loss_prob = np.asarray(preds["loss"], dtype=np.float32)
    delta_pred = np.asarray(preds["delta"], dtype=np.float32)
    gate_score = delta_pred + win_prob - 2.0 * loss_prob
    required_gate_score = _context_required_gate_score(data["x"], data["feature_names"], thresholds)
    passed = (
        eligible
        & (win_prob >= float(thresholds["min_win_prob"]))
        & (loss_prob <= float(thresholds["max_loss_prob"]))
        & (delta_pred >= float(thresholds["min_delta_pred"]))
        & (gate_score >= required_gate_score)
    )
    rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        group_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        passed_indices = group_indices[passed[group_indices]]
        if passed_indices.size == 0:
            continue
        best = int(
            min(
                (int(index) for index in passed_indices),
                key=lambda index: (-float(gate_score[index]), int(metadata.at[index, "candidate_index"])),
            )
        )
        row = metadata.loc[best]
        rows.append(
            module._override_row(
                row=row,
                group=group,
                row_index=best,
                win_prob=float(win_prob[best]),
                loss_prob=float(loss_prob[best]),
                delta_pred=float(delta_pred[best]),
                selector_score=float(gate_score[best]),
            )
        )
    rows.sort(key=lambda row: (-float(row["selector_score"]), int(row["candidate_index"]), int(row["group_id"])))
    return rows


def _selected_override_rows_for_thresholds(
    module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    override_rows = _top_rows_for_thresholds(module, data, preds, thresholds)
    selected = {int(row["group_id"]): dict(row) for row in override_rows}
    return [selected[int(group["group_id"].iloc[0])] for _, group in metadata.groupby("group_id", sort=False) if int(group["group_id"].iloc[0]) in selected]


def _rows_with_overrides(module: Any, metadata: pd.DataFrame, overrides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {int(row["group_id"]): dict(row) for row in overrides}
    rows: list[dict[str, Any]] = []
    for _, group in metadata.groupby("group_id", sort=False):
        gid = int(group["group_id"].iloc[0])
        rows.append(selected.get(gid, module._no_override_row(group)))
    return rows


def _prefix_metrics(module: Any, metadata: pd.DataFrame, overrides: list[dict[str, Any]], count: int) -> dict[str, Any]:
    rows = _rows_with_overrides(module, metadata, overrides[: int(count)])
    return module._selection_metrics(rows, metadata)


def _metrics_for_thresholds(
    module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    metadata = data["metadata"].reset_index(drop=True)
    rows = _selected_override_rows_for_thresholds(module, data, preds, thresholds)
    return module._selection_metrics(_rows_with_overrides(module, metadata, rows), metadata)


def _tune_budget(
    module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    *,
    override_budget: float,
    min_budget_fraction: float,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    metadata = data["metadata"].reset_index(drop=True)
    group_count = int(metadata["group_id"].nunique())
    max_count = max(int(min_override_count), int(np.floor(float(override_budget) * max(group_count, 1))))
    min_count = max(
        int(min_override_count),
        int(np.floor(float(override_budget) * float(min_budget_fraction) * max(group_count, 1))),
    )
    best_key: tuple[float, float, float, float] | None = None
    best_thresholds: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] | None = None
    best_count = 0

    for thresholds in module._threshold_grid():
        rows = _top_rows_for_thresholds(module, data, preds, thresholds)
        if len(rows) < int(min_count):
            continue
        accepted = np.asarray([float(row["accepted_delta_vs_base"]) for row in rows], dtype=np.float64)
        reward = np.asarray([float(row["reward_delta_vs_base"]) for row in rows], dtype=np.float64)
        loss = (accepted < 0.0).astype(np.float64)
        accepted_cum = np.cumsum(accepted)
        reward_cum = np.cumsum(reward)
        loss_cum = np.cumsum(loss)
        limit = min(max_count, len(rows))
        for count in range(int(min_count), int(limit) + 1):
            total_delta = float(accepted_cum[count - 1])
            if total_delta <= float(min_total_delta):
                continue
            loss_rate = float(loss_cum[count - 1] / max(count, 1))
            if loss_rate > float(max_loss_rate):
                continue
            key = (total_delta, float(reward_cum[count - 1] / max(count, 1)), -loss_rate, float(count))
            if best_key is None or key > best_key:
                best_key = key
                best_thresholds = dict(thresholds)
                best_rows = rows
                best_count = int(count)
    if best_thresholds is None or best_rows is None:
        empty_rows = _rows_with_overrides(module, metadata, [])
        metrics = module._selection_metrics(empty_rows, metadata)
        thresholds = {
            "min_win_prob": 1.000001,
            "max_loss_prob": -0.000001,
            "min_delta_pred": 1.0e9,
            "min_gate_score": 1.0e9,
            "override_budget": float(override_budget),
            "min_budget_fraction": float(min_budget_fraction),
            "tune_found_feasible": 0.0,
        }
        return thresholds, metrics | {"tune_found_feasible": 0}

    cutoff = float(best_rows[best_count - 1]["selector_score"])
    best_thresholds["min_gate_score"] = cutoff
    best_thresholds["override_budget"] = float(override_budget)
    best_thresholds["min_budget_fraction"] = float(min_budget_fraction)
    best_thresholds["tune_found_feasible"] = 1.0
    best_metrics = _prefix_metrics(module, metadata, best_rows, best_count)
    best_metrics["tune_found_feasible"] = 1
    best_metrics["override_budget"] = float(override_budget)
    best_metrics["min_budget_fraction"] = float(min_budget_fraction)
    best_metrics["min_gate_score"] = cutoff
    return best_thresholds, best_metrics


def _candidate_condition_values(values: np.ndarray, *, direction: str) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return []
    quantiles = (0.60, 0.75, 0.90) if direction == "high" else (0.10, 0.25, 0.40)
    result: list[float] = []
    for quantile in quantiles:
        value = float(np.quantile(finite, float(quantile)))
        if not any(abs(value - old) < 1.0e-6 for old in result):
            result.append(value)
    return result


def _build_context_rule_candidates(
    data: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    *,
    min_match_count: int,
    max_rule_candidates: int,
) -> list[dict[str, Any]]:
    if not selected_rows:
        return []
    feature_names = list(data["feature_names"])
    feature_index = {name: int(position) for position, name in enumerate(feature_names)}
    selected_positions = np.asarray(
        [int(row["row_index"]) for row in selected_rows if row.get("row_index") is not None],
        dtype=np.int64,
    )
    if selected_positions.size == 0:
        return []
    features = np.asarray(data["x"], dtype=np.float32)
    selected_features = features[selected_positions]
    specs = [
        ("global_load", "ge", "high"),
        ("global_fragmentation", "ge", "high"),
        ("valid_candidates_norm", "le", "low"),
        ("pool_size_norm", "le", "low"),
        ("energy_rank_delta", "le", "low"),
        ("fragmentation_delta", "le", "low"),
        ("qot_margin_delta", "le", "low"),
    ]
    single_rules: list[dict[str, Any]] = []
    for feature, op, direction in specs:
        if feature not in feature_index:
            continue
        for value in _candidate_condition_values(selected_features[:, feature_index[feature]], direction=direction):
            rule = {"conditions": [{"feature": feature, "op": op, "value": float(value)}]}
            match_count = int(_context_rule_mask(features, feature_names, rule)[selected_positions].sum())
            if match_count >= int(min_match_count):
                rule["match_count"] = match_count
                single_rules.append(rule)

    pair_names = {
        ("global_load", "valid_candidates_norm"),
        ("global_load", "pool_size_norm"),
        ("global_load", "energy_rank_delta"),
        ("global_load", "fragmentation_delta"),
        ("global_fragmentation", "valid_candidates_norm"),
        ("valid_candidates_norm", "energy_rank_delta"),
    }
    rules = list(single_rules)
    for left in single_rules:
        left_feature = str(left["conditions"][0]["feature"])
        for right in single_rules:
            right_feature = str(right["conditions"][0]["feature"])
            if left_feature == right_feature:
                continue
            if (left_feature, right_feature) not in pair_names and (right_feature, left_feature) not in pair_names:
                continue
            conditions = [dict(left["conditions"][0]), dict(right["conditions"][0])]
            key = tuple((condition["feature"], condition["op"], round(float(condition["value"]), 6)) for condition in conditions)
            rule = {"conditions": conditions}
            match_count = int(_context_rule_mask(features, feature_names, rule)[selected_positions].sum())
            if match_count >= int(min_match_count):
                rule["match_count"] = match_count
                rule["key"] = key
                rules.append(rule)

    unique: dict[tuple[tuple[str, str, float], ...], dict[str, Any]] = {}
    for rule in rules:
        key = tuple(
            (str(condition["feature"]), str(condition.get("op", "ge")), round(float(condition["value"]), 6))
            for condition in rule["conditions"]
        )
        existing = unique.get(key)
        if existing is None or int(rule.get("match_count", 0)) > int(existing.get("match_count", 0)):
            unique[key] = rule
    ranked = sorted(
        unique.values(),
        key=lambda rule: (
            -int(rule.get("match_count", 0)),
            len(list(rule.get("conditions") or [])),
            json.dumps(rule.get("conditions") or [], sort_keys=True),
        ),
    )
    return ranked[: max(1, int(max_rule_candidates))]


def _tune_context_gate(
    module: Any,
    data: dict[str, Any],
    preds: dict[str, np.ndarray],
    thresholds: dict[str, Any],
    *,
    max_rules: int,
    min_rule_match_count: int,
    max_rule_candidates: int,
    min_override_fraction: float,
    max_loss_rate: float,
    min_total_delta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_thresholds = dict(thresholds)
    current_thresholds["context_gate_rules"] = list(current_thresholds.get("context_gate_rules") or [])
    current_metrics = _metrics_for_thresholds(module, data, preds, current_thresholds)
    baseline_override_count = int(current_metrics.get("override_count", 0))
    if baseline_override_count <= 0 or int(max_rules) <= 0:
        current_metrics["context_tune_found_feasible"] = 0
        return current_thresholds, current_metrics

    minimum_override_count = max(
        int(min_rule_match_count),
        int(round(float(min_override_fraction) * float(baseline_override_count))),
    )
    searched = 0
    added_rules: list[dict[str, Any]] = []
    for rule_index in range(int(max_rules)):
        selected_rows = _selected_override_rows_for_thresholds(module, data, preds, current_thresholds)
        candidates = _build_context_rule_candidates(
            data,
            selected_rows,
            min_match_count=int(min_rule_match_count),
            max_rule_candidates=int(max_rule_candidates),
        )
        best_key: tuple[float, float, float, float] | None = None
        best_thresholds: dict[str, Any] | None = None
        best_metrics: dict[str, Any] | None = None
        base_min = float(current_thresholds.get("min_gate_score", -np.inf))
        selected_scores = np.asarray([float(row["selector_score"]) for row in selected_rows], dtype=np.float32)
        if selected_scores.size == 0:
            break
        default_cutoffs = [float(np.quantile(selected_scores, quantile)) for quantile in (0.60, 0.75, 0.90)]
        for rule in candidates:
            mask = _context_rule_mask(data["x"], data["feature_names"], rule)
            matched_scores = selected_scores[
                np.asarray([mask[int(row["row_index"])] for row in selected_rows if row.get("row_index") is not None], dtype=bool)
            ]
            if matched_scores.size < int(min_rule_match_count):
                continue
            cutoffs = sorted(
                {
                    float(value)
                    for value in [
                        *default_cutoffs,
                        *[float(np.quantile(matched_scores, quantile)) for quantile in (0.40, 0.60, 0.80)],
                    ]
                    if float(value) > base_min + 1.0e-6
                }
            )
            for cutoff in cutoffs:
                searched += 1
                candidate_rule = {
                    "name": f"context_rule_{rule_index + 1}",
                    "min_gate_score": float(cutoff),
                    "conditions": [dict(condition) for condition in rule["conditions"]],
                    "calibration_match_count": int(rule.get("match_count", 0)),
                }
                candidate_thresholds = dict(current_thresholds)
                candidate_thresholds["context_gate_rules"] = [
                    *list(current_thresholds.get("context_gate_rules") or []),
                    candidate_rule,
                ]
                metrics = _metrics_for_thresholds(module, data, preds, candidate_thresholds)
                override_count = int(metrics.get("override_count", 0))
                if override_count < int(minimum_override_count):
                    continue
                loss_rate = metrics.get("selected_loss_rate_when_overridden")
                if loss_rate is not None and float(loss_rate) > float(max_loss_rate):
                    continue
                total_delta = float(metrics.get("total_selected_accepted_delta_vs_base", 0.0))
                if total_delta <= float(min_total_delta):
                    continue
                key = (
                    total_delta,
                    float(metrics.get("mean_selected_reward_delta_vs_base", 0.0)),
                    0.0 if loss_rate is None else -float(loss_rate),
                    -float(override_count),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = candidate_thresholds
                    best_metrics = dict(metrics)
                    best_metrics["context_candidate_rule"] = candidate_rule
        if best_thresholds is None or best_metrics is None:
            break
        if float(best_metrics.get("total_selected_accepted_delta_vs_base", 0.0)) <= float(
            current_metrics.get("total_selected_accepted_delta_vs_base", 0.0)
        ):
            break
        current_thresholds = best_thresholds
        current_metrics = best_metrics
        added_rules.append(dict(best_metrics["context_candidate_rule"]))

    current_metrics["context_tune_found_feasible"] = 1 if added_rules else 0
    current_metrics["context_rules_added"] = len(added_rules)
    current_metrics["context_rules_searched"] = int(searched)
    current_metrics["context_min_override_fraction"] = float(min_override_fraction)
    current_metrics["context_minimum_override_count"] = int(minimum_override_count)
    return current_thresholds, current_metrics


def _copy_model_files(source_artifact: Path, output_dir: Path, meta: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gate = dict(meta["advantage_gate"])
    names = {
        "model_path": str(meta["model_path"]),
        "win_model_path": str(gate["win_model_path"]),
        "loss_model_path": str(gate["loss_model_path"]),
        "delta_model_path": str(gate["delta_model_path"]),
    }
    copied: dict[str, str] = {}
    for key, name in names.items():
        src = source_artifact.parent / name
        dst = output_dir / Path(name).name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[key] = dst.name
    return copied


def calibrate(
    *,
    run_dir: Path,
    artifact_path: Path,
    output_dir: Path,
    budgets: list[float],
    min_budget_fraction: float,
    top_k: int,
    threshold_fraction: float,
    max_loss_rate: float,
    min_override_count: int,
    min_total_delta: float,
    context_gate_search: bool,
    context_max_rules: int,
    context_min_rule_match_count: int,
    context_max_rule_candidates: int,
    context_min_override_fraction: float,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    module = _load_quick_module()
    meta = json.loads(artifact_path.read_text(encoding="utf-8"))
    original_train = module._load_split(run_dir, "train")
    original_eval = module._load_split(run_dir, "eval")
    train_pool = module._add_runtime_features(module._filter_small_pool(original_train, top_k=top_k))
    eval_pool = module._add_runtime_features(module._filter_small_pool(original_eval, top_k=top_k))
    _, threshold_val = module._split_train_threshold(train_pool, threshold_fraction=threshold_fraction, seed=seed)
    models = _load_torch_models(torch, artifact_path, meta)
    threshold_heads = _full_head_preds(torch, models, threshold_val)
    eval_heads = _full_head_preds(torch, models, eval_pool)
    copied_paths = _copy_model_files(artifact_path, output_dir, meta)

    variants: dict[str, Any] = {}
    for budget in budgets:
        thresholds, threshold_metrics = _tune_budget(
            module,
            threshold_val,
            threshold_heads,
            override_budget=float(budget),
            min_budget_fraction=float(min_budget_fraction),
            max_loss_rate=float(max_loss_rate),
            min_override_count=int(min_override_count),
            min_total_delta=float(min_total_delta),
        )
        base_threshold_metrics = dict(threshold_metrics)
        if bool(context_gate_search) and float(thresholds.get("tune_found_feasible", 0.0)) > 0.5:
            thresholds, threshold_metrics = _tune_context_gate(
                module,
                threshold_val,
                threshold_heads,
                thresholds,
                max_rules=int(context_max_rules),
                min_rule_match_count=int(context_min_rule_match_count),
                max_rule_candidates=int(context_max_rule_candidates),
                min_override_fraction=float(context_min_override_fraction),
                max_loss_rate=float(max_loss_rate),
                min_total_delta=float(min_total_delta),
            )
        eval_rows = _top_rows_for_thresholds(module, eval_pool, eval_heads, thresholds)
        eval_rows = [row for row in eval_rows if float(row["selector_score"]) >= float(thresholds["min_gate_score"])]
        eval_metrics = module._selection_metrics(
            _rows_with_overrides(module, eval_pool["metadata"].reset_index(drop=True), eval_rows),
            eval_pool["metadata"].reset_index(drop=True),
        )

        pct = int(round(float(budget) * 100.0))
        suffix = "_pressure" if bool(context_gate_search) else ""
        artifact_name = f"torch_dqn_distill_old10_orate{pct}{suffix}_tree_ranker.json"
        calibrated_meta = dict(meta)
        calibrated_meta["model_path"] = copied_paths["model_path"]
        gate = dict(calibrated_meta["advantage_gate"])
        gate.update(
            {
                "win_model_path": copied_paths["win_model_path"],
                "loss_model_path": copied_paths["loss_model_path"],
                "delta_model_path": copied_paths["delta_model_path"],
                **thresholds,
            }
        )
        calibrated_meta["advantage_gate"] = gate
        calibrated_meta["training"] = {
            **dict(meta.get("training") or {}),
            "override_rate_calibration": {
                "source_artifact": str(artifact_path),
                "override_budget": float(budget),
                "min_budget_fraction": float(min_budget_fraction),
                "threshold_fraction": float(threshold_fraction),
                "max_loss_rate": float(max_loss_rate),
                "min_override_count": int(min_override_count),
                "min_total_delta": float(min_total_delta),
                "context_gate_search": bool(context_gate_search),
                "context_max_rules": int(context_max_rules),
                "context_min_rule_match_count": int(context_min_rule_match_count),
                "context_max_rule_candidates": int(context_max_rule_candidates),
                "context_min_override_fraction": float(context_min_override_fraction),
                "seed": int(seed),
            },
        }
        artifact_output = output_dir / artifact_name
        _write_json(artifact_output, calibrated_meta)
        variant_name = f"orate{pct}{suffix}"
        variants[variant_name] = {
            "artifact_path": str(artifact_output),
            "thresholds": thresholds,
            "threshold_val_base": base_threshold_metrics,
            "threshold_val": threshold_metrics,
            "eval": eval_metrics,
        }

    summary = {
        "run_dir": str(run_dir),
        "source_artifact": str(artifact_path),
        "output_dir": str(output_dir),
        "budgets": budgets,
        "top_k": int(top_k),
        "min_budget_fraction": float(min_budget_fraction),
        "threshold_fraction": float(threshold_fraction),
        "max_loss_rate": float(max_loss_rate),
        "context_gate_search": bool(context_gate_search),
        "context_max_rules": int(context_max_rules),
        "context_min_rule_match_count": int(context_min_rule_match_count),
        "context_max_rule_candidates": int(context_max_rule_candidates),
        "context_min_override_fraction": float(context_min_override_fraction),
        "variants": variants,
    }
    _write_json(output_dir / "torch_dqn_override_rate_calibration_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate DQN exception policy by override-rate budget.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--min-budget-fraction", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--threshold-fraction", type=float, default=0.1)
    parser.add_argument("--max-loss-rate", type=float, default=0.10)
    parser.add_argument("--min-override-count", type=int, default=10)
    parser.add_argument("--min-total-delta", type=float, default=1.0)
    parser.add_argument("--context-gate-search", action="store_true")
    parser.add_argument("--context-max-rules", type=int, default=1)
    parser.add_argument("--context-min-rule-match-count", type=int, default=25)
    parser.add_argument("--context-max-rule-candidates", type=int, default=120)
    parser.add_argument("--context-min-override-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()
    budgets = [float(item.strip()) for item in str(args.budgets).split(",") if item.strip()]
    summary = calibrate(
        run_dir=Path(args.run_dir),
        artifact_path=Path(args.artifact),
        output_dir=Path(args.output_dir),
        budgets=budgets,
        min_budget_fraction=float(args.min_budget_fraction),
        top_k=int(args.top_k),
        threshold_fraction=float(args.threshold_fraction),
        max_loss_rate=float(args.max_loss_rate),
        min_override_count=int(args.min_override_count),
        min_total_delta=float(args.min_total_delta),
        context_gate_search=bool(args.context_gate_search),
        context_max_rules=int(args.context_max_rules),
        context_min_rule_match_count=int(args.context_min_rule_match_count),
        context_max_rule_candidates=int(args.context_max_rule_candidates),
        context_min_override_fraction=float(args.context_min_override_fraction),
        seed=int(args.seed),
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
