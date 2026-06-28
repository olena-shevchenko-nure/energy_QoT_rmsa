from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cse2026.data_generation.summarize import summarize_dataset
from cse2026.data_generation.validation import validate_dataset

from ..config import ExperimentConfig, expand_env_vars


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_data(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("validate_data requires dataset_path")
    report = validate_dataset(_dataset_argument(config))
    run_path = Path(run_dir)
    _write_json(run_path / "reports" / "validation_report.json", report)
    checks = report.get("checks", [])
    metrics = {
        "stage": "validate_data",
        "validation_passed": bool(report.get("passed", False)),
        "number_of_checks": int(len(checks)),
        "failed_checks": int(sum(1 for check in checks if not check.get("passed", False))),
        "splits": report.get("summary", {}),
    }
    _write_json(run_path / "metrics.json", metrics)
    if not metrics["validation_passed"]:
        raise RuntimeError("Dataset validation failed")
    return metrics


def _dataset_argument(config: ExperimentConfig) -> str | Path:
    raw_value = config.raw.get("dataset_path")
    expanded = expand_env_vars(raw_value)
    if isinstance(expanded, str) and "$" not in expanded and not Path(expanded).is_absolute():
        return expanded
    return config.dataset_path


def summarize_data(config: ExperimentConfig, run_dir: str | Path) -> dict[str, Any]:
    if config.dataset_path is None:
        raise ValueError("summarize_data requires dataset_path")
    summary = summarize_dataset(config.dataset_path)
    run_path = Path(run_dir)
    summary_path = run_path / "reports" / "dataset_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    dataset_summary = config.dataset_path / "reports" / "dataset_summary.md"
    if dataset_summary.exists() and dataset_summary.resolve() != summary_path.resolve():
        shutil.copy2(dataset_summary, summary_path)
    metrics = {
        "stage": "summarize_data",
        "summary_path": str(summary_path),
        "dataset_path": str(config.dataset_path),
    }
    _write_json(run_path / "metrics.json", metrics)
    return metrics
