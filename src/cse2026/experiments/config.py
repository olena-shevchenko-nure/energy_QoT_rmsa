from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cse2026.data_generation.io_utils import project_root


PLACEHOLDER_STAGES = {
    "train_supervised_ranker",
}

SUPPORTED_STAGES = {
    "validate_data",
    "summarize_data",
    "topn_baseline_eval",
    "dataloader_smoke",
    "ong_solver_eval",
    "collect_ong_expert_dataset",
    "pretrain_gnn",
    "pretrain_cnn",
    "train_dqn",
    "train_dqn_online",
    "train_deeprmsa_a3c",
    "train_deeprmsa_a3c_windowed_online",
    "train_gnn_cnn_a3c_windowed_online",
    "train_xlron_graph_transformer_ppo",
    "train_top32_xlron_stabilized_ppo",
    "evaluate_policy",
    "lookahead_oracle_eval",
    "train_lookahead_override",
    "train_tree_ranker",
    "train_dagger_tree_ranker",
    *PLACEHOLDER_STAGES,
}


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def _resolve_path(value: str | Path | None, root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    stage: str
    dataset_path: Path | None
    runs_root: Path
    seed: int = 0
    device: str = "auto"
    splits: tuple[str, ...] | None = None
    batch_size: int = 64
    max_batches: int = 3
    raw: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        root: str | Path | None = None,
    ) -> "ExperimentConfig":
        repo_root = Path(root) if root is not None else project_root()
        expanded = expand_env_vars(data)
        stage = str(expanded.get("stage", ""))
        if stage not in SUPPORTED_STAGES:
            raise ValueError(f"Unsupported experiment stage: {stage}")

        experiment_name = str(expanded.get("experiment_name", "")).strip()
        if not experiment_name:
            raise ValueError("experiment_name is required")

        runs_root_value = expanded.get("runs_root") or os.environ.get("CSE2026_RUNS_ROOT") or "runs/eon"
        dataset_value = expanded.get("dataset_path")
        dataset_path = _resolve_path(dataset_value, repo_root) if dataset_value else None
        runs_root = _resolve_path(runs_root_value, repo_root)
        if runs_root is None:
            raise ValueError("runs_root is required")

        splits_value = expanded.get("splits")
        splits = tuple(str(split) for split in splits_value) if splits_value else None

        resolved = dict(expanded)
        resolved["experiment_name"] = experiment_name
        resolved["stage"] = stage
        resolved["dataset_path"] = None if dataset_path is None else str(dataset_path)
        resolved["runs_root"] = str(runs_root)
        resolved["seed"] = int(expanded.get("seed", 0))
        resolved["device"] = str(expanded.get("device", "auto"))
        if splits is not None:
            resolved["splits"] = list(splits)
        resolved["batch_size"] = int(expanded.get("batch_size", 64))
        resolved["max_batches"] = int(expanded.get("max_batches", 3))

        return cls(
            experiment_name=experiment_name,
            stage=stage,
            dataset_path=dataset_path,
            runs_root=runs_root,
            seed=int(expanded.get("seed", 0)),
            device=str(expanded.get("device", "auto")),
            splits=splits,
            batch_size=int(expanded.get("batch_size", 64)),
            max_batches=int(expanded.get("max_batches", 3)),
            raw=dict(data),
            resolved=resolved,
            config_path=None if config_path is None else Path(config_path),
        )

    @classmethod
    def from_file(cls, path: str | Path, *, root: str | Path | None = None) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")
        return cls.from_mapping(data, config_path=config_path, root=root)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_file(path)
