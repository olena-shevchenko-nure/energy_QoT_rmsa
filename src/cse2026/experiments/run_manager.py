from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig
from .system_info import collect_env_report, git_info, write_report


@dataclass(frozen=True)
class RunContext:
    config: ExperimentConfig
    run_dir: Path
    git_info: dict[str, Any]
    env_report: dict[str, Any]
    resumed: bool = False


class RunManager:
    def __init__(self, config: ExperimentConfig, *, root: str | Path | None = None) -> None:
        self.config = config
        self.root = Path(root) if root is not None else Path.cwd()

    def create(self, *, resume: str | None = None) -> RunContext:
        run_dir = self._resolve_resume(resume) if resume else self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=bool(resume))
        info = git_info(self.root)
        env_report = collect_env_report(
            dataset=self.config.dataset_path,
            runs_root=self.config.runs_root,
            root=self.root,
        )
        self._write_yaml(run_dir / "config.yaml", self.config.raw)
        self._write_yaml(run_dir / "config.resolved.yaml", self.config.resolved)
        write_report(run_dir / "git_info.json", info)
        write_report(run_dir / "env_report.json", env_report)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        return RunContext(
            config=self.config,
            run_dir=run_dir,
            git_info=info,
            env_report=env_report,
            resumed=bool(resume),
        )

    def _new_run_dir(self) -> Path:
        short_sha = git_info(self.root).get("short_commit") or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.config.runs_root / self.config.experiment_name / f"{timestamp}_{short_sha}"
        if not base.exists():
            return base
        for index in range(1, 100):
            candidate = base.with_name(f"{base.name}_{index:02d}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(f"Could not create a unique run directory under {base.parent}")

    def _resolve_resume(self, resume: str | None) -> Path:
        if not resume:
            raise ValueError("resume value is required")
        if resume == "latest":
            parent = self.config.runs_root / self.config.experiment_name
            if not parent.exists():
                raise FileNotFoundError(f"No runs found for resume latest: {parent}")
            candidates = sorted(path for path in parent.iterdir() if path.is_dir())
            if not candidates:
                raise FileNotFoundError(f"No runs found for resume latest: {parent}")
            return candidates[-1]
        path = Path(resume)
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")

    @staticmethod
    def copy_artifact(src: str | Path, dst_dir: str | Path) -> Path:
        source = Path(src)
        target_dir = Path(dst_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        return target
