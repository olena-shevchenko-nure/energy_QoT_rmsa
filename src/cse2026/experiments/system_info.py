from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from cse2026.data_generation.io_utils import project_root


def _run_command(args: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)
    except OSError as exc:
        return 127, str(exc)
    return int(proc.returncode), (proc.stdout + proc.stderr).strip()


def git_info(root: str | Path | None = None) -> dict[str, Any]:
    repo = Path(root) if root is not None else project_root()
    branch_code, branch = _run_command(["git", "branch", "--show-current"], cwd=repo)
    commit_code, commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo)
    short_code, short = _run_command(["git", "rev-parse", "--short", "HEAD"], cwd=repo)
    status_code, status = _run_command(["git", "status", "--short"], cwd=repo)
    return {
        "branch": branch if branch_code == 0 else "",
        "commit": commit if commit_code == 0 else "",
        "short_commit": short if short_code == 0 else "unknown",
        "dirty": bool(status.strip()) if status_code == 0 else None,
        "status_short": status.splitlines() if status_code == 0 else [],
    }


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _disk_usage(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = _nearest_existing(Path(path))
    try:
        usage = shutil.disk_usage(resolved)
    except OSError:
        return None
    return {
        "path": str(Path(path)),
        "measured_path": str(resolved),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _memory_info() -> dict[str, Any] | None:
    try:
        import psutil
    except ImportError:
        return None
    mem = psutil.virtual_memory()
    return {
        "total_bytes": int(mem.total),
        "available_bytes": int(mem.available),
        "percent": float(mem.percent),
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def _nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    code, output = _run_command(
        [
            exe,
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ]
    )
    return {"available": code == 0, "returncode": code, "summary": output.splitlines()}


def collect_env_report(
    *,
    dataset: str | Path | None = None,
    data_root: str | Path | None = None,
    runs_root: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(root) if root is not None else project_root()
    dataset_path = Path(dataset) if dataset is not None else None
    manifest = dataset_path / "manifest.json" if dataset_path is not None else None
    validation = dataset_path / "reports" / "validation_report.json" if dataset_path is not None else None
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "memory": _memory_info(),
        "disk": {
            "project_root": _disk_usage(repo),
            "data_root": _disk_usage(data_root or os.environ.get("CSE2026_DATA_ROOT") or repo / "data"),
            "runs_root": _disk_usage(runs_root or os.environ.get("CSE2026_RUNS_ROOT") or repo / "runs"),
        },
        "git": git_info(repo),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "CSE2026_PROJECT_ROOT": os.environ.get("CSE2026_PROJECT_ROOT", ""),
            "CSE2026_DATA_ROOT": os.environ.get("CSE2026_DATA_ROOT", ""),
            "CSE2026_RUNS_ROOT": os.environ.get("CSE2026_RUNS_ROOT", ""),
            "CSE2026_DEVICE": os.environ.get("CSE2026_DEVICE", ""),
        },
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "dataset": None
        if dataset_path is None
        else {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "manifest_exists": manifest.exists(),
            "validation_report_exists": validation.exists(),
        },
    }


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
