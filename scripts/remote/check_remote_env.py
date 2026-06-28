#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cse2026.experiments.system_info import collect_env_report, write_report


def _run_validator(dataset: Path) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "scripts/data/validate_eon_data.py", "--dataset", str(dataset)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check a remote CSE 2026 experiment environment.")
    parser.add_argument("--dataset", help="Dataset root to inspect.")
    parser.add_argument("--runs-root", default=None, help="Root where env check reports are written.")
    parser.add_argument("--run-validation", action="store_true", help="Run the EON data validator for --dataset.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset = Path(args.dataset) if args.dataset else None
    runs_root = Path(args.runs_root) if args.runs_root else ROOT / "runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = runs_root / "env_checks" / timestamp
    report = collect_env_report(dataset=dataset, runs_root=runs_root, root=ROOT)
    if args.run_validation and dataset is not None:
        report["validation_command"] = _run_validator(dataset)
    write_report(out_dir / "env_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"ENV_REPORT={out_dir / 'env_report.json'}")
    if dataset is not None and not dataset.exists():
        return 1
    if args.run_validation and report.get("validation_command", {}).get("returncode") != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
