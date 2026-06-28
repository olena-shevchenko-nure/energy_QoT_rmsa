#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/evaluation/mvp80_selected_topn_p95_compare_clean.yaml")
RUNNER = Path("scripts/experiments/run_eon_experiment.py")

PATH_KEYS = (
    "dataset_path",
    "dqn_checkpoint",
    "gnn_cnn_a3c_checkpoint",
    "top32_xlron_stabilized_ppo_checkpoint",
    "lightgbm_candidate_ranker_old10_path",
    "torch_dqn_candidate_ranker_distill_old10_path",
)

DATASET_TEST_FILES = (
    "manifest.json",
    "traffic/test.parquet",
    "candidates/test.parquet",
    "candidates/test_full.parquet",
    "cnn/test_index.parquet",
    "cnn/test_tensors.npz",
    "gnn/test_graphs.npz",
    "gnn/test_routes.parquet",
    "dqn/test_transitions.parquet",
    "topology/directed_links.csv",
)

REFERENCE_OUTPUTS = (
    "results/mvp80/tables/mvp80_selected_topn_p95_comparison_20260626.csv",
    "results/mvp80/raw/mvp80_selected_topn_p95_policy_summary_20260626.csv",
    "results/mvp80/raw/mvp80_selected_topn_p95_policy_episode_metrics_20260626.csv",
    "results/mvp80/statistics/mvp80_statistical_summary_20260626.csv",
    "results/mvp80/statistics/mvp80_paired_tests_vs_energy_aware_20260626.csv",
)


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _read_top_level_yaml_scalars(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.startswith(("#", "-")):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            values[key] = value.strip("'\"")
    return values


def _format_yaml_value(value: str | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("\\", "/")
    if not text:
        return '""'
    if any(ch in text for ch in ":#[]{}*,&!|>'\"%@`"):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _write_overridden_config(source: Path, destination: Path, updates: dict[str, str | int]) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and ":" in line and not line.startswith(("#", "-")):
            key = line.split(":", 1)[0].strip()
            if key in updates:
                lines.append(f"{key}: {_format_yaml_value(updates[key])}")
                seen.add(key)
                continue
        lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}: {_format_yaml_value(value)}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _command_display(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    pieces = [str(ROOT / "src"), str(ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        pieces.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pieces)
    return env


def _collect_missing_paths(config: dict[str, str], skip_ong_check: bool) -> list[str]:
    missing: list[str] = []
    for path, label in ((_repo_path(RUNNER), "experiment runner"),):
        if not path.exists():
            missing.append(f"{label}: {path}")

    for key in PATH_KEYS:
        raw = config.get(key)
        if not raw:
            missing.append(f"config key `{key}` is absent")
            continue
        path = _repo_path(raw)
        if not path.exists():
            missing.append(f"{key}: {path}")

    dataset_raw = config.get("dataset_path")
    if dataset_raw:
        dataset_path = _repo_path(dataset_raw)
        for relative in DATASET_TEST_FILES:
            path = dataset_path / relative
            if not path.exists():
                missing.append(f"dataset file: {path}")

    if not skip_ong_check:
        ong_raw = config.get("ong_source_path")
        if not ong_raw:
            missing.append("config key `ong_source_path` is absent")
        else:
            ong_path = _repo_path(ong_raw)
            if not ong_path.exists():
                missing.append(
                    f"ong_source_path: {ong_path} "
                    "(clone the pinned Optical Networking Gym commit or pass --ong-source-path)"
                )
    return missing


def _print_reference_outputs() -> None:
    print("Reference paper outputs:")
    for relative in REFERENCE_OUTPUTS:
        print(f"  - {relative}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the MVP80 policy comparison from the CSE 2026 RMSA paper."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Evaluation config to run. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch scripts/experiments/run_eon_experiment.py.",
    )
    parser.add_argument(
        "--ong-source-path",
        help="Override the Optical Networking Gym checkout path without editing the checked-in config.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), help="Override config device.")
    parser.add_argument("--max-episodes", type=int, help="Override config max_episodes for smoke runs.")
    parser.add_argument(
        "--max-requests-per-episode",
        type=int,
        help="Override config max_requests_per_episode for smoke runs.",
    )
    parser.add_argument("--runs-root", help="Override config runs_root.")
    parser.add_argument(
        "--runner-dry-run",
        action="store_true",
        help="Call the underlying experiment runner with --dry-run to print its resolved config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the command, but do not launch the experiment.",
    )
    parser.add_argument(
        "--skip-ong-check",
        action="store_true",
        help="Do not require ong_source_path to exist during validation.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Launch without checking local dataset, model artifact, and ONG paths.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config_path = _repo_path(args.config)
    if not config_path.exists():
        print(f"Missing config: {config_path}", file=sys.stderr)
        return 2

    updates: dict[str, str | int] = {}
    if args.ong_source_path:
        updates["ong_source_path"] = args.ong_source_path
    if args.device:
        updates["device"] = args.device
    if args.max_episodes is not None:
        updates["max_episodes"] = args.max_episodes
    if args.max_requests_per_episode is not None:
        updates["max_requests_per_episode"] = args.max_requests_per_episode
    if args.runs_root:
        updates["runs_root"] = args.runs_root

    with tempfile.TemporaryDirectory(prefix="reproduce_mvp80_") as tmp_dir:
        runtime_config = config_path
        if updates:
            runtime_config = Path(tmp_dir) / "mvp80_runtime_config.yaml"
            _write_overridden_config(config_path, runtime_config, updates)

        config_values = _read_top_level_yaml_scalars(runtime_config)
        if not args.skip_validation:
            missing = _collect_missing_paths(config_values, skip_ong_check=args.skip_ong_check)
            if missing:
                print("Reproduction input check failed:", file=sys.stderr)
                for item in missing:
                    print(f"  - {item}", file=sys.stderr)
                print("\nSee docs/environment_reproduction.md and docs/optical_networking_gym_setup.md.", file=sys.stderr)
                return 2

        command = [args.python, str(_repo_path(RUNNER)), "--config", str(runtime_config)]
        if args.runner_dry_run:
            command.append("--dry-run")

        print(f"Repository root: {ROOT}")
        print(f"Config: {runtime_config}")
        print(f"Command: {_command_display(command)}")
        _print_reference_outputs()

        if args.dry_run:
            return 0

        completed = subprocess.run(command, cwd=str(ROOT), env=_pythonpath_env())
        return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
