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
DEFAULT_ONG_LOCK = Path("third_party/optical-networking-gym.lock")

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


def _raw_bool(config: dict[str, str], key: str, default: bool) -> bool:
    value = config.get(key)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_ong_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key in {"upstream_repository", "commit", "clean_repo_expected_path"}:
            values[key] = value
    missing = {"upstream_repository", "commit"} - set(values)
    if missing:
        raise RuntimeError(f"ONG lock file is missing keys: {', '.join(sorted(missing))}")
    return values


def _run_command(command: list[str], *, dry_run: bool = False) -> None:
    print(f"ONG setup: {_command_display(command)}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {_command_display(command)}")


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {_command_display(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _ensure_ong_checkout(config: dict[str, str], *, dry_run: bool, no_install: bool) -> bool:
    if no_install or not _raw_bool(config, "ong_auto_install", True):
        return False

    raw_path = config.get("ong_source_path")
    if not raw_path:
        return False

    lock_path = _repo_path(config.get("ong_lock_file", str(DEFAULT_ONG_LOCK)))
    lock = _read_ong_lock(lock_path)
    repo_url = lock["upstream_repository"]
    commit = lock["commit"]
    ong_path = _repo_path(raw_path)

    if not ong_path.exists():
        if not dry_run:
            ong_path.parent.mkdir(parents=True, exist_ok=True)
        _run_command(["git", "clone", repo_url, str(ong_path)], dry_run=dry_run)
        _run_command(["git", "-C", str(ong_path), "checkout", commit], dry_run=dry_run)
        return True

    if not (ong_path / ".git").exists():
        return True

    current_commit = _command_output(["git", "-C", str(ong_path), "rev-parse", "HEAD"])
    if current_commit == commit:
        return True

    status = _command_output(["git", "-C", str(ong_path), "status", "--porcelain"])
    if status:
        raise RuntimeError(
            f"Existing ONG checkout is not clean: {ong_path}. "
            "Commit was not changed automatically."
        )
    _run_command(["git", "-C", str(ong_path), "fetch", "--tags", "origin"], dry_run=dry_run)
    _run_command(["git", "-C", str(ong_path), "checkout", commit], dry_run=dry_run)
    return True


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
                    "(auto-install failed or was disabled; pass --ong-source-path or check the lock file)"
                )
    return missing


def _print_reference_outputs() -> None:
    print("Reference paper outputs:", flush=True)
    for relative in REFERENCE_OUTPUTS:
        print(f"  - {relative}", flush=True)


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
        "--no-install-ong",
        action="store_true",
        help="Do not clone or checkout the pinned Optical Networking Gym source automatically.",
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
        planned_ong_install = False
        if not args.skip_ong_check:
            try:
                planned_ong_install = _ensure_ong_checkout(
                    config_values,
                    dry_run=args.dry_run,
                    no_install=args.no_install_ong,
                )
            except RuntimeError as exc:
                print(f"ONG setup failed: {exc}", file=sys.stderr, flush=True)
                return 2

        if not args.skip_validation:
            skip_ong_validation = args.skip_ong_check or (args.dry_run and planned_ong_install)
            missing = _collect_missing_paths(config_values, skip_ong_check=skip_ong_validation)
            if missing:
                print("Reproduction input check failed:", file=sys.stderr, flush=True)
                for item in missing:
                    print(f"  - {item}", file=sys.stderr, flush=True)
                print(
                    "\nSee docs/environment_reproduction.md and docs/optical_networking_gym_setup.md.",
                    file=sys.stderr,
                    flush=True,
                )
                return 2

        command = [args.python, str(_repo_path(RUNNER)), "--config", str(runtime_config)]
        if args.runner_dry_run:
            command.append("--dry-run")

        print(f"Repository root: {ROOT}", flush=True)
        print(f"Config: {runtime_config}", flush=True)
        print(f"Command: {_command_display(command)}", flush=True)
        _print_reference_outputs()

        if args.dry_run:
            return 0

        completed = subprocess.run(command, cwd=str(ROOT), env=_pythonpath_env())
        return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
