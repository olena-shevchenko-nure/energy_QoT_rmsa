from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .generate import generate_dataset
from .io_utils import load_yaml
from .validation import validate_dataset


def estimate_config_size(cfg: dict[str, Any]) -> int:
    scenarios = len(cfg.get("traffic_scenarios", []))
    loads = len(cfg.get("loads", []))
    total = 0
    for split_cfg in cfg.get("splits", {}).values():
        if "seeds" in split_cfg:
            seeds = len(split_cfg["seeds"])
        else:
            seeds = int(split_cfg["seed_stop"]) - int(split_cfg["seed_start"])
        total += seeds * scenarios * loads * int(split_cfg["requests_per_seed"])
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Estimate load profiles by regenerating small KSP-style probes.")
    parser.add_argument("--config", required=True, help="Base YAML config.")
    parser.add_argument("--output", default="/tmp/eon_calibration_probe", help="Probe output directory.")
    parser.add_argument("--max-requests", type=int, default=2400, help="Maximum probe requests.")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    print("Configured load profiles:")
    for name, value in cfg.get("load_profiles", {}).items():
        print(f"- {name}: lambda={value}")
    print("")
    print("Targets:")
    print("- low: < 3% blocking")
    print("- medium: 5-10% blocking")
    print("- high: 10-25% blocking")
    print("- overload: > 25% blocking")
    print("")
    total = estimate_config_size(cfg)
    if total > args.max_requests:
        print(f"Base config has {total} requests; calibration script is intentionally conservative.")
        print("Create a smaller probe config by reducing seeds/scenarios/loads, then rerun this command.")
        return
    manifest = generate_dataset(args.config, args.output)
    report = validate_dataset(args.output)
    print(f"Generated probe dataset {manifest['dataset_name']} at {Path(args.output)}")
    for split, summary in report["summary"].items():
        print(f"{split}: blocking_rate={summary['blocking_rate']:.4f}")


if __name__ == "__main__":
    main()

