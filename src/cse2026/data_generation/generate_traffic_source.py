from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import build_checksums, ensure_dir, load_yaml, utc_timestamp, write_checksums, write_json, write_parquet
from .topology import copy_topology_files, load_topology
from .traffic import generate_requests_for_episode


def _split_counts(output: Path, splits: list[str]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split in splits:
        path = output / "traffic" / f"{split}.parquet"
        if not path.exists():
            counts[split] = {"traffic_rows": 0}
            continue
        traffic = pd.read_parquet(path)
        counts[split] = {
            "traffic_rows": int(len(traffic)),
            "episodes": int(traffic["episode_id"].nunique()) if "episode_id" in traffic else 0,
            "scenarios": sorted(str(value) for value in traffic["traffic_scenario"].dropna().unique())
            if "traffic_scenario" in traffic
            else [],
            "loads": sorted(str(value) for value in traffic["load_name"].dropna().unique()) if "load_name" in traffic else [],
        }
    return counts


def generate_traffic_source(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    output = Path(output_path)
    if output.exists():
        shutil.rmtree(output)
    for subdir in ("topology", "traffic", "reports"):
        ensure_dir(output / subdir)

    topology = load_topology(str(cfg.get("topology", "nsfnet_deeprmsa_14_22")), int(cfg.get("slots", 100)))
    copy_topology_files(topology.name, output / "topology")

    split_names = list(cfg.get("splits", {}).keys())
    for split, split_cfg in cfg["splits"].items():
        rows: list[dict[str, Any]] = []
        seeds = [int(seed) for seed in split_cfg.get("seeds", [])]
        if not seeds:
            seeds = list(range(int(split_cfg["seed_start"]), int(split_cfg["seed_stop"])))
        for seed in seeds:
            for scenario in cfg.get("traffic_scenarios", ["uniform"]):
                for load_name in cfg.get("loads", ["medium"]):
                    episode_id = f"{split}-{scenario}-{load_name}-seed{seed}"
                    rows.extend(
                        generate_requests_for_episode(
                            split=split,
                            seed=int(seed),
                            scenario=str(scenario),
                            load_name=str(load_name),
                            requests_per_seed=int(split_cfg["requests_per_seed"]),
                            cfg=cfg,
                            episode_id=episode_id,
                            node_count=topology.node_count,
                        )
                    )
        write_parquet(output / "traffic" / f"{split}.parquet", rows)
        print(f"{split}: wrote {len(rows)} traffic rows")

    manifest = {
        "dataset_name": cfg.get("dataset_name", output.name),
        "dataset_type": "traffic_source",
        "config_path": str(config_path),
        "generation_timestamp": utc_timestamp(),
        "topology": topology.name,
        "slot_total": topology.slot_total,
        "splits": _split_counts(output, split_names),
        "parameters": cfg,
    }
    checksums = build_checksums(output)
    manifest["checksums"] = checksums
    write_json(output / "manifest.json", manifest)
    write_checksums(output / "checksums.sha256", checksums)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate traffic-only EON source datasets for ONG replay.")
    parser.add_argument("--config", required=True, help="Path to YAML data-generation config.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_traffic_source(args.config, args.output)
    total_requests = sum(split.get("traffic_rows", 0) for split in manifest["splits"].values())
    print(f"Generated traffic source {manifest['dataset_name']} with {total_requests} requests at {args.output}")


if __name__ == "__main__":
    main()
