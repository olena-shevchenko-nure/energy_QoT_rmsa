from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import read_json


def summarize_dataset(dataset_path: str | Path) -> str:
    dataset = Path(dataset_path)
    manifest = read_json(dataset / "manifest.json")
    lines: list[str] = []
    lines.append(f"# Dataset Summary: {manifest['dataset_name']}")
    lines.append("")
    lines.append("## Topology")
    lines.append("")
    lines.append(f"- Topology: `{manifest['topology']}`")
    lines.append(f"- Slots per directed link: {manifest['slot_total']}")
    lines.append(f"- K routes: {manifest['k_routes']}")
    lines.append(f"- Top-N: {manifest['n_max']}")
    lines.append("")
    lines.append("## Split Counts")
    lines.append("")
    lines.append("| split | requests | blocking rate | full candidates | avg feasible before Top-N | avg candidate_mask sum | CNN tensors | GNN samples | DQN transitions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for split in manifest["splits"]:
        traffic = pd.read_parquet(dataset / "traffic" / f"{split}.parquet")
        candidates = pd.read_parquet(dataset / "candidates" / f"{split}.parquet")
        candidates_full = pd.read_parquet(dataset / "candidates" / f"{split}_full.parquet")
        cnn_index = pd.read_parquet(dataset / "cnn" / f"{split}_index.parquet")
        dqn = pd.read_parquet(dataset / "dqn" / f"{split}_transitions.parquet")
        gnn_shape = manifest["splits"][split].get("gnn_graph_shape", [0])
        avg_mask = float(candidates.groupby(["episode_id", "request_id"])["candidate_mask"].sum().mean()) if len(candidates) else 0.0
        avg_feasible = float(traffic["num_feasible"].mean()) if len(traffic) else 0.0
        blocking_rate = float(dqn["blocked"].mean()) if len(dqn) else 0.0
        lines.append(
            f"| {split} | {len(traffic)} | {blocking_rate:.4f} | {len(candidates_full)} | "
            f"{avg_feasible:.2f} | {avg_mask:.2f} | {len(cnn_index)} | {gnn_shape[0]} | {len(dqn)} |"
        )

    lines.append("")
    lines.append("## Candidate Statistics")
    lines.append("")
    all_candidates = []
    for split in manifest["splits"]:
        frame = pd.read_parquet(dataset / "candidates" / f"{split}.parquet")
        all_candidates.append(frame[frame["candidate_mask"] == 1])
    real = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    if len(real):
        lines.append(f"- Real Top-N candidates: {len(real)}")
        lines.append(f"- Mean route length (km): {real['route_length_km'].mean():.2f}")
        lines.append(f"- Mean required slots: {real['required_slots'].mean():.2f}")
        lines.append(f"- Mean energy increment (W): {real['energy_increment'].mean():.2f}")
        lines.append(f"- Mean J_total: {real['j_total'].mean():.4f}")
    else:
        lines.append("- Real Top-N candidates: 0")

    lines.append("")
    lines.append("## Validation")
    lines.append("")
    validation_path = dataset / "reports" / "validation_report.json"
    if validation_path.exists():
        validation = read_json(validation_path)
        lines.append(f"- Validation passed: {validation['passed']}")
        lines.append(f"- Checks run: {len(validation['checks'])}")
    else:
        lines.append("- Validation report has not been generated yet.")

    summary = "\n".join(lines) + "\n"
    (dataset / "reports" / "dataset_summary.md").write_text(summary, encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize generated EON dataset artifacts.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = summarize_dataset(args.dataset)
    print(summary)


if __name__ == "__main__":
    main()

