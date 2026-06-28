from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import build_checksums, parse_int_list, read_json, write_json


def _ok(report: dict[str, Any], name: str, passed: bool, detail: str = "") -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        report["passed"] = False


def _load_split_frames(dataset: Path, split: str) -> dict[str, pd.DataFrame]:
    return {
        "traffic": pd.read_parquet(dataset / "traffic" / f"{split}.parquet"),
        "candidates": pd.read_parquet(dataset / "candidates" / f"{split}.parquet"),
        "candidates_full": pd.read_parquet(dataset / "candidates" / f"{split}_full.parquet"),
        "cnn_index": pd.read_parquet(dataset / "cnn" / f"{split}_index.parquet"),
        "gnn_routes": pd.read_parquet(dataset / "gnn" / f"{split}_routes.parquet"),
        "dqn": pd.read_parquet(dataset / "dqn" / f"{split}_transitions.parquet"),
    }


def validate_topology(dataset: Path, report: dict[str, Any]) -> None:
    nodes = pd.read_csv(dataset / "topology" / "nodes.csv")
    undirected = pd.read_csv(dataset / "topology" / "undirected_links.csv")
    directed = pd.read_csv(dataset / "topology" / "directed_links.csv")
    _ok(report, "topology_node_count", len(nodes) == 14, f"nodes={len(nodes)}")
    _ok(report, "topology_undirected_link_count", len(undirected) == 22, f"undirected={len(undirected)}")
    _ok(report, "topology_directed_link_count", len(directed) == 44, f"directed={len(directed)}")
    pairs = {(int(row.src), int(row.dst)) for row in directed.itertuples(index=False)}
    reverse_ok = all((dst, src) in pairs for src, dst in pairs)
    _ok(report, "topology_reverse_links", reverse_ok)
    _ok(report, "topology_positive_lengths", bool((directed["length_km"] > 0).all()))


def validate_candidates(frames: dict[str, pd.DataFrame], n_max: int, report: dict[str, Any], split: str) -> None:
    candidates = frames["candidates"]
    _ok(report, f"{split}_candidate_mask_values", set(candidates["candidate_mask"].unique()).issubset({0, 1, False, True}))
    real = candidates[candidates["candidate_mask"] == 1]
    padding = candidates[candidates["candidate_mask"] == 0]
    _ok(report, f"{split}_mask_real_feasible", bool(real["is_feasible"].all()) if len(real) else True)
    _ok(report, f"{split}_padding_mask_zero", bool((padding["candidate_id"] == -1).all()) if len(padding) else True)
    group_sizes = candidates.groupby(["episode_id", "request_id"]).size()
    _ok(report, f"{split}_candidate_mask_length", bool((group_sizes == n_max).all()), f"groups={len(group_sizes)}")
    mask_sums = candidates.groupby(["episode_id", "request_id"])["candidate_mask"].sum()
    real_counts = candidates[candidates["candidate_mask"] == 1].groupby(["episode_id", "request_id"]).size()
    aligned = mask_sums.sort_index().equals(real_counts.reindex(mask_sums.index, fill_value=0).astype(mask_sums.dtype).sort_index())
    _ok(report, f"{split}_candidate_mask_sum", aligned)
    dqn = frames["dqn"]
    _ok(report, f"{split}_padding_action_selected", bool((dqn["padding_action_selected"] == 0).all()))
    _ok(report, f"{split}_invalid_action_selected", bool((dqn["invalid_action_selected"] == 0).all()))
    blocked = dqn[dqn["blocked"] == 1]
    if len(blocked):
        _ok(report, f"{split}_blocked_when_no_feasible", bool((blocked["num_feasible_before_topn"] == 0).all()))


def validate_dqn(frames: dict[str, pd.DataFrame], cfg: dict[str, Any], report: dict[str, Any], split: str) -> None:
    dqn = frames["dqn"]
    accepted = dqn[dqn["blocked"] == 0]
    delay_bound = float(cfg.get("delay_bound_ms", 50.0))
    _ok(report, f"{split}_accepted_qot_margin", bool((accepted["qot_margin"] >= -1e-9).all()) if len(accepted) else True)
    _ok(report, f"{split}_accepted_delay_bound", bool((accepted["delay_ms"] <= delay_bound + 1e-9).all()) if len(accepted) else True)
    _ok(report, f"{split}_candidate_mask_valid", bool(dqn["candidate_mask_valid"].all()) if len(dqn) else True)


def validate_cnn(dataset: Path, frames: dict[str, pd.DataFrame], cfg: dict[str, Any], report: dict[str, Any], split: str) -> None:
    tensors = np.load(dataset / "cnn" / f"{split}_tensors.npz")["X_spec"]
    index = frames["cnn_index"]
    slots = int(cfg.get("slots", 100))
    _ok(report, f"{split}_cnn_shape", list(tensors.shape)[1:] == [6, slots], str(tensors.shape))
    _ok(report, f"{split}_cnn_index_match", len(index) == tensors.shape[0], f"index={len(index)} tensors={tensors.shape[0]}")
    finite_cols = ["delta_frag", "frag_after", "lmax_after_norm", "nseg_after_norm", "compactness", "placement_score", "J_total"]
    finite_ok = bool(np.isfinite(index[finite_cols].to_numpy(dtype=float)).all()) if len(index) else True
    _ok(report, f"{split}_cnn_labels_finite", finite_ok)
    block_ok = True
    for row in index.itertuples(index=False):
        selected = tensors[int(row.sample_id), 1, :]
        expected = np.zeros(slots, dtype=selected.dtype)
        expected[int(row.b_start) : int(row.b_start) + int(row.w)] = 1
        if not np.array_equal(selected, expected):
            block_ok = False
            break
    _ok(report, f"{split}_cnn_selected_block_indicator", block_ok)


def validate_gnn(dataset: Path, frames: dict[str, pd.DataFrame], cfg: dict[str, Any], report: dict[str, Any], split: str) -> None:
    graphs = np.load(dataset / "gnn" / f"{split}_graphs.npz")
    node_features = graphs["node_features"]
    link_features = graphs["link_features"]
    edge_index = graphs["edge_index"]
    _ok(report, f"{split}_gnn_node_shape", node_features.shape[1:] == (14, 4), str(node_features.shape))
    _ok(report, f"{split}_gnn_link_shape", link_features.shape[1:] == (44, 8), str(link_features.shape))
    _ok(report, f"{split}_gnn_edge_index_shape", edge_index.shape == (2, 44), str(edge_index.shape))
    routes = frames["gnn_routes"]
    valid_links = True
    for value in routes["route_directed_link_ids"].head(5000):
        links = parse_int_list(value)
        if any(link < 0 or link >= 44 for link in links):
            valid_links = False
            break
    _ok(report, f"{split}_gnn_route_link_ids_valid", valid_links)


def validate_splits(all_frames: dict[str, dict[str, pd.DataFrame]], report: dict[str, Any]) -> None:
    episodes_by_split = {split: set(frames["traffic"]["episode_id"].unique()) for split, frames in all_frames.items()}
    seeds_by_split = {split: set(int(x) for x in frames["traffic"]["seed"].unique()) for split, frames in all_frames.items()}
    split_names = list(all_frames)
    episode_ok = True
    seed_ok = True
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            episode_ok = episode_ok and episodes_by_split[left].isdisjoint(episodes_by_split[right])
            seed_ok = seed_ok and seeds_by_split[left].isdisjoint(seeds_by_split[right])
    _ok(report, "split_episode_leakage", episode_ok)
    _ok(report, "split_seed_leakage", seed_ok)


def validate_checksums(dataset: Path, report: dict[str, Any]) -> None:
    manifest = read_json(dataset / "manifest.json")
    expected = dict(manifest.get("checksums", {}))
    actual = build_checksums(dataset)
    for generated_later in ("reports/validation_report.json", "reports/dataset_summary.md"):
        actual.pop(generated_later, None)
    _ok(report, "manifest_checksums_match", expected == actual, f"expected={len(expected)} actual={len(actual)}")


def validate_dataset(dataset_path: str | Path) -> dict[str, Any]:
    dataset = Path(dataset_path)
    manifest = read_json(dataset / "manifest.json")
    cfg = manifest["parameters"]
    report: dict[str, Any] = {
        "dataset": str(dataset),
        "passed": True,
        "checks": [],
        "summary": {},
    }
    validate_topology(dataset, report)
    split_names = list(manifest["splits"].keys())
    all_frames = {split: _load_split_frames(dataset, split) for split in split_names}
    for split, frames in all_frames.items():
        validate_candidates(frames, int(cfg.get("n_max", 32)), report, split)
        validate_dqn(frames, cfg, report, split)
        validate_cnn(dataset, frames, cfg, report, split)
        validate_gnn(dataset, frames, cfg, report, split)
    validate_splits(all_frames, report)
    validate_checksums(dataset, report)
    report["summary"] = {
        split: {
            "requests": int(len(frames["traffic"])),
            "topn_rows": int(len(frames["candidates"])),
            "full_candidate_rows": int(len(frames["candidates_full"])),
            "cnn_samples": int(len(frames["cnn_index"])),
            "dqn_transitions": int(len(frames["dqn"])),
            "blocking_rate": float(frames["dqn"]["blocked"].mean()) if len(frames["dqn"]) else math.nan,
        }
        for split, frames in all_frames.items()
    }
    write_json(dataset / "reports" / "validation_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated EON dataset artifacts.")
    parser.add_argument("--dataset", required=True, help="Dataset root directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = validate_dataset(args.dataset)
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Validation {status}: {args.dataset}")
    for split, summary in report["summary"].items():
        print(f"{split}: requests={summary['requests']} blocking_rate={summary['blocking_rate']:.4f}")
    if not report["passed"]:
        failed = [check for check in report["checks"] if not check["passed"]]
        for check in failed[:10]:
            print(f"FAILED {check['name']}: {check['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

