from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .build_cnn_dataset import cnn_index_row, select_cnn_candidates, tensor_for_candidate
from .build_dqn_dataset import choose_collection_action, dqn_transition_row
from .build_gnn_dataset import GnnBuffers
from .candidates import generate_candidates_for_request, serialize_candidate_row, sorted_feasible
from .io_utils import (
    build_checksums,
    ensure_dir,
    expand_seeds,
    load_yaml,
    save_npz_deterministic,
    stable_seed,
    utc_timestamp,
    write_checksums,
    write_json,
    write_parquet,
)
from .modulation import load_modulations
from .routing import precompute_k_shortest_routes
from .safety_filter import make_padding_candidate
from .spectrum import fragmentation
from .topology import Topology, copy_topology_files, load_topology
from .traffic import generate_requests_for_episode


def _release_expired(
    *,
    active_lightpaths: list[dict[str, Any]],
    occupancy: np.ndarray,
    release_times: np.ndarray,
    active_link_counts: np.ndarray,
    active_node_counts: np.ndarray,
    now: float,
) -> list[dict[str, Any]]:
    remaining: list[dict[str, Any]] = []
    for lightpath in active_lightpaths:
        if float(lightpath["departure_time"]) > now:
            remaining.append(lightpath)
            continue
        start = int(lightpath["b_start"])
        width = int(lightpath["w"])
        for link_id in lightpath["route_directed_link_ids"]:
            occupancy[int(link_id), start : start + width] = 0
            release_times[int(link_id), start : start + width] = 0.0
            active_link_counts[int(link_id)] = max(active_link_counts[int(link_id)] - 1, 0)
        for node_id in lightpath["route_node_ids"]:
            idx = int(node_id) - 1
            active_node_counts[idx] = max(active_node_counts[idx] - 1, 0)
    return remaining


def _apply_candidate(
    *,
    candidate: dict[str, Any],
    request: dict[str, Any],
    active_lightpaths: list[dict[str, Any]],
    occupancy: np.ndarray,
    release_times: np.ndarray,
    active_link_counts: np.ndarray,
    active_node_counts: np.ndarray,
) -> None:
    start = int(candidate["b_start"])
    width = int(candidate["w"])
    links = [int(x) for x in candidate["route_directed_link_ids"]]
    nodes = [int(x) for x in candidate["route_node_ids"]]
    departure = float(request["departure_time"])
    for link_id in links:
        occupancy[link_id, start : start + width] = 1
        release_times[link_id, start : start + width] = departure
        active_link_counts[link_id] += 1
    for node_id in nodes:
        active_node_counts[node_id - 1] += 1
    active_lightpaths.append(
        {
            "departure_time": departure,
            "route_directed_link_ids": links,
            "route_node_ids": nodes,
            "b_start": start,
            "w": width,
        }
    )


def _mean_link_fragmentation(occupancy: np.ndarray) -> float:
    frags = [fragmentation((row == 0).astype(np.uint8)) for row in occupancy]
    return float(np.mean(frags))


def _schema_files(output: Path, cfg: dict[str, Any]) -> None:
    write_json(
        output / "gnn" / "schema.json",
        {
            "node_features": ["is_source", "is_destination", "degree_norm", "active_lightpaths_norm"],
            "link_features": [
                "length_norm",
                "delay_norm",
                "occupancy_ratio",
                "largest_free_block_norm",
                "fragmentation_indicator",
                "active_lightpaths_norm",
                "incremental_energy_cost_norm",
                "qot_risk",
            ],
            "global_features": [
                "mean_link_occupancy",
                "max_link_occupancy",
                "mean_fragmentation",
                "max_fragmentation",
                "mean_largest_free_block",
                "min_largest_free_block",
                "total_active_lightpaths_norm",
                "total_energy_state_norm",
            ],
            "request_features": ["bit_rate_norm", "holding_time_norm", "required_slots_min_norm"],
        },
    )
    write_json(
        output / "cnn" / "schema.json",
        {
            "shape": ["num_samples", 6, int(cfg.get("slots", 100))],
            "channels": [
                "route_availability",
                "selected_block_indicator",
                "local_fragmentation_context",
                "future_release_time",
                "route_occupancy_fraction",
                "distance_to_selected_block",
            ],
            "labels": [
                "delta_frag",
                "frag_after",
                "lmax_after_norm",
                "nseg_after_norm",
                "created_small_gap",
                "compactness",
                "placement_score",
                "J_total",
            ],
        },
    )
    write_json(
        output / "dqn" / "schema.json",
        {
            "selected_candidate_index": "Index into the Top-N candidate table for the same request; -1 means block.",
            "candidate_mask_valid": "True when real candidates fit inside N_max and padding is ignored.",
            "best_candidate_index": "Best real Top-N candidate according to q_head_score; -1 when blocked.",
            "reward": "Proxy reward for accepted/block decisions.",
        },
    )


def _split_counts_manifest(output: Path, splits: list[str]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split in splits:
        split_counts: dict[str, Any] = {}
        for group, filename in (
            ("traffic_rows", output / "traffic" / f"{split}.parquet"),
            ("candidate_topn_rows", output / "candidates" / f"{split}.parquet"),
            ("candidate_full_rows", output / "candidates" / f"{split}_full.parquet"),
            ("gnn_route_rows", output / "gnn" / f"{split}_routes.parquet"),
            ("cnn_index_rows", output / "cnn" / f"{split}_index.parquet"),
            ("dqn_transition_rows", output / "dqn" / f"{split}_transitions.parquet"),
        ):
            if filename.exists():
                import pandas as pd

                split_counts[group] = int(len(pd.read_parquet(filename)))
        graph_path = output / "gnn" / f"{split}_graphs.npz"
        if graph_path.exists():
            data = np.load(graph_path)
            split_counts["gnn_graph_shape"] = list(data["node_features"].shape)
        cnn_path = output / "cnn" / f"{split}_tensors.npz"
        if cnn_path.exists():
            data = np.load(cnn_path)
            split_counts["cnn_tensor_shape"] = list(data["X_spec"].shape)
        counts[split] = split_counts
    return counts


def _write_split_artifacts(
    *,
    split: str,
    output: Path,
    topology: Topology,
    traffic_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    full_candidate_rows: list[dict[str, Any]],
    gnn: GnnBuffers,
    cnn_tensors: list[np.ndarray],
    cnn_index_rows: list[dict[str, Any]],
    dqn_rows: list[dict[str, Any]],
) -> None:
    write_parquet(output / "traffic" / f"{split}.parquet", traffic_rows)
    write_parquet(output / "candidates" / f"{split}.parquet", candidate_rows)
    write_parquet(output / "candidates" / f"{split}_full.parquet", full_candidate_rows)
    write_parquet(output / "gnn" / f"{split}_routes.parquet", gnn.route_rows)
    write_parquet(output / "cnn" / f"{split}_index.parquet", cnn_index_rows)
    write_parquet(output / "dqn" / f"{split}_transitions.parquet", dqn_rows)
    save_npz_deterministic(output / "gnn" / f"{split}_graphs.npz", **gnn.arrays(topology.edge_index))
    if cnn_tensors:
        x_spec = np.stack(cnn_tensors, axis=0).astype(np.float16)
    else:
        x_spec = np.zeros((0, 6, topology.slot_total), dtype=np.float16)
    save_npz_deterministic(output / "cnn" / f"{split}_tensors.npz", X_spec=x_spec)


def generate_dataset(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    output = Path(output_path)
    if output.exists():
        shutil.rmtree(output)
    ensure_dir(output)
    topology = load_topology(str(cfg.get("topology", "nsfnet_deeprmsa_14_22")), int(cfg.get("slots", 100)))
    modulations = load_modulations()
    routes_by_od = precompute_k_shortest_routes(topology, int(cfg.get("k_routes", 5)))
    n_max = int(cfg.get("n_max", 32))
    splits = list(cfg.get("splits", {}).keys())

    for subdir in ("topology", "traffic", "candidates", "gnn", "cnn", "dqn", "reports"):
        ensure_dir(output / subdir)
    copy_topology_files(topology.name, output / "topology")
    _schema_files(output, cfg)

    for split, split_cfg in cfg["splits"].items():
        seeds = expand_seeds(split_cfg)
        traffic_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        full_candidate_rows: list[dict[str, Any]] = []
        gnn = GnnBuffers()
        cnn_tensors: list[np.ndarray] = []
        cnn_index_rows: list[dict[str, Any]] = []
        dqn_rows: list[dict[str, Any]] = []

        for seed in seeds:
            for scenario in cfg.get("traffic_scenarios", ["uniform"]):
                for load_name in cfg.get("loads", ["medium"]):
                    episode_id = f"{split}-{scenario}-{load_name}-seed{seed}"
                    requests = generate_requests_for_episode(
                        split=split,
                        seed=int(seed),
                        scenario=str(scenario),
                        load_name=str(load_name),
                        requests_per_seed=int(split_cfg["requests_per_seed"]),
                        cfg=cfg,
                        episode_id=episode_id,
                        node_count=topology.node_count,
                    )
                    occupancy = np.zeros((topology.directed_link_count, topology.slot_total), dtype=np.uint8)
                    release_times = np.zeros_like(occupancy, dtype=np.float32)
                    active_link_counts = np.zeros(topology.directed_link_count, dtype=np.int32)
                    active_node_counts = np.zeros(topology.node_count, dtype=np.int32)
                    active_lightpaths: list[dict[str, Any]] = []
                    episode_rng = np.random.default_rng(stable_seed("episode", split, seed, scenario, load_name))

                    for request_idx, request in enumerate(requests):
                        active_lightpaths = _release_expired(
                            active_lightpaths=active_lightpaths,
                            occupancy=occupancy,
                            release_times=release_times,
                            active_link_counts=active_link_counts,
                            active_node_counts=active_node_counts,
                            now=float(request["arrival_time"]),
                        )
                        od_routes = routes_by_od[(int(request["src"]), int(request["dst"]))]
                        generated, summaries = generate_candidates_for_request(
                            request=request,
                            routes=od_routes,
                            topology=topology,
                            modulations=modulations,
                            occupancy=occupancy,
                            cfg=cfg,
                            rng=episode_rng,
                        )
                        feasible_sorted = sorted_feasible(generated)
                        topn_real = [dict(row) for row in feasible_sorted[:n_max]]
                        for topn_index, row in enumerate(topn_real):
                            row["topn_index"] = topn_index
                            row["in_topn"] = True
                            row["candidate_mask"] = 1
                            row["n_max"] = n_max
                            row["split"] = split
                            row["seed"] = int(seed)
                            row["traffic_scenario"] = scenario
                            row["load_name"] = load_name

                        for row in feasible_sorted:
                            row["topn_index"] = next((real["topn_index"] for real in topn_real if real["candidate_id"] == row["candidate_id"]), -1)
                            row["in_topn"] = row["topn_index"] >= 0
                            row["candidate_mask"] = int(row["in_topn"])
                            row["n_max"] = n_max
                            row["split"] = split
                            row["seed"] = int(seed)
                            row["traffic_scenario"] = scenario
                            row["load_name"] = load_name
                            full_candidate_rows.append(serialize_candidate_row(row))

                        for topn_index in range(n_max):
                            if topn_index < len(topn_real):
                                candidate_rows.append(serialize_candidate_row(topn_real[topn_index]))
                            else:
                                candidate_rows.append(
                                    make_padding_candidate(
                                        episode_id=episode_id,
                                        request_id=int(request["request_id"]),
                                        topn_index=topn_index,
                                        n_max=n_max,
                                        split=split,
                                        seed=int(seed),
                                        traffic_scenario=scenario,
                                        load_name=load_name,
                                    )
                                )

                        global_frag = _mean_link_fragmentation(occupancy)
                        gnn.append(
                            topology=topology,
                            occupancy=occupancy,
                            active_link_counts=active_link_counts,
                            active_node_counts=active_node_counts,
                            request=request,
                            routes=od_routes,
                            modulations=modulations,
                            summaries=summaries,
                            num_feasible=len(feasible_sorted),
                            global_fragmentation=global_frag,
                            cfg=cfg,
                        )

                        selected_for_cnn = select_cnn_candidates(
                            topn_real,
                            rng=episode_rng,
                            max_samples_per_request=int(cfg.get("max_cnn_samples_per_request", 12)),
                        )
                        for candidate in selected_for_cnn:
                            cnn_sample_id = len(cnn_index_rows)
                            cnn_tensors.append(
                                tensor_for_candidate(
                                    candidate=candidate,
                                    occupancy=occupancy,
                                    release_times=release_times,
                                    now=float(request["arrival_time"]),
                                    cfg=cfg,
                                )
                            )
                            cnn_index_rows.append(cnn_index_row(sample_id=cnn_sample_id, candidate=candidate, request=request))

                        selected = choose_collection_action(topn_real, episode_rng, cfg)
                        state_id = f"{episode_id}:{request['request_id']}"
                        next_state_id = f"{episode_id}:{int(request['request_id']) + 1}"
                        done = request_idx == len(requests) - 1
                        dqn_rows.append(
                            dqn_transition_row(
                                transition_id=len(dqn_rows),
                                request=request,
                                state_id=state_id,
                                next_state_id=next_state_id,
                                selected=selected,
                                topn_real_candidates=topn_real,
                                full_candidate_count=len(feasible_sorted),
                                n_max=n_max,
                                done=done,
                                cfg=cfg,
                            )
                        )

                        traffic_row = dict(request)
                        traffic_row["num_feasible"] = int(len(feasible_sorted))
                        traffic_row["num_topn_real"] = int(len(topn_real))
                        traffic_row["blocked_by_feasibility"] = bool(len(feasible_sorted) == 0)
                        traffic_rows.append(traffic_row)

                        if selected is not None:
                            _apply_candidate(
                                candidate=selected,
                                request=request,
                                active_lightpaths=active_lightpaths,
                                occupancy=occupancy,
                                release_times=release_times,
                                active_link_counts=active_link_counts,
                                active_node_counts=active_node_counts,
                            )

        _write_split_artifacts(
            split=split,
            output=output,
            topology=topology,
            traffic_rows=traffic_rows,
            candidate_rows=candidate_rows,
            full_candidate_rows=full_candidate_rows,
            gnn=gnn,
            cnn_tensors=cnn_tensors,
            cnn_index_rows=cnn_index_rows,
            dqn_rows=dqn_rows,
        )

    manifest = {
        "dataset_name": cfg.get("dataset_name", output.name),
        "config_path": str(config_path),
        "generation_timestamp": utc_timestamp(),
        "topology": topology.name,
        "slot_total": topology.slot_total,
        "k_routes": int(cfg.get("k_routes", 5)),
        "n_max": n_max,
        "splits": _split_counts_manifest(output, splits),
        "parameters": cfg,
    }
    checksums = build_checksums(output)
    manifest["checksums"] = checksums
    write_json(output / "manifest.json", manifest)
    write_checksums(output / "checksums.sha256", checksums)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate EON experiment datasets.")
    parser.add_argument("--config", required=True, help="Path to YAML data-generation config.")
    parser.add_argument("--output", required=True, help="Output dataset directory.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_dataset(args.config, args.output)
    total_requests = sum(split.get("traffic_rows", 0) for split in manifest["splits"].values())
    print(f"Generated {manifest['dataset_name']} with {total_requests} requests at {args.output}")


if __name__ == "__main__":
    main()
