from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io_utils import ensure_dir, project_root, write_json


DEFAULT_UNDIRECTED_LINKS: list[tuple[int, int, int]] = [
    (1, 2, 1050),
    (1, 3, 1500),
    (1, 8, 2400),
    (2, 3, 600),
    (2, 4, 750),
    (3, 6, 1800),
    (4, 5, 600),
    (4, 11, 1950),
    (5, 6, 1200),
    (5, 7, 600),
    (6, 10, 1050),
    (6, 14, 1800),
    (7, 8, 750),
    (7, 10, 1350),
    (8, 9, 750),
    (9, 10, 750),
    (9, 12, 300),
    (9, 13, 300),
    (11, 12, 600),
    (11, 13, 750),
    (12, 14, 300),
    (13, 14, 150),
]

NODE_COORDS: dict[int, tuple[float, float]] = {
    1: (0.05, 0.72),
    2: (0.18, 0.82),
    3: (0.25, 0.62),
    4: (0.34, 0.86),
    5: (0.45, 0.72),
    6: (0.43, 0.48),
    7: (0.56, 0.62),
    8: (0.67, 0.76),
    9: (0.78, 0.62),
    10: (0.66, 0.42),
    11: (0.52, 0.92),
    12: (0.75, 0.90),
    13: (0.88, 0.82),
    14: (0.92, 0.50),
}


@dataclass(frozen=True)
class DirectedLink:
    directed_link_id: int
    undirected_link_id: int
    src: int
    dst: int
    length_km: float
    delay_ms: float
    span_count: int
    amplifier_count: int
    base_energy_cost: float
    base_qot_risk: float


@dataclass(frozen=True)
class Topology:
    name: str
    slot_total: int
    nodes: pd.DataFrame
    undirected_links: pd.DataFrame
    directed_links: pd.DataFrame
    directed_link_by_pair: dict[tuple[int, int], int]
    reverse_directed_link: dict[int, int]

    @property
    def node_count(self) -> int:
        return int(len(self.nodes))

    @property
    def directed_link_count(self) -> int:
        return int(len(self.directed_links))

    @property
    def undirected_link_count(self) -> int:
        return int(len(self.undirected_links))

    @property
    def edge_index(self):
        import numpy as np

        return np.asarray(
            [
                self.directed_links["src"].to_numpy(dtype=int) - 1,
                self.directed_links["dst"].to_numpy(dtype=int) - 1,
            ],
            dtype=np.int64,
        )


def topology_source_dir(name: str) -> Path:
    return project_root() / "data" / "eon" / "topologies" / name


def build_directed_links(
    span_length_km: float = 80.0,
    amplifier_power_w: float = 20.0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    directed_id = 0
    for undirected_id, (u, v, length) in enumerate(DEFAULT_UNDIRECTED_LINKS):
        for src, dst in ((u, v), (v, u)):
            amplifier_count = int(math.ceil(length / span_length_km))
            rows.append(
                {
                    "directed_link_id": directed_id,
                    "undirected_link_id": undirected_id,
                    "src": src,
                    "dst": dst,
                    "length_km": float(length),
                    "delay_ms": float(length) / 200.0,
                    "span_count": amplifier_count,
                    "amplifier_count": amplifier_count,
                    "base_energy_cost": amplifier_count * float(amplifier_power_w),
                    "base_qot_risk": min(float(length) / 4000.0, 1.0),
                }
            )
            directed_id += 1
    return pd.DataFrame(rows)


def build_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node_id": node, "node_id_zero_based": node - 1, "x": xy[0], "y": xy[1]}
            for node, xy in NODE_COORDS.items()
        ]
    )


def build_undirected_links() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"undirected_link_id": idx, "u": u, "v": v, "length_km": length}
            for idx, (u, v, length) in enumerate(DEFAULT_UNDIRECTED_LINKS)
        ]
    )


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    for col in frame.columns:
        if frame[col].dtype == object:
            try:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
            except (TypeError, ValueError):
                pass
    return frame


def load_topology(name: str = "nsfnet_deeprmsa_14_22", slot_total: int = 100) -> Topology:
    src = topology_source_dir(name)
    if src.exists():
        nodes = _coerce_numeric(pd.read_csv(src / "nodes.csv"))
        undirected = _coerce_numeric(pd.read_csv(src / "undirected_links.csv"))
        directed = _coerce_numeric(pd.read_csv(src / "directed_links.csv"))
    else:
        nodes = build_nodes()
        undirected = build_undirected_links()
        directed = build_directed_links()

    pair_to_id = {
        (int(row.src), int(row.dst)): int(row.directed_link_id)
        for row in directed.itertuples(index=False)
    }
    reverse = {
        int(row.directed_link_id): pair_to_id[(int(row.dst), int(row.src))]
        for row in directed.itertuples(index=False)
    }
    return Topology(
        name=name,
        slot_total=slot_total,
        nodes=nodes,
        undirected_links=undirected,
        directed_links=directed,
        directed_link_by_pair=pair_to_id,
        reverse_directed_link=reverse,
    )


def copy_topology_files(name: str, output_dir: str | Path) -> None:
    output = ensure_dir(output_dir)
    src = topology_source_dir(name)
    if src.exists():
        for filename in ("nodes.csv", "undirected_links.csv", "directed_links.csv", "topology.json"):
            shutil.copyfile(src / filename, output / filename)
        return

    build_nodes().to_csv(output / "nodes.csv", index=False)
    build_undirected_links().to_csv(output / "undirected_links.csv", index=False)
    build_directed_links().to_csv(output / "directed_links.csv", index=False)
    write_json(
        output / "topology.json",
        {
            "name": name,
            "topology_variant": "DeepRMSA-compatible NSFNET 14-node / 22-link / 44-directed-link",
            "source_note": "Numeric topology encoded from experiment prompt; no external source code copied.",
            "slot_total": 100,
            "directed": True,
            "node_count": 14,
            "undirected_link_count": 22,
            "directed_link_count": 44,
        },
    )


def route_links_for_path(topology: Topology, node_path: Iterable[int]) -> list[int]:
    nodes = list(node_path)
    return [topology.directed_link_by_pair[(u, v)] for u, v in zip(nodes[:-1], nodes[1:])]


def path_length(topology: Topology, link_ids: Iterable[int]) -> float:
    directed = topology.directed_links.set_index("directed_link_id")
    return float(sum(float(directed.loc[int(link_id), "length_km"]) for link_id in link_ids))
