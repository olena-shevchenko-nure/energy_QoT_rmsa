from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import networkx as nx

from .topology import Topology, route_links_for_path


@dataclass(frozen=True)
class Route:
    route_id: int
    src: int
    dst: int
    node_ids: list[int]
    directed_link_ids: list[int]
    length_km: float
    delay_ms: float
    hop_count: int


def build_graph(topology: Topology) -> nx.Graph:
    graph = nx.Graph()
    for node_id in topology.nodes["node_id"].astype(int).tolist():
        graph.add_node(node_id)
    for row in topology.undirected_links.itertuples(index=False):
        graph.add_edge(int(row.u), int(row.v), length_km=float(row.length_km))
    return graph


def precompute_k_shortest_routes(topology: Topology, k_routes: int) -> dict[tuple[int, int], list[Route]]:
    graph = build_graph(topology)
    directed = topology.directed_links.set_index("directed_link_id")
    routes: dict[tuple[int, int], list[Route]] = {}
    node_ids = topology.nodes["node_id"].astype(int).tolist()
    for src in node_ids:
        for dst in node_ids:
            if src == dst:
                continue
            od_routes: list[Route] = []
            for route_id, node_path in enumerate(islice(nx.shortest_simple_paths(graph, src, dst, weight="length_km"), k_routes)):
                link_ids = route_links_for_path(topology, node_path)
                length = float(sum(float(directed.loc[link_id, "length_km"]) for link_id in link_ids))
                delay = float(sum(float(directed.loc[link_id, "delay_ms"]) for link_id in link_ids))
                od_routes.append(
                    Route(
                        route_id=route_id,
                        src=src,
                        dst=dst,
                        node_ids=[int(node) for node in node_path],
                        directed_link_ids=[int(link_id) for link_id in link_ids],
                        length_km=length,
                        delay_ms=delay,
                        hop_count=len(link_ids),
                    )
                )
            routes[(src, dst)] = od_routes
    return routes

