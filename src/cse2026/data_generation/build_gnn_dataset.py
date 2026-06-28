from __future__ import annotations

from typing import Any

import numpy as np

from .features import graph_features, route_label_row
from .io_utils import stringify


class GnnBuffers:
    def __init__(self) -> None:
        self.node_features: list[np.ndarray] = []
        self.link_features: list[np.ndarray] = []
        self.global_features: list[np.ndarray] = []
        self.request_features: list[np.ndarray] = []
        self.sample_ids: list[str] = []
        self.route_rows: list[dict[str, Any]] = []

    def append(
        self,
        *,
        topology,
        occupancy: np.ndarray,
        active_link_counts: np.ndarray,
        active_node_counts: np.ndarray,
        request: dict[str, Any],
        routes: list[Any],
        modulations: list[Any],
        summaries: dict[tuple[int, int], dict[str, Any]],
        num_feasible: int,
        global_fragmentation: float,
        cfg: dict[str, Any],
    ) -> None:
        sample_id = f"{request['episode_id']}:{request['request_id']}"
        node, link, global_feat, request_feat = graph_features(
            topology=topology,
            occupancy=occupancy,
            active_link_counts=active_link_counts,
            active_node_counts=active_node_counts,
            request=request,
            cfg=cfg,
        )
        self.node_features.append(node)
        self.link_features.append(link)
        self.global_features.append(global_feat)
        self.request_features.append(request_feat)
        self.sample_ids.append(sample_id)
        for route in routes:
            for modulation in modulations:
                summary = summaries.get((route.route_id, modulation.modulation_id), {})
                row = route_label_row(
                    sample_id=sample_id,
                    request=request,
                    route=route,
                    modulation=modulation,
                    occupancy=occupancy,
                    summary=summary,
                    cfg=cfg,
                )
                row["block_now"] = int(num_feasible == 0)
                row["num_feasible"] = int(num_feasible)
                row["num_feasible_norm"] = min(float(num_feasible), float(cfg.get("n_max", 32))) / float(cfg.get("n_max", 32))
                row["global_fragmentation"] = float(global_fragmentation)
                row["route_node_ids"] = stringify(row["route_node_ids"])
                row["route_directed_link_ids"] = stringify(row["route_directed_link_ids"])
                self.route_rows.append(row)

    def arrays(self, edge_index: np.ndarray) -> dict[str, np.ndarray]:
        if self.node_features:
            node_features = np.stack(self.node_features, axis=0)
            link_features = np.stack(self.link_features, axis=0)
            global_features = np.stack(self.global_features, axis=0)
            request_features = np.stack(self.request_features, axis=0)
        else:
            node_features = np.zeros((0, 14, 4), dtype=np.float32)
            link_features = np.zeros((0, 44, 8), dtype=np.float32)
            global_features = np.zeros((0, 8), dtype=np.float32)
            request_features = np.zeros((0, 3), dtype=np.float32)
        return {
            "node_features": node_features.astype(np.float32),
            "link_features": link_features.astype(np.float32),
            "global_features": global_features.astype(np.float32),
            "request_features": request_features.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "sample_ids": np.asarray(self.sample_ids, dtype="U128"),
        }

