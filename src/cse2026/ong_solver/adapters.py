from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .common import (
    Candidate,
    CandidateBatch,
    SolverConfig,
    StateView,
    build_candidate_batch,
    candidate_starts,
    route_availability,
    score_candidate,
)


class OngAdapter(Protocol):
    def candidate_batch(self, env: Any, cfg: SolverConfig, rng: np.random.Generator) -> CandidateBatch: ...

    def block_action(self, env: Any) -> Any: ...


def select_adapter(env: Any) -> OngAdapter:
    if hasattr(env, "heuristic_context") or _looks_like_v2_simulator(getattr(env, "simulator", env)):
        return V2OpticalNetworkingGymAdapter()
    if hasattr(env, "k_shortest_paths") and hasattr(env, "topology") and hasattr(env, "current_service"):
        return LegacyOpticalRLGymAdapter()
    raise TypeError(
        "Unsupported optical networking environment. Expected optical_networking_gym_v2.OpticalEnv "
        "or legacy optical_rl_gym RMSA/DeepRMSA environment."
    )


def _looks_like_v2_simulator(obj: Any) -> bool:
    return all(hasattr(obj, attr) for attr in ("current_analysis", "current_request", "state", "config", "topology"))


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def _modulation_reach(modulation: Any) -> float:
    return float(_get_field(modulation, "maximum_length", "reach_km", default=4000.0))


def _modulation_efficiency(modulation: Any) -> float:
    return float(_get_field(modulation, "spectral_efficiency", default=1.0))


def _modulation_name(modulation: Any) -> str:
    return str(_get_field(modulation, "name", default="modulation"))


def _release_time_for_service(service: Any) -> float:
    request = _get_field(service, "request")
    if request is not None:
        arrival = float(_get_field(request, "arrival_time", default=0.0))
        holding = float(_get_field(request, "holding_time", default=0.0))
        return arrival + holding
    arrival = float(_get_field(service, "arrival_time", default=0.0))
    holding = float(_get_field(service, "holding_time", default=0.0))
    return arrival + holding


class V2OpticalNetworkingGymAdapter:
    """Adapter for optical_networking_gym_v2.OpticalEnv / Simulator."""

    def candidate_batch(self, env: Any, cfg: SolverConfig, rng: np.random.Generator) -> CandidateBatch:
        context = self._context(env)
        state_view = self._state_view(context)
        candidates = self._candidates(context, state_view, cfg, rng)
        return build_candidate_batch(state_view, candidates, cfg)

    def block_action(self, env: Any) -> int:
        context = self._context(env)
        return int(getattr(context, "reject_action", self._total_actions(context) - 1))

    def _context(self, env: Any) -> Any:
        if hasattr(env, "heuristic_context"):
            return env.heuristic_context()
        simulator = getattr(env, "simulator", env)
        if _looks_like_v2_simulator(simulator):
            try:
                from optical_networking_gym_v2.heuristics.runtime_heuristics import build_runtime_heuristic_context

                return build_runtime_heuristic_context(simulator)
            except Exception:
                return _DuckV2Context(simulator)
        return env

    def _total_actions(self, context: Any) -> int:
        simulator = getattr(context, "simulator", None)
        if simulator is not None and hasattr(simulator, "total_actions"):
            return int(simulator.total_actions)
        config = context.config
        slots = int(config.num_spectrum_resources)
        paths = int(getattr(config, "k_paths", len(context.analysis.paths)))
        mod_count = int(len(context.analysis.modulation_indices))
        return paths * mod_count * slots + 1

    def _state_view(self, context: Any) -> StateView:
        topology = context.topology
        state = context.state
        request = context.request
        node_names = tuple(topology.node_names)
        links = sorted(tuple(topology.links), key=lambda link: int(link.id))
        edge_index = np.asarray(
            [
                [int(link.source_index) for link in links],
                [int(link.target_index) for link in links],
            ],
            dtype=np.int64,
        )
        edge_lengths = np.asarray([float(link.length_km) for link in links], dtype=np.float32)
        occupancy = (np.asarray(state.slot_allocation, dtype=np.int64) != -1).astype(np.uint8)
        release_times = np.zeros_like(occupancy, dtype=np.float32)
        active_link_counts = np.zeros(occupancy.shape[0], dtype=np.int32)
        active_node_counts = np.zeros(len(node_names), dtype=np.int32)
        link_active = getattr(state, "link_active_service_ids", None)
        if link_active is not None:
            for link_id, active_ids in enumerate(link_active):
                active_link_counts[link_id] = len(active_ids)

        active_services = getattr(state, "active_services_by_id", {})
        for service in active_services.values():
            release_time = _release_time_for_service(service)
            path = _get_field(service, "path")
            link_ids = tuple(int(link_id) for link_id in _get_field(path, "link_ids", default=()))
            node_indices = tuple(int(node_id) for node_id in _get_field(path, "node_indices", default=()))
            start = int(_get_field(service, "occupied_slot_start", "service_slot_start", default=0))
            end = int(_get_field(service, "occupied_slot_end_exclusive", "service_slot_end_exclusive", default=start))
            for link_id in link_ids:
                if 0 <= link_id < release_times.shape[0]:
                    release_times[link_id, start:end] = release_time
            for node_id in node_indices:
                if 0 <= node_id < len(active_node_counts):
                    active_node_counts[node_id] += 1

        return StateView(
            node_names=node_names,
            edge_index=edge_index,
            edge_lengths_km=edge_lengths,
            occupancy=occupancy,
            release_times=release_times,
            active_link_counts=active_link_counts,
            active_node_counts=active_node_counts,
            src=node_names[int(request.source_id)],
            dst=node_names[int(request.destination_id)],
            bit_rate_gbps=float(request.bit_rate),
            holding_time=float(request.holding_time),
            current_time=float(getattr(state, "current_time", 0.0)),
            topology_name=str(getattr(topology, "topology_id", "")),
        )

    def _valid_start_flags(self, context: Any) -> np.ndarray:
        mask_mode = str(getattr(context.config, "mask_mode", "resource_and_qot"))
        if mask_mode.endswith("RESOURCE_ONLY") or mask_mode.endswith("resource_only"):
            return np.asarray(context.analysis.resource_valid_starts, dtype=bool)
        return np.asarray(context.analysis.qot_valid_starts, dtype=bool)

    def _encode_action(self, context: Any, path_index: int, modulation_offset: int, initial_slot: int) -> int:
        try:
            from optical_networking_gym_v2.runtime.action_codec import encode_action

            return int(
                encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=initial_slot,
                )
            )
        except Exception:
            slots = int(context.config.num_spectrum_resources)
            modulation_count = int(len(context.analysis.modulation_indices))
            return int((path_index * modulation_count + modulation_offset) * slots + initial_slot)

    def _candidates(
        self,
        context: Any,
        state: StateView,
        cfg: SolverConfig,
        rng: np.random.Generator,
    ) -> list[Candidate]:
        flags = self._valid_start_flags(context)
        analysis = context.analysis
        slots = int(context.config.num_spectrum_resources)
        candidates: list[Candidate] = []
        for path_index, path in enumerate(analysis.paths):
            link_ids = tuple(int(link_id) for link_id in path.link_ids)
            route_node_ids = tuple(path.node_names)
            route_length = float(path.length_km)
            hop_count = int(path.hops)
            for modulation_offset, modulation_index in enumerate(analysis.modulation_indices):
                starts_mask = flags[path_index, modulation_offset, :].astype(np.uint8)
                if not starts_mask.any():
                    continue
                service_slots = int(analysis.required_slots_by_path_mod[path_index, modulation_offset])
                if service_slots <= 0:
                    continue
                starts = candidate_starts(starts_mask, width=1, random_count=cfg.random_starts_per_route, rng=rng)
                modulation = context.config.modulations[int(modulation_index)]
                for start in starts:
                    occupied_width = service_slots + (1 if start + service_slots < slots else 0)
                    availability = route_availability(state.occupancy, link_ids)
                    if start + occupied_width > slots or int(availability[start : start + occupied_width].sum()) != occupied_width:
                        continue
                    action = self._encode_action(context, path_index, modulation_offset, int(start))
                    candidates.append(
                        score_candidate(
                            action=action,
                            route_id=path_index,
                            modulation_index=int(modulation_index),
                            modulation_offset=modulation_offset,
                            b_start=int(start),
                            width=int(occupied_width),
                            route_node_ids=route_node_ids,
                            route_link_ids=link_ids,
                            route_length_km=route_length,
                            hop_count=hop_count,
                            spectral_efficiency=_modulation_efficiency(modulation),
                            modulation_name=_modulation_name(modulation),
                            modulation_reach_km=_modulation_reach(modulation),
                            transponder_power_w=float(_get_field(modulation, "transponder_power_w", default=cfg.transponder_power_w)),
                            state=state,
                            cfg=cfg,
                        )
                    )
        return candidates


class _DuckV2Context:
    def __init__(self, simulator: Any) -> None:
        self.simulator = simulator
        self.config = simulator.config
        self.topology = simulator.topology
        self.state = simulator.state
        self.request = simulator.current_request
        self.analysis = simulator.current_analysis
        self.action_mask = getattr(simulator, "current_mask", None)

    @property
    def reject_action(self) -> int:
        total_actions = getattr(self.simulator, "total_actions", None)
        if total_actions is not None:
            return int(total_actions) - 1
        slots = int(self.config.num_spectrum_resources)
        return int(len(self.analysis.paths) * len(self.analysis.modulation_indices) * slots)


class LegacyOpticalRLGymAdapter:
    """Adapter for legacy optical_rl_gym RMSAEnv and DeepRMSAEnv."""

    def candidate_batch(self, env: Any, cfg: SolverConfig, rng: np.random.Generator) -> CandidateBatch:
        state = self._state_view(env)
        candidates = self._candidates(env, state, cfg, rng)
        return build_candidate_batch(state, candidates, cfg)

    def block_action(self, env: Any) -> Any:
        if hasattr(env, "j"):
            return int(env.k_paths * env.j)
        return (int(env.k_paths), int(env.num_spectrum_resources))

    def _edge_rows(self, env: Any) -> list[tuple[Any, Any, dict[str, Any]]]:
        return sorted(
            list(env.topology.edges(data=True)),
            key=lambda row: int(row[2].get("index", row[2].get("id", 0))),
        )

    def _state_view(self, env: Any) -> StateView:
        edges = self._edge_rows(env)
        node_names = tuple(env.topology.nodes())
        node_index = {node: index for index, node in enumerate(node_names)}
        edge_index = np.asarray(
            [
                [node_index[src] for src, _dst, _data in edges],
                [node_index[dst] for _src, dst, _data in edges],
            ],
            dtype=np.int64,
        )
        edge_lengths = np.asarray([float(data.get("length", data.get("length_km", 0.0))) for _src, _dst, data in edges], dtype=np.float32)
        available_slots = np.asarray(env.topology.graph["available_slots"], dtype=np.uint8)
        occupancy = (1 - available_slots).astype(np.uint8)
        release_times = np.zeros_like(occupancy, dtype=np.float32)
        active_link_counts = np.zeros(occupancy.shape[0], dtype=np.int32)
        active_node_counts = np.zeros(len(node_names), dtype=np.int32)

        running_services = getattr(env.topology, "graph", {}).get("running_services", [])
        for service in running_services:
            release = _release_time_for_service(service)
            path = _get_field(service, "path")
            node_list = tuple(_get_field(path, "node_list", default=()))
            link_ids = tuple(self._route_link_ids(env, node_list))
            start = int(_get_field(service, "initial_slot", default=0))
            width = int(_get_field(service, "number_slots", default=0))
            for link_id in link_ids:
                active_link_counts[link_id] += 1
                release_times[link_id, start : start + width] = release
            for node in node_list:
                if node in node_index:
                    active_node_counts[node_index[node]] += 1

        service = env.current_service
        return StateView(
            node_names=node_names,
            edge_index=edge_index,
            edge_lengths_km=edge_lengths,
            occupancy=occupancy,
            release_times=release_times,
            active_link_counts=active_link_counts,
            active_node_counts=active_node_counts,
            src=service.source,
            dst=service.destination,
            bit_rate_gbps=float(service.bit_rate),
            holding_time=float(service.holding_time),
            current_time=float(getattr(env, "current_time", 0.0)),
            topology_name=str(env.topology.graph.get("name", "")),
        )

    def _route_link_ids(self, env: Any, node_list: tuple[Any, ...]) -> list[int]:
        link_ids: list[int] = []
        for src, dst in zip(node_list[:-1], node_list[1:]):
            data = env.topology[src][dst]
            link_ids.append(int(data.get("index", data.get("id", 0))))
        return link_ids

    def _deep_rmsa_blocks(self, env: Any, route_id: int) -> list[tuple[int, int]]:
        starts, lengths = env.get_available_blocks(route_id)
        return [(int(start), int(length)) for start, length in zip(starts, lengths)]

    def _legacy_action(self, env: Any, route_id: int, start: int, block_index: int) -> Any:
        if hasattr(env, "j"):
            return int(route_id * env.j + block_index)
        return (int(route_id), int(start))

    def _candidates(self, env: Any, state: StateView, cfg: SolverConfig, rng: np.random.Generator) -> list[Candidate]:
        service = env.current_service
        paths = env.k_shortest_paths[(service.source, service.destination)]
        candidates: list[Candidate] = []
        for route_id, path in enumerate(paths):
            link_ids = tuple(self._route_link_ids(env, tuple(path.node_list)))
            width = int(env.get_number_slots(path))
            modulation = _get_field(path, "best_modulation", "current_modulation")
            if modulation is None:
                modulation = getattr(service, "best_modulation", None)
            route_length = float(_get_field(path, "length", "length_km", default=0.0))
            hop_count = int(_get_field(path, "hops", default=max(len(path.node_list) - 1, 0)))
            if route_length > _modulation_reach(modulation):
                continue
            availability = route_availability(state.occupancy, link_ids)
            if hasattr(env, "j"):
                starts = self._deep_rmsa_blocks(env, route_id)
            else:
                starts = [(start, -1) for start in candidate_starts(availability, width, cfg.random_starts_per_route, rng)]
            for block_index, (start, _length) in enumerate(starts):
                if start + width > state.slot_count or int(availability[start : start + width].sum()) != width:
                    continue
                action = self._legacy_action(env, route_id, start, block_index)
                candidates.append(
                    score_candidate(
                        action=action,
                        route_id=route_id,
                        modulation_index=0,
                        modulation_offset=0,
                        b_start=int(start),
                        width=int(width),
                        route_node_ids=tuple(path.node_list),
                        route_link_ids=link_ids,
                        route_length_km=route_length,
                        hop_count=hop_count,
                        spectral_efficiency=_modulation_efficiency(modulation),
                        modulation_name=_modulation_name(modulation),
                        modulation_reach_km=_modulation_reach(modulation),
                        transponder_power_w=float(_get_field(modulation, "transponder_power_w", default=cfg.transponder_power_w)),
                        state=state,
                        cfg=cfg,
                    )
                )
        return candidates
