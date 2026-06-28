from __future__ import annotations

import numpy as np


def route_availability(occupancy: np.ndarray, route_link_ids: list[int]) -> np.ndarray:
    if not route_link_ids:
        return np.ones(occupancy.shape[1], dtype=np.uint8)
    return (occupancy[np.asarray(route_link_ids, dtype=np.int64), :].sum(axis=0) == 0).astype(np.uint8)


def route_occupancy_fraction(occupancy: np.ndarray, route_link_ids: list[int]) -> np.ndarray:
    if not route_link_ids:
        return np.zeros(occupancy.shape[1], dtype=np.float32)
    return occupancy[np.asarray(route_link_ids, dtype=np.int64), :].mean(axis=0).astype(np.float32)


def contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.uint8)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def feasible_starts(mask: np.ndarray, width: int) -> list[int]:
    starts: list[int] = []
    for start, end in contiguous_segments(mask):
        if end - start >= width:
            starts.extend(range(start, end - width + 1))
    return starts


def largest_free_block(mask: np.ndarray) -> int:
    segments = contiguous_segments(mask)
    return int(max((end - start for start, end in segments), default=0))


def free_count(mask: np.ndarray) -> int:
    return int(np.asarray(mask).sum())


def segment_count(mask: np.ndarray) -> int:
    return len(contiguous_segments(mask))


def fragmentation(mask: np.ndarray, epsilon: float = 1e-9) -> float:
    n_free = free_count(mask)
    if n_free == 0:
        return 1.0
    return float(1.0 - largest_free_block(mask) / (n_free + epsilon))


def allocate_on_mask(mask: np.ndarray, start: int, width: int) -> np.ndarray:
    after = np.asarray(mask, dtype=np.uint8).copy()
    after[start : start + width] = 0
    return after


def local_fragmentation_context(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask), dtype=np.float32)
    total = float(len(mask))
    for start, end in contiguous_segments(mask):
        out[start:end] = 1.0 - float(end - start) / total
    return out


def future_release_for_route(release_times: np.ndarray, route_link_ids: list[int], now: float) -> np.ndarray:
    if not route_link_ids:
        return np.zeros(release_times.shape[1], dtype=np.float32)
    route_release = release_times[np.asarray(route_link_ids, dtype=np.int64), :]
    residual = np.maximum(route_release - float(now), 0.0)
    return residual.max(axis=0).astype(np.float32)

