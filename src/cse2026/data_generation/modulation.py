from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io_utils import project_root


@dataclass(frozen=True)
class Modulation:
    modulation_id: int
    name: str
    spectral_efficiency: float
    reach_km: float
    snr_threshold_db: float
    transponder_power_w: float


DEFAULT_MODULATIONS: list[Modulation] = [
    Modulation(0, "BPSK", 1.0, 4000.0, 6.5, 90.0),
    Modulation(1, "QPSK", 2.0, 2000.0, 9.5, 100.0),
    Modulation(2, "8QAM", 3.0, 1000.0, 12.5, 120.0),
    Modulation(3, "16QAM", 4.0, 500.0, 15.5, 140.0),
]


def modulation_source_path() -> Path:
    return project_root() / "data" / "eon" / "modulations" / "flexgrid_modulations.csv"


def load_modulations(path: str | Path | None = None) -> list[Modulation]:
    src = Path(path) if path else modulation_source_path()
    if not src.exists():
        return DEFAULT_MODULATIONS
    frame = pd.read_csv(src)
    return [
        Modulation(
            modulation_id=int(row.modulation_id),
            name=str(row.name),
            spectral_efficiency=float(row.spectral_efficiency),
            reach_km=float(row.reach_km),
            snr_threshold_db=float(row.snr_threshold_db),
            transponder_power_w=float(row.transponder_power_w),
        )
        for row in frame.itertuples(index=False)
    ]


def required_slots(
    bit_rate_gbps: float,
    modulation: Modulation,
    slot_capacity_gbps_at_1bpshz: float,
    guard_band_slots: int,
) -> int:
    import math

    payload_slots = math.ceil(float(bit_rate_gbps) / (slot_capacity_gbps_at_1bpshz * modulation.spectral_efficiency))
    return int(payload_slots + guard_band_slots)


def modulations_by_id(modulations: list[Modulation]) -> dict[int, Modulation]:
    return {mod.modulation_id: mod for mod in modulations}

