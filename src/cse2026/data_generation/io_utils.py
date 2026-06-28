from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(stable_json(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % (2**32)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet(path: str | Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_npz_deterministic(path: str | Path, **arrays: np.ndarray) -> None:
    """Write an NPZ file with stable ZIP metadata for checksum reproducibility."""
    path = Path(path)
    ensure_dir(path.parent)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            buffer = BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, buffer.getvalue())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_artifact_files(root: str | Path) -> Iterable[Path]:
    root = Path(root)
    excluded = {"manifest.json", "checksums.sha256"}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in excluded:
            continue
        yield path


def build_checksums(root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in iter_artifact_files(root)}


def write_checksums(path: str | Path, checksums: dict[str, str]) -> None:
    lines = [f"{value}  {name}" for name, value in sorted(checksums.items())]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def stringify(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(x) for x in json.loads(value)]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [int(x) for x in value]


def expand_seeds(split_cfg: dict[str, Any]) -> list[int]:
    if "seeds" in split_cfg:
        return [int(seed) for seed in split_cfg["seeds"]]
    return list(range(int(split_cfg["seed_start"]), int(split_cfg["seed_stop"])))

