from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Iterator, TextIO


def configure_logging(run_dir: str | Path) -> logging.Logger:
    log_dir = Path(run_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cse2026.experiments")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def tee_stdout(path: str | Path) -> Iterator[None]:
    original = sys.stdout
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            yield
        finally:
            sys.stdout = original
