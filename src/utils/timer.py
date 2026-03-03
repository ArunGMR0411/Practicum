#!/usr/bin/env python3

"""Lightweight wall-clock timing utilities for pipeline stages."""

from __future__ import annotations

import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class StageTimer:
    """Collect and persist per-stage wall-clock timings."""

    def __init__(self, output_path: str | Path = "outputs/timing.json") -> None:
        """Initialise the timer with an output path for the timing summary."""
        self.output_path = Path(output_path)
        self.records: dict[str, float] = {}

    @contextmanager
    def track(self, stage_name: str) -> Iterator[None]:
        """Context manager that records elapsed wall-clock time for one stage."""
        started_at = time.perf_counter()
        try:
            yield
        finally:
            self.records[stage_name] = self.records.get(stage_name, 0.0) + (
                time.perf_counter() - started_at
            )

    def save(self) -> None:
        """Write the timing summary as JSON using an atomic file replace."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.output_path.parent,
            delete=False,
        ) as handle:
            json.dump(self.records, handle, indent=2)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(self.output_path)
