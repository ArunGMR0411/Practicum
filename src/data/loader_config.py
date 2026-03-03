"""Configuration helpers for the CASTLE dataset loader."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.system_config import read_safe_batch_size


def _available_memory_bytes() -> int | None:
    """Return available system memory on POSIX systems when detectable."""
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None
    return int(page_size) * int(pages)


@dataclass
class LoaderConfig:
    """Serialisable configuration for native-resolution CASTLE frame loading.

    Defaults preserve source-resolution, single-frame, no-preprocessing behaviour.
    Optional tiling, staged resizing, and runtime recommendations are explicit and
    must be enabled by the caller when a downstream method requires them.
    """

    manifest_path: str = "data/castle2024/raw_dataset_index.csv"
    """Relative path to the CASTLE manifest CSV."""

    return_format: str = "pil"
    """Image return type: either ``'pil'`` or ``'numpy'``."""

    eval_scale: float | None = None
    """Optional post-decode evaluation-time scale factor; ``None`` keeps native size."""

    enable_tiling: bool = False
    """Enable tile-based downstream handling only when a method explicitly needs it."""

    tile_size: int = 768
    """Tile size in pixels when tiling is enabled."""

    tile_overlap: int = 64
    """Tile overlap in pixels when tiling is enabled."""

    staged_resize_steps: list[int] | None = None
    """Optional opt-in staged downscale targets, highest to lowest resolution."""

    filters: dict[str, str] | None = None
    """Optional manifest-column equality filters."""

    batch_size: int = 1
    """Default batch size; single-frame processing remains the default."""

    num_workers: int = 0
    """Default worker count; zero disables multiprocessing by default."""

    def validate(self) -> None:
        """Validate configuration consistency and raise clear errors on conflicts."""
        if self.return_format not in {"pil", "numpy"}:
            raise ValueError("return_format must be 'pil' or 'numpy'")
        if self.eval_scale is not None and self.eval_scale <= 0:
            raise ValueError("eval_scale must be greater than 0 when provided")
        if self.enable_tiling and self.eval_scale is not None:
            raise ValueError("enable_tiling cannot be combined with eval_scale")
        if self.enable_tiling and self.staged_resize_steps:
            raise ValueError("enable_tiling cannot be combined with staged_resize_steps")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be greater than 0")
        if self.tile_overlap < 0:
            raise ValueError("tile_overlap must be non-negative")
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.staged_resize_steps:
            if any(step <= 0 for step in self.staged_resize_steps):
                raise ValueError("staged_resize_steps values must be greater than 0")
            if self.staged_resize_steps != sorted(self.staged_resize_steps, reverse=True):
                raise ValueError("staged_resize_steps must be ordered from largest to smallest")

    def to_json(self, path: str | Path) -> None:
        """Write the configuration as indented JSON using an atomic rename."""
        self.validate()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(self), indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(output_path)

    @classmethod
    def from_json(cls, path: str | Path) -> "LoaderConfig":
        """Load configuration from a JSON file and validate the result."""
        input_path = Path(path)
        with input_path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
        config = cls(**data)
        config.validate()
        return config

    @classmethod
    def with_system_recommendations(cls, **overrides: Any) -> "LoaderConfig":
        """Create a config with advisory runtime settings based on available resources.

        This method does not change preprocessing defaults. It only suggests
        conservative worker and batch settings intended to use up to roughly 80% of
        available CPU parallelism and memory headroom while preserving one-frame
        processing semantics unless the caller overrides them.
        """

        cpu_count = os.cpu_count() or 1
        try:
            from src.utils.compute_policy import build_compute_policy

            policy = build_compute_policy()
            recommended_workers = max(0, int(policy.detection_num_workers))
            recommended_batch_size = max(1, int(policy.detection_batch_size))
        except Exception:
            recommended_workers = max(0, int(cpu_count * 0.8) - 1)
            recommended_batch_size = read_safe_batch_size("detection_4k", fallback=1)

        memory_bytes = _available_memory_bytes()
        if memory_bytes is not None:
            memory_budget_bytes = int(memory_bytes * 0.8)
            estimated_frame_bytes = 3840 * 2160 * 3
            memory_limited_batch = max(1, memory_budget_bytes // estimated_frame_bytes)
            memory_limited_batch = min(memory_limited_batch, max(1, cpu_count))
            recommended_batch_size = max(1, min(recommended_batch_size, memory_limited_batch))

        config = cls(
            batch_size=max(1, recommended_batch_size),
            num_workers=max(0, recommended_workers),
            **overrides,
        )
        config.validate()
        return config
