#!/usr/bin/env python3

"""Helpers for loading the project runtime system configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "system_config.json"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]


def load_system_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the runtime system configuration from JSON."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_safe_batch_size(
    workload: str,
    fallback: int = 1,
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> int:
    """Read a safe batch size for the named workload with a minimum of one."""
    payload = load_system_config(path)
    safe_batch_sizes = payload.get("safe_batch_sizes", {})
    value = int(safe_batch_sizes.get(workload, fallback))
    return max(1, value)


def read_accelerator_memory_gb(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[float, float]:
    """Read live accelerator memory in GiB, falling back to the profiled config."""
    if torch is not None and torch.cuda.is_available():
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gib = 1024**3
            return round(total_bytes / gib, 4), round(free_bytes / gib, 4)
        except Exception:
            pass
    payload = load_system_config(path)
    total = float(payload.get("accelerator_memory_total_gb", 0.0) or 0.0)
    available = float(payload.get("accelerator_memory_available_gb", 0.0) or 0.0)
    return total, available


def read_host_memory_gb(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[float, float]:
    """Read live host RAM, falling back to the profiled configuration."""
    try:
        memory = psutil.virtual_memory()
        gib = 1024**3
        return round(memory.total / gib, 4), round(memory.available / gib, 4)
    except Exception:
        payload = load_system_config(path)
        total = float(payload.get("ram_total_gb", 0.0) or 0.0)
        available = float(payload.get("ram_available_gb", 0.0) or 0.0)
        return total, available


def resolve_torch_device(path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    """Resolve the preferred PyTorch device with safe runtime fallback."""
    payload = load_system_config(path)
    preferred = str(payload.get("runtime_label") or payload.get("device_type") or "cpu").strip().lower()
    if preferred == "cuda" and torch is not None and torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None) if torch is not None else None
    if preferred == "mps" and mps is not None and mps.is_available():
        return "mps"
    if preferred == "xla":
        return "cpu"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
