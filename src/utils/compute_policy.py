"""Runtime compute-policy helpers for automatic device and batch selection."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from src.utils.system_config import (
    read_host_memory_gb,
    load_system_config,
    read_accelerator_memory_gb,
    read_safe_batch_size,
    resolve_torch_device,
)


@dataclass(frozen=True)
class ComputePolicy:
    device: str
    accelerator_total_gb: float
    accelerator_available_gb: float
    host_ram_total_gb: float
    host_ram_available_gb: float
    detection_batch_size: int
    detection_num_workers: int
    reid_batch_size: int
    fid_batch_size: int
    ocr_region_batch_size: int
    generative_control_max_workers: int
    generative_method_workers: dict[str, int]
    use_mixed_precision: bool
    use_low_vram_mode: bool
    source_config: dict[str, Any]


def _safe_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return max(1, int(value))
    except ValueError:
        return None


def _scaled_batch(base: int, accelerator_total_gb: float, *, floor: int = 1, cap: int | None = None) -> int:
    if accelerator_total_gb <= 0:
        value = base
    elif accelerator_total_gb >= 70:
        value = base * 4
    elif accelerator_total_gb >= 40:
        value = base * 2
    elif accelerator_total_gb >= 20:
        value = int(base * 1.5)
    else:
        value = base
    value = max(floor, value)
    if cap is not None:
        value = min(value, cap)
    return value


def _derive_reid_batch(accelerator_total_gb: float) -> int:
    """Choose the Re-ID batch size from available VRAM."""
    override = _safe_int_env("REID_BATCH_SIZE")
    if override is not None:
        return override
    if accelerator_total_gb >= 70:
        return 256
    if accelerator_total_gb >= 40:
        return 128
    if accelerator_total_gb >= 20:
        return 64
    if accelerator_total_gb >= 10:
        return 48
    if accelerator_total_gb > 0:
        return 16
    return 8


def _derive_generative_workers(
    *,
    device: str,
    accelerator_total_gb: float,
    accelerator_available_gb: float,
    ram_total_gb: float,
    ram_available_gb: float,
) -> int:
    override = _safe_int_env("GENERATIVE_CONTROL_MAX_WORKERS")
    if override is not None:
        return override

    if device != "cuda":
        return 1

    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_budget_threads = max(1, math.floor(logical_cpus * 0.8))
    # These heavy workers are GPU-led, so keep the per-worker CPU reservation
    # modest enough that accelerator boxes can run several model processes in
    # parallel rather than stalling on an overly conservative host-core budget.
    min_threads_per_worker = 4
    cpu_limit = max(1, cpu_budget_threads // min_threads_per_worker)

    gpu_available = accelerator_available_gb if accelerator_available_gb > 0 else accelerator_total_gb
    ram_available = ram_available_gb if ram_available_gb > 0 else ram_total_gb

    # NullFace-class heavy workers keep a full generative stack resident.
    # Keep explicit headroom and derive concurrency from the tightest resource.
    gpu_headroom_gb = 4.0
    ram_headroom_gb = 16.0 if ram_total_gb >= 64 else max(4.0, ram_total_gb * 0.15)
    per_worker_gpu_gb = 7.5
    per_worker_ram_gb = 10.0

    gpu_limit = max(1, math.floor(max(0.0, gpu_available - gpu_headroom_gb) / per_worker_gpu_gb))
    ram_limit = max(1, math.floor(max(0.0, ram_available - ram_headroom_gb) / per_worker_ram_gb))

    derived = min(cpu_limit, gpu_limit, ram_limit)

    # Avoid exploding process count on very large hosts without real evidence that
    # a higher count helps these heavy image-generation backends.
    if accelerator_total_gb >= 70:
        cap = 6
    elif accelerator_total_gb >= 36:
        cap = 4
    elif accelerator_total_gb >= 20:
        cap = 3
    else:
        cap = 2
    return max(1, min(derived, cap))


def _derive_detection_num_workers(*, ram_available_gb: float) -> int:
    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_limit = max(1, math.floor(logical_cpus * 0.8) - 1)
    if ram_available_gb <= 0:
        ram_limit = 2
    else:
        # Native 4K image decode is memory-heavy; reserve room for the OS and the caller.
        ram_limit = max(1, math.floor(max(0.0, ram_available_gb - 4.0) / 2.5))
    return max(1, min(cpu_limit, ram_limit))


def _derive_generative_workers_for_method(
    method_name: str,
    *,
    device: str,
    accelerator_total_gb: float,
    accelerator_available_gb: float,
    ram_total_gb: float,
    ram_available_gb: float,
) -> int:
    if device != "cuda":
        return 1

    worker_override = _safe_int_env(f"{method_name.upper()}_MAX_WORKERS")
    if worker_override is not None:
        return worker_override

    profiles: dict[str, dict[str, float]] = {
        "stylegan": {"gpu_gb": 5.0, "ram_gb": 5.0, "gpu_headroom_gb": 3.0, "ram_headroom_gb": 8.0, "cap": 6.0},
        "nullface": {"gpu_gb": 7.5, "ram_gb": 10.0, "gpu_headroom_gb": 4.0, "ram_headroom_gb": 16.0, "cap": 4.0},
        "fams": {"gpu_gb": 9.0, "ram_gb": 12.0, "gpu_headroom_gb": 4.0, "ram_headroom_gb": 16.0, "cap": 3.0},
        "reverse_personalization": {"gpu_gb": 14.0, "ram_gb": 14.0, "gpu_headroom_gb": 6.0, "ram_headroom_gb": 20.0, "cap": 2.0},
        "diffusion": {"gpu_gb": 8.0, "ram_gb": 8.0, "gpu_headroom_gb": 4.0, "ram_headroom_gb": 12.0, "cap": 3.0},
    }
    profile = profiles.get(
        method_name,
        {"gpu_gb": 8.0, "ram_gb": 8.0, "gpu_headroom_gb": 4.0, "ram_headroom_gb": 12.0, "cap": 2.0},
    )

    logical_cpus = max(1, os.cpu_count() or 1)
    cpu_budget_threads = max(1, math.floor(logical_cpus * 0.8))
    min_threads_per_worker = 4
    cpu_limit = max(1, cpu_budget_threads // min_threads_per_worker)

    gpu_available = accelerator_available_gb if accelerator_available_gb > 0 else accelerator_total_gb
    ram_available = ram_available_gb if ram_available_gb > 0 else ram_total_gb
    gpu_limit = max(1, math.floor(max(0.0, gpu_available - profile["gpu_headroom_gb"]) / profile["gpu_gb"]))
    ram_limit = max(1, math.floor(max(0.0, ram_available - profile["ram_headroom_gb"]) / profile["ram_gb"]))
    return max(1, min(cpu_limit, gpu_limit, ram_limit, int(profile["cap"])))


def build_compute_policy() -> ComputePolicy:
    payload = load_system_config()
    device = resolve_torch_device()
    accelerator_total_gb, accelerator_available_gb = read_accelerator_memory_gb()
    ram_total_gb, ram_available_gb = read_host_memory_gb()

    # Use existing conservative safe sizes as the floor, then scale up on larger accelerators.
    base_fid = read_safe_batch_size("fid_eval_299", fallback=32)
    base_detection = read_safe_batch_size("detection_4k", fallback=1)

    fid_batch = _scaled_batch(base_fid, accelerator_total_gb, floor=8, cap=256)
    reid_batch = _derive_reid_batch(accelerator_total_gb)
    ocr_batch = _scaled_batch(8, accelerator_total_gb, floor=4, cap=64)
    detection_batch = max(1, min(base_detection, 8))
    detection_workers = _derive_detection_num_workers(ram_available_gb=ram_available_gb)

    generative_workers = _derive_generative_workers(
        device=device,
        accelerator_total_gb=accelerator_total_gb,
        accelerator_available_gb=accelerator_available_gb,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
    )
    generative_method_workers = {
        method_name: _derive_generative_workers_for_method(
            method_name,
            device=device,
            accelerator_total_gb=accelerator_total_gb,
            accelerator_available_gb=accelerator_available_gb,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
        )
        for method_name in ("stylegan", "nullface", "fams", "reverse_personalization", "diffusion")
    }

    return ComputePolicy(
        device=device,
        accelerator_total_gb=accelerator_total_gb,
        accelerator_available_gb=accelerator_available_gb,
        host_ram_total_gb=ram_total_gb,
        host_ram_available_gb=ram_available_gb,
        detection_batch_size=detection_batch,
        detection_num_workers=detection_workers,
        reid_batch_size=reid_batch,
        fid_batch_size=fid_batch,
        ocr_region_batch_size=ocr_batch,
        generative_control_max_workers=generative_workers,
        generative_method_workers=generative_method_workers,
        use_mixed_precision=device == "cuda",
        use_low_vram_mode=0.0 < accelerator_total_gb < 10.0,
        source_config=payload,
    )
