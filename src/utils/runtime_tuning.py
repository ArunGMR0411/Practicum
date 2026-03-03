"""Runtime tuning helpers for GPU-backed research backends."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from src.utils.system_config import resolve_torch_device

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TorchRuntimeTuning:
    device: str
    cpu_threads: int
    interop_threads: int
    tf32_enabled: bool
    cudnn_benchmark: bool
    float32_matmul_precision: str


def _recommended_cpu_threads(device: str, process_share: float) -> int:
    total_cpus = os.cpu_count() or 1
    if device == "cuda":
        # Use most host cores on accelerator boxes while keeping a small margin
        # for the OS and any auxiliary worker processes.
        target = max(3, int(total_cpus * 0.8))
        scaled = max(1, int(math.floor(target * max(0.0, min(1.0, process_share)))))
        return max(1, min(total_cpus, scaled))
    return max(1, min(total_cpus, 8))


def configure_torch_runtime(device: str, *, process_share: float = 1.0) -> TorchRuntimeTuning:
    """Apply safe runtime tuning for the current host.

    The goal is not to peg every resource blindly. Instead, we enable the
    highest-throughput settings that are safe for repeated inference on the
    current accelerator class without materially increasing OOM risk.
    """

    cpu_threads = _recommended_cpu_threads(device, process_share)
    interop_threads = max(1, min(4, cpu_threads))
    tf32_enabled = False
    cudnn_benchmark = False
    matmul_precision = "highest"

    if torch is not None:
        try:
            torch.set_num_threads(cpu_threads)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass

        if device == "cuda":
            tf32_enabled = True
            cudnn_benchmark = True
            matmul_precision = "high"
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except AttributeError:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            except AttributeError:
                pass
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(matmul_precision)

    return TorchRuntimeTuning(
        device=device,
        cpu_threads=cpu_threads,
        interop_threads=interop_threads,
        tf32_enabled=tf32_enabled,
        cudnn_benchmark=cudnn_benchmark,
        float32_matmul_precision=matmul_precision,
    )


def runtime_device_from_config() -> str:
    """Resolve ``auto`` against the live host instead of treating it as CPU."""
    device = resolve_torch_device()
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"
