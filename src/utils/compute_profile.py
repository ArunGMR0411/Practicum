#!/usr/bin/env python3

"""Profile runtime capacity under memory and compute-constrained resources."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "compute_profile_local.json"
ENVIRONMENT_LABEL = "memory-and-compute-constrained"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for compute profiling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)),
        help="Output JSON path relative to the project root.",
    )
    return parser.parse_args()


def detect_device_type() -> str:
    """Detect the active accelerator class visible to PyTorch."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def accelerator_memory(device_type: str) -> tuple[float, float]:
    """Return total and available accelerator memory in GiB when accessible."""
    if torch is None:
        return 0.0, 0.0
    if device_type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        available = max(props.total_memory - max(reserved, allocated), 0) / (1024 ** 3)
        return total, available
    return 0.0, 0.0


def safe_batch_size(available_bytes: float, model_footprint_mb: float, input_shape: tuple[int, ...]) -> int:
    """Estimate a conservative safe batch size from memory budget and tensor shape."""
    element_count = 1
    for dim in input_shape:
        element_count *= dim
    bytes_per_sample = element_count * 4
    model_bytes = model_footprint_mb * 1024 * 1024
    usable_budget = max(available_bytes * 0.8 - model_bytes, bytes_per_sample)
    return max(1, int(usable_budget // bytes_per_sample))


def write_atomic(payload: dict[str, object], output_path: Path) -> None:
    """Write JSON atomically to avoid partial output corruption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    """Compute a conservative resource profile and save it as JSON."""
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    vm = psutil.virtual_memory()
    device_type = detect_device_type()
    accel_total_gb, accel_available_gb = accelerator_memory(device_type)

    accel_available_bytes = accel_available_gb * (1024 ** 3)
    ram_available_bytes = vm.available
    working_bytes = accel_available_bytes if accel_available_bytes > 0 else ram_available_bytes

    notes = (
        "Profile generated for memory and compute-constrained resources. "
        "Batch sizes are conservative estimates only and do not load model weights."
    )
    if torch is None:
        notes += " PyTorch was not available in the execution environment, so the profile used CPU-only fallback logic."

    payload = {
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "environment_label": ENVIRONMENT_LABEL,
        "device_type": device_type,
        "ram_total_gb": round(vm.total / (1024 ** 3), 4),
        "ram_available_gb": round(vm.available / (1024 ** 3), 4),
        "accelerator_memory_total_gb": round(accel_total_gb, 4),
        "accelerator_memory_available_gb": round(accel_available_gb, 4),
        "safe_batch_sizes": {
            "detection_4k": safe_batch_size(working_bytes, 200, (3, 2160, 3840)),
            "anonymisation_crop_256": safe_batch_size(working_bytes, 500, (3, 256, 256)),
            "fid_eval_299": safe_batch_size(working_bytes, 100, (3, 299, 299)),
        },
        "notes": notes,
    }

    write_atomic(payload, output_path)
    print(f"Saved compute profile to {output_path.relative_to(PROJECT_ROOT)}")
    print(
        f"Environment: {payload['environment_label']} | Device: {device_type} | "
        f"RAM available: {payload['ram_available_gb']:.2f} GiB | "
        f"Accelerator available: {payload['accelerator_memory_available_gb']:.2f} GiB"
    )
    print(
        "Safe batch sizes | "
        f"detection_4k={payload['safe_batch_sizes']['detection_4k']} "
        f"anonymisation_crop_256={payload['safe_batch_sizes']['anonymisation_crop_256']} "
        f"fid_eval_299={payload['safe_batch_sizes']['fid_eval_299']}"
    )


if __name__ == "__main__":
    main()
