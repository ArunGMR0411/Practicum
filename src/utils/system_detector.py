#!/usr/bin/env python3

"""Detect runtime class and write a reusable constrained-resource system config."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "system_config.json"
DEFAULT_ENV_LABEL_PATH = PROJECT_ROOT / "outputs" / "detected_environment_label.txt"
ENVIRONMENT_LABEL = "memory-and-compute-constrained"

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    """Parse output paths for config generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)),
        help="Config JSON path relative to project root.",
    )
    parser.add_argument(
        "--label-output",
        default=str(DEFAULT_ENV_LABEL_PATH.relative_to(PROJECT_ROOT)),
        help="Detected environment label file path relative to project root.",
    )
    return parser.parse_args()


def has_xla() -> bool:
    """Return True when a TPU/XLA runtime is importable."""
    return shutil.which("python") is not None and _importable("torch_xla")


def _importable(module_name: str) -> bool:
    """Return True when a Python module can be imported in the current interpreter."""
    try:
        __import__(module_name)
    except Exception:
        return False
    return True


def detect_runtime_label() -> tuple[str, str]:
    """Detect the preferred runtime branch and runtime device type."""
    if has_xla():
        return "xla", "xla"
    if torch is not None and torch.cuda.is_available():
        return "cuda", "cuda"
    return "cpu_or_laptop", "cpu"


def accelerator_memory_gb(device_type: str) -> tuple[float, float]:
    """Return total and available accelerator memory in GiB when accessible."""
    if device_type == "cuda" and torch is not None:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        available = max(props.total_memory - max(reserved, allocated), 0) / (1024 ** 3)
        return total, available
    return 0.0, 0.0


def safe_batch_size(available_bytes: float, model_footprint_mb: float, input_shape: tuple[int, ...]) -> int:
    """Estimate a conservative safe batch size under constrained resources."""
    element_count = 1
    for dim in input_shape:
        element_count *= dim
    bytes_per_sample = element_count * 4
    model_bytes = model_footprint_mb * 1024 * 1024
    usable_budget = max(available_bytes * 0.8 - model_bytes, bytes_per_sample)
    return max(1, int(usable_budget // bytes_per_sample))


def write_atomic_text(path: Path, text: str) -> None:
    """Write a text file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(path)


def main() -> None:
    """Detect the runtime and persist a reusable system configuration."""
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    label_output_path = PROJECT_ROOT / args.label_output

    runtime_label, device_type = detect_runtime_label()
    vm = psutil.virtual_memory()
    accel_total_gb, accel_available_gb = accelerator_memory_gb(device_type)
    working_bytes = int(accel_available_gb * (1024 ** 3)) if accel_available_gb > 0 else vm.available

    payload = {
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "environment_label": ENVIRONMENT_LABEL,
        "runtime_label": runtime_label,
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
    }

    write_atomic_json(output_path, payload)
    write_atomic_text(label_output_path, f"{runtime_label}\n")

    print(f"Saved system config to {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Detected runtime label: {runtime_label}")
    print(
        f"Environment: {ENVIRONMENT_LABEL} | Device: {device_type} | "
        f"RAM total: {payload['ram_total_gb']:.2f} GiB | "
        f"Accelerator total: {payload['accelerator_memory_total_gb']:.2f} GiB"
    )
    print(
        "Safe batch sizes | "
        f"detection_4k={payload['safe_batch_sizes']['detection_4k']} "
        f"anonymisation_crop_256={payload['safe_batch_sizes']['anonymisation_crop_256']} "
        f"fid_eval_299={payload['safe_batch_sizes']['fid_eval_299']}"
    )


if __name__ == "__main__":
    main()
