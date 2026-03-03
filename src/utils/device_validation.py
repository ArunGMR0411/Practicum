#!/usr/bin/env python3

"""Run basic environment checks under memory and compute-constrained resources."""

from __future__ import annotations

import csv
import io
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from PIL import Image

MANIFEST_PATH = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"


def detect_device(torch_module) -> str:
    """Detect the active device class available to the current runtime."""
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def image_shape(image_obj) -> tuple[int, ...]:
    """Return a non-empty image shape tuple for PIL or NumPy image objects."""
    if hasattr(image_obj, "shape"):
        return tuple(int(dim) for dim in image_obj.shape)
    if hasattr(image_obj, "size"):
        width, height = image_obj.size
        bands = len(getattr(image_obj, "getbands", lambda: ())()) or 0
        return (height, width, bands) if bands else (height, width)
    return ()


def estimate_memory(torch_module, device_type: str) -> tuple[float | None, float | None]:
    """Estimate total and available accelerator memory in GiB when accessible."""
    if device_type == "cuda":
        props = torch_module.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        reserved = torch_module.cuda.memory_reserved(0)
        allocated = torch_module.cuda.memory_allocated(0)
        available = max(props.total_memory - max(reserved, allocated), 0) / (1024 ** 3)
        return total, available
    return None, None


def main() -> int:
    """Execute 10 smoke-test checks and exit non-zero if any check fails."""
    passed = 0
    total = 10
    torch_module = None
    active_device = "unavailable"

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal passed
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"{status} | {name} | {detail}")

    py_ok = sys.version_info >= (3, 10)
    report("python_version", py_ok, platform.python_version())

    try:
        import torch

        torch_module = torch
        report("torch_import", True, f"imported torch {torch.__version__}")
    except Exception as exc:  # pragma: no cover - environment-dependent
        report("torch_import", False, f"{type(exc).__name__}: {exc}")

    if torch_module is not None:
        try:
            active_device = detect_device(torch_module)
            report("device_availability", True, f"active device: {active_device}")
        except Exception as exc:
            report("device_availability", False, f"{type(exc).__name__}: {exc}")

        try:
            tensor = torch_module.zeros((1, 3, 224, 224), dtype=torch_module.float32, device=active_device)
            report("tensor_creation", True, f"shape={tuple(tensor.shape)} device={tensor.device}")
        except Exception as exc:
            tensor = None
            report("tensor_creation", False, f"{type(exc).__name__}: {exc}")

        try:
            doubled = tensor * 2  # type: ignore[operator]
            ok = str(doubled.dtype).endswith("float32")
            report("basic_arithmetic", ok, f"dtype={doubled.dtype}")
        except Exception as exc:
            report("basic_arithmetic", False, f"{type(exc).__name__}: {exc}")
    else:
        report("device_availability", False, "torch unavailable")
        report("tensor_creation", False, "torch unavailable")
        report("basic_arithmetic", False, "torch unavailable")

    try:
        image = Image.new("RGB", (8, 8), color=(12, 34, 56))
        buffer = io.BytesIO()
        image.save(buffer, format="WEBP")
        buffer.seek(0)
        decoded = Image.open(buffer)
        decoded.load()
        report("pillow_webp_decode", True, f"decoded size={decoded.size}")
    except Exception as exc:
        report("pillow_webp_decode", False, f"{type(exc).__name__}: {exc}")

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            first_row = next(reader)
        ok = bool(header and first_row)
        report("manifest_read", ok, f"path={MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        report("manifest_read", False, f"{type(exc).__name__}: {exc}")

    try:
        from src.data.castle_loader import CASTLEDataset

        report("loader_import", True, f"imported {CASTLEDataset.__name__}")
    except Exception as exc:
        CASTLEDataset = None  # type: ignore[assignment]
        report("loader_import", False, f"{type(exc).__name__}: {exc}")

    try:
        dataset = CASTLEDataset("data/castle2024/raw_dataset_index.csv")  # type: ignore[misc]
        item = dataset[0]
        image_obj = item["image"]
        shape = image_shape(image_obj)
        non_zero = bool(shape) and all(int(dim) > 0 for dim in shape[:2])
        report("single_frame_load", non_zero, f"image_shape={shape}")
    except Exception as exc:
        report("single_frame_load", False, f"{type(exc).__name__}: {exc}")

    try:
        ram = psutil.virtual_memory()
        accel_total, accel_available = (None, None)
        if torch_module is not None:
            accel_total, accel_available = estimate_memory(torch_module, active_device)
        detail = (
            f"ram_total_gb={ram.total / (1024 ** 3):.2f} "
            f"ram_available_gb={ram.available / (1024 ** 3):.2f} "
            f"accelerator_total_gb={0.0 if accel_total is None else accel_total:.2f} "
            f"accelerator_available_gb={0.0 if accel_available is None else accel_available:.2f}"
        )
        report("memory_ceiling_estimate", True, detail)
    except Exception as exc:
        report("memory_ceiling_estimate", False, f"{type(exc).__name__}: {exc}")

    print(f"{passed}/{total} checks passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
