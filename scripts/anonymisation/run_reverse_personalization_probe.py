#!/usr/bin/env python3

"""Run a tiny low-VRAM probe through Reverse Personalization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUNNER = PROJECT_ROOT / "scripts" / "anonymisation" / "run_reverse_personalization.py"
SOURCE_IMAGE = (
    PROJECT_ROOT
    / "third_party"
    / "reverse_personalization"
    / "assets"
    / "images"
    / "teaser"
    / "06496-orig.jpg"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reverse_personalization_probe"
DEFAULT_CROP_PATH = DEFAULT_OUTPUT_DIR / "erling_face_crop.png"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "erling_face_crop_anon.png"

from PIL import Image

from src.utils.system_config import read_accelerator_memory_gb


def tail_text(value: str | bytes | None, length: int) -> str:
    """Return a JSON-safe tail string from subprocess output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.strip()[-length:]


def build_face_crop(source_image: Path, output_path: Path) -> Path:
    """Create a deterministic face-focused crop for the probe."""
    with Image.open(source_image) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        crop_box = (
            max(0, width // 2 - 224),
            max(0, height // 2 - 224),
            min(width, width // 2 + 224),
            min(height, height // 2 + 224),
        )
        rgb.crop(crop_box).save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Hard timeout for the low-VRAM probe subprocess.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = build_face_crop(SOURCE_IMAGE, DEFAULT_CROP_PATH)
    total_gb, available_gb = read_accelerator_memory_gb()
    previous_output_exists = DEFAULT_OUTPUT_PATH.is_file()
    command = [
        str(PYTHON),
        str(RUNNER),
        "--input",
        str(crop_path),
        "--output",
        str(DEFAULT_OUTPUT_PATH),
        "--num-inversion-steps",
        "1",
        "--face-image-size",
        "256",
        "--det-size",
        "256",
        "--use-model-cpu-offload",
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=args.timeout_seconds,
        )
        returncode = result.returncode
        stdout_tail = tail_text(result.stdout, 500)
        stderr_tail = tail_text(result.stderr, 1000)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout_tail = tail_text(exc.stdout, 500)
        stderr_tail = tail_text(exc.stderr, 1000)

    summary = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_profile": "rtx_constrained-compute profile_mobile_8gb_offload",
        "accelerator_memory_total_gb": total_gb,
        "accelerator_memory_available_gb": available_gb,
        "timeout_seconds": args.timeout_seconds,
        "validated_command": " ".join(command),
        "input_path": str(crop_path),
        "output_path": str(DEFAULT_OUTPUT_PATH),
        "previous_output_exists": previous_output_exists,
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "output_exists": DEFAULT_OUTPUT_PATH.is_file(),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "notes": [
            "This is the smallest verified verified execution path for the 8 GB laptop GPU.",
            "It validates environment wiring and low-VRAM offload behavior, not anonymization quality.",
            "Meaningful generative quality evaluation should still be run on constrained-compute or constrained-compute after environment verification.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
