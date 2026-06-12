#!/usr/bin/env python3

"""Run the smallest real CASTLE probe for the FAMS adapter."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
RUNNER = PROJECT_ROOT / "scripts" / "run_fams.py"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_BOXES = PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500" / "manifest.csv"
DEFAULT_RELATIVE_PATH = "day1/members/allie/10_0122.webp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fams_probe"


def tail_text(value: str | bytes | None, length: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value.strip()[-length:]


def manifest_raw_root(manifest_path: Path) -> Path:
    if manifest_path.name == "raw_dataset_index.csv":
        return manifest_path.parent / "raw"
    return manifest_path.parent.parent / "raw"


def load_largest_box(boxes_path: Path, relative_path: str) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    with boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["image_id"] != relative_path:
                continue
            box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            candidates.append(box)
    if not candidates:
        raise ValueError(f"No face boxes found for {relative_path} in {boxes_path}")
    return max(candidates, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes-csv", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--relative-path", default=DEFAULT_RELATIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = manifest_raw_root(args.manifest) / args.relative_path
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    box = load_largest_box(args.boxes_csv, args.relative_path)
    boxes_json = output_dir / "boxes.json"
    output_path = output_dir / "fams_output.webp"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.csv"
    boxes_json.write_text(json.dumps({"boxes": [box]}, indent=2) + "\n", encoding="utf-8")

    command = [
        str(PYTHON),
        str(RUNNER),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--boxes-json",
        str(boxes_json),
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
        stdout_tail = tail_text(result.stdout, 1000)
        stderr_tail = tail_text(result.stderr, 2000)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout_tail = tail_text(exc.stdout, 1000)
        stderr_tail = tail_text(exc.stderr, 2000)

    runtime_seconds = round(time.perf_counter() - started, 3)
    output_exists = output_path.is_file()
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "method", "output_path", "boxes_processed", "runtime_seconds"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "relative_path": args.relative_path,
                "method": "fams",
                "output_path": str(output_path.relative_to(PROJECT_ROOT)) if output_exists else "",
                "boxes_processed": 1 if output_exists and returncode == 0 else 0,
                "runtime_seconds": runtime_seconds,
            }
        )

    summary = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "fams",
        "backend": "Face Anonymization Made Simple",
        "relative_path": args.relative_path,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "boxes_json": str(boxes_json),
        "selected_box": box,
        "timeout_seconds": args.timeout_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": runtime_seconds,
        "output_exists": output_exists,
        "validated_command": " ".join(command),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "notes": [
            "Minimum real CASTLE smoke for FAMS adapter wiring.",
            "This validates command, dependency, model-loading, and image-write paths before broader dev comparison.",
            "If Hugging Face weights are not cached, the first run may spend most of its time downloading.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
