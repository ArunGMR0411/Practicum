#!/usr/bin/env python3

"""Run a tiny reviewed qualitative gate across advanced anonymisers."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_BOXES = PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500" / "manifest.csv"
DEFAULT_RELATIVE_PATHS = (
    "day1/members/allie/10_0122.webp",
    "day1/members/bjorn/11_0481.webp",
    "day4/members/florian/14_0144.webp",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "advanced_qualitative_gate"
STYLEGAN_MODEL = PROJECT_ROOT / "third_party" / "styleid" / "pretrained_models" / "stylegan2-ffhq-config-f.pt"


def manifest_raw_root(manifest_path: Path) -> Path:
    if manifest_path.name == "raw_dataset_index.csv":
        return manifest_path.parent / "raw"
    return manifest_path.parent.parent / "raw"


def load_boxes_by_image(boxes_path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    with boxes_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped.setdefault(row["image_id"], []).append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return grouped


def tail_text(value: str | None, length: int) -> str:
    if value is None:
        return ""
    return value.strip()[-length:]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def build_method_command(method_name: str, input_path: Path, output_path: Path, boxes_json: Path) -> list[str]:
    if method_name == "nullface":
        return [str(PYTHON), str(PROJECT_ROOT / "scripts" / "run_nullface.py"), "--input", str(input_path), "--output", str(output_path), "--boxes-json", str(boxes_json)]
    if method_name == "fams":
        return [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_fams.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--boxes-json",
            str(boxes_json),
            "--face-image-size",
            "96",
            "--num-inference-steps",
            "6",
            "--guidance-scale",
            "2.0",
        ]
    if method_name == "stylegan":
        return [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_stylegan.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--boxes-json",
            str(boxes_json),
            "--model-path",
            str(STYLEGAN_MODEL),
        ]
    raise ValueError(f"Unsupported method: {method_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes-csv", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--methods", nargs="+", default=["nullface", "fams", "stylegan"])
    parser.add_argument("--relative-path", dest="relative_paths", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relative_paths = tuple(args.relative_paths) if args.relative_paths else DEFAULT_RELATIVE_PATHS
    boxes_by_image = load_boxes_by_image(args.boxes_csv)
    raw_root = manifest_raw_root(args.manifest)
    output_root = args.output_root
    jobs_dir = output_root / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    results: list[dict[str, object]] = []
    for relative_path in relative_paths:
        boxes = boxes_by_image.get(relative_path, [])
        if not boxes:
            raise ValueError(f"No reviewed boxes found for {relative_path}")
        input_path = raw_root / relative_path
        if not input_path.is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        boxes_json = jobs_dir / (relative_path.replace("/", "__") + ".json")
        boxes_json.write_text(json.dumps({"boxes": boxes}, indent=2) + "\n", encoding="utf-8")
        for method_name in args.methods:
            output_path = output_root / method_name / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and output_path.is_file():
                results.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "boxes_count": len(boxes),
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "returncode": 0,
                        "runtime_seconds": 0.0,
                        "stdout_tail": "resumed: existing output retained",
                        "stderr_tail": "",
                        "status": "resumed",
                    }
                )
                continue
            command = build_method_command(method_name, input_path, output_path, boxes_json)
            started = time.perf_counter()
            result = run_command(command)
            runtime_seconds = round(time.perf_counter() - started, 3)
            results.append(
                {
                    "relative_path": relative_path,
                    "method": method_name,
                    "boxes_count": len(boxes),
                    "output_path": str(output_path.relative_to(PROJECT_ROOT)) if output_path.is_file() else "",
                    "returncode": result.returncode,
                    "runtime_seconds": runtime_seconds,
                    "stdout_tail": tail_text(result.stdout, 1000),
                    "stderr_tail": tail_text(result.stderr, 2000),
                    "status": "ok" if result.returncode == 0 and output_path.is_file() else "error",
                }
            )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "method", "boxes_count", "output_path", "returncode", "runtime_seconds"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "relative_path": row["relative_path"],
                    "method": row["method"],
                    "boxes_count": row["boxes_count"],
                    "output_path": row["output_path"],
                    "returncode": row["returncode"],
                    "runtime_seconds": row["runtime_seconds"],
                }
            )

    summary = {
        "relative_paths": list(relative_paths),
        "methods": list(args.methods),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "results": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
