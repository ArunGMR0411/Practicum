#!/usr/bin/env python3

"""Run a small StyleID/FAMS tuning grid over reviewed CASTLE frames."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_BOXES = (
    PROJECT_ROOT
    / "data"
    / "castle2024"
    / "annotations"
    / "face_detection" / "02_egocentric_stress_500"
    / "manifest.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs" / "styleid_fams_tuning"
DEFAULT_STYLEID_MODEL = (
    PROJECT_ROOT / "third_party" / "styleid" / "pretrained_models" / "stylegan2-ffhq-config-f.pt"
)
DEFAULT_RELATIVE_PATHS = (
    "day1/members/allie/10_0122.webp",
    "day1/members/bjorn/11_0481.webp",
    "day4/members/florian/14_0144.webp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes-csv", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--relative-path", dest="relative_paths", action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", default=["stylegan", "fams"])
    parser.add_argument("--stylegan-truncation", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--stylegan-seed", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--fams-face-image-size", type=int, nargs="+", default=[96, 256, 512])
    parser.add_argument("--fams-steps", type=int, nargs="+", default=[6, 15, 25])
    parser.add_argument("--fams-guidance-scale", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--fams-anonymization-degree", type=float, nargs="+", default=[1.0, 1.25])
    parser.add_argument("--fams-overlap-iou-threshold", type=float, nargs="+", default=[0.15, 0.3])
    parser.add_argument("--max-jobs", type=int, default=0, help="Optional cap for smoke/debug runs.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def raw_root_from_manifest(manifest_path: Path) -> Path:
    if manifest_path.name == "raw_dataset_index.csv":
        return manifest_path.parent / "raw"
    return manifest_path.parent.parent / "raw"


def load_boxes(path: Path) -> dict[str, list[tuple[int, int, int, int]]]:
    grouped: dict[str, list[tuple[int, int, int, int]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped.setdefault(row["image_id"], []).append(
                (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            )
    return grouped


def write_boxes_json(path: Path, boxes: list[tuple[int, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"boxes": boxes}, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def tail(value: str, limit: int = 1200) -> str:
    return (value or "").strip()[-limit:]


def stylegan_jobs(args: argparse.Namespace, input_path: Path, output_dir: Path, boxes_json: Path) -> list[tuple[str, list[str], Path]]:
    jobs: list[tuple[str, list[str], Path]] = []
    for truncation, seed in itertools.product(args.stylegan_truncation, args.stylegan_seed):
        label = f"stylegan_trunc_{truncation:g}_seed_{seed}"
        output_path = output_dir / label / input_path.name
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_stylegan.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--boxes-json",
            str(boxes_json),
            "--model-path",
            str(DEFAULT_STYLEID_MODEL),
            "--truncation-psi",
            str(truncation),
            "--seed",
            str(seed),
        ]
        jobs.append((label, command, output_path))
    return jobs


def fams_jobs(args: argparse.Namespace, input_path: Path, output_dir: Path, boxes_json: Path) -> list[tuple[str, list[str], Path]]:
    jobs: list[tuple[str, list[str], Path]] = []
    grid = itertools.product(
        args.fams_face_image_size,
        args.fams_steps,
        args.fams_guidance_scale,
        args.fams_anonymization_degree,
        args.fams_overlap_iou_threshold,
    )
    for size, steps, guidance, degree, overlap in grid:
        label = f"fams_size_{size}_steps_{steps}_guide_{guidance:g}_degree_{degree:g}_iou_{overlap:g}"
        output_path = output_dir / label / input_path.name
        command = [
            str(PYTHON),
            str(PROJECT_ROOT / "scripts" / "run_fams.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--boxes-json",
            str(boxes_json),
            "--face-image-size",
            str(size),
            "--num-inference-steps",
            str(steps),
            "--guidance-scale",
            str(guidance),
            "--anonymization-degree",
            str(degree),
            "--overlap-iou-threshold",
            str(overlap),
        ]
        jobs.append((label, command, output_path))
    return jobs


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    relative_paths = tuple(args.relative_paths or DEFAULT_RELATIVE_PATHS)
    raw_root = raw_root_from_manifest(args.manifest)
    boxes_by_image = load_boxes(args.boxes_csv)
    jobs_dir = args.output_root / "_jobs"
    rows: list[dict[str, object]] = []
    job_count = 0

    for relative_path in relative_paths:
        input_path = raw_root / relative_path
        boxes = boxes_by_image.get(relative_path, [])
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if not boxes:
            raise ValueError(f"No reviewed boxes found for {relative_path}")
        boxes_json = jobs_dir / f"{relative_path.replace('/', '__')}.json"
        write_boxes_json(boxes_json, boxes)

        method_jobs: list[tuple[str, list[str], Path]] = []
        if "stylegan" in args.methods:
            method_jobs.extend(stylegan_jobs(args, input_path, args.output_root / "stylegan", boxes_json))
        if "fams" in args.methods:
            method_jobs.extend(fams_jobs(args, input_path, args.output_root / "fams", boxes_json))

        for label, command, output_path in method_jobs:
            if args.max_jobs and job_count >= args.max_jobs:
                break
            job_count += 1
            started = time.perf_counter()
            if args.resume and output_path.is_file():
                result = None
                status = "resumed"
                runtime = 0.0
                stdout_tail = "existing output retained"
                stderr_tail = ""
                returncode = 0
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result = run_command(command)
                runtime = round(time.perf_counter() - started, 3)
                returncode = result.returncode
                status = "ok" if returncode == 0 and output_path.is_file() else "error"
                stdout_tail = tail(result.stdout)
                stderr_tail = tail(result.stderr, 2500)
            rows.append(
                {
                    "relative_path": relative_path,
                    "variant": label,
                    "output_path": str(output_path.relative_to(PROJECT_ROOT)) if output_path.is_file() else "",
                    "boxes_count": len(boxes),
                    "runtime_seconds": runtime,
                    "returncode": returncode,
                    "status": status,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail,
                }
            )
        if args.max_jobs and job_count >= args.max_jobs:
            break

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "styleid_fams_tuning_grid_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "relative_path",
            "variant",
            "output_path",
            "boxes_count",
            "runtime_seconds",
            "returncode",
            "status",
            "stdout_tail",
            "stderr_tail",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "requested_images": list(relative_paths),
        "methods": args.methods,
        "jobs_attempted": len(rows),
        "jobs_ok": sum(1 for row in rows if row["status"] in {"ok", "resumed"}),
        "jobs_error": sum(1 for row in rows if row["status"] == "error"),
        "manifest": str(manifest_path),
    }
    (args.output_root / "styleid_fams_tuning_grid_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
