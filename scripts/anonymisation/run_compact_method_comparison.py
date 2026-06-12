#!/usr/bin/env python3

"""Run a compact reviewed RQ2 benchmark on a small reviewed slice."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.blur_anonymiser import BlurAnonymiser
from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
from src.anonymisation.fams_anonymiser import FAMSAnonymiser
from src.anonymisation.nullface_anonymiser import NullFaceAnonymiser
from src.anonymisation.pixelate_anonymiser import PixelateAnonymiser
from src.anonymisation.stylegan_anonymiser import StyleGANAnonymiser


DEFAULT_IMAGES = (
    "day1/members/allie/09_0380.webp",
    "day1/members/bjorn/08_0558.webp",
    "day4/members/allie/08_0010.webp",
    "day4/members/florian/14_0144.webp",
    "day4/members/luca/12_0591.webp",
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "runs" / "compact_benchmark"
BOX_SOURCES = (
    PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500" / "manifest.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relative-path", dest="relative_paths", action="append")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["blur", "pixelate", "diffusion", "nullface", "stylegan", "fams"],
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def build_manifest_rows(relative_paths: tuple[str, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative_path in relative_paths:
        day_id, view_type, camera_stream_id, filename = relative_path.split("/")
        rows.append(
            {
                "relative_path": relative_path,
                "day_id": day_id,
                "view_type": view_type,
                "camera_stream_id": camera_stream_id,
                "filename": filename,
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "day_id", "view_type", "camera_stream_id", "filename"],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_reviewed_boxes(relative_paths: tuple[str, ...]) -> dict[str, list[tuple[int, int, int, int]]]:
    requested = set(relative_paths)
    found: dict[str, list[tuple[int, int, int, int]]] = {path: [] for path in relative_paths}
    for source in BOX_SOURCES:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row["image_id"]
                if image_id not in requested:
                    continue
                found[image_id].append(
                    (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
                )
    missing = [path for path, boxes in found.items() if not boxes]
    if missing:
        raise ValueError(f"Missing reviewed boxes for: {missing}")
    return found


def write_detection_csv(path: Path, boxes_by_image: dict[str, list[tuple[int, int, int, int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "x1", "y1", "x2", "y2", "score"])
        writer.writeheader()
        for image_id, boxes in boxes_by_image.items():
            for box in boxes:
                writer.writerow(
                    {
                        "image_id": image_id,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "score": 1.0,
                    }
                )


def build_method(method_name: str):
    if method_name == "blur":
        return BlurAnonymiser()
    if method_name == "pixelate":
        return PixelateAnonymiser()
    if method_name == "diffusion":
        return DiffusionAnonymiser()
    if method_name == "nullface":
        return NullFaceAnonymiser(
            crop_padding_ratio=0.6,
            guidance_scale=7.0,
            num_diffusion_steps=30,
            skip=20,
            mask_delay_steps=6,
        )
    if method_name == "stylegan":
        return StyleGANAnonymiser()
    if method_name == "fams":
        # Low-VRAM profile accepted for constrained-compute benchmarking.
        return FAMSAnonymiser(face_image_size=96, num_inference_steps=6, guidance_scale=2.0)
    raise ValueError(f"Unsupported method: {method_name}")


def main() -> None:
    args = parse_args()
    relative_paths = tuple(args.relative_paths) if args.relative_paths else DEFAULT_IMAGES
    output_root = args.output_root
    raw_root = PROJECT_ROOT / "data" / "castle2024" / "raw"
    resume = not args.no_resume

    manifest_rows = build_manifest_rows(relative_paths)
    slice_manifest_path = output_root / "slice_manifest.csv"
    write_manifest(slice_manifest_path, manifest_rows)

    boxes_by_image = load_reviewed_boxes(relative_paths)
    detections_path = output_root / "slice_detections.csv"
    write_detection_csv(detections_path, boxes_by_image)

    results: list[dict[str, object]] = []
    output_manifest_rows: list[dict[str, object]] = []

    for method_name in args.methods:
        anonymiser = build_method(method_name)
        for relative_path in relative_paths:
            input_path = raw_root / relative_path
            output_path = output_root / method_name / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            boxes = boxes_by_image[relative_path]
            if resume and output_path.is_file():
                results.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "boxes_count": len(boxes),
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "runtime_seconds": 0.0,
                        "status": "resumed",
                    }
                )
                output_manifest_rows.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "boxes_processed": len(boxes),
                        "tiling_required": False,
                    }
                )
                continue

            image = Image.open(input_path).convert("RGB")
            started = time.perf_counter()
            try:
                result = anonymiser.anonymise(image, boxes)
                runtime_seconds = round(time.perf_counter() - started, 3)
                result.image.save(output_path)
                results.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "boxes_count": len(boxes),
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "runtime_seconds": runtime_seconds,
                        "status": "ok",
                        "metadata": result.metadata,
                    }
                )
                output_manifest_rows.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
                        "boxes_processed": int(result.metadata.get("boxes_processed", 0)),
                        "tiling_required": bool(result.metadata.get("tiling_required", False)),
                    }
                )
            except Exception as exc:
                runtime_seconds = round(time.perf_counter() - started, 3)
                results.append(
                    {
                        "relative_path": relative_path,
                        "method": method_name,
                        "boxes_count": len(boxes),
                        "output_path": "",
                        "runtime_seconds": runtime_seconds,
                        "status": "error",
                        "error": str(exc),
                    }
                )

    output_manifest_path = output_root / "anonymised_manifest.csv"
    with output_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "method", "output_path", "boxes_processed", "tiling_required"],
        )
        writer.writeheader()
        writer.writerows(output_manifest_rows)

    summary = {
        "relative_paths": list(relative_paths),
        "methods": list(args.methods),
        "slice_manifest": str(slice_manifest_path.relative_to(PROJECT_ROOT)),
        "slice_detections": str(detections_path.relative_to(PROJECT_ROOT)),
        "anonymised_manifest": str(output_manifest_path.relative_to(PROJECT_ROOT)),
        "results": results,
    }
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
