#!/usr/bin/env python3

"""Validate completed face-box annotations for CASTLE detector training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = PROJECT_ROOT / "data" / "castle2024" / "annotations" / "face_detection" / "02_egocentric_stress_500"
TASKS_CSV_PATH = PROJECT_ROOT / "data" / "thesis_manifests" / "final_face_detection_500.csv"
RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"


def load_task_map() -> dict[str, Path]:
    """Map annotation image identifiers to on-disk image paths."""
    task_map: dict[str, Path] = {}
    with TASKS_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_path = PROJECT_ROOT / row.get("image_path", "")
            if not image_path.is_file():
                image_path = RAW_ROOT / row["relative_path"]
            task_map[row["relative_path"]] = image_path
    return task_map


def main() -> None:
    """Validate annotation CSV rows and report summary counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        default=str(PACK_ROOT / "manifest.csv"),
        help="Path to the completed face-box CSV.",
    )
    args = parser.parse_args()

    annotation_path = Path(args.annotations)
    task_map = load_task_map()
    required_fields = [
        "image_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "annotator_id",
        "annotation_round",
        "condition_label",
        "notes",
    ]
    allowed_optional_fields = {"score"}
    issues: list[str] = []
    counts = Counter()

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            issues.append("Missing header row")
        else:
            missing = [field for field in required_fields if field not in reader.fieldnames]
            unexpected = [field for field in reader.fieldnames if field not in required_fields and field not in allowed_optional_fields]
            if missing or unexpected:
                issues.append(
                    f"Unexpected header. Missing {missing}; unexpected {unexpected}; got {reader.fieldnames}"
                )
        for line_number, row in enumerate(reader, start=2):
            image_id = row["image_id"]
            if image_id not in task_map:
                issues.append(f"Line {line_number}: unknown image_id {image_id}")
                continue
            image_path = task_map[image_id]
            if not image_path.exists():
                issues.append(f"Line {line_number}: missing image file {image_path}")
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            try:
                x1 = int(row["x1"])
                y1 = int(row["y1"])
                x2 = int(row["x2"])
                y2 = int(row["y2"])
            except ValueError:
                issues.append(f"Line {line_number}: non-integer box coordinate")
                continue
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                issues.append(f"Line {line_number}: box out of bounds for {image_id}")
                continue
            counts[image_id] += 1

    annotated_images = len(counts)
    total_boxes = sum(counts.values())
    print(f"Annotated images: {annotated_images}")
    print(f"Total boxes: {total_boxes}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Validation passed.")


if __name__ == "__main__":
    main()
