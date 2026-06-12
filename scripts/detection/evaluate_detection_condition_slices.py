#!/usr/bin/env python3

"""Evaluate detector performance on reviewed condition-specific slices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision


def load_slice_image_ids(manifest_path: Path, condition_label: str) -> list[str]:
    """Return reviewed image ids for one condition label."""
    image_ids: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("annotation_status", "") != "reviewed":
                continue
            if row.get("condition_label", "") != condition_label:
                continue
            image_ids.append(row["relative_path"])
    return image_ids


def load_ground_truth_subset(path: Path, image_id_set: set[str]) -> list[GroundTruthBox]:
    """Load GT boxes for a restricted slice of image ids."""
    rows: list[GroundTruthBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            if image_id not in image_id_set:
                continue
            rows.append(
                GroundTruthBox(
                    image_id=image_id,
                    box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    metadata={"condition_label": row.get("condition_label", "")},
                )
            )
    return rows


def load_prediction_subset(path: Path, image_id_set: set[str]) -> list[ScoredBox]:
    """Load detector predictions for a restricted slice of image ids."""
    rows: list[ScoredBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["image_id"]
            if image_id not in image_id_set:
                continue
            rows.append(
                ScoredBox(
                    image_id=image_id,
                    box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    score=float(row["score"]),
                    metadata={"condition_label": row.get("condition_label", "")},
                )
            )
    return rows


def save_json(payload: dict, output_path: Path) -> None:
    """Atomically save JSON output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--detector-a", required=True)
    parser.add_argument("--detector-a-name", default="detector_a")
    parser.add_argument("--detector-b", required=True)
    parser.add_argument("--detector-b-name", default="detector_b")
    parser.add_argument("--condition", action="append", required=True)
    parser.add_argument("--output", default="outputs/detection_condition_slices.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    gt_path = Path(args.ground_truth)
    detector_a_path = Path(args.detector_a)
    detector_b_path = Path(args.detector_b)

    payload: dict[str, object] = {"conditions": {}}
    for condition in args.condition:
        image_ids = load_slice_image_ids(manifest_path, condition)
        image_id_set = set(image_ids)
        gt_subset = load_ground_truth_subset(gt_path, image_id_set)
        det_a_subset = load_prediction_subset(detector_a_path, image_id_set)
        det_b_subset = load_prediction_subset(detector_b_path, image_id_set)
        payload["conditions"][condition] = {
            "image_count": len(image_ids),
            "ground_truth_box_count": len(gt_subset),
            args.detector_a_name: compute_average_precision(det_a_subset, gt_subset, iou_threshold=0.5),
            args.detector_b_name: compute_average_precision(det_b_subset, gt_subset, iou_threshold=0.5),
        }

    save_json(payload, Path(args.output))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
