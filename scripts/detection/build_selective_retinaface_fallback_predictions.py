#!/usr/bin/env python3

"""Merge RetinaFace detections into YOLO+SCRFD predictions on low-count images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import compute_iou


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "x1", "y1", "x2", "y2", "score", "detector_name"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/detection_eval_subset_yolo_scrfd_fallback.csv")
    parser.add_argument("--fallback", default="outputs/detection_eval_subset_retinaface.csv")
    parser.add_argument("--prediction-count-threshold", type=int, default=3)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-score-scale", type=float, default=0.95)
    parser.add_argument("--output", default="outputs/detection_eval_subset_yolo_scrfd_retinaface_le3.csv")
    parser.add_argument("--summary-output", default="outputs/detection_eval_subset_yolo_scrfd_retinaface_le3_summary.json")
    args = parser.parse_args()

    base_rows = load_rows(PROJECT_ROOT / args.base)
    fallback_rows = load_rows(PROJECT_ROOT / args.fallback)
    base_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    fallback_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in base_rows:
        base_by_image[row["image_id"]].append(row)
    for row in fallback_rows:
        fallback_by_image[row["image_id"]].append(row)

    merged_rows: list[dict[str, object]] = []
    triggered_images = 0
    fallback_added_count = 0
    all_image_ids = sorted(set(base_by_image) | set(fallback_by_image))
    for image_id in all_image_ids:
        base_image_rows = list(base_by_image.get(image_id, []))
        merged_rows.extend(
            {
                "image_id": row["image_id"],
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "score": float(row["score"]),
                "detector_name": row["detector_name"],
            }
            for row in base_image_rows
        )
        if len(base_image_rows) > args.prediction_count_threshold:
            continue

        triggered_images += 1
        existing_boxes = [
            (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            for row in base_image_rows
        ]
        for row in fallback_by_image.get(image_id, []):
            candidate_box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            if any(compute_iou(candidate_box, existing_box) >= args.duplicate_iou_threshold for existing_box in existing_boxes):
                continue
            merged_rows.append(
                {
                    "image_id": image_id,
                    "x1": candidate_box[0],
                    "y1": candidate_box[1],
                    "x2": candidate_box[2],
                    "y2": candidate_box[3],
                    "score": float(row["score"]) * args.fallback_score_scale,
                    "detector_name": "yolo_scrfd_retinaface_selective",
                }
            )
            existing_boxes.append(candidate_box)
            fallback_added_count += 1

    save_rows(merged_rows, PROJECT_ROOT / args.output)
    summary = {
        "base": args.base,
        "fallback": args.fallback,
        "prediction_count_threshold": args.prediction_count_threshold,
        "duplicate_iou_threshold": args.duplicate_iou_threshold,
        "fallback_score_scale": args.fallback_score_scale,
        "triggered_image_count": triggered_images,
        "fallback_added_count": fallback_added_count,
        "output": args.output,
        "merged_detection_count": len(merged_rows),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=(PROJECT_ROOT / args.summary_output).parent, delete=False) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(PROJECT_ROOT / args.summary_output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
