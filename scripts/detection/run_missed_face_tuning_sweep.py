#!/usr/bin/env python3

"""Sweep low-risk YOLO inference settings on the missed-face audit subset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_detector import YOLODetector
from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load one CSV file into a list of dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def group_ground_truth(rows: list[dict[str, str]]) -> tuple[list[GroundTruthBox], dict[str, list[GroundTruthBox]]]:
    """Convert reviewed face boxes into grouped ground-truth objects."""
    all_boxes: list[GroundTruthBox] = []
    grouped: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for row in rows:
        gt = GroundTruthBox(
            image_id=row["image_id"],
            box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
            metadata={"condition_label": row.get("condition_label", "")},
        )
        all_boxes.append(gt)
        grouped[gt.image_id].append(gt)
    return all_boxes, dict(grouped)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Return IoU for one prediction-ground-truth pair."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def image_level_correct(
    predictions_by_image: dict[str, list[ScoredBox]],
    ground_truth_by_image: dict[str, list[GroundTruthBox]],
    iou_threshold: float,
) -> tuple[int, int]:
    """Return count of correct images and images with at least one missed face."""
    correct_images = 0
    missed_images = 0
    for image_id, ground_truths in ground_truth_by_image.items():
        predictions = predictions_by_image.get(image_id, [])
        matched_prediction_indices: set[int] = set()
        matched_count = 0
        for gt in ground_truths:
            best_prediction_index = -1
            best_iou = 0.0
            for prediction_index, prediction in enumerate(predictions):
                if prediction_index in matched_prediction_indices:
                    continue
                iou = compute_iou(prediction.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_prediction_index = prediction_index
            if best_prediction_index >= 0 and best_iou >= iou_threshold:
                matched_prediction_indices.add(best_prediction_index)
                matched_count += 1
        if matched_count == len(ground_truths):
            correct_images += 1
        else:
            missed_images += 1
    return correct_images, missed_images


def atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    """Atomically write one JSON payload."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", default="outputs/01_protocol/supporting_protocols/06_missed_face_audit.csv")
    parser.add_argument("--ground-truth", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--model-path", default="data/models/yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-thresholds", nargs="+", type=float, default=[0.25, 0.15, 0.10, 0.05])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[960, 1280, 1600])
    parser.add_argument("--output-json", default="outputs/missed_face_tuning_sweep.json")
    args = parser.parse_args()

    audit_manifest_rows = load_csv_rows(PROJECT_ROOT / args.audit_manifest)
    audit_image_ids = [row["relative_path"] for row in audit_manifest_rows]
    ground_truth_rows = [row for row in load_csv_rows(PROJECT_ROOT / args.ground_truth) if row["image_id"] in set(audit_image_ids)]
    all_ground_truths, ground_truth_by_image = group_ground_truth(ground_truth_rows)

    grid_results: list[dict[str, object]] = []
    best_config: dict[str, object] | None = None

    for confidence_threshold in args.confidence_thresholds:
        for image_size in args.image_sizes:
            detector = YOLODetector(
                model_path=args.model_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=args.iou_threshold,
                device=args.device,
                image_size=image_size,
            )
            predictions: list[ScoredBox] = []
            predictions_by_image: dict[str, list[ScoredBox]] = defaultdict(list)

            for row in audit_manifest_rows:
                image_id = row["relative_path"]
                image_path = PROJECT_ROOT / "data" / "castle2024" / "raw" / image_id
                with Image.open(image_path) as image:
                    result = detector.detect(image)
                for detection in result.detections:
                    prediction = ScoredBox(
                        image_id=image_id,
                        box=detection.box,
                        score=float(detection.confidence),
                        metadata={"condition_label": row.get("condition_label", "")},
                    )
                    predictions.append(prediction)
                    predictions_by_image[image_id].append(prediction)

            metrics = compute_average_precision(predictions, all_ground_truths, iou_threshold=args.iou_threshold)
            correct_images, missed_images = image_level_correct(
                predictions_by_image=predictions_by_image,
                ground_truth_by_image=ground_truth_by_image,
                iou_threshold=args.iou_threshold,
            )
            config_result = {
                "confidence_threshold": confidence_threshold,
                "image_size": image_size,
                "ap": round(float(metrics["ap"]), 6),
                "precision": round(float(metrics["precision"]), 6),
                "recall": round(float(metrics["recall"]), 6),
                "f1": round(float(metrics["f1"]), 6),
                "true_positives": int(metrics["true_positives"]),
                "false_positives": int(metrics["false_positives"]),
                "false_negatives": int(metrics["false_negatives"]),
                "images_fully_correct": int(correct_images),
                "images_with_missed_faces": int(missed_images),
                "image_level_full_match_rate": round(float(correct_images / len(audit_image_ids)), 6) if audit_image_ids else 0.0,
                "detections_written": int(len(predictions)),
            }
            grid_results.append(config_result)
            if best_config is None or (
                config_result["recall"],
                config_result["f1"],
                -config_result["false_positives"],
                config_result["image_level_full_match_rate"],
            ) > (
                best_config["recall"],
                best_config["f1"],
                -best_config["false_positives"],
                best_config["image_level_full_match_rate"],
            ):
                best_config = config_result

    max_recall = max(float(item["recall"]) for item in grid_results) if grid_results else 0.0
    balanced_candidates = [item for item in grid_results if float(item["recall"]) >= max_recall - 0.01]
    recommended_config = None
    if balanced_candidates:
        recommended_config = max(
            balanced_candidates,
            key=lambda item: (
                float(item["f1"]),
                float(item["precision"]),
                -int(item["false_positives"]),
                float(item["image_level_full_match_rate"]),
            ),
        )

    payload = {
        "audit_manifest": args.audit_manifest,
        "ground_truth": args.ground_truth,
        "model_path": args.model_path,
        "device": args.device,
        "iou_threshold": args.iou_threshold,
        "images_evaluated": len(audit_image_ids),
        "ground_truth_faces": len(all_ground_truths),
        "grid_results": grid_results,
        "best_config": best_config,
        "recommended_config": recommended_config,
    }
    atomic_write_json(payload, PROJECT_ROOT / args.output_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
