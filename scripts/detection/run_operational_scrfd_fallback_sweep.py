#!/usr/bin/env python3

"""Search a runtime-available trigger for selective YOLO + SCRFD fallback."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision
from src.routing.quality_assessor import QualityAssessor
from src.data.subset_building import detect_text_score, resize_for_analysis


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def atomic_write_csv(rows: list[dict[str, object]], fieldnames: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def load_ground_truth(path: Path) -> list[GroundTruthBox]:
    rows: list[GroundTruthBox] = []
    for row in load_csv_rows(path):
        rows.append(
            GroundTruthBox(
                image_id=row["image_id"],
                box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                metadata={"condition_label": row.get("condition_label", "")},
            )
        )
    return rows


def load_predictions(path: Path) -> dict[str, list[ScoredBox]]:
    grouped: dict[str, list[ScoredBox]] = defaultdict(list)
    for row in load_csv_rows(path):
        grouped[row["image_id"]].append(
            ScoredBox(
                image_id=row["image_id"],
                box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                score=float(row["score"]),
                metadata={"detector_name": row.get("detector_name", "")},
            )
        )
    return dict(grouped)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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


def merge_boxes(
    primary: list[ScoredBox],
    fallback: list[ScoredBox],
    duplicate_iou_threshold: float,
    fallback_score_scale: float,
) -> list[ScoredBox]:
    merged = list(primary)
    for candidate in fallback:
        if any(compute_iou(candidate.box, existing.box) >= duplicate_iou_threshold for existing in merged):
            continue
        merged.append(
            ScoredBox(
                image_id=candidate.image_id,
                box=candidate.box,
                score=float(candidate.score) * fallback_score_scale,
                metadata=candidate.metadata,
            )
        )
    return merged


def image_level_correctness(
    predictions_by_image: dict[str, list[ScoredBox]],
    ground_truths: list[GroundTruthBox],
    iou_threshold: float,
) -> tuple[int, int]:
    gt_by_image: dict[str, list[GroundTruthBox]] = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt.image_id].append(gt)

    correct = 0
    missed = 0
    for image_id, image_gts in gt_by_image.items():
        image_predictions = predictions_by_image.get(image_id, [])
        matched_prediction_indices: set[int] = set()
        matched_gt_count = 0
        for gt in image_gts:
            best_prediction_index = -1
            best_iou = 0.0
            for prediction_index, prediction in enumerate(image_predictions):
                if prediction_index in matched_prediction_indices:
                    continue
                iou = compute_iou(prediction.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_prediction_index = prediction_index
            if best_prediction_index >= 0 and best_iou >= iou_threshold:
                matched_prediction_indices.add(best_prediction_index)
                matched_gt_count += 1
        if matched_gt_count == len(image_gts):
            correct += 1
        else:
            missed += 1
    return correct, missed


def compute_runtime_signal_rows(
    annotation_manifest_rows: list[dict[str, str]],
    primary_predictions: dict[str, list[ScoredBox]],
    raw_root: Path,
) -> list[dict[str, object]]:
    assessor = QualityAssessor()
    signal_rows: list[dict[str, object]] = []
    for row in annotation_manifest_rows:
        image_id = row["relative_path"]
        image_path = raw_root / image_id
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            detections = primary_predictions.get(image_id, [])
            face_boxes = [item.box for item in detections]
            assessment = assessor.assess(rgb_image, face_boxes=face_boxes)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            resized = resize_for_analysis(image_bgr, analysis_width=192)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            text_score = int(detect_text_score(gray))
            dominant_box = assessment.metadata.get("dominant_face_box")
            center_y_ratio = 0.0
            min_face_size_px = 0.0
            if dominant_box is not None:
                _, y1, _, y2 = dominant_box
                center_y_ratio = ((float(y1) + float(y2)) / 2.0) / max(rgb_image.height, 1)
            if face_boxes:
                min_face_size_px = min(float(min(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in face_boxes)
            signal_rows.append(
                {
                    "image_id": image_id,
                    "condition_label": row.get("condition_label", ""),
                    "yolo_face_box_count": int(assessment.metadata["face_box_count"]),
                    "dominant_face_size_px": float(assessment.signals.face_size_px),
                    "min_detected_face_size_px": float(min_face_size_px),
                    "dominant_center_y_ratio": float(center_y_ratio),
                    "text_score": text_score,
                }
            )
    return signal_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--ground-truth", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--primary-predictions", default="outputs/02_face_detection/01_yolo_predictions_run.csv")
    parser.add_argument("--fallback-predictions", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-score-scale", type=float, default=0.95)
    parser.add_argument("--output-json", default="outputs/operational_scrfd_fallback_sweep.json")
    parser.add_argument("--output-signals-csv", default="outputs/reviewed_runtime_trigger_signals.csv")
    args = parser.parse_args()

    annotation_manifest_rows = [row for row in load_csv_rows(PROJECT_ROOT / args.annotation_manifest) if row.get("annotation_status", "") == "reviewed"]
    ground_truths = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    primary_predictions = load_predictions(PROJECT_ROOT / args.primary_predictions)
    fallback_predictions = load_predictions(PROJECT_ROOT / args.fallback_predictions)
    signal_rows = compute_runtime_signal_rows(annotation_manifest_rows, primary_predictions, PROJECT_ROOT / args.raw_root)

    atomic_write_csv(signal_rows, list(signal_rows[0].keys()) if signal_rows else [], PROJECT_ROOT / args.output_signals_csv)

    primary_all = [item for items in primary_predictions.values() for item in items]
    baseline_metrics = compute_average_precision(primary_all, ground_truths, iou_threshold=args.iou_threshold)
    baseline_correct, baseline_missed = image_level_correctness(primary_predictions, ground_truths, iou_threshold=args.iou_threshold)

    size_thresholds = [24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
    center_y_thresholds = [0.60, 0.62, 0.65, 0.68]
    text_thresholds = [8, 12, 14, 18]

    policy_results: list[dict[str, object]] = []
    best_policy: dict[str, object] | None = None

    signal_by_image = {row["image_id"]: row for row in signal_rows}

    for size_threshold, center_y_threshold, text_threshold in itertools.product(
        size_thresholds, center_y_thresholds, text_thresholds
    ):
        merged_by_image: dict[str, list[ScoredBox]] = {}
        triggered_images = 0
        trigger_breakdown = {"small_face_like": 0, "downward_like": 0, "text_like": 0}

        for row in annotation_manifest_rows:
            image_id = row["relative_path"]
            primary_items = primary_predictions.get(image_id, [])
            signal_row = signal_by_image[image_id]
            min_detected_face_size_px = float(signal_row["min_detected_face_size_px"])
            small_face_like = 0.0 < min_detected_face_size_px <= size_threshold
            downward_like = float(signal_row["dominant_center_y_ratio"]) >= center_y_threshold
            text_like = int(signal_row["text_score"]) >= text_threshold
            should_trigger = small_face_like or downward_like or text_like
            if small_face_like:
                trigger_breakdown["small_face_like"] += 1
            if downward_like:
                trigger_breakdown["downward_like"] += 1
            if text_like:
                trigger_breakdown["text_like"] += 1

            if should_trigger:
                triggered_images += 1
                merged_by_image[image_id] = merge_boxes(
                    primary=primary_items,
                    fallback=fallback_predictions.get(image_id, []),
                    duplicate_iou_threshold=args.duplicate_iou_threshold,
                    fallback_score_scale=args.fallback_score_scale,
                )
            else:
                merged_by_image[image_id] = list(primary_items)

        merged_all = [item for items in merged_by_image.values() for item in items]
        metrics = compute_average_precision(merged_all, ground_truths, iou_threshold=args.iou_threshold)
        correct_images, missed_images = image_level_correctness(merged_by_image, ground_truths, iou_threshold=args.iou_threshold)
        result = {
            "size_threshold_px": size_threshold,
            "center_y_threshold": center_y_threshold,
            "text_score_threshold": text_threshold,
            "triggered_images": triggered_images,
            "trigger_breakdown": trigger_breakdown,
            "ap": round(float(metrics["ap"]), 6),
            "precision": round(float(metrics["precision"]), 6),
            "recall": round(float(metrics["recall"]), 6),
            "f1": round(float(metrics["f1"]), 6),
            "true_positives": int(metrics["true_positives"]),
            "false_positives": int(metrics["false_positives"]),
            "false_negatives": int(metrics["false_negatives"]),
            "images_fully_correct": int(correct_images),
            "images_with_missed_faces": int(missed_images),
            "image_level_full_match_rate": round(float(correct_images / len(annotation_manifest_rows)), 6) if annotation_manifest_rows else 0.0,
        }
        policy_results.append(result)
        if best_policy is None or (
            result["f1"],
            result["recall"],
            -result["false_positives"],
            -result["triggered_images"],
        ) > (
            best_policy["f1"],
            best_policy["recall"],
            -best_policy["false_positives"],
            -best_policy["triggered_images"],
        ):
            best_policy = result

    payload = {
        "annotation_manifest": args.annotation_manifest,
        "ground_truth": args.ground_truth,
        "primary_predictions": args.primary_predictions,
        "fallback_predictions": args.fallback_predictions,
        "baseline": {
            "ap": round(float(baseline_metrics["ap"]), 6),
            "precision": round(float(baseline_metrics["precision"]), 6),
            "recall": round(float(baseline_metrics["recall"]), 6),
            "f1": round(float(baseline_metrics["f1"]), 6),
            "true_positives": int(baseline_metrics["true_positives"]),
            "false_positives": int(baseline_metrics["false_positives"]),
            "false_negatives": int(baseline_metrics["false_negatives"]),
            "images_fully_correct": int(baseline_correct),
            "images_with_missed_faces": int(baseline_missed),
            "image_level_full_match_rate": round(float(baseline_correct / len(annotation_manifest_rows)), 6) if annotation_manifest_rows else 0.0,
        },
        "policy_results": policy_results,
        "recommended_policy": best_policy,
    }
    atomic_write_json(payload, PROJECT_ROOT / args.output_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
