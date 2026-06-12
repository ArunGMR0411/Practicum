#!/usr/bin/env python3

"""Evaluate selective MTCNN fallback policies on the reviewed face-annotation pack."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision


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


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
                metadata={
                    "condition_label": row.get("condition_label", ""),
                    "detector_name": row.get("detector_name", ""),
                },
            )
        )
    return dict(grouped)


def load_condition_labels(path: Path) -> dict[str, str]:
    rows = load_csv_rows(path)
    return {row["relative_path"]: row.get("condition_label", "") for row in rows}


def merge_boxes(
    primary: list[ScoredBox],
    fallback: list[ScoredBox],
    duplicate_iou_threshold: float,
    fallback_score_scale: float,
) -> list[ScoredBox]:
    """Merge fallback detections into primary detections with duplicate suppression."""
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
    """Return counts of fully correct images and images with at least one missed face."""
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


def atomic_write_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--ground-truth", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--primary-predictions", default="outputs/02_face_detection/01_yolo_predictions_run.csv")
    parser.add_argument("--fallback-predictions", default="outputs/mtcnn_face_predictions_reviewed.csv")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-score-scale", type=float, default=0.95)
    parser.add_argument(
        "--trigger-labels",
        nargs="+",
        default=["small_face", "downward_view", "visible_text", "motion_blur", "extreme_pose"],
    )
    parser.add_argument("--output-json", default="outputs/conditional_fallback_detector_experiment.json")
    args = parser.parse_args()

    ground_truths = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    primary_predictions = load_predictions(PROJECT_ROOT / args.primary_predictions)
    fallback_predictions = load_predictions(PROJECT_ROOT / args.fallback_predictions)
    condition_by_image = load_condition_labels(PROJECT_ROOT / args.annotation_manifest)

    primary_all = [item for items in primary_predictions.values() for item in items]
    baseline_metrics = compute_average_precision(primary_all, ground_truths, iou_threshold=args.iou_threshold)
    baseline_correct, baseline_missed = image_level_correctness(primary_predictions, ground_truths, iou_threshold=args.iou_threshold)

    candidate_labels = list(dict.fromkeys(args.trigger_labels))
    policy_results: list[dict[str, object]] = []
    best_policy: dict[str, object] | None = None

    for r in range(1, len(candidate_labels) + 1):
        for trigger_labels in itertools.combinations(candidate_labels, r):
            trigger_set = set(trigger_labels)
            merged_by_image: dict[str, list[ScoredBox]] = {}
            triggered_images = 0
            for image_id, condition_label in condition_by_image.items():
                primary_items = primary_predictions.get(image_id, [])
                if condition_label in trigger_set:
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
                "trigger_labels": list(trigger_labels),
                "triggered_images": triggered_images,
                "ap": round(float(metrics["ap"]), 6),
                "precision": round(float(metrics["precision"]), 6),
                "recall": round(float(metrics["recall"]), 6),
                "f1": round(float(metrics["f1"]), 6),
                "true_positives": int(metrics["true_positives"]),
                "false_positives": int(metrics["false_positives"]),
                "false_negatives": int(metrics["false_negatives"]),
                "images_fully_correct": int(correct_images),
                "images_with_missed_faces": int(missed_images),
                "image_level_full_match_rate": round(float(correct_images / len(condition_by_image)), 6) if condition_by_image else 0.0,
            }
            policy_results.append(result)
            if best_policy is None or (
                result["recall"],
                result["f1"],
                -result["false_positives"],
                -result["triggered_images"],
            ) > (
                best_policy["recall"],
                best_policy["f1"],
                -best_policy["false_positives"],
                -best_policy["triggered_images"],
            ):
                best_policy = result

    max_recall = max(float(item["recall"]) for item in policy_results) if policy_results else 0.0
    balanced_candidates = [item for item in policy_results if float(item["recall"]) >= max_recall - 0.005]
    recommended_policy = None
    if balanced_candidates:
        recommended_policy = max(
            balanced_candidates,
            key=lambda item: (
                float(item["f1"]),
                float(item["precision"]),
                -int(item["false_positives"]),
                -int(item["triggered_images"]),
            ),
        )

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
            "image_level_full_match_rate": round(float(baseline_correct / len(condition_by_image)), 6) if condition_by_image else 0.0,
        },
        "policy_results": policy_results,
        "best_policy": best_policy,
        "recommended_policy": recommended_policy,
    }
    atomic_write_json(payload, PROJECT_ROOT / args.output_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
