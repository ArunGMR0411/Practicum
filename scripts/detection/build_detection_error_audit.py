#!/usr/bin/env python3

"""Build a reproducible reviewed error audit for CASTLE face detection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_iou

T = TypeVar("T")


def load_ground_truth(path: Path) -> list[GroundTruthBox]:
    rows: list[GroundTruthBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                GroundTruthBox(
                    image_id=row["image_id"],
                    box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    metadata={"condition_label": row.get("condition_label", "")},
                )
            )
    return rows


def load_predictions(path: Path) -> list[ScoredBox]:
    rows: list[ScoredBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ScoredBox(
                    image_id=row["image_id"],
                    box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    score=float(row["score"]),
                    metadata={"detector_name": row.get("detector_name", "")},
                )
            )
    return rows


def load_reviewed_image_status(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("reviewed") != "yes":
                continue
            rows[row["image_id"]] = dict(row)
    return rows


def load_annotation_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["relative_path"]] = dict(row)
    return rows


def group_by_image(rows: list[T], image_id_getter: Callable[[T], str]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for row in rows:
        grouped[image_id_getter(row)].append(row)
    return dict(grouped)


def box_metrics(box: tuple[int, int, int, int]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    aspect_ratio = float(width) / float(height) if height > 0 else math.inf
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return {
        "width": float(width),
        "height": float(height),
        "area": float(area),
        "aspect_ratio": float(aspect_ratio),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "min_side": float(min(width, height)),
        "max_side": float(max(width, height)),
    }


def match_image_predictions(
    image_predictions: list[ScoredBox],
    image_ground_truths: list[GroundTruthBox],
    iou_threshold: float,
) -> tuple[list[dict[str, object]], set[int], set[int]]:
    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()
    matches: list[dict[str, object]] = []

    ranked_predictions = sorted(enumerate(image_predictions), key=lambda item: item[1].score, reverse=True)
    for pred_index, prediction in ranked_predictions:
        best_gt_index = -1
        best_iou = 0.0
        for gt_index, gt in enumerate(image_ground_truths):
            if gt_index in matched_gt_indices:
                continue
            iou = compute_iou(prediction.box, gt.box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index >= 0 and best_iou >= iou_threshold:
            matched_gt_indices.add(best_gt_index)
            matched_pred_indices.add(pred_index)
            matches.append(
                {
                    "pred_index": pred_index,
                    "gt_index": best_gt_index,
                    "iou": round(best_iou, 6),
                    "score": prediction.score,
                }
            )
    return matches, matched_pred_indices, matched_gt_indices


def duplicate_like_rows(image_predictions: list[ScoredBox], duplicate_iou_threshold: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_index, left in enumerate(image_predictions):
        for right_index in range(left_index + 1, len(image_predictions)):
            right = image_predictions[right_index]
            iou = compute_iou(left.box, right.box)
            if iou < duplicate_iou_threshold:
                continue
            left_metrics = box_metrics(left.box)
            right_metrics = box_metrics(right.box)
            rows.append(
                {
                    "image_id": left.image_id,
                    "left_index": left_index,
                    "right_index": right_index,
                    "iou": round(iou, 6),
                    "left_score": round(left.score, 6),
                    "right_score": round(right.score, 6),
                    "left_box": "|".join(str(value) for value in left.box),
                    "right_box": "|".join(str(value) for value in right.box),
                    "left_min_side": round(left_metrics["min_side"], 3),
                    "right_min_side": round(right_metrics["min_side"], 3),
                }
            )
    return rows


def summarise_counter(counter: Counter[str]) -> list[dict[str, object]]:
    return [{"key": key, "count": count} for key, count in counter.most_common()]


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def save_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth",
        default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
    )
    parser.add_argument(
        "--review-status",
        default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv",
    )
    parser.add_argument(
        "--predictions",
        default="outputs/detection_eval_subset_yolo_scrfd_fallback.csv",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="outputs/detection_error_audit_yolo_scrfd_fallback",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    predictions = load_predictions(PROJECT_ROOT / args.predictions)
    reviewed_status = load_reviewed_image_status(PROJECT_ROOT / args.review_status)
    annotation_manifest = load_annotation_manifest(PROJECT_ROOT / args.manifest)

    gt_by_image = group_by_image(ground_truth, lambda row: row.image_id)
    pred_by_image = group_by_image(predictions, lambda row: row.image_id)
    reviewed_image_ids = sorted(reviewed_status.keys())

    false_positive_rows: list[dict[str, object]] = []
    false_negative_rows: list[dict[str, object]] = []
    zero_face_false_positive_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    image_summary_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []

    fp_by_day: Counter[str] = Counter()
    fp_by_view: Counter[str] = Counter()
    fp_zero_face_by_day: Counter[str] = Counter()
    fn_by_day: Counter[str] = Counter()
    fn_by_view: Counter[str] = Counter()

    for image_id in reviewed_image_ids:
        image_gts = gt_by_image.get(image_id, [])
        image_preds = pred_by_image.get(image_id, [])
        image_meta = annotation_manifest.get(image_id, {})
        day_id = image_meta.get("day_id", "")
        view_type = image_meta.get("view_type", "")
        stream_id = image_meta.get("camera_stream_id", "")

        matches, matched_pred_indices, matched_gt_indices = match_image_predictions(
            image_preds,
            image_gts,
            iou_threshold=args.iou_threshold,
        )
        duplicate_rows.extend(duplicate_like_rows(image_preds, duplicate_iou_threshold=args.duplicate_iou_threshold))

        for row in matches:
            prediction = image_preds[int(row["pred_index"])]
            gt = image_gts[int(row["gt_index"])]
            pred_metrics = box_metrics(prediction.box)
            gt_metrics = box_metrics(gt.box)
            match_rows.append(
                {
                    "image_id": image_id,
                    "day_id": day_id,
                    "view_type": view_type,
                    "camera_stream_id": stream_id,
                    "pred_index": row["pred_index"],
                    "gt_index": row["gt_index"],
                    "iou": row["iou"],
                    "score": round(prediction.score, 6),
                    "pred_box": "|".join(str(value) for value in prediction.box),
                    "gt_box": "|".join(str(value) for value in gt.box),
                    "pred_min_side": round(pred_metrics["min_side"], 3),
                    "gt_min_side": round(gt_metrics["min_side"], 3),
                }
            )

        for pred_index, prediction in enumerate(image_preds):
            if pred_index in matched_pred_indices:
                continue
            metrics = box_metrics(prediction.box)
            row = {
                "image_id": image_id,
                "day_id": day_id,
                "view_type": view_type,
                "camera_stream_id": stream_id,
                "score": round(prediction.score, 6),
                "x1": prediction.box[0],
                "y1": prediction.box[1],
                "x2": prediction.box[2],
                "y2": prediction.box[3],
                "width": round(metrics["width"], 3),
                "height": round(metrics["height"], 3),
                "min_side": round(metrics["min_side"], 3),
                "max_side": round(metrics["max_side"], 3),
                "area": round(metrics["area"], 3),
                "aspect_ratio": round(metrics["aspect_ratio"], 6),
                "zero_face_image": not image_gts,
            }
            false_positive_rows.append(row)
            fp_by_day[day_id] += 1
            fp_by_view[view_type] += 1
            if not image_gts:
                zero_face_false_positive_rows.append(row)
                fp_zero_face_by_day[day_id] += 1

        for gt_index, gt in enumerate(image_gts):
            if gt_index in matched_gt_indices:
                continue
            metrics = box_metrics(gt.box)
            row = {
                "image_id": image_id,
                "day_id": day_id,
                "view_type": view_type,
                "camera_stream_id": stream_id,
                "x1": gt.box[0],
                "y1": gt.box[1],
                "x2": gt.box[2],
                "y2": gt.box[3],
                "width": round(metrics["width"], 3),
                "height": round(metrics["height"], 3),
                "min_side": round(metrics["min_side"], 3),
                "max_side": round(metrics["max_side"], 3),
                "area": round(metrics["area"], 3),
                "aspect_ratio": round(metrics["aspect_ratio"], 6),
            }
            false_negative_rows.append(row)
            fn_by_day[day_id] += 1
            fn_by_view[view_type] += 1

        image_summary_rows.append(
            {
                "image_id": image_id,
                "day_id": day_id,
                "view_type": view_type,
                "camera_stream_id": stream_id,
                "gt_count": len(image_gts),
                "prediction_count": len(image_preds),
                "matched_gt_count": len(matched_gt_indices),
                "matched_prediction_count": len(matched_pred_indices),
                "false_positive_count": len(image_preds) - len(matched_pred_indices),
                "false_negative_count": len(image_gts) - len(matched_gt_indices),
                "zero_face_image": not image_gts,
            }
        )

    output_dir = PROJECT_ROOT / args.output_dir
    save_csv(false_positive_rows, output_dir / "false_positives.csv")
    save_csv(false_negative_rows, output_dir / "false_negatives.csv")
    save_csv(zero_face_false_positive_rows, output_dir / "zero_face_false_positives.csv")
    save_csv(duplicate_rows, output_dir / "duplicate_like_pairs.csv")
    save_csv(image_summary_rows, output_dir / "image_summary.csv")
    save_csv(match_rows, output_dir / "matched_pairs.csv")

    summary = {
        "ground_truth_path": args.ground_truth,
        "predictions_path": args.predictions,
        "reviewed_image_count": len(reviewed_image_ids),
        "ground_truth_box_count": len(ground_truth),
        "prediction_count": len(predictions),
        "false_positive_count": len(false_positive_rows),
        "false_negative_count": len(false_negative_rows),
        "zero_face_false_positive_count": len(zero_face_false_positive_rows),
        "duplicate_like_pair_count_iou_ge_threshold": len(duplicate_rows),
        "iou_threshold": args.iou_threshold,
        "duplicate_iou_threshold": args.duplicate_iou_threshold,
        "false_positive_by_day": summarise_counter(fp_by_day),
        "false_positive_by_view": summarise_counter(fp_by_view),
        "zero_face_false_positive_by_day": summarise_counter(fp_zero_face_by_day),
        "false_negative_by_day": summarise_counter(fn_by_day),
        "false_negative_by_view": summarise_counter(fn_by_view),
        "top_false_positive_images": sorted(
            (
                row
                for row in image_summary_rows
                if int(row["false_positive_count"]) > 0
            ),
            key=lambda row: (int(row["false_positive_count"]), int(row["prediction_count"])),
            reverse=True,
        )[:25],
        "top_false_negative_images": sorted(
            (
                row
                for row in image_summary_rows
                if int(row["false_negative_count"]) > 0
            ),
            key=lambda row: (int(row["false_negative_count"]), int(row["gt_count"])),
            reverse=True,
        )[:25],
    }
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
