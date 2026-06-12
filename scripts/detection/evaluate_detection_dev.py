#!/usr/bin/env python3

"""Evaluate detector predictions against annotated CASTLE frames."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision


def load_ground_truth(path: Path) -> list[GroundTruthBox]:
    """Load ground-truth face boxes from a CSV file."""
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
    """Load scored detector predictions from a CSV file."""
    rows: list[ScoredBox] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ScoredBox(
                    image_id=row["image_id"],
                    box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
                    score=float(row["score"]),
                    metadata={"condition_label": row.get("condition_label", "")},
                )
            )
    return rows


def load_image_ids_from_manifest(path: Path) -> list[str]:
    """Load the full reviewed image-id list from the annotation manifest."""
    rows: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("annotation_status", "") == "reviewed":
                rows.append(row["relative_path"])
    if rows:
        return rows

    # Browser reviews store completion status in the adjacent manifest.
    status_path = path.parent / "manifest.csv"
    if status_path.exists():
        with status_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("reviewed") == "yes":
                    rows.append(row["image_id"])
    return rows


def mcnemar_from_counts(n01: int, n10: int) -> dict[str, float]:
    """Compute McNemar's test statistic and p-value from discordant counts."""
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = float(chi2.sf(statistic, df=1))
    return {"statistic": statistic, "p_value": p_value}


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
    if inter_area == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


def group_ground_truths(ground_truths: list[GroundTruthBox]) -> dict[str, list[GroundTruthBox]]:
    """Group GT boxes by image id."""
    grouped: dict[str, list[GroundTruthBox]] = {}
    for gt in ground_truths:
        grouped.setdefault(gt.image_id, []).append(gt)
    return grouped


def group_predictions(predictions: list[ScoredBox]) -> dict[str, list[ScoredBox]]:
    """Group predictions by image id."""
    grouped: dict[str, list[ScoredBox]] = {}
    for prediction in predictions:
        grouped.setdefault(prediction.image_id, []).append(prediction)
    return grouped


def image_level_correctness(
    predictions: list[ScoredBox],
    ground_truths: list[GroundTruthBox],
    image_ids: list[str],
    iou_threshold: float = 0.5,
) -> dict[str, bool]:
    """Return per-image correctness for McNemar comparison.

    Positive image: correct when every GT face is matched by a unique prediction with IoU >= threshold.
    Negative image: correct when the detector emits zero predictions.
    """
    gt_by_image = group_ground_truths(ground_truths)
    pred_by_image = group_predictions(predictions)
    correctness: dict[str, bool] = {}
    for image_id in image_ids:
        image_gts = gt_by_image.get(image_id, [])
        image_preds = pred_by_image.get(image_id, [])
        if not image_gts:
            correctness[image_id] = len(image_preds) == 0
            continue
        matched_pred_indices: set[int] = set()
        matched_gt_count = 0
        for gt in image_gts:
            best_index = -1
            best_iou = 0.0
            for index, prediction in enumerate(image_preds):
                if index in matched_pred_indices:
                    continue
                iou = compute_iou(prediction.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_index = index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched_pred_indices.add(best_index)
                matched_gt_count += 1
        correctness[image_id] = matched_gt_count == len(image_gts)
    return correctness


def build_mcnemar_payload(
    detector_a_predictions: list[ScoredBox],
    detector_b_predictions: list[ScoredBox],
    ground_truths: list[GroundTruthBox],
    image_ids: list[str],
) -> dict[str, object]:
    """Build paired image-level correctness and McNemar statistics."""
    correct_a = image_level_correctness(detector_a_predictions, ground_truths, image_ids)
    correct_b = image_level_correctness(detector_b_predictions, ground_truths, image_ids)
    n00 = n01 = n10 = n11 = 0
    paired_rows: list[dict[str, object]] = []
    for image_id in image_ids:
        a_ok = bool(correct_a.get(image_id, False))
        b_ok = bool(correct_b.get(image_id, False))
        if not a_ok and not b_ok:
            n00 += 1
        elif not a_ok and b_ok:
            n01 += 1
        elif a_ok and not b_ok:
            n10 += 1
        else:
            n11 += 1
        paired_rows.append(
            {
                "image_id": image_id,
                "detector_a_correct": a_ok,
                "detector_b_correct": b_ok,
            }
        )
    stats = mcnemar_from_counts(n01=n01, n10=n10)
    return {
        "status": "computed",
        "n00_both_incorrect": n00,
        "n01_a_incorrect_b_correct": n01,
        "n10_a_correct_b_incorrect": n10,
        "n11_both_correct": n11,
        "statistic": stats["statistic"],
        "p_value": stats["p_value"],
        "paired_rows": paired_rows,
    }


def save_json(payload: dict, output_path: Path) -> None:
    """Atomically save JSON output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def save_paired_table(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save paired image-level correctness table for auditability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "detector_a_correct", "detector_b_correct"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    """Evaluate one or two detector prediction files on the annotated dev set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--detector-a", required=True)
    parser.add_argument("--detector-a-name", default="detector_a")
    parser.add_argument("--detector-b", default=None)
    parser.add_argument("--detector-b-name", default="detector_b")
    parser.add_argument("--manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--output", default="outputs/detection_dev_results.json")
    parser.add_argument("--paired-output", default=None)
    args = parser.parse_args()

    gt = load_ground_truth(Path(args.ground_truth))
    det_a = load_predictions(Path(args.detector_a))
    payload: dict[str, object] = {
        "ground_truth_count": len(gt),
        args.detector_a_name: compute_average_precision(det_a, gt, iou_threshold=0.5),
    }

    if args.detector_b:
        det_b = load_predictions(Path(args.detector_b))
        payload[args.detector_b_name] = compute_average_precision(det_b, gt, iou_threshold=0.5)
        image_ids = load_image_ids_from_manifest(Path(args.manifest))
        mcnemar_payload = build_mcnemar_payload(det_a, det_b, gt, image_ids=image_ids)
        paired_rows = mcnemar_payload.pop("paired_rows")
        payload["mcnemar"] = mcnemar_payload
        if args.paired_output:
            save_paired_table(paired_rows, Path(args.paired_output))

    save_json(payload, Path(args.output))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
