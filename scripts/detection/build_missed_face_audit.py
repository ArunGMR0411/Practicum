#!/usr/bin/env python3

"""Build a reproducible missed-face audit subset from reviewed annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def load_grouped_boxes(path: Path, image_column: str) -> dict[str, list[tuple[int, int, int, int]]]:
    """Load one CSV of boxes grouped by image id."""
    rows = pd.read_csv(path)
    grouped: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        grouped[getattr(row, image_column)].append((int(row.x1), int(row.y1), int(row.x2), int(row.y2)))
    return dict(grouped)


def match_ground_truths(
    ground_truth_boxes: list[tuple[int, int, int, int]],
    predicted_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> tuple[int, list[int]]:
    """Return matched GT count and indices of unmatched GT boxes."""
    matched_predictions: set[int] = set()
    unmatched_gt_indices: list[int] = []
    matched_gt_count = 0
    for gt_index, gt_box in enumerate(ground_truth_boxes):
        best_prediction_index = -1
        best_iou = 0.0
        for prediction_index, prediction_box in enumerate(predicted_boxes):
            if prediction_index in matched_predictions:
                continue
            iou = compute_iou(gt_box, prediction_box)
            if iou > best_iou:
                best_iou = iou
                best_prediction_index = prediction_index
        if best_prediction_index >= 0 and best_iou >= iou_threshold:
            matched_predictions.add(best_prediction_index)
            matched_gt_count += 1
        else:
            unmatched_gt_indices.append(gt_index)
    return matched_gt_count, unmatched_gt_indices


def atomic_write_csv(rows: list[dict[str, object]], fieldnames: list[str], output_path: Path) -> None:
    """Atomically write CSV rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


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
    parser.add_argument("--annotation-manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--ground-truth", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--detector-predictions", default="outputs/02_face_detection/01_yolo_predictions_run.csv")
    parser.add_argument("--detector-name", default="yolo_local")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", default="outputs/missed_face_audit.csv")
    parser.add_argument("--output-json", default="outputs/missed_face_audit_summary.json")
    parser.add_argument("--output-manifest", default="outputs/01_protocol/supporting_protocols/06_missed_face_audit.csv")
    args = parser.parse_args()

    annotation_manifest = pd.read_csv(PROJECT_ROOT / args.annotation_manifest)
    reviewed_manifest = annotation_manifest[annotation_manifest["annotation_status"].fillna("") == "reviewed"].copy()
    reviewed_manifest["box_count"] = reviewed_manifest["box_count"].fillna(0).astype(int)
    reviewed_manifest = reviewed_manifest[reviewed_manifest["box_count"] > 0].copy()

    ground_truth_by_image = load_grouped_boxes(PROJECT_ROOT / args.ground_truth, image_column="image_id")
    predictions_by_image = load_grouped_boxes(PROJECT_ROOT / args.detector_predictions, image_column="image_id")

    audit_rows: list[dict[str, object]] = []
    missed_condition_counter: Counter[str] = Counter()
    missed_day_counter: Counter[str] = Counter()
    missed_view_counter: Counter[str] = Counter()
    total_gt_faces = 0
    total_matched_faces = 0

    for row in reviewed_manifest.itertuples(index=False):
        image_id = row.relative_path
        ground_truth_boxes = ground_truth_by_image.get(image_id, [])
        predicted_boxes = predictions_by_image.get(image_id, [])
        if not ground_truth_boxes:
            continue
        matched_gt_count, unmatched_gt_indices = match_ground_truths(
            ground_truth_boxes=ground_truth_boxes,
            predicted_boxes=predicted_boxes,
            iou_threshold=args.iou_threshold,
        )
        total_gt_faces += len(ground_truth_boxes)
        total_matched_faces += matched_gt_count
        missed_face_count = len(unmatched_gt_indices)
        row_payload = {
            "relative_path": image_id,
            "camera_stream_id": row.camera_stream_id,
            "day_id": row.day_id,
            "view_type": row.view_type,
            "participant_id": row.participant_id,
            "condition_label": row.condition_label,
            "ground_truth_face_count": len(ground_truth_boxes),
            "predicted_face_count": len(predicted_boxes),
            "matched_face_count": matched_gt_count,
            "missed_face_count": missed_face_count,
            "missed_gt_indices": json.dumps(unmatched_gt_indices),
            "detector_name": args.detector_name,
            "iou_threshold": args.iou_threshold,
            "is_missed_face_case": missed_face_count > 0,
        }
        audit_rows.append(row_payload)
        if missed_face_count > 0:
            missed_condition_counter[str(row.condition_label)] += 1
            missed_day_counter[str(row.day_id)] += 1
            missed_view_counter[str(row.view_type)] += 1

    audit_df = pd.DataFrame(audit_rows).sort_values(
        by=["is_missed_face_case", "missed_face_count", "ground_truth_face_count", "relative_path"],
        ascending=[False, False, False, True],
    )
    missed_df = audit_df[audit_df["is_missed_face_case"]].copy()

    manifest_rows = reviewed_manifest.merge(
        missed_df[["relative_path", "ground_truth_face_count", "predicted_face_count", "matched_face_count", "missed_face_count"]],
        on="relative_path",
        how="inner",
    ).sort_values(by=["missed_face_count", "box_count", "relative_path"], ascending=[False, False, True])

    summary = {
        "detector_name": args.detector_name,
        "annotation_manifest": args.annotation_manifest,
        "ground_truth": args.ground_truth,
        "detector_predictions": args.detector_predictions,
        "reviewed_positive_images": int(len(reviewed_manifest)),
        "missed_face_images": int(len(missed_df)),
        "missed_face_rate_image_level": round(float(len(missed_df) / len(reviewed_manifest)), 4) if len(reviewed_manifest) else 0.0,
        "ground_truth_faces": int(total_gt_faces),
        "matched_ground_truth_faces": int(total_matched_faces),
        "missed_ground_truth_faces": int(total_gt_faces - total_matched_faces),
        "missed_face_rate_face_level": round(float((total_gt_faces - total_matched_faces) / total_gt_faces), 4) if total_gt_faces else 0.0,
        "condition_counts_missed_images": dict(sorted(missed_condition_counter.items())),
        "day_counts_missed_images": dict(sorted(missed_day_counter.items())),
        "view_counts_missed_images": dict(sorted(missed_view_counter.items())),
        "top_missed_cases": missed_df.head(15)[
            ["relative_path", "condition_label", "ground_truth_face_count", "predicted_face_count", "matched_face_count", "missed_face_count"]
        ].to_dict(orient="records"),
    }

    atomic_write_csv(audit_df.to_dict(orient="records"), list(audit_df.columns), PROJECT_ROOT / args.output_csv)
    atomic_write_csv(manifest_rows.to_dict(orient="records"), list(manifest_rows.columns), PROJECT_ROOT / args.output_manifest)
    atomic_write_json(summary, PROJECT_ROOT / args.output_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
