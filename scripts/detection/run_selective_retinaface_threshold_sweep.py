#!/usr/bin/env python3

"""Sweep selective RetinaFace merge thresholds against the reviewed 500-image subset."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.build_selective_retinaface_fallback_predictions import load_rows
from src.evaluation.detection_metrics import GroundTruthBox, ScoredBox, compute_average_precision
from scripts.detection.evaluate_detection_dev import build_mcnemar_payload, load_ground_truth, load_image_ids_from_manifest


def merge_rows(
    base_rows: list[dict[str, str]],
    fallback_rows: list[dict[str, str]],
    prediction_count_threshold: int,
    duplicate_iou_threshold: float,
    fallback_score_scale: float,
) -> list[dict[str, object]]:
    from collections import defaultdict
    from src.evaluation.detection_metrics import compute_iou

    base_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    fallback_by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in base_rows:
        base_by_image[row["image_id"]].append(row)
    for row in fallback_rows:
        fallback_by_image[row["image_id"]].append(row)

    merged_rows: list[dict[str, object]] = []
    for image_id in sorted(set(base_by_image) | set(fallback_by_image)):
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
        if len(base_image_rows) > prediction_count_threshold:
            continue
        existing_boxes = [
            (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            for row in base_image_rows
        ]
        for row in fallback_by_image.get(image_id, []):
            candidate_box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            if any(compute_iou(candidate_box, existing_box) >= duplicate_iou_threshold for existing_box in existing_boxes):
                continue
            merged_rows.append(
                {
                    "image_id": image_id,
                    "x1": candidate_box[0],
                    "y1": candidate_box[1],
                    "x2": candidate_box[2],
                    "y2": candidate_box[3],
                    "score": float(row["score"]) * fallback_score_scale,
                    "detector_name": f"yolo_scrfd_retinaface_le{prediction_count_threshold}",
                }
            )
            existing_boxes.append(candidate_box)
    return merged_rows


def to_scored(rows: list[dict[str, object]]) -> list[ScoredBox]:
    return [
        ScoredBox(
            image_id=str(row["image_id"]),
            box=(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])),
            score=float(row["score"]),
            metadata={"detector_name": str(row.get("detector_name", ""))},
        )
        for row in rows
    ]


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(handle.name)
    tmp.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/detection_eval_subset_yolo_scrfd_fallback.csv")
    parser.add_argument("--fallback", default="outputs/detection_eval_subset_retinaface.csv")
    parser.add_argument("--ground-truth", default="outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
    parser.add_argument("--manifest", default="outputs/01_protocol/thesis_manifests/final_face_detection_500.csv")
    parser.add_argument("--output-json", default="outputs/detection_selective_retinaface_threshold_sweep.json")
    parser.add_argument("--write-csv-dir", default="outputs/detection_selective_retinaface_threshold_sweep")
    parser.add_argument("--thresholds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--fallback-score-scale", type=float, default=0.95)
    args = parser.parse_args()

    base_rows = load_rows(PROJECT_ROOT / args.base)
    fallback_rows = load_rows(PROJECT_ROOT / args.fallback)
    gt = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    image_ids = load_image_ids_from_manifest(PROJECT_ROOT / args.manifest)
    base_scored = to_scored(
        [
            {
                "image_id": row["image_id"],
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "score": float(row["score"]),
                "detector_name": row.get("detector_name", ""),
            }
            for row in base_rows
        ]
    )
    baseline = compute_average_precision(base_scored, gt, iou_threshold=0.5)

    results = []
    best_f1 = None
    for threshold in args.thresholds:
        merged = merge_rows(
            base_rows=base_rows,
            fallback_rows=fallback_rows,
            prediction_count_threshold=threshold,
            duplicate_iou_threshold=args.duplicate_iou_threshold,
            fallback_score_scale=args.fallback_score_scale,
        )
        merged_path = Path(args.write_csv_dir) / f"detection_eval_subset_yolo_scrfd_retinaface_le{threshold}_sweep.csv"
        save_csv(merged, PROJECT_ROOT / merged_path)
        merged_scored = to_scored(merged)
        metrics = compute_average_precision(merged_scored, gt, iou_threshold=0.5)
        mcnemar = build_mcnemar_payload(base_scored, merged_scored, gt, image_ids=image_ids)
        row = {
            "threshold": threshold,
            "output_csv": str(merged_path),
            "ap": metrics["ap"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "delta_ap": metrics["ap"] - baseline["ap"],
            "delta_precision": metrics["precision"] - baseline["precision"],
            "delta_recall": metrics["recall"] - baseline["recall"],
            "delta_f1": metrics["f1"] - baseline["f1"],
            "mcnemar_p_value": mcnemar["p_value"],
            "mcnemar_n01": mcnemar["n01_a_incorrect_b_correct"],
            "mcnemar_n10": mcnemar["n10_a_correct_b_incorrect"],
        }
        results.append(row)
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row

    payload = {
        "version": "detection_selective_retinaface_threshold_sweep",
        "baseline": baseline,
        "threshold_results": results,
        "best_f1": best_f1,
    }
    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
