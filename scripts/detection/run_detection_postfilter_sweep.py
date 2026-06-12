#!/usr/bin/env python3

"""Sweep low-risk post-filters over saved CASTLE detector predictions."""

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


IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160


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


def box_metrics(box: tuple[int, int, int, int]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    min_side = min(width, height)
    return {
        "width": float(width),
        "height": float(height),
        "min_side": float(min_side),
    }


def is_edge_touching(box: tuple[int, int, int, int], margin: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= IMAGE_WIDTH - margin or y2 >= IMAGE_HEIGHT - margin


def should_drop(
    prediction: ScoredBox,
    edge_margin_px: int,
    large_min_side_px: int,
    max_score: float,
) -> bool:
    metrics = box_metrics(prediction.box)
    if not is_edge_touching(prediction.box, edge_margin_px):
        return False
    if metrics["min_side"] < large_min_side_px:
        return False
    if prediction.score > max_score:
        return False
    return True


def filter_predictions(
    predictions: list[ScoredBox],
    edge_margin_px: int,
    large_min_side_px: int,
    max_score: float,
) -> tuple[list[ScoredBox], int]:
    kept: list[ScoredBox] = []
    removed = 0
    for prediction in predictions:
        if should_drop(
            prediction,
            edge_margin_px=edge_margin_px,
            large_min_side_px=large_min_side_px,
            max_score=max_score,
        ):
            removed += 1
            continue
        kept.append(prediction)
    return kept, removed


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
        "--predictions",
        default="outputs/detection_eval_subset_yolo_scrfd_fallback.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/detection_postfilter_sweep_yolo_scrfd_fallback.json",
    )
    args = parser.parse_args()

    ground_truth = load_ground_truth(PROJECT_ROOT / args.ground_truth)
    predictions = load_predictions(PROJECT_ROOT / args.predictions)
    baseline = compute_average_precision(predictions, ground_truth, iou_threshold=0.5)

    results: list[dict[str, object]] = []
    for edge_margin_px in (0, 8, 16, 24, 32):
        for large_min_side_px in (160, 200, 240, 280, 320, 360, 400):
            for max_score in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65):
                filtered_predictions, removed = filter_predictions(
                    predictions,
                    edge_margin_px=edge_margin_px,
                    large_min_side_px=large_min_side_px,
                    max_score=max_score,
                )
                metrics = compute_average_precision(filtered_predictions, ground_truth, iou_threshold=0.5)
                results.append(
                    {
                        "edge_margin_px": edge_margin_px,
                        "large_min_side_px": large_min_side_px,
                        "max_score": max_score,
                        "removed_count": removed,
                        **metrics,
                        "delta_ap": metrics["ap"] - baseline["ap"],
                        "delta_precision": metrics["precision"] - baseline["precision"],
                        "delta_recall": metrics["recall"] - baseline["recall"],
                        "delta_f1": metrics["f1"] - baseline["f1"],
                        "delta_false_positives": metrics["false_positives"] - baseline["false_positives"],
                        "delta_false_negatives": metrics["false_negatives"] - baseline["false_negatives"],
                    }
                )

    viable = [
        row
        for row in results
        if row["delta_recall"] >= -0.003 and row["delta_false_positives"] < 0
    ]
    best_viable = max(viable, key=lambda row: (row["ap"], row["f1"], -row["false_positives"])) if viable else None
    best_ap = max(results, key=lambda row: (row["ap"], row["f1"], -row["false_positives"]))
    best_f1 = max(results, key=lambda row: (row["f1"], row["ap"], -row["false_positives"]))

    payload = {
        "baseline": baseline,
        "search_space": {
            "edge_margin_px": [0, 8, 16, 24, 32],
            "large_min_side_px": [160, 200, 240, 280, 320, 360, 400],
            "max_score": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        },
        "best_viable": best_viable,
        "best_ap": best_ap,
        "best_f1": best_f1,
        "top_viable": sorted(viable, key=lambda row: (row["ap"], row["f1"], -row["false_positives"]), reverse=True)[:20],
        "top_overall": sorted(results, key=lambda row: (row["ap"], row["f1"], -row["false_positives"]), reverse=True)[:20],
    }
    save_json(payload, PROJECT_ROOT / args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
