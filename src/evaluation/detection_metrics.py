"""Detection metrics for CASTLE face-detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ScoredBox:
    """One scored detection tied to an image identifier."""

    image_id: str
    box: BoundingBox
    score: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GroundTruthBox:
    """One ground-truth face box tied to an image identifier."""

    image_id: str
    box: BoundingBox
    metadata: dict[str, Any] | None = None


def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Return the intersection-over-union between two boxes."""
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


def match_detections(
    predictions: list[ScoredBox],
    ground_truths: list[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Greedily match scored detections to ground-truth boxes by IoU."""
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt.image_id, []).append(gt)

    matched_gt_indices: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    matches: list[dict[str, Any]] = []

    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        image_gts = gt_by_image.get(prediction.image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(image_gts):
            if index in matched_gt_indices[prediction.image_id]:
                continue
            iou = compute_iou(prediction.box, gt.box)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        is_true_positive = best_index >= 0 and best_iou >= iou_threshold
        if is_true_positive:
            matched_gt_indices[prediction.image_id].add(best_index)
        matches.append(
            {
                "image_id": prediction.image_id,
                "score": prediction.score,
                "iou": best_iou,
                "true_positive": is_true_positive,
                "false_positive": not is_true_positive,
            }
        )
    return matches


def compute_average_precision(
    predictions: list[ScoredBox],
    ground_truths: list[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute AP, precision, recall, and F1 from scored detections."""
    total_ground_truths = len(ground_truths)
    matches = match_detections(predictions, ground_truths, iou_threshold=iou_threshold)
    if total_ground_truths == 0:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_positives": len(predictions),
            "false_negatives": 0,
        }

    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for item in matches:
        if item["true_positive"]:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        precision = cumulative_tp / max(1, cumulative_tp + cumulative_fp)
        recall = cumulative_tp / total_ground_truths
        precisions.append(precision)
        recalls.append(recall)

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls, strict=False):
        ap += precision * max(0.0, recall - previous_recall)
        previous_recall = recall

    true_positives = cumulative_tp
    false_positives = cumulative_fp
    false_negatives = max(0, total_ground_truths - true_positives)
    final_precision = true_positives / max(1, true_positives + false_positives)
    final_recall = true_positives / max(1, total_ground_truths)
    final_f1 = (
        0.0
        if final_precision + final_recall == 0
        else 2 * final_precision * final_recall / (final_precision + final_recall)
    )
    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "f1": final_f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
