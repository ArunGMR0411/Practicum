#!/usr/bin/env python3

"""Targeted error-bank hardening for the face detector policy.

This uses retained detector candidate boxes. It does not rerun detector
inference. The goal is to test whether the current RF-DETR-aware reranker can
be pushed materially higher by recovering false negatives and suppressing false
positives from the existing candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_face_detector_hardening_experiment import (  # noqa: E402
    Protocol,
    build_subgroup_membership,
    load_records,
    score_predictions,
    write_csv,
)
from scripts.detection.run_low_compute_detector_policy_experiment import (  # noqa: E402
    SCENE_PREDICTIONS,
    load_scene_predictions,
)
from src.evaluation.detection_metrics import ScoredBox, compute_iou  # noqa: E402


OUTPUT_DEFAULT = "outputs/02_face_detection/12_detector_error_hardening"
CANDIDATE_BOXES = PROJECT_ROOT / "outputs/02_face_detection/11_detector_candidate_box_telemetry/detector_candidate_boxes.csv"
CONDITION_DATASET = PROJECT_ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv"
CURRENT_POLICY = "cv_box_reranker_with_rfdetr_predicted_conditions"

SOURCE_SETS = {
    "rfdetr_clean": [
        "rfdetr_medium_face_030",
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
        "yolo8s_widerface_repo_640",
        "yolo11n_pose_widerface_640",
    ],
    # Live App bank: only detectors the Gradio runtime generates (RF-DETR + YOLO11-1280 + SCRFD).
    "runtime_3": [
        "rfdetr_medium_face_030",
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
    ],
    "all_raw": [
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
        "yolo8s_widerface_repo_640",
        "yolo11n_pose_widerface_640",
        "yolo11s_widerface_640",
        "sliced_yolo11s_widerface_1280",
        "rfdetr_medium_face_030",
    ],
    "raw_plus_fusions": [
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
        "yolo8s_widerface_repo_640",
        "yolo11n_pose_widerface_640",
        "yolo11s_widerface_640",
        "sliced_yolo11s_widerface_1280",
        "rfdetr_medium_face_030",
        "fusion_yolo11s1280_scrfd10g",
        "fusion_yolo11s_scrfd10g",
        "fusion_rfdetr_scrfd10g",
    ],
}

CONDITION_NAMES = [
    "no_face",
    "single_face",
    "multi_face",
    "small_face",
    "medium_face",
    "large_face",
    "mixed_scale_face",
    "very_small_or_distant_face",
    "edge_or_partial_face",
    "profile_or_occluded_face",
    "downward_egocentric_view",
    "motion_blur_or_low_sharpness",
    "low_light_or_dim",
    "high_clutter",
]


def write_csv_union(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol_records():
    protocols = [
        Protocol("01_baseline_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        Protocol(
            "02_egocentric_stress_500",
            PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
        ),
    ]
    return load_records(protocols)


def load_image_features() -> dict[str, dict[str, float]]:
    protocol_map = {
        "baseline_500": "01_baseline_500",
        "egocentric_stress_500": "02_egocentric_stress_500",
    }
    df = pd.read_csv(CONDITION_DATASET)
    rows: dict[str, dict[str, float]] = {}
    for row in df.itertuples(index=False):
        protocol = protocol_map.get(str(row.protocol), str(row.protocol))
        key = f"{protocol}::{row.relative_path}"
        rows[key] = {
            "brightness": float(getattr(row, "img_brightness_mean", 0.0)),
            "sharpness": float(getattr(row, "img_sharpness_laplacian_var", 0.0)),
            "edge_density": float(getattr(row, "img_edge_density", 0.0)),
            "saturation": float(getattr(row, "img_saturation_mean", 0.0)),
        }
    return rows


def candidate_boxes_to_predictions(candidate_df: pd.DataFrame, sources: list[str], method_name: str) -> list[ScoredBox]:
    rows: list[ScoredBox] = []
    for row in candidate_df[candidate_df["detector_name"].isin(sources)].itertuples(index=False):
        rows.append(
            ScoredBox(
                image_id=str(row.scoped_image_id),
                box=(int(row.x1), int(row.y1), int(row.x2), int(row.y2)),
                score=float(row.score),
                metadata={"source_detector": method_name if len(sources) == 1 else str(row.detector_name)},
            )
        )
    return rows


def cluster_candidate_boxes(candidate_df: pd.DataFrame, sources: list[str], iou_threshold: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    subset = candidate_df[candidate_df["detector_name"].isin(sources)]
    for image_id, group in subset.groupby("scoped_image_id"):
        pending = [
            {
                "box": (int(row.x1), int(row.y1), int(row.x2), int(row.y2)),
                "score": float(row.score),
                "source": str(row.detector_name),
            }
            for row in group.itertuples(index=False)
        ]
        pending = sorted(pending, key=lambda item: item["score"], reverse=True)
        while pending:
            seed = pending.pop(0)
            cluster = [seed]
            remaining = []
            for item in pending:
                if compute_iou(seed["box"], item["box"]) >= iou_threshold:
                    cluster.append(item)
                else:
                    remaining.append(item)
            pending = remaining
            weights = [max(item["score"], 1e-6) for item in cluster]
            total = sum(weights)
            box = tuple(
                int(round(sum(item["box"][idx] * weight for item, weight in zip(cluster, weights, strict=False)) / total))
                for idx in range(4)
            )
            source_scores = {
                source: max([item["score"] for item in cluster if item["source"] == source] or [0.0])
                for source in sources
            }
            clusters.append(
                {
                    "image_id": str(image_id),
                    "box": box,
                    "score": max(item["score"] for item in cluster),
                    "sources": {item["source"] for item in cluster},
                    "source_scores": source_scores,
                }
            )
    return clusters


def assign_cluster_labels(clusters: list[dict[str, Any]], records_by_id: dict[str, Any]) -> None:
    for cluster in clusters:
        record = records_by_id[cluster["image_id"]]
        cluster["label"] = int(any(compute_iou(cluster["box"], gt_box) >= 0.5 for gt_box in record.gt_boxes))


def cluster_features(
    clusters: list[dict[str, Any]],
    sources: list[str],
    records_by_id: dict[str, Any],
    scene_predictions: dict[str, dict[str, int]],
    image_features: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for cluster in clusters:
        record = records_by_id[cluster["image_id"]]
        width = float(record.attributes.get("image_width") or 3840)
        height = float(record.attributes.get("image_height") or 2160)
        x1, y1, x2, y2 = cluster["box"]
        bw = max(0.0, (x2 - x1) / max(width, 1.0))
        bh = max(0.0, (y2 - y1) / max(height, 1.0))
        cx = ((x1 + x2) / 2.0) / max(width, 1.0)
        cy = ((y1 + y2) / 2.0) / max(height, 1.0)
        edge_distance = min(
            x1 / max(width, 1.0),
            y1 / max(height, 1.0),
            (width - x2) / max(width, 1.0),
            (height - y2) / max(height, 1.0),
        )
        source_scores = np.asarray([cluster["source_scores"].get(source, 0.0) for source in sources], dtype=float)
        feature_row = [
            float(cluster["score"]),
            float(len(cluster["sources"])),
            bw,
            bh,
            bw * bh,
            cx,
            cy,
            edge_distance,
            bh / max(bw, 1e-9),
            float(source_scores.max()) if len(source_scores) else 0.0,
            float(source_scores.mean()) if len(source_scores) else 0.0,
            float(source_scores.std()) if len(source_scores) else 0.0,
            float(source_scores.max() - source_scores.min()) if len(source_scores) else 0.0,
            float((source_scores > 0).sum()),
        ]
        for source in sources:
            feature_row.extend(
                [
                    float(source in cluster["sources"]),
                    float(cluster["source_scores"].get(source, 0.0)),
                ]
            )
        scene = scene_predictions.get(cluster["image_id"], {})
        feature_row.extend(float(scene.get(name, 0)) for name in CONDITION_NAMES)
        image = image_features.get(cluster["image_id"], {})
        feature_row.extend(
            [
                float(image.get("brightness", 0.0)),
                float(image.get("sharpness", 0.0)),
                float(image.get("edge_density", 0.0)),
                float(image.get("saturation", 0.0)),
            ]
        )
        rows.append(feature_row)
        labels.append(int(cluster["label"]))
        groups.append(cluster["image_id"])
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), np.asarray(groups)


def classifier_factory(name: str):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight={0: 1.0, 1: 2.5}, max_iter=1000, solver="liblinear"),
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight={0: 1.0, 1: 2.5},
            random_state=7,
            n_jobs=-1,
        )
    if name == "extra":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=2,
            class_weight={0: 1.0, 1: 2.5},
            random_state=8,
            n_jobs=-1,
        )
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=24,
            learning_rate=0.04,
            l2_regularization=0.02,
            random_state=9,
        )
    raise ValueError(f"Unknown classifier: {name}")


def predictions_from_clusters(
    clusters: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    method_name: str,
) -> list[ScoredBox]:
    return [
        ScoredBox(
            image_id=cluster["image_id"],
            box=cluster["box"],
            score=float(probability),
            metadata={"source_detector": method_name},
        )
        for cluster, probability in zip(clusters, probabilities, strict=True)
        if probability >= threshold
    ]


def all_images_score(method_name: str, predictions: list[ScoredBox], records: list[Any]) -> dict[str, Any]:
    rows = score_predictions(method_name, predictions, records, build_subgroup_membership(records), "combined_1000")
    return next(row for row in rows if row["subgroup"] == "all_images")


def choose_threshold(
    train_clusters: list[dict[str, Any]],
    train_probabilities: np.ndarray,
    train_records: list[Any],
    method_name: str,
) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_row: dict[str, Any] | None = None
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = predictions_from_clusters(train_clusters, train_probabilities, float(threshold), method_name)
        row = all_images_score(method_name, predictions, train_records)
        score = float(row["oapr_detector_score"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_row = row
    return best_threshold, best_row or {}


def cross_validated_variant(
    clusters: list[dict[str, Any]],
    records: list[Any],
    sources: list[str],
    scene_predictions: dict[str, dict[str, int]],
    image_features: dict[str, dict[str, float]],
    classifier_name: str,
    method_name: str,
) -> tuple[list[ScoredBox], list[dict[str, Any]]]:
    records_by_id = {record.scoped_id: record for record in records}
    X, y, groups = cluster_features(clusters, sources, records_by_id, scene_predictions, image_features)
    folds = GroupKFold(n_splits=5)
    predictions: list[ScoredBox] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X, y, groups=groups), start=1):
        classifier = classifier_factory(classifier_name)
        classifier.fit(X[train_idx], y[train_idx])
        train_probabilities = classifier.predict_proba(X[train_idx])[:, 1]
        test_probabilities = classifier.predict_proba(X[test_idx])[:, 1]
        train_clusters = [clusters[index] for index in train_idx]
        test_clusters = [clusters[index] for index in test_idx]
        train_record_ids = {cluster["image_id"] for cluster in train_clusters}
        train_records = [record for record in records if record.scoped_id in train_record_ids]
        threshold, train_row = choose_threshold(train_clusters, train_probabilities, train_records, method_name)
        predictions.extend(predictions_from_clusters(test_clusters, test_probabilities, threshold, method_name))
        fold_rows.append(
            {
                "method": method_name,
                "fold": fold_idx,
                "threshold": round(float(threshold), 4),
                "train_clusters": len(train_idx),
                "test_clusters": len(test_idx),
                "positive_train_rate": round(float(y[train_idx].mean()), 6),
                "positive_test_rate": round(float(y[test_idx].mean()), 6),
                "train_oapr_detector_score": round(float(train_row.get("oapr_detector_score", 0.0)), 6),
            }
        )
    return predictions, fold_rows


def match_error_rows(
    method_name: str,
    predictions: list[ScoredBox],
    records: list[Any],
    candidate_pool: list[ScoredBox],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    pred_by_image: dict[str, list[ScoredBox]] = defaultdict(list)
    candidate_by_image: dict[str, list[ScoredBox]] = defaultdict(list)
    for prediction in predictions:
        pred_by_image[prediction.image_id].append(prediction)
    for candidate in candidate_pool:
        candidate_by_image[candidate.image_id].append(candidate)

    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for record in records:
        image_predictions = sorted(pred_by_image.get(record.scoped_id, []), key=lambda item: item.score, reverse=True)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for pred_index, prediction in enumerate(image_predictions):
            best_gt_index = -1
            best_iou = 0.0
            for gt_index, gt_box in enumerate(record.gt_boxes):
                if gt_index in matched_gt:
                    continue
                iou = compute_iou(prediction.box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index
            if best_gt_index >= 0 and best_iou >= 0.5:
                matched_gt.add(best_gt_index)
                matched_pred.add(pred_index)
                counts["tp"] += 1
        for pred_index, prediction in enumerate(image_predictions):
            if pred_index in matched_pred:
                continue
            counts["fp"] += 1
            fp_rows.append(
                {
                    "method": method_name,
                    "protocol": record.protocol,
                    "image_id": record.scoped_id,
                    "relative_path": record.relative_path,
                    "x1": prediction.box[0],
                    "y1": prediction.box[1],
                    "x2": prediction.box[2],
                    "y2": prediction.box[3],
                    "score": round(float(prediction.score), 6),
                    "source_detector": (prediction.metadata or {}).get("source_detector", ""),
                    "face_count_category": record.attributes.get("face_count_category", ""),
                    "face_scale_category": record.attributes.get("face_scale_category", ""),
                    "edge_partial_face": record.attributes.get("edge_partial_face", ""),
                    "blur_low_sharpness": record.attributes.get("blur_low_sharpness", ""),
                    "clutter_level": record.attributes.get("clutter_level", ""),
                }
            )
        for gt_index, gt_box in enumerate(record.gt_boxes):
            if gt_index in matched_gt:
                continue
            counts["fn"] += 1
            best_candidate_iou = 0.0
            best_candidate_source = ""
            best_candidate_score = 0.0
            for candidate in candidate_by_image.get(record.scoped_id, []):
                iou = compute_iou(candidate.box, gt_box)
                if iou > best_candidate_iou:
                    best_candidate_iou = iou
                    best_candidate_source = str((candidate.metadata or {}).get("source_detector", ""))
                    best_candidate_score = float(candidate.score)
            fn_rows.append(
                {
                    "method": method_name,
                    "protocol": record.protocol,
                    "image_id": record.scoped_id,
                    "relative_path": record.relative_path,
                    "gt_index": gt_index,
                    "x1": gt_box[0],
                    "y1": gt_box[1],
                    "x2": gt_box[2],
                    "y2": gt_box[3],
                    "best_candidate_iou": round(best_candidate_iou, 6),
                    "best_candidate_source": best_candidate_source,
                    "best_candidate_score": round(best_candidate_score, 6),
                    "recoverable_at_iou_0_5": int(best_candidate_iou >= 0.5),
                    "recoverable_at_iou_0_4": int(best_candidate_iou >= 0.4),
                    "recoverable_at_iou_0_3": int(best_candidate_iou >= 0.3),
                    "face_count_category": record.attributes.get("face_count_category", ""),
                    "face_scale_category": record.attributes.get("face_scale_category", ""),
                    "edge_partial_face": record.attributes.get("edge_partial_face", ""),
                    "blur_low_sharpness": record.attributes.get("blur_low_sharpness", ""),
                    "clutter_level": record.attributes.get("clutter_level", ""),
                }
            )
    return fp_rows, fn_rows, counts


def markdown_summary(path: Path, score_rows: list[dict[str, Any]], recoverability_rows: list[dict[str, Any]]) -> None:
    ranked = sorted(score_rows, key=lambda row: float(row["oapr_detector_score"]), reverse=True)
    current = next(row for row in score_rows if row["model"] == CURRENT_POLICY)
    best = ranked[0]
    lines = [
        "# Detector Error Hardening",
        "",
        "Purpose: test whether the final face detector policy can be pushed toward `0.95` by mining false positives/false negatives from retained detector candidate boxes.",
        "",
        "Result:",
        "",
        f"- Current final policy `{CURRENT_POLICY}`: OAPR detector score `{float(current['oapr_detector_score']):.4f}`, precision `{float(current['precision']):.4f}`, recall `{float(current['recall']):.4f}`, F1 `{float(current['f1']):.4f}`, TP `{current['true_positives']}`, FP `{current['false_positives']}`, FN `{current['false_negatives']}`.",
        f"- Best hardening variant `{best['model']}`: OAPR detector score `{float(best['oapr_detector_score']):.4f}`, precision `{float(best['precision']):.4f}`, recall `{float(best['recall']):.4f}`, F1 `{float(best['f1']):.4f}`, TP `{best['true_positives']}`, FP `{best['false_positives']}`, FN `{best['false_negatives']}`.",
        "- The retained candidate pool does not support a defensible `0.95` deployable score by reranking alone.",
        "",
        "Candidate-pool recoverability:",
        "",
        "| Pool | Current FNs | Recoverable @ IoU 0.5 | Recoverable @ IoU 0.4 | Recoverable @ IoU 0.3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in recoverability_rows:
        lines.append(
            f"| {row['candidate_pool']} | {row['current_false_negatives']} | "
            f"{row['recoverable_iou_0_5']} | {row['recoverable_iou_0_4']} | {row['recoverable_iou_0_3']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Some current missed faces are recoverable from retained candidates, but accepting them without new evidence increases false positives.",
            "- The best fold-safe reranker improves precision and the OAPR detector score modestly, but does not recover enough faces to reach `0.95`.",
            "- Reaching `0.95` likely requires new candidate generation for the remaining non-recoverable false negatives or stronger crop-level face/non-face evidence, not just threshold tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_protocol_records()
    records_by_id = {record.scoped_id: record for record in records}
    candidate_df = pd.read_csv(CANDIDATE_BOXES)
    scene_predictions = load_scene_predictions(SCENE_PREDICTIONS)
    image_features = load_image_features()

    current_predictions = candidate_boxes_to_predictions(candidate_df, [CURRENT_POLICY], CURRENT_POLICY)
    raw_pool = candidate_boxes_to_predictions(candidate_df, SOURCE_SETS["all_raw"], "raw_candidate_pool")
    all_pool = candidate_boxes_to_predictions(
        candidate_df,
        SOURCE_SETS["raw_plus_fusions"] + [CURRENT_POLICY],
        "all_candidate_pool",
    )
    current_fp_rows, current_fn_rows, _ = match_error_rows(CURRENT_POLICY, current_predictions, records, raw_pool)

    recoverability_rows: list[dict[str, Any]] = []
    for pool_name, pool in [("raw_pool", raw_pool), ("raw_plus_fusions_plus_current", all_pool)]:
        _, fn_rows, _ = match_error_rows(CURRENT_POLICY, current_predictions, records, pool)
        recoverability_rows.append(
            {
                "candidate_pool": pool_name,
                "current_false_negatives": len(fn_rows),
                "recoverable_iou_0_5": sum(int(row["recoverable_at_iou_0_5"]) for row in fn_rows),
                "recoverable_iou_0_4": sum(int(row["recoverable_at_iou_0_4"]) for row in fn_rows),
                "recoverable_iou_0_3": sum(int(row["recoverable_at_iou_0_3"]) for row in fn_rows),
            }
        )

    score_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    current_score = all_images_score(CURRENT_POLICY, current_predictions, records)
    score_rows.append({"model": CURRENT_POLICY, "variant_set": "current", "classifier": "", "cluster_iou": "", **current_score})

    variant_predictions: dict[str, list[ScoredBox]] = {CURRENT_POLICY: current_predictions}
    for set_name, sources in SOURCE_SETS.items():
        for cluster_iou in [0.45, 0.50, 0.60]:
            clusters = cluster_candidate_boxes(candidate_df, sources, cluster_iou)
            assign_cluster_labels(clusters, records_by_id)
            for classifier_name in ["logreg", "rf", "extra", "histgb"]:
                method_name = f"error_hardened_{set_name}_{classifier_name}_iou{str(cluster_iou).replace('.', '_')}"
                predictions, folds = cross_validated_variant(
                    clusters=clusters,
                    records=records,
                    sources=sources,
                    scene_predictions=scene_predictions,
                    image_features=image_features,
                    classifier_name=classifier_name,
                    method_name=method_name,
                )
                variant_predictions[method_name] = predictions
                row = all_images_score(method_name, predictions, records)
                score_rows.append(
                    {
                        "model": method_name,
                        "variant_set": set_name,
                        "classifier": classifier_name,
                        "cluster_iou": cluster_iou,
                        "cluster_count": len(clusters),
                        "prediction_count": len(predictions),
                        **row,
                    }
                )
                for fold in folds:
                    fold["variant_set"] = set_name
                    fold["classifier"] = classifier_name
                    fold["cluster_iou"] = cluster_iou
                threshold_rows.extend(folds)
                print(
                    f"{method_name}: score={float(row['oapr_detector_score']):.6f}, "
                    f"P={float(row['precision']):.4f}, R={float(row['recall']):.4f}, F1={float(row['f1']):.4f}",
                    flush=True,
                )

    ranked = sorted(score_rows, key=lambda row: float(row["oapr_detector_score"]), reverse=True)
    best_name = ranked[0]["model"]
    best_predictions = variant_predictions[best_name]
    best_fp_rows, best_fn_rows, _ = match_error_rows(best_name, best_predictions, records, raw_pool)

    prediction_rows = [
        {
            "method": best_name,
            "image_id": prediction.image_id,
            "x1": prediction.box[0],
            "y1": prediction.box[1],
            "x2": prediction.box[2],
            "y2": prediction.box[3],
            "score": round(float(prediction.score), 6),
            "source_detector": (prediction.metadata or {}).get("source_detector", ""),
        }
        for prediction in best_predictions
    ]

    write_csv_union(output_dir / "detector_error_hardening_scores.csv", score_rows)
    write_csv_union(output_dir / "detector_error_hardening_thresholds.csv", threshold_rows)
    write_csv_union(output_dir / "detector_error_hardening_recoverability.csv", recoverability_rows)
    write_csv_union(output_dir / "detector_error_bank_current_false_positives.csv", current_fp_rows)
    write_csv_union(output_dir / "detector_error_bank_current_false_negatives.csv", current_fn_rows)
    write_csv_union(output_dir / "detector_error_bank_best_false_positives.csv", best_fp_rows)
    write_csv_union(output_dir / "detector_error_bank_best_false_negatives.csv", best_fn_rows)
    write_csv_union(output_dir / "detector_error_hardened_best_predictions.csv", prediction_rows)
    markdown_summary(output_dir / "detector_error_hardening_summary.md", score_rows, recoverability_rows)
    print(f"Wrote detector error-hardening outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
