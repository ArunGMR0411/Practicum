#!/usr/bin/env python3

"""Low-compute detector-policy and box-reranker experiment.

This follows the detector-hardening result without adding heavyweight detector
families. It compares the current best agreement fusion against category-aware
policies and a cross-validated box-level reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detection.run_face_detector_hardening_experiment import (  # noqa: E402
    Candidate,
    ImageRecord,
    Protocol,
    build_subgroup_membership,
    load_records,
    nms_fusion,
    run_candidate,
    score_predictions,
    write_csv,
)
from src.evaluation.detection_metrics import ScoredBox, compute_iou  # noqa: E402


OUTPUT_DEFAULT = "outputs/02_face_detection/07_low_compute_detector_policy_experiment"
SCENE_PREDICTIONS = PROJECT_ROOT / "outputs/02_face_detection/04_scene_condition_router/08_final_predictions.csv"
SCENE_METHOD = "handcrafted_yolo_multiscale__logistic_regression"


@dataclass
class Cluster:
    image_id: str
    box: tuple[int, int, int, int]
    score: float
    sources: set[str]
    source_scores: dict[str, float]
    label: int
    protocol: str


def load_scene_predictions(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    if not path.exists():
        return rows
    protocol_map = {
        "baseline_500": "01_baseline_500",
        "egocentric_stress_500": "02_egocentric_stress_500",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method_id"] != SCENE_METHOD:
                continue
            protocol = protocol_map.get(row["protocol"], row["protocol"])
            key = f"{protocol}::{row['relative_path']}"
            rows[key] = {
                name.replace("pred_", ""): int(row[name])
                for name in row
                if name.startswith("pred_")
            }
    return rows


def true_condition_features(record: ImageRecord) -> dict[str, int]:
    attrs = record.attributes
    return {
        "no_face": int(attrs.get("face_count_category") == "no_face"),
        "single_face": int(attrs.get("face_count_category") == "single_face"),
        "multi_face": int(attrs.get("face_count_category") == "multi_face"),
        "small_face": int(attrs.get("face_scale_category") == "small"),
        "medium_face": int(attrs.get("face_scale_category") == "medium"),
        "large_face": int(attrs.get("face_scale_category") == "large"),
        "mixed_scale_face": int(attrs.get("face_scale_category") == "mixed_scale"),
        "very_small_or_distant_face": int(attrs.get("face_scale_category") == "very_small_or_distant"),
        "edge_or_partial_face": int(attrs.get("edge_partial_face") == "yes"),
        "profile_or_occluded_face": int(attrs.get("profile_occluded_face") == "yes"),
        "downward_egocentric_view": int(attrs.get("downward_egocentric_view") == "yes"),
        "motion_blur_or_low_sharpness": int(attrs.get("blur_low_sharpness") == "yes"),
        "low_light_or_dim": int(attrs.get("low_light_dim") == "yes"),
        "high_clutter": int(attrs.get("clutter_level") == "high"),
        "outdoor_or_vehicle_scene": int(attrs.get("outdoor_vehicle_scene") == "yes"),
    }


def cluster_predictions(
    predictions_by_model: dict[str, list[ScoredBox]],
    source_names: list[str],
    records_by_id: dict[str, ImageRecord],
    iou_threshold: float = 0.5,
) -> list[Cluster]:
    by_image: dict[str, list[ScoredBox]] = defaultdict(list)
    for source in source_names:
        for item in predictions_by_model[source]:
            by_image[item.image_id].append(item)

    clusters: list[Cluster] = []
    for image_id, boxes in by_image.items():
        pending = sorted(boxes, key=lambda item: item.score, reverse=True)
        record = records_by_id[image_id]
        while pending:
            seed = pending.pop(0)
            group = [seed]
            rest = []
            for item in pending:
                if compute_iou(seed.box, item.box) >= iou_threshold:
                    group.append(item)
                else:
                    rest.append(item)
            pending = rest
            weights = [max(item.score, 1e-6) for item in group]
            total = sum(weights)
            coords = tuple(
                int(round(sum(item.box[idx] * weight for item, weight in zip(group, weights, strict=False)) / total))
                for idx in range(4)
            )
            sources = {
                str(item.metadata.get("source_detector"))
                for item in group
                if item.metadata and item.metadata.get("source_detector")
            }
            source_scores = {
                source: max(
                    [item.score for item in group if item.metadata and item.metadata.get("source_detector") == source]
                    or [0.0]
                )
                for source in source_names
            }
            label = int(any(compute_iou(coords, gt_box) >= 0.5 for gt_box in record.gt_boxes))
            clusters.append(
                Cluster(
                    image_id=image_id,
                    box=coords,
                    score=max(item.score for item in group),
                    sources=sources,
                    source_scores=source_scores,
                    label=label,
                    protocol=record.protocol,
                )
            )
    return clusters


def cluster_to_features(
    cluster: Cluster,
    record: ImageRecord,
    source_names: list[str],
    scene_predictions: dict[str, dict[str, int]],
    use_oracle_conditions: bool,
) -> list[float]:
    width = float(record.attributes.get("image_width") or 3840)
    height = float(record.attributes.get("image_height") or 2160)
    x1, y1, x2, y2 = cluster.box
    bw = max(0.0, (x2 - x1) / max(width, 1.0))
    bh = max(0.0, (y2 - y1) / max(height, 1.0))
    cx = ((x1 + x2) / 2.0) / max(width, 1.0)
    cy = ((y1 + y2) / 2.0) / max(height, 1.0)
    edge_distance = min(x1 / max(width, 1.0), y1 / max(height, 1.0), (width - x2) / max(width, 1.0), (height - y2) / max(height, 1.0))
    geom = [
        cluster.score,
        len(cluster.sources),
        bw,
        bh,
        bw * bh,
        cx,
        cy,
        edge_distance,
        bh / max(bw, 1e-6),
    ]
    source_features: list[float] = []
    for source in source_names:
        source_features.append(float(source in cluster.sources))
        source_features.append(float(cluster.source_scores.get(source, 0.0)))

    if use_oracle_conditions:
        condition_map = true_condition_features(record)
    else:
        condition_map = scene_predictions.get(cluster.image_id, {})
    condition_names = [
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
        "outdoor_or_vehicle_scene",
    ]
    return geom + source_features + [float(condition_map.get(name, 0)) for name in condition_names]


def predictions_from_clusters(
    clusters: list[Cluster],
    probabilities: np.ndarray,
    threshold: float,
    method_name: str,
) -> list[ScoredBox]:
    rows: list[ScoredBox] = []
    for cluster, prob in zip(clusters, probabilities, strict=True):
        if prob < threshold:
            continue
        rows.append(
            ScoredBox(
                image_id=cluster.image_id,
                box=cluster.box,
                score=float(prob),
                metadata={"source_detector": method_name},
            )
        )
    return rows


def score_model(model_name: str, predictions: list[ScoredBox], records: list[ImageRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol in sorted({record.protocol for record in records}):
        protocol_records = [record for record in records if record.protocol == protocol]
        protocol_predictions = [item for item in predictions if item.image_id.startswith(f"{protocol}::")]
        rows.extend(score_predictions(model_name, protocol_predictions, protocol_records, build_subgroup_membership(protocol_records), protocol))
    rows.extend(score_predictions(model_name, predictions, records, build_subgroup_membership(records), "combined_1000"))
    return rows


def combined_score(rows: list[dict[str, Any]], subgroup: str = "all_images") -> float:
    for row in rows:
        if row["protocol"] == "combined_1000" and row["subgroup"] == subgroup:
            return float(row["oapr_detector_score"])
    raise ValueError(f"missing combined_1000/{subgroup}")


def choose_threshold(train_clusters: list[Cluster], train_probs: np.ndarray, train_records: list[ImageRecord]) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.15, 0.90, 31):
        predictions = predictions_from_clusters(train_clusters, train_probs, float(threshold), "threshold_search")
        rows = score_model("threshold_search", predictions, train_records)
        score = combined_score(rows)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def cross_validated_reranker(
    clusters: list[Cluster],
    records: list[ImageRecord],
    source_names: list[str],
    scene_predictions: dict[str, dict[str, int]],
    use_oracle_conditions: bool,
) -> tuple[list[ScoredBox], list[dict[str, Any]]]:
    records_by_id = {record.scoped_id: record for record in records}
    groups = np.array([cluster.image_id for cluster in clusters])
    indices = np.arange(len(clusters))
    folds = GroupKFold(n_splits=5)
    out_probs = np.zeros(len(clusters), dtype=float)
    thresholds: list[dict[str, Any]] = []

    X = np.array(
        [
            cluster_to_features(cluster, records_by_id[cluster.image_id], source_names, scene_predictions, use_oracle_conditions)
            for cluster in clusters
        ],
        dtype=float,
    )
    y = np.array([cluster.label for cluster in clusters], dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(folds.split(X, y, groups=groups), start=1):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight={0: 1.0, 1: 2.5},
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        )
        clf.fit(X[train_idx], y[train_idx])
        train_probs = clf.predict_proba(X[train_idx])[:, 1]
        test_probs = clf.predict_proba(X[test_idx])[:, 1]
        train_clusters = [clusters[i] for i in train_idx]
        test_clusters = [clusters[i] for i in test_idx]
        train_record_ids = {cluster.image_id for cluster in train_clusters}
        train_records = [record for record in records if record.scoped_id in train_record_ids]
        threshold = choose_threshold(train_clusters, train_probs, train_records)
        out_probs[test_idx] = test_probs
        thresholds.append(
            {
                "fold": fold_idx,
                "threshold": round(threshold, 4),
                "train_clusters": len(train_idx),
                "test_clusters": len(test_idx),
                "positive_train_rate": round(float(y[train_idx].mean()), 6),
                "positive_test_rate": round(float(y[test_idx].mean()), 6),
            }
        )
    final_threshold = float(np.median([row["threshold"] for row in thresholds]))
    return predictions_from_clusters(clusters, out_probs, final_threshold, "cv_box_reranker"), thresholds


def category_policy_predictions(
    predictions_by_model: dict[str, list[ScoredBox]],
    records: list[ImageRecord],
    scene_predictions: dict[str, dict[str, int]],
    use_oracle_conditions: bool,
) -> list[ScoredBox]:
    """Select a detector/fusion source per image from condition evidence."""
    records_by_id = {record.scoped_id: record for record in records}
    source_by_image: dict[str, str] = {}
    for record in records:
        if use_oracle_conditions:
            c = true_condition_features(record)
        else:
            c = scene_predictions.get(record.scoped_id, {})
        if c.get("no_face"):
            source = "scrfd_10g_current_640"
        elif c.get("large_face") or c.get("low_light_or_dim"):
            source = "yolo11n_pose_widerface_640"
        elif c.get("motion_blur_or_low_sharpness") or c.get("single_face"):
            source = "yolo8s_widerface_repo_640"
        elif c.get("medium_face") or c.get("profile_or_occluded_face"):
            source = "fusion_yolo11s_scrfd10g_agreement"
        else:
            source = "fusion_yolo11s1280_scrfd10g_agreement"
        source_by_image[record.scoped_id] = source
    selected: list[ScoredBox] = []
    for source, predictions in predictions_by_model.items():
        for item in predictions:
            if source_by_image.get(item.image_id) == source:
                selected.append(
                    ScoredBox(
                        image_id=item.image_id,
                        box=item.box,
                        score=item.score,
                        metadata={"source_detector": f"category_policy:{source}"},
                    )
                )
    return selected


def write_markdown_summary(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    overall = [row for row in rows if row["subgroup"] == "all_images"]
    overall = sorted(overall, key=lambda row: (row["protocol"], -float(row["oapr_detector_score"])))
    lines = [
        f"# {title}",
        "",
        "| Protocol | Method | OAPR score | F1 | Precision | Recall | TP | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {float(row['oapr_detector_score']):.4f} | "
            f"{float(row['f1']):.4f} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
            f"{row['true_positives']} | {row['false_positives']} | {row['false_negatives']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    started = perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocols = [
        Protocol("01_baseline_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        Protocol("02_egocentric_stress_500", PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"),
    ]
    records = load_records(protocols)
    records_by_id = {record.scoped_id: record for record in records}
    scene_predictions = load_scene_predictions(SCENE_PREDICTIONS)

    candidates = [
        Candidate("yolo11s_widerface_1280", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 1280),
        Candidate("scrfd_10g_current_640", "scrfd", "detector", "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx", 640),
        Candidate("yolo8s_widerface_repo_640", "yolo", "detector", "data/models/face_detection_candidates/yolov8s_widerface.pt", 640),
        Candidate("yolo11n_pose_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11n-pose_widerface.pt", 640),
        Candidate("yolo11s_widerface_640", "yolo", "detector", "data/models/face_detection_candidates/yolo11s_widerface.pt", 640),
    ]

    predictions_by_model: dict[str, list[ScoredBox]] = {}
    runtime_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"\n=== Running {candidate.name} ===", flush=True)
        predictions, runtime = run_candidate(candidate, records)
        predictions_by_model[candidate.name] = predictions
        runtime_rows.append(runtime)

    fusion_specs = [
        ("fusion_yolo11s1280_scrfd10g_agreement", ["yolo11s_widerface_1280", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
        ("fusion_yolo11s_scrfd10g_agreement", ["yolo11s_widerface_640", "scrfd_10g_current_640"], 0.50, 0.22, 0.08, 0.90),
    ]
    for name, sources, iou, min_score, bonus, penalty in fusion_specs:
        predictions_by_model[name] = nms_fusion(predictions_by_model, sources, name, iou, min_score, bonus, penalty)
        runtime_rows.append(
            {
                "candidate": name,
                "family": "fusion",
                "model_path": "+".join(sources),
                "image_size": "",
                "confidence": min_score,
                "runtime_seconds": "",
                "images": len(records),
                "detections": len(predictions_by_model[name]),
                "failures": 0,
                "failure_examples": "",
            }
        )

    source_names = [
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
        "yolo8s_widerface_repo_640",
        "yolo11n_pose_widerface_640",
    ]
    clusters = cluster_predictions(predictions_by_model, source_names, records_by_id)

    deployable_reranker_preds, deploy_thresholds = cross_validated_reranker(
        clusters,
        records,
        source_names,
        scene_predictions,
        use_oracle_conditions=False,
    )
    oracle_reranker_preds, oracle_thresholds = cross_validated_reranker(
        clusters,
        records,
        source_names,
        scene_predictions,
        use_oracle_conditions=True,
    )

    method_predictions = {
        "fixed_fusion_yolo11s1280_scrfd10g": predictions_by_model["fusion_yolo11s1280_scrfd10g_agreement"],
        "deployable_category_policy": category_policy_predictions(predictions_by_model, records, scene_predictions, use_oracle_conditions=False),
        "oracle_category_policy": category_policy_predictions(predictions_by_model, records, scene_predictions, use_oracle_conditions=True),
        "cv_box_reranker_predicted_conditions": deployable_reranker_preds,
        "cv_box_reranker_oracle_conditions": oracle_reranker_preds,
    }

    score_rows: list[dict[str, Any]] = []
    for method_name, predictions in method_predictions.items():
        score_rows.extend(score_model(method_name, predictions, records))

    write_csv(output_dir / "low_compute_policy_scores.csv", score_rows)
    write_csv(output_dir / "low_compute_policy_runtime.csv", runtime_rows)
    write_csv(output_dir / "low_compute_reranker_thresholds_predicted_conditions.csv", deploy_thresholds)
    write_csv(output_dir / "low_compute_reranker_thresholds_oracle_conditions.csv", oracle_thresholds)

    cluster_rows = [
        {
            "clusters": len(clusters),
            "positive_clusters": sum(cluster.label for cluster in clusters),
            "positive_rate": round(sum(cluster.label for cluster in clusters) / max(1, len(clusters)), 6),
            "source_detectors": "|".join(source_names),
            "runtime_total_seconds": round(perf_counter() - started, 3),
        }
    ]
    write_csv(output_dir / "low_compute_reranker_cluster_summary.csv", cluster_rows)
    write_markdown_summary(output_dir / "low_compute_policy_scores.md", score_rows, "Low-Compute Detector Policy Scores")

    lines = [
        "# Low-Compute Detector Policy Experiment",
        "",
        "This experiment tests low-compute refinements after detector hardening. It does not use RF-DETR, full-resolution slicing, or unavailable specialised models.",
        "",
        "Compared methods:",
        "",
        "- `fixed_fusion_yolo11s1280_scrfd10g`: current best detector-hardening fusion.",
        "- `deployable_category_policy`: category-specific detector selection using predicted Scene-Condition Router labels.",
        "- `oracle_category_policy`: upper-bound category selection using reviewed condition labels.",
        "- `cv_box_reranker_predicted_conditions`: five-fold image-level cross-validated box reranker using detector features and predicted condition labels.",
        "- `cv_box_reranker_oracle_conditions`: same reranker with reviewed condition labels as an upper bound.",
        "",
        "Use predicted-condition results for deployable claims. Use oracle-condition rows only as upper-bound analysis.",
    ]
    output_dir.joinpath("low_compute_policy_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote low-compute detector policy outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
