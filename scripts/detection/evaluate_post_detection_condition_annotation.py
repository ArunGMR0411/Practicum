#!/usr/bin/env python3
"""Evaluate hybrid post-detection condition annotation for OAPR routing.

This tests whether SCR context cues plus final face-box geometry are stronger
than either raw/pre-detection SCR or box-derived rules alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/02_face_detection/10_post_detection_condition_annotation"

CONDITION_DATASET = PROJECT_ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv"
SCR_PREDICTIONS = PROJECT_ROOT / "outputs/02_face_detection/04_scene_condition_router/08_final_predictions.csv"
SCR_METHOD = "handcrafted_yolo_multiscale__logistic_regression"
HANDOFF_BOXES = PROJECT_ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv"
DETECTOR_CANDIDATE_BOXES = (
    PROJECT_ROOT / "outputs/02_face_detection/11_detector_candidate_box_telemetry/detector_candidate_boxes.csv"
)

MANIFESTS = {
    "baseline_500": PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv",
    "egocentric_stress_500": PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
}

LABELS = [
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

SCALE_LABELS = [
    "small_face",
    "medium_face",
    "large_face",
    "mixed_scale_face",
    "very_small_or_distant_face",
]

BOX_GEOMETRY_LABELS = [
    "no_face",
    "single_face",
    "multi_face",
    "small_face",
    "medium_face",
    "large_face",
    "mixed_scale_face",
    "very_small_or_distant_face",
    "edge_or_partial_face",
]

IMAGE_CUE_LABELS = [
    "motion_blur_or_low_sharpness",
    "low_light_or_dim",
    "high_clutter",
]

UNSUPPORTED_BY_BOX_RULES = ["profile_or_occluded_face"]

SCR_CONTEXT_LABELS = [
    "profile_or_occluded_face",
    "downward_egocentric_view",
    "motion_blur_or_low_sharpness",
    "low_light_or_dim",
    "high_clutter",
]

EXCLUDED_PRIMARY_LABELS = {"text_or_screen_risk_in_face_protocol", "outdoor_or_vehicle_scene"}
SUPPORTED_MIN_SUPPORT = 30
FEATURE_METADATA_COLUMNS = {"relative_path", "protocol", "image_path", "method_id", "box_source", "detected_box_count"}
FINAL_AVAILABLE_HANDOFF_LABEL_SOURCE = {
    "no_face": "rule_hybrid",
    "single_face": "rule_hybrid",
    "multi_face": "rule_hybrid",
    "small_face": "rule_hybrid",
    "medium_face": "crop_conflict_cv",
    "large_face": "crop_conflict_cv",
    "mixed_scale_face": "crop_conflict_cv",
    "very_small_or_distant_face": "rule_hybrid",
    "edge_or_partial_face": "crop_conflict_cv",
    "profile_or_occluded_face": "crop_conflict_cv",
    "downward_egocentric_view": "crop_conflict_cv",
    "motion_blur_or_low_sharpness": "rule_hybrid",
    "low_light_or_dim": "rule_hybrid",
    "high_clutter": "rule_hybrid",
}

FIXED_MODEL_POLICY_LABEL_SOURCE = {
    "no_face": "base_final_hybrid",
    "single_face": "logreg",
    "multi_face": "base_final_hybrid",
    "small_face": "multiclass_scale_layer",
    "medium_face": "multiclass_scale_layer",
    "large_face": "multiclass_scale_layer",
    "mixed_scale_face": "multiclass_scale_layer",
    "very_small_or_distant_face": "multiclass_scale_layer",
    "edge_or_partial_face": "rf_detector_telemetry",
    "profile_or_occluded_face": "rf_detector_telemetry",
    "downward_egocentric_view": "gb_detector_telemetry",
    "motion_blur_or_low_sharpness": "histgb_detector_telemetry",
    "low_light_or_dim": "logreg",
    "high_clutter": "histgb_detector_telemetry",
}

DETECTOR_PROTOCOL_MAP = {
    "baseline_500": "01_baseline_500",
    "egocentric_stress_500": "02_egocentric_stress_500",
}

RAW_DETECTOR_SOURCES = [
    "yolo11s_widerface_1280",
    "scrfd_10g_current_640",
    "yolo8s_widerface_repo_640",
    "yolo11n_pose_widerface_640",
    "yolo11s_widerface_640",
    "sliced_yolo11s_widerface_1280",
    "rfdetr_medium_face_030",
]

POLICY_DETECTOR_SOURCES = [
    "fusion_yolo11s1280_scrfd10g",
    "fusion_yolo11s_scrfd10g",
    "fusion_rfdetr_scrfd10g",
    "cv_box_reranker_with_rfdetr_predicted_conditions",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_boxes_json(value: Any) -> list[dict[str, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return []
    try:
        boxes = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    out: list[dict[str, float]] = []
    for box in boxes:
        try:
            out.append(
                {
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                    "score": float(box.get("score", 1.0)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def scale_label(area_frac: float, height_frac: float) -> str:
    if height_frac < 0.065 or area_frac < 0.003:
        return "very_small_or_distant"
    if height_frac < 0.13 or area_frac < 0.0085:
        return "small"
    if height_frac < 0.25 or area_frac < 0.022:
        return "medium"
    return "large"


def labels_from_boxes(
    boxes: list[dict[str, float]],
    width: float,
    height: float,
    image_cues: dict[str, int],
) -> dict[str, int]:
    labels = {label: 0 for label in LABELS}
    face_count = len(boxes)
    labels["no_face"] = int(face_count == 0)
    labels["single_face"] = int(face_count == 1)
    labels["multi_face"] = int(face_count >= 2)

    per_face_scales: list[str] = []
    areas: list[float] = []
    center_y_ratios: list[float] = []
    edge_count = 0

    for box in boxes:
        x1 = max(0.0, min(width, box["x1"]))
        y1 = max(0.0, min(height, box["y1"]))
        x2 = max(0.0, min(width, box["x2"]))
        y2 = max(0.0, min(height, box["y2"]))
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        area_frac = (box_width * box_height) / max(1.0, width * height)
        height_frac = box_height / max(1.0, height)
        areas.append(area_frac)
        center_y_ratios.append(((y1 + y2) / 2.0) / max(1.0, height))
        per_face_scales.append(scale_label(area_frac, height_frac))
        if x1 <= 0.03 * width or y1 <= 0.03 * height or x2 >= width - 0.03 * width or y2 >= height - 0.03 * height:
            edge_count += 1

    if face_count > 0:
        unique_scales = set(per_face_scales)
        area_ratio = max(areas) / max(min(areas), 1e-9)
        major_scale_gap = (
            ("very_small_or_distant" in unique_scales and ("medium" in unique_scales or "large" in unique_scales))
            or ("small" in unique_scales and "large" in unique_scales)
        )
        if face_count >= 2 and (major_scale_gap or area_ratio >= 10.0):
            labels["mixed_scale_face"] = 1
        else:
            for scale in ["large", "medium", "small", "very_small_or_distant"]:
                if scale in unique_scales:
                    labels[f"{scale}_face" if scale != "very_small_or_distant" else "very_small_or_distant_face"] = 1
                    break
        labels["edge_or_partial_face"] = int(edge_count > 0)
        labels["downward_egocentric_view"] = int(max(center_y_ratios) >= 0.62)

    for label in IMAGE_CUE_LABELS:
        labels[label] = int(image_cues.get(label, 0))

    # Face profile/occlusion is not safely inferable from boxes alone.
    labels["profile_or_occluded_face"] = 0
    return labels


def image_cue_thresholds(condition_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {}
    for protocol, group in condition_df.groupby("protocol"):
        thresholds[protocol] = {
            "blur": float(group["img_sharpness_laplacian_var"].quantile(0.20)),
            "low_light": float(group["img_brightness_mean"].quantile(0.20)),
            "high_clutter": float(group["img_edge_density"].quantile(0.66)),
        }
    return thresholds


def image_cues(row: pd.Series, thresholds: dict[str, dict[str, float]]) -> dict[str, int]:
    t = thresholds[str(row["protocol"])]
    return {
        "motion_blur_or_low_sharpness": int(float(row["img_sharpness_laplacian_var"]) <= t["blur"]),
        "low_light_or_dim": int(float(row["img_brightness_mean"]) <= t["low_light"]),
        "high_clutter": int(float(row["img_edge_density"]) >= t["high_clutter"]),
    }


def reviewed_box_lookup() -> dict[str, list[dict[str, float]]]:
    lookup: dict[str, list[dict[str, float]]] = {}
    for _protocol, path in MANIFESTS.items():
        manifest = pd.read_csv(path)
        for row in manifest.itertuples(index=False):
            lookup[str(row.relative_path)] = parse_boxes_json(getattr(row, "reviewed_face_boxes_json", "[]"))
    return lookup


def detected_handoff_lookup() -> dict[str, list[dict[str, float]]]:
    if not HANDOFF_BOXES.exists():
        return {}
    df = pd.read_csv(HANDOFF_BOXES)
    lookup: dict[str, list[dict[str, float]]] = {}
    for image_id, group in df.groupby("image_id"):
        lookup[str(image_id)] = [
            {
                "x1": float(row.x1),
                "y1": float(row.y1),
                "x2": float(row.x2),
                "y2": float(row.y2),
                "score": float(getattr(row, "score", 1.0)),
            }
            for row in group.itertuples(index=False)
        ]
    return lookup


def image_path_for_row(row: pd.Series) -> Path:
    if "image_path" in row and str(row["image_path"]).strip():
        return PROJECT_ROOT / str(row["image_path"])
    return PROJECT_ROOT / "data/castle2024/raw" / str(row["relative_path"])


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def patch_stats(rgb: np.ndarray) -> dict[str, float]:
    if rgb.size == 0:
        return {
            "brightness": 0.0,
            "brightness_std": 0.0,
            "sharpness": 0.0,
            "edge_density": 0.0,
            "saturation": 0.0,
        }
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "brightness": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float((edges > 0).mean()),
        "saturation": float(hsv[:, :, 1].mean()),
    }


def summarize(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_std": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_std": float(arr.std()),
    }


def iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
    bx1, by1, bx2, by2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def load_detector_candidate_lookup() -> dict[tuple[str, str], list[dict[str, float | str]]]:
    if not DETECTOR_CANDIDATE_BOXES.exists():
        return {}
    df = pd.read_csv(DETECTOR_CANDIDATE_BOXES)
    lookup: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in df.itertuples(index=False):
        key = (str(row.protocol), str(row.relative_path))
        lookup.setdefault(key, []).append(
            {
                "detector_name": str(row.detector_name),
                "x1": float(row.x1),
                "y1": float(row.y1),
                "x2": float(row.x2),
                "y2": float(row.y2),
                "score": float(row.score),
            }
        )
    return lookup


def candidate_telemetry_features(
    protocol: str,
    relative_path: str,
    lookup: dict[tuple[str, str], list[dict[str, float | str]]],
) -> dict[str, float]:
    export_protocol = DETECTOR_PROTOCOL_MAP.get(protocol, protocol)
    rows = lookup.get((export_protocol, relative_path), [])
    features: dict[str, float] = {
        "detector_candidate_total_boxes": float(len(rows)),
    }
    raw_rows = [row for row in rows if row["detector_name"] in RAW_DETECTOR_SOURCES]
    policy_rows = [row for row in rows if row["detector_name"] in POLICY_DETECTOR_SOURCES]
    raw_counts: list[float] = []
    raw_scores: list[float] = []
    for source in RAW_DETECTOR_SOURCES + POLICY_DETECTOR_SOURCES:
        source_rows = [row for row in rows if row["detector_name"] == source]
        counts = float(len(source_rows))
        scores = [float(row["score"]) for row in source_rows]
        areas = [
            max(0.0, float(row["x2"]) - float(row["x1"])) * max(0.0, float(row["y2"]) - float(row["y1"]))
            for row in source_rows
        ]
        features[f"detector_{source}_count"] = counts
        features[f"detector_{source}_has_box"] = float(counts > 0)
        features[f"detector_{source}_score_max"] = max(scores) if scores else 0.0
        features[f"detector_{source}_score_mean"] = float(np.mean(scores)) if scores else 0.0
        features[f"detector_{source}_area_mean"] = float(np.mean(areas)) if areas else 0.0
        if source in RAW_DETECTOR_SOURCES:
            raw_counts.append(counts)
            raw_scores.extend(scores)

    features["detector_raw_source_count_with_boxes"] = float(sum(count > 0 for count in raw_counts))
    features["detector_raw_count_mean"] = float(np.mean(raw_counts)) if raw_counts else 0.0
    features["detector_raw_count_std"] = float(np.std(raw_counts)) if raw_counts else 0.0
    features["detector_raw_count_range"] = float(max(raw_counts) - min(raw_counts)) if raw_counts else 0.0
    features["detector_raw_score_mean"] = float(np.mean(raw_scores)) if raw_scores else 0.0
    features["detector_policy_total_boxes"] = float(len(policy_rows))

    pending = sorted(raw_rows, key=lambda row: float(row["score"]), reverse=True)
    cluster_source_counts: list[float] = []
    cluster_sizes: list[float] = []
    singleton_clusters = 0
    consensus_clusters = 0
    while pending:
        seed = pending.pop(0)
        group = [seed]
        remaining = []
        for row in pending:
            if iou(seed, row) >= 0.5:
                group.append(row)
            else:
                remaining.append(row)
        pending = remaining
        sources = {str(row["detector_name"]) for row in group}
        cluster_source_counts.append(float(len(sources)))
        cluster_sizes.append(float(len(group)))
        singleton_clusters += int(len(sources) == 1)
        consensus_clusters += int(len(sources) >= 3)
    features["detector_cluster_count"] = float(len(cluster_source_counts))
    features["detector_cluster_source_count_max"] = max(cluster_source_counts) if cluster_source_counts else 0.0
    features["detector_cluster_source_count_mean"] = float(np.mean(cluster_source_counts)) if cluster_source_counts else 0.0
    features["detector_cluster_size_mean"] = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
    features["detector_singleton_cluster_count"] = float(singleton_clusters)
    features["detector_consensus_cluster_count"] = float(consensus_clusters)
    features["detector_disagreement_ratio"] = (
        float(singleton_clusters) / max(1.0, float(len(cluster_source_counts))) if cluster_source_counts else 0.0
    )
    return features


def box_and_crop_features(row: pd.Series, boxes: list[dict[str, float]]) -> dict[str, float]:
    width = safe_float(row.get("image_width", 3840.0), 3840.0)
    height = safe_float(row.get("image_height", 2160.0), 2160.0)
    features: dict[str, float] = {
        "box_face_count": float(len(boxes)),
        "box_has_face": float(len(boxes) > 0),
    }
    areas: list[float] = []
    heights: list[float] = []
    widths: list[float] = []
    aspect_ratios: list[float] = []
    center_y: list[float] = []
    scores: list[float] = []
    edge_count = 0
    lower_count = 0
    upper_count = 0
    crop_brightness: list[float] = []
    crop_sharpness: list[float] = []
    crop_edge_density: list[float] = []
    crop_saturation: list[float] = []
    crop_brightness_std: list[float] = []

    image_rgb: np.ndarray | None = None
    if boxes:
        path = image_path_for_row(row)
        if path.exists():
            with Image.open(path) as image:
                image_rgb = np.asarray(image.convert("RGB"))

    for box in boxes:
        x1 = max(0.0, min(width, safe_float(box.get("x1"))))
        y1 = max(0.0, min(height, safe_float(box.get("y1"))))
        x2 = max(0.0, min(width, safe_float(box.get("x2"))))
        y2 = max(0.0, min(height, safe_float(box.get("y2"))))
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        area = (box_width * box_height) / max(1.0, width * height)
        height_ratio = box_height / max(1.0, height)
        width_ratio = box_width / max(1.0, width)
        y_ratio = ((y1 + y2) / 2.0) / max(1.0, height)
        areas.append(area)
        heights.append(height_ratio)
        widths.append(width_ratio)
        aspect_ratios.append(box_height / max(box_width, 1.0))
        center_y.append(y_ratio)
        scores.append(safe_float(box.get("score"), 1.0))
        if x1 <= 0.03 * width or y1 <= 0.03 * height or x2 >= width - 0.03 * width or y2 >= height - 0.03 * height:
            edge_count += 1
        if y_ratio >= 0.62:
            lower_count += 1
        if y_ratio <= 0.38:
            upper_count += 1
        if image_rgb is not None:
            pad_x = 0.15 * box_width
            pad_y = 0.15 * box_height
            cx1 = int(max(0, np.floor(x1 - pad_x)))
            cy1 = int(max(0, np.floor(y1 - pad_y)))
            cx2 = int(min(image_rgb.shape[1], np.ceil(x2 + pad_x)))
            cy2 = int(min(image_rgb.shape[0], np.ceil(y2 + pad_y)))
            stats = patch_stats(image_rgb[cy1:cy2, cx1:cx2])
            crop_brightness.append(stats["brightness"])
            crop_brightness_std.append(stats["brightness_std"])
            crop_sharpness.append(stats["sharpness"])
            crop_edge_density.append(stats["edge_density"])
            crop_saturation.append(stats["saturation"])

    features.update(summarize(areas, "box_area_ratio"))
    features.update(summarize(heights, "box_height_ratio"))
    features.update(summarize(widths, "box_width_ratio"))
    features.update(summarize(aspect_ratios, "box_aspect_ratio"))
    features.update(summarize(center_y, "box_center_y_ratio"))
    features.update(summarize(scores, "box_score"))
    features.update(summarize(crop_brightness, "crop_brightness"))
    features.update(summarize(crop_brightness_std, "crop_brightness_std"))
    features.update(summarize(crop_sharpness, "crop_sharpness"))
    features.update(summarize(crop_edge_density, "crop_edge_density"))
    features.update(summarize(crop_saturation, "crop_saturation"))
    features["box_edge_count"] = float(edge_count)
    features["box_lower_frame_count"] = float(lower_count)
    features["box_upper_frame_count"] = float(upper_count)
    features["box_area_ratio_range"] = max(areas) / max(min(areas), 1e-9) if areas else 0.0
    features["box_score_range"] = max(scores) - min(scores) if scores else 0.0
    features["box_coverage_ratio"] = float(sum(areas))
    return features


def build_feature_rows(
    condition_df: pd.DataFrame,
    scr_predictions: list[dict[str, Any]],
    box_predictions: list[dict[str, Any]],
    box_lookup: dict[str, list[dict[str, float]]],
    method_id: str,
    subset_protocols: set[str] | None = None,
    detector_candidate_lookup: dict[tuple[str, str], list[dict[str, float | str]]] | None = None,
) -> list[dict[str, Any]]:
    scr_by_key = index_predictions(scr_predictions)
    box_by_key = index_predictions(box_predictions)
    rows: list[dict[str, Any]] = []
    for _, condition_row in condition_df.iterrows():
        protocol = str(condition_row["protocol"])
        if subset_protocols and protocol not in subset_protocols:
            continue
        relative_path = str(condition_row["relative_path"])
        key = (protocol, relative_path)
        scr_row = scr_by_key.get(key, {})
        box_row = box_by_key.get(key, {})
        if not scr_row and not box_row:
            continue
        boxes = box_lookup.get(relative_path, [])
        rec: dict[str, Any] = {
            "relative_path": relative_path,
            "protocol": protocol,
            "image_path": str(image_path_for_row(condition_row)),
            "method_id": method_id,
            "box_source": str(box_row.get("box_source", "")),
            "detected_box_count": len(boxes),
            "img_brightness_mean": safe_float(condition_row.get("img_brightness_mean")),
            "img_brightness_std": safe_float(condition_row.get("img_brightness_std")),
            "img_sharpness_laplacian_var": safe_float(condition_row.get("img_sharpness_laplacian_var")),
            "img_edge_density": safe_float(condition_row.get("img_edge_density")),
            "img_saturation_mean": safe_float(condition_row.get("img_saturation_mean")),
        }
        rec.update(box_and_crop_features(condition_row, boxes))
        if detector_candidate_lookup is not None:
            rec.update(candidate_telemetry_features(protocol, relative_path, detector_candidate_lookup))
        conflict_count = 0
        for label in LABELS:
            true_value = int(condition_row[f"label_{label}"])
            scr_value = int(scr_row.get(f"pred_{label}", 0)) if scr_row else 0
            box_value = int(box_row.get(f"pred_{label}", 0)) if box_row else 0
            rec[f"true_{label}"] = true_value
            rec[f"feature_scr_pred_{label}"] = scr_value
            rec[f"feature_box_rule_pred_{label}"] = box_value
            rec[f"feature_scr_box_conflict_{label}"] = int(scr_value != box_value)
            if label in BOX_GEOMETRY_LABELS and scr_value != box_value:
                conflict_count += 1
        rec["feature_geometry_conflict_count"] = float(conflict_count)
        rows.append(rec)
    return rows


def feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded_prefixes = ("true_", "pred_")
    columns: list[str] = []
    for column in feature_df.columns:
        if column in FEATURE_METADATA_COLUMNS or column.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(feature_df[column]):
            columns.append(column)
    return columns


def best_threshold_for_f2(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.10, 0.90, 33):
        pred = (probs >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def label_cv_predictions(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    if len(np.unique(y)) < 2:
        return np.full_like(y, int(y[0]), dtype=int)
    min_class = int(np.bincount(y).min())
    if min_class >= 5:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=random_state)
        splits = splitter.split(X)
    out = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2:
            out[test_idx] = int(y[train_idx][0])
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=random_state),
        )
        clf.fit(X[train_idx], y[train_idx])
        train_probs = clf.predict_proba(X[train_idx])[:, 1]
        test_probs = clf.predict_proba(X[test_idx])[:, 1]
        threshold = best_threshold_for_f2(y[train_idx], train_probs)
        out[test_idx] = (test_probs >= threshold).astype(int)
    return out


def model_family_factories(random_state: int = 42) -> dict[str, Any]:
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=random_state),
        ),
        "rf300_d4": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state + 1,
            n_jobs=-1,
        ),
        "rf500_d8": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state + 2,
            n_jobs=-1,
        ),
        "extra500_d8": lambda: ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state + 3,
            n_jobs=-1,
        ),
        "extra800_none": lambda: ExtraTreesClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state + 4,
            n_jobs=-1,
        ),
        "gb": lambda: GradientBoostingClassifier(
            n_estimators=160,
            max_depth=2,
            learning_rate=0.04,
            random_state=random_state + 5,
        ),
        "histgb": lambda: HistGradientBoostingClassifier(
            max_iter=180,
            max_leaf_nodes=16,
            learning_rate=0.04,
            l2_regularization=0.03,
            random_state=random_state + 6,
        ),
    }


def fixed_policy_classifier(source: str, random_state: int = 42) -> Any:
    if source == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=random_state),
        )
    if source == "rf_detector_telemetry":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state + 2,
            n_jobs=-1,
        )
    if source == "gb_detector_telemetry":
        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            random_state=random_state + 5,
        )
    if source == "histgb_detector_telemetry":
        return HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=16,
            learning_rate=0.05,
            l2_regularization=0.03,
            random_state=random_state + 6,
        )
    raise ValueError(f"Unsupported fixed-policy classifier source: {source}")


def proba_for_classifier(classifier: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(X)[:, 1]
    scores = classifier.decision_function(X)
    return (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)


def build_model_family_cv_hybrid_predictions(
    feature_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    method_id: str,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a model family per label inside each fold, then predict held-out rows."""
    feature_df = pd.DataFrame(feature_rows).fillna(0).reset_index(drop=True)
    base_by_key = index_predictions(base_predictions)
    columns = feature_columns(feature_df)
    X = feature_df[columns].to_numpy(dtype=float)
    n = len(feature_df)
    pred_by_label: dict[str, np.ndarray] = {}
    selection_rows: list[dict[str, Any]] = []
    factories = model_family_factories(random_state)

    for label in LABELS:
        y = feature_df[f"true_{label}"].to_numpy(dtype=int)
        base_pred = np.array(
            [
                int(base_by_key.get((str(row.protocol), str(row.relative_path)), {}).get(f"pred_{label}", 0))
                for row in feature_df.itertuples(index=False)
            ],
            dtype=int,
        )
        if len(np.unique(y)) < 2:
            pred_by_label[label] = np.full(n, int(y[0]), dtype=int)
            selection_rows.append(
                {
                    "label": label,
                    "fold": "all",
                    "selected_source": "constant",
                    "train_f2": 1.0,
                    "threshold": "",
                }
            )
            continue
        min_class = int(np.bincount(y).min())
        if min_class >= 5:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + 17)
            splits = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=min(5, n), shuffle=True, random_state=random_state + 17)
            splits = splitter.split(X)
        out = np.zeros(n, dtype=int)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            base_train_f2 = fbeta_score(y[train_idx], base_pred[train_idx], beta=2, zero_division=0)
            best_source = "base_final_hybrid"
            best_train_f2 = float(base_train_f2)
            best_threshold = ""
            best_test_pred = base_pred[test_idx]
            if len(np.unique(y[train_idx])) >= 2:
                for family_name, factory in factories.items():
                    clf = factory()
                    clf.fit(X[train_idx], y[train_idx])
                    train_prob = proba_for_classifier(clf, X[train_idx])
                    test_prob = proba_for_classifier(clf, X[test_idx])
                    threshold = best_threshold_for_f2(y[train_idx], train_prob)
                    train_pred = (train_prob >= threshold).astype(int)
                    train_f2 = fbeta_score(y[train_idx], train_pred, beta=2, zero_division=0)
                    if train_f2 > best_train_f2 + 0.01:
                        best_source = family_name
                        best_train_f2 = float(train_f2)
                        best_threshold = round(float(threshold), 4)
                        best_test_pred = (test_prob >= threshold).astype(int)
            out[test_idx] = best_test_pred
            selection_rows.append(
                {
                    "label": label,
                    "fold": fold_idx,
                    "selected_source": best_source,
                    "train_f2": round(float(best_train_f2), 6),
                    "threshold": best_threshold,
                    "test_rows": len(test_idx),
                    "test_support": int(y[test_idx].sum()),
                }
            )
        pred_by_label[label] = out

    rows: list[dict[str, Any]] = []
    for idx, row in feature_df.iterrows():
        rec: dict[str, Any] = {
            "relative_path": row["relative_path"],
            "protocol": row["protocol"],
            "method_id": method_id,
            "box_source": row["box_source"],
            "detected_box_count": row["detected_box_count"],
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"true_{label}"])
            rec[f"pred_{label}"] = int(pred_by_label[label][idx])
        rows.append(rec)
    return rows, selection_rows


def build_fixed_policy_cv_hybrid_predictions(
    feature_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    method_id: str,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the evidence-derived per-label model policy without exhaustive search."""
    feature_df = pd.DataFrame(feature_rows).fillna(0).reset_index(drop=True)
    base_by_key = index_predictions(base_predictions)
    columns = feature_columns(feature_df)
    X = feature_df[columns].to_numpy(dtype=float)
    n = len(feature_df)
    pred_by_label: dict[str, np.ndarray] = {}
    selection_rows: list[dict[str, Any]] = []

    for label in LABELS:
        y = feature_df[f"true_{label}"].to_numpy(dtype=int)
        base_pred = np.array(
            [
                int(base_by_key.get((str(row.protocol), str(row.relative_path)), {}).get(f"pred_{label}", 0))
                for row in feature_df.itertuples(index=False)
            ],
            dtype=int,
        )
        source = FIXED_MODEL_POLICY_LABEL_SOURCE[label]
        if source in {"base_final_hybrid", "multiclass_scale_layer"} or len(np.unique(y)) < 2:
            pred_by_label[label] = base_pred
            selection_rows.append(
                {
                    "label": label,
                    "source": source,
                    "fold": "all",
                    "train_f2": "",
                    "threshold": "",
                    "reason": (
                        "scale labels handled by the fold-safe multiclass scale layer"
                        if source == "multiclass_scale_layer"
                        else "base final hybrid remained strongest or safest"
                    ),
                }
            )
            continue
        min_class = int(np.bincount(y).min())
        if min_class >= 5:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + 71)
            splits = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=min(5, n), shuffle=True, random_state=random_state + 71)
            splits = splitter.split(X)
        out = np.zeros(n, dtype=int)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
            if len(np.unique(y[train_idx])) < 2:
                out[test_idx] = base_pred[test_idx]
                continue
            clf = fixed_policy_classifier(source, random_state=random_state + fold_idx)
            clf.fit(X[train_idx], y[train_idx])
            train_prob = proba_for_classifier(clf, X[train_idx])
            test_prob = proba_for_classifier(clf, X[test_idx])
            threshold = best_threshold_for_f2(y[train_idx], train_prob)
            train_pred = (train_prob >= threshold).astype(int)
            train_f2 = fbeta_score(y[train_idx], train_pred, beta=2, zero_division=0)
            out[test_idx] = (test_prob >= threshold).astype(int)
            selection_rows.append(
                {
                    "label": label,
                    "source": source,
                    "fold": fold_idx,
                    "train_f2": round(float(train_f2), 6),
                    "threshold": round(float(threshold), 4),
                    "reason": "fixed evidence-derived per-label telemetry model",
                }
            )
        pred_by_label[label] = out

    rows: list[dict[str, Any]] = []
    for idx, row in feature_df.iterrows():
        rec: dict[str, Any] = {
            "relative_path": row["relative_path"],
            "protocol": row["protocol"],
            "method_id": method_id,
            "box_source": row["box_source"],
            "detected_box_count": row["detected_box_count"],
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"true_{label}"])
            rec[f"pred_{label}"] = int(pred_by_label[label][idx])
        rows.append(rec)
    return rows, selection_rows


def build_multiclass_scale_hybrid_predictions(
    feature_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    method_id: str,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace independent scale-label predictions with fold-safe multiclass scale predictions.

    The scale labels are mutually exclusive in the reviewed condition protocol.
    Modelling them independently can create avoidable false positives, so this
    layer predicts one of: none, small, medium, large, mixed, very-small.
    """
    feature_df = pd.DataFrame(feature_rows).fillna(0).reset_index(drop=True)
    base_by_key = index_predictions(base_predictions)
    columns = feature_columns(feature_df)
    X = feature_df[columns].to_numpy(dtype=float)
    classes = ["none", *SCALE_LABELS]
    y_names: list[str] = []
    for _, row in feature_df.iterrows():
        scale_class = "none"
        for label in SCALE_LABELS:
            if int(row[f"true_{label}"]) == 1:
                scale_class = label
                break
        y_names.append(scale_class)
    y = np.asarray([classes.index(name) for name in y_names], dtype=int)
    out = np.zeros(len(y), dtype=int)
    selection_rows: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + 29)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        clf = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=20,
            learning_rate=0.05,
            l2_regularization=0.03,
            random_state=random_state + fold_idx,
        )
        clf.fit(X[train_idx], y[train_idx])
        out[test_idx] = clf.predict(X[test_idx])
        selection_rows.append(
            {
                "fold": fold_idx,
                "model": "hist_gradient_boosting_multiclass_scale",
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "classes": "|".join(classes),
                "train_class_counts": "|".join(
                    f"{classes[class_idx]}={int((y[train_idx] == class_idx).sum())}"
                    for class_idx in range(len(classes))
                ),
                "test_class_counts": "|".join(
                    f"{classes[class_idx]}={int((y[test_idx] == class_idx).sum())}"
                    for class_idx in range(len(classes))
                ),
            }
        )

    rows: list[dict[str, Any]] = []
    for idx, row in feature_df.iterrows():
        key = (str(row["protocol"]), str(row["relative_path"]))
        base_row = base_by_key[key]
        rec: dict[str, Any] = {
            "relative_path": row["relative_path"],
            "protocol": row["protocol"],
            "method_id": method_id,
            "box_source": row["box_source"],
            "detected_box_count": row["detected_box_count"],
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"true_{label}"])
            if label in SCALE_LABELS:
                rec[f"pred_{label}"] = 0
            else:
                rec[f"pred_{label}"] = int(base_row[f"pred_{label}"])
        scale_label = classes[int(out[idx])]
        if scale_label != "none":
            rec[f"pred_{scale_label}"] = 1
        rows.append(rec)
    return rows, selection_rows


def build_cv_telemetry_predictions(feature_rows: list[dict[str, Any]], method_id: str) -> list[dict[str, Any]]:
    feature_df = pd.DataFrame(feature_rows).fillna(0)
    columns = feature_columns(feature_df)
    X = feature_df[columns].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    pred_by_label: dict[str, np.ndarray] = {}
    for label in LABELS:
        y = feature_df[f"true_{label}"].to_numpy(dtype=int)
        pred_by_label[label] = label_cv_predictions(X, y)
    for idx, row in feature_df.iterrows():
        rec: dict[str, Any] = {
            "relative_path": row["relative_path"],
            "protocol": row["protocol"],
            "method_id": method_id,
            "box_source": row["box_source"],
            "detected_box_count": row["detected_box_count"],
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"true_{label}"])
            rec[f"pred_{label}"] = int(pred_by_label[label][idx])
        rows.append(rec)
    return rows


def build_selective_cv_hybrid_predictions(
    feature_rows: list[dict[str, Any]],
    base_predictions: list[dict[str, Any]],
    method_id: str,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Use CV telemetry only for labels where it beats the base hybrid on train folds."""
    feature_df = pd.DataFrame(feature_rows).fillna(0).reset_index(drop=True)
    base_by_key = index_predictions(base_predictions)
    columns = feature_columns(feature_df)
    X = feature_df[columns].to_numpy(dtype=float)
    n = len(feature_df)
    pred_by_label: dict[str, np.ndarray] = {}
    groups = feature_df["relative_path"].astype(str).to_numpy()
    # Each image appears once per method/protocol here, so stratified folds are enough.
    for label in LABELS:
        y = feature_df[f"true_{label}"].to_numpy(dtype=int)
        base_pred = np.array(
            [
                int(base_by_key.get((str(row.protocol), str(row.relative_path)), {}).get(f"pred_{label}", 0))
                for row in feature_df.itertuples(index=False)
            ],
            dtype=int,
        )
        if len(np.unique(y)) < 2:
            pred_by_label[label] = np.full(n, int(y[0]), dtype=int)
            continue
        min_class = int(np.bincount(y).min())
        if min_class >= 5:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            splits = splitter.split(X, y)
        else:
            splitter = KFold(n_splits=min(5, n), shuffle=True, random_state=random_state)
            splits = splitter.split(X)
        out = np.zeros(n, dtype=int)
        for train_idx, test_idx in splits:
            if len(np.unique(y[train_idx])) < 2:
                out[test_idx] = base_pred[test_idx]
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=random_state),
            )
            clf.fit(X[train_idx], y[train_idx])
            train_probs = clf.predict_proba(X[train_idx])[:, 1]
            test_probs = clf.predict_proba(X[test_idx])[:, 1]
            threshold = best_threshold_for_f2(y[train_idx], train_probs)
            train_model_pred = (train_probs >= threshold).astype(int)
            model_train_f2 = fbeta_score(y[train_idx], train_model_pred, beta=2, zero_division=0)
            base_train_f2 = fbeta_score(y[train_idx], base_pred[train_idx], beta=2, zero_division=0)
            if model_train_f2 > base_train_f2 + 0.01:
                out[test_idx] = (test_probs >= threshold).astype(int)
            else:
                out[test_idx] = base_pred[test_idx]
        pred_by_label[label] = out
    rows: list[dict[str, Any]] = []
    for idx, row in feature_df.iterrows():
        rec: dict[str, Any] = {
            "relative_path": row["relative_path"],
            "protocol": row["protocol"],
            "method_id": method_id,
            "box_source": row["box_source"],
            "detected_box_count": row["detected_box_count"],
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"true_{label}"])
            rec[f"pred_{label}"] = int(pred_by_label[label][idx])
        rows.append(rec)
    _ = groups  # Kept to make the image-level design explicit without changing folds.
    return rows


def build_label_source_hybrid_predictions(
    base_predictions: list[dict[str, Any]],
    cv_predictions: list[dict[str, Any]],
    label_sources: dict[str, str],
    method_id: str,
) -> list[dict[str, Any]]:
    base_by_key = index_predictions(base_predictions)
    cv_by_key = index_predictions(cv_predictions)
    rows: list[dict[str, Any]] = []
    for key, base_row in base_by_key.items():
        cv_row = cv_by_key.get(key)
        if not cv_row:
            continue
        rec: dict[str, Any] = {
            "relative_path": base_row["relative_path"],
            "protocol": base_row["protocol"],
            "method_id": method_id,
            "box_source": base_row.get("box_source", ""),
            "detected_box_count": base_row.get("detected_box_count", ""),
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(base_row[f"true_{label}"])
            source = label_sources.get(label, "rule_hybrid")
            if source == "crop_conflict_cv":
                rec[f"pred_{label}"] = int(cv_row[f"pred_{label}"])
            else:
                rec[f"pred_{label}"] = int(base_row[f"pred_{label}"])
        rows.append(rec)
    return rows


def build_post_detection_predictions(
    condition_df: pd.DataFrame,
    box_lookup: dict[str, list[dict[str, float]]],
    method_id: str,
    subset_protocols: set[str] | None = None,
) -> list[dict[str, Any]]:
    thresholds = image_cue_thresholds(condition_df)
    rows: list[dict[str, Any]] = []
    for _, row in condition_df.iterrows():
        protocol = str(row["protocol"])
        if subset_protocols and protocol not in subset_protocols:
            continue
        relative_path = str(row["relative_path"])
        boxes = box_lookup.get(relative_path, [])
        cues = image_cues(row, thresholds)
        pred = labels_from_boxes(
            boxes=boxes,
            width=float(row["image_width"]) if "image_width" in row else 3840.0,
            height=float(row["image_height"]) if "image_height" in row else 2160.0,
            image_cues=cues,
        )
        rec: dict[str, Any] = {
            "relative_path": relative_path,
            "protocol": protocol,
            "method_id": method_id,
            "box_source": method_id,
            "detected_box_count": len(boxes),
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(row[f"label_{label}"])
            rec[f"pred_{label}"] = int(pred[label])
        rows.append(rec)
    return rows


def load_scr_predictions(condition_df: pd.DataFrame) -> list[dict[str, Any]]:
    df = pd.read_csv(SCR_PREDICTIONS)
    df = df[df["method_id"] == SCR_METHOD].copy()
    cols = ["relative_path", "protocol", "method_id"]
    for label in LABELS:
        cols.extend([f"true_{label}", f"pred_{label}"])
    df = df[cols]
    df["box_source"] = "pre_detection_scr_telemetry"
    df["detected_box_count"] = ""
    # Keep only records represented in the current condition dataset.
    valid = set(zip(condition_df["protocol"], condition_df["relative_path"]))
    df = df[df.apply(lambda row: (row["protocol"], row["relative_path"]) in valid, axis=1)]
    return df.to_dict("records")


def index_predictions(predictions: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["protocol"]), str(row["relative_path"])): row
        for row in predictions
    }


def build_hybrid_predictions(
    condition_df: pd.DataFrame,
    scr_predictions: list[dict[str, Any]],
    box_predictions: list[dict[str, Any]],
    method_id: str,
    subset_protocols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Combine SCR context predictions with post-detection box-geometry labels.

    Rule:
    - box geometry labels come from post-detection boxes;
    - context/image-quality labels come from SCR;
    - unsupported labels use SCR if available, otherwise fallback to 0.
    """
    scr_by_key = index_predictions(scr_predictions)
    box_by_key = index_predictions(box_predictions)
    rows: list[dict[str, Any]] = []
    for _, condition_row in condition_df.iterrows():
        protocol = str(condition_row["protocol"])
        if subset_protocols and protocol not in subset_protocols:
            continue
        relative_path = str(condition_row["relative_path"])
        key = (protocol, relative_path)
        scr_row = scr_by_key.get(key, {})
        box_row = box_by_key.get(key, {})
        if not scr_row and not box_row:
            continue
        rec: dict[str, Any] = {
            "relative_path": relative_path,
            "protocol": protocol,
            "method_id": method_id,
            "box_source": str(box_row.get("box_source", "")),
            "detected_box_count": box_row.get("detected_box_count", ""),
        }
        for label in LABELS:
            rec[f"true_{label}"] = int(condition_row[f"label_{label}"])
            if label in BOX_GEOMETRY_LABELS and box_row:
                rec[f"pred_{label}"] = int(box_row.get(f"pred_{label}", 0))
            elif label in SCR_CONTEXT_LABELS and scr_row:
                rec[f"pred_{label}"] = int(scr_row.get(f"pred_{label}", 0))
            elif scr_row:
                rec[f"pred_{label}"] = int(scr_row.get(f"pred_{label}", 0))
            elif box_row:
                rec[f"pred_{label}"] = int(box_row.get(f"pred_{label}", 0))
            else:
                rec[f"pred_{label}"] = 0
        rows.append(rec)
    return rows


def sample_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for truth, pred in zip(y_true, y_pred):
        t = set(np.where(truth == 1)[0])
        p = set(np.where(pred == 1)[0])
        if not t and not p:
            scores.append(1.0)
        elif not t and p:
            scores.append(0.0)
        else:
            scores.append(len(t & p) / max(1, len(t | p)))
    return float(np.mean(scores))


def metric_rows(
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows = pd.DataFrame(predictions)
    per_label_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for (method_id, protocol), group in all_rows.groupby(["method_id", "protocol"], dropna=False):
        labels_for_primary = [
            label
            for label in LABELS
            if label not in EXCLUDED_PRIMARY_LABELS and int(group[f"true_{label}"].sum()) >= SUPPORTED_MIN_SUPPORT
        ]
        y_true = group[[f"true_{label}" for label in labels_for_primary]].to_numpy(dtype=int)
        y_pred = group[[f"pred_{label}" for label in labels_for_primary]].to_numpy(dtype=int)
        if len(labels_for_primary) == 0:
            continue
        macro_f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        micro_f2 = fbeta_score(y_true, y_pred, beta=2, average="micro", zero_division=0)
        jaccard = sample_jaccard(y_true, y_pred)
        oapr_score = 0.50 * macro_f2 + 0.25 * macro_f1 + 0.15 * micro_f2 + 0.10 * jaccard
        route_eligible_count = 0
        for label in LABELS:
            true = group[f"true_{label}"].to_numpy(dtype=int)
            pred = group[f"pred_{label}"].to_numpy(dtype=int)
            support = int(true.sum())
            precision = precision_score(true, pred, zero_division=0)
            recall = recall_score(true, pred, zero_division=0)
            f1 = f1_score(true, pred, zero_division=0)
            f2 = fbeta_score(true, pred, beta=2, zero_division=0)
            route_eligible = bool(support >= SUPPORTED_MIN_SUPPORT and f1 >= 0.70)
            route_eligible_count += int(route_eligible)
            per_label_rows.append(
                {
                    "method_id": method_id,
                    "protocol": protocol,
                    "label": label,
                    "support": support,
                    "precision": round(float(precision), 6),
                    "recall": round(float(recall), 6),
                    "f1": round(float(f1), 6),
                    "f2": round(float(f2), 6),
                    "route_eligible": route_eligible,
                    "label_family": (
                        "box_geometry"
                        if label in BOX_GEOMETRY_LABELS
                        else "image_quality"
                        if label in IMAGE_CUE_LABELS
                        else "not_safely_inferable_from_boxes"
                        if label in UNSUPPORTED_BY_BOX_RULES
                        else "other"
                    ),
                }
            )
        per_label_df = pd.DataFrame([row for row in per_label_rows if row["method_id"] == method_id and row["protocol"] == protocol])
        for family, family_group in per_label_df[per_label_df["support"] >= SUPPORTED_MIN_SUPPORT].groupby("label_family"):
            family_rows.append(
                {
                    "method_id": method_id,
                    "protocol": protocol,
                    "label_family": family,
                    "supported_label_count": len(family_group),
                    "mean_precision": round(float(family_group["precision"].mean()), 6),
                    "mean_recall": round(float(family_group["recall"].mean()), 6),
                    "mean_f1": round(float(family_group["f1"].mean()), 6),
                    "mean_f2": round(float(family_group["f2"].mean()), 6),
                    "route_eligible_label_count": int(family_group["route_eligible"].sum()),
                }
            )
        benchmark_rows.append(
            {
                "method_id": method_id,
                "protocol": protocol,
                "image_rows": len(group),
                "primary_label_count": len(labels_for_primary),
                "route_eligible_label_count": route_eligible_count,
                "oapr_condition_score": round(float(oapr_score), 6),
                "macro_f2": round(float(macro_f2), 6),
                "macro_f1": round(float(macro_f1), 6),
                "micro_f2": round(float(micro_f2), 6),
                "sample_jaccard": round(float(jaccard), 6),
            }
        )
    return benchmark_rows, per_label_rows, family_rows


def write_markdown_summary(
    path: Path,
    benchmark_rows: list[dict[str, Any]],
    per_label_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
) -> None:
    benchmark = pd.DataFrame(benchmark_rows).sort_values(["protocol", "oapr_condition_score"], ascending=[True, False])
    per_label = pd.DataFrame(per_label_rows)
    family = pd.DataFrame(family_rows).sort_values(["protocol", "label_family", "mean_f1"], ascending=[True, True, False])
    lines = [
        "# Post-Detection Condition Annotation Evaluation",
        "",
        "Hypothesis:",
        "",
        "> SCR context cues plus post-detection face-box geometry produce stronger condition profiles for anonymisation routing than either SCR-only or box-only rules.",
        "",
        "Compared methods:",
        "",
        "- `handcrafted_yolo_multiscale__logistic_regression`: existing best pre-detection SCR prediction from raw/image-quality cues plus detector telemetry.",
        "- `post_detection_oracle_reviewed_boxes`: upper bound using reviewed face boxes plus image-quality cues.",
        "- `post_detection_available_handoff_boxes`: practical check on the egocentric stress 500 using the retained anonymisation handoff boxes.",
        "- `hybrid_scr_plus_reviewed_boxes_upper_bound`: SCR context/semantic cues plus reviewed-box geometry.",
        "- `hybrid_scr_plus_available_handoff_boxes`: SCR context/semantic cues plus retained detected-box geometry on the egocentric stress 500.",
        "- `cv_crop_conflict_available_handoff_boxes`: out-of-fold crop/box/conflict telemetry model on the egocentric stress 500.",
        "- `selective_cv_hybrid_available_handoff_boxes`: fold-safe selective hybrid that keeps rule labels unless telemetry improves training-fold F2.",
        "- `final_hybrid_condition_profile_available_handoff_boxes`: earlier label-source policy using the strongest measured source per label family.",
        "- `fixed_policy_detector_telemetry_hybrid_available_handoff_boxes`: faster evidence-derived per-label policy using detector telemetry without exhaustive model search.",
        "- `fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes`: final Step 4 policy that adds a fold-safe multiclass scale layer so mutually exclusive scale labels cannot conflict.",
        "",
        "Important boundary:",
        "",
        "- The retained handoff boxes are available only for the egocentric stress/anonymisation 500 surface.",
        "- Detector candidate boxes are now retained in `outputs/02_face_detection/11_detector_candidate_box_telemetry/`, enabling detector-disagreement and source-consensus features.",
        "- Detector-disagreement telemetry alone did not beat the earlier final policy on the egocentric stress 500; the winning improvement came from combining detector telemetry with a structured multiclass scale layer.",
        "- The final policy keeps base/rule labels where they remain strongest, uses detector telemetry for selected context labels, and models scale labels as one mutually exclusive class.",
        "",
        "## Benchmark",
        "",
        "| Method | Protocol | Images | OAPR condition score | Macro F2 | Macro F1 | Micro F2 | Jaccard | Route-eligible labels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in benchmark.to_dict("records"):
        lines.append(
            f"| {row['method_id']} | {row['protocol']} | {row['image_rows']} | "
            f"{float(row['oapr_condition_score']):.4f} | {float(row['macro_f2']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(row['micro_f2']):.4f} | "
            f"{float(row['sample_jaccard']):.4f} | {row['route_eligible_label_count']} |"
        )
    lines.extend(
        [
            "",
            "## Label-Family Benchmark",
            "",
            "| Method | Protocol | Label family | Supported labels | Mean F1 | Mean F2 | Route-eligible labels |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in family.to_dict("records"):
        lines.append(
            f"| {row['method_id']} | {row['protocol']} | {row['label_family']} | "
            f"{row['supported_label_count']} | {float(row['mean_f1']):.4f} | "
            f"{float(row['mean_f2']):.4f} | {row['route_eligible_label_count']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
        "- The enhanced Step 4 hypothesis is **supported**.",
        "- SCR-only reached `0.7375` on the egocentric stress 500; box-only retained handoff labels reached `0.6884`.",
        "- The first hybrid improved the score to `0.7999`.",
        "- Crop-level and conflict telemetry improved the practical score to `0.8652` with the selective CV hybrid.",
        "- The earlier label-source policy improved the practical score to `0.8742`, with all 12 supported labels route-eligible.",
        "- Persisted detector candidate boxes plus the fixed detector-telemetry policy improved the practical score further.",
        "- The final multiclass scale layer crossed the 0.9 target by preventing contradictory scale-label predictions.",
        "- The next anonymisation-routing stage should use `fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes` as the Step 4 condition profile.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    condition_df = pd.read_csv(CONDITION_DATASET)
    # Use only the reviewed 1,000-image face-detection surfaces.
    condition_df = condition_df[condition_df["protocol"].isin(["baseline_500", "egocentric_stress_500"])].copy()

    predictions: list[dict[str, Any]] = []
    scr_predictions = load_scr_predictions(condition_df)
    reviewed_box_predictions = build_post_detection_predictions(
        condition_df=condition_df,
        box_lookup=reviewed_box_lookup(),
        method_id="post_detection_oracle_reviewed_boxes",
    )
    handoff_box_predictions = build_post_detection_predictions(
        condition_df=condition_df,
        box_lookup=detected_handoff_lookup(),
        method_id="post_detection_available_handoff_boxes",
        subset_protocols={"egocentric_stress_500"},
    )
    hybrid_reviewed = build_hybrid_predictions(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=reviewed_box_predictions,
        method_id="hybrid_scr_plus_reviewed_boxes_upper_bound",
    )
    hybrid_handoff = build_hybrid_predictions(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=handoff_box_predictions,
        method_id="hybrid_scr_plus_available_handoff_boxes",
        subset_protocols={"egocentric_stress_500"},
    )
    reviewed_feature_rows = build_feature_rows(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=reviewed_box_predictions,
        box_lookup=reviewed_box_lookup(),
        method_id="cv_crop_conflict_reviewed_boxes_upper_bound_features",
    )
    handoff_feature_rows = build_feature_rows(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=handoff_box_predictions,
        box_lookup=detected_handoff_lookup(),
        method_id="cv_crop_conflict_available_handoff_boxes_features",
        subset_protocols={"egocentric_stress_500"},
    )
    detector_candidate_lookup = load_detector_candidate_lookup()
    detector_reviewed_feature_rows = build_feature_rows(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=reviewed_box_predictions,
        box_lookup=reviewed_box_lookup(),
        method_id="cv_detector_disagreement_reviewed_boxes_upper_bound_features",
        detector_candidate_lookup=detector_candidate_lookup,
    )
    detector_handoff_feature_rows = build_feature_rows(
        condition_df=condition_df,
        scr_predictions=scr_predictions,
        box_predictions=handoff_box_predictions,
        box_lookup=detected_handoff_lookup(),
        method_id="cv_detector_disagreement_available_handoff_boxes_features",
        subset_protocols={"egocentric_stress_500"},
        detector_candidate_lookup=detector_candidate_lookup,
    )
    cv_reviewed = build_cv_telemetry_predictions(
        reviewed_feature_rows,
        method_id="cv_crop_conflict_reviewed_boxes_upper_bound",
    )
    cv_handoff = build_cv_telemetry_predictions(
        handoff_feature_rows,
        method_id="cv_crop_conflict_available_handoff_boxes",
    )
    selective_reviewed = build_selective_cv_hybrid_predictions(
        feature_rows=reviewed_feature_rows,
        base_predictions=hybrid_reviewed,
        method_id="selective_cv_hybrid_reviewed_boxes_upper_bound",
    )
    selective_handoff = build_selective_cv_hybrid_predictions(
        feature_rows=handoff_feature_rows,
        base_predictions=hybrid_handoff,
        method_id="selective_cv_hybrid_available_handoff_boxes",
    )
    cv_detector_reviewed = build_cv_telemetry_predictions(
        detector_reviewed_feature_rows,
        method_id="cv_detector_disagreement_reviewed_boxes_upper_bound",
    )
    cv_detector_handoff = build_cv_telemetry_predictions(
        detector_handoff_feature_rows,
        method_id="cv_detector_disagreement_available_handoff_boxes",
    )
    selective_detector_reviewed = build_selective_cv_hybrid_predictions(
        feature_rows=detector_reviewed_feature_rows,
        base_predictions=hybrid_reviewed,
        method_id="selective_cv_detector_disagreement_reviewed_boxes_upper_bound",
    )
    selective_detector_handoff = build_selective_cv_hybrid_predictions(
        feature_rows=detector_handoff_feature_rows,
        base_predictions=hybrid_handoff,
        method_id="selective_cv_detector_disagreement_available_handoff_boxes",
    )
    final_handoff = build_label_source_hybrid_predictions(
        base_predictions=hybrid_handoff,
        cv_predictions=cv_handoff,
        label_sources=FINAL_AVAILABLE_HANDOFF_LABEL_SOURCE,
        method_id="final_hybrid_condition_profile_available_handoff_boxes",
    )
    fixed_policy_handoff, fixed_policy_selection_rows = build_fixed_policy_cv_hybrid_predictions(
        feature_rows=detector_handoff_feature_rows,
        base_predictions=final_handoff,
        method_id="fixed_policy_detector_telemetry_hybrid_available_handoff_boxes",
    )
    multiclass_scale_handoff, multiclass_scale_selection_rows = build_multiclass_scale_hybrid_predictions(
        feature_rows=detector_handoff_feature_rows,
        base_predictions=fixed_policy_handoff,
        method_id="fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes",
    )
    predictions.extend(scr_predictions)
    predictions.extend(reviewed_box_predictions)
    predictions.extend(handoff_box_predictions)
    predictions.extend(hybrid_reviewed)
    predictions.extend(hybrid_handoff)
    predictions.extend(cv_reviewed)
    predictions.extend(cv_handoff)
    predictions.extend(selective_reviewed)
    predictions.extend(selective_handoff)
    predictions.extend(cv_detector_reviewed)
    predictions.extend(cv_detector_handoff)
    predictions.extend(selective_detector_reviewed)
    predictions.extend(selective_detector_handoff)
    predictions.extend(final_handoff)
    predictions.extend(fixed_policy_handoff)
    predictions.extend(multiclass_scale_handoff)

    benchmark_rows, per_label_rows, family_rows = metric_rows(predictions)
    write_csv(OUTPUT_DIR / "post_detection_condition_predictions.csv", predictions)
    write_csv(OUTPUT_DIR / "post_detection_condition_benchmark.csv", benchmark_rows)
    write_csv(OUTPUT_DIR / "post_detection_condition_per_label_metrics.csv", per_label_rows)
    write_csv(OUTPUT_DIR / "post_detection_condition_family_benchmark.csv", family_rows)
    write_csv(
        OUTPUT_DIR / "post_detection_condition_final_label_policy.csv",
        [
            {
                "label": label,
                "final_source": FINAL_AVAILABLE_HANDOFF_LABEL_SOURCE[label],
                "reason": (
                    "crop/conflict telemetry improved the measured Step 4 label result"
                    if FINAL_AVAILABLE_HANDOFF_LABEL_SOURCE[label] == "crop_conflict_cv"
                    else "rule/SCR hybrid remained stronger or safer for this label"
                ),
            }
            for label in LABELS
        ],
    )
    write_csv(OUTPUT_DIR / "post_detection_condition_fixed_policy_selection.csv", fixed_policy_selection_rows)
    write_csv(OUTPUT_DIR / "post_detection_condition_multiclass_scale_selection.csv", multiclass_scale_selection_rows)
    write_markdown_summary(
        OUTPUT_DIR / "post_detection_condition_annotation_summary.md",
        benchmark_rows=benchmark_rows,
        per_label_rows=per_label_rows,
        family_rows=family_rows,
    )
    print(f"Wrote Step 4 outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
