#!/usr/bin/env python3
"""Build the reviewed condition dataset for the Scene-Condition Router."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data" / "castle2024" / "raw"
OUTPUT_ROOT = Path("/tmp/practicum_router_runs/condition_dataset")

MANIFESTS = {
    "baseline_500": ROOT
    / "data"
    / "castle2024"
    / "annotations"
    / "face_detection"
    / "01_baseline_500"
    / "manifest.csv",
    "egocentric_stress_500": ROOT
    / "data"
    / "castle2024"
    / "annotations"
    / "face_detection"
    / "02_egocentric_stress_500"
    / "manifest.csv",
}

LABEL_COLUMNS = {
    "no_face": ("face_count_category", "no_face"),
    "single_face": ("face_count_category", "single_face"),
    "multi_face": ("face_count_category", "multi_face"),
    "small_face": ("face_scale_category", "small"),
    "medium_face": ("face_scale_category", "medium"),
    "large_face": ("face_scale_category", "large"),
    "mixed_scale_face": ("face_scale_category", "mixed_scale"),
    "very_small_or_distant_face": ("face_scale_category", "very_small_or_distant"),
    "edge_or_partial_face": ("edge_partial_face", "yes"),
    "profile_or_occluded_face": ("profile_occluded_face", "yes"),
    "downward_egocentric_view": ("downward_egocentric_view", "yes"),
    "motion_blur_or_low_sharpness": ("blur_low_sharpness", "yes"),
    "low_light_or_dim": ("low_light_dim", "yes"),
    "high_clutter": ("clutter_level", "high"),
    "outdoor_or_vehicle_scene": ("outdoor_vehicle_scene", "yes"),
}


def parse_boxes(value: Any) -> list[dict[str, float]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    boxes = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            x1 = float(item["x1"])
            y1 = float(item["y1"])
            x2 = float(item["x2"])
            y2 = float(item["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def geometry_features(boxes: list[dict[str, float]], width: float, height: float) -> dict[str, float]:
    if width <= 0 or height <= 0 or not boxes:
        return {
            "geom_face_count": float(len(boxes)),
            "geom_max_area_ratio": 0.0,
            "geom_mean_area_ratio": 0.0,
            "geom_min_area_ratio": 0.0,
            "geom_face_coverage_ratio": 0.0,
            "geom_max_height_ratio": 0.0,
            "geom_mean_height_ratio": 0.0,
            "geom_edge_face_count": 0.0,
            "geom_lower_frame_face_count": 0.0,
            "geom_upper_frame_face_count": 0.0,
            "geom_center_face_count": 0.0,
        }
    areas, heights, edge, lower, upper, center = [], [], 0, 0, 0, 0
    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        areas.append((bw * bh) / (width * height))
        heights.append(bh / height)
        if x1 / width < 0.05 or y1 / height < 0.05 or x2 / width > 0.95 or y2 / height > 0.95:
            edge += 1
        if cy > 0.62:
            lower += 1
        if cy < 0.38:
            upper += 1
        if 0.33 <= cx <= 0.67 and 0.33 <= cy <= 0.67:
            center += 1
    return {
        "geom_face_count": float(len(boxes)),
        "geom_max_area_ratio": float(np.max(areas)),
        "geom_mean_area_ratio": float(np.mean(areas)),
        "geom_min_area_ratio": float(np.min(areas)),
        "geom_face_coverage_ratio": float(np.sum(areas)),
        "geom_max_height_ratio": float(np.max(heights)),
        "geom_mean_height_ratio": float(np.mean(heights)),
        "geom_edge_face_count": float(edge),
        "geom_lower_frame_face_count": float(lower),
        "geom_upper_frame_face_count": float(upper),
        "geom_center_face_count": float(center),
    }


def image_quality_features(image_path: Path) -> dict[str, float | str]:
    if not image_path.exists():
        return {
            "image_read_status": "missing",
            "img_brightness_mean": np.nan,
            "img_brightness_std": np.nan,
            "img_sharpness_laplacian_var": np.nan,
            "img_edge_density": np.nan,
            "img_saturation_mean": np.nan,
        }
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "image_read_status": "unreadable",
            "img_brightness_mean": np.nan,
            "img_brightness_std": np.nan,
            "img_sharpness_laplacian_var": np.nan,
            "img_edge_density": np.nan,
            "img_saturation_mean": np.nan,
        }
    # Downsample for deterministic lightweight feature extraction.
    image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    return {
        "image_read_status": "ok",
        "img_brightness_mean": float(gray.mean()),
        "img_brightness_std": float(gray.std()),
        "img_sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "img_edge_density": float((edges > 0).mean()),
        "img_saturation_mean": float(hsv[:, :, 1].mean()),
    }


def yes_no_label(row: pd.Series, source_col: str, expected: str) -> int:
    value = str(row.get(source_col, "")).strip().lower()
    return int(value == expected)


def source_group(relative_path: str) -> str:
    parts = Path(relative_path).parts
    day = parts[0] if len(parts) > 0 else "unknown_day"
    camera = parts[2] if len(parts) > 2 else "unknown_camera"
    stem = Path(relative_path).stem
    hour = stem.split("_", 1)[0] if "_" in stem else "unknown_hour"
    return f"{day}/{camera}/{hour}"


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    manifest_summary: list[dict[str, Any]] = []
    for protocol, manifest_path in MANIFESTS.items():
        df = pd.read_csv(manifest_path)
        manifest_summary.append(
            {
                "protocol": protocol,
                "manifest_path": str(manifest_path.relative_to(ROOT)),
                "rows": len(df),
                "unique_images": df["relative_path"].nunique(),
                "manual_reviewed_rows": int((df.get("manual_review_status", "") == "yes").sum()),
                "category_reviewed_rows": int((df.get("category_review_status", "") == "reviewed").sum()),
                "manual_face_boxes": int(pd.to_numeric(df.get("manual_face_count", 0), errors="coerce").fillna(0).sum()),
            }
        )
        for _, row in df.iterrows():
            relative_path = str(row["relative_path"])
            image_path = RAW_ROOT / relative_path
            width = float(row.get("image_width", 0) or 0)
            height = float(row.get("image_height", 0) or 0)
            boxes = parse_boxes(row.get("reviewed_face_boxes_json"))
            out: dict[str, Any] = {
                "protocol": protocol,
                "relative_path": relative_path,
                "image_id": row.get("image_id", relative_path),
                "image_path": str(image_path.relative_to(ROOT)),
                "source_group": source_group(relative_path),
                "day_id": row.get("day_id", ""),
                "camera_stream_id": row.get("camera_stream_id", ""),
                "timestamp_id": row.get("timestamp_id", ""),
                "manual_review_status": row.get("manual_review_status", ""),
                "category_review_status": row.get("category_review_status", ""),
                "manual_face_count": row.get("manual_face_count", len(boxes)),
                "face_count_category": row.get("face_count_category", ""),
                "face_scale_category": row.get("face_scale_category", ""),
                "condition_label": row.get("condition_label", ""),
            }
            for label_name, (col, expected) in LABEL_COLUMNS.items():
                out[f"label_{label_name}"] = yes_no_label(row, col, expected)
            text_screen_value = str(row.get("text_screen_risk", "")).strip().lower()
            out["label_text_or_screen_risk_in_face_protocol"] = int(
                text_screen_value not in {"", "no", "not_assessed_in_face_review", "nan"}
            )
            out.update(geometry_features(boxes, width, height))
            out.update(image_quality_features(image_path))
            rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(manifest_summary)


def write_markdown_summary(dataset: pd.DataFrame, manifests: pd.DataFrame, path: Path) -> None:
    label_cols = [c for c in dataset.columns if c.startswith("label_")]
    lines = [
        "# Scene-Condition Dataset Summary",
        "",
        "This dataset is generated from the two manually reviewed 500-image face protocols.",
        "",
        "## Protocol Summary",
        "",
        manifests.to_markdown(index=False),
        "",
        "## Label Support",
        "",
    ]
    support = []
    for col in label_cols:
        support.append({"label": col.removeprefix("label_"), "positive_count": int(dataset[col].sum())})
    support_df = pd.DataFrame(support).sort_values("label")
    lines.append(support_df.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Feature Boundary",
            "",
            "- Geometry features are derived from reviewed face boxes.",
            "- Image-quality features are deterministic OpenCV measurements.",
            "- Visual embeddings are extracted separately so DINOv3 or another frozen backbone can be swapped without changing labels.",
            "- Text/screen labels in the face protocol are not treated as full multimodal ground truth; the dedicated multimodal protocol remains separate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, manifests = build_dataset()
    dataset_path = args.output_dir / "condition_dataset.csv"
    summary_path = args.output_dir / "condition_dataset_summary.md"
    manifest_summary_path = args.output_dir / "condition_manifest_summary.csv"
    dataset.to_csv(dataset_path, index=False)
    manifests.to_csv(manifest_summary_path, index=False)
    write_markdown_summary(dataset, manifests, summary_path)
    print(f"wrote {dataset_path} rows={len(dataset)} cols={len(dataset.columns)}")
    print(f"wrote {manifest_summary_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
