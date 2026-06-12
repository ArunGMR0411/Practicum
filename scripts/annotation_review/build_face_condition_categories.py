#!/usr/bin/env python3

"""Build candidate face-condition categories from reviewed boxes and image cues."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


FACE_COUNT_LABELS = {
    0: "no_face",
    1: "single_face",
}

SUBGROUP_RULES = {
    "multi_face": lambda row: row["face_count_category"] == "multi_face",
    "single_face": lambda row: row["face_count_category"] == "single_face",
    "no_face": lambda row: row["face_count_category"] == "no_face",
    "very_small_or_distant_face": lambda row: row["face_scale_category"] == "very_small_or_distant",
    "small_face": lambda row: row["face_scale_category"] == "small",
    "medium_face": lambda row: row["face_scale_category"] == "medium",
    "large_face": lambda row: row["face_scale_category"] == "large",
    "mixed_scale_face": lambda row: row["face_scale_category"] == "mixed_scale",
    "edge_or_partial_face": lambda row: row["edge_partial_face"] == "yes",
    "profile_or_occluded_face": lambda row: row["profile_occluded_face"] == "yes",
    "downward_egocentric_view": lambda row: row["downward_egocentric_view"] == "yes",
    "motion_blur_or_low_sharpness": lambda row: row["blur_low_sharpness"] == "yes",
    "low_light_or_dim": lambda row: row["low_light_dim"] == "yes",
    "high_clutter": lambda row: row["clutter_level"] == "high",
    "outdoor_or_vehicle_scene": lambda row: row["outdoor_vehicle_scene"] == "yes",
}


def scale_label(area_frac: float, height_frac: float) -> str:
    """Map one face box to a scale label calibrated from the reviewed global protocol."""
    if height_frac < 0.065 or area_frac < 0.003:
        return "very_small_or_distant"
    if height_frac < 0.13 or area_frac < 0.0085:
        return "small"
    if height_frac < 0.25 or area_frac < 0.022:
        return "medium"
    return "large"


def image_quality(path: Path) -> dict[str, float]:
    """Return lightweight image cues used for category prefill."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float((edges > 0).mean())
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    green_blue_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 125) & (hsv[:, :, 1] > 45)).mean()
    return {
        "laplacian_var": laplacian_var,
        "brightness": brightness,
        "edge_density": edge_density,
        "green_blue_ratio": float(green_blue_mask),
    }


def build_category_rows(
    manifest_path: Path,
    boxes_path: Path,
    raw_root: Path,
    review_status: str,
) -> pd.DataFrame:
    """Build one category row per manifest image."""
    manifest = pd.read_csv(manifest_path)
    boxes = pd.read_csv(boxes_path)
    grouped_boxes = {image_id: group.copy() for image_id, group in boxes.groupby("image_id")}

    quality_rows: list[dict[str, float | str]] = []
    for row in manifest.itertuples(index=False):
        image_id = str(row.relative_path)
        cues = image_quality(raw_root / image_id)
        quality_rows.append({"image_id": image_id, **cues})
    quality = pd.DataFrame(quality_rows)
    blur_threshold = float(quality["laplacian_var"].quantile(0.20))
    low_light_threshold = float(quality["brightness"].quantile(0.20))
    high_clutter_threshold = float(quality["edge_density"].quantile(0.66))
    low_clutter_threshold = float(quality["edge_density"].quantile(0.33))

    rows: list[dict[str, str]] = []
    for manifest_row in manifest.itertuples(index=False):
        image_id = str(manifest_row.relative_path)
        width = int(manifest_row.image_width)
        height = int(manifest_row.image_height)
        image_boxes = grouped_boxes.get(image_id, pd.DataFrame())
        face_count = len(image_boxes)
        face_count_category = FACE_COUNT_LABELS.get(face_count, "multi_face")

        per_face_scales: list[str] = []
        edge_partial = "no"
        downward = "no"
        if face_count == 0:
            face_scale_category = "none"
        else:
            areas = []
            center_y_ratios = []
            for box in image_boxes.itertuples(index=False):
                box_width = max(1, int(box.x2) - int(box.x1))
                box_height = max(1, int(box.y2) - int(box.y1))
                area_frac = (box_width * box_height) / max(1, width * height)
                height_frac = box_height / max(1, height)
                areas.append(area_frac)
                center_y_ratios.append(((int(box.y1) + int(box.y2)) / 2.0) / max(1, height))
                per_face_scales.append(scale_label(area_frac, height_frac))
                margin_x = 0.03 * width
                margin_y = 0.03 * height
                if int(box.x1) <= margin_x or int(box.y1) <= margin_y or int(box.x2) >= width - margin_x or int(box.y2) >= height - margin_y:
                    edge_partial = "yes"

            unique_scales = set(per_face_scales)
            area_ratio = max(areas) / max(min(areas), 1e-9)
            major_scale_gap = (
                ("very_small_or_distant" in unique_scales and ("medium" in unique_scales or "large" in unique_scales))
                or ("small" in unique_scales and "large" in unique_scales)
            )
            if face_count >= 2 and (major_scale_gap or area_ratio >= 10.0):
                face_scale_category = "mixed_scale"
            else:
                scale_priority = ["large", "medium", "small", "very_small_or_distant"]
                face_scale_category = next(label for label in scale_priority if label in unique_scales)
            if max(center_y_ratios) >= 0.62:
                downward = "yes"

        cues = quality.loc[quality["image_id"] == image_id].iloc[0]
        blur = "yes" if float(cues.laplacian_var) <= blur_threshold else "no"
        low_light = "yes" if float(cues.brightness) <= low_light_threshold else "no"
        edge_density = float(cues.edge_density)
        if edge_density >= high_clutter_threshold:
            clutter = "high"
        elif edge_density <= low_clutter_threshold:
            clutter = "low"
        else:
            clutter = "medium"

        outdoor_vehicle = "yes" if float(cues.green_blue_ratio) >= 0.22 and float(cues.brightness) >= 95 else "no"
        profile_occluded = "pending_review"

        labels = [face_count_category]
        if face_scale_category != "none":
            labels.append(face_scale_category)
        for flag, label in [
            (edge_partial, "edge_or_partial"),
            (profile_occluded, "profile_or_occluded"),
            (downward, "downward_egocentric"),
            (blur, "motion_blur_or_low_sharpness"),
            (low_light, "low_light_or_dim"),
            (outdoor_vehicle, "outdoor_or_vehicle"),
        ]:
            if flag == "yes":
                labels.append(label)
        labels.append(f"clutter_{clutter}")

        rows.append(
            {
                "image_id": image_id,
                "face_count_category": face_count_category,
                "face_scale_category": face_scale_category,
                "edge_partial_face": edge_partial,
                "profile_occluded_face": profile_occluded,
                "downward_egocentric_view": downward,
                "blur_low_sharpness": blur,
                "low_light_dim": low_light,
                "clutter_level": clutter,
                "text_screen_risk": "not_assessed_in_face_review",
                "outdoor_vehicle_scene": outdoor_vehicle,
                "category_review_status": review_status,
                "condition_label": "|".join(labels),
                "category_notes": (
                    "Candidate labels generated from reviewed face boxes and image-quality cues. "
                    "Profile/occlusion and semantic scene labels require reviewer confirmation."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_subgroups(category_df: pd.DataFrame, manifest_path: Path, output_dir: Path) -> None:
    """Write one CSV per subgroup using manifest metadata plus category labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.csv"):
        stale_file.unlink()
    manifest = pd.read_csv(manifest_path)
    merged = manifest.merge(category_df, left_on="relative_path", right_on="image_id", how="inner")
    for subgroup, rule in SUBGROUP_RULES.items():
        mask = merged.apply(rule, axis=1)
        merged.loc[mask].to_csv(output_dir / f"{subgroup}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--category-output", required=True)
    parser.add_argument("--subgroup-dir", required=True)
    parser.add_argument("--review-status", default="pending")
    args = parser.parse_args()

    category_df = build_category_rows(
        manifest_path=Path(args.manifest),
        boxes_path=Path(args.boxes),
        raw_root=Path(args.raw_root),
        review_status=args.review_status,
    )
    category_path = Path(args.category_output)
    category_path.parent.mkdir(parents=True, exist_ok=True)
    category_df.to_csv(category_path, index=False)
    write_subgroups(category_df, Path(args.manifest), Path(args.subgroup_dir))
    print(f"wrote {category_path} ({len(category_df)} rows)")
    print(f"wrote subgroups to {args.subgroup_dir}")


if __name__ == "__main__":
    main()
