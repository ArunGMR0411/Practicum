#!/usr/bin/env python3

"""Helpers for defended CASTLE development and calibration subset building."""

from __future__ import annotations

import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.system_config import resolve_torch_device

try:
    from facenet_pytorch import MTCNN
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    MTCNN = None  # type: ignore[assignment]


CONDITION_LABELS = [
    "small_face",
    "motion_blur",
    "extreme_pose",
    "downward_view",
    "visible_screen",
    "visible_text",
    "multiple_faces",
    "no_face",
]

QUALITY_DIMENSIONS = [
    "blur_level",
    "face_size",
    "occlusion_ratio",
    "webp_artefact_severity",
]

FACE_ANALYSIS_COLUMNS = [
    "face_count",
    "largest_face_size_px",
    "min_face_size_px",
    "occlusion_ratio_value",
    "small_face_flag",
    "extreme_pose_flag",
    "downward_view_flag",
    "multiple_faces_flag",
    "no_face_flag",
]

FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)

ANALYSIS_COLUMNS = [
    "blur_score",
    "webp_blockiness_score",
    "face_count",
    "largest_face_size_px",
    "min_face_size_px",
    "occlusion_ratio_value",
    "screen_score",
    "text_score",
    "small_face_flag",
    "motion_blur_flag",
    "extreme_pose_flag",
    "downward_view_flag",
    "visible_screen_flag",
    "visible_text_flag",
    "multiple_faces_flag",
    "no_face_flag",
    "blur_level",
    "face_size",
    "occlusion_ratio",
    "webp_artefact_severity",
]

BOOLEAN_ANALYSIS_COLUMNS = [
    "small_face_flag",
    "motion_blur_flag",
    "extreme_pose_flag",
    "downward_view_flag",
    "visible_screen_flag",
    "visible_text_flag",
    "multiple_faces_flag",
    "no_face_flag",
]

NUMERIC_ANALYSIS_COLUMNS = [
    "blur_score",
    "webp_blockiness_score",
    "face_count",
    "largest_face_size_px",
    "min_face_size_px",
    "occlusion_ratio_value",
    "screen_score",
    "text_score",
]

CATEGORICAL_ANALYSIS_COLUMNS = [
    "blur_level",
    "face_size",
    "occlusion_ratio",
    "webp_artefact_severity",
]

LEGACY_CATEGORY_MAP = {
    "blur_level": {
        "sharp": "high",
        "moderate": "medium",
        "blurred": "low",
    },
    "occlusion_ratio": {
        "moderate": "medium",
    },
}


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load the main CASTLE manifest and attach convenience aliases."""
    df = pd.read_csv(manifest_path)
    df = df[df["integrity_status"] == "valid"].copy()
    df["day_id"] = df["day_or_session_id"]
    df["timestamp_id"] = df["file_name"].str.rsplit(".", n=1).str[0]
    return df.reset_index(drop=True)


def ensure_analysis_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected analysis columns exist even for sparse or empty frames."""
    result = df.copy()
    for column in NUMERIC_ANALYSIS_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0
    for column in BOOLEAN_ANALYSIS_COLUMNS:
        if column not in result.columns:
            result[column] = False
    for column in CATEGORICAL_ANALYSIS_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    if "condition_matches" not in result.columns:
        result["condition_matches"] = [[] for _ in range(len(result))]
    result = normalise_legacy_category_labels(result)
    return result


def normalise_legacy_category_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map older categorical label names into the current project schema."""
    result = df.copy()
    for column, value_map in LEGACY_CATEGORY_MAP.items():
        if column not in result.columns:
            continue
        result[column] = result[column].replace(value_map)
    return result


def stable_hash(seed: int, value: str) -> int:
    """Return a deterministic integer hash derived from SHA-256."""
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stream_day_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Sample at least one frame per present stream-day combination."""
    sampled_groups = [
        group.sample(n=1, random_state=seed)
        for _, group in df.groupby(["day_id", "camera_stream_id"], dropna=False, sort=True)
    ]
    if not sampled_groups:
        return df.head(0).reset_index(drop=True)
    return pd.concat(sampled_groups, ignore_index=True).reset_index(drop=True)


def analyse_rows(
    df: pd.DataFrame,
    raw_root: str | Path,
    analysis_width: int = 192,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Compute lightweight heuristic condition and quality proxies per frame."""
    raw_root = Path(raw_root)
    records = df.to_dict("records")

    def analyse_record(row: dict[str, Any]) -> dict[str, Any]:
        image_path = raw_root / row["relative_path"]
        record = dict(row)
        record.update(analyse_image(image_path, analysis_width=analysis_width))
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        analysed_records = list(executor.map(analyse_record, records))
    return ensure_analysis_schema(pd.DataFrame(analysed_records))


def analyse_image(image_path: Path, analysis_width: int = 192) -> dict[str, Any]:
    """Analyse one image using lightweight OpenCV heuristics."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image for subset analysis: {image_path}")

    resized = resize_for_analysis(image, analysis_width=analysis_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blockiness_score = compute_blockiness(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(14, 14),
    )
    face_count = int(len(faces))
    face_boxes = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    largest_face = face_boxes[0] if face_boxes else None
    largest_face_size = int(min(largest_face[2], largest_face[3])) if largest_face is not None else 0
    min_face_size = int(min(min(box[2], box[3]) for box in face_boxes)) if face_boxes else 0

    multiple_faces = face_count >= 2
    no_face = face_count == 0
    small_face = any(min(box[2], box[3]) < 30 for box in face_boxes) if face_boxes else False

    extreme_pose = False
    downward_view = False
    occlusion_ratio_value = 1.0 if no_face else 0.0
    if largest_face is not None:
        x, y, w, h = [int(v) for v in largest_face]
        center_y = y + h / 2.0
        downward_view = center_y > gray.shape[0] * 0.62
        border_margin_x = gray.shape[1] * 0.05
        border_margin_y = gray.shape[0] * 0.05
        border_contacts = sum(
            [
                x <= border_margin_x,
                y <= border_margin_y,
                x + w >= gray.shape[1] - border_margin_x,
                y + h >= gray.shape[0] - border_margin_y,
            ]
        )
        occlusion_ratio_value = border_contacts / 4.0
        aspect_ratio = w / max(h, 1)
        extreme_pose = aspect_ratio < 0.78 or aspect_ratio > 1.22 or border_contacts >= 2

    screen_score = detect_screen_score(gray)
    text_score = detect_text_score(gray)

    return {
        "blur_score": blur_score,
        "webp_blockiness_score": blockiness_score,
        "face_count": face_count,
        "largest_face_size_px": largest_face_size,
        "min_face_size_px": min_face_size,
        "occlusion_ratio_value": round(float(occlusion_ratio_value), 4),
        "screen_score": round(float(screen_score), 4),
        "text_score": int(text_score),
        "small_face_flag": bool(small_face),
        "motion_blur_flag": False,
        "extreme_pose_flag": bool(extreme_pose),
        "downward_view_flag": bool(downward_view),
        "visible_screen_flag": screen_score >= 0.10,
        "visible_text_flag": text_score >= 14,
        "multiple_faces_flag": bool(multiple_faces),
        "no_face_flag": bool(no_face),
    }


def resize_for_analysis(image: np.ndarray, analysis_width: int = 192) -> np.ndarray:
    """Resize images for lightweight heuristic analysis while preserving aspect ratio."""
    height, width = image.shape[:2]
    if width <= analysis_width:
        return image
    scale = analysis_width / width
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (analysis_width, resized_height), interpolation=cv2.INTER_AREA)


def compute_blockiness(gray: np.ndarray, block_size: int = 8) -> float:
    """Compute a simple WebP-style block boundary artifact proxy."""
    vertical = 0.0
    horizontal = 0.0
    for col in range(block_size, gray.shape[1], block_size):
        vertical += float(np.mean(np.abs(gray[:, col] - gray[:, col - 1])))
    for row in range(block_size, gray.shape[0], block_size):
        horizontal += float(np.mean(np.abs(gray[row, :] - gray[row - 1, :])))
    norm = max((gray.shape[1] // block_size) + (gray.shape[0] // block_size), 1)
    return (vertical + horizontal) / norm


def detect_screen_score(gray: np.ndarray) -> float:
    """Estimate the likelihood of a visible screen using rectangularity and edge structure."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    best_ratio = 0.0
    line_score = 0.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=max(gray.shape[1] // 8, 20), maxLineGap=12)
    if lines is not None:
        horizontal = 0
        vertical = 0
        for line in lines[:, 0]:
            x1, y1, x2, y2 = [int(value) for value in line]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx >= dy * 3:
                horizontal += 1
            elif dy >= dx * 3:
                vertical += 1
        if horizontal > 0 and vertical > 0:
            line_score = min((horizontal + vertical) / 30.0, 1.0)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        area = cv2.contourArea(contour)
        if len(approx) != 4 or area <= image_area * 0.03:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        rect_area = max(w * h, 1)
        fill_ratio = area / rect_area
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        if fill_ratio < 0.65 or aspect_ratio > 2.5:
            continue
        candidate_ratio = area / image_area
        best_ratio = max(best_ratio, min(candidate_ratio * 2.5, 1.0))
    return float(min(best_ratio + line_score * 0.35, 1.0))


def detect_text_score(gray: np.ndarray) -> int:
    """Estimate visible text density using fast connected-component counts."""
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(merged)
    count = 0
    for index in range(1, num_labels):
        x, y, w, h, area = stats[index]
        if area < 12:
            continue
        if 3 <= h <= 30 and 8 <= w <= 120 and w >= h:
            count += 1
    return count


def get_mtcnn(device: str | None = None) -> MTCNN | None:
    """Create an MTCNN detector when the dependency is available."""
    if MTCNN is None:
        return None
    resolved_device = device or resolve_torch_device()
    return MTCNN(keep_all=True, device=resolved_device)


def analyse_face_conditions_with_mtcnn(
    df: pd.DataFrame,
    raw_root: str | Path,
    analysis_width: int = 640,
    device: str | None = None,
) -> pd.DataFrame:
    """Refine face-condition candidates with MTCNN and simple landmark-derived pose cues."""
    mtcnn = get_mtcnn(device=device)
    if mtcnn is None:
        raise RuntimeError("facenet_pytorch.MTCNN is not available in the current environment")
    raw_root = Path(raw_root)
    refined_rows: list[dict[str, Any]] = []
    for row in df.to_dict("records"):
        image_path = raw_root / str(row["relative_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        resized = resize_for_analysis(image_bgr, analysis_width=analysis_width)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(resized_rgb)
        boxes, probs, landmarks = mtcnn.detect(pil_image, landmarks=True)
        face_boxes: list[tuple[float, float, float, float, float, Any]] = []
        if boxes is not None and probs is not None:
            for index, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or float(prob) < 0.75:
                    continue
                lm = landmarks[index] if landmarks is not None else None
                x1, y1, x2, y2 = [float(v) for v in box]
                face_boxes.append((x1, y1, x2, y2, float(prob), lm))

        face_count = len(face_boxes)
        multiple_faces = face_count >= 2
        no_face = face_count == 0
        largest_face_size_px = 0.0
        min_face_size_px = 0.0
        dominant_conf = 0.0
        downward_view = False
        extreme_pose = False
        occlusion_ratio_value = 1.0 if no_face else 0.0
        small_face = False
        if face_boxes:
            face_boxes.sort(key=lambda item: (item[2] - item[0]) * (item[3] - item[1]), reverse=True)
            min_face_size_px = min(min(max(item[2] - item[0], 1.0), max(item[3] - item[1], 1.0)) for item in face_boxes)
            x1, y1, x2, y2, dominant_conf, dominant_landmarks = face_boxes[0]
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            largest_face_size_px = min(width, height)
            small_face = largest_face_size_px < 32.0
            center_y = ((y1 + y2) / 2.0) / max(resized.shape[0], 1)
            downward_view = center_y > 0.62
            border_margin_x = resized.shape[1] * 0.05
            border_margin_y = resized.shape[0] * 0.05
            border_contacts = sum(
                [
                    x1 <= border_margin_x,
                    y1 <= border_margin_y,
                    x2 >= resized.shape[1] - border_margin_x,
                    y2 >= resized.shape[0] - border_margin_y,
                ]
            )
            occlusion_ratio_value = border_contacts / 4.0
            aspect_ratio = width / height
            if dominant_landmarks is not None:
                left_eye = dominant_landmarks[0]
                right_eye = dominant_landmarks[1]
                nose = dominant_landmarks[2]
                eye_distance = max(abs(float(right_eye[0]) - float(left_eye[0])), 1.0)
                eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
                nose_offset = abs(float(nose[0]) - eye_mid_x) / eye_distance
                extreme_pose = nose_offset > 0.18 or aspect_ratio < 0.72 or aspect_ratio > 1.28 or border_contacts >= 2
            else:
                extreme_pose = aspect_ratio < 0.72 or aspect_ratio > 1.28 or border_contacts >= 2

        refined = dict(row)
        refined.update(
            {
                "face_count": face_count,
                "largest_face_size_px": round(float(largest_face_size_px), 2),
                "min_face_size_px": round(float(min_face_size_px), 2),
                "occlusion_ratio_value": round(float(occlusion_ratio_value), 4),
                "small_face_flag": bool(small_face),
                "extreme_pose_flag": bool(extreme_pose),
                "downward_view_flag": bool(downward_view),
                "multiple_faces_flag": bool(multiple_faces),
                "no_face_flag": bool(no_face),
                "dominant_face_confidence": round(float(dominant_conf), 3),
            }
        )
        refined_rows.append(refined)
    refined_df = pd.DataFrame(refined_rows)
    if refined_df.empty:
        return ensure_analysis_schema(refined_df)
    return ensure_analysis_schema(attach_quantile_labels(refined_df))


def attach_quantile_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Attach categorical quality-dimension labels for calibration sampling."""
    result = ensure_analysis_schema(df)
    result["motion_blur_flag"] = result["blur_score"] <= result["blur_score"].quantile(0.25)
    result["blur_level"] = quantile_label(result["blur_score"], labels=["low", "medium", "high"])
    result["webp_artefact_severity"] = quantile_label(
        result["webp_blockiness_score"],
        labels=["low", "medium", "high"],
    )
    result["face_size"] = result["largest_face_size_px"].apply(face_size_bucket)
    result["occlusion_ratio"] = result["occlusion_ratio_value"].apply(occlusion_bucket)
    result["condition_matches"] = result.apply(condition_membership, axis=1)
    return result


def quantile_label(series: pd.Series, labels: list[str]) -> pd.Series:
    """Convert a numeric series into stable quantile labels."""
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels)


def face_size_bucket(size_px: int) -> str:
    """Bucket detected face size for routing calibration."""
    if size_px <= 0:
        return "none"
    if size_px < 30:
        return "small"
    if size_px < 80:
        return "medium"
    return "large"


def occlusion_bucket(value: float) -> str:
    """Bucket the heuristic occlusion ratio proxy."""
    if value <= 0.0:
        return "low"
    if value <= 0.25:
        return "medium"
    return "high"


def condition_membership(row: pd.Series) -> list[str]:
    """Return all hard-condition labels satisfied by one analysed row."""
    matches: list[str] = []
    for label, flag_name in [
        ("small_face", "small_face_flag"),
        ("motion_blur", "motion_blur_flag"),
        ("extreme_pose", "extreme_pose_flag"),
        ("downward_view", "downward_view_flag"),
        ("visible_screen", "visible_screen_flag"),
        ("visible_text", "visible_text_flag"),
        ("multiple_faces", "multiple_faces_flag"),
        ("no_face", "no_face_flag"),
    ]:
        if bool(row[flag_name]):
            matches.append(label)
    return matches


def apply_face_bucket_updates(df: pd.DataFrame) -> pd.DataFrame:
    """Refresh only face-derived categorical labels after a stronger face-analysis pass."""
    result = ensure_analysis_schema(df)
    result["face_size"] = result["largest_face_size_px"].apply(face_size_bucket)
    result["occlusion_ratio"] = result["occlusion_ratio_value"].apply(occlusion_bucket)
    return result


def choose_condition_frame(row: pd.Series, target_condition: str) -> bool:
    """Return True when a row is eligible for the requested hard condition."""
    return target_condition in row["condition_matches"]


def merge_with_analysis_cache(
    df: pd.DataFrame,
    cache_path: str | Path,
    raw_root: str | Path,
    analysis_width: int = 192,
    max_workers: int = 4,
    face_backend: str = "heuristic",
    face_analysis_width: int = 640,
    face_device: str | None = None,
) -> pd.DataFrame:
    """Merge cached image-analysis features and analyse only uncached rows."""
    cache_path = Path(cache_path)
    cache_df = pd.DataFrame()
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
        cache_df = cache_df.drop_duplicates("relative_path")
        missing_columns = [column for column in ANALYSIS_COLUMNS if column not in cache_df.columns]
        for column in missing_columns:
            cache_df[column] = np.nan

    if not cache_df.empty:
        complete_mask = ~cache_df[ANALYSIS_COLUMNS].isna().any(axis=1)
        cached_paths = set(cache_df.loc[complete_mask, "relative_path"].tolist())
    else:
        cached_paths = set()
    missing_df = df[~df["relative_path"].isin(cached_paths)].copy()
    if not missing_df.empty:
        analysed_missing = attach_quantile_labels(
            analyse_rows(
                missing_df,
                raw_root=raw_root,
                analysis_width=analysis_width,
                max_workers=max_workers,
            )
        )
        cache_df = pd.concat([cache_df, analysed_missing], ignore_index=True)
        cache_df = cache_df.drop_duplicates("relative_path", keep="last").reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cache_path, index=False)

    merged_df = df.merge(
        cache_df[["relative_path"] + ANALYSIS_COLUMNS],
        on="relative_path",
        how="left",
    )
    if merged_df[ANALYSIS_COLUMNS].isna().any().any():
        raise ValueError("Analysis cache merge left missing analysis values")
    merged_df = ensure_analysis_schema(merged_df)
    if face_backend == "mtcnn":
        mtcnn_face_df = analyse_face_conditions_with_mtcnn(
            merged_df[["relative_path"]],
            raw_root=raw_root,
            analysis_width=face_analysis_width,
            device=face_device,
        )
        mtcnn_face_df = mtcnn_face_df[["relative_path"] + FACE_ANALYSIS_COLUMNS]
        merged_df = merged_df.drop(columns=FACE_ANALYSIS_COLUMNS, errors="ignore").merge(
            mtcnn_face_df,
            on="relative_path",
            how="left",
        )
        if merged_df[FACE_ANALYSIS_COLUMNS].isna().any().any():
            raise ValueError("MTCNN face-analysis merge left missing face-analysis values")
        merged_df = ensure_analysis_schema(merged_df)
        merged_df = apply_face_bucket_updates(merged_df)
    merged_df["condition_matches"] = merged_df.apply(condition_membership, axis=1)
    return merged_df


def choose_diverse_candidate_pool(
    manifest_df: pd.DataFrame,
    pool_size: int,
    seed: int,
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Choose a diversity-first candidate pool instead of lexical-path-first rows."""
    if group_columns is None:
        group_columns = ["day_id", "camera_stream_id", "view_type"]
    working_df = manifest_df.copy()
    for column in group_columns:
        if column not in working_df.columns:
            working_df[column] = ""
    anchors = (
        working_df.sort_values("relative_path")
        .groupby(group_columns, group_keys=False, dropna=False)
        .head(2)
        .reset_index(drop=True)
    )
    remaining = working_df[~working_df["relative_path"].isin(anchors["relative_path"])].copy()
    remaining["group_key"] = remaining[group_columns].astype(str).agg("|".join, axis=1)
    remaining["stable_order"] = remaining.apply(
        lambda row: (
            stable_hash(seed, str(row["group_key"])),
            stable_hash(seed, str(row["relative_path"])),
            str(row["relative_path"]),
        ),
        axis=1,
    )
    ordered_remaining = remaining.sort_values("stable_order").reset_index(drop=True)
    extra_count = max(pool_size - len(anchors), 0)
    extras = ordered_remaining.head(extra_count).drop(columns=["group_key", "stable_order"], errors="ignore")
    return pd.concat([anchors, extras], ignore_index=True).drop_duplicates("relative_path").reset_index(drop=True)


def condition_score(row: pd.Series, condition: str) -> float:
    """Return a sortable confidence score for one condition family."""
    if condition == "small_face":
        if int(row.get("face_count", 0)) <= 0:
            return -1.0
        min_face = float(row.get("min_face_size_px", row.get("largest_face_size_px", 0.0)) or 0.0)
        face_conf = 1.0 if bool(row.get("small_face_flag", False)) else 0.0
        return (1000.0 if face_conf else 0.0) + max(0.0, 80.0 - min_face)
    if condition == "motion_blur":
        return -float(row.get("blur_score", 0.0)) + float(row.get("face_count", 0.0)) * 25.0
    if condition == "extreme_pose":
        return (1000.0 if bool(row.get("extreme_pose_flag", False)) else 0.0) + float(row.get("occlusion_ratio_value", 0.0)) * 10.0
    if condition == "downward_view":
        return (1000.0 if bool(row.get("downward_view_flag", False)) else 0.0) + float(row.get("largest_face_size_px", 0.0)) / 10.0
    if condition == "visible_screen":
        return (
            float(row.get("screen_score", 0.0)) * 100.0
            + float(row.get("text_score", 0.0)) * 0.35
            + (6.0 if str(row.get("camera_stream_id", "")) in {"meeting", "reading"} else 0.0)
            - (8.0 if str(row.get("camera_stream_id", "")) in {"living1", "living2"} else 0.0)
        )
    if condition == "visible_text":
        return (
            float(row.get("text_score", 0.0)) * 3.0
            + min(float(row.get("blur_score", 0.0)) / 150.0, 10.0)
            - float(row.get("screen_score", 0.0)) * 10.0
        )
    if condition == "multiple_faces":
        return float(row.get("face_count", 0.0))
    if condition == "no_face":
        return (
            (1000.0 if bool(row.get("no_face_flag", False)) else 0.0)
            - float(row.get("screen_score", 0.0)) * 20.0
            - float(row.get("text_score", 0.0))
            + min(float(row.get("blur_score", 0.0)) / 200.0, 5.0)
        )
    return 0.0


def build_condition_candidate_bank(
    analysed_df: pd.DataFrame,
    condition: str,
    top_k: int = 50,
    seed: int = 42,
    diversity_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a ranked, diversity-aware candidate bank for one condition."""
    if diversity_columns is None:
        diversity_columns = ["day_id", "camera_stream_id"]
    candidates = ensure_analysis_schema(analysed_df)
    for column in diversity_columns:
        if column not in candidates.columns:
            candidates[column] = ""
        candidates[column] = candidates[column].fillna("").astype(str)
    candidates["candidate_score"] = candidates.apply(lambda row: condition_score(row, condition), axis=1)
    if condition == "motion_blur":
        candidates = candidates[
            candidates["motion_blur_flag"].astype(bool)
            & (candidates["face_count"] > 0)
            & (candidates["screen_score"] < 0.20)
        ]
    elif condition in {"visible_screen", "visible_text"}:
        threshold_column = "visible_screen_flag" if condition == "visible_screen" else "visible_text_flag"
        if condition == "visible_screen":
            candidates = candidates[
                (
                    (candidates[threshold_column].astype(bool))
                    | (candidates["screen_score"] >= 0.08)
                    | ((candidates["text_score"] >= 12) & (~candidates["motion_blur_flag"].astype(bool)))
                )
                & (candidates["text_score"] <= 40)
                & (candidates["face_count"] > 0)
            ]
        else:
            candidates = candidates[
                candidates[threshold_column].astype(bool)
                & (~candidates["motion_blur_flag"].astype(bool))
                & (candidates["text_score"] >= 12)
            ]
    elif condition == "small_face":
        candidates = candidates[candidates["face_count"] > 0]
    elif condition == "no_face":
        candidates = candidates[
            candidates["no_face_flag"].astype(bool)
            & (candidates["screen_score"] < 0.12)
            & (candidates["text_score"] <= 5)
        ]
    else:
        flag_column = f"{condition}_flag"
        if flag_column in candidates.columns:
            candidates = candidates[candidates[flag_column].astype(bool)]
        if condition == "multiple_faces":
            candidates = candidates[candidates["face_count"] >= 3]
    if candidates.empty:
        return ensure_analysis_schema(candidates)
    candidates["diversity_key"] = candidates[diversity_columns].astype(str).agg("|".join, axis=1)
    candidates["stable_tiebreak"] = candidates["relative_path"].map(lambda value: (stable_hash(seed, str(value)), str(value)))
    candidates = candidates.sort_values(["candidate_score", "stable_tiebreak"], ascending=[False, True]).reset_index(drop=True)
    diverse_rows: list[pd.Series] = []
    used_diversity: set[str] = set()
    used_paths: set[str] = set()
    for _, row in candidates.iterrows():
        if row["relative_path"] in used_paths:
            continue
        if row["diversity_key"] in used_diversity and len(diverse_rows) < top_k // 2:
            continue
        diverse_rows.append(row)
        used_paths.add(str(row["relative_path"]))
        used_diversity.add(str(row["diversity_key"]))
        if len(diverse_rows) >= top_k:
            break
    if len(diverse_rows) < top_k:
        for _, row in candidates.iterrows():
            if row["relative_path"] in used_paths:
                continue
            diverse_rows.append(row)
            used_paths.add(str(row["relative_path"]))
            if len(diverse_rows) >= top_k:
                break
    result = ensure_analysis_schema(pd.DataFrame(diverse_rows).reset_index(drop=True))
    result["target_condition"] = condition
    result["label_source"] = "condition_specific_proxy"
    result["label_confidence"] = result["candidate_score"].rank(method="dense", ascending=False, pct=True).round(3)
    return ensure_analysis_schema(result.drop(columns=["diversity_key", "stable_tiebreak"], errors="ignore"))


def build_screen_candidate_bank(
    analysed_df: pd.DataFrame,
    top_k: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a looser but ranked screen candidate bank for manual validation."""
    bank = build_condition_candidate_bank(analysed_df, condition="visible_screen", top_k=top_k, seed=seed)
    if bank.empty:
        return ensure_analysis_schema(bank)
    bank["label_source"] = "screen_specific_proxy"
    return bank


def build_face_condition_candidate_banks(
    analysed_df: pd.DataFrame,
    raw_root: str | Path,
    conditions: list[str] | None = None,
    top_k: int = 50,
    seed: int = 42,
    analysis_width: int = 640,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Refine sparse face-condition banks with MTCNN-backed analysis."""
    if conditions is None:
        conditions = ["small_face", "extreme_pose", "downward_view", "multiple_faces", "no_face"]
    banks: dict[str, pd.DataFrame] = {}
    for condition in conditions:
        if condition == "no_face":
            seed_df = analysed_df[
                analysed_df["no_face_flag"].astype(bool)
                & (analysed_df["screen_score"] < 0.12)
                & (analysed_df["text_score"] <= 5)
            ].copy()
            seed_df["face_priority_score"] = (
                seed_df.get("screen_score", 0).astype(float) * -10.0
                + seed_df.get("text_score", 0).astype(float) * -1.0
                + seed_df.get("blur_score", 0).astype(float) / 200.0
            )
            seed_df = seed_df.sort_values("face_priority_score", ascending=False).head(max(top_k * 10, 250)).copy()
        else:
            seed_df = analysed_df.copy()
            seed_df["face_priority_score"] = (
                seed_df.get("face_count", 0).astype(float) * 5.0
                + seed_df.get("largest_face_size_px", 0).astype(float)
            )
            seed_df = seed_df.sort_values("face_priority_score", ascending=False).head(max(top_k * 10, 250)).copy()
        refined_df = analyse_face_conditions_with_mtcnn(
            seed_df,
            raw_root=raw_root,
            analysis_width=analysis_width,
            device=device,
        )
        bank = build_condition_candidate_bank(refined_df, condition=condition, top_k=top_k, seed=seed)
        if not bank.empty:
            bank["label_source"] = "mtcnn_refined_proxy"
        banks[condition] = ensure_analysis_schema(bank)
    return banks


def build_face_quality_candidate_banks(
    analysed_df: pd.DataFrame,
    raw_root: str | Path,
    dimensions: list[str] | None = None,
    per_value: int = 10,
    seed: int = 42,
    analysis_width: int = 640,
    device: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Refine face-dependent calibration dimensions with MTCNN-backed analysis."""
    if dimensions is None:
        dimensions = ["face_size", "occlusion_ratio"]
    candidate_seed_df = analysed_df.copy()
    candidate_seed_df["face_priority_score"] = (
        candidate_seed_df.get("face_count", 0).astype(float) * 5.0
        + candidate_seed_df.get("largest_face_size_px", 0).astype(float)
    )
    candidate_seed_df = candidate_seed_df.sort_values("face_priority_score", ascending=False).head(max(per_value * 20, 300)).copy()
    refined_df = analyse_face_conditions_with_mtcnn(
        candidate_seed_df,
        raw_root=raw_root,
        analysis_width=analysis_width,
        device=device,
    )
    banks: dict[str, pd.DataFrame] = {}
    target_values = {
        "face_size": ["none", "small", "medium", "large"],
        "occlusion_ratio": ["low", "moderate", "high"],
    }
    for dimension in dimensions:
        values = target_values[dimension]
        bank = build_quality_candidate_bank(refined_df, dimension, values, per_value=per_value, seed=seed)
        if not bank.empty:
            bank["label_source"] = "mtcnn_refined_quality_proxy"
        banks[dimension] = ensure_analysis_schema(bank)
    return banks


def build_quality_candidate_bank(
    analysed_df: pd.DataFrame,
    column: str,
    values: list[str],
    per_value: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a ranked calibration candidate bank for one quality dimension."""
    bank_parts: list[pd.DataFrame] = []
    for value in values:
        subset = analysed_df[analysed_df[column].astype(str) == value].copy()
        if subset.empty:
            continue
        subset["candidate_score"] = subset["relative_path"].map(lambda item: stable_hash(seed, f"{column}:{value}:{item}"))
        subset = subset.sort_values("candidate_score").head(per_value).copy()
        subset["target_quality_dimension"] = column
        subset["target_quality_value"] = value
        subset["label_source"] = "quality_specific_proxy"
        subset["label_confidence"] = 0.75
        bank_parts.append(subset)
    return ensure_analysis_schema(pd.concat(bank_parts, ignore_index=True) if bank_parts else pd.DataFrame())
