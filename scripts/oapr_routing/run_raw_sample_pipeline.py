#!/usr/bin/env python3
"""Run the evidence-supported routing policy on unseen raw CASTLE images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.src.privacy_pipeline_app.pipeline_demo import apply_layered, apply_solid
from src.data.subset_building import detect_text_score, resize_for_analysis
from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector


def read_manifest_paths() -> set[str]:
    paths: set[str] = set()
    canonical_manifests = [
        PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv",
        PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_face_detection_500.csv",
        PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_multimodal_250.csv",
        PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
    ]
    for manifest in canonical_manifests:
        try:
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    for key in ("relative_path", "image_id", "image_path"):
                        value = row.get(key, "")
                        if value and value.endswith((".webp", ".jpg", ".jpeg", ".png")):
                            paths.add(value.replace("data/castle2024/raw/", ""))
        except (UnicodeDecodeError, OSError):
            continue
    return paths


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quality_signals(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> dict[str, object]:
    small = image.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
    array = np.asarray(small)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    edge_density = float((cv2.Canny(gray, 80, 160) > 0).mean())
    analysis = resize_for_analysis(cv2.cvtColor(array, cv2.COLOR_RGB2BGR), analysis_width=192)
    text_score = int(detect_text_score(cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)))
    height = image.height
    largest = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), default=None)
    face_scale = "none"
    if largest is not None:
        face_height = largest[3] - largest[1]
        ratio = face_height / max(height, 1)
        face_scale = "small" if ratio < 0.13 else "medium" if ratio < 0.25 else "large"
    return {
        "sharpness_score": round(sharpness, 3),
        "brightness_mean": round(brightness, 3),
        "edge_density": round(edge_density, 5),
        "text_risk_signal": text_score >= 12,
        "text_signal_score": text_score,
        "face_scale_signal": face_scale,
        "face_count": len(boxes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "sample_test")
    args = parser.parse_args()
    raw_root = PROJECT_ROOT / "data/castle2024/raw"
    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(
        path for path in raw_root.rglob("*")
        if path.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}
    )
    excluded = read_manifest_paths()
    candidates = [p for p in source_paths if str(p.relative_to(raw_root)) not in excluded]
    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.count)

    detector = YOLOSCRFDFallbackDetector(
        yolo_model_path="data/models/face_detection_candidates/yolo11s_widerface.pt",
        yolo_confidence_threshold=0.25,
        yolo_iou_threshold=0.5,
        yolo_device="0",
        yolo_image_size=1280,
        min_face_size_threshold_px=96.0,
        center_y_threshold=0.65,
        text_score_threshold=12,
    )
    rows: list[dict[str, object]] = []
    for index, source in enumerate(selected, start=1):
        relative = source.relative_to(raw_root)
        sample_id = f"sample_{index:02d}_{source.stem}"
        started = time.perf_counter()
        with Image.open(source).convert("RGB") as image:
            result = detector.detect(image)
            boxes = [tuple(map(int, detection.box)) for detection in result.detections]
            signals = quality_signals(image, boxes)
            safety_candidate = len(boxes) > 0
            if not safety_candidate:
                method = "no_action_copy"
                action = "verified_no_face_copy"
                output_path = args.output / "outputs" / f"{sample_id}.webp"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, format="WEBP", quality=95, method=6)
            elif len(boxes) == 1:
                method = "solid_mask_black"
                action = "single_face_policy"
                output_path = args.output / "outputs" / f"{sample_id}.webp"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                apply_solid(image, boxes).save(output_path, format="WEBP", quality=95, method=6)
            else:
                method = "layered_blur_downscale_noise"
                action = "face_positive_fallback_policy"
                output_path = args.output / "outputs" / f"{sample_id}.webp"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                apply_layered(image, boxes).save(output_path, format="WEBP", quality=95, method=6)
        elapsed = time.perf_counter() - started
        rows.append({
            "sample_id": sample_id,
            "source_relative_path": str(relative),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_width": image.width,
            "source_height": image.height,
            "detector": detector.detector_name,
            "detector_device": "cuda:0",
            "detector_box_count": len(boxes),
            "detector_boxes_json": str(boxes),
            **signals,
            "selected_method": method,
            "selected_action": action,
            "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            "output_exists": output_path.exists(),
            "runtime_seconds": round(elapsed, 4),
            "status": "ok",
        })

    write_csv(args.output / "routing_log.csv", rows)
    manifest_rows = [{k: row[k] for k in ("sample_id", "source_relative_path", "selected_method", "output_path", "status")} for row in rows]
    write_csv(args.output / "output_manifest.csv", manifest_rows)
    readme = [
        "# Unseen Raw CASTLE Sample Run",
        "",
        "This sample contains five raw CASTLE images selected with a fixed seed and excluded from all CSV manifests found under `data/`.",
        "",
        "The run used the GPU-configured YOLO/SCRFD detector, lightweight condition telemetry, the no-face safety rule, and the existing solid-mask/layered anonymisation functions.",
        "",
        "This is a functional pipeline demonstration, not a benchmark: these images have no human ground truth and therefore no accuracy score can be claimed.",
        "",
        "The original raw images were not modified. Derived outputs and routing metadata are in this directory.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"processed={len(rows)} output={args.output}")
    print("method_distribution", {method: sum(row["selected_method"] == method for row in rows) for method in sorted({row["selected_method"] for row in rows})})


if __name__ == "__main__":
    main()
