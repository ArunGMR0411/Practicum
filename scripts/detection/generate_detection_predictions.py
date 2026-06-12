#!/usr/bin/env python3

"""Run one detector over reviewed CASTLE annotation images and save prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANNOTATION_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "thesis_manifests" / "final_face_detection_500.csv"
)
RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"


def build_detector(args: argparse.Namespace):
    """Instantiate the requested detector backend."""
    if args.detector == "yolo":
        from src.detection.yolo_detector import YOLODetector

        return YOLODetector(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device,
            image_size=args.image_size,
        )
    if args.detector == "retinaface":
        from src.detection.retinaface_detector import RetinaFaceDetector

        return RetinaFaceDetector(threshold=args.confidence_threshold)
    if args.detector == "mtcnn":
        from src.detection.mtcnn_detector import MTCNNDetector

        return MTCNNDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
    if args.detector == "yolo_scrfd_fallback":
        from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector

        return YOLOSCRFDFallbackDetector(
            yolo_model_path=args.model_path,
            yolo_confidence_threshold=args.confidence_threshold,
            yolo_iou_threshold=args.iou_threshold,
            yolo_device=args.device,
            yolo_image_size=args.image_size,
            min_face_size_threshold_px=args.min_face_size_threshold_px,
            center_y_threshold=args.center_y_threshold,
            text_score_threshold=args.text_score_threshold,
        )
    raise ValueError(f"Unsupported detector: {args.detector}")


def iter_reviewed_images(limit: int | None = None) -> list[dict[str, str]]:
    """Return final 500-frame detection-manifest rows with valid image paths."""
    with ANNOTATION_MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        reviewed = list(csv.DictReader(handle))
    if limit is not None:
        reviewed = reviewed[:limit]
    return reviewed


def resolve_image_path(
    row: dict[str, object],
    override_root: Path | None,
    override_suffix: str | None,
) -> Path:
    """Resolve the source image path, optionally swapping root and suffix for ablations."""
    if override_root is None:
        annotation_path = PROJECT_ROOT / str(row.get("annotation_image_path", ""))
        if annotation_path.is_file():
            return annotation_path
        return RAW_ROOT / str(row["relative_path"])
    relative_path = Path(str(row["relative_path"]))
    if override_suffix:
        relative_path = relative_path.with_suffix(override_suffix)
    return override_root / relative_path


def write_predictions(rows: list[dict[str, object]], output_path: Path) -> None:
    """Atomically save detector predictions as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "x1", "y1", "x2", "y2", "score", "condition_label", "detector_name"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo", "retinaface", "mtcnn", "yolo_scrfd_fallback"], required=True)
    parser.add_argument("--model-path", default="data/models/yolov8n.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-face-size-threshold-px", type=float, default=96.0)
    parser.add_argument("--center-y-threshold", type=float, default=0.65)
    parser.add_argument("--text-score-threshold", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--image-suffix", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    detector = build_detector(args)
    records = iter_reviewed_images(limit=args.limit)
    image_root = Path(args.image_root) if args.image_root else None
    prediction_rows: list[dict[str, object]] = []
    image_count = 0
    detection_count = 0

    for row in records:
        image_path = resolve_image_path(row, image_root, args.image_suffix)
        image = Image.open(image_path)
        result = detector.detect(image)
        image_id = str(row["relative_path"])
        condition_label = str(row.get("condition_label", ""))
        image_count += 1
        for detection in result.detections:
            x1, y1, x2, y2 = detection.box
            prediction_rows.append(
                {
                    "image_id": image_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": detection.confidence,
                    "condition_label": condition_label,
                    "detector_name": detector.detector_name,
                }
            )
            detection_count += 1

    output_path = Path(args.output)
    write_predictions(prediction_rows, output_path)
    print(
        json.dumps(
            {
                "detector": args.detector,
                "images_processed": image_count,
                "detections_written": detection_count,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
