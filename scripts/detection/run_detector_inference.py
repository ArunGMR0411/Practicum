#!/usr/bin/env python3

"""Run a configured face detector over a manifest-backed image set and save detections."""

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

from src.data.castle_loader import CASTLEDataset


def build_detector(args: argparse.Namespace):
    """Instantiate one detector from CLI arguments."""
    if args.detector == "mtcnn":
        from src.detection.mtcnn_detector import MTCNNDetector

        return MTCNNDetector(
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
    if args.detector == "yolo":
        from src.detection.yolo_detector import YOLODetector

        return YOLODetector(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            device=args.device,
            image_size=args.image_size,
            allowed_class_ids=args.allowed_class_id,
        )
    if args.detector == "scrfd":
        from src.detection.scrfd_detector import SCRFDDetector

        return SCRFDDetector(
            model_path=args.scrfd_model_path,
            confidence_threshold=args.confidence_threshold,
            input_size=(args.scrfd_input_size, args.scrfd_input_size),
        )
    if args.detector == "yunet":
        from src.detection.yunet_detector import YuNetDetector

        return YuNetDetector(
            model_path=args.yunet_model_path,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.yunet_nms_threshold,
            top_k=args.yunet_top_k,
            max_input_size=args.yunet_max_input_size,
        )
    if args.detector in {"dsfd", "retinaface_resnet50", "retinaface_mobilenet", "retinanet_resnet50", "retinanet_mobilenet"}:
        from src.detection.face_detection_package_detector import FaceDetectionPackageDetector

        backend_names = {
            "dsfd": "DSFDDetector",
            "retinaface_resnet50": "RetinaNetResNet50",
            "retinaface_mobilenet": "RetinaNetMobileNetV1",
            "retinanet_resnet50": "RetinaNetResNet50",
            "retinanet_mobilenet": "RetinaNetMobileNetV1",
        }
        return FaceDetectionPackageDetector(
            backend_name=backend_names[args.detector],
            detector_name=args.detector,
            confidence_threshold=args.confidence_threshold,
            nms_iou_threshold=args.face_detection_nms_iou_threshold,
            device=args.device,
            max_resolution=args.face_detection_max_resolution,
            fp16_inference=not args.face_detection_disable_fp16,
        )
    if args.detector == "retinaface":
        from src.detection.retinaface_detector import RetinaFaceDetector

        return RetinaFaceDetector(threshold=args.confidence_threshold)
    if args.detector == "yolo_scrfd_fallback":
        from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector

        return YOLOSCRFDFallbackDetector(
            yolo_model_path=args.model_path,
            yolo_confidence_threshold=args.confidence_threshold,
            yolo_iou_threshold=args.iou_threshold,
            yolo_device=args.device,
            yolo_image_size=args.image_size,
            yolo_allowed_class_ids=args.allowed_class_id,
            min_face_size_threshold_px=args.min_face_size_threshold_px,
            center_y_threshold=args.center_y_threshold,
            text_score_threshold=args.text_score_threshold,
        )
    if args.detector == "yolo_scrfd_retinaface_selective":
        from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector
        from src.detection.yolo_scrfd_retinaface_selective_detector import YOLOSCRFDRetinaFaceSelectiveDetector

        base_detector = YOLOSCRFDFallbackDetector(
            yolo_model_path=args.model_path,
            yolo_confidence_threshold=args.confidence_threshold,
            yolo_iou_threshold=args.iou_threshold,
            yolo_device=args.device,
            yolo_image_size=args.image_size,
            yolo_allowed_class_ids=args.allowed_class_id,
            min_face_size_threshold_px=args.min_face_size_threshold_px,
            center_y_threshold=args.center_y_threshold,
            text_score_threshold=args.text_score_threshold,
        )
        return YOLOSCRFDRetinaFaceSelectiveDetector(
            base_detector=base_detector,
            retinaface_threshold=args.confidence_threshold,
            trigger_prediction_count_threshold=args.retinaface_trigger_prediction_count_threshold,
            duplicate_iou_threshold=args.retinaface_duplicate_iou_threshold,
            fallback_score_scale=args.retinaface_fallback_score_scale,
        )
    raise ValueError(f"Unsupported detector: {args.detector}")


def save_detections(rows: list[dict[str, object]], output_path: Path) -> None:
    """Atomically save detector predictions as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "x1", "y1", "x2", "y2", "score", "condition_label", "detector_name"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=output_path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def resolve_source_path(default_path: Path, args: argparse.Namespace) -> Path:
    """Resolve the image path, optionally swapping source root/suffix for ablations."""
    if args.image_root is None:
        return default_path
    raw_root = PROJECT_ROOT / "data" / "castle2024" / "raw"
    absolute_default_path = default_path if default_path.is_absolute() else PROJECT_ROOT / default_path
    relative_path = absolute_default_path.relative_to(raw_root)
    if args.image_suffix:
        relative_path = relative_path.with_suffix(args.image_suffix)
    return Path(args.image_root) / relative_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--detector",
        choices=[
            "mtcnn",
            "yolo",
            "retinaface",
            "scrfd",
            "yunet",
            "dsfd",
            "retinaface_resnet50",
            "retinaface_mobilenet",
            "retinanet_resnet50",
            "retinanet_mobilenet",
            "yolo_scrfd_fallback",
            "yolo_scrfd_retinaface_selective",
        ],
        required=True,
    )
    parser.add_argument("--model-path", default="data/models/yolov8n.pt")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--allowed-class-id", type=int, action="append", default=None)
    parser.add_argument("--scrfd-model-path", default="/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx")
    parser.add_argument("--scrfd-input-size", type=int, default=640)
    parser.add_argument("--yunet-model-path", default="data/models/face_detection_yunet_2026may.onnx")
    parser.add_argument("--yunet-nms-threshold", type=float, default=0.3)
    parser.add_argument("--yunet-top-k", type=int, default=5000)
    parser.add_argument("--yunet-max-input-size", type=int, default=1280)
    parser.add_argument("--face-detection-nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--face-detection-max-resolution", type=int, default=1920)
    parser.add_argument("--face-detection-disable-fp16", action="store_true")
    parser.add_argument("--min-face-size-threshold-px", type=float, default=96.0)
    parser.add_argument("--center-y-threshold", type=float, default=0.65)
    parser.add_argument("--text-score-threshold", type=int, default=12)
    parser.add_argument("--retinaface-trigger-prediction-count-threshold", type=int, default=8)
    parser.add_argument("--retinaface-duplicate-iou-threshold", type=float, default=0.5)
    parser.add_argument("--retinaface-fallback-score-scale", type=float, default=0.95)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-root", default=None, help="Optional alternate image root for ablations.")
    parser.add_argument("--image-suffix", default=None, help="Optional suffix swap, e.g. .png.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = CASTLEDataset(args.manifest, return_format="pil", filters={})
    detector = build_detector(args)

    rows: list[dict[str, object]] = []
    image_count = 0
    detection_count = 0
    for item in dataset:
        if args.max_images is not None and image_count >= args.max_images:
            break
        image_count += 1
        image = item["image"]
        if args.image_root is not None:
            image_path = resolve_source_path(Path(item["path"]), args)
            with Image.open(image_path) as loaded:
                image = loaded.copy()
        result = detector.detect(image)
        image_id = item["metadata"]["relative_path"]
        condition_label = item["metadata"].get("condition_label", "")
        for detection in result.detections:
            x1, y1, x2, y2 = detection.box
            rows.append(
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
    save_detections(rows, output_path)
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
