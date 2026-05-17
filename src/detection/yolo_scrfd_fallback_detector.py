"""Two-pass face detector: YOLO primary with selective SCRFD fallback."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, Detection, DetectionResult
from src.detection.yolo_detector import YOLODetector
from src.data.subset_building import detect_text_score, resize_for_analysis

try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    get_model = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    ort = None  # type: ignore[assignment]


class YOLOSCRFDFallbackDetector(BaseDetector):
    """Run YOLO first, then selectively add SCRFD detections on hard frames."""

    detector_name = "yolo_scrfd_fallback"

    def __init__(
        self,
        yolo_model_path: str = "data/models/yolov9c.pt",
        yolo_confidence_threshold: float = 0.25,
        yolo_iou_threshold: float = 0.5,
        yolo_device: str | None = None,
        yolo_image_size: int | None = None,
        yolo_allowed_class_ids: list[int] | None = None,
        scrfd_model_path: str = "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx",
        scrfd_input_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
        min_face_size_threshold_px: float = 96.0,
        center_y_threshold: float = 0.65,
        text_score_threshold: int = 12,
        duplicate_iou_threshold: float = 0.5,
        fallback_score_scale: float = 0.95,
    ) -> None:
        self.primary_detector = YOLODetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence_threshold,
            iou_threshold=yolo_iou_threshold,
            device=yolo_device,
            image_size=yolo_image_size,
            allowed_class_ids=yolo_allowed_class_ids,
        )
        self.scrfd_model_path = str(scrfd_model_path)
        self.scrfd_input_size = tuple(int(value) for value in scrfd_input_size)
        self.providers = list(providers or self._default_providers())
        self.min_face_size_threshold_px = float(min_face_size_threshold_px)
        self.center_y_threshold = float(center_y_threshold)
        self.text_score_threshold = int(text_score_threshold)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.fallback_score_scale = float(fallback_score_scale)
        self._scrfd_model: Any | None = None

    @staticmethod
    def _default_providers() -> list[str]:
        """Prefer GPU execution for SCRFD whenever the environment supports it."""
        if ort is None:
            return ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get_scrfd_model(self) -> Any:
        if get_model is None:
            raise ModuleNotFoundError(
                "YOLOSCRFDFallbackDetector requires the 'insightface' package, which is not installed in the current environment."
            )
        if self._scrfd_model is None:
            detector = get_model(self.scrfd_model_path, providers=self.providers)
            detector.prepare(ctx_id=0, input_size=self.scrfd_input_size)
            self._scrfd_model = detector
        return self._scrfd_model

    def _compute_trigger_signals(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> dict[str, float | int | bool]:
        face_boxes = [item.box for item in detections]
        min_face_size_px = 0.0
        center_y_ratio = 0.0
        if face_boxes:
            min_face_size_px = min(float(min(x2 - x1, y2 - y1)) for x1, y1, x2, y2 in face_boxes)
            dominant_box = max(face_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            _, y1, _, y2 = dominant_box
            center_y_ratio = ((float(y1) + float(y2)) / 2.0) / max(image.height, 1)

        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        resized = resize_for_analysis(image_bgr, analysis_width=192)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        text_score = int(detect_text_score(gray))

        small_face_like = 0.0 < min_face_size_px <= self.min_face_size_threshold_px
        downward_like = center_y_ratio >= self.center_y_threshold
        text_like = text_score >= self.text_score_threshold
        should_trigger = bool(small_face_like or downward_like or text_like)
        return {
            "min_detected_face_size_px": round(float(min_face_size_px), 3),
            "dominant_center_y_ratio": round(float(center_y_ratio), 6),
            "text_score": text_score,
            "small_face_like": bool(small_face_like),
            "downward_like": bool(downward_like),
            "text_like": bool(text_like),
            "fallback_triggered": should_trigger,
        }

    def _compute_iou(self, box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return 0.0 if union <= 0 else inter_area / union

    def _run_scrfd(self, image: Image.Image) -> list[Detection]:
        detector = self._get_scrfd_model()
        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces, _ = detector.detect(image_bgr, input_size=self.scrfd_input_size, max_num=0, metric="default")
        if faces is None:
            return []
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        for face in faces:
            boxes.append(tuple(int(value) for value in face[:4]))
            confidences.append(float(face[4]) if len(face) >= 5 else 1.0)
        return self.normalise_detections(
            image=image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=[{"detector_stage": "scrfd_fallback"} for _ in boxes],
        )

    def detect(self, image: Image.Image) -> DetectionResult:
        started = perf_counter()
        primary_result = self.primary_detector.detect(image)
        merged_detections = list(primary_result.detections)
        trigger_signals = self._compute_trigger_signals(image, merged_detections)
        scrfd_added = 0

        if bool(trigger_signals["fallback_triggered"]):
            for candidate in self._run_scrfd(image):
                if any(self._compute_iou(candidate.box, existing.box) >= self.duplicate_iou_threshold for existing in merged_detections):
                    continue
                merged_detections.append(
                    Detection(
                        box=candidate.box,
                        confidence=float(candidate.confidence) * self.fallback_score_scale,
                        metadata={**candidate.metadata, "detector_stage": "scrfd_fallback_merged"},
                    )
                )
                scrfd_added += 1

        elapsed = perf_counter() - started
        return DetectionResult(
            detections=merged_detections,
            metadata={
                "detector": self.detector_name,
                "runtime_seconds": elapsed,
                "primary_detector": self.primary_detector.detector_name,
                "primary_detection_count": len(primary_result.detections),
                "final_detection_count": len(merged_detections),
                "scrfd_added_count": scrfd_added,
                **trigger_signals,
            },
        )
