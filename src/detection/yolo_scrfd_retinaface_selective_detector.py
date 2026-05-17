"""Three-stage face detector: YOLO primary, SCRFD fallback, RetinaFace selective refinement."""

from __future__ import annotations

from time import perf_counter

from PIL import Image

from src.detection.base_detector import BaseDetector, Detection, DetectionResult
from src.detection.retinaface_detector import RetinaFaceDetector
from src.detection.yolo_scrfd_fallback_detector import YOLOSCRFDFallbackDetector


class YOLOSCRFDRetinaFaceSelectiveDetector(BaseDetector):
    """Use RetinaFace only on images where the operational detector returns few faces."""

    detector_name = "yolo_scrfd_retinaface_selective"

    def __init__(
        self,
        base_detector: YOLOSCRFDFallbackDetector | None = None,
        retinaface_threshold: float = 0.5,
        trigger_prediction_count_threshold: int = 8,
        duplicate_iou_threshold: float = 0.5,
        fallback_score_scale: float = 0.95,
    ) -> None:
        self.base_detector = base_detector or YOLOSCRFDFallbackDetector()
        self.retinaface_detector = RetinaFaceDetector(threshold=retinaface_threshold)
        self.trigger_prediction_count_threshold = int(trigger_prediction_count_threshold)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.fallback_score_scale = float(fallback_score_scale)

    @staticmethod
    def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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

    def detect(self, image: Image.Image) -> DetectionResult:
        started = perf_counter()
        base_result = self.base_detector.detect(image)
        merged_detections = list(base_result.detections)
        retinaface_added = 0
        retinaface_triggered = len(merged_detections) <= self.trigger_prediction_count_threshold

        if retinaface_triggered:
            retinaface_result = self.retinaface_detector.detect(image)
            for candidate in retinaface_result.detections:
                if any(self._compute_iou(candidate.box, existing.box) >= self.duplicate_iou_threshold for existing in merged_detections):
                    continue
                merged_detections.append(
                    Detection(
                        box=candidate.box,
                        confidence=float(candidate.confidence) * self.fallback_score_scale,
                        metadata={**candidate.metadata, "detector_stage": "retinaface_selective_merged"},
                    )
                )
                retinaface_added += 1

        elapsed = perf_counter() - started
        return DetectionResult(
            detections=merged_detections,
            metadata={
                "detector": self.detector_name,
                "runtime_seconds": elapsed,
                "base_detector": self.base_detector.detector_name,
                "trigger_prediction_count_threshold": self.trigger_prediction_count_threshold,
                "duplicate_iou_threshold": self.duplicate_iou_threshold,
                "fallback_score_scale": self.fallback_score_scale,
                "base_detection_count": len(base_result.detections),
                "final_detection_count": len(merged_detections),
                "retinaface_triggered": retinaface_triggered,
                "retinaface_added_count": retinaface_added,
                **base_result.metadata,
            },
        )
