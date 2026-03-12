"""GPU-backed detectors from the `face-detection` package."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult

try:
    import face_detection
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    face_detection = None  # type: ignore[assignment]


class FaceDetectionPackageDetector(BaseDetector):
    """Wrap DSFD and RetinaNet face detectors behind the common interface."""

    def __init__(
        self,
        backend_name: str = "DSFDDetector",
        detector_name: str | None = None,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = "cuda",
        max_resolution: int | None = 1920,
        fp16_inference: bool = True,
    ) -> None:
        self.backend_name = backend_name
        self.detector_name = detector_name or backend_name.lower()
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.device = device
        self.max_resolution = max_resolution
        self.fp16_inference = bool(fp16_inference)
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if face_detection is None:
            raise ModuleNotFoundError("FaceDetectionPackageDetector requires the 'face_detection' package.")
        if self._model is None:
            self._model = face_detection.build_detector(
                name=self.backend_name,
                confidence_threshold=self.confidence_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
                device=self.device,
                max_resolution=self.max_resolution,
                fp16_inference=self.fp16_inference,
                clip_boxes=True,
            )
        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        model = self._get_model()
        rgb_image = image.convert("RGB")
        started = perf_counter()
        faces = model.detect(np.array(rgb_image))
        elapsed = perf_counter() - started

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        for face in faces:
            score = float(face[4]) if len(face) >= 5 else 1.0
            if score < self.confidence_threshold:
                continue
            boxes.append(tuple(int(round(float(value))) for value in face[:4]))
            confidences.append(score)

        detections = self.normalise_detections(
            image=rgb_image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=[{"detector_stage": self.detector_name} for _ in boxes],
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "backend_name": self.backend_name,
                "confidence_threshold": self.confidence_threshold,
                "nms_iou_threshold": self.nms_iou_threshold,
                "device": self.device,
                "max_resolution": self.max_resolution,
                "fp16_inference": self.fp16_inference,
                "runtime_seconds": elapsed,
                "raw_count": int(len(faces)),
            },
        )
