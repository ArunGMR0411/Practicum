"""Optional MTCNN detector wrapper used for face-sensitive subset and routing analysis."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult
from src.utils.system_config import resolve_torch_device

try:
    from facenet_pytorch import MTCNN
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    MTCNN = None  # type: ignore[assignment]


class MTCNNDetector(BaseDetector):
    """Wrap facenet-pytorch MTCNN behind the shared detector interface."""

    detector_name = "mtcnn"

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        device: str | None = None,
        keep_all: bool = True,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.device = device or resolve_torch_device()
        self.keep_all = bool(keep_all)
        self._backend_available = MTCNN is not None
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if not self._backend_available:
            raise ModuleNotFoundError(
                "MTCNNDetector requires the 'facenet_pytorch' package, which is not installed in the current environment."
            )
        if self._model is None:
            self._model = MTCNN(keep_all=self.keep_all, device=self.device)
        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        model = self._get_model()
        rgb_image = image.convert("RGB")
        started = perf_counter()
        boxes, probs, landmarks = model.detect(rgb_image, landmarks=True)
        elapsed = perf_counter() - started

        if boxes is None or probs is None:
            return DetectionResult(
                detections=[],
                metadata={
                    "detector": self.detector_name,
                    "confidence_threshold": self.confidence_threshold,
                    "device": self.device,
                    "runtime_seconds": elapsed,
                    "backend_available": True,
                    "raw_count": 0,
                },
            )

        filtered_boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        per_detection_metadata: list[dict[str, Any]] = []
        for index, (box, prob) in enumerate(zip(boxes, probs, strict=False)):
            if prob is None or float(prob) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in box]
            filtered_boxes.append((x1, y1, x2, y2))
            confidences.append(float(prob))
            landmark_row = landmarks[index].tolist() if landmarks is not None else []
            per_detection_metadata.append({"landmarks": landmark_row})

        detections = self.normalise_detections(
            rgb_image,
            boxes=filtered_boxes,
            confidences=confidences,
            per_detection_metadata=per_detection_metadata,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "confidence_threshold": self.confidence_threshold,
                "device": self.device,
                "runtime_seconds": elapsed,
                "backend_available": True,
                "raw_count": 0 if boxes is None else int(len(boxes)),
            },
        )
