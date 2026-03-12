"""Optional RetinaFace wrapper used as the baseline face detector."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult

try:
    from retinaface import RetinaFace
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    RetinaFace = None  # type: ignore[assignment]


class RetinaFaceDetector(BaseDetector):
    """Wrap the optional RetinaFace package behind the shared detector interface."""

    detector_name = "retinaface"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)
        self._backend_available = RetinaFace is not None

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run RetinaFace detection on a PIL image and return structured detections."""
        if not self._backend_available:
            raise ModuleNotFoundError(
                "RetinaFaceDetector requires the 'retinaface' package, which is not installed in the current environment."
            )

        rgb_image = image.convert("RGB")
        array = np.array(rgb_image)
        started = perf_counter()
        raw_result = RetinaFace.detect_faces(array, threshold=self.threshold)
        elapsed = perf_counter() - started

        if not raw_result or (isinstance(raw_result, dict) and "error" in raw_result):
            return DetectionResult(
                detections=[],
                metadata={
                    "detector": self.detector_name,
                    "threshold": self.threshold,
                    "runtime_seconds": elapsed,
                    "backend_available": True,
                    "raw_count": 0,
                },
            )

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        per_detection_metadata: list[dict[str, Any]] = []

        for item in raw_result.values():
            x1, y1, x2, y2 = item["facial_area"]
            boxes.append((x1, y1, x2, y2))
            confidences.append(float(item.get("score", 1.0)))
            per_detection_metadata.append(
                {
                    "landmarks": item.get("landmarks", {}),
                }
            )

        detections = self.normalise_detections(
            rgb_image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=per_detection_metadata,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "threshold": self.threshold,
                "runtime_seconds": elapsed,
                "backend_available": True,
                "raw_count": len(raw_result),
            },
        )
