"""Standalone SCRFD detector wrapper using InsightFace ONNX Runtime backends."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult

try:
    from insightface.model_zoo import get_model
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    get_model = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    ort = None  # type: ignore[assignment]


class SCRFDDetector(BaseDetector):
    """Run SCRFD as a standalone face detector."""

    detector_name = "scrfd"

    def __init__(
        self,
        model_path: str = "/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx",
        confidence_threshold: float = 0.25,
        input_size: tuple[int, int] = (640, 640),
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.input_size = tuple(int(value) for value in input_size)
        self.providers = list(providers or self._default_providers())
        self._model: Any | None = None

    @staticmethod
    def _default_providers() -> list[str]:
        if ort is None:
            return ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get_model(self) -> Any:
        if get_model is None:
            raise ModuleNotFoundError("SCRFDDetector requires the 'insightface' package.")
        if self._model is None:
            detector = get_model(self.model_path, providers=self.providers)
            detector.prepare(ctx_id=0, input_size=self.input_size)
            self._model = detector
        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        model = self._get_model()
        rgb_image = image.convert("RGB")
        started = perf_counter()
        image_bgr = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        faces, _ = model.detect(image_bgr, input_size=self.input_size, max_num=0, metric="default")
        elapsed = perf_counter() - started

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        if faces is not None:
            for face in faces:
                score = float(face[4]) if len(face) >= 5 else 1.0
                if score < self.confidence_threshold:
                    continue
                boxes.append(tuple(int(value) for value in face[:4]))
                confidences.append(score)

        detections = self.normalise_detections(
            image=rgb_image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=[{"detector_stage": "scrfd"} for _ in boxes],
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "input_size": self.input_size,
                "providers": self.providers,
                "runtime_seconds": elapsed,
                "raw_count": 0 if faces is None else int(len(faces)),
            },
        )
