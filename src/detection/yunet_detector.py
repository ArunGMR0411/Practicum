"""OpenCV YuNet face detector wrapper."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult


class YuNetDetector(BaseDetector):
    """Run the OpenCV Zoo YuNet ONNX face detector."""

    detector_name = "yunet"

    def __init__(
        self,
        model_path: str = "data/models/face_detection_yunet_2026may.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_input_size: int = 1280,
    ) -> None:
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_input_size = int(max_input_size)
        self._detector = None

    def _get_detector(self, image_size: tuple[int, int]):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YuNet model not found: {self.model_path}")
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                self.model_path,
                "",
                image_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k,
            )
        else:
            self._detector.setInputSize(image_size)
        return self._detector

    def detect(self, image: Image.Image) -> DetectionResult:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        scale = 1.0
        inference_image = rgb_image
        longest_side = max(width, height)
        if self.max_input_size > 0 and longest_side > self.max_input_size:
            scale = self.max_input_size / float(longest_side)
            resized_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            inference_image = rgb_image.resize(resized_size, Image.Resampling.BILINEAR)

        detector = self._get_detector(inference_image.size)
        started = perf_counter()
        image_bgr = cv2.cvtColor(np.array(inference_image), cv2.COLOR_RGB2BGR)
        _, faces = detector.detect(image_bgr)
        elapsed = perf_counter() - started

        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        if faces is not None:
            for row in faces:
                x, y, w, h = [float(value) / scale for value in row[:4]]
                score = float(row[-1])
                if score < self.confidence_threshold:
                    continue
                boxes.append(
                    (
                        int(round(x)),
                        int(round(y)),
                        int(round(x + w)),
                        int(round(y + h)),
                    )
                )
                confidences.append(score)

        detections = self.normalise_detections(
            image=rgb_image,
            boxes=boxes,
            confidences=confidences,
            per_detection_metadata=[{"detector_stage": "yunet"} for _ in boxes],
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "nms_threshold": self.nms_threshold,
                "top_k": self.top_k,
                "max_input_size": self.max_input_size,
                "inference_scale": round(scale, 6),
                "runtime_seconds": elapsed,
                "raw_count": 0 if faces is None else int(len(faces)),
            },
        )
