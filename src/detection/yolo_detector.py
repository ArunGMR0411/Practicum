"""Optional YOLO detector wrapper used for the primary face-detection candidate."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    YOLO = None  # type: ignore[assignment]


class YOLODetector(BaseDetector):
    """Wrap an optional Ultralytics YOLO model behind the shared detector interface."""

    detector_name = "yolo"

    def __init__(
        self,
        model_path: str = "data/models/yolov9c.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str | None = None,
        image_size: int | None = None,
        allowed_class_ids: list[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.image_size = None if image_size is None else int(image_size)
        self.allowed_class_ids = None if allowed_class_ids is None else {int(value) for value in allowed_class_ids}
        self._backend_available = YOLO is not None
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Initialise the YOLO backend lazily."""
        if not self._backend_available:
            raise ModuleNotFoundError(
                "YOLODetector requires the 'ultralytics' package, which is not installed in the current environment."
            )
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run YOLO detection on a PIL image and return structured detections."""
        model = self._get_model()
        rgb_image = image.convert("RGB")
        started = perf_counter()
        predict_kwargs = {
            "source": rgb_image,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "device": self.device,
            "verbose": False,
        }
        if self.image_size is not None:
            predict_kwargs["imgsz"] = self.image_size
        prediction = model.predict(
            **predict_kwargs,
        )
        elapsed = perf_counter() - started

        if not prediction:
            return DetectionResult(
                detections=[],
                metadata={
                    "detector": self.detector_name,
                    "model_path": self.model_path,
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                    "runtime_seconds": elapsed,
                    "backend_available": True,
                    "raw_count": 0,
                },
            )

        result = prediction[0]
        boxes_tensor = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
        filtered_boxes: list[tuple[int, int, int, int]] = []
        filtered_confidences: list[float] = []
        per_detection_metadata: list[dict[str, int]] = []
        for box, score, class_id in zip(boxes_tensor, confidences, classes, strict=False):
            class_id_int = int(class_id)
            if self.allowed_class_ids is not None and class_id_int not in self.allowed_class_ids:
                continue
            filtered_boxes.append(tuple(map(int, box)))
            filtered_confidences.append(float(score))
            per_detection_metadata.append({"class_id": class_id_int})

        detections = self.normalise_detections(
            rgb_image,
            boxes=filtered_boxes,
            confidences=filtered_confidences,
            per_detection_metadata=per_detection_metadata,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector": self.detector_name,
                "model_path": self.model_path,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "runtime_seconds": elapsed,
                "backend_available": True,
                "raw_count": len(boxes_tensor),
                "filtered_count": len(filtered_boxes),
                "allowed_class_ids": None if self.allowed_class_ids is None else sorted(self.allowed_class_ids),
                "image_size": self.image_size,
            },
        )
