"""Screen detector using a practical YOLO fallback over screen-like COCO classes."""

from __future__ import annotations

from typing import Any

from PIL import Image

from src.detection.base_detector import BaseDetector, DetectionResult


class ScreenDetector(BaseDetector):
    """Detect screen-like objects using Ultralytics YOLO as the current fallback."""

    detector_name = "screen_yolo_fallback"
    DEFAULT_SCREEN_CLASS_IDS = (62, 63, 67)  # tv/monitor, laptop, cell phone

    def __init__(
        self,
        model_path: str = "data/models/yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = "0",
        class_ids: tuple[int, ...] | None = None,
        image_size: int = 640,
        half_precision: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ScreenDetector requires the 'ultralytics' package. Install it with "
                "`pip install ultralytics`."
            ) from exc

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_ids = tuple(class_ids or self.DEFAULT_SCREEN_CLASS_IDS)
        self.image_size = int(image_size)
        self.half_precision = bool(half_precision)
        self._model = YOLO(model_path)

    def detect(self, image: Image.Image) -> DetectionResult:
        predictions = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=list(self.class_ids),
            imgsz=self.image_size,
            half=self.half_precision,
            verbose=False,
        )
        if not predictions:
            return DetectionResult(
                detections=[],
                metadata={"detector_name": self.detector_name, "class_ids": self.class_ids},
            )

        result = predictions[0]
        boxes: list[tuple[int, int, int, int]] = []
        scores: list[float] = []
        metadata_rows: list[dict[str, Any]] = []

        names = getattr(result, "names", {})
        for raw_box in result.boxes:
            class_id = int(raw_box.cls.item())
            if class_id not in self.class_ids:
                continue
            x1, y1, x2, y2 = raw_box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            scores.append(float(raw_box.conf.item()))
            metadata_rows.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                }
            )

        detections = self.normalise_detections(
            image=image,
            boxes=boxes,
            confidences=scores,
            per_detection_metadata=metadata_rows,
        )
        return DetectionResult(
            detections=detections,
            metadata={
                "detector_name": self.detector_name,
                "model_path": self.model_path,
                "class_ids": self.class_ids,
                "image_size": self.image_size,
                "half_precision": self.half_precision,
            },
        )
