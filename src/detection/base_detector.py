"""Abstract base classes and shared result containers for face detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    """One detector prediction with a bounding box, score, and optional metadata."""

    box: BoundingBox
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    """Container for one detector invocation and its predictions."""

    detections: list[Detection]
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Abstract detector interface used by all face-detection backends."""

    detector_name = "base"

    @abstractmethod
    def detect(self, image: Image.Image) -> DetectionResult:
        """Run face detection on a PIL image and return a structured result."""

    def clamp_box(self, image: Image.Image, box: BoundingBox) -> BoundingBox | None:
        """Clamp one box to image bounds and drop degenerate coordinates."""
        width, height = image.size
        x1, y1, x2, y2 = box
        left = max(0, min(int(x1), width))
        top = max(0, min(int(y1), height))
        right = max(0, min(int(x2), width))
        bottom = max(0, min(int(y2), height))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def normalise_detections(
        self,
        image: Image.Image,
        boxes: list[BoundingBox],
        confidences: list[float] | None = None,
        per_detection_metadata: list[dict[str, Any]] | None = None,
    ) -> list[Detection]:
        """Convert raw detector outputs into validated structured detections."""
        scores = confidences if confidences is not None else [1.0] * len(boxes)
        metadata_rows = (
            per_detection_metadata
            if per_detection_metadata is not None
            else [{} for _ in range(len(boxes))]
        )
        detections: list[Detection] = []
        for box, score, metadata in zip(boxes, scores, metadata_rows, strict=False):
            clamped = self.clamp_box(image, box)
            if clamped is None:
                continue
            detections.append(
                Detection(
                    box=clamped,
                    confidence=float(score),
                    metadata=dict(metadata),
                )
            )
        return detections
