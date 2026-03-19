"""Base abstractions for face anonymisation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from PIL import Image


BoundingBox = tuple[int, int, int, int]


@dataclass
class AnonymiserResult:
    """Container for anonymised image outputs and method metadata."""

    image: Image.Image
    metadata: dict[str, Any]


class BaseAnonymiser(ABC):
    """Abstract interface implemented by every anonymisation method."""

    method_name = "base"

    @abstractmethod
    def anonymise(self, image: Image.Image, boxes: list[BoundingBox]) -> AnonymiserResult:
        """Return an anonymised image and method metadata for the supplied face boxes."""

    def validate_boxes(self, image: Image.Image, boxes: list[BoundingBox]) -> list[BoundingBox]:
        """Clamp supplied boxes to the image bounds and discard degenerate entries."""
        width, height = image.size
        valid_boxes: list[BoundingBox] = []
        for x1, y1, x2, y2 in boxes:
            left = max(0, min(int(x1), width))
            top = max(0, min(int(y1), height))
            right = max(0, min(int(x2), width))
            bottom = max(0, min(int(y2), height))
            if right <= left or bottom <= top:
                continue
            valid_boxes.append((left, top, right, bottom))
        return valid_boxes
