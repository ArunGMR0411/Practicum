"""Gaussian-blur baseline anonymiser."""

from __future__ import annotations

from PIL import Image, ImageFilter

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser


class BlurAnonymiser(BaseAnonymiser):
    """Apply Gaussian blur to each supplied face region."""

    method_name = "blur"

    def __init__(self, radius: float = 16.0) -> None:
        self.radius = radius

    def anonymise(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> AnonymiserResult:
        """Blur each validated face crop and return the modified image."""
        output = image.copy()
        valid_boxes = self.validate_boxes(output, boxes)
        for left, top, right, bottom in valid_boxes:
            region = output.crop((left, top, right, bottom))
            blurred = region.filter(ImageFilter.GaussianBlur(radius=self.radius))
            output.paste(blurred, (left, top))
        return AnonymiserResult(
            image=output,
            metadata={
                "method": self.method_name,
                "radius": self.radius,
                "boxes_processed": len(valid_boxes),
                "tiling_required": False,
            },
        )
