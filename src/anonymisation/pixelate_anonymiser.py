"""Pixelation baseline anonymiser."""

from __future__ import annotations

from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser


class PixelateAnonymiser(BaseAnonymiser):
    """Pixelate each supplied face crop with a coarse resize-down/up pass."""

    method_name = "pixelate"

    def __init__(self, scale_factor: int = 12) -> None:
        self.scale_factor = max(2, scale_factor)

    def anonymise(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> AnonymiserResult:
        """Pixelate each validated face crop and return the modified image."""
        output = image.copy()
        valid_boxes = self.validate_boxes(output, boxes)
        for left, top, right, bottom in valid_boxes:
            region = output.crop((left, top, right, bottom))
            width, height = region.size
            downsampled = region.resize(
                (max(1, width // self.scale_factor), max(1, height // self.scale_factor)),
                resample=Image.Resampling.BILINEAR,
            )
            pixelated = downsampled.resize((width, height), resample=Image.Resampling.NEAREST)
            output.paste(pixelated, (left, top))
        return AnonymiserResult(
            image=output,
            metadata={
                "method": self.method_name,
                "scale_factor": self.scale_factor,
                "boxes_processed": len(valid_boxes),
                "tiling_required": False,
            },
        )
