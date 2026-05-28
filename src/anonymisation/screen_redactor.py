"""Region redactor for detected screen boxes."""

from __future__ import annotations

from PIL import Image, ImageFilter

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser


class ScreenRedactor(BaseAnonymiser):
    """Redact detected screens using blur or solid fill."""

    method_name = "screen_redactor"

    def __init__(self, mode: str = "blur", blur_radius: float = 18.0, fill_color: tuple[int, int, int] = (0, 0, 0)) -> None:
        if mode not in {"blur", "fill"}:
            raise ValueError("mode must be 'blur' or 'fill'")
        self.mode = mode
        self.blur_radius = blur_radius
        self.fill_color = fill_color

    def anonymise(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> AnonymiserResult:
        output = image.copy()
        valid_boxes = self.validate_boxes(output, boxes)
        for left, top, right, bottom in valid_boxes:
            if self.mode == "blur":
                region = output.crop((left, top, right, bottom))
                redacted = region.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
                output.paste(redacted, (left, top))
            else:
                fill_region = Image.new("RGB", (right - left, bottom - top), self.fill_color)
                output.paste(fill_region, (left, top))
        return AnonymiserResult(
            image=output,
            metadata={
                "method": self.method_name,
                "mode": self.mode,
                "boxes_processed": len(valid_boxes),
                "tiling_required": False,
            },
        )
