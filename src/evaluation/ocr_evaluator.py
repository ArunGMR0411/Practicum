"""OCR-based redaction evaluation using TrOCR."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


BoundingBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRRegionResult:
    """OCR output for one detected region."""

    box: BoundingBox
    text: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None


class OCREvaluator:
    """Evaluate text recognisability before and after redaction with TrOCR."""

    def __init__(
        self,
        model_name: str = "microsoft/trocr-small-printed",
        device: str = "cuda",
        max_new_tokens: int = 64,
        region_batch_size: int = 8,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.region_batch_size = max(1, int(region_batch_size))

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def _prepare_crop(self, image: Image.Image, box: BoundingBox) -> Image.Image | None:
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image.crop((x1, y1, x2, y2)).convert("RGB")
        if crop.width < 4 or crop.height < 4:
            return None
        return crop

    @torch.no_grad()
    def recognise_region(self, image: Image.Image, box: BoundingBox) -> OCRRegionResult:
        return self.recognise_regions(image, [box])[0]

    def recognise_regions(
        self, image: Image.Image, boxes: list[BoundingBox]
    ) -> list[OCRRegionResult]:
        if not boxes:
            return []

        indexed_crops: list[tuple[int, BoundingBox, Image.Image | None]] = [
            (index, box, self._prepare_crop(image, box)) for index, box in enumerate(boxes)
        ]
        results: list[OCRRegionResult | None] = [None] * len(boxes)

        for start in range(0, len(indexed_crops), self.region_batch_size):
            batch = indexed_crops[start : start + self.region_batch_size]
            valid = [(index, box, crop) for index, box, crop in batch if crop is not None]
            for index, box, crop in batch:
                if crop is None:
                    results[index] = OCRRegionResult(box=box, text="", confidence=None, metadata={"skipped": True})

            if not valid:
                continue

            valid_indices = [index for index, _box, _crop in valid]
            valid_boxes = [box for _index, box, _crop in valid]
            valid_crops = [crop for _index, _box, crop in valid]
            pixel_values = self.processor(images=valid_crops, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(pixel_values, max_new_tokens=self.max_new_tokens)
            texts = [text.strip() for text in self.processor.batch_decode(generated_ids, skip_special_tokens=True)]

            for index, box, text in zip(valid_indices, valid_boxes, texts, strict=True):
                results[index] = OCRRegionResult(
                    box=box,
                    text=text,
                    confidence=None,
                    metadata={"model_name": self.model_name},
                )

        return [result for result in results if result is not None]

    @staticmethod
    def text_similarity(reference: str, candidate: str) -> float:
        """Return a normalized character similarity in [0, 1]."""
        reference = (reference or "").strip()
        candidate = (candidate or "").strip()
        if not reference and not candidate:
            return 1.0
        if not reference or not candidate:
            return 0.0
        return float(SequenceMatcher(a=reference, b=candidate).ratio())

    @classmethod
    def suppression_rate(
        cls,
        original_results: list[OCRRegionResult],
        redacted_results: list[OCRRegionResult],
        similarity_threshold: float = 0.5,
    ) -> float:
        """Fraction of regions whose OCR readability is suppressed below threshold."""
        if len(original_results) != len(redacted_results):
            raise ValueError("original_results and redacted_results must have the same length")
        if not original_results:
            return 0.0

        suppressed = 0
        for original, redacted in zip(original_results, redacted_results, strict=True):
            similarity = cls.text_similarity(original.text, redacted.text)
            if similarity < similarity_threshold:
                suppressed += 1
        return float(suppressed / len(original_results))

    @classmethod
    def paired_accuracy_summary(
        cls,
        original_results: list[OCRRegionResult],
        redacted_results: list[OCRRegionResult],
    ) -> dict[str, float]:
        """Summarise pre/post OCR readability with normalized similarity scores."""
        if len(original_results) != len(redacted_results):
            raise ValueError("original_results and redacted_results must have the same length")
        similarities = [
            cls.text_similarity(original.text, redacted.text)
            for original, redacted in zip(original_results, redacted_results, strict=True)
        ]
        if not similarities:
            return {
                "mean_similarity": 0.0,
                "median_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            }
        values = np.array(similarities, dtype=np.float32)
        return {
            "mean_similarity": float(values.mean()),
            "median_similarity": float(np.median(values)),
            "min_similarity": float(values.min()),
            "max_similarity": float(values.max()),
        }
