"""Tests for OCR evaluator batching behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from src.evaluation.ocr_evaluator import OCREvaluator


class _FakeTensor:
    def to(self, _device):
        return self


class _FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        class _Payload:
            pixel_values = _FakeTensor()

        assert return_tensors == "pt"
        self.last_batch_size = len(images)
        return _Payload()

    def batch_decode(self, generated_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return [f"text_{index}" for index, _ in enumerate(generated_ids)]


class _FakeModel:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def generate(self, _pixel_values, max_new_tokens=64):
        assert max_new_tokens == 64
        return [0, 1]


class OCRBatchingTest(unittest.TestCase):
    @patch("src.evaluation.ocr_evaluator.VisionEncoderDecoderModel.from_pretrained", return_value=_FakeModel())
    @patch("src.evaluation.ocr_evaluator.TrOCRProcessor.from_pretrained", return_value=_FakeProcessor())
    def test_recognise_regions_batches_and_preserves_order(self, *_mocks) -> None:
        evaluator = OCREvaluator(device="cpu", region_batch_size=2)
        image = Image.new("RGB", (64, 64), color=(255, 255, 255))
        boxes = [(0, 0, 16, 16), (16, 0, 32, 16)]
        results = evaluator.recognise_regions(image, boxes)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].box, boxes[0])
        self.assertEqual(results[1].box, boxes[1])
        self.assertEqual(results[0].text, "text_0")
        self.assertEqual(results[1].text, "text_1")


if __name__ == "__main__":
    unittest.main()
