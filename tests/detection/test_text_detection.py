"""Tests for the text detector wrapper."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.base_detector import DetectionResult
from src.detection.text_detector import TextDetector


class TestTextDetection(unittest.TestCase):
    def test_invalid_backend_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TextDetector(backend="invalid", device="cpu")

    def test_text_detector_sets_backend_specific_name(self) -> None:
        detector = TextDetector(device="cpu")
        self.assertEqual(detector.backend, "easyocr")
        self.assertEqual(detector.detector_name, "craft_easyocr")

    def test_text_detector_import_and_detect(self) -> None:
        detector = TextDetector(device="cpu")
        image = Image.new("RGB", (320, 160), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((40, 50), "DCU TEST", fill="black")
        result = detector.detect(image)
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.detections, list)
        self.assertEqual(result.metadata["backend"], "easyocr")
        self.assertEqual(result.metadata["device"], "cpu")

    def test_box_iou_zero_for_disjoint_boxes(self) -> None:
        self.assertEqual(
            TextDetector._box_iou((0, 0, 10, 10), (20, 20, 30, 30)),
            0.0,
        )

    def test_merge_boxes_removes_duplicate_overlap(self) -> None:
        detector = TextDetector(device="cpu")
        boxes, metadata = detector._merge_boxes(
            [(0, 0, 20, 20), (1, 1, 19, 19), (30, 30, 40, 40)],
            [{"id": 1}, {"id": 2}, {"id": 3}],
            iou_threshold=0.5,
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(metadata), 2)


if __name__ == "__main__":
    unittest.main()
