"""Tests for the screen detector wrapper."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.base_detector import DetectionResult
from src.detection.screen_detector import ScreenDetector


class TestScreenDetection(unittest.TestCase):
    def test_screen_detector_instantiation(self) -> None:
        detector = ScreenDetector(model_path="data/models/yolov8n.pt", device="cpu")
        self.assertEqual(detector.detector_name, "screen_yolo_fallback")

    def test_screen_detector_custom_class_ids_are_retained(self) -> None:
        detector = ScreenDetector(model_path="data/models/yolov8n.pt", device="cpu", class_ids=(63,))
        self.assertEqual(detector.class_ids, (63,))

    def test_screen_detector_detect_empty(self) -> None:
        detector = ScreenDetector(model_path="data/models/yolov8n.pt", device="cpu")
        image = Image.new("RGB", (224, 224), color="white")
        result = detector.detect(image)
        self.assertIsInstance(result, DetectionResult)
        self.assertIsInstance(result.detections, list)
        self.assertEqual(result.metadata["class_ids"], detector.class_ids)


if __name__ == "__main__":
    unittest.main()
