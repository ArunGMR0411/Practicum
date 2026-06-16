#!/usr/bin/env python3

"""Interface probes for CASTLE detector wrappers."""

from __future__ import annotations

import unittest

from PIL import Image

from src.detection import (
    BaseDetector,
    DetectionResult,
    YOLODetector,
    RetinaFaceDetector,
    YOLOSCRFDRetinaFaceSelectiveDetector,
)


class DetectionInterfaceTest(unittest.TestCase):
    """Check detector scaffolding behavior without requiring model backends."""

    def test_base_detector_imports(self) -> None:
        """The abstract base should remain importable."""
        self.assertEqual(BaseDetector.detector_name, "base")

    def test_retinaface_wrapper_fails_clearly_without_backend(self) -> None:
        """RetinaFace should raise a clear dependency error if not installed."""
        detector = RetinaFaceDetector()
        detector._backend_available = False
        with self.assertRaises(ModuleNotFoundError):
            detector.detect(Image.new("RGB", (64, 64), color="black"))

    def test_yolo_wrapper_fails_clearly_without_backend(self) -> None:
        """YOLO should raise a clear dependency error if not installed."""
        detector = YOLODetector()
        detector._backend_available = False
        with self.assertRaises(ModuleNotFoundError):
            detector.detect(Image.new("RGB", (64, 64), color="black"))

    def test_detection_result_container(self) -> None:
        """The structured result type should be usable without model inference."""
        result = DetectionResult(detections=[], metadata={"detector": "dummy"})
        self.assertEqual(result.metadata["detector"], "dummy")
        self.assertEqual(result.detections, [])

    def test_selective_retinaface_wrapper_imports(self) -> None:
        """The selective high-accuracy wrapper should remain importable."""
        detector = YOLOSCRFDRetinaFaceSelectiveDetector()
        self.assertEqual(detector.detector_name, "yolo_scrfd_retinaface_selective")


if __name__ == "__main__":
    unittest.main()
