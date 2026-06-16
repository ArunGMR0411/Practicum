"""Unit tests for OCR evaluator helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ocr_evaluator import OCREvaluator, OCRRegionResult


class TestOCREvaluatorHelpers(unittest.TestCase):
    def test_text_similarity_identity(self) -> None:
        self.assertAlmostEqual(OCREvaluator.text_similarity("castle", "castle"), 1.0)

    def test_text_similarity_empty_cases(self) -> None:
        self.assertEqual(OCREvaluator.text_similarity("", "text"), 0.0)
        self.assertEqual(OCREvaluator.text_similarity("", ""), 1.0)

    def test_suppression_rate(self) -> None:
        original = [
            OCRRegionResult((0, 0, 10, 10), "hello"),
            OCRRegionResult((0, 0, 10, 10), "world"),
        ]
        redacted = [
            OCRRegionResult((0, 0, 10, 10), ""),
            OCRRegionResult((0, 0, 10, 10), "world"),
        ]
        rate = OCREvaluator.suppression_rate(original, redacted, similarity_threshold=0.5)
        self.assertAlmostEqual(rate, 0.5)

    def test_paired_accuracy_summary(self) -> None:
        original = [OCRRegionResult((0, 0, 10, 10), "abcd")]
        redacted = [OCRRegionResult((0, 0, 10, 10), "ab")]
        summary = OCREvaluator.paired_accuracy_summary(original, redacted)
        self.assertIn("mean_similarity", summary)
        self.assertGreater(summary["mean_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
