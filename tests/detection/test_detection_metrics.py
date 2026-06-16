#!/usr/bin/env python3

"""Known-answer tests for CASTLE detection metrics."""

from __future__ import annotations

import unittest

from src.evaluation.detection_metrics import (
    GroundTruthBox,
    ScoredBox,
    compute_average_precision,
    compute_iou,
    match_detections,
)


class DetectionMetricsTest(unittest.TestCase):
    """Validate core detection metrics on simple synthetic cases."""

    def test_iou_known_answer(self) -> None:
        """IoU should match the expected overlap ratio."""
        iou = compute_iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(iou, 25 / 175)

    def test_average_precision_known_answer(self) -> None:
        """A perfect top-ranked hit followed by one false positive should yield AP 1.0."""
        ground_truths = [GroundTruthBox(image_id="img1", box=(0, 0, 10, 10))]
        predictions = [
            ScoredBox(image_id="img1", box=(0, 0, 10, 10), score=0.9),
            ScoredBox(image_id="img1", box=(20, 20, 30, 30), score=0.1),
        ]
        metrics = compute_average_precision(predictions, ground_truths, iou_threshold=0.5)
        self.assertAlmostEqual(metrics["ap"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 2 * 0.5 * 1.0 / 1.5)

    def test_match_detections_greedy_one_gt(self) -> None:
        ground_truths = [GroundTruthBox(image_id="img1", box=(0, 0, 10, 10))]
        predictions = [
            ScoredBox(image_id="img1", box=(0, 0, 10, 10), score=0.9),
            ScoredBox(image_id="img1", box=(0, 0, 10, 10), score=0.8),
        ]
        matches = match_detections(predictions, ground_truths, iou_threshold=0.5)
        self.assertTrue(matches[0]["true_positive"])
        self.assertTrue(matches[1]["false_positive"])

    def test_average_precision_with_no_ground_truths_returns_zeroes(self) -> None:
        metrics = compute_average_precision(
            predictions=[ScoredBox(image_id="img1", box=(0, 0, 10, 10), score=0.9)],
            ground_truths=[],
            iou_threshold=0.5,
        )
        self.assertEqual(metrics["ap"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
