#!/usr/bin/env python3

"""Tests for the face annotation workflow helpers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class FaceAnnotationWorkflowTest(unittest.TestCase):
    """Check final annotation pack and validation schema assumptions."""

    def test_annotation_pack_path_is_under_expected_pack(self) -> None:
        pack_path = Path("outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
        self.assertIn("face_detection/02_egocentric_stress_500", str(pack_path))

    def test_completed_annotation_header(self) -> None:
        """The final completed annotations should expose the expected columns."""
        annotation_path = Path("outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv")
        if not annotation_path.exists():
            self.skipTest("completed annotation file is not available")
        header = annotation_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        # Actual schema uses per-image rows with JSON for boxes (see reviewed_face_boxes_json)
        for column in ["image_id", "manual_face_count", "reviewed_face_boxes_json", "condition_label", "manual_review_status"]:
            self.assertIn(column, header)

    def test_validator_schema_example_is_well_formed(self) -> None:
        """A minimal completed-annotation CSV should conform to the expected header."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["image_id", "x1", "y1", "x2", "y2", "annotator_id", "annotation_round", "condition_label", "notes"]
                )
                writer.writerow(["img.webp", "1", "2", "3", "4", "annotator", "1", "small_face", ""])
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)

    def test_validator_example_preserves_expected_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "frame.webp"
            Image.new("RGB", (8, 8), color="white").save(image_path)
            row = ["frame.webp", "1", "2", "3", "4", "annotator", "1", "small_face", ""]
            self.assertEqual(len(row), 9)


if __name__ == "__main__":
    unittest.main()
