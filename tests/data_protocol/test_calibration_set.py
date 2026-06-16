"""Tests for the locked CASTLE calibration-set manifest."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.data_protocol.build_calibration_set import SEED, TARGET_COUNT, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEV_SET_PATH = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"
CALIBRATION_SET_PATH = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "02_calibration_200.csv"
QUALITY_DIMENSIONS = ["blur_level", "face_size", "occlusion_ratio", "webp_artefact_severity"]


class CalibrationSetTest(unittest.TestCase):
    """Validate the generated calibration-set manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the saved manifests once for all checks."""
        cls.dev_df = pd.read_csv(DEV_SET_PATH)
        cls.calibration_df = pd.read_csv(CALIBRATION_SET_PATH)

    def test_correct_frame_count(self) -> None:
        """The calibration set must contain exactly 200 frames."""
        self.assertEqual(len(self.calibration_df), TARGET_COUNT)

    def test_no_overlap_with_dev_set(self) -> None:
        """The calibration set must be disjoint from the development set."""
        overlap = set(self.dev_df["relative_path"]) & set(self.calibration_df["relative_path"])
        self.assertFalse(overlap, msg=f"Calibration/dev overlap detected: {sorted(overlap)[:5]}")

    def test_all_quality_dimensions_represented(self) -> None:
        """Each quality dimension must contain all three bucket labels."""
        for column in QUALITY_DIMENSIONS:
            values = sorted(self.calibration_df[column].dropna().astype(str).unique().tolist())
            self.assertEqual(values, ["high", "low", "medium"] if column != "face_size" else ["large", "medium", "small"])

    def test_fixed_seed_ordering_is_reproducible(self) -> None:
        """The deterministic ordering primitive must be stable for the locked seed."""
        paths = self.calibration_df["relative_path"].head(20).astype(str).tolist()
        first = sorted(paths, key=lambda value: (stable_hash(value), value))
        second = sorted(reversed(paths), key=lambda value: (stable_hash(value), value))
        self.assertEqual(first, second)
        self.assertIsInstance(SEED, int)


if __name__ == "__main__":
    unittest.main()
