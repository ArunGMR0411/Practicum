"""Tests for the locked CASTLE development-set manifest."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.data_protocol.build_dev_set import CONDITION_LABELS, MIN_PER_CONDITION, SEED, TARGET_COUNT, stable_order_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEV_SET_PATH = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"


class DevelopmentSetTest(unittest.TestCase):
    """Validate the generated development-set manifest."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the saved development-set manifest once for all checks."""
        cls.dev_df = pd.read_csv(DEV_SET_PATH)

    def test_correct_frame_count(self) -> None:
        """The development set must contain exactly 300 frames."""
        self.assertEqual(len(self.dev_df), TARGET_COUNT)

    def test_no_duplicates(self) -> None:
        """The development set must not contain duplicate relative paths."""
        self.assertEqual(self.dev_df["relative_path"].nunique(), len(self.dev_df))

    def test_all_stream_ids_covered(self) -> None:
        """The development set must cover every stream ID present in the manifest."""
        manifest_df = pd.read_csv(MANIFEST_PATH)
        manifest_streams = set(manifest_df["camera_stream_id"].dropna().astype(str))
        dev_streams = set(self.dev_df["camera_stream_id"].dropna().astype(str))
        self.assertEqual(dev_streams, manifest_streams)

    def test_minimum_condition_counts_met(self) -> None:
        """Every required condition label must appear at least 30 times."""
        counts = self.dev_df["condition_label"].value_counts().to_dict()
        for label in CONDITION_LABELS:
            self.assertGreaterEqual(
                counts.get(label, 0),
                MIN_PER_CONDITION,
                msg=f"Condition {label} is underrepresented: {counts.get(label, 0)}",
            )

    def test_fixed_seed_ordering_is_reproducible(self) -> None:
        """The deterministic ordering primitive must be stable for the locked seed."""
        paths = self.dev_df["relative_path"].head(20).astype(str).tolist()
        first = sorted(paths, key=stable_order_key)
        second = sorted(reversed(paths), key=stable_order_key)
        self.assertEqual(first, second)
        self.assertIsInstance(SEED, int)


if __name__ == "__main__":
    unittest.main()
