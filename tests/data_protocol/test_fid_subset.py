"""Tests for the Phase 3 FID subset builder."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.data_protocol.build_fid_subset import build_fid_subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"


class FIDSubsetBuilderTest(unittest.TestCase):
    """Validate the locked FID subset generation contract."""

    def test_build_fid_subset_has_expected_size(self) -> None:
        fid_df = build_fid_subset(MANIFEST_PATH)
        self.assertEqual(len(fid_df), 50000)

    def test_build_fid_subset_is_reproducible(self) -> None:
        first = build_fid_subset(MANIFEST_PATH)
        second = build_fid_subset(MANIFEST_PATH)
        self.assertEqual(first["relative_path"].tolist(), second["relative_path"].tolist())


if __name__ == "__main__":
    unittest.main()
