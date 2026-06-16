"""Tests for locked full-scale evaluation subset manifests."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.data_protocol.build_eval_subsets import build_eval_subsets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"


class EvalSubsetBuilderTest(unittest.TestCase):
    """Validate the detection and anonymisation evaluation subsets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.subsets = build_eval_subsets(MANIFEST_PATH)

    def test_expected_sizes(self) -> None:
        self.assertEqual(len(self.subsets["DETECTION_EVAL_SUBSET"]), 500)
        self.assertEqual(len(self.subsets["ANONYMISATION_EVAL_SUBSET"]), 500)

    def test_detection_and_anonymisation_are_egocentric(self) -> None:
        for name in ("DETECTION_EVAL_SUBSET", "ANONYMISATION_EVAL_SUBSET"):
            values = sorted(self.subsets[name]["view_type"].dropna().astype(str).unique().tolist())
            self.assertEqual(values, ["egocentric"])

    def test_subsets_are_pairwise_disjoint(self) -> None:
        names = ["DETECTION_EVAL_SUBSET", "ANONYMISATION_EVAL_SUBSET"]
        for index, left_name in enumerate(names):
            left_paths = set(self.subsets[left_name]["relative_path"].astype(str).tolist())
            for right_name in names[index + 1 :]:
                right_paths = set(self.subsets[right_name]["relative_path"].astype(str).tolist())
                self.assertFalse(left_paths & right_paths, msg=f"Overlap detected between {left_name} and {right_name}")

    def test_build_is_reproducible(self) -> None:
        second = build_eval_subsets(MANIFEST_PATH)
        for name in ("DETECTION_EVAL_SUBSET", "ANONYMISATION_EVAL_SUBSET"):
            self.assertEqual(
                self.subsets[name]["relative_path"].tolist(),
                second[name]["relative_path"].tolist(),
            )


if __name__ == "__main__":
    unittest.main()
