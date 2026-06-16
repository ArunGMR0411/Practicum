"""Tests for the generative control-pack batch runner."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.anonymisation.run_generative_control_batch import run_batch
from scripts.anonymisation.summarise_generative_control_runs import build_summary


class GenerativeControlBatchTest(unittest.TestCase):
    def test_blur_control_batch_writes_manifest_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_root = root / "pack"
            image_rel = "day1/members/allie/10_0122.webp"
            image_path = pack_root / "data" / "castle2024" / "raw" / image_rel
            boxes_path = pack_root / "boxes" / "day1" / "members" / "allie" / "10_0122.boxes.json"
            output_root = root / "outputs"

            image_path.parent.mkdir(parents=True)
            boxes_path.parent.mkdir(parents=True)
            Image.new("RGB", (128, 128), color=(180, 180, 180)).save(image_path)
            boxes_path.write_text(json.dumps({"boxes": [[16, 16, 96, 96]]}), encoding="utf-8")
            (pack_root / "control_pack_summary.json").write_text(
                json.dumps(
                    {
                        "box_files": [
                            {
                                "relative_path": image_rel,
                                "boxes_json": str(boxes_path.relative_to(pack_root)),
                                "box_count": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = run_batch(
                argparse.Namespace(
                    pack_root=pack_root,
                    method="blur",
                    output_root=output_root,
                    max_images=None,
                    continue_on_error=False,
                    max_workers=0,
                )
            )

            self.assertEqual(result["ok_count"], 1)
            self.assertEqual(result["error_count"], 0)
            self.assertTrue((output_root / "blur" / image_rel).is_file())
            self.assertTrue((output_root / "blur_control_manifest.csv").is_file())
            self.assertTrue((output_root / "blur_control_summary.json").is_file())

            summary = build_summary(
                argparse.Namespace(
                    output_root=output_root,
                    methods=["blur"],
                    output_json=output_root / "summary.json",
                    allow_missing_methods=False,
                )
            )
            self.assertTrue(summary["ready_for_larger_run"])
            self.assertEqual(summary["missing_methods"], [])
            self.assertEqual(summary["failed_methods"], [])

    def test_control_summary_fails_missing_method_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = build_summary(
                argparse.Namespace(
                    output_root=Path(tmpdir),
                    methods=["stylegan"],
                    output_json=Path(tmpdir) / "summary.json",
                    allow_missing_methods=False,
                )
            )
        self.assertFalse(summary["ready_for_larger_run"])
        self.assertEqual(summary["missing_methods"], ["stylegan"])
        self.assertEqual(summary["failed_methods"], ["stylegan"])


if __name__ == "__main__":
    unittest.main()
