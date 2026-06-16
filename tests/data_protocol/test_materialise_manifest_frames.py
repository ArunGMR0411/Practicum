"""Tests for manifest-based raw-frame transfer materialisation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "data_protocol" / "materialise_manifest_frames.py"


class MaterialiseManifestFramesTest(unittest.TestCase):
    """Validate deterministic manifest-driven frame copying for constrained-compute transfer packs."""

    def write_manifest(self, path: Path, relative_paths: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["relative_path"])
            writer.writeheader()
            for relative_path in relative_paths:
                writer.writerow({"relative_path": relative_path})

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_dry_run_reports_size_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            source = raw_root / "day1" / "members" / "allie" / "10_0001.webp"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"frame-bytes")
            manifest = root / "manifest.csv"
            output_root = root / "transfer"
            self.write_manifest(manifest, ["day1/members/allie/10_0001.webp"])

            result = self.run_script(
                "--manifest",
                str(manifest),
                "--raw-root",
                str(raw_root),
                "--output-root",
                str(output_root),
                "--dry-run",
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["unique_requested_paths"], 1)
            self.assertEqual(payload["missing_count"], 0)
            self.assertEqual(payload["total_bytes"], len(b"frame-bytes"))
            self.assertFalse((output_root / "data" / "castle2024" / "raw" / "day1" / "members" / "allie" / "10_0001.webp").exists())

    def test_copy_preserves_raw_layout_and_deduplicates_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            rel_a = "day1/members/allie/10_0001.webp"
            rel_b = "day1/members/bjorn/10_0002.webp"
            for rel_path, content in [(rel_a, b"a"), (rel_b, b"bb")]:
                source = raw_root / rel_path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(content)
            manifest_a = root / "manifest_a.csv"
            manifest_b = root / "manifest_b.csv"
            self.write_manifest(manifest_a, [rel_a, rel_b])
            self.write_manifest(manifest_b, [rel_a])
            output_root = root / "transfer"

            result = self.run_script(
                "--manifest",
                str(manifest_a),
                "--manifest",
                str(manifest_b),
                "--raw-root",
                str(raw_root),
                "--output-root",
                str(output_root),
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["unique_requested_paths"], 2)
            self.assertEqual(payload["copied"], 2)
            copied_a = output_root / "data" / "castle2024" / "raw" / rel_a
            copied_b = output_root / "data" / "castle2024" / "raw" / rel_b
            self.assertEqual(copied_a.read_bytes(), b"a")
            self.assertEqual(copied_b.read_bytes(), b"bb")

    def test_missing_frame_fails_with_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            raw_root.mkdir()
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, ["missing.webp"])

            result = self.run_script(
                "--manifest",
                str(manifest),
                "--raw-root",
                str(raw_root),
                "--output-root",
                str(root / "transfer"),
                "--dry-run",
            )

            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["missing_count"], 1)
            self.assertEqual(payload["missing_sample"], ["missing.webp"])


if __name__ == "__main__":
    unittest.main()
