"""Tests for the standalone app package entrypoints."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_app_package_imports() -> None:
    app_src = PROJECT_ROOT / "app" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))

    from privacy_pipeline_app.production_runner import ALL_MODES, SELECTOR_OBJECTIVES

    assert "objective_profile" in ALL_MODES
    assert "privacy_first" in SELECTOR_OBJECTIVES
    assert "multimodal_risk" in SELECTOR_OBJECTIVES


def test_app_cli_smoke_on_tiny_manifest(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT
        / "outputs"
        / "01_protocol"
        / "thesis_manifests"
        / "final_multimodal_250.csv"
    )
    with source.open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "local_path", "relative_path"])
        writer.writeheader()
        writer.writerow(
            {
                "image_id": first_row.get("image_id", "smoke_0"),
                "local_path": first_row.get("local_path", ""),
                "relative_path": first_row.get("relative_path", ""),
            }
        )

    output_dir = tmp_path / "app_out"
    completed = subprocess.run(
        [
            sys.executable,
            "app/run_cli.py",
            "--manifest",
            str(manifest),
            "--mode",
            "objective_profile",
            "--objective",
            "privacy_first",
            "--limit",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "routing_log.csv").exists()
    assert (output_dir / "runtime_summary.json").exists()
