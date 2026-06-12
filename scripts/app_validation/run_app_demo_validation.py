#!/usr/bin/env python3
"""Run App demonstrator validation under current objective_profile implementation.

Writes under outputs/07_app_validation/ (canonical tree).
Legacy oapr_* runs are retained as historical subfolders if already present;
this script regenerates objective_profile focus runs + deterministic baselines.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
APP_OUT = ROOT / "outputs" / "07_app_validation"
# Face-anonymisation locked 500 has protocol face boxes used by pipeline_demo detections.
# Multimodal 250 does not share that face-box surface (0 overlap), so objective_profile
# would collapse to copy-only without live detector inference.
DEFAULT_MANIFEST = ROOT / "outputs" / "01_protocol" / "thesis_manifests" / "final_face_anonymisation_500.csv"

# App profiles and deterministic regression modes.
RUNS = [
    ("objective_profile_balanced", "objective_profile", "utility_under_privacy_floor"),
    ("objective_profile_privacy", "objective_profile", "privacy_first"),
    ("objective_profile_utility", "objective_profile", "utility_priority"),
    ("blur", "blur", "privacy_first"),
    ("layered", "layered", "privacy_first"),
    ("pixelate", "pixelate", "privacy_first"),
    ("solid_mask", "solid_mask", "privacy_first"),
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def main() -> None:
    manifest = DEFAULT_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {manifest}")

    output_root = APP_OUT / "run_logs"
    run_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for run_name, mode, objective in RUNS:
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "app" / "run_cli.py"),
            "--manifest",
            str(manifest),
            "--mode",
            mode,
            "--objective",
            objective,
            "--resolution",
            "native",
            "--limit",
            "100",
            "--output-dir",
            str(run_dir),
        ]
        print("RUN", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode != 0:
            print(completed.stdout[-2000:] if completed.stdout else "", flush=True)
            print(completed.stderr[-2000:] if completed.stderr else "", flush=True)
            raise SystemExit(f"App validation failed for {run_name}: exit {completed.returncode}")
        stdout = completed.stdout.strip()
        start = stdout.find("{")
        end = stdout.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Could not parse app summary JSON for {run_name}: {stdout[-500:]}")
        summary = json.loads(stdout[start : end + 1])
        numbered_outputs = {
            "runtime_summary.json": "01_runtime_summary.json",
            "routing_log.csv": "02_routing_log.csv",
            "per_image_manifest.csv": "03_per_image_manifest.csv",
        }
        for source_name, target_name in numbered_outputs.items():
            source_path = run_dir / source_name
            target_path = run_dir / target_name
            if source_path.exists():
                source_path.replace(target_path)
        # also accept already-numbered
        runtime_path = run_dir / "01_runtime_summary.json"
        if runtime_path.is_file() and "frames_ok" not in summary:
            summary = json.loads(runtime_path.read_text(encoding="utf-8"))
        run_rows.append(
            {
                "run": run_name,
                "mode": mode,
                "objective": objective,
                "workers": summary.get("workers", ""),
                "compute_profile": summary.get("compute_profile", ""),
                "frames_ok": summary.get("frames_ok", 0),
                "frames_failed": summary.get("frames_failed", 0),
                "mean_runtime_seconds": summary.get("runtime_mean_seconds", summary.get("mean_runtime_seconds", "")),
                "method_counts": json.dumps(summary.get("method_counts", {}), sort_keys=True),
                "output_dir": str(run_dir.relative_to(ROOT)),
            }
        )
        manifest_rows.append(
            {
                "run_name": run_name,
                "mode": mode,
                "objective": objective,
                "manifest_path": str(manifest.relative_to(ROOT)),
                "routing_log": str((run_dir / "02_routing_log.csv").relative_to(ROOT)),
                "per_image_manifest": str((run_dir / "03_per_image_manifest.csv").relative_to(ROOT)),
                "sample_outputs": "run_logs images/",
                "side_by_side_outputs": "run_logs side_by_side/",
            }
        )
        fail_log = run_dir / "failure_log.csv"
        if fail_log.is_file():
            with fail_log.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    failure_rows.append({"run": run_name, **row})

    write_csv(
        APP_OUT / "02_app_runtime_summary.csv",
        run_rows,
        [
            "run",
            "mode",
            "objective",
            "workers",
            "compute_profile",
            "frames_ok",
            "frames_failed",
            "mean_runtime_seconds",
            "method_counts",
            "output_dir",
        ],
    )
    write_csv(
        APP_OUT / "01_app_validation_manifest.csv",
        manifest_rows,
        [
            "run_name",
            "mode",
            "objective",
            "manifest_path",
            "routing_log",
            "per_image_manifest",
            "sample_outputs",
            "side_by_side_outputs",
        ],
    )
    write_csv(APP_OUT / "03_app_failure_log.csv", failure_rows, ["run", "image_id", "status", "error"])
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "app_policy_id": "objective_profile",
        "n_runs": len(run_rows),
        "runs": run_rows,
        "note": "Regenerated under current objective_profile implementation (plus deterministic baselines).",
    }
    (APP_OUT / "05_app_validation_regeneration_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote {APP_OUT}")


if __name__ == "__main__":
    main()
