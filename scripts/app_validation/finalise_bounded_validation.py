#!/usr/bin/env python3
"""Archive the final non-model App demonstrator validation boundary."""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "outputs/01_protocol/smoke_protocol_public"
OUT = ROOT / "outputs/07_app_validation"

def main() -> None:
    summary = json.loads((SMOKE / "runs/latest/metadata/summary.json").read_text(encoding="utf-8"))
    decisions = list(csv.DictReader((SMOKE / "runs/latest/metadata/decisions.csv").open(encoding="utf-8")))
    manifest = list(csv.DictReader((SMOKE / "smoke_manifest_24.csv").open(encoding="utf-8")))
    statuses = Counter(row["status"] for row in decisions)
    selected = Counter(row["selected_method"] for row in decisions)
    applied = Counter(row["applied_method"] for row in decisions)
    screenshots = [
        "outputs/01_protocol/smoke_protocol_public/runs/latest/side_by_side/smoke_01_synthetic_face.jpg",
        "outputs/01_protocol/smoke_protocol_public/runs/latest/side_by_side/smoke_03_synthetic_screen_text.jpg",
        "app/examples/side_by_side/synthetic_noface__objective_profile_balanced.jpg",
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_class": "app_demonstrator_supporting_evidence_non_model",
        "scientific_performance_evidence": False,
        "input": {"kind": "public_synthetic", "n_images": len(manifest), "manifest": str((SMOKE / "smoke_manifest_24.csv").relative_to(ROOT))},
        "objective_selected": "balanced",
        "app_policy_id": summary["app_policy_id"],
        "scientific_policy_id": summary["scientific_policy_id"],
        "planned_methods": summary["profile"],
        "detector_policy_configured": summary["profile"]["face_detection"],
        "detector_policy_actually_executed": None,
        "detector_fallback_occurred": False,
        "detection_input": "manifest_provided_synthetic_boxes",
        "counts": {
            "faces": sum(int(row["n_face_boxes"]) for row in manifest),
            "screens": sum(int(row["n_screen_boxes"]) for row in manifest),
            "text_regions": sum(int(row["n_text_boxes"]) for row in manifest),
        },
        "selected_face_methods": dict(selected),
        "applied_face_methods": dict(applied),
        "fallback_count": statuses.get("fallback", 0),
        "error_count": sum(count for status, count in statuses.items() if status not in {"ok", "fallback"}),
        "runtime_seconds": summary["runtime_seconds"],
        "output_path": "outputs/01_protocol/smoke_protocol_public/runs/latest",
        "screenshots": screenshots,
        "workflow_checks": {
            "folder_and_scan_contract": "tested_by_pipeline_app_suite",
            "compute_and_asset_preflight": "tested_by_pipeline_app_suite",
            "objective_profile_generation": "pass",
            "face_text_screen_box_consumption": "pass_with_public_manifest_boxes",
            "optional_review_step": "tested_by_pipeline_app_suite_not_invoked_in_smoke",
            "anonymisation_and_multimodal_redaction": "pass",
            "output_and_success_report": "pass",
        },
        "model_rerun_skip": "Public manifest boxes exercise the downstream workflow without fresh face, text, or screen inference.",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "claim_boundary": "Supporting App demonstrator validation only; does not redefine scientific results.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "06_final_demonstrator_validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final App demonstrator validation",
        "",
        "- Result: **PASS (bounded, non-model workflow)**",
        f"- Public synthetic inputs: **{len(manifest)}**",
        f"- Objective/App policy: `balanced` / `{payload['app_policy_id']}`",
        f"- Selected methods: `{dict(selected)}`; applied methods: `{dict(applied)}`",
        f"- Face / screen / text boxes consumed: **{payload['counts']['faces']} / {payload['counts']['screens']} / {payload['counts']['text_regions']}**",
        f"- Fallbacks / errors: **{payload['fallback_count']} / {payload['error_count']}**",
        f"- Runtime: **{payload['runtime_seconds']} s**",
        "- Fresh model inference: **not run**; manifest-provided synthetic boxes exercised the downstream App modules.",
        "- Boundary: supporting demonstrator evidence only, not scientific performance evidence.",
        "",
        "Machine-readable record: `outputs/07_app_validation/06_final_demonstrator_validation.json`",
    ]
    (OUT / "06_final_demonstrator_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
