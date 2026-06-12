#!/usr/bin/env python3

"""Run OAPR decisions over the locked 500-frame protocol without writing images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

from privacy_pipeline_app.pipeline_demo import (  # noqa: E402
    DetectionSummary,
    build_profile,
    load_detections,
    objective_aware_privacy_router,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "01_protocol"
    / "01_locked_500_input_manifest.csv"
)
DEFAULT_DETECTIONS = [
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "classical_baselines"
    / "anonymisation_eval_subset_yolo_scrfd_fallback.csv"
]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "runs" / "oapr_metadata"
OBJECTIVES = [
    "privacy_first",
    "utility_priority",
    "utility_under_privacy_floor",
    "runtime_aware",
    "compute_profile_adaptive",
    "failure_avoidance",
    "multimodal_risk",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def serialise(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def normalise_manifest_row(row: dict[str, str]) -> dict[str, str]:
    relative_path = row.get("relative_path") or row.get("image_id") or ""
    raw_path = row.get("raw_path") or row.get("local_path") or str(PROJECT_ROOT / "data" / "castle2024" / "raw" / relative_path)
    return {
        **row,
        "relative_path": relative_path,
        "image_id": relative_path,
        "local_path": raw_path,
        # The locked comparable manifest does not include reviewed leakage labels.
        # Use conservative defaults; face risk is inferred from detector boxes.
        "visible_face_leakage_risk": row.get("visible_face_leakage_risk", "high" if int(row.get("box_count") or 0) else "none"),
        "visible_text_leakage_risk": row.get("visible_text_leakage_risk", "none"),
        "visible_screen_leakage_risk": row.get("visible_screen_leakage_risk", "none"),
        "visual_disruption": row.get("visual_disruption", "unknown"),
        "obvious_anonymisation_artifact_risk": row.get("obvious_anonymisation_artifact_risk", "unknown"),
        "selector_recommendation": row.get("selector_recommendation", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detections", type=Path, nargs="*", default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objectives", nargs="+", default=OBJECTIVES)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [normalise_manifest_row(row) for row in read_csv(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    detections = load_detections(args.detections)
    profile = build_profile()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_manifest: list[dict[str, Any]] = []
    routing_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for objective in args.objectives:
        counts: Counter[str] = Counter()
        started_objective = time.perf_counter()
        failures = 0
        total_runtime = 0.0
        for index, row in enumerate(rows, start=1):
            started = time.perf_counter()
            image_id = row["image_id"]
            try:
                image = Image.open(row["local_path"]).convert("RGB")
                summary = detections.get(image_id, DetectionSummary([], [], [], []))
                decision = objective_aware_privacy_router(
                    row,
                    image,
                    summary,
                    objective,
                    profile,
                    detector_mode="cached_yolo_scrfd_fallback",
                    resolution="native",
                )
                elapsed = round(time.perf_counter() - started, 6)
                total_runtime += elapsed
                counts[decision.selected_method] += 1
                payload = {
                    "case_id": f"oapr500_{objective}_{index:04d}",
                    "relative_path": image_id,
                    "objective_mode": objective,
                    "selected_method": decision.selected_method,
                    "selected_action": decision.selected_action,
                    "fallback_method": decision.fallback_method,
                    "eligible_methods": decision.eligible_methods,
                    "rejected_methods": decision.rejected_methods,
                    "rejection_reasons": decision.rejection_reasons,
                    "privacy_floor_status": decision.privacy_floor_status,
                    "utility_status": decision.utility_status,
                    "runtime_status": decision.runtime_status,
                    "quality_gate_status": decision.quality_gate_status,
                    "apparent_demographic_quality_gate_status": decision.apparent_demographic_quality_gate_status,
                    "face_risk": decision.face_risk,
                    "text_risk": decision.text_risk,
                    "screen_risk": decision.screen_risk,
                    "face_count": decision.face_count,
                    "dominant_face_height_px": decision.dominant_face_height_px,
                    "dominant_face_area_ratio": decision.dominant_face_area_ratio,
                    "pose_condition": decision.pose_condition,
                    "source_quality_condition": decision.source_quality_condition,
                    "expected_utility_cost": decision.expected_utility_cost,
                    "expected_runtime_cost": decision.expected_runtime_cost,
                    "evidence_level": decision.evidence_level,
                    "residual_risk_note": decision.residual_risk_note,
                    "explanation": decision.explanation,
                    "resource_concurrency_policy": decision.resource_concurrency_policy,
                    "runtime_seconds": elapsed,
                    "status": "ok",
                    "error": "",
                }
                routing_rows.append({k: serialise(v) for k, v in payload.items()})
            except Exception as exc:  # pragma: no cover - evidence execution path
                elapsed = round(time.perf_counter() - started, 6)
                total_runtime += elapsed
                failures += 1
                failure_rows.append(
                    {
                        "objective_mode": objective,
                        "relative_path": image_id,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "runtime_seconds": elapsed,
                    }
                )
        wall = round(time.perf_counter() - started_objective, 6)
        runtime_rows.append(
            {
                "objective_mode": objective,
                "frames_requested": len(rows),
                "frames_ok": len(rows) - failures,
                "frames_failed": failures,
                "method_counts": json.dumps(dict(counts), sort_keys=True),
                "runtime_total_seconds": round(total_runtime, 6),
                "wall_seconds": wall,
                "mean_runtime_seconds": round(total_runtime / max(1, len(rows)), 6),
                "compute_profile": profile["compute_profile"],
            }
        )
        run_manifest.append(
            {
                "objective_mode": objective,
                "manifest": str(args.manifest),
                "detections": ";".join(str(path) for path in args.detections),
                "routing_log": str(output_root / "oapr_500_routing_log.csv"),
                "runtime_summary": str(output_root / "oapr_500_runtime_summary.csv"),
                "failure_log": str(output_root / "routing_failure_log.csv"),
            }
        )

    routing_fields = [
        "case_id",
        "relative_path",
        "objective_mode",
        "selected_method",
        "selected_action",
        "fallback_method",
        "eligible_methods",
        "rejected_methods",
        "rejection_reasons",
        "privacy_floor_status",
        "utility_status",
        "runtime_status",
        "quality_gate_status",
        "apparent_demographic_quality_gate_status",
        "face_risk",
        "text_risk",
        "screen_risk",
        "face_count",
        "dominant_face_height_px",
        "dominant_face_area_ratio",
        "pose_condition",
        "source_quality_condition",
        "expected_utility_cost",
        "expected_runtime_cost",
        "evidence_level",
        "residual_risk_note",
        "explanation",
        "resource_concurrency_policy",
        "runtime_seconds",
        "status",
        "error",
    ]
    write_csv(output_root / "oapr_500_routing_log.csv", routing_rows, routing_fields)
    write_csv(output_root / "oapr_500_runtime_summary.csv", runtime_rows, list(runtime_rows[0]))
    write_csv(output_root / "routing_failure_log.csv", failure_rows, ["objective_mode", "relative_path", "status", "error", "runtime_seconds"])
    write_csv(output_root / "oapr_500_run_manifest.csv", run_manifest, list(run_manifest[0]))

    summary = {
        "objectives": args.objectives,
        "frames_per_objective": len(rows),
        "routing_decisions": len(routing_rows),
        "failures": len(failure_rows),
        "runtime_summary": str(output_root / "oapr_500_runtime_summary.csv"),
        "routing_log": str(output_root / "oapr_500_routing_log.csv"),
    }
    (output_root / "oapr_500_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
