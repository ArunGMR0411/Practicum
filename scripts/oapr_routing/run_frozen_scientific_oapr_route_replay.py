#!/usr/bin/env python3
"""Replay and materialise the frozen scientific OAPR 286/81/133 decisions.

Runs: retained scientific route decision → live detector-derived application
boxes → App anonymisation modules on CASTLE raw frames. This is not fresh OAPR
routing because the route decision is read from a frozen table (or replayed
from retained condition rows).

Default: full locked 500-frame face anonymisation protocol (may take a while).
Use --limit for a subset; full run should recover ~286/81/133 route counts when
using the offline condition profiler (same rule as e2e validation).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "app" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "app" / "src"))

from privacy_pipeline_app.runtime_env import configure_app_runtime  # noqa: E402
from privacy_pipeline_app.thesis_face_detector import (  # noqa: E402
    ThesisFaceDetector,
    resolve_runnable_face_policy,
)
from privacy_pipeline_app.wizard_workflow import _apply_method  # noqa: E402
from src.policy.registry import get_app_policy_semantics, load_policy_registry  # noqa: E402

CASTLE_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
MANIFEST = PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv"
CONDITIONS = (
    PROJECT_ROOT
    / "outputs/02_face_detection/10_post_detection_condition_annotation/post_detection_condition_predictions.csv"
)
# Fallback: visual-safe policy already materialised routes
VISUAL_SAFE_POLICY = (
    PROJECT_ROOT
    / "outputs/03_anonymisation/16_visual_quality_hardening/04_final_visual_safe_policy.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "outputs/10_final_enhancement_evaluation/06_frozen_scientific_oapr_route_replay"

METHOD_MAP = {
    "layered_blur_downscale_noise": "layered",
    "solid_mask_black": "solid_mask",
    "no_action_copy": "copy",
    "layered": "layered",
    "solid_mask": "solid_mask",
    "copy": "copy",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def route_from_conditions(row: pd.Series) -> tuple[str, str]:
    """Same scientific routing rule as run_end_to_end_thesis_validation.route_method."""
    if int(row.get("pred_no_face", 0)) == 1 and int(row.get("safety_candidate_count", 0)) == 0:
        return "no_action_copy", "confident_no_face"
    if int(row.get("pred_no_face", 0)) == 1:
        return "layered_blur_downscale_noise", "no_face_safety_gate_override"
    if int(row.get("pred_single_face", 0)) == 1:
        return "solid_mask_black", "single_face_privacy_first_policy"
    return "layered_blur_downscale_noise", "face_positive_practical_fallback"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="0 = full 500")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--face-policy-id",
        default=None,
        help="Detector application-box policy (default: bounded App runtime detector)",
    )
    p.add_argument(
        "--use-offline-routes",
        action="store_true",
        help="Use precomputed visual-safe policy routes (exact 286/81/133) instead of re-routing from conditions",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_app_runtime(force=True)
    if not CASTLE_ROOT.is_dir():
        raise SystemExit(f"CASTLE raw missing at {CASTLE_ROOT}")
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing {MANIFEST}")

    out_dir = args.output_dir
    (out_dir / "anonymised").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    semantics = get_app_policy_semantics()
    detector_id = args.face_policy_id or str(load_policy_registry()["app_runtime_detector"]["policy_id"])
    runtime = resolve_runnable_face_policy(detector_id)
    detector = ThesisFaceDetector(runtime["policy_id"])

    manifest = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    if args.limit and args.limit > 0:
        manifest = manifest[: args.limit]

    # Route table
    if args.use_offline_routes and VISUAL_SAFE_POLICY.is_file():
        vs = pd.read_csv(VISUAL_SAFE_POLICY)
        route_by_path = {
            str(r["relative_path"]): (str(r["selected_method"]), str(r.get("selection_reason", "offline_visual_safe")))
            for _, r in vs.iterrows()
        }
        route_source = "04_final_visual_safe_policy.csv"
    elif CONDITIONS.is_file():
        cond = pd.read_csv(CONDITIONS)
        route_by_path = {}
        for _, r in cond.iterrows():
            method, reason = route_from_conditions(r)
            route_by_path[str(r["relative_path"])] = (method, reason)
        route_source = CONDITIONS.name
    else:
        raise SystemExit("Need conditions CSV or visual-safe policy for scientific routing")

    frame_rows: list[dict] = []
    method_counts: Counter[str] = Counter()
    failures: list[dict] = []
    t0 = time.perf_counter()

    for index, row in enumerate(manifest, start=1):
        rel = row["relative_path"]
        src = CASTLE_ROOT / rel
        image_id = rel.replace("/", "__")
        scientific_method, reason = route_by_path.get(rel, ("layered_blur_downscale_noise", "fallback_missing_route"))
        app_method = METHOD_MAP.get(scientific_method, "layered")
        rec = {
            "index": index,
            "relative_path": rel,
            "image_id": image_id,
            "scientific_selected_method": scientific_method,
            "selection_reason": reason,
            "app_applied_method": app_method,
            "face_count_live": 0,
            "detector_application_policy_id": runtime["policy_id"],
            "status": "ok",
            "error": "",
            "output_path": "",
            "output_sha256": "",
            "runtime_seconds": 0.0,
        }
        if not src.is_file():
            rec["status"] = "missing_source"
            rec["error"] = f"missing {src}"
            failures.append(rec)
            frame_rows.append(rec)
            continue
        t1 = time.perf_counter()
        try:
            with Image.open(src) as loaded:
                image = loaded.convert("RGB")
            boxes_scored, det_meta = detector.detect(image)
            boxes = [(b[0], b[1], b[2], b[3]) for b in boxes_scored]
            rec["face_count_live"] = len(boxes)
            rec["candidate_counts"] = json.dumps(det_meta.get("candidate_counts") or {})
            # Scientific route decides operator; live boxes define where to apply
            if app_method == "copy":
                result = _apply_method(image, boxes, "copy", fallback_method="solid_mask")
            else:
                # If scientific says anonymise but live detector sees no faces, still copy (safety)
                if not boxes and app_method != "copy":
                    # Prefer scientific mask/layered on empty → copy with note
                    result = _apply_method(image, boxes, "copy", fallback_method="solid_mask")
                    rec["app_applied_method"] = "copy"
                    rec["selection_reason"] = reason + "|live_zero_faces_override_to_copy"
                else:
                    result = _apply_method(image, boxes, app_method, fallback_method="solid_mask")
                    rec["app_applied_method"] = result.applied_method
            rec["status"] = result.status
            rec["error"] = result.error or ""
            out_path = out_dir / "anonymised" / f"{image_id}.webp"
            result.image.save(out_path, format="WEBP", quality=90)
            rec["output_path"] = str(out_path.relative_to(PROJECT_ROOT))
            rec["output_sha256"] = sha256_file(out_path)
            method_counts[rec["app_applied_method"]] += 1
            method_counts[f"scientific::{scientific_method}"] += 1
            if result.status != "ok":
                failures.append(rec)
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(rec)
        rec["runtime_seconds"] = round(time.perf_counter() - t1, 4)
        frame_rows.append(rec)
        if index % 25 == 0 or index == len(manifest):
            print(
                f"[{index}/{len(manifest)}] {rel} sci={scientific_method} "
                f"app={rec['app_applied_method']} faces={rec['face_count_live']} {rec['status']}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    sci_counts = Counter(r["scientific_selected_method"] for r in frame_rows)
    app_counts = Counter(r["app_applied_method"] for r in frame_rows if r["status"] == "ok")
    expected = {"layered_blur_downscale_noise": 286, "solid_mask_black": 81, "no_action_copy": 133}
    summary = {
        "n_frames": len(frame_rows),
        "route_source": route_source,
        "scientific_policy_id": semantics.get("scientific_policy_id"),
        "app_policy_id": semantics.get("app_policy_id"),
        "detector_application_policy_id": runtime["policy_id"],
        "detector_candidate_sources": runtime.get("candidate_sources", []),
        "detector_runtime": {k: runtime[k] for k in runtime if k != "components"},
        "scientific_route_counts": dict(sci_counts),
        "app_applied_counts": dict(app_counts),
        "expected_full_500_counts": expected,
        "n_failures": len(failures),
        "runtime_total_seconds": round(elapsed, 3),
        "execution_mode": "frozen_scientific_oapr_route_replay",
        "pipeline": "frozen route decision → detector-derived application boxes → App anonymisation modules",
        "fresh_routing": False,
        "route_application_difference_note": "Scientific route counts describe retained decisions; app_applied_counts describe operators actually applied after live zero-box overrides.",
        "matches_expected_286_81_133": (
            len(frame_rows) == 500
            and sci_counts.get("layered_blur_downscale_noise", 0) == 286
            and sci_counts.get("solid_mask_black", 0) == 81
            and sci_counts.get("no_action_copy", 0) == 133
        ),
    }
    fields = list(frame_rows[0].keys()) if frame_rows else []
    with (out_dir / "metadata" / "per_frame_actions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(frame_rows)
    (out_dir / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
