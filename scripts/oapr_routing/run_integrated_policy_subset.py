#!/usr/bin/env python3
"""True integrated execution on a locked reproducible subset.

Runs final face detector + objective_profile visual-safe anonymisation on live
pixels (not retained anonymisation outputs). Writes per-frame provenance,
output hashes, failures, and aggregate comparisons.

Default subset: first N frames of final_face_anonymisation_500 (seed order in
the locked manifest). Requires CASTLE raw under data/castle2024/raw/.
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
from src.policy.registry import (  # noqa: E402
    get_app_policy_semantics,
    get_profile,
    load_policy_registry,
)

CASTLE_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
MANIFEST = (
    PROJECT_ROOT
    / "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv"
)
DEFAULT_OUT = (
    PROJECT_ROOT
    / "outputs/10_final_enhancement_evaluation/05_integrated_policy_subset"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Locked subset size")
    parser.add_argument("--focus", default="balanced", choices=["privacy", "balanced", "utility"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--face-policy-id",
        default=None,
        help="Override detector policy id (default: registry accelerated_full primary)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_app_runtime(force=True)
    if not CASTLE_ROOT.is_dir():
        raise SystemExit(f"CASTLE raw missing at {CASTLE_ROOT}")
    if not MANIFEST.is_file():
        raise SystemExit(f"Locked manifest missing: {MANIFEST}")

    out_dir = args.output_dir
    (out_dir / "anonymised").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    semantics = get_app_policy_semantics()
    profile = get_profile(args.focus)
    face_method = str(profile["face_anonymisation"])
    detector_id = args.face_policy_id or str(
        load_policy_registry()["scientific_policies"]["face_detector_primary"]["policy_id"]
    )
    runtime = resolve_runnable_face_policy(detector_id)
    detector = ThesisFaceDetector(runtime["policy_id"])

    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))[: max(1, args.limit)]
    frame_rows: list[dict] = []
    method_counts: Counter[str] = Counter()
    failures: list[dict] = []
    t0 = time.perf_counter()

    for index, row in enumerate(rows, start=1):
        rel = row["relative_path"]
        src = CASTLE_ROOT / rel
        image_id = rel.replace("/", "__")
        rec = {
            "index": index,
            "relative_path": rel,
            "image_id": image_id,
            "status": "ok",
            "error": "",
            "selected_method": face_method,
            "applied_method": face_method,
            "face_count": 0,
            "detector_policy_id": runtime["policy_id"],
            "detector_requested_policy_id": runtime.get("requested_policy_id", detector_id),
            "detector_fallback_applied": bool(runtime.get("fallback_applied")),
            "candidate_counts": {},
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
            rec["face_count"] = len(boxes)
            rec["candidate_counts"] = det_meta.get("candidate_counts") or {}
            rec["detector_policy_id"] = det_meta.get("policy_id") or runtime["policy_id"]
            selected = "copy" if not boxes else face_method
            rec["selected_method"] = selected
            result = _apply_method(image, boxes, selected, fallback_method="solid_mask")
            rec["applied_method"] = result.applied_method
            rec["status"] = result.status
            rec["error"] = result.error or ""
            out_path = out_dir / "anonymised" / f"{image_id}.webp"
            result.image.save(out_path, format="WEBP", quality=90)
            rec["output_path"] = str(out_path.relative_to(PROJECT_ROOT))
            rec["output_sha256"] = sha256_file(out_path)
            method_counts[result.applied_method] += 1
            if result.status != "ok":
                failures.append(rec)
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(rec)
        rec["runtime_seconds"] = round(time.perf_counter() - t1, 4)
        frame_rows.append(rec)
        print(
            f"[{index}/{len(rows)}] {rel} faces={rec['face_count']} "
            f"{rec['selected_method']}→{rec['applied_method']} {rec['status']}",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "subset_size": len(rows),
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "focus": args.focus,
        "app_policy_id": semantics["app_policy_id"],
        "scientific_policy_id": semantics["scientific_policy_id"],
        "simplification": semantics["simplification"],
        "detector_policy_id": runtime["policy_id"],
        "detector_runtime": runtime,
        "profile": profile,
        "method_counts_applied": dict(method_counts),
        "n_failures": len(failures),
        "runtime_total_seconds": round(elapsed, 3),
        "note": (
            "Live integrated subset execution. App path uses objective_profile fixed face "
            "operator (plus copy when no faces), not the scientific 286/81/133 OAPR router."
        ),
    }
    # Aggregate comparison placeholder vs scientific route counts (not same protocol subset)
    scientific_counts = semantics.get("scientific_route_counts") or {}
    summary["scientific_route_counts_full_500"] = scientific_counts
    summary["comparison_note"] = (
        "Subset method counts are not expected to match full-500 scientific OAPR "
        "route counts; comparison is structural (detector live + visual-safe ops only)."
    )

    fields = list(frame_rows[0].keys()) if frame_rows else []
    with (out_dir / "metadata" / "per_frame_actions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in frame_rows:
            row = dict(rec)
            if isinstance(row.get("candidate_counts"), dict):
                row["candidate_counts"] = json.dumps(row["candidate_counts"])
            writer.writerow(row)
    (out_dir / "metadata" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metadata" / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metadata" / "policy_semantics.json").write_text(
        json.dumps(semantics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
