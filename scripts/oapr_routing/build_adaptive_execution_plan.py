#!/usr/bin/env python3

"""Build a strict adaptive execution plan from defended detector outputs.

This planner is intentionally evidence-constrained:
- `blur` remains the safe default operational branch,
- `nullface` is the only promoted advanced branch,
- `fams` is retained only as bounded optional evidence on stronger compute,
- `stylegan` is kept as a very narrow academic/GAN-only candidate lane.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.compute_policy import build_compute_policy

DEFAULT_MANIFEST = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"
DEFAULT_DETECTIONS = PROJECT_ROOT / "outputs" / "dev_set_detections_yolo_scrfd_fallback.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "adaptive_execution_plan_dev.json"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "adaptive_execution_plan_dev.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def load_manifest_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}


def load_detection_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["image_id"]].append(row)
    return grouped


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def box_metrics(detections: list[dict[str, str]]) -> tuple[float, float, float]:
    if not detections:
        return 0.0, 0.0, 0.0
    widths = [float(row["x2"]) - float(row["x1"]) for row in detections]
    heights = [float(row["y2"]) - float(row["y1"]) for row in detections]
    scores = [float(row["score"]) for row in detections]
    largest_dim = max(max(w, h) for w, h in zip(widths, heights, strict=True))
    largest_area = max(w * h for w, h in zip(widths, heights, strict=True))
    best_score = max(scores)
    return largest_dim, largest_area, best_score


def compute_tier_name(policy: Any) -> str:
    if policy.device != "cuda":
        return "low"
    if policy.accelerator_total_gb >= 30:
        return "high"
    if policy.accelerator_total_gb >= 10:
        return "medium"
    return "low"


def determine_method_plan(
    row: dict[str, str],
    detections: list[dict[str, str]],
    compute_tier: str,
) -> dict[str, Any]:
    face_count = len(detections)
    largest_dim, largest_area, best_det_score = box_metrics(detections)

    has_face = face_count > 0
    has_text = is_true(row.get("visible_text_flag"))
    has_screen = is_true(row.get("visible_screen_flag"))
    small_face = is_true(row.get("small_face_flag"))
    motion_blur = is_true(row.get("motion_blur_flag"))
    extreme_pose = is_true(row.get("extreme_pose_flag"))
    downward_view = is_true(row.get("downward_view_flag"))
    multiple_faces = is_true(row.get("multiple_faces_flag")) or face_count > 1
    no_face = face_count == 0

    content_partition: list[str] = []
    content_partition.append("has_face" if has_face else "no_face")
    if has_text:
        content_partition.append("has_text")
    if has_screen:
        content_partition.append("has_screen")

    stylegan_eligible = (
        has_face
        and face_count == 1
        and largest_dim >= 220.0
        and largest_area >= 52000.0
        and best_det_score >= 0.90
        and not motion_blur
        and not small_face
        and not multiple_faces
        and not downward_view
        and not extreme_pose
        and not has_text
        and not has_screen
    )
    nullface_eligible = (
        has_face
        and face_count == 1
        and largest_dim >= 160.0
        and best_det_score >= 0.82
        and not motion_blur
        and not small_face
        and not multiple_faces
    )
    fams_eligible = (
        has_face
        and face_count == 1
        and largest_dim >= 180.0
        and best_det_score >= 0.85
        and not small_face
        and not motion_blur
        and not multiple_faces
        and not downward_view
        and not extreme_pose
        and not has_text
        and not has_screen
        and compute_tier in {"medium", "high"}
    )
    blur_eligible = has_face
    pixelate_eligible = has_face

    method_bucket = "blur_only"
    recommended_method = "none"
    compute_requirement = "none"
    route_reason = "no_face_detected"
    utility_risk = "none"

    if has_face:
        recommended_method = "blur"
        compute_requirement = "low"
        route_reason = "safe_default_face_path"
        utility_risk = "low"
        if stylegan_eligible and compute_tier == "high":
            method_bucket = "stylegan_candidate"
            recommended_method = "stylegan"
            compute_requirement = "high"
            route_reason = "single_clean_large_face_gan_candidate"
            utility_risk = "high"
        elif nullface_eligible:
            method_bucket = "nullface_candidate"
            recommended_method = "nullface"
            compute_requirement = "low"
            route_reason = "single_clean_face_low_compute_candidate"
            utility_risk = "medium"
        elif fams_eligible:
            method_bucket = "fams_candidate"
            recommended_method = "fams"
            compute_requirement = "medium"
            route_reason = "clean_face_advanced_diffusion_candidate"
            utility_risk = "high"

    if no_face:
        method_bucket = "non_face_only"

    return {
        "face_count_detected": face_count,
        "largest_detected_face_px": round(largest_dim, 2),
        "largest_detected_face_area_px2": round(largest_area, 2),
        "best_detection_score": round(best_det_score, 4),
        "content_partition": "|".join(content_partition),
        "method_bucket": method_bucket,
        "recommended_method": recommended_method,
        "compute_requirement": compute_requirement,
        "route_reason": route_reason,
        "utility_risk": utility_risk,
        "stylegan_eligible": stylegan_eligible,
        "nullface_eligible": nullface_eligible,
        "fams_eligible": fams_eligible,
        "blur_eligible": blur_eligible,
        "pixelate_eligible": pixelate_eligible,
        "flags_summary": {
            "small_face": small_face,
            "motion_blur": motion_blur,
            "extreme_pose": extreme_pose,
            "downward_view": downward_view,
            "multiple_faces": multiple_faces,
            "visible_text": has_text,
            "visible_screen": has_screen,
        },
    }


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest_rows(args.manifest)
    detection_rows = load_detection_rows(args.detections)
    compute_policy = build_compute_policy()
    compute_tier = compute_tier_name(compute_policy)

    per_image_rows: list[dict[str, Any]] = []
    bucket_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()

    for relative_path, row in manifest_rows.items():
        detections = detection_rows.get(relative_path, [])
        plan = determine_method_plan(row, detections, compute_tier)
        record = {
            "relative_path": relative_path,
            "condition_label": row.get("condition_label", ""),
            "condition_matches": row.get("condition_matches", ""),
            "content_partition": plan["content_partition"],
            "method_bucket": plan["method_bucket"],
            "recommended_method": plan["recommended_method"],
            "compute_requirement": plan["compute_requirement"],
            "route_reason": plan["route_reason"],
            "utility_risk": plan["utility_risk"],
            "face_count_detected": plan["face_count_detected"],
            "largest_detected_face_px": plan["largest_detected_face_px"],
            "largest_detected_face_area_px2": plan["largest_detected_face_area_px2"],
            "best_detection_score": plan["best_detection_score"],
            "stylegan_eligible": plan["stylegan_eligible"],
            "nullface_eligible": plan["nullface_eligible"],
            "fams_eligible": plan["fams_eligible"],
            "blur_eligible": plan["blur_eligible"],
            "pixelate_eligible": plan["pixelate_eligible"],
        }
        per_image_rows.append(record)
        bucket_counts[record["method_bucket"]] += 1
        method_counts[record["recommended_method"]] += 1
        content_counts[record["content_partition"]] += 1

    summary = {
        "manifest": str(args.manifest.relative_to(PROJECT_ROOT)),
        "detections": str(args.detections.relative_to(PROJECT_ROOT)),
        "compute_policy": {
            "device": compute_policy.device,
            "accelerator_total_gb": compute_policy.accelerator_total_gb,
            "accelerator_available_gb": compute_policy.accelerator_available_gb,
            "compute_tier": compute_tier,
            "reid_batch_size": compute_policy.reid_batch_size,
            "fid_batch_size": compute_policy.fid_batch_size,
            "ocr_region_batch_size": compute_policy.ocr_region_batch_size,
            "generative_control_max_workers": compute_policy.generative_control_max_workers,
            "use_mixed_precision": compute_policy.use_mixed_precision,
            "use_low_vram_mode": compute_policy.use_low_vram_mode,
        },
        "strict_rules_version": "adaptive_execution_plan",
        "notes": [
            "This is a strict pre-execution planner, not a final accuracy claim.",
            "StyleGAN eligibility is intentionally very narrow and academic-only.",
            "NullFace is the only promoted advanced branch in the operational selector.",
            "FAMS remains bounded optional evidence and is only eligible on stronger compute.",
            "Blur remains the safe fallback for weak or ambiguous face conditions.",
            "Utility risk is recorded explicitly because privacy gain alone is not sufficient for promotion.",
        ],
        "bucket_counts": dict(bucket_counts),
        "recommended_method_counts": dict(method_counts),
        "content_partition_counts": dict(content_counts),
        "per_image": per_image_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "condition_label",
                "condition_matches",
                "content_partition",
                "method_bucket",
                "recommended_method",
                "compute_requirement",
                "route_reason",
                "utility_risk",
                "face_count_detected",
                "largest_detected_face_px",
                "largest_detected_face_area_px2",
                "best_detection_score",
                "stylegan_eligible",
                "nullface_eligible",
                "fams_eligible",
                "blur_eligible",
                "pixelate_eligible",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps({
        "output_json": str(args.output_json.relative_to(PROJECT_ROOT)),
        "output_csv": str(args.output_csv.relative_to(PROJECT_ROOT)),
        "bucket_counts": dict(bucket_counts),
        "recommended_method_counts": dict(method_counts),
        "compute_tier": compute_tier,
    }, indent=2))


if __name__ == "__main__":
    main()
