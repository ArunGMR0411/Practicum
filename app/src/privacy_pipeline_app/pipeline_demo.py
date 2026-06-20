#!/usr/bin/env python3
"""Shareable pipeline demonstrator with Objective-Aware Privacy Router logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = ROOT / "data/castle2024/raw"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.utils.compute_policy import build_compute_policy
except Exception:  # pragma: no cover - environment dependent fallback
    build_compute_policy = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectionSummary:
    boxes: list[tuple[int, int, int, int]]
    scores: list[float]
    detector_names: list[str]
    condition_labels: list[str]


@dataclass(frozen=True)
class SelectorDecision:
    selected_method: str
    selected_action: str
    fallback_method: str
    eligible_methods: list[str]
    rejected_methods: list[str]
    rejection_reasons: dict[str, str]
    objective_mode: str
    compute_profile: str
    detector_mode: str
    resolution_path: str
    face_risk: str
    text_risk: str
    screen_risk: str
    multimodal_actions: list[str]
    privacy_floor_status: str
    utility_status: str
    runtime_status: str
    quality_gate_status: str
    apparent_demographic_quality_gate_status: str
    evidence_level: str
    source_artifacts: list[str]
    residual_risk_note: str
    explanation: str
    face_count: int
    dominant_face_height_px: int
    dominant_face_area_ratio: float
    face_confidence_min: str
    face_confidence_max: str
    face_confidence_mean: str
    pose_condition: str
    source_quality_condition: str
    expected_utility_cost: str
    expected_runtime_cost: str
    resource_concurrency_policy: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_detections(paths: list[Path]) -> dict[str, DetectionSummary]:
    grouped_boxes: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_detectors: dict[str, list[str]] = defaultdict(list)
    grouped_conditions: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for row in read_csv(path):
            image_id = row.get("image_id") or row.get("relative_path")
            if not image_id:
                continue
            try:
                grouped_boxes[image_id].append(
                    (
                        int(float(row["x1"])),
                        int(float(row["y1"])),
                        int(float(row["x2"])),
                        int(float(row["y2"])),
                    )
                )
                if row.get("score"):
                    grouped_scores[image_id].append(float(row["score"]))
                if row.get("detector_name"):
                    grouped_detectors[image_id].append(row["detector_name"])
                if row.get("condition_label"):
                    grouped_conditions[image_id].append(row["condition_label"])
            except (KeyError, ValueError):
                continue
    return {
        image_id: DetectionSummary(
            boxes=boxes,
            scores=grouped_scores.get(image_id, []),
            detector_names=sorted(set(grouped_detectors.get(image_id, []))),
            condition_labels=sorted(set(grouped_conditions.get(image_id, []))),
        )
        for image_id, boxes in grouped_boxes.items()
    }


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    left = max(0, min(x1, width))
    top = max(0, min(y1, height))
    right = max(0, min(x2, width))
    bottom = max(0, min(y2, height))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def valid_boxes(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    return [box for box in (clamp_box(box, *image.size) for box in boxes) if box is not None]


def apply_blur(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    output = image.copy()
    for box in valid_boxes(output, boxes):
        region = output.crop(box).filter(ImageFilter.GaussianBlur(radius=18))
        output.paste(region, box[:2])
    return output


def apply_pixelate(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    output = image.copy()
    for left, top, right, bottom in valid_boxes(output, boxes):
        region = output.crop((left, top, right, bottom))
        width, height = region.size
        tiny = region.resize((max(1, width // 14), max(1, height // 14)), Image.Resampling.BILINEAR)
        output.paste(tiny.resize((width, height), Image.Resampling.NEAREST), (left, top))
    return output


def apply_solid(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box in valid_boxes(output, boxes):
        draw.rectangle(box, fill=(0, 0, 0))
    return output


def apply_layered(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    output = image.copy()
    for left, top, right, bottom in valid_boxes(output, boxes):
        region = output.crop((left, top, right, bottom)).convert("RGB")
        width, height = region.size
        tiny = region.resize((max(1, width // 24), max(1, height // 24)), Image.Resampling.BILINEAR)
        layered = tiny.resize((width, height), Image.Resampling.NEAREST)
        layered = layered.filter(ImageFilter.GaussianBlur(radius=max(4, min(width, height) / 18)))
        pixels = layered.load()
        for y in range(height):
            for x in range(width):
                if (x + y) % 7 == 0:
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (max(0, r - 16), max(0, g - 16), max(0, b - 16))
        output.paste(layered, (left, top))
    return output


def apply_text_screen_redaction(image: Image.Image, row: dict[str, str]) -> Image.Image:
    """Conservative demonstrator redaction when exact text/screen boxes are unavailable."""
    text = row.get("visible_text_leakage_risk", "none")
    screen = row.get("visible_screen_leakage_risk", "none")
    if text not in {"high", "possible"} and screen not in {"high", "possible"}:
        return image
    output = image.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    if screen in {"high", "possible"}:
        draw.rectangle((int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.55)), fill=(0, 0, 0))
    if text in {"high", "possible"}:
        draw.rectangle((int(width * 0.05), int(height * 0.55), int(width * 0.95), int(height * 0.88)), fill=(0, 0, 0))
    return output


METHODS = {
    "blur": apply_blur,
    "pixelate": apply_pixelate,
    "solid_mask": apply_solid,
    "layered": apply_layered,
}


def build_profile() -> dict[str, Any]:
    if build_compute_policy is None:
        return {
            "compute_profile": "unknown_or_unstable_profile",
            "device": "unknown",
            "workers": 1,
            "device_summary": "compute_policy_unavailable",
        }
    try:
        policy = build_compute_policy()
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "compute_profile": "unknown_or_unstable_profile",
            "device": "unknown",
            "workers": 1,
            "device_summary": f"compute_policy_error:{type(exc).__name__}",
        }
    if policy.device != "cuda":
        profile = "cpu_or_very_low_resource"
    elif policy.accelerator_total_gb >= 36:
        profile = "stronger_accelerator_constrained_compute"
    elif policy.accelerator_total_gb >= 8:
        profile = "low_vram_local_gpu"
    else:
        profile = "unknown_or_unstable_profile"
    workers = max(1, min(policy.detection_num_workers, 8))
    return {
        "compute_profile": profile,
        "device": policy.device,
        "workers": workers,
        "device_summary": json.dumps(
            {
                "device": policy.device,
                "accelerator_total_gb": policy.accelerator_total_gb,
                "accelerator_available_gb": policy.accelerator_available_gb,
                "ram_total_gb": policy.host_ram_total_gb,
                "ram_available_gb": policy.host_ram_available_gb,
                "detection_num_workers": policy.detection_num_workers,
                "generative_control_max_workers": policy.generative_control_max_workers,
            },
            sort_keys=True,
        ),
    }


def face_geometry(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> tuple[int, float]:
    valid = valid_boxes(image, boxes)
    if not valid:
        return 0, 0.0
    heights = [bottom - top for _, top, _, bottom in valid]
    areas = [(right - left) * (bottom - top) for left, top, right, bottom in valid]
    return max(heights), max(areas) / float(image.width * image.height)


def infer_pose_condition(row: dict[str, str], detections: DetectionSummary, dominant_height: int) -> str:
    labels = set(detections.condition_labels)
    recommendation = row.get("selector_recommendation", "")
    if labels:
        return "+".join(sorted(labels))
    for token in ("downward", "extreme_pose", "side_face", "profile", "small_face", "multi_face", "multiple_faces"):
        if token in recommendation:
            return token
    if dominant_height and dominant_height < 150:
        return "small_face"
    return "unknown"


def confidence_summary(scores: list[float]) -> tuple[str, str, str]:
    if not scores:
        return "not_available", "not_available", "not_available"
    return f"{min(scores):.4f}", f"{max(scores):.4f}", f"{sum(scores) / len(scores):.4f}"


def reject_base_methods(compute_profile: str, quality_condition: str) -> dict[str, str]:
    rejected = {
        "StyleGAN": "quality_limited_visual_misalignment_on_egocentric_faces",
        "FAMS": "quality_limited_palette_or_blotchy_artifacts",
        "reverse_personalization": "partial_comparable_with_method_failures_and_runtime_limited_for_routine_pipeline_use",
        "G2Face": "dependency_limited",
        "ReFaceX": "literature_only_or_not_reproducibly_executable",
        "DeepPrivacy2": "checkpoint_or_access_limited_not_comparable",
        "diffusion_low_step": "full_comparable_but_visual_quality_gated_and_not_enabled_in_local_deterministic_demo",
        "NullFace": "full_metric_but_quality_and_runtime_limited_non_default",
    }
    if compute_profile in {"cpu_or_very_low_resource", "unknown_or_unstable_profile"}:
        rejected["NullFace"] = "not_eligible_for_current_compute_profile"
        rejected["diffusion_low_step"] = "not_eligible_for_current_compute_profile"
    if "high_source_blur" in quality_condition:
        rejected["NullFace"] = "rejected_by_source_quality_gate"
        rejected["diffusion_low_step"] = "rejected_by_source_quality_gate"
    return rejected


def objective_aware_privacy_router(
    row: dict[str, str],
    image: Image.Image,
    detections: DetectionSummary,
    objective: str,
    profile: dict[str, Any],
    detector_mode: str,
    resolution: str,
) -> SelectorDecision:
    boxes = detections.boxes
    face_count = len(valid_boxes(image, boxes))
    dominant_height, area_ratio = face_geometry(image, boxes)
    conf_min, conf_max, conf_mean = confidence_summary(detections.scores)
    pose_condition = infer_pose_condition(row, detections, dominant_height)
    text_risk = row.get("visible_text_leakage_risk", "none") or "unknown"
    screen_risk = row.get("visible_screen_leakage_risk", "none") or "unknown"
    face_risk = row.get("visible_face_leakage_risk", "none") or "unknown"
    quality_condition = row.get("visual_disruption", "unknown") or "unknown"
    artifact = row.get("obvious_anonymisation_artifact_risk", "unknown") or "unknown"
    compute_profile = profile["compute_profile"]

    rejected = reject_base_methods(compute_profile, quality_condition)
    eligible = ["blur", "solid_mask", "layered"]
    if objective in {"utility_priority", "utility_under_privacy_floor"}:
        eligible.append("pixelate")
    if compute_profile in {"low_vram_local_gpu", "stronger_accelerator_constrained_compute"} and face_count == 1 and dominant_height >= 180:
        rejected["NullFace"] = "requires_apparent_demographic_and_quality_gate_before_promotion"
    if compute_profile == "stronger_accelerator_constrained_compute" and face_count >= 1 and dominant_height >= 150:
        eligible.append("diffusion_low_step_research_candidate")
        rejected["diffusion_low_step"] = "eligible_as_full_comparable_research_branch_but_not_enabled_in_local_deterministic_demo"

    multimodal_actions: list[str] = []
    if text_risk in {"high", "possible"}:
        multimodal_actions.append("text_redaction_demo_region")
    if screen_risk in {"high", "possible"}:
        multimodal_actions.append("screen_redaction_demo_region")

    selected = "blur"
    action = "face_anonymisation_blur"
    fallback = "blur"
    privacy_floor = "passed_by_confirmatory_or_privacy_first_method"
    utility_status = "not_optimized"
    runtime_status = "within_local_deterministic_budget"
    quality_gate = "passed_for_deterministic_method"
    demographic_gate = "not_applicable_for_deterministic_method"
    evidence = "confirmatory"
    expected_utility = "medium"
    expected_runtime = "low"
    residual = "residual text/screen risk depends on detector coverage and demo redaction boundary"
    explanation = "Blur selected as practical balanced fallback."

    if face_count == 0 and face_risk == "none" and not multimodal_actions:
        selected = "copy"
        action = "copy_without_privacy_action"
        privacy_floor = "not_applicable_no_detected_or_reviewed_risk"
        utility_status = "preserved"
        quality_gate = "not_applicable"
        evidence = "demo"
        residual = "no face/text/screen risk in review metadata; residual unknown risk still possible"
        explanation = "No face boxes and no reviewed non-face risk."
    elif objective == "privacy_first":
        if multimodal_actions:
            selected = "solid_mask"
            action = "face_anonymisation_plus_multimodal_demo_redaction"
            expected_utility = "high_cost"
            explanation = "Privacy-first objective selects solid mask when text/screen risk is present."
        elif face_count >= 1:
            selected = "solid_mask"
            action = "face_anonymisation_solid_mask"
            expected_utility = "high_cost"
            explanation = "Privacy-first objective selects solid masking because the final RQ2 evidence ranks it as the strongest privacy-first fallback."
        else:
            selected = "blur"
            explanation = "Privacy-first objective keeps blur when stronger action is not triggered."
    elif objective in {"utility_priority", "utility_under_privacy_floor"}:
        if text_risk in {"high", "possible"} or screen_risk in {"high", "possible"}:
            selected = "solid_mask"
            action = "face_anonymisation_plus_multimodal_demo_redaction"
            utility_status = "utility_deprioritized_due_to_non_face_privacy_risk"
            explanation = "Utility mode cannot override high text/screen privacy risk."
        elif artifact == "low_method_failure_risk" and face_risk != "high":
            selected = "pixelate"
            privacy_floor = "bounded_low_privacy_utility_mode"
            utility_status = "high_utility_low_privacy"
            evidence = "bounded"
            expected_utility = "low_cost"
            explanation = "Pixelate selected only in utility-priority mode where reviewed risk is not high."
        else:
            selected = "blur"
            utility_status = "privacy_floor_prevents_pixelate"
            explanation = "Utility recovery rejected because privacy/failure risk is not low."
    elif objective == "runtime_aware":
        selected = "blur" if artifact == "high_method_failure_risk" or face_count <= 1 else "layered"
        action = f"face_anonymisation_{selected}"
        runtime_status = "fast_deterministic_method_selected"
        explanation = "Runtime-aware mode uses deterministic methods and rejects RP because the final RP evidence is partial-comparable with high runtime and method failures."
    elif objective == "compute_profile_adaptive":
        if compute_profile in {"cpu_or_very_low_resource", "unknown_or_unstable_profile"}:
            selected = "blur"
            explanation = "Compute-profile mode fails closed to blur on constrained/unknown profile."
        elif multimodal_actions:
            selected = "solid_mask"
            action = "face_anonymisation_plus_multimodal_demo_redaction"
            explanation = "Compute-profile mode allows stronger deterministic privacy action for multimodal risk."
        elif face_count >= 2 and face_risk == "high":
            selected = "layered"
            explanation = "Compute-profile mode selects layered deterministic method for high multiface risk."
        else:
            selected = "blur"
            explanation = "Compute-profile mode keeps balanced deterministic fallback; diffusion_low_step is logged as a full-comparable research candidate only when quality-gated."
    elif objective == "failure_avoidance":
        selected = "blur" if artifact in {"high_method_failure_risk", "medium_method_failure_risk"} or "small" in pose_condition else "layered"
        action = f"face_anonymisation_{selected}"
        explanation = "Failure-avoidance mode rejects fragile or partial-comparable advanced methods, including RP and quality-gated Diffusion, and uses safe deterministic methods."
    elif objective == "multimodal_risk":
        if multimodal_actions:
            selected = "solid_mask"
            action = "face_anonymisation_plus_multimodal_demo_redaction"
            explanation = "Multimodal-risk mode prioritizes non-face PII response."
        elif face_count == 0:
            selected = "copy"
            action = "copy_without_face_action"
            explanation = "No reviewed multimodal risk and no face boxes."
        else:
            selected = "blur"
            explanation = "Multimodal-risk mode falls back to face blur."

    if selected == "pixelate":
        rejected["blur"] = "not_selected_in_low_risk_utility_mode"
    if selected == "solid_mask":
        rejected["pixelate"] = "privacy_floor_not_sufficient_for_text_or_screen_risk"
    if selected == "layered":
        rejected["pixelate"] = "privacy_floor_not_sufficient_for_high_face_risk"

    if selected in {"solid_mask", "layered", "blur"} and face_count == 0 and face_risk != "none":
        residual = "review metadata indicates possible risk but no face boxes were available; method can only process available regions"

    source_artifacts = [
        "outputs/04_multimodal_privacy/03_residual_leakage_review.csv",
        "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv",
        "outputs/09_traceability/01_evidence_index.csv",
        "outputs/03_anonymisation/01_all_methods_comparison.csv",
    ]
    return SelectorDecision(
        selected_method=selected,
        selected_action=action,
        fallback_method=fallback,
        eligible_methods=eligible,
        rejected_methods=sorted(rejected),
        rejection_reasons=rejected,
        objective_mode=objective,
        compute_profile=compute_profile,
        detector_mode=detector_mode,
        resolution_path=resolution,
        face_risk=face_risk,
        text_risk=text_risk,
        screen_risk=screen_risk,
        multimodal_actions=multimodal_actions,
        privacy_floor_status=privacy_floor,
        utility_status=utility_status,
        runtime_status=runtime_status,
        quality_gate_status=quality_gate,
        apparent_demographic_quality_gate_status=demographic_gate,
        evidence_level=evidence,
        source_artifacts=source_artifacts,
        residual_risk_note=residual,
        explanation=explanation,
        face_count=face_count,
        dominant_face_height_px=dominant_height,
        dominant_face_area_ratio=round(area_ratio, 6),
        face_confidence_min=conf_min,
        face_confidence_max=conf_max,
        face_confidence_mean=conf_mean,
        pose_condition=pose_condition,
        source_quality_condition=quality_condition,
        expected_utility_cost=expected_utility,
        expected_runtime_cost=expected_runtime,
        resource_concurrency_policy="resource-derived concurrency rather than a fixed cap",
    )


def resize_for_processing(image: Image.Image, resolution: str) -> tuple[Image.Image, float]:
    if resolution == "native":
        return image, 1.0
    scale = 0.5 if resolution == "half" else 0.25
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR), scale


def scale_boxes(boxes: list[tuple[int, int, int, int]], scale: float) -> list[tuple[int, int, int, int]]:
    if scale == 1.0:
        return boxes
    return [(int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)) for x1, y1, x2, y2 in boxes]


def side_by_side(original: Image.Image, anonymised: Image.Image) -> Image.Image:
    width = min(1200, original.width)
    scale = width / original.width
    left = original.resize((width, int(original.height * scale)), Image.Resampling.BILINEAR)
    right = anonymised.resize(left.size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (left.width * 2, left.height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def manifest_rows(input_path: Path | None, input_dir: Path | None, manifest: Path | None, limit: int | None) -> list[dict[str, str]]:
    if manifest:
        rows = read_csv(manifest)
        if not rows:
            return []
        if "local_path" in rows[0]:
            selected = [{"relative_path": row.get("image_id", Path(row["local_path"]).name), "local_path": row["local_path"], **row} for row in rows]
        elif "relative_path" in rows[0]:
            selected = [{"relative_path": row["relative_path"], "image_id": row["relative_path"], "local_path": str(RAW_ROOT / row["relative_path"]), **row} for row in rows]
        else:
            raise ValueError("Manifest must contain local_path/image_id or relative_path.")
    elif input_path:
        selected = [{"relative_path": input_path.name, "image_id": input_path.name, "local_path": str(input_path)}]
    elif input_dir:
        selected = [
            {"relative_path": p.name, "image_id": p.name, "local_path": str(p)}
            for p in sorted(input_dir.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    else:
        raise ValueError("Provide --input, --input-dir, or --manifest.")
    return selected[:limit] if limit else selected


def process_one(
    index: int,
    row: dict[str, str],
    args: argparse.Namespace,
    detections_by_image: dict[str, DetectionSummary],
    profile: dict[str, Any],
    images_dir: Path,
    side_by_side_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None, float, str]:
    image_id = row.get("image_id") or row.get("relative_path") or Path(row["local_path"]).name
    start = time.perf_counter()
    try:
        input_path = Path(row["local_path"])
        original = Image.open(input_path).convert("RGB")
        image, scale = resize_for_processing(original, args.resolution)
        detections = detections_by_image.get(image_id, DetectionSummary([], [], [], []))
        decision = None
        if args.mode in {"objective_profile", "profile"}:
            # Select the face operator from the policy registry.
            from src.policy.registry import get_profile

            focus = {
                "privacy_first": "privacy",
                "utility_under_privacy_floor": "balanced",
                "utility_priority": "utility",
                "runtime_aware": "balanced",
                "compute_profile_adaptive": "balanced",
                "failure_avoidance": "privacy",
                "multimodal_risk": "privacy",
            }.get(str(args.objective), "balanced")
            prof = get_profile(focus)
            face_op = str(prof["face_anonymisation"])
            method = "copy" if not detections.boxes else face_op
            reason = (
                f"objective_profile focus={focus} face_anonymisation={face_op} "
                f"(App policy; not scientific 286/81/133 OAPR)"
            )
        else:
            method = args.mode
            reason = f"fixed {args.mode} mode"
        boxes = scale_boxes(detections.boxes, scale)
        if method == "copy":
            anonymised = image.copy()
        else:
            anonymised = METHODS[method](image, boxes)
        if decision and decision.multimodal_actions:
            anonymised = apply_text_screen_redaction(anonymised, row)
        if scale != 1.0:
            anonymised = anonymised.resize(original.size, Image.Resampling.BILINEAR)
        output_path = images_dir / Path(image_id).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anonymised.save(output_path, quality=90)
        sbs_path = side_by_side_dir / Path(image_id).with_suffix(".jpg")
        sbs_path.parent.mkdir(parents=True, exist_ok=True)
        side_by_side(original, anonymised).save(sbs_path, quality=88)
        elapsed = time.perf_counter() - start
        status = "ok"
        error = ""
    except Exception as exc:  # pragma: no cover - validation path
        elapsed = time.perf_counter() - start
        method = "error"
        reason = "exception"
        output_path = Path("")
        sbs_path = Path("")
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        decision = None

    decision_payload = asdict(decision) if decision else {}
    route_row: dict[str, object] = {
        "case_id": f"app_demo_{index:03d}",
        "image_id": image_id,
        "mode": args.mode,
        "selected_method": method,
        "route_reason": reason,
        "boxes_available": len(detections_by_image.get(image_id, DetectionSummary([], [], [], [])).boxes),
        "runtime_seconds": round(elapsed, 6),
        "status": status,
        "error": error,
        "device_summary": profile.get("device_summary", ""),
        **{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in decision_payload.items()},
    }
    manifest_row = {
        "image_id": image_id,
        "input_path": row.get("local_path", ""),
        "output_path": str(output_path),
        "side_by_side_path": str(sbs_path),
        "selected_method": method,
        "objective_mode": args.objective,
        "status": status,
    }
    failure = {"image_id": image_id, "status": status, "error": error} if status != "ok" else None
    return route_row, manifest_row, failure, elapsed, method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--mode",
        choices=["blur", "pixelate", "solid_mask", "layered", "objective_profile"],
        default="objective_profile",
    )
    parser.add_argument(
        "--objective",
        choices=[
            "privacy_first",
            "utility_priority",
            "utility_under_privacy_floor",
            "runtime_aware",
            "compute_profile_adaptive",
            "failure_avoidance",
            "multimodal_risk",
        ],
        default="utility_under_privacy_floor",
    )
    parser.add_argument("--resolution", choices=["native", "half", "quarter"], default="native")
    parser.add_argument("--detector-mode", default="cached_yolo_scrfd_fallback")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--detections", nargs="*", type=Path, default=[
        ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv",
        ROOT / "outputs/yolo_scrfd_fallback_predictions_reviewed.csv",
        ROOT / "outputs/detection_eval_subset_yolo_scrfd_fallback.csv",
        ROOT / "outputs/02_face_detection/06_detection_eval_subset_yolo_scrfd_fallback.csv",
    ])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/runs/app_sample_outputs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    side_by_side_dir = output_dir / "side_by_side"
    images_dir.mkdir(parents=True, exist_ok=True)
    side_by_side_dir.mkdir(parents=True, exist_ok=True)

    profile = build_profile()
    worker_count = max(1, args.workers if args.workers is not None else int(profile["workers"]))
    detections_by_image = load_detections(args.detections)
    rows = manifest_rows(args.input, args.input_dir, args.manifest, args.limit)

    route_rows: list[dict[str, object]] = []
    manifest_out: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    runtimes: list[float] = []
    method_counts: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_one, index, row, args, detections_by_image, profile, images_dir, side_by_side_dir)
            for index, row in enumerate(rows, start=1)
        ]
        for future in as_completed(futures):
            route_row, manifest_row, failure, elapsed, method = future.result()
            route_rows.append(route_row)
            manifest_out.append(manifest_row)
            if failure:
                failures.append(failure)
            if route_row["status"] == "ok":
                runtimes.append(elapsed)
                method_counts[method] += 1

    route_rows.sort(key=lambda row: str(row["case_id"]))
    manifest_out.sort(key=lambda row: str(row["image_id"]))
    route_fields = [
        "case_id", "image_id", "mode", "objective_mode", "compute_profile", "detector_mode", "resolution_path",
        "selected_method", "selected_action", "fallback_method", "route_reason", "eligible_methods",
        "rejected_methods", "rejection_reasons", "privacy_floor_status", "utility_status", "runtime_status",
        "quality_gate_status", "apparent_demographic_quality_gate_status", "face_risk", "text_risk", "screen_risk",
        "multimodal_actions", "face_count", "boxes_available", "dominant_face_height_px", "dominant_face_area_ratio",
        "face_confidence_min", "face_confidence_max", "face_confidence_mean", "pose_condition",
        "source_quality_condition", "expected_utility_cost", "expected_runtime_cost", "evidence_level",
        "source_artifacts", "residual_risk_note", "explanation", "resource_concurrency_policy", "device_summary",
        "runtime_seconds", "status", "error",
    ]
    write_csv(output_dir / "routing_log.csv", route_rows, route_fields)
    write_csv(output_dir / "per_image_manifest.csv", manifest_out, ["image_id", "input_path", "output_path", "side_by_side_path", "selected_method", "objective_mode", "status"])
    write_csv(output_dir / "failure_log.csv", failures, ["image_id", "status", "error"])
    summary = {
        "frames_requested": len(rows),
        "frames_ok": sum(1 for row in route_rows if row["status"] == "ok"),
        "frames_failed": len(failures),
        "mode": args.mode,
        "objective": args.objective,
        "resolution": args.resolution,
        "workers": worker_count,
        "compute_profile": profile["compute_profile"],
        "method_counts": dict(method_counts),
        "runtime_total_seconds": round(sum(runtimes), 6),
        "runtime_mean_seconds": round(sum(runtimes) / len(runtimes), 6) if runtimes else 0.0,
        "output_dir": str(output_dir),
    }
    (output_dir / "runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copyfile(output_dir / "routing_log.csv", output_dir.parent / "app_routing_log.csv")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
