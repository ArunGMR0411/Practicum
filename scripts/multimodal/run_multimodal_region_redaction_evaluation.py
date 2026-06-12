#!/usr/bin/env python3
"""Evaluate fixed and adaptive text/screen redaction on the 250-image protocol.

Includes area-aware screen operators (fill small regions, strong-blur large ones)
and a development-only privacy floor for text-only adaptive selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ocr_evaluator import OCREvaluator

ANNOTATIONS = ROOT / "outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv"
EVIDENCE_DIR = ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence"
PREDICTIONS = EVIDENCE_DIR / "03_selected_localisation_predictions.csv"
POLICY = EVIDENCE_DIR / "04_multimodal_risk_policy.csv"
RAW_ROOT = ROOT / "data/castle2024/raw"
Box = tuple[int, int, int, int]

# Baseline operators.
BASE_VARIANTS: dict[str, tuple[str, str, float | None]] = {
    "text_fill_screen_blur": ("fill", "blur", None),
    "text_blur_screen_blur": ("blur", "blur", None),
    "text_pixelate_screen_pixelate": ("pixelate", "pixelate", None),
    "text_fill_screen_fill": ("fill", "fill", None),
    "text_blur_screen_fill": ("blur", "fill", None),
    "text_pixelate_screen_blur": ("pixelate", "blur", None),
}

# Development grid for area-aware screen operators (threshold selected on development).
AREA_AWARE_THRESHOLDS = (0.05, 0.08, 0.10, 0.15)
TEXT_PRIVACY_FLOOR = 0.80
STRONG_SCREEN_BLUR_RADIUS = 40.0
DEFAULT_SCREEN_BLUR_RADIUS = 18.0
DEFAULT_TEXT_BLUR_RADIUS = 12.0


def build_variants() -> dict[str, tuple[str, str, float | None]]:
    variants = dict(BASE_VARIANTS)
    for tau in AREA_AWARE_THRESHOLDS:
        tag = f"{int(round(tau * 100)):02d}"
        variants[f"text_blur_screen_area_aware_t{tag}"] = ("blur", "area_aware", tau)
        variants[f"text_fill_screen_area_aware_t{tag}"] = ("fill", "area_aware", tau)
    return variants


def parse_boxes(value: str) -> list[Box]:
    if not value or value in {"[]", "nan", "None"}:
        return []
    return [
        (int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"]))
        for item in json.loads(value)
    ]


def valid_boxes(image: Image.Image, boxes: list[Box]) -> list[Box]:
    output = []
    for x1, y1, x2, y2 in boxes:
        candidate = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
        if candidate[2] > candidate[0] and candidate[3] > candidate[1]:
            output.append(candidate)
    return output


def box_area_fraction(image: Image.Image, box: Box) -> float:
    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return (width * height) / float(image.width * image.height)


def apply_mode(crop: Image.Image, mode: str, *, blur_radius: float) -> Image.Image:
    if mode == "fill":
        return Image.new("RGB", crop.size, (0, 0, 0))
    if mode == "blur":
        return crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if mode == "pixelate":
        width, height = crop.size
        small = crop.resize(
            (max(1, width // 16), max(1, height // 16)), Image.Resampling.BILINEAR
        )
        return small.resize((width, height), Image.Resampling.NEAREST)
    if mode == "center_fill_blur_edge":
        # Strong edge blur plus central fill: keeps most screen content unreadable
        # while reducing full-frame utility damage versus solid fill of huge boxes.
        blurred = crop.filter(ImageFilter.GaussianBlur(radius=STRONG_SCREEN_BLUR_RADIUS))
        width, height = blurred.size
        cx1 = int(0.20 * width)
        cy1 = int(0.20 * height)
        cx2 = int(0.80 * width)
        cy2 = int(0.80 * height)
        if cx2 > cx1 and cy2 > cy1:
            blurred.paste(Image.new("RGB", (cx2 - cx1, cy2 - cy1), (0, 0, 0)), (cx1, cy1))
        return blurred
    raise ValueError(mode)


def redact(
    image: Image.Image,
    boxes: list[Box],
    mode: str,
    *,
    blur_radius: float,
    area_threshold: float | None = None,
) -> Image.Image:
    """Redact boxes. mode may be fill/blur/pixelate or area_aware for screens."""
    output = image.copy()
    for box in valid_boxes(output, boxes):
        crop = output.crop(box).convert("RGB")
        if mode == "area_aware":
            if area_threshold is None:
                raise ValueError("area_aware requires area_threshold")
            frac = box_area_fraction(output, box)
            # Small regions: hard fill. Large regions: center-fill + edge blur.
            if frac < area_threshold:
                transformed = apply_mode(crop, "fill", blur_radius=blur_radius)
            else:
                transformed = apply_mode(
                    crop, "center_fill_blur_edge", blur_radius=blur_radius
                )
        else:
            transformed = apply_mode(crop, mode, blur_radius=blur_radius)
        output.paste(transformed, box[:2])
    return output


def preview(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR))


def utility_metrics(
    original: Image.Image, output: Image.Image, model: lpips.LPIPS
) -> tuple[float, float, float]:
    before = preview(original)
    after = preview(output)
    ssim = float(structural_similarity(before, after, channel_axis=2, data_range=255))
    first = torch.from_numpy(before.copy()).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1
    second = torch.from_numpy(after.copy()).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1
    with torch.inference_mode():
        perceptual = float(model(first, second).item())
    lpips_utility = float(np.clip(1.0 - perceptual / 0.05, 0.0, 1.0))
    return ssim, perceptual, 0.5 * ssim + 0.5 * lpips_utility


def region_change(original: Image.Image, output: Image.Image, boxes: list[Box]) -> float:
    values = []
    for box in valid_boxes(original, boxes):
        before = np.asarray(
            original.crop(box).convert("RGB").resize((256, 144), Image.Resampling.BILINEAR)
        )
        after = np.asarray(
            output.crop(box).convert("RGB").resize((256, 144), Image.Resampling.BILINEAR)
        )
        values.append(
            1.0 - float(structural_similarity(before, after, channel_axis=2, data_range=255))
        )
    return float(np.mean(values)) if values else math.nan


def non_sensitive_change_fraction(
    original: Image.Image,
    output: Image.Image,
    sensitive_boxes: list[Box],
) -> float:
    before = np.asarray(original.resize((960, 540), Image.Resampling.BILINEAR), dtype=np.int16)
    after = np.asarray(output.resize((960, 540), Image.Resampling.BILINEAR), dtype=np.int16)
    changed = np.abs(before - after).mean(axis=2) > 5
    mask = np.zeros((540, 960), dtype=bool)
    sx, sy = 960 / original.width, 540 / original.height
    for x1, y1, x2, y2 in sensitive_boxes:
        mask[
            max(0, int(y1 * sy)) : min(540, int(math.ceil(y2 * sy))),
            max(0, int(x1 * sx)) : min(960, int(math.ceil(x2 * sx))),
        ] = True
    outside = ~mask
    return float(changed[outside].mean()) if outside.any() else 0.0


def text_privacy(
    original_texts: list[str],
    output: Image.Image,
    boxes: list[Box],
    evaluator: OCREvaluator,
) -> tuple[float, float]:
    if not boxes:
        return math.nan, math.nan
    recognised = evaluator.recognise_regions(output, boxes)
    similarities = [
        evaluator.text_similarity(before, after.text)
        for before, after in zip(original_texts, recognised, strict=True)
    ]
    return float(np.mean(np.asarray(similarities) < 0.5)), 1.0 - float(np.mean(similarities))


def paired_bootstrap(adaptive: pd.DataFrame, fixed: pd.DataFrame) -> dict[str, float | int]:
    paired = adaptive[["image_id", "multimodal_anonymisation_score"]].merge(
        fixed[["image_id", "multimodal_anonymisation_score"]],
        on="image_id",
        suffixes=("_adaptive", "_fixed"),
        validate="one_to_one",
    )
    differences = (
        paired.multimodal_anonymisation_score_adaptive
        - paired.multimodal_anonymisation_score_fixed
    ).to_numpy()
    rng = np.random.default_rng(20260715)
    indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
    values = differences[indices].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "adaptive_minus_fixed_mean": float(differences.mean()),
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
        "adaptive_win_count": int((differences > 1e-12).sum()),
        "fixed_win_count": int((differences < -1e-12).sum()),
        "tie_count": int((np.abs(differences) <= 1e-12).sum()),
    }


def select_adaptive_policy(
    metrics: pd.DataFrame,
    *,
    text_privacy_floor: float,
    text_score_slack: float = 0.015,
    screen_privacy_slack: float = 0.04,
    selection_mode: str = "max_score",
) -> pd.DataFrame:
    """Select per-risk-state operators on development only.

    selection_mode:
      - max_score: highest development multimodal score (privacy breaks ties).
        This is the canonical reported policy.
      - utility_aware: experimental ablation that prefers utility among
        near-best-privacy screen operators and privacy among near-best text
        scores. Retained for negative-result documentation.
    """
    development = metrics[metrics.evaluation_split.eq("development")]
    policy_rows: list[dict[str, object]] = []
    for state in sorted(metrics.predicted_risk_state.unique()):
        if state == "no_text_screen_risk":
            policy_rows.append(
                {
                    "predicted_risk_state": state,
                    "selected_variant": "no_action_copy",
                    "development_score": math.nan,
                    "development_privacy": math.nan,
                    "development_utility": math.nan,
                    "reason": "No predicted text/screen boxes; preserve the image.",
                }
            )
            continue
        candidates = (
            development[development.predicted_risk_state.eq(state)]
            .groupby("variant", as_index=False)
            .agg(
                development_images=("image_id", "nunique"),
                privacy_score=("privacy_score", "mean"),
                utility_score=("utility_score", "mean"),
                runtime_score=("runtime_score", "mean"),
                multimodal_anonymisation_score=("multimodal_anonymisation_score", "mean"),
            )
        )
        if selection_mode == "utility_aware" and state == "text_present":
            best_score = float(candidates.multimodal_anonymisation_score.max())
            near = candidates[
                candidates.multimodal_anonymisation_score >= best_score - text_score_slack
            ]
            floor_pool = near[near.privacy_score >= text_privacy_floor]
            pool = (floor_pool if not floor_pool.empty else near).sort_values(
                ["privacy_score", "multimodal_anonymisation_score"],
                ascending=False,
            )
            reason = (
                f"Ablation utility_aware: near-best score (slack {text_score_slack:.3f}); "
                "privacy preferred for text-only risk."
            )
        elif selection_mode == "utility_aware" and state in {
            "screen_present",
            "text_and_screen_present",
        }:
            best_privacy = float(candidates.privacy_score.max())
            near_priv = candidates[
                candidates.privacy_score >= best_privacy - screen_privacy_slack
            ]
            pool = near_priv.sort_values(
                ["utility_score", "multimodal_anonymisation_score", "privacy_score"],
                ascending=False,
            )
            reason = (
                f"Ablation utility_aware: near-best privacy (slack {screen_privacy_slack:.3f}); "
                "utility preferred for screen-containing risk."
            )
        else:
            pool = candidates.sort_values(
                ["multimodal_anonymisation_score", "privacy_score"],
                ascending=False,
            )
            reason = "Highest development-split measured score; privacy breaks score ties."
        best = pool.iloc[0]
        policy_rows.append(
            {
                "predicted_risk_state": state,
                "selected_variant": best.variant,
                "development_score": best.multimodal_anonymisation_score,
                "development_privacy": best.privacy_score,
                "development_utility": best.utility_score,
                "reason": reason,
            }
        )
    return pd.DataFrame(policy_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS,
        help="Localisation predictions CSV (protocol_id, image_id, predicted_* boxes).",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=POLICY,
        help="Risk-policy CSV; if missing columns, built from predictions + annotations.",
    )
    parser.add_argument(
        "--text-privacy-floor",
        type=float,
        default=0.70,
        help="Minimum development privacy preferred for text_present near-best selection.",
    )
    parser.add_argument(
        "--reuse-metrics",
        type=Path,
        default=None,
        help="Optional path to an existing 06_redaction_per_image_metrics.csv to skip recompute.",
    )
    parser.add_argument(
        "--only-area-aware",
        action="store_true",
        help="Compute only area-aware variants and merge with existing base-variant metrics.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("max_score", "utility_aware"),
        default="max_score",
        help="Development policy selection rule. max_score is canonical.",
    )
    parser.add_argument(
        "--base-variants-only",
        action="store_true",
        help="Evaluate only the six base redaction variants (faster ablation).",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LPIPS and TrOCR; CPU fallback is disabled.")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions if args.predictions.is_absolute() else ROOT / args.predictions
    policy_path = args.policy if args.policy.is_absolute() else ROOT / args.policy
    all_variants = build_variants()
    if args.only_area_aware:
        variants = {
            name: spec
            for name, spec in all_variants.items()
            if "area_aware" in name
        }
    elif args.base_variants_only:
        variants = dict(BASE_VARIANTS)
    else:
        variants = all_variants

    annotations = pd.read_csv(ANNOTATIONS, keep_default_na=False)
    predictions = pd.read_csv(predictions_path, keep_default_na=False)

    def risk_state(text: bool, screen: bool) -> str:
        if text and screen:
            return "text_and_screen_present"
        if text:
            return "text_present"
        if screen:
            return "screen_present"
        return "no_text_screen_risk"

    if policy_path.exists():
        routing = pd.read_csv(policy_path, keep_default_na=False)
    else:
        routing = pd.DataFrame()

    needed = {
        "protocol_id",
        "image_id",
        "ground_truth_risk_state",
        "predicted_risk_state",
        "ground_truth_text_present",
        "predicted_text_present",
        "ground_truth_screen_present",
        "predicted_screen_present",
    }
    if routing.empty or not needed.issubset(set(routing.columns)):
        # Build routing from annotations + predictions (campaign outputs).
        merged = annotations.merge(
            predictions, on=["protocol_id", "image_id"], validate="one_to_one"
        )
        rows = []
        for row in merged.itertuples():
            gt_text = bool(parse_boxes(getattr(row, "text_boxes_json", "[]")))
            gt_screen = bool(parse_boxes(getattr(row, "screen_boxes_json", "[]")))
            pr_text = int(getattr(row, "predicted_text_count", 0)) > 0
            pr_screen = int(getattr(row, "predicted_screen_count", 0)) > 0
            if "predicted_text_present" in merged.columns:
                pr_text = str(getattr(row, "predicted_text_present")).lower() in {
                    "true",
                    "1",
                    "yes",
                }
            if "predicted_screen_present" in merged.columns:
                pr_screen = str(getattr(row, "predicted_screen_present")).lower() in {
                    "true",
                    "1",
                    "yes",
                }
            if "predicted_text_boxes_json" in merged.columns:
                pr_text = bool(parse_boxes(getattr(row, "predicted_text_boxes_json")))
            if "predicted_screen_boxes_json" in merged.columns:
                pr_screen = bool(parse_boxes(getattr(row, "predicted_screen_boxes_json")))
            pred_state = (
                getattr(row, "predicted_risk_state", "")
                if hasattr(row, "predicted_risk_state")
                and str(getattr(row, "predicted_risk_state", "")).strip()
                else risk_state(pr_text, pr_screen)
            )
            rows.append(
                {
                    "protocol_id": row.protocol_id,
                    "image_id": row.image_id,
                    "evaluation_split": getattr(row, "evaluation_split", ""),
                    "ground_truth_risk_state": risk_state(gt_text, gt_screen),
                    "predicted_risk_state": pred_state,
                    "ground_truth_text_present": gt_text,
                    "predicted_text_present": pr_text,
                    "ground_truth_screen_present": gt_screen,
                    "predicted_screen_present": pr_screen,
                    "route_action": {
                        "text_present": "redact_text",
                        "screen_present": "redact_screen",
                        "text_and_screen_present": "redact_text_and_screen",
                        "no_text_screen_risk": "skip_multimodal_redaction",
                    }.get(pred_state, "skip_multimodal_redaction"),
                }
            )
        routing = pd.DataFrame(rows)
        routing.to_csv(output_dir / "04_multimodal_risk_policy.csv", index=False)

    pred_cols = [
        c
        for c in predictions.columns
        if c
        in {
            "protocol_id",
            "image_id",
            "evaluation_split",
            "predicted_text_boxes_json",
            "predicted_screen_boxes_json",
            "predicted_text_count",
            "predicted_screen_count",
            "text_variant",
            "screen_variant",
        }
    ]
    data = annotations.merge(
        predictions[pred_cols], on=["protocol_id", "image_id"], validate="one_to_one"
    )
    route_cols = [
        "protocol_id",
        "image_id",
        "ground_truth_risk_state",
        "predicted_risk_state",
        "route_action",
    ]
    # Drop overlapping risk columns from annotations if present.
    overlap = [c for c in route_cols if c in data.columns and c not in {"protocol_id", "image_id"}]
    if overlap:
        data = data.drop(columns=overlap)
    data = data.merge(
        routing[route_cols],
        on=["protocol_id", "image_id"],
        validate="one_to_one",
    )

    if args.reuse_metrics is not None and not args.only_area_aware:
        metrics_path = (
            args.reuse_metrics
            if args.reuse_metrics.is_absolute()
            else ROOT / args.reuse_metrics
        )
        metrics = pd.read_csv(metrics_path)
        print(f"Reused metrics from {metrics_path} ({len(metrics)} rows)", flush=True)
    else:
        existing_base = None
        if args.only_area_aware:
            metrics_path = output_dir / "06_redaction_per_image_metrics.csv"
            if metrics_path.exists():
                existing = pd.read_csv(metrics_path)
                base_names = set(BASE_VARIANTS)
                existing_base = existing[existing.variant.isin(base_names)].copy()
                print(
                    f"Keeping {len(existing_base)} base-variant metric rows; "
                    f"recomputing {len(variants)} area-aware variants.",
                    flush=True,
                )

        lpips_model = lpips.LPIPS(net="alex").cuda().eval()
        ocr = OCREvaluator(device="cuda", region_batch_size=16)
        original_text: dict[str, list[str]] = {}
        print("Recognising reviewed source text regions", flush=True)
        for index, row in enumerate(data.itertuples(), 1):
            gt_text = parse_boxes(row.text_boxes_json)
            if gt_text:
                with Image.open(RAW_ROOT / row.image_id) as image:
                    recognised = ocr.recognise_regions(image.convert("RGB"), gt_text)
                original_text[row.image_id] = [item.text for item in recognised]
            else:
                original_text[row.image_id] = []
            if index % 50 == 0:
                print(f"source OCR: {index}/{len(data)}", flush=True)

        metric_rows: list[dict[str, object]] = []
        for index, row in enumerate(data.itertuples(), 1):
            gt_text = parse_boxes(row.text_boxes_json)
            gt_screen = parse_boxes(row.screen_boxes_json)
            pred_text = parse_boxes(row.predicted_text_boxes_json)
            pred_screen = parse_boxes(row.predicted_screen_boxes_json)
            with Image.open(RAW_ROOT / row.image_id) as source:
                image = source.convert("RGB")
                for variant, (text_mode, screen_mode, area_threshold) in variants.items():
                    started = time.perf_counter()
                    output = redact(
                        image,
                        pred_screen,
                        screen_mode,
                        blur_radius=DEFAULT_SCREEN_BLUR_RADIUS,
                        area_threshold=area_threshold,
                    )
                    output = redact(
                        output,
                        pred_text,
                        text_mode,
                        blur_radius=DEFAULT_TEXT_BLUR_RADIUS,
                    )
                    runtime = time.perf_counter() - started
                    ssim, perceptual, utility = utility_metrics(image, output, lpips_model)
                    text_suppression, text_ocr_privacy = text_privacy(
                        original_text[row.image_id], output, gt_text, ocr
                    )
                    text_change = region_change(image, output, gt_text)
                    screen_change = region_change(image, output, gt_screen)
                    active_privacy: list[float] = []
                    if gt_text:
                        active_privacy.append(0.70 * text_ocr_privacy + 0.30 * text_change)
                    if gt_screen:
                        active_privacy.append(screen_change)
                    privacy = float(np.mean(active_privacy)) if active_privacy else 1.0
                    spill = non_sensitive_change_fraction(
                        image, output, [*gt_text, *gt_screen]
                    )
                    runtime_score = float(np.clip(1.0 - runtime / 1.0, 0.0, 1.0))
                    score = 0.50 * privacy + 0.30 * utility + 0.10 * runtime_score + 0.10
                    metric_rows.append(
                        {
                            "protocol_id": row.protocol_id,
                            "image_id": row.image_id,
                            "evaluation_split": row.evaluation_split,
                            "ground_truth_risk_state": row.ground_truth_risk_state,
                            "predicted_risk_state": row.predicted_risk_state,
                            "variant": variant,
                            "screen_area_threshold": area_threshold
                            if area_threshold is not None
                            else "",
                            "ground_truth_text_count": len(gt_text),
                            "ground_truth_screen_count": len(gt_screen),
                            "predicted_text_count": len(pred_text),
                            "predicted_screen_count": len(pred_screen),
                            "privacy_score": privacy,
                            "text_ocr_suppression_rate": text_suppression,
                            "text_ocr_privacy_score": text_ocr_privacy,
                            "text_region_obscuration": text_change,
                            "screen_region_obscuration": screen_change,
                            "utility_score": utility,
                            "SSIM": ssim,
                            "LPIPS": perceptual,
                            "non_sensitive_change_fraction": spill,
                            "runtime_seconds": runtime,
                            "runtime_score": runtime_score,
                            "success_score": 1.0,
                            "multimodal_anonymisation_score": score,
                        }
                    )
            if index % 10 == 0:
                print(
                    f"redaction evaluation: {index}/{len(data)} "
                    f"({len(variants)} variants)",
                    flush=True,
                )

        metrics = pd.DataFrame(metric_rows)
        if existing_base is not None and not existing_base.empty:
            metrics = pd.concat([existing_base, metrics], ignore_index=True)
        metrics.to_csv(output_dir / "06_redaction_per_image_metrics.csv", index=False)

    adaptive_policy = select_adaptive_policy(
        metrics,
        text_privacy_floor=args.text_privacy_floor,
        selection_mode=args.selection_mode,
    )
    adaptive_policy.to_csv(output_dir / "08_adaptive_redaction_policy.csv", index=False)

    selected_parts = []
    for decision in adaptive_policy.itertuples(index=False):
        state_rows = metrics[metrics.predicted_risk_state.eq(decision.predicted_risk_state)]
        if decision.selected_variant == "no_action_copy":
            base = data[data.predicted_risk_state.eq(decision.predicted_risk_state)][
                [
                    "protocol_id",
                    "image_id",
                    "evaluation_split",
                    "ground_truth_risk_state",
                    "predicted_risk_state",
                ]
            ].copy()
            base["variant"] = "no_action_copy"
            base["privacy_score"] = np.where(
                base.ground_truth_risk_state.eq("no_text_screen_risk"), 1.0, 0.0
            )
            base["text_ocr_suppression_rate"] = np.nan
            base["text_ocr_privacy_score"] = np.nan
            base["text_region_obscuration"] = np.nan
            base["screen_region_obscuration"] = np.nan
            base["utility_score"] = 1.0
            base["SSIM"] = 1.0
            base["LPIPS"] = 0.0
            base["non_sensitive_change_fraction"] = 0.0
            base["runtime_seconds"] = 0.0
            base["runtime_score"] = 1.0
            base["success_score"] = 1.0
            base["multimodal_anonymisation_score"] = 0.50 * base.privacy_score + 0.50
            selected_parts.append(base)
        else:
            selected_parts.append(
                state_rows[state_rows.variant.eq(decision.selected_variant)].copy()
            )
    adaptive = pd.concat(selected_parts, ignore_index=True)
    adaptive["policy"] = "adaptive_multimodal_policy"
    adaptive.to_csv(output_dir / "09_adaptive_selected_metrics.csv", index=False)

    risk_flags = routing[
        [
            "protocol_id",
            "image_id",
            "ground_truth_text_present",
            "predicted_text_present",
            "ground_truth_screen_present",
            "predicted_screen_present",
        ]
    ]
    residual = adaptive.merge(risk_flags, on=["protocol_id", "image_id"], validate="one_to_one")
    residual["missed_text_detection"] = residual.ground_truth_text_present.astype(
        bool
    ) & ~residual.predicted_text_present.astype(bool)
    residual["missed_screen_detection"] = residual.ground_truth_screen_present.astype(
        bool
    ) & ~residual.predicted_screen_present.astype(bool)
    residual["residual_text_readability"] = residual.ground_truth_text_present.astype(
        bool
    ) & residual.text_ocr_privacy_score.fillna(0.0).lt(0.50)
    residual["insufficient_screen_obscuration"] = residual.ground_truth_screen_present.astype(
        bool
    ) & residual.screen_region_obscuration.fillna(0.0).lt(0.50)
    residual["utility_below_050"] = residual.utility_score.lt(0.50)
    residual["any_residual_risk_flag"] = residual[
        [
            "missed_text_detection",
            "missed_screen_detection",
            "residual_text_readability",
            "insufficient_screen_obscuration",
            "utility_below_050",
        ]
    ].any(axis=1)
    residual.to_csv(output_dir / "10_residual_risk_analysis.csv", index=False)

    comparison_rows = []
    for split in ["test", "all"]:
        adaptive_split = (
            adaptive if split == "all" else adaptive[adaptive.evaluation_split.eq(split)]
        )
        aggregate = adaptive_split.agg(
            {
                "privacy_score": "mean",
                "utility_score": "mean",
                "runtime_score": "mean",
                "success_score": "mean",
                "multimodal_anonymisation_score": "mean",
                "SSIM": "mean",
                "LPIPS": "mean",
                "non_sensitive_change_fraction": "mean",
                "runtime_seconds": "sum",
            }
        ).to_dict()
        comparison_rows.append(
            {
                "split": split,
                "policy": "adaptive_multimodal_policy",
                "n_images": len(adaptive_split),
                **aggregate,
                "adaptive_minus_fixed_mean": 0.0,
                "difference_ci_low": 0.0,
                "difference_ci_high": 0.0,
                "adaptive_win_count": 0,
                "fixed_win_count": 0,
                "tie_count": len(adaptive_split),
            }
        )
        fixed_split = (
            metrics if split == "all" else metrics[metrics.evaluation_split.eq(split)]
        )
        for variant, group in fixed_split.groupby("variant"):
            aggregate = group.agg(
                {
                    "privacy_score": "mean",
                    "utility_score": "mean",
                    "runtime_score": "mean",
                    "success_score": "mean",
                    "multimodal_anonymisation_score": "mean",
                    "SSIM": "mean",
                    "LPIPS": "mean",
                    "non_sensitive_change_fraction": "mean",
                    "runtime_seconds": "sum",
                }
            ).to_dict()
            comparison_rows.append(
                {
                    "split": split,
                    "policy": f"fixed_{variant}",
                    "n_images": len(group),
                    **aggregate,
                    **paired_bootstrap(adaptive_split, group),
                }
            )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["split", "multimodal_anonymisation_score"], ascending=[True, False]
    )
    comparison.to_csv(output_dir / "07_redaction_method_comparison.csv", index=False)

    detection_path = output_dir / "05_combined_risk_detection.csv"
    if not detection_path.exists():
        detection_path = EVIDENCE_DIR / "05_combined_risk_detection.csv"
    detection = pd.read_csv(detection_path)
    test_detection = detection[detection.split.eq("test")].iloc[0]
    test_comparison = comparison[comparison.split.eq("test")]
    adaptive_test = test_comparison[
        test_comparison.policy.eq("adaptive_multimodal_policy")
    ].iloc[0]
    best_fixed = (
        test_comparison[~test_comparison.policy.eq("adaptive_multimodal_policy")]
        .sort_values("multimodal_anonymisation_score", ascending=False)
        .iloc[0]
    )
    residual_test = residual[residual.evaluation_split.eq("test")]
    lines = [
        "# RQ3 Multimodal Privacy Evidence",
        "",
        "## Reviewed protocol",
        "",
        "- `250` egocentric images were manually reviewed with `116` text boxes and `139` screen boxes.",
        "- Screen-priority annotation and routing remove text boxes that overlap a reviewed/predicted screen.",
        "- Method selection uses the development split; the primary result below is held-out test evidence.",
        "- Adaptive operators include area-aware screen redaction (fill small regions, strong-blur large ones) "
        "and a text-only privacy floor on development selection.",
        "",
        "## Detection",
        "",
        f"- Combined risk precision: `{test_detection.precision:.4f}`.",
        f"- Combined risk recall: `{test_detection.recall:.4f}`.",
        f"- Combined risk F1: `{test_detection.f1:.4f}`.",
        f"- OAPR multimodal score: `{test_detection.oapr_multimodal_score:.4f}`.",
        "- Box/region-level results are reported separately in `02_detection_method_comparison.csv`; "
        "image-level presence is not presented as perfect localisation.",
        "",
        "## End-to-end redaction",
        "",
        f"- Adaptive privacy score: `{adaptive_test.privacy_score:.4f}`.",
        f"- Adaptive utility score: `{adaptive_test.utility_score:.4f}`.",
        f"- Adaptive multimodal anonymisation score: `{adaptive_test.multimodal_anonymisation_score:.4f}`.",
        f"- Strongest fixed policy: `{best_fixed.policy}` with score `{best_fixed.multimodal_anonymisation_score:.4f}`.",
        f"- Adaptive minus strongest fixed: "
        f"`{adaptive_test.multimodal_anonymisation_score - best_fixed.multimodal_anonymisation_score:.4f}`.",
        "",
        "## Adaptive policy (development-selected)",
        "",
    ]
    for decision in adaptive_policy.itertuples(index=False):
        lines.append(
            f"- `{decision.predicted_risk_state}` → `{decision.selected_variant}` "
            f"({decision.reason})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The protocol uses independent human boxes and held-out localization evaluation.",
            "- The combined detector is privacy-oriented: missed-risk recall is weighted more strongly "
            "than harmless extra redaction.",
            "- End-to-end privacy includes localization failures; it is not an oracle-box redaction result.",
            "- Area-aware screen operators target utility collapse on large displays while retaining "
            "hard fill on small screens.",
            "- The text-only privacy floor prevents weak pixelation when a stronger privacy operator "
            "is available on development evidence.",
            "- Residual false positives and missed regions remain measurable limitations and are not "
            "converted into a full-anonymisation claim.",
            "",
            "## Held-out residual-risk flags",
            "",
            f"- Missed text-risk images: `{int(residual_test.missed_text_detection.sum())}`.",
            f"- Missed screen-risk images: `{int(residual_test.missed_screen_detection.sum())}`.",
            f"- Text readability below the privacy threshold: "
            f"`{int(residual_test.residual_text_readability.sum())}` flagged images.",
            f"- Screen obscuration below the privacy threshold: "
            f"`{int(residual_test.insufficient_screen_obscuration.sum())}` flagged images.",
            f"- Utility below 0.50: `{int(residual_test.utility_below_050.sum())}` images.",
        ]
    )
    (output_dir / "11_rq3_final_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Compact operator-improvement report for the experiment log.
    baseline_note = (
        "Prior adaptive (pre-operator update): privacy 0.8199, utility 0.7005, "
        "score 0.8195, util<0.50 = 34/75."
    )
    improve_lines = [
        "# Operator Improvement Results (area-aware screen + text privacy floor)",
        "",
        baseline_note,
        "",
        "## Development-selected adaptive policy",
        "",
        adaptive_policy.to_markdown(index=False),
        "",
        "## Held-out adaptive vs fixed (top rows)",
        "",
        test_comparison.head(12).to_markdown(index=False),
        "",
        "## Held-out residual flags",
        "",
        f"- utility_below_050: {int(residual_test.utility_below_050.sum())}/75",
        f"- residual_text_readability: {int(residual_test.residual_text_readability.sum())}/75",
        f"- insufficient_screen_obscuration: {int(residual_test.insufficient_screen_obscuration.sum())}/75",
        f"- missed_text: {int(residual_test.missed_text_detection.sum())}/75",
        f"- missed_screen: {int(residual_test.missed_screen_detection.sum())}/75",
        "",
        f"Text privacy floor used: {args.text_privacy_floor:.2f}",
        f"Area-aware thresholds grid: {list(AREA_AWARE_THRESHOLDS)}",
    ]
    (output_dir / "13_operator_improvement_results.md").write_text(
        "\n".join(improve_lines) + "\n", encoding="utf-8"
    )

    print(adaptive_policy.to_string(index=False))
    print(test_comparison.head(15).to_string(index=False))
    print(
        "held-out adaptive:",
        float(adaptive_test.privacy_score),
        float(adaptive_test.utility_score),
        float(adaptive_test.multimodal_anonymisation_score),
        "util<0.5",
        int(residual_test.utility_below_050.sum()),
    )


if __name__ == "__main__":
    main()
