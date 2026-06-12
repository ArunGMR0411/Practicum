#!/usr/bin/env python3

"""Build category-aware anonymisation policy scores from completed per-image metrics."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/03_anonymisation/09_policy_scoring"
METRIC_DIR = PROJECT_ROOT / "outputs/03_anonymisation/08_policy_metric_readiness"
REID_DIR = METRIC_DIR / "reid"

ANON_MANIFEST = PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv"
CONDITION_MANIFEST = PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
MULTIMODAL_POLICY = PROJECT_ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/04_multimodal_risk_policy.csv"
ALL_METHODS = PROJECT_ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv"

DETERMINISTIC_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/02_deterministic_baselines/01_output_manifest.csv"
NULLFACE_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/01_nullface_full_500_manifest.csv"
DIFFUSION_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/01_diffusion_manifest_full_500.csv"
RP_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/01_rp_final_manifest.csv"
NULLFACE_RUNTIME = PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/03_nullface_runtime_summary.csv"

METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
]

CONDITION_LABELS = [
    "no_face",
    "single_face",
    "multi_face",
    "small_face",
    "medium_face",
    "large_face",
    "mixed_scale_face",
    "very_small_or_distant_face",
    "edge_or_partial_face",
    "profile_or_occluded_face",
    "downward_egocentric_view",
    "motion_blur_or_low_sharpness",
    "low_light_or_dim",
    "high_clutter",
]

OBJECTIVE_WEIGHTS = {
    "privacy_first_score": {
        "privacy": 0.70,
        "utility": 0.15,
        "runtime": 0.05,
        "success": 0.10,
    },
    "balanced_oapr_anonymisation_score": {
        "privacy": 0.50,
        "utility": 0.30,
        "runtime": 0.10,
        "success": 0.10,
    },
    "utility_preserving_score": {
        "privacy": 0.30,
        "utility": 0.50,
        "runtime": 0.10,
        "success": 0.10,
    },
    "runtime_practical_score": {
        "privacy": 0.40,
        "utility": 0.20,
        "runtime": 0.30,
        "success": 0.10,
    },
}


def save_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_perceptual(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("detailed", [])
    if not rows:
        return pd.DataFrame(columns=["relative_path", "method", "SSIM", "LPIPS"])
    return pd.DataFrame(rows).rename(columns={"ssim": "SSIM", "lpips": "LPIPS"})


def read_reid(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["relative_path", "method"])
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return pd.DataFrame(columns=["relative_path", "method"])
    df = pd.DataFrame(rows).rename(columns={"image_id": "relative_path"})
    return (
        df.groupby(["relative_path", "method"], dropna=False)
        .agg(
            AdaFace_cosine_mean=("adaface_cosine_sim", "mean"),
            AdaFace_reid_rate=("adaface_hit", "mean"),
            ArcFace_cosine_mean=("arcface_cosine_sim", "mean"),
            ArcFace_reid_rate=("arcface_hit", "mean"),
            face_crop_count=("box_idx", "count"),
        )
        .reset_index()
    )


def method_manifest(method: str) -> pd.DataFrame:
    if method in {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"}:
        df = pd.read_csv(DETERMINISTIC_MANIFEST)
        return df[df["method"] == method].copy()
    if method == "nullface":
        df = pd.read_csv(NULLFACE_MANIFEST)
        df["status"] = "ok"
        df["runtime_seconds"] = pd.NA
        return df
    if method == "diffusion_low_step":
        return pd.read_csv(DIFFUSION_MANIFEST)
    if method == "reverse_personalization":
        return pd.read_csv(RP_MANIFEST)
    return pd.DataFrame()


def method_perceptual(method: str) -> pd.DataFrame:
    if method in {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"}:
        return read_perceptual(METRIC_DIR / "deterministic_per_image_perceptual.json")
    if method == "nullface":
        return read_perceptual(METRIC_DIR / "nullface_per_image_perceptual.json")
    if method == "diffusion_low_step":
        return read_perceptual(PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/04_diffusion_full_500_perceptual.json")
    if method == "reverse_personalization":
        return read_perceptual(PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/06_rp_final_perceptual.json")
    return pd.DataFrame(columns=["relative_path", "method", "SSIM", "LPIPS"])


def method_reid(method: str) -> pd.DataFrame:
    if method in {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise", "nullface"}:
        return read_reid(REID_DIR / f"{method}_reid_details.json")
    if method == "diffusion_low_step":
        return read_reid(PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/05_diffusion_full_500_reid_details.json")
    if method == "reverse_personalization":
        return read_reid(PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/07_rp_final_reid_details.json")
    return pd.DataFrame(columns=["relative_path", "method"])


def nullface_runtime_mean() -> float:
    if not NULLFACE_RUNTIME.is_file():
        return math.nan
    df = pd.read_csv(NULLFACE_RUNTIME)
    row = df[df["segment"].eq("compute_constrained_remaining_298")]
    if row.empty:
        return math.nan
    return float(row.iloc[0]["mean_runtime_seconds"])


def build_condition_frame() -> pd.DataFrame:
    base = pd.read_csv(ANON_MANIFEST)[["relative_path"]].copy()
    manual = pd.read_csv(CONDITION_MANIFEST)
    keep = [
        "relative_path",
        "box_count",
        "face_count_category",
        "face_scale_category",
        "condition_label",
        "edge_partial_face",
        "profile_occluded_face",
        "downward_egocentric_view",
        "blur_low_sharpness",
        "low_light_dim",
        "clutter_level",
    ]
    manual = manual[[col for col in keep if col in manual.columns]].copy()
    condition_text = manual.get("condition_label", "").astype(str)
    face_count = manual.get("face_count_category", "").astype(str)
    face_scale = manual.get("face_scale_category", "").astype(str)
    manual["manual_no_face"] = face_count.eq("no_face")
    manual["manual_single_face"] = face_count.eq("single_face")
    manual["manual_multi_face"] = face_count.eq("multi_face")
    manual["manual_small_face"] = face_scale.eq("small") | condition_text.str.contains("small", regex=False)
    manual["manual_medium_face"] = face_scale.eq("medium") | condition_text.str.contains("medium", regex=False)
    manual["manual_large_face"] = face_scale.eq("large") | condition_text.str.contains("large", regex=False)
    manual["manual_mixed_scale_face"] = face_scale.eq("mixed_scale") | condition_text.str.contains("mixed_scale", regex=False)
    manual["manual_very_small_or_distant_face"] = face_scale.eq("very_small_or_distant") | condition_text.str.contains(
        "very_small_or_distant", regex=False
    )
    manual["manual_edge_or_partial_face"] = (
        manual.get("edge_partial_face", "").astype(str).str.lower().eq("yes")
        | condition_text.str.contains("edge_or_partial", regex=False)
    )
    manual["manual_profile_or_occluded_face"] = (
        manual.get("profile_occluded_face", "").astype(str).str.lower().eq("yes")
        | condition_text.str.contains("profile_or_occluded", regex=False)
    )
    manual["manual_downward_egocentric_view"] = (
        manual.get("downward_egocentric_view", "").astype(str).str.lower().eq("yes")
        | condition_text.str.contains("downward_egocentric", regex=False)
    )
    manual["manual_motion_blur_or_low_sharpness"] = (
        manual.get("blur_low_sharpness", "").astype(str).str.lower().eq("yes")
        | condition_text.str.contains("motion_blur_or_low_sharpness", regex=False)
    )
    manual["manual_low_light_or_dim"] = (
        manual.get("low_light_dim", "").astype(str).str.lower().eq("yes")
        | condition_text.str.contains("low_light", regex=False)
    )
    manual["manual_high_clutter"] = (
        manual.get("clutter_level", "").astype(str).str.lower().eq("high")
        | condition_text.str.contains("clutter_high", regex=False)
    )
    merged = base.merge(manual, on="relative_path", how="left")
    if MULTIMODAL_POLICY.is_file():
        mm = pd.read_csv(MULTIMODAL_POLICY).rename(columns={"image_id": "relative_path"})
        merged = merged.merge(
            mm[
                [
                    "relative_path",
                    "text_present",
                    "screen_present",
                    "no_text_screen_risk",
                    "multimodal_risk_state",
                ]
            ],
            on="relative_path",
            how="left",
        )
    else:
        merged["text_present"] = False
        merged["screen_present"] = False
        merged["no_text_screen_risk"] = False
        merged["multimodal_risk_state"] = "not_available"
    return merged


def status_success(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["ok", "existing", "completed", "copied"])


def build_per_image_metrics(condition_df: pd.DataFrame) -> pd.DataFrame:
    all_methods = pd.read_csv(ALL_METHODS).set_index("method", drop=False)
    nullface_runtime = nullface_runtime_mean()
    rows: list[pd.DataFrame] = []
    for method in METHODS:
        manifest = method_manifest(method)
        perceptual = method_perceptual(method)
        reid = method_reid(method)
        if not perceptual.empty:
            perceptual = perceptual[perceptual["method"].eq(method)].copy()
        if not reid.empty:
            reid = reid[reid["method"].eq(method)].copy()
        detailed = condition_df.merge(manifest, on="relative_path", how="left", suffixes=("", "_manifest"))
        detailed["method"] = method
        if "status" not in detailed.columns:
            detailed["status"] = "missing"
        detailed["success"] = status_success(detailed["status"]).astype(float)
        detailed = detailed.merge(perceptual[["relative_path", "method", "SSIM", "LPIPS"]], on=["relative_path", "method"], how="left")
        if not reid.empty:
            detailed = detailed.merge(reid, on=["relative_path", "method"], how="left")
        else:
            detailed["AdaFace_cosine_mean"] = pd.NA
            detailed["AdaFace_reid_rate"] = pd.NA
            detailed["ArcFace_cosine_mean"] = pd.NA
            detailed["ArcFace_reid_rate"] = pd.NA
            detailed["face_crop_count"] = pd.NA

        method_row = all_methods.loc[method] if method in all_methods.index else None
        if method == "nullface":
            runtime_mean = nullface_runtime
        elif method == "reverse_personalization":
            runtime_mean = float(method_row["runtime_mean"]) if method_row is not None and pd.notna(method_row["runtime_mean"]) else math.nan
        else:
            runtime_mean = pd.to_numeric(detailed.get("runtime_seconds"), errors="coerce")
        if isinstance(runtime_mean, pd.Series):
            detailed["runtime_for_score"] = runtime_mean
        else:
            detailed["runtime_for_score"] = runtime_mean

        has_face = pd.to_numeric(detailed.get("box_count"), errors="coerce").fillna(0).gt(0)
        no_face = ~has_face
        detailed["privacy_residual_rate"] = (
            pd.to_numeric(detailed["AdaFace_reid_rate"], errors="coerce").fillna(0)
            + pd.to_numeric(detailed["ArcFace_reid_rate"], errors="coerce").fillna(0)
        ) / 2
        detailed.loc[no_face, "privacy_residual_rate"] = 0.0
        detailed.loc[detailed["success"].eq(0), "privacy_residual_rate"] = 1.0
        detailed["privacy_score"] = (1.0 - detailed["privacy_residual_rate"]).clip(0, 1)

        detailed["SSIM"] = pd.to_numeric(detailed["SSIM"], errors="coerce")
        detailed["LPIPS"] = pd.to_numeric(detailed["LPIPS"], errors="coerce")
        detailed["lpips_utility_score"] = (1.0 - (detailed["LPIPS"].fillna(0.05) / 0.05)).clip(0, 1)
        detailed["utility_score"] = ((0.50 * detailed["SSIM"].fillna(0)) + (0.50 * detailed["lpips_utility_score"])).clip(0, 1)
        detailed.loc[detailed["success"].eq(0), "utility_score"] = 0.0

        detailed["runtime_for_score"] = pd.to_numeric(detailed["runtime_for_score"], errors="coerce").fillna(30.0)
        detailed["runtime_score"] = (1.0 - (detailed["runtime_for_score"] / 5.0)).clip(0, 1)
        detailed["success_score"] = detailed["success"].astype(float)

        for score_name, weights in OBJECTIVE_WEIGHTS.items():
            detailed[score_name] = (
                weights["privacy"] * detailed["privacy_score"]
                + weights["utility"] * detailed["utility_score"]
                + weights["runtime"] * detailed["runtime_score"]
                + weights["success"] * detailed["success_score"]
            ).clip(0, 1)
        rows.append(detailed)
    return pd.concat(rows, ignore_index=True)


def summarize_category(metrics: pd.DataFrame, category: str, source: str, mask: pd.Series) -> list[dict[str, Any]]:
    total_support = int(metrics.loc[mask, "relative_path"].nunique())
    if category != "no_face" and source == "manual_condition_review":
        face_positive = pd.to_numeric(metrics.get("box_count"), errors="coerce").fillna(0).gt(0)
        mask = mask & face_positive
    sub = metrics[mask].copy()
    if sub.empty:
        return []
    scored_support = int(sub["relative_path"].nunique())
    rows: list[dict[str, Any]] = []
    grouped = sub.groupby("method", dropna=False)
    for method, df in grouped:
        rows.append(
            {
                "category": category,
                "category_source": source,
                "support_count": total_support,
                "scored_frame_count": scored_support,
                "method": method,
                "success_rate": float(df["success_score"].mean()),
                "SSIM_mean": float(df["SSIM"].mean(skipna=True)),
                "LPIPS_mean": float(df["LPIPS"].mean(skipna=True)),
                "privacy_score_mean": float(df["privacy_score"].mean()),
                "utility_score_mean": float(df["utility_score"].mean()),
                "runtime_score_mean": float(df["runtime_score"].mean()),
                "privacy_first_score": float(df["privacy_first_score"].mean()),
                "balanced_oapr_anonymisation_score": float(df["balanced_oapr_anonymisation_score"].mean()),
                "utility_preserving_score": float(df["utility_preserving_score"].mean()),
                "runtime_practical_score": float(df["runtime_practical_score"].mean()),
                "runtime_mean_for_score": float(df["runtime_for_score"].mean()),
            }
        )
    return rows


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    ranked: list[pd.DataFrame] = []
    for category, sub in df.groupby("category", dropna=False):
        sub = sub.copy()
        for score in OBJECTIVE_WEIGHTS:
            sub[f"{score}_rank"] = sub[score].rank(ascending=False, method="dense").astype(int)
        ranked.append(sub)
    return pd.concat(ranked, ignore_index=True).to_dict("records")


def build_category_tables(metrics: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    category_rows: list[dict[str, Any]] = []
    for label in CONDITION_LABELS:
        col = f"manual_{label}"
        if col in metrics.columns:
            mask = metrics[col].astype(str).str.lower().isin(["true", "1", "yes"])
            if mask.any():
                category_rows.extend(summarize_category(metrics, label, "manual_condition_review", mask))
    for col, name in [
        ("text_present", "text_present"),
        ("screen_present", "screen_present"),
        ("no_text_screen_risk", "no_text_screen_risk"),
    ]:
        if col in metrics.columns:
            mask = metrics[col].astype(str).str.lower().isin(["true", "1", "yes"])
            if mask.any():
                category_rows.extend(summarize_category(metrics, name, "multimodal_policy", mask))

    category_rows = rank_rows(category_rows)
    df = pd.DataFrame(category_rows)
    decision_rows: list[dict[str, Any]] = []
    for category, sub in df.groupby("category", dropna=False):
        support = int(sub.iloc[0]["support_count"])
        scored_support = int(sub.iloc[0].get("scored_frame_count", support))
        if category == "no_face":
            decision_rows.append(
                {
                    "category": category,
                    "category_source": sub.iloc[0]["category_source"],
                    "support_count": support,
                    "scored_frame_count": scored_support,
                    "policy_confidence": "high",
                    "privacy_first_method": "no_face_copy_or_multimodal_only",
                    "balanced_default_method": "no_face_copy_or_multimodal_only",
                    "utility_method": "no_face_copy_or_multimodal_only",
                    "runtime_method": "no_face_copy_or_multimodal_only",
                    "final_policy_decision": "skip_face_anonymisation_then_apply_multimodal_policy_if_needed",
                    "decision_reason": "Reviewed protocol has no face boxes, so face anonymisation is unnecessary; preserve utility and route only text/screen risk if present.",
                }
            )
            continue
        winners = {}
        for score in OBJECTIVE_WEIGHTS:
            ordered = sub.sort_values(score, ascending=False)
            winners[score] = ordered.iloc[0]["method"]
        default = winners["balanced_oapr_anonymisation_score"]
        confidence = "high" if scored_support >= 20 else "low_support"
        reason = "Highest balanced OAPR anonymisation score for this category."
        if confidence == "low_support":
            default = "layered_blur_downscale_noise"
            reason = (
                "Category has fewer than 20 face-positive scored frames; use the stable stronger-privacy practical fallback "
                "rather than overclaiming a low-support category winner."
            )
        if category in {"text_present", "screen_present"}:
            reason += " Multimodal redaction must also be applied; face anonymisation score alone is not sufficient."
        decision_rows.append(
            {
                "category": category,
                "category_source": sub.iloc[0]["category_source"],
                "support_count": support,
                "scored_frame_count": scored_support,
                "policy_confidence": confidence,
                "privacy_first_method": winners["privacy_first_score"],
                "balanced_default_method": default,
                "utility_method": winners["utility_preserving_score"],
                "runtime_method": winners["runtime_practical_score"],
                "final_policy_decision": default,
                "decision_reason": reason,
            }
        )
    return category_rows, decision_rows


def write_summary(category_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> None:
    decisions = pd.DataFrame(decision_rows)
    default_counts = decisions["final_policy_decision"].value_counts().to_dict() if not decisions.empty else {}
    lines = [
        "# Category-Aware Anonymisation Policy Scores",
        "",
        "Purpose: derive evidence-based anonymisation policy choices from per-image metrics joined with the reviewed egocentric-stress 500 condition profile and the 250-frame multimodal risk policy.",
        "",
        "Score definition:",
        "",
        "- `privacy_score = 1 - mean(AdaFace_reid_rate, ArcFace_reid_rate)`; no-face frames get privacy residual `0`, while failed method outputs get residual `1`.",
        "- `utility_score = 0.5 * SSIM + 0.5 * (1 - LPIPS/0.05)`, clipped to `[0, 1]`.",
        "- `runtime_score = 1 - runtime_seconds/5`, clipped to `[0, 1]`; methods without full per-frame runtime use the documented aggregate runtime boundary.",
        "- `success_score = 1` for successful outputs and `0` for failures.",
        "- `balanced_oapr_anonymisation_score = 0.50*privacy + 0.30*utility + 0.10*runtime + 0.10*success`.",
        "",
        "Additional objective scores:",
        "",
        "- `privacy_first_score = 0.70*privacy + 0.15*utility + 0.05*runtime + 0.10*success`.",
        "- `utility_preserving_score = 0.30*privacy + 0.50*utility + 0.10*runtime + 0.10*success`.",
        "- `runtime_practical_score = 0.40*privacy + 0.20*utility + 0.30*runtime + 0.10*success`.",
        "",
        "Policy findings:",
        "",
    ]
    for method, count in default_counts.items():
        lines.append(f"- `{method}` is the balanced default decision for {count} category rows.")
    lines.extend(
        [
            "",
            "Boundary:",
            "",
            "- This is a category-aware scoring table, not a claim that one method globally dominates.",
            "- `no_face` routes to no face anonymisation; multimodal policy still applies if text/screen risk exists.",
            "- Reverse Personalization is included with its 18-frame failure penalty and high runtime cost; it is not a default deployment candidate.",
            "- StyleID/FAMS are excluded because the final decision is quality-limited after systematic tuning.",
            "",
            "Canonical outputs:",
            "",
            f"- Per-image metric table: `{rel(OUTPUT_DIR / 'anonymisation_policy_per_image_metrics.csv')}`.",
            f"- Category score table: `{rel(OUTPUT_DIR / 'anonymisation_policy_category_scores.csv')}`.",
            f"- Final category decision table: `{rel(OUTPUT_DIR / 'anonymisation_policy_decision_table.csv')}`.",
        ]
    )
    (OUTPUT_DIR / "anonymisation_policy_score_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    condition_df = build_condition_frame()
    metrics = build_per_image_metrics(condition_df)
    metrics.to_csv(OUTPUT_DIR / "anonymisation_policy_per_image_metrics.csv", index=False)
    category_rows, decision_rows = build_category_tables(metrics)
    category_fields = [
        "category",
        "category_source",
        "support_count",
        "scored_frame_count",
        "method",
        "success_rate",
        "SSIM_mean",
        "LPIPS_mean",
        "privacy_score_mean",
        "utility_score_mean",
        "runtime_score_mean",
        "privacy_first_score",
        "balanced_oapr_anonymisation_score",
        "utility_preserving_score",
        "runtime_practical_score",
        "runtime_mean_for_score",
        "privacy_first_score_rank",
        "balanced_oapr_anonymisation_score_rank",
        "utility_preserving_score_rank",
        "runtime_practical_score_rank",
    ]
    save_csv(category_rows, category_fields, OUTPUT_DIR / "anonymisation_policy_category_scores.csv")
    decision_fields = [
        "category",
        "category_source",
        "support_count",
        "scored_frame_count",
        "policy_confidence",
        "privacy_first_method",
        "balanced_default_method",
        "utility_method",
        "runtime_method",
        "final_policy_decision",
        "decision_reason",
    ]
    save_csv(decision_rows, decision_fields, OUTPUT_DIR / "anonymisation_policy_decision_table.csv")
    write_summary(category_rows, decision_rows)
    print(
        {
            "per_image_rows": len(metrics),
            "category_score_rows": len(category_rows),
            "decision_rows": len(decision_rows),
            "output_dir": rel(OUTPUT_DIR),
        }
    )


if __name__ == "__main__":
    main()
