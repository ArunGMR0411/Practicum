#!/usr/bin/env python3
"""Build the complete adaptive-versus-fixed evidence comparison pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/07_adaptive_full_comparison"
ADAPTIVE_DETECTOR = "cv_box_reranker_with_rfdetr_predicted_conditions"
ANON_METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
]


def save(df: pd.DataFrame, name: str, title: str, columns: list[str] | None = None) -> None:
    if columns:
        df = df.reindex(columns=columns)
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}.md").write_text(
        f"# {title}\n\n{df.to_markdown(index=False)}\n",
        encoding="utf-8",
    )


def detection_comparisons() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(
        ROOT / "outputs/02_face_detection/08_sliced_rfdetr_detector_experiment/"
        "sliced_detector_policy_scores.csv"
    )
    combined = scores.loc[scores["protocol"].eq("combined_1000")].copy()
    overall = combined.loc[combined["subgroup"].eq("all_images")].copy()
    adaptive_score = overall.loc[
        overall["model"].eq(ADAPTIVE_DETECTOR), "oapr_detector_score"
    ].iloc[0]
    overall["adaptive_score_delta"] = overall["oapr_detector_score"] - adaptive_score
    overall["adaptive_win"] = overall["oapr_detector_score"] > adaptive_score + 1e-12
    overall["adaptive_tie"] = (overall["oapr_detector_score"] - adaptive_score).abs() <= 1e-12
    overall = overall.sort_values("oapr_detector_score", ascending=False)

    subgroup = combined.loc[~combined["subgroup"].eq("all_images")].copy()
    wide = subgroup.pivot_table(
        index="subgroup", columns="model", values="oapr_detector_score", aggfunc="first"
    )
    rows = []
    for category, values in wide.iterrows():
        adaptive = values.get(ADAPTIVE_DETECTOR, np.nan)
        row = {
            "category": category,
            "adaptive_detector": ADAPTIVE_DETECTOR,
            "adaptive_oapr_detector_score": adaptive,
        }
        for method in ANON_METHODS:
            pass
        competitors = values.drop(labels=[ADAPTIVE_DETECTOR], errors="ignore").dropna()
        if pd.isna(adaptive) or competitors.empty:
            row.update({"best_competitor": "not_available", "best_competitor_score": np.nan, "adaptive_margin": np.nan})
        else:
            best_name = competitors.idxmax()
            row.update({"best_competitor": best_name, "best_competitor_score": competitors.max(), "adaptive_margin": adaptive - competitors.max()})
        rows.append(row)
    subgroup_summary = pd.DataFrame(rows)
    return overall, subgroup_summary


def multimodal_detection() -> pd.DataFrame:
    """Build multimodal detection comparison from the canonical 250-image package.

    Prefer method-level rows (text/screen localisation) plus the selected
    combined-risk held-out summary so the pack reflects region-level evidence.
    """
    evidence = ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence"
    methods = pd.read_csv(evidence / "02_detection_method_comparison.csv")
    methods = methods[methods["split"].eq("test")].copy()
    methods["evidence_surface"] = "held_out_test_reviewed_250_region_level"
    methods["adaptive_reference"] = False

    combined = pd.read_csv(evidence / "05_combined_risk_detection.csv")
    combined = combined[combined["split"].eq("test")].copy()
    combined["evidence_surface"] = "held_out_test_reviewed_250_combined_risk"
    # Mark the selected combined policy as the adaptive/reference row.
    combined["adaptive_reference"] = True
    ref = float(combined["oapr_multimodal_score"].iloc[0])

    scores = pd.concat([methods, combined], ignore_index=True, sort=False)
    scores["adaptive_score_delta"] = scores["oapr_multimodal_score"] - ref
    scores["adaptive_win"] = scores["oapr_multimodal_score"] > ref + 1e-12
    scores["adaptive_tie"] = (scores["oapr_multimodal_score"] - ref).abs() <= 1e-12
    return scores.sort_values(
        ["modality", "oapr_multimodal_score"], ascending=[True, False]
    )


def face_anonymisation() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final_root = ROOT / "outputs/03_anonymisation/15_materialised_adaptive_policy"
    fixed_summary = pd.read_csv(
        ROOT / "outputs/03_anonymisation/14_group2_comparison/"
        "03_all_method_and_policy_summary.csv"
    )
    direct = pd.read_csv(final_root / "13_final_policy_metric_summary.csv").iloc[0]
    summary = fixed_summary.copy()
    summary.insert(0, "policy", summary["method"].map(lambda value: f"fixed_{value}"))
    summary["n_input_frames"] = summary["n_images"]
    summary["n_success"] = (summary["n_images"] * summary["success_rate"]).round().astype(int)
    summary["n_failure"] = summary["n_input_frames"] - summary["n_success"]
    summary = summary.rename(
        columns={
            "privacy_score": "privacy_score_mean",
            "utility_score": "utility_score_mean",
            "mean_runtime_seconds": "runtime_mean_seconds",
            "balanced_score": "balanced_oapr_anonymisation_score_mean",
        }
    )
    adaptive_row = {
        "policy": "adaptive_artifact_gated_policy",
        "method": "mixed_policy",
        "n_images": int(direct.n_input_frames),
        "success_rate": direct.n_success / direct.n_input_frames,
        "privacy_score_mean": direct.three_attacker_privacy_score,
        "utility_score_mean": direct.hardened_utility_score,
        "runtime_mean_seconds": direct.component_runtime_mean_seconds,
        "balanced_oapr_anonymisation_score_mean": direct.hardened_balanced_score,
        "high_compute_score": np.nan,
        "n_input_frames": int(direct.n_input_frames),
        "n_success": int(direct.n_success),
        "n_failure": int(direct.n_failure),
    }
    summary = pd.concat([summary, pd.DataFrame([adaptive_row])], ignore_index=True)

    pairwise = pd.read_csv(final_root / "14_final_policy_paired_comparison.csv").rename(
        columns={
            "comparator": "fixed_method",
            "policy_mean": "adaptive_mean",
            "comparator_mean": "fixed_mean",
            "mean_difference": "adaptive_minus_fixed_mean",
            "policy_wins": "adaptive_win_count",
            "comparator_wins": "fixed_win_count",
            "ties": "tie_count",
        }
    )
    pairwise.insert(1, "n_paired_frames", 500)

    routes = pd.read_csv(final_root / "04_final_policy_routing_log.csv")
    base_scores = pd.read_csv(
        ROOT / "outputs/03_anonymisation/11_policy_hardening/"
        "02_enhanced_per_image_policy_metrics.csv"
    )
    advanced_scores = pd.read_csv(
        ROOT / "outputs/03_anonymisation/14_group2_comparison/"
        "02_advanced_per_image_scores.csv"
    )
    fixed_full = pd.concat([base_scores, advanced_scores], ignore_index=True, sort=False)
    fixed_full = fixed_full.drop_duplicates(["relative_path", "method"], keep="last")
    category_rows = []
    for category, adaptive_group in routes.groupby("policy_category"):
        paths = set(adaptive_group["relative_path"])
        adaptive_mean = adaptive_group["balanced_score"].mean()
        for method, fixed_group in fixed_full.loc[
            fixed_full["relative_path"].isin(paths)
        ].groupby("method"):
            fixed_mean = fixed_group["enhanced_balanced_score"].mean()
            category_rows.append(
                {
                    "category": category,
                    "fixed_method": method,
                    "adaptive_mean": adaptive_mean,
                    "fixed_mean": fixed_mean,
                    "adaptive_minus_fixed": adaptive_mean - fixed_mean,
                    "adaptive_n": len(adaptive_group),
                    "fixed_n": len(fixed_group),
                    "evidence_status": "final_materialised_policy_category_comparison",
                }
            )
    return summary, pairwise, pd.DataFrame(category_rows)


def multimodal_anonymisation() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/"
        "07_redaction_method_comparison.csv"
    )
    frame = frame[frame["split"].eq("test")].copy()
    adaptive = frame.loc[
        frame["policy"].eq("adaptive_multimodal_policy"),
        "multimodal_anonymisation_score",
    ].iloc[0]
    frame["adaptive_score_delta"] = frame["multimodal_anonymisation_score"] - adaptive
    frame["adaptive_win"] = frame["multimodal_anonymisation_score"] > adaptive + 1e-12
    frame["adaptive_tie"] = (
        frame["multimodal_anonymisation_score"] - adaptive
    ).abs() <= 1e-12
    frame["evidence_status"] = "held_out_test_on_reviewed_250_image_protocol"
    return frame.sort_values("multimodal_anonymisation_score", ascending=False)


def anonymisation_eligibility() -> pd.DataFrame:
    methods = pd.read_csv(ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv")
    return methods[[
        "method", "evidence_level", "final_status", "n_input_frames", "n_success",
        "n_failure", "oapr_role", "limitation", "report_safe_claim",
    ]].copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    det_overall, det_categories = detection_comparisons()
    save(det_overall, "01_face_detection_overall", "Face Detection Overall Comparison")
    save(det_categories, "02_face_detection_category_comparison", "Face Detection Category Comparison")
    mm_det = multimodal_detection()
    save(mm_det, "03_multimodal_detection_comparison", "Multimodal Detection Comparison")
    anon_summary, anon_pairwise, anon_categories = face_anonymisation()
    save(anon_summary, "04_face_anonymisation_overall", "Face Anonymisation Overall Comparison")
    save(anon_pairwise, "05_face_anonymisation_pairwise", "Adaptive Face Anonymisation Pairwise Comparison")
    save(anon_categories, "06_face_anonymisation_category_comparison", "Face Anonymisation Category Comparison")
    mm_anon = multimodal_anonymisation()
    save(mm_anon, "07_multimodal_anonymisation_comparison", "Multimodal Anonymisation Comparison")
    eligibility = anonymisation_eligibility()
    save(eligibility, "08_face_anonymisation_policy_eligibility", "Face Anonymisation Policy Eligibility")

    best_det = det_overall.iloc[0]
    adaptive_det = det_overall.loc[det_overall.model.eq(ADAPTIVE_DETECTOR)].iloc[0]
    adaptive_anon = anon_summary.loc[anon_summary.policy.eq("adaptive_artifact_gated_policy")].iloc[0]
    layered = anon_summary.loc[anon_summary.policy.eq("fixed_layered_blur_downscale_noise")].iloc[0]

    mm_combined = mm_det.loc[mm_det["adaptive_reference"]].iloc[0]
    mm_adaptive = mm_anon.loc[mm_anon["policy"].eq("adaptive_multimodal_policy")].iloc[0]
    mm_fixed = mm_anon.loc[~mm_anon["policy"].eq("adaptive_multimodal_policy")].iloc[0]
    mm_delta = float(mm_adaptive["multimodal_anonymisation_score"] - mm_fixed["multimodal_anonymisation_score"])
    mm_ci_low = float(mm_fixed.get("difference_ci_low", float("nan")))
    mm_ci_high = float(mm_fixed.get("difference_ci_high", float("nan")))
    # 07 table stores adaptive_minus_fixed on fixed rows as adaptive - fixed.
    if "adaptive_minus_fixed_mean" in mm_fixed.index and pd.notna(mm_fixed["adaptive_minus_fixed_mean"]):
        mm_delta = float(mm_fixed["adaptive_minus_fixed_mean"])
    if "difference_ci_low" in mm_fixed.index and pd.notna(mm_fixed["difference_ci_low"]):
        mm_ci_low = float(mm_fixed["difference_ci_low"])
    if "difference_ci_high" in mm_fixed.index and pd.notna(mm_fixed["difference_ci_high"]):
        mm_ci_high = float(mm_fixed["difference_ci_high"])

    summary = [
        "# Adaptive Full Pipeline Comparison",
        "",
        "This pack compares adaptive and fixed methods separately for face detection, multimodal detection, face anonymisation, and multimodal anonymisation.",
        "",
        "## Face Detection",
        "",
        f"The adaptive RF-DETR-aware reranker scored `{adaptive_det.oapr_detector_score:.4f}` on the combined 1,000-image reviewed protocol. The highest score in the table is `{best_det.model}` at `{best_det.oapr_detector_score:.4f}`. The adaptive method is therefore the strongest measured policy by a very small margin only if it is the highest row; otherwise the table is the authority and no dominance claim is made.",
        "",
        "## Multimodal Detection",
        "",
        (
            "Canonical multimodal evidence is the reviewed 250-image region-level "
            "protocol (`outputs/04_multimodal_privacy/01_multimodal_250_evidence/`). "
            f"On the held-out 75-image split, the selected combined text/screen "
            f"policy (`{mm_combined.variant}`) reaches precision "
            f"`{float(mm_combined.precision):.4f}`, recall "
            f"`{float(mm_combined.recall):.4f}`, F1 "
            f"`{float(mm_combined.f1):.4f}`, and OAPR multimodal score "
            f"`{float(mm_combined.oapr_multimodal_score):.4f}`. "
            "Region-level text precision remains low because environmental text "
            "is proposed; image-level presence is not treated as perfect localisation."
        ),
        "",
        "## Face Anonymisation",
        "",
        f"The materialised RiDDLE-heavy policy ablation scored `{adaptive_anon.balanced_oapr_anonymisation_score_mean:.4f}` versus `{layered.balanced_oapr_anonymisation_score_mean:.4f}` for fixed layered obfuscation. It produced all 500 outputs and used deterministic fallback for predicted RiDDLE artifacts. A subsequent uniform visual-quality investigation found that the grouped gate recalled only `11/14` reviewed artifacts and did not reliably cover pose/gaze failures. The current visual-safe runtime policy therefore uses 286 layered routes, 81 solid-mask routes, and 133 reviewed no-face copy-through routes; it completes `500/500` routes with zero generative selections and zero failures.",
        "",
        "## Multimodal Anonymisation",
        "",
        (
            "Adaptive risk-state routing is measured end-to-end on predicted "
            "boxes over the held-out 75-image split of the reviewed 250-image "
            "protocol. "
            f"Adaptive scored privacy `{float(mm_adaptive.privacy_score):.4f}`, "
            f"utility `{float(mm_adaptive.utility_score):.4f}`, and objective "
            f"`{float(mm_adaptive.multimodal_anonymisation_score):.4f}`. "
            f"The strongest fixed policy was `{mm_fixed.policy}` at "
            f"`{float(mm_fixed.multimodal_anonymisation_score):.4f}` "
            f"(privacy `{float(mm_fixed.privacy_score):.4f}`, utility "
            f"`{float(mm_fixed.utility_score):.4f}`). "
            f"Adaptive-minus-fixed mean difference was `{mm_delta:+.4f}` with "
            f"95% bootstrap CI `[{mm_ci_low:.4f}, {mm_ci_high:.4f}]`, so "
            "superiority over that strongest fixed policy is not statistically "
            "established. Screen privacy is measured as region obscuration, "
            "not semantic leakage elimination."
        ),
        "",
        "## Claim Boundary",
        "",
        "The adaptive pipeline is evidence-supported and objective-aware. It must not be described as globally superior to every fixed method. A method-by-method and category-by-category claim is required.",
        "",
        "Canonical multimodal sources: `outputs/04_multimodal_privacy/01_multimodal_250_evidence/11_rq3_final_summary.md` and the machine-readable tables in the same directory.",
        "",
        "Canonical source for the current 500-frame face route: `outputs/03_anonymisation/16_visual_quality_hardening/04_final_visual_safe_policy.csv`.",
        "",
        "The RiDDLE-heavy materialised comparison remains available at `outputs/03_anonymisation/15_materialised_adaptive_policy/04_final_policy_routing_log.csv` as quantitative ablation evidence.",
    ]
    (OUT / "09_full_comparison_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("generated", OUT)


if __name__ == "__main__":
    main()
