#!/usr/bin/env python3
"""Summarise direct metrics and paired evidence for the final adaptive policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/03_anonymisation/15_materialised_adaptive_policy"


def main() -> None:
    routes = pd.read_csv(OUT / "04_final_policy_routing_log.csv")
    runtime = pd.read_csv(OUT / "07_final_runtime_summary.csv").iloc[0]
    perceptual = json.loads((OUT / "08_final_perceptual_metrics.json").read_text())["summary"]["adaptive_artifact_gated_policy"]
    reid = json.loads((OUT / "10_final_reid_summary.json").read_text())
    facenet = json.loads((OUT / "12_final_facenet_summary.json").read_text())
    gate = pd.read_csv(OUT / "01_artifact_gate_validation.csv").iloc[0]
    review = pd.read_csv(OUT / "02_artifact_gate_review_predictions.csv")

    privacy_three_attacker = 1.0 - np.mean([
        reid["adaface_reid_rate"],
        reid["arcface_reid_rate"],
        facenet["facenet_reid_rate_060"],
    ])
    summary = pd.DataFrame([{
        "method": "adaptive_artifact_gated_policy",
        "n_input_frames": len(routes),
        "n_success": int(routes.status.eq("ok").sum()),
        "n_failure": int(routes.status.ne("ok").sum()),
        "face_crop_count": reid["face_crop_count"],
        "SSIM_mean": perceptual["ssim_mean"],
        "LPIPS_mean": perceptual["lpips_mean"],
        "AdaFace_cosine_mean": reid["adaface_cosine_sim_mean"],
        "AdaFace_reid_rate": reid["adaface_reid_rate"],
        "ArcFace_cosine_mean": reid["arcface_cosine_sim_mean"],
        "ArcFace_reid_rate": reid["arcface_reid_rate"],
        "FaceNet_cosine_mean": facenet["facenet_cosine_similarity_mean"],
        "FaceNet_reid_rate_060": facenet["facenet_reid_rate_060"],
        "three_attacker_privacy_score": privacy_three_attacker,
        "hardened_utility_score": routes.utility_score.mean(),
        "hardened_balanced_score": routes.balanced_score.mean(),
        "component_runtime_mean_seconds": runtime.component_runtime_mean_seconds,
        "artifact_gate_recall": gate.recall,
        "artifact_gate_precision": gate.precision,
        "artifact_gate_f2": gate.f2_privacy_utility_weighted,
        "artifact_gate_fallbacks": int(runtime.artifact_gate_fallbacks),
    }])
    summary.to_csv(OUT / "13_final_policy_metric_summary.csv", index=False)

    existing = pd.read_csv(ROOT / "outputs/03_anonymisation/11_policy_hardening/02_enhanced_per_image_policy_metrics.csv")
    advanced = pd.read_csv(ROOT / "outputs/03_anonymisation/14_group2_comparison/02_advanced_per_image_scores.csv")
    fixed = pd.concat([
        existing[["relative_path", "method", "enhanced_balanced_score"]],
        advanced[["relative_path", "method", "enhanced_balanced_score"]],
    ], ignore_index=True)
    rng = np.random.default_rng(20260713)
    comparisons = []
    for method, values in fixed.groupby("method"):
        paired = routes[["relative_path", "balanced_score"]].merge(
            values[["relative_path", "enhanced_balanced_score"]],
            on="relative_path",
            validate="one_to_one",
        )
        difference = (paired.balanced_score - paired.enhanced_balanced_score).to_numpy()
        indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
        low, high = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        comparisons.append({
            "comparator": method,
            "policy_mean": paired.balanced_score.mean(),
            "comparator_mean": paired.enhanced_balanced_score.mean(),
            "mean_difference": difference.mean(),
            "ci_low": low,
            "ci_high": high,
            "policy_wins": int((difference > 1e-12).sum()),
            "comparator_wins": int((difference < -1e-12).sum()),
            "ties": int((np.abs(difference) <= 1e-12).sum()),
        })
    ungated = pd.read_csv(ROOT / "outputs/03_anonymisation/14_group2_comparison/04_grouped_heldout_policy_routes.csv")
    paired = routes[["relative_path", "balanced_score"]].merge(
        ungated[["relative_path", "score"]], on="relative_path", validate="one_to_one"
    )
    difference = (paired.balanced_score - paired.score).to_numpy()
    indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
    low, high = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
    comparisons.append({
        "comparator": "ungated_grouped_heldout_policy",
        "policy_mean": paired.balanced_score.mean(),
        "comparator_mean": paired.score.mean(),
        "mean_difference": difference.mean(),
        "ci_low": low,
        "ci_high": high,
        "policy_wins": int((difference > 1e-12).sum()),
        "comparator_wins": int((difference < -1e-12).sum()),
        "ties": int((np.abs(difference) <= 1e-12).sum()),
    })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUT / "14_final_policy_paired_comparison.csv", index=False)

    labels = review.riddle_obvious_artifact.astype(int)
    predictions = review.grouped_out_of_fold_prediction.astype(int)
    quality = pd.DataFrame([{
        "review_sample_size": len(review),
        "initial_riddle_artifacts": int(labels.sum()),
        "grouped_validation_artifacts_detected": int(((labels == 1) & (predictions == 1)).sum()),
        "grouped_validation_artifacts_missed": int(((labels == 1) & (predictions == 0)).sum()),
        "full_fit_known_artifacts_flagged": int(((labels == 1) & (review.full_fit_prediction == 1)).sum()),
        "known_grouped_misses_flagged_by_final_gate": int(((labels == 1) & (predictions == 0) & (review.full_fit_prediction == 1)).sum()),
        "grouped_validation_plausible_outputs_flagged": int(((labels == 0) & (predictions == 1)).sum()),
        "estimated_unseen_participant_miss_rate": float(((labels == 1) & (predictions == 0)).mean()),
        "validation_fallback_rate": predictions.mean(),
        "interpretation": "participant-grouped generalisation estimate; all three grouped misses were flagged by the final full-fit gate, but independent review is still required",
    }])
    quality.to_csv(OUT / "15_final_quality_gate_effect.csv", index=False)

    layered = comparisons[comparisons.comparator.eq("layered_blur_downscale_noise")].iloc[0]
    ungated_row = comparisons[comparisons.comparator.eq("ungated_grouped_heldout_policy")].iloc[0]
    distribution = pd.read_csv(OUT / "06_final_method_distribution.csv")
    lines = [
        "# Final Materialised Adaptive Policy",
        "",
        "The final policy was materialised as 500 output images. RiDDLE candidates pass through a participant-validated SigLIP2 artifact gate; predicted artifacts fall back to layered obfuscation.",
        "",
        "## Direct output metrics",
        "",
        summary.to_markdown(index=False),
        "",
        "## Route distribution",
        "",
        distribution.to_markdown(index=False),
        "",
        "## Quality-gate effect",
        "",
        quality.to_markdown(index=False),
        "",
        "## Paired comparison",
        "",
        comparisons.to_markdown(index=False),
        "",
        f"Against fixed layered obfuscation, the materialised policy gain is `{layered.mean_difference:.6f}`, 95% CI `[{layered.ci_low:.6f}, {layered.ci_high:.6f}]`.",
        f"The quality gate changes the balanced score by `{ungated_row.mean_difference:.6f}` relative to the ungated held-out policy; this is the measured cost of conservative artifact fallback.",
        "All three grouped-validation misses were flagged by the final full-fit gate and use layered fallback in the materialised outputs. This does not prove that every artifact in the unreviewed outputs was detected.",
        "",
    ]
    (OUT / "16_final_policy_evidence_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()
