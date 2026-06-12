#!/usr/bin/env python3
"""Validate the evidence-supported routing pipeline on the 500-frame protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION_METHOD = (
    "fixed_policy_detector_telemetry_multiclass_scale_"
    "hybrid_available_handoff_boxes"
)
FIXED_METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
]


def success_mask(series: pd.Series) -> pd.Series:
    return series.map(
        lambda value: value is True
        or (isinstance(value, (int, float)) and not pd.isna(value) and value == 1)
        or str(value).strip().lower() in {"true", "yes", "ok", "success"}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/06_end_to_end_thesis_validation"),
    )
    return parser.parse_args()


def load_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    manifest = pd.read_csv("outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv")
    conditions = pd.read_csv(
        "outputs/02_face_detection/10_post_detection_condition_annotation/"
        "post_detection_condition_predictions.csv"
    )
    conditions = conditions.loc[
        (conditions["protocol"] == "egocentric_stress_500")
        & (conditions["method_id"] == CONDITION_METHOD)
    ].copy()
    metrics = pd.read_csv(
        "outputs/03_anonymisation/09_policy_scoring/"
        "anonymisation_policy_per_image_metrics.csv"
    )
    multimodal = pd.read_csv(
        "outputs/04_multimodal_privacy/01_multimodal_250_evidence/"
        "04_multimodal_risk_policy.csv"
    )
    candidates = pd.read_csv(
        "outputs/02_face_detection/11_detector_candidate_box_telemetry/"
        "detector_candidate_boxes.csv"
    )
    return manifest, conditions, metrics, multimodal, candidates


def selected_condition(row: pd.Series) -> str:
    if int(row["pred_no_face"]) == 1:
        return "no_face"
    if int(row["pred_single_face"]) == 1:
        return "single_face"
    labels = [
        name.removeprefix("pred_")
        for name, value in row.items()
        if name.startswith("pred_") and int(value) == 1
    ]
    return "|".join(labels) if labels else "fallback_unknown"


def route_method(row: pd.Series) -> tuple[str, str]:
    if int(row["pred_no_face"]) == 1 and int(row["safety_candidate_count"]) == 0:
        return "no_action_copy", "confident_no_face"
    if int(row["pred_no_face"]) == 1:
        return "layered_blur_downscale_noise", "no_face_safety_gate_override"
    if int(row["pred_single_face"]) == 1:
        return "solid_mask_black", "single_face_privacy_first_policy"
    return "layered_blur_downscale_noise", "face_positive_practical_fallback"


def add_copy_metrics(routes: pd.DataFrame) -> pd.DataFrame:
    copy = routes.loc[routes["selected_method"] == "no_action_copy"].copy()
    copy["method"] = "no_action_copy"
    copy["output_path"] = copy["relative_path"].map(lambda p: f"data/castle2024/raw/{p}")
    copy["status"] = "no_face_action_skipped"
    copy["success"] = True
    copy["SSIM"] = 1.0
    copy["LPIPS"] = 0.0
    copy["AdaFace_reid_rate"] = np.nan
    copy["ArcFace_reid_rate"] = np.nan
    copy["privacy_score"] = 1.0
    copy["utility_score"] = 1.0
    copy["runtime_score"] = 1.0
    copy["success_score"] = 1.0
    copy["balanced_oapr_anonymisation_score"] = 1.0
    copy["runtime_seconds"] = 0.0
    return copy


def build_routing_log(
    manifest: pd.DataFrame,
    conditions: pd.DataFrame,
    metrics: pd.DataFrame,
    multimodal: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    if len(manifest) != 500 or manifest["relative_path"].nunique() != 500:
        raise ValueError("Face anonymisation manifest is not the canonical unique 500-frame set")
    if len(conditions) != 500 or conditions["relative_path"].nunique() != 500:
        raise ValueError("Final condition-profiler output does not cover 500 unique frames")

    routes = manifest[["relative_path", "checksum_sha256"]].merge(
        conditions, on="relative_path", how="left", validate="one_to_one"
    )
    safety_candidates = candidates.loc[
        (candidates["protocol"] == "02_egocentric_stress_500")
        & (candidates["detector_name"] == "rfdetr_medium_face_030")
        & (candidates["score"] >= 0.70)
    ].groupby("relative_path").size()
    routes["safety_candidate_count"] = (
        routes["relative_path"].map(safety_candidates).fillna(0).astype(int)
    )
    routed = routes.apply(route_method, axis=1, result_type="expand")
    routed.columns = ["selected_method", "route_reason"]
    routes = pd.concat([routes, routed], axis=1)
    routes["selected_condition_profile"] = routes.apply(selected_condition, axis=1)

    metric_columns = [
        "relative_path",
        "method",
        "output_path",
        "status",
        "success",
        "SSIM",
        "LPIPS",
        "AdaFace_reid_rate",
        "ArcFace_reid_rate",
        "privacy_score",
        "utility_score",
        "runtime_score",
        "success_score",
        "balanced_oapr_anonymisation_score",
        "runtime_seconds",
    ]
    selected = routes.loc[routes["selected_method"] != "no_action_copy"].merge(
        metrics[metric_columns],
        left_on=["relative_path", "selected_method"],
        right_on=["relative_path", "method"],
        how="left",
        validate="one_to_one",
    )
    copy = add_copy_metrics(routes)
    selected = pd.concat([selected, copy], ignore_index=True, sort=False)

    mm = multimodal[["image_id", "multimodal_risk_state", "multimodal_route_action"]].rename(
        columns={"image_id": "relative_path"}
    )
    selected = selected.merge(mm, on="relative_path", how="left", validate="one_to_one")
    selected["multimodal_protocol_covered"] = selected["multimodal_risk_state"].notna()
    selected["multimodal_risk_state"] = selected["multimodal_risk_state"].fillna(
        "separate_protocol_not_overlapping"
    )
    selected["multimodal_route_action"] = selected["multimodal_route_action"].fillna(
        "not_evaluated_on_face_protocol"
    )
    selected["output_exists"] = selected["output_path"].map(
        lambda value: Path(str(value)).exists() if pd.notna(value) else False
    )
    selected["pipeline_status"] = np.where(
        selected["output_exists"] & success_mask(selected["success"]), "ok", "failed"
    )
    return selected.sort_values("relative_path").reset_index(drop=True)


def summarise_policy(name: str, frame_rows: pd.DataFrame) -> dict[str, object]:
    successful = frame_rows.loc[frame_rows["pipeline_status"] == "ok"]
    face_positive = successful.loc[successful["method"] != "no_action_copy"]
    effective_scores = frame_rows["balanced_oapr_anonymisation_score"].where(
        frame_rows["pipeline_status"] == "ok", 0.0
    )
    return {
        "policy": name,
        "n_input_frames": len(frame_rows),
        "n_success": len(successful),
        "n_failure": len(frame_rows) - len(successful),
        "method_distribution": json.dumps(
            successful["method"].value_counts().sort_index().to_dict(), sort_keys=True
        ),
        "SSIM_mean": successful["SSIM"].mean(),
        "LPIPS_mean": successful["LPIPS"].mean(),
        "AdaFace_reid_rate": face_positive["AdaFace_reid_rate"].mean(),
        "ArcFace_reid_rate": face_positive["ArcFace_reid_rate"].mean(),
        "privacy_score_mean": successful["privacy_score"].mean(),
        "utility_score_mean": successful["utility_score"].mean(),
        "runtime_score_mean": successful["runtime_score"].mean(),
        "success_score_mean": successful["success_score"].mean(),
        "balanced_oapr_anonymisation_score_mean": effective_scores.mean(),
        "runtime_total_seconds": successful["runtime_seconds"].fillna(0).sum(),
        "runtime_mean_seconds": successful["runtime_seconds"].fillna(0).mean(),
    }


def fixed_policy_rows(metrics: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = metrics.loc[metrics["method"] == method].copy()
    output_exists = rows["output_path"].notna() & rows["output_path"].map(
        lambda value: Path(str(value)).exists() if pd.notna(value) else False
    )
    final_success = output_exists & ~rows["status"].fillna("").str.contains(
        "fail|error", case=False, regex=True
    )
    rows["pipeline_status"] = np.where(
        final_success, "ok", "failed"
    )
    rows["success_score"] = final_success.astype(float)
    rows["balanced_oapr_anonymisation_score"] = (
        0.50 * rows["privacy_score"]
        + 0.30 * rows["utility_score"]
        + 0.10 * rows["runtime_score"]
        + 0.10 * rows["success_score"]
    )
    return rows


def bootstrap_mean_ci(values: pd.Series, seed: int = 42) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_pairwise_comparison(routes: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    integrated = routes[["relative_path", "balanced_oapr_anonymisation_score"]].rename(
        columns={"balanced_oapr_anonymisation_score": "integrated_score"}
    )
    rows: list[dict[str, object]] = []
    for method in FIXED_METHODS:
        fixed = fixed_policy_rows(metrics, method)[
            ["relative_path", "balanced_oapr_anonymisation_score", "pipeline_status"]
        ].copy()
        fixed["fixed_score"] = fixed["balanced_oapr_anonymisation_score"].where(
            fixed["pipeline_status"] == "ok", 0.0
        )
        paired = integrated.merge(
            fixed[["relative_path", "fixed_score"]],
            on="relative_path",
            how="inner",
            validate="one_to_one",
        )
        paired["difference"] = paired["integrated_score"] - paired["fixed_score"]
        low, high = bootstrap_mean_ci(paired["difference"])
        tolerance = 1e-12
        rows.append(
            {
                "fixed_method": method,
                "n_paired_frames": len(paired),
                "integrated_mean": paired["integrated_score"].mean(),
                "fixed_mean": paired["fixed_score"].mean(),
                "mean_difference_integrated_minus_fixed": paired["difference"].mean(),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "integrated_win_count": int((paired["difference"] > tolerance).sum()),
                "fixed_win_count": int((paired["difference"] < -tolerance).sum()),
                "tie_count": int((paired["difference"].abs() <= tolerance).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_difference_integrated_minus_fixed", ascending=False
    )


def build_comparison(routes: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    summaries = [summarise_policy("integrated_condition_aware_policy", routes)]
    for method in FIXED_METHODS:
        summaries.append(summarise_policy(f"fixed_{method}", fixed_policy_rows(metrics, method)))
    result = pd.DataFrame(summaries)
    integrated_score = result.loc[
        result["policy"] == "integrated_condition_aware_policy",
        "balanced_oapr_anonymisation_score_mean",
    ].iloc[0]
    result["score_difference_vs_integrated"] = (
        result["balanced_oapr_anonymisation_score_mean"] - integrated_score
    )
    result["comparison_boundary"] = (
        "Same 500-frame protocol; no-action rows are valid only where the final condition "
        "profiler predicts no face."
    )
    return result.sort_values(
        "balanced_oapr_anonymisation_score_mean", ascending=False
    ).reset_index(drop=True)


def build_previous_oapr_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    previous = pd.read_csv("outputs/05_oapr/12_oapr_full_metric_summary.csv")
    prior = previous[[
        "method",
        "objective_mode",
        "n_input_frames",
        "n_success",
        "n_failure",
        "method_counts",
        "SSIM_mean",
        "LPIPS_mean",
        "AdaFace_reid_rate",
        "ArcFace_reid_rate",
        "runtime_mean_seconds",
        "runtime_total_seconds",
    ]].copy()
    prior.insert(0, "policy_generation", "previous_oapr")
    prior = prior.rename(columns={"method_counts": "method_distribution"})

    current = comparison.loc[
        comparison["policy"] == "integrated_condition_aware_policy"
    ].copy()
    current = current.rename(columns={"policy": "method"})
    current.insert(0, "policy_generation", "integrated_current")
    current["objective_mode"] = "balanced_condition_aware"
    columns = list(prior.columns)
    return pd.concat([prior, current.reindex(columns=columns)], ignore_index=True)


def write_summary(
    output: Path,
    routes: pd.DataFrame,
    comparison: pd.DataFrame,
    prior: pd.DataFrame,
) -> None:
    current = comparison.loc[
        comparison["policy"] == "integrated_condition_aware_policy"
    ].iloc[0]
    best_fixed = comparison.loc[comparison["policy"].str.startswith("fixed_")].iloc[0]
    distribution = routes["method"].value_counts().sort_index().to_dict()
    unsafe_skips = int(
        (
            (routes["method"] == "no_action_copy")
            & (routes["true_no_face"] == 0)
        ).sum()
    )
    safety_overrides = int((routes["route_reason"] == "no_face_safety_gate_override").sum())
    lines = [
        "# End-to-End Thesis Pipeline Validation",
        "",
        "## Result",
        "",
        f"The integrated policy completed {int(current.n_success)}/500 frames with "
        f"{int(current.n_failure)} failures. It routed {json.dumps(distribution, sort_keys=True)}.",
        "",
        f"Its balanced OAPR anonymisation score was "
        f"{current.balanced_oapr_anonymisation_score_mean:.4f}. The strongest fixed "
        f"comparator was `{best_fixed.policy}` at "
        f"{best_fixed.balanced_oapr_anonymisation_score_mean:.4f}.",
        "",
        "## Interpretation",
        "",
        "This experiment tests whether the completed condition profiler changes the "
        "anonymisation decision usefully. It does not retrain or regenerate any "
        "anonymisation method. Existing outputs are selected through the runtime-safe "
        "category policy and validated for presence, success, privacy, utility, and runtime.",
        "",
        "The integrated result is objective-specific. It does not establish global "
        "dominance over every fixed method. No-face copying is permitted only after the "
        "evidence-supported condition profiler predicts no face; residual detector risk "
        "therefore remains part of the claim boundary.",
        "",
        f"After the RF-DETR candidate safety gate, the final policy retained "
        f"{unsafe_skips} unsafe false `no_face` skip(s). The gate overrides a no-action "
        f"decision whenever a high-confidence face candidate remains; it fired on "
        f"{safety_overrides} frame(s).",
        "",
        "## Multimodal Boundary",
        "",
        f"The face-anonymisation 500 and reviewed multimodal 250 overlap on "
        f"{int(routes.multimodal_protocol_covered.sum())} frames. Multimodal routing is "
        "validated separately and is not imputed onto these 500 frames.",
        "",
        "## Previous OAPR Comparison",
        "",
        "The previous OAPR modes remain valid objective-specific baselines. Their metrics "
        "were measured from materialised routed outputs; the current row uses the same "
        "500-frame face protocol but adds the final condition-aware anonymisation policy.",
        "",
        prior.to_markdown(index=False),
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, conditions, metrics, multimodal, candidates = load_inputs()
    routes = build_routing_log(manifest, conditions, metrics, multimodal, candidates)
    comparison = build_comparison(routes, metrics)
    pairwise = build_pairwise_comparison(routes, metrics)
    prior = build_previous_oapr_comparison(comparison)

    routes.to_csv(args.output_dir / "01_integrated_routing_log.csv", index=False)
    routes[[
        "relative_path",
        "selected_method",
        "output_path",
        "output_exists",
        "pipeline_status",
    ]].to_csv(args.output_dir / "02_integrated_output_manifest.csv", index=False)
    routes.loc[routes["pipeline_status"] != "ok", [
        "relative_path",
        "selected_method",
        "output_path",
        "status",
        "pipeline_status",
    ]].to_csv(args.output_dir / "03_integrated_failure_log.csv", index=False)
    comparison.to_csv(args.output_dir / "04_integrated_vs_fixed_methods.csv", index=False)
    pairwise.to_csv(args.output_dir / "05_integrated_pairwise_evidence.csv", index=False)
    prior.to_csv(args.output_dir / "06_integrated_vs_previous_oapr.csv", index=False)
    write_summary(
        args.output_dir / "07_end_to_end_validation_summary.md",
        routes,
        comparison,
        prior,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
