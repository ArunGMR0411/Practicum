#!/usr/bin/env python3
"""Execute the complete multi-objective anonymisation policy on 500 frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/03_anonymisation/10_full_policy"
SCORE_COLUMNS = {
    "balanced_standard": "balanced_oapr_anonymisation_score",
    "balanced_high_compute": "balanced_oapr_anonymisation_score",
    "privacy_first": "privacy_first_score",
    "utility_priority": "utility_preserving_score",
    "runtime_practical": "runtime_practical_score",
    "failure_avoidance": "balanced_oapr_anonymisation_score",
}
STANDARD = {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"}
ADVANCED = {"nullface", "diffusion_low_step", "reverse_personalization"}


def primary_category(row: pd.Series) -> str:
    if row["route_reason"] == "confident_no_face":
        return "no_face"
    priority = [
        "single_face", "very_small_or_distant_face", "large_face", "motion_blur_or_low_sharpness",
        "small_face", "medium_face", "mixed_scale_face", "edge_or_partial_face",
        "profile_or_occluded_face", "low_light_or_dim", "high_clutter",
        "multi_face", "downward_egocentric_view",
    ]
    for category in priority:
        if int(row.get(f"pred_{category}", 0)) == 1:
            return category
    return "multi_face"


def eligible(mode: str) -> set[str]:
    if mode in {"balanced_high_compute", "utility_priority"}:
        return STANDARD | ADVANCED
    if mode == "failure_avoidance":
        return STANDARD | {"nullface", "diffusion_low_step"}
    return STANDARD


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(ROOT / "outputs/06_end_to_end_thesis_validation/01_integrated_routing_log.csv")
    category_scores = pd.read_csv(ROOT / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_category_scores.csv")
    metrics = pd.read_csv(ROOT / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv")
    readiness = pd.read_csv(ROOT / "outputs/03_anonymisation/08_policy_metric_readiness/anonymisation_policy_metric_readiness.csv")
    routes["policy_category"] = routes.apply(primary_category, axis=1)

    logs: list[dict[str, object]] = []
    for mode, score_column in SCORE_COLUMNS.items():
        allowed = eligible(mode)
        for row in routes.to_dict("records"):
            category = row["policy_category"]
            if category == "no_face":
                logs.append({
                    "relative_path": row["relative_path"], "objective_mode": mode,
                    "policy_category": category, "selected_method": "no_action_copy",
                    "selected_score": 1.0, "eligible_methods": json.dumps(sorted(allowed)),
                    "candidate_scores": "{}", "rejected_methods": "{}",
                    "selection_reason": "condition_profiler_and_detector_safety_gate_confirm_no_face",
                    "actual_privacy_score": 1.0, "actual_utility_score": 1.0,
                    "actual_runtime_score": 1.0, "actual_success_score": 1.0,
                    "actual_objective_score": 1.0,
                    "output_path": f"data/castle2024/raw/{row['relative_path']}", "status": "ok",
                })
                continue
            candidates = category_scores.loc[
                category_scores["category"].eq(category)
                & category_scores["method"].isin(allowed)
            ].copy()
            support = int(candidates["scored_frame_count"].max()) if not candidates.empty else 0
            if candidates.empty or support < 20:
                selected = "layered_blur_downscale_noise"
                reason = "low_support_or_missing_category_fallback"
            else:
                candidates = candidates.sort_values(score_column, ascending=False)
                selected = str(candidates.iloc[0]["method"])
                reason = f"highest_{score_column}_for_{category}"
            scores = {
                str(item.method): round(float(getattr(item, score_column)), 6)
                for item in candidates.itertuples(index=False)
            }
            rejected = {}
            for method in sorted(STANDARD | ADVANCED | {"styleid_stylegan", "fams"}):
                if method == selected:
                    continue
                if method in {"styleid_stylegan", "fams"}:
                    rejected[method] = "quality_limited_not_policy_eligible"
                elif method not in allowed:
                    rejected[method] = "compute_or_objective_gate"
                elif method == "reverse_personalization" and mode == "failure_avoidance":
                    rejected[method] = "failure_gate"
                else:
                    rejected[method] = "lower_objective_score"
            metric = metrics.loc[
                metrics["relative_path"].eq(row["relative_path"])
                & metrics["method"].eq(selected)
            ]
            if metric.empty:
                output_path, status, selected_score = "", "missing_metric_row", np.nan
                privacy = utility = runtime = success = actual_objective = np.nan
            else:
                item = metric.iloc[0]
                output_path = item.get("output_path", "")
                status = "ok" if not str(item.get("status", "")).lower().startswith("fail") else "failed"
                selected_score = scores.get(selected, float(item.get(score_column, np.nan)))
                privacy = float(item.get("privacy_score", np.nan))
                utility = float(item.get("utility_score", np.nan))
                runtime = float(item.get("runtime_score", np.nan))
                success = 1.0 if status == "ok" else 0.0
                actual_objective = float(item.get(score_column, np.nan))
            logs.append({
                "relative_path": row["relative_path"], "objective_mode": mode,
                "policy_category": category, "selected_method": selected,
                "selected_score": selected_score, "eligible_methods": json.dumps(sorted(allowed)),
                "candidate_scores": json.dumps(scores, sort_keys=True),
                "rejected_methods": json.dumps(rejected, sort_keys=True),
                "selection_reason": reason,
                "actual_privacy_score": privacy, "actual_utility_score": utility,
                "actual_runtime_score": runtime, "actual_success_score": success,
                "actual_objective_score": actual_objective,
                "output_path": output_path, "status": status,
            })
    log = pd.DataFrame(logs)
    log.to_csv(OUT / "01_full_policy_routing_log.csv", index=False)
    distribution = log.groupby(["objective_mode", "selected_method"]).size().rename("image_count").reset_index()
    distribution.to_csv(OUT / "02_full_policy_method_distribution.csv", index=False)
    mode_summary = log.groupby("objective_mode").agg(
        n_input_frames=("relative_path", "size"),
        n_success=("status", lambda values: int((values == "ok").sum())),
        n_failure=("status", lambda values: int((values != "ok").sum())),
        category_policy_score_mean=("selected_score", "mean"),
        actual_privacy_score_mean=("actual_privacy_score", "mean"),
        actual_utility_score_mean=("actual_utility_score", "mean"),
        actual_runtime_score_mean=("actual_runtime_score", "mean"),
        actual_objective_score_mean=("actual_objective_score", "mean"),
    ).reset_index()
    mode_summary.to_csv(OUT / "03_full_policy_mode_summary.csv", index=False)
    readiness.to_csv(OUT / "04_full_policy_method_eligibility.csv", index=False)
    lines = ["# Comprehensive Anonymisation Policy Evaluation", "", "## Method distributions", "", distribution.to_markdown(index=False), "", "## Mode summary", "", mode_summary.to_markdown(index=False), "", "Every eligible method is scored before selection. A method can receive zero selections when another eligible method has a higher category/objective score. This evaluation does not claim complete anonymisation.", ""]
    (OUT / "05_full_policy_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(distribution.to_string(index=False))
    print("\n", mode_summary.to_string(index=False))


if __name__ == "__main__":
    main()
