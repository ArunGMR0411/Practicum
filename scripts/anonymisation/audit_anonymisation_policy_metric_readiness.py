#!/usr/bin/env python3

"""Audit whether anonymisation metrics are ready for condition-aware OAPR scoring."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs/03_anonymisation/08_policy_metric_readiness"
ANON_MANIFEST = PROJECT_ROOT / "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv"
CONDITION_MANIFEST = PROJECT_ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
CONDITION_PREDICTIONS = PROJECT_ROOT / "outputs/02_face_detection/10_post_detection_condition_annotation/post_detection_condition_predictions.csv"
MULTIMODAL_POLICY = PROJECT_ROOT / "outputs/04_multimodal_privacy/01_multimodal_250_evidence/04_multimodal_risk_policy.csv"
ALL_METHODS = PROJECT_ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv"

FINAL_CONDITION_METHOD = "fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes"

METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
    "styleid_stylegan",
    "fams",
]

DETERMINISTIC_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/02_deterministic_baselines/01_output_manifest.csv"
NULLFACE_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/01_nullface_full_500_manifest.csv"
NULLFACE_QUALITY = PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/quality_review/01_nullface_quality_metrics.csv"
DIFFUSION_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/01_diffusion_manifest_full_500.csv"
DIFFUSION_PERCEPTUAL = PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/04_diffusion_full_500_perceptual.json"
DIFFUSION_REID = PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/05_diffusion_full_500_reid_details.json"
RP_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/01_rp_final_manifest.csv"
RP_PERCEPTUAL = PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/06_rp_final_perceptual.json"
RP_REID = PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/07_rp_final_reid_details.json"
STYLEID_FAMS_MANIFEST = PROJECT_ROOT / "outputs/03_anonymisation/06_styleid_fams/02_pilot_manifest.csv"
POLICY_METRIC_DIR = PROJECT_ROOT / "outputs/03_anonymisation/08_policy_metric_readiness"
DETERMINISTIC_PERCEPTUAL_COMPLETED = POLICY_METRIC_DIR / "deterministic_per_image_perceptual.json"
NULLFACE_PERCEPTUAL_COMPLETED = POLICY_METRIC_DIR / "nullface_per_image_perceptual.json"
POLICY_REID_DIR = POLICY_METRIC_DIR / "reid"


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


def save_csv(rows: list[dict[str, Any]], fieldnames: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_perceptual_details(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["relative_path", "method", "SSIM", "LPIPS"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("detailed", []) if isinstance(payload, dict) else []
    if not details:
        return pd.DataFrame(columns=["relative_path", "method", "SSIM", "LPIPS"])
    df = pd.DataFrame(details)
    return df.rename(columns={"ssim": "SSIM", "lpips": "LPIPS"})


def load_reid_details(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "relative_path",
                "method",
                "AdaFace_cosine_mean",
                "AdaFace_reid_rate",
                "ArcFace_cosine_mean",
                "ArcFace_reid_rate",
                "face_crop_count",
            ]
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"image_id": "relative_path"})
    grouped = (
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
    return grouped


def load_method_status(method: str, all_methods: pd.DataFrame) -> dict[str, Any]:
    row = all_methods[all_methods["method"] == method]
    if row.empty:
        return {
            "n_input_frames": 0,
            "n_success": 0,
            "n_failure": 0,
            "evidence_level": "not_available",
            "final_status": "not_available",
        }
    data = row.iloc[0].to_dict()
    return {
        "n_input_frames": data.get("n_input_frames", "not_available"),
        "n_success": data.get("n_success", "not_available"),
        "n_failure": data.get("n_failure", "not_available"),
        "evidence_level": data.get("evidence_level", "not_available"),
        "final_status": data.get("final_status", "not_available"),
    }


def build_condition_frame() -> pd.DataFrame:
    base = pd.read_csv(ANON_MANIFEST)[["relative_path"]].copy()
    manual = pd.read_csv(CONDITION_MANIFEST)
    manual_cols = ["relative_path", "box_count", "face_count_category", "face_scale_category"]
    for label in CONDITION_LABELS:
        manual[f"manual_{label}"] = manual.get("condition_label", "").astype(str).str.contains(label, regex=False)
        if label == "no_face":
            manual[f"manual_{label}"] = manual.get("face_count_category", "").astype(str).eq("no_face")
    manual_cols.extend([f"manual_{label}" for label in CONDITION_LABELS])
    merged = base.merge(manual[manual_cols], on="relative_path", how="left")

    predictions = pd.read_csv(CONDITION_PREDICTIONS)
    final = predictions[
        (predictions["protocol"] == "egocentric_stress_500")
        & (predictions["method_id"] == FINAL_CONDITION_METHOD)
    ].copy()
    pred_cols = ["relative_path", "detected_box_count"]
    for label in CONDITION_LABELS:
        col = f"pred_{label}"
        if col in final.columns:
            pred_cols.append(col)
    merged = merged.merge(final[pred_cols], on="relative_path", how="left")

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
        merged["text_present"] = "not_available"
        merged["screen_present"] = "not_available"
        merged["no_text_screen_risk"] = "not_available"
        merged["multimodal_risk_state"] = "not_available"
    return merged


def method_artifacts(method: str) -> dict[str, Any]:
    if method in {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"}:
        return {
            "manifest": DETERMINISTIC_MANIFEST,
            "perceptual": DETERMINISTIC_PERCEPTUAL_COMPLETED,
            "reid": POLICY_REID_DIR / f"{method}_reid_details.json",
            "quality": None,
            "runtime_source": DETERMINISTIC_MANIFEST,
        }
    if method == "nullface":
        return {
            "manifest": NULLFACE_MANIFEST,
            "perceptual": NULLFACE_PERCEPTUAL_COMPLETED,
            "reid": POLICY_REID_DIR / "nullface_reid_details.json",
            "quality": NULLFACE_QUALITY,
            "runtime_source": PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/03_nullface_runtime_summary.csv",
        }
    if method == "diffusion_low_step":
        return {
            "manifest": DIFFUSION_MANIFEST,
            "perceptual": DIFFUSION_PERCEPTUAL,
            "reid": DIFFUSION_REID,
            "quality": PROJECT_ROOT / "outputs/03_anonymisation/04_diffusion/quality_review/04_diffusion_quality_adjudicated.csv",
            "runtime_source": DIFFUSION_MANIFEST,
        }
    if method == "reverse_personalization":
        return {
            "manifest": RP_MANIFEST,
            "perceptual": RP_PERCEPTUAL,
            "reid": RP_REID,
            "quality": PROJECT_ROOT / "outputs/03_anonymisation/05_reverse_personalization/04_rp_failure_root_cause.csv",
            "runtime_source": RP_MANIFEST,
        }
    if method in {"styleid_stylegan", "fams"}:
        return {
            "manifest": STYLEID_FAMS_MANIFEST,
            "perceptual": None,
            "reid": None,
            "quality": None,
            "runtime_source": STYLEID_FAMS_MANIFEST,
        }
    return {"manifest": None, "perceptual": None, "reid": None, "quality": None, "runtime_source": None}


def available_output_rows(method: str) -> pd.DataFrame:
    artifacts = method_artifacts(method)
    manifest = artifacts["manifest"]
    if not manifest or not manifest.is_file():
        return pd.DataFrame(columns=["relative_path", "method", "status", "runtime_seconds", "output_path"])
    df = pd.read_csv(manifest)
    if method in {"blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"}:
        df = df[df["method"] == method].copy()
    elif method in {"styleid_stylegan", "fams"}:
        if method == "styleid_stylegan":
            df = df[df["variant"].astype(str).str.startswith("stylegan")].copy()
        else:
            df = df[df["variant"].astype(str).str.startswith("fams")].copy()
        df["method"] = method
    if "status" not in df.columns:
        df["status"] = "ok"
    if "runtime_seconds" not in df.columns:
        df["runtime_seconds"] = pd.NA
    if "output_path" not in df.columns:
        df["output_path"] = pd.NA
    return df


def count_existing_outputs(df: pd.DataFrame) -> int:
    if "output_path" not in df.columns or df.empty:
        return 0
    count = 0
    for path in df["output_path"].dropna().astype(str):
        if (PROJECT_ROOT / path).is_file():
            count += 1
    return count


def build_metric_availability() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    all_methods = pd.read_csv(ALL_METHODS)
    detailed_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        status = load_method_status(method, all_methods)
        manifest_df = available_output_rows(method)
        artifacts = method_artifacts(method)
        perceptual_df = load_perceptual_details(artifacts["perceptual"]) if artifacts["perceptual"] else pd.DataFrame()
        reid_df = load_reid_details(artifacts["reid"]) if artifacts["reid"] else pd.DataFrame()

        detailed = manifest_df[["relative_path", "method", "status", "runtime_seconds", "output_path"]].copy() if not manifest_df.empty else pd.DataFrame(columns=["relative_path", "method", "status", "runtime_seconds", "output_path"])
        if not perceptual_df.empty:
            detailed = detailed.merge(
                perceptual_df[["relative_path", "method", "SSIM", "LPIPS"]],
                on=["relative_path", "method"],
                how="left",
            )
        else:
            detailed["SSIM"] = pd.NA
            detailed["LPIPS"] = pd.NA
        if not reid_df.empty:
            detailed = detailed.merge(reid_df, on=["relative_path", "method"], how="left")
        else:
            for col in [
                "AdaFace_cosine_mean",
                "AdaFace_reid_rate",
                "ArcFace_cosine_mean",
                "ArcFace_reid_rate",
                "face_crop_count",
            ]:
                detailed[col] = pd.NA
        detailed_frames[method] = detailed

        manifest_rows = int(len(manifest_df))
        success_rows = int((manifest_df["status"].astype(str).isin(["ok", "existing", "completed", "copied"])).sum()) if manifest_rows else 0
        output_files_present = count_existing_outputs(manifest_df)
        per_image_perceptual_rows = int(detailed["SSIM"].notna().sum()) if not detailed.empty else 0
        per_image_reid_rows = int(detailed["AdaFace_cosine_mean"].notna().sum()) if not detailed.empty else 0
        runtime_rows = int(detailed["runtime_seconds"].notna().sum()) if not detailed.empty else 0
        condition_join_rows = 0
        if not detailed.empty:
            condition_join_rows = int(detailed["relative_path"].isin(pd.read_csv(ANON_MANIFEST)["relative_path"]).sum())

        aggregate_runtime_available = (
            method == "nullface"
            and artifacts["runtime_source"] is not None
            and artifacts["runtime_source"].is_file()
        )

        if method in {"styleid_stylegan", "fams"}:
            readiness = "not_policy_ready_quality_limited"
            next_action = "Do not include as policy candidate unless systematic tuning is reopened."
        elif int(status.get("n_success", 0) or 0) < 500:
            readiness = "bounded_policy_candidate_partial_success"
            next_action = "Use as bounded/non-default candidate; category scores need failure penalty."
        elif (
            per_image_perceptual_rows == 500
            and per_image_reid_rows > 0
            and (runtime_rows >= 450 or aggregate_runtime_available)
        ):
            readiness = "policy_ready_per_image_metrics"
            next_action = (
                "Can compute category-aware policy score now; use aggregate runtime boundary."
                if aggregate_runtime_available and runtime_rows < 450
                else "Can compute category-aware policy score now."
            )
        elif output_files_present >= 450:
            readiness = "local_metric_recompute_needed"
            next_action = "Output images exist; recompute missing per-image metrics before category-specific scoring."
        else:
            readiness = "not_ready_missing_outputs"
            next_action = "Requires generation/output recovery before scoring."

        rows.append(
            {
                "method": method,
                **status,
                "manifest_rows": manifest_rows,
                "success_rows_in_manifest": success_rows,
                "output_files_present": output_files_present,
                "per_image_perceptual_rows": per_image_perceptual_rows,
                "per_image_reid_rows": per_image_reid_rows,
                "runtime_rows": runtime_rows,
                "condition_join_rows": condition_join_rows,
                "manifest_source": rel(artifacts["manifest"]) if artifacts["manifest"] else "not_available",
                "perceptual_source": rel(artifacts["perceptual"]) if artifacts["perceptual"] else "not_available",
                "reid_source": rel(artifacts["reid"]) if artifacts["reid"] else "not_available",
                "runtime_source": rel(artifacts["runtime_source"]) if artifacts["runtime_source"] else "not_available",
                "readiness_status": readiness,
                "next_action": next_action,
            }
        )
    return rows, detailed_frames


def build_category_coverage(condition_df: pd.DataFrame, detailed_frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories: list[tuple[str, str]] = []
    for label in CONDITION_LABELS:
        categories.append((f"manual_{label}", "manual"))
        categories.append((f"pred_{label}", "predicted_oapr_condition_profile"))
    categories.extend(
        [
            ("text_present", "multimodal"),
            ("screen_present", "multimodal"),
            ("no_text_screen_risk", "multimodal"),
        ]
    )

    for col, source in categories:
        if col not in condition_df.columns:
            continue
        mask = condition_df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        support = int(mask.sum())
        if support == 0:
            continue
        category_paths = set(condition_df.loc[mask, "relative_path"].astype(str))
        for method, detailed in detailed_frames.items():
            if detailed.empty:
                coverage = {
                    "output_rows": 0,
                    "success_rows": 0,
                    "perceptual_rows": 0,
                    "reid_rows": 0,
                    "runtime_rows": 0,
                }
            else:
                sub = detailed[detailed["relative_path"].astype(str).isin(category_paths)]
                coverage = {
                    "output_rows": int(len(sub)),
                    "success_rows": int(sub["status"].astype(str).isin(["ok", "existing", "completed", "copied"]).sum()),
                    "perceptual_rows": int(sub["SSIM"].notna().sum()) if "SSIM" in sub else 0,
                    "reid_rows": int(sub["AdaFace_cosine_mean"].notna().sum()) if "AdaFace_cosine_mean" in sub else 0,
                    "runtime_rows": int(sub["runtime_seconds"].notna().sum()) if "runtime_seconds" in sub else 0,
                }
            rows.append(
                {
                    "category": col,
                    "category_source": source,
                    "support_count": support,
                    "method": method,
                    **coverage,
                    "category_policy_ready": (
                        "yes"
                        if coverage["perceptual_rows"] == support
                        and (coverage["reid_rows"] > 0 or col == "manual_no_face" or col == "pred_no_face")
                        and (
                            coverage["runtime_rows"] > 0
                            or (
                                method == "nullface"
                                and (PROJECT_ROOT / "outputs/03_anonymisation/03_nullface/03_nullface_runtime_summary.csv").is_file()
                            )
                        )
                        else "no"
                    ),
                }
            )
    return rows


def write_summary(avail_rows: list[dict[str, Any]], category_rows: list[dict[str, Any]]) -> None:
    ready = [row["method"] for row in avail_rows if row["readiness_status"] == "policy_ready_per_image_metrics"]
    recompute = [row["method"] for row in avail_rows if row["readiness_status"] == "local_metric_recompute_needed"]
    partial = [row["method"] for row in avail_rows if row["readiness_status"] == "bounded_policy_candidate_partial_success"]
    limited = [row["method"] for row in avail_rows if row["readiness_status"] == "not_policy_ready_quality_limited"]

    lines = [
        "# Anonymisation Policy Metric Readiness",
        "",
        "Purpose: decide whether category-aware OAPR anonymisation scoring can run from retained evidence, or whether method generation/metric recomputation is needed.",
        "",
        "Inputs:",
        "",
        f"- 500-frame anonymisation manifest: `{rel(ANON_MANIFEST)}`.",
        f"- Manual condition profile: `{rel(CONDITION_MANIFEST)}`.",
        f"- Final OAPR condition profile: `{rel(CONDITION_PREDICTIONS)}` filtered to `{FINAL_CONDITION_METHOD}`.",
        f"- Multimodal risk policy: `{rel(MULTIMODAL_POLICY)}`.",
        f"- Consolidated method metrics: `{rel(ALL_METHODS)}`.",
        "",
        "Findings:",
        "",
        f"- Policy-ready per-image metric methods now: `{', '.join(ready) if ready else 'none'}`.",
        f"- Methods with output images but missing per-image metric detail: `{', '.join(recompute) if recompute else 'none'}`.",
        f"- Partial/bounded candidates requiring failure penalty: `{', '.join(partial) if partial else 'none'}`.",
        f"- Quality-limited/non-policy methods: `{', '.join(limited) if limited else 'none'}`.",
        "",
        "Compute decision:",
        "",
        "- Additional high-memory compute is not needed for the policy-readiness audit or scoring logic.",
        "- Deterministic and NullFace output regeneration is not needed because those outputs already exist.",
        "- Per-image perceptual/ReID details are now complete for deterministic baselines and NullFace.",
        "- Diffusion already had full per-image perceptual and per-face ReID detail.",
        "- Reverse Personalization has detailed metrics for the successful 482 outputs and must be scored with an explicit 18-frame failure penalty.",
        "",
        "Boundary:",
        "",
        "- Do not choose category-specific anonymisation winners from aggregate method means alone.",
        "- The next step is category-aware OAPR anonymisation score computation from the completed per-image evidence.",
    ]
    (OUTPUT_DIR / "anonymisation_policy_metric_readiness_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR = output_dir

    condition_df = build_condition_frame()
    condition_df.to_csv(OUTPUT_DIR / "anonymisation_policy_condition_join_base.csv", index=False)

    avail_rows, detailed_frames = build_metric_availability()
    availability_fields = [
        "method",
        "n_input_frames",
        "n_success",
        "n_failure",
        "evidence_level",
        "final_status",
        "manifest_rows",
        "success_rows_in_manifest",
        "output_files_present",
        "per_image_perceptual_rows",
        "per_image_reid_rows",
        "runtime_rows",
        "condition_join_rows",
        "manifest_source",
        "perceptual_source",
        "reid_source",
        "runtime_source",
        "readiness_status",
        "next_action",
    ]
    save_csv(avail_rows, availability_fields, OUTPUT_DIR / "anonymisation_policy_metric_readiness.csv")

    category_rows = build_category_coverage(condition_df, detailed_frames)
    category_fields = [
        "category",
        "category_source",
        "support_count",
        "method",
        "output_rows",
        "success_rows",
        "perceptual_rows",
        "reid_rows",
        "runtime_rows",
        "category_policy_ready",
    ]
    save_csv(category_rows, category_fields, OUTPUT_DIR / "anonymisation_policy_category_metric_coverage.csv")

    write_summary(avail_rows, category_rows)
    print(
        {
            "output_dir": rel(OUTPUT_DIR),
            "methods": len(avail_rows),
            "category_rows": len(category_rows),
            "ready_methods": [
                row["method"]
                for row in avail_rows
                if row["readiness_status"] == "policy_ready_per_image_metrics"
            ],
            "local_recompute_needed": [
                row["method"]
                for row in avail_rows
                if row["readiness_status"] == "local_metric_recompute_needed"
            ],
        }
    )


if __name__ == "__main__":
    main()
