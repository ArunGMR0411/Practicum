#!/usr/bin/env python3
"""Evaluate Group 2 anonymisers under the established policy-hardening protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import face_alignment
import lpips
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_policy_hardening as hardening  # noqa: E402


RAW = ROOT / "data/castle2024/raw"
REVIEWED = ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
ROUTES = ROOT / "outputs/06_end_to_end_thesis_validation/01_integrated_routing_log.csv"
EXISTING = ROOT / "outputs/03_anonymisation/11_policy_hardening/02_enhanced_per_image_policy_metrics.csv"
OUT = ROOT / "outputs/03_anonymisation/14_group2_comparison"
METHOD_DIRS = {
    "riddle": ROOT / "outputs/03_anonymisation/12_riddle",
    "falco": ROOT / "outputs/03_anonymisation/13_falco",
}
PRIVACY_FLOOR = 0.95


def load_base_rows(method: str, directory: Path) -> pd.DataFrame:
    manifest = pd.read_csv(directory / f"01_{method}_500_manifest.csv")
    perceptual = json.loads((directory / f"04_{method}_perceptual.json").read_text())
    perceptual = pd.DataFrame(perceptual["detailed"]).rename(columns={"ssim": "SSIM", "lpips": "LPIPS"})
    reid = pd.DataFrame(json.loads((directory / f"05_{method}_reid_details.json").read_text()))
    if reid.empty:
        grouped = pd.DataFrame(columns=["relative_path"])
    else:
        grouped = reid.groupby("image_id").agg(
            AdaFace_cosine_mean=("adaface_cosine_sim", "mean"),
            AdaFace_reid_rate=("adaface_hit", "mean"),
            ArcFace_cosine_mean=("arcface_cosine_sim", "mean"),
            ArcFace_reid_rate=("arcface_hit", "mean"),
        ).reset_index().rename(columns={"image_id": "relative_path"})
    data = manifest.merge(perceptual[["relative_path", "SSIM", "LPIPS"]], on="relative_path", how="left", validate="one_to_one")
    data = data.merge(grouped, on="relative_path", how="left", validate="one_to_one")
    data["method"] = method
    data["success"] = data["status"].isin(["ok", "copied_no_face"]).astype(float)
    data["runtime_for_score"] = data["runtime_seconds"]
    data["privacy_residual_rate"] = data[["AdaFace_reid_rate", "ArcFace_reid_rate"]].mean(axis=1, skipna=True)
    data.loc[data["box_count"].eq(0), "privacy_residual_rate"] = 0.0
    data["privacy_score"] = (1.0 - data["privacy_residual_rate"]).clip(0.0, 1.0)
    data["lpips_utility_score"] = (1.0 - data["LPIPS"] / 0.05).clip(0.0, 1.0)
    data["utility_score"] = (0.5 * data["SSIM"] + 0.5 * data["lpips_utility_score"]).clip(0.0, 1.0)
    return data


def evaluate_face_region(base: pd.DataFrame, reuse: bool) -> pd.DataFrame:
    target = OUT / "01_advanced_face_region_metrics.csv"
    if reuse and target.exists():
        return pd.read_csv(target)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = Path("/tmp/practicum_work/group2_face_region.csv")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(checkpoint).to_dict("records") if checkpoint.exists() else []
    done = {(row["relative_path"], row["method"]) for row in rows}
    reviewed = pd.read_csv(REVIEWED)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().cuda()
    perceptual = lpips.LPIPS(net="alex").eval().cuda()
    landmarks = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cuda",
        face_detector="sfd",
        flip_input=False,
    )
    started = time.perf_counter()
    for image_index, item in enumerate(reviewed.itertuples(index=False), start=1):
        boxes = json.loads(item.reviewed_face_boxes_json)
        original = hardening.Image.open(RAW / item.relative_path).convert("RGB")
        original_crops = [hardening.image_crop(original, box) for box in boxes]
        original_embeddings = hardening.facenet_embeddings(original_crops, facenet)
        original_landmarks = []
        for crop in original_crops:
            detected = landmarks.get_landmarks_from_image(crop)
            original_landmarks.append(None if not detected else detected[0])
        for method in METHOD_DIRS:
            if (item.relative_path, method) in done:
                continue
            source = base[(base.relative_path == item.relative_path) & (base.method == method)].iloc[0]
            output = hardening.Image.open(ROOT / source.output_path).convert("RGB")
            if not boxes:
                rows.append({
                    "relative_path": item.relative_path,
                    "method": method,
                    "face_crop_count": 0,
                    "crop_ssim_mean": np.nan,
                    "crop_lpips_mean": np.nan,
                    "facenet_cosine_mean": np.nan,
                    "facenet_reid_rate_060": np.nan,
                    "landmark_pair_rate": np.nan,
                    "landmark_geometry_score": np.nan,
                    "background_preservation_score": hardening.background_score(original, output, boxes),
                    "face_region_utility_score": np.nan,
                    "metric_status": "no_face",
                })
                continue
            output_crops = [hardening.image_crop(output, box) for box in boxes]
            cosine = np.sum(original_embeddings * hardening.facenet_embeddings(output_crops, facenet), axis=1)
            crop_ssim = [
                hardening.structural_similarity(a, b, channel_axis=2, data_range=255)
                for a, b in zip(original_crops, output_crops, strict=True)
            ]
            crop_lpips = hardening.lpips_values(original_crops, output_crops, perceptual)
            errors = []
            detected_pairs = 0
            for source_landmarks, crop in zip(original_landmarks, output_crops, strict=True):
                detected = landmarks.get_landmarks_from_image(crop)
                target_landmarks = None if not detected else detected[0]
                if source_landmarks is not None and target_landmarks is not None:
                    detected_pairs += 1
                    errors.append(hardening.procrustes_error(source_landmarks, target_landmarks))
            pair_rate = detected_pairs / len(boxes)
            error_mean = float(np.mean(errors)) if errors else np.nan
            geometry = float(np.clip(1.0 - error_mean / 0.35, 0.0, 1.0)) if errors else 0.0
            lpips_utility = float(np.mean(np.clip(1.0 - crop_lpips / 0.50, 0.0, 1.0)))
            local_utility = 0.35 * np.mean(crop_ssim) + 0.25 * lpips_utility + 0.25 * geometry + 0.15 * pair_rate
            rows.append({
                "relative_path": item.relative_path,
                "method": method,
                "face_crop_count": len(boxes),
                "crop_ssim_mean": float(np.mean(crop_ssim)),
                "crop_lpips_mean": float(np.mean(crop_lpips)),
                "facenet_cosine_mean": float(np.mean(cosine)),
                "facenet_reid_rate_060": float(np.mean(cosine >= 0.60)),
                "landmark_pair_rate": pair_rate,
                "landmark_geometry_score": geometry,
                "background_preservation_score": hardening.background_score(original, output, boxes),
                "face_region_utility_score": local_utility,
                "metric_status": "ok",
            })
        if image_index % 10 == 0:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
            elapsed = time.perf_counter() - started
            print(f"face_region={image_index}/500 elapsed={elapsed/60:.1f}m eta={elapsed/image_index*(500-image_index)/60:.1f}m", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(target, index=False)
    checkpoint.unlink(missing_ok=True)
    return result


def score_candidates(base: pd.DataFrame, face_region: pd.DataFrame) -> pd.DataFrame:
    data = base.merge(face_region, on=["relative_path", "method"], how="left", validate="one_to_one")
    rates = data[["AdaFace_reid_rate", "ArcFace_reid_rate", "facenet_reid_rate_060"]]
    data["privacy_score_three_attacker"] = 1.0 - rates.mean(axis=1, skipna=True)
    data.loc[data.box_count.eq(0), "privacy_score_three_attacker"] = 1.0
    data["enhanced_success_score"] = data["success"]
    data["enhanced_utility_score"] = data["utility_score"]
    positive = data.box_count.gt(0)
    data.loc[positive, "enhanced_utility_score"] = (
        0.40 * data.loc[positive, "utility_score"]
        + 0.50 * data.loc[positive, "face_region_utility_score"]
        + 0.10 * data.loc[positive, "background_preservation_score"]
    )
    data["enhanced_utility_score"] = data["enhanced_utility_score"].fillna(data["utility_score"])
    data["runtime_score_constrained"] = np.exp(-data.runtime_for_score.fillna(150.0) / 5.0)
    data["runtime_score_high_compute"] = (
        1.0 - np.log1p(data.runtime_for_score.fillna(150.0)) / np.log1p(150.0)
    ).clip(0.0, 1.0)
    data["enhanced_balanced_score"] = (
        0.50 * data.privacy_score_three_attacker
        + 0.30 * data.enhanced_utility_score
        + 0.10 * data.runtime_score_constrained
        + 0.10 * data.enhanced_success_score
    )
    data["enhanced_high_compute_score"] = (
        0.50 * data.privacy_score_three_attacker
        + 0.35 * data.enhanced_utility_score
        + 0.05 * data.runtime_score_high_compute
        + 0.10 * data.enhanced_success_score
    )
    data.to_csv(OUT / "02_advanced_per_image_scores.csv", index=False)
    return data


def heldout_policy(all_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_csv(ROUTES)
    routes["policy_category"] = routes.apply(hardening.route_category, axis=1)
    image_rows = routes[["relative_path", "policy_category", "route_reason"]].copy()
    image_rows["participant"] = image_rows.relative_path.str.split("/").str[2]
    all_data = all_data.merge(image_rows[["relative_path", "policy_category"]], on="relative_path", how="left", validate="many_to_one")
    methods = sorted(all_data.method.unique())
    decisions = []
    splitter = GroupKFold(n_splits=5)
    for fold, (train_index, test_index) in enumerate(splitter.split(image_rows, groups=image_rows.participant), start=1):
        train_paths = set(image_rows.iloc[train_index].relative_path)
        train = all_data[all_data.relative_path.isin(train_paths)]
        aggregates = train.groupby(["policy_category", "method"]).agg(
            score=("enhanced_balanced_score", "mean"),
            privacy=("privacy_score_three_attacker", "mean"),
            success=("enhanced_success_score", "mean"),
        ).reset_index()
        global_values = train.groupby("method").agg(
            score=("enhanced_balanced_score", "mean"),
            privacy=("privacy_score_three_attacker", "mean"),
            success=("enhanced_success_score", "mean"),
        ).reset_index()
        for item in image_rows.iloc[test_index].itertuples(index=False):
            if item.route_reason == "confident_no_face":
                decisions.append({"fold": fold, "relative_path": item.relative_path, "policy_category": "no_face", "selected_method": "no_action_copy", "score": 1.0, "privacy": 1.0, "utility": 1.0, "selection_source": "verified_no_face"})
                continue
            candidates = aggregates[aggregates.policy_category.eq(item.policy_category)]
            eligible = candidates[(candidates.privacy >= PRIVACY_FLOOR) & (candidates.success >= 0.98)]
            source = "train_fold_category"
            if eligible.empty:
                eligible = global_values[(global_values.privacy >= PRIVACY_FLOOR) & (global_values.success >= 0.98)]
                source = "train_fold_global_fallback"
            if eligible.empty:
                selected = "solid_mask_black"
                source = "terminal_privacy_fallback"
            else:
                selected = str(eligible.sort_values("score", ascending=False).iloc[0].method)
            actual = all_data[(all_data.relative_path == item.relative_path) & (all_data.method == selected)].iloc[0]
            if actual.enhanced_success_score < 1.0:
                fallback = all_data[
                    (all_data.relative_path == item.relative_path)
                    & (all_data.method.isin(["layered_blur_downscale_noise", "solid_mask_black"]))
                    & (all_data.enhanced_success_score == 1.0)
                    & (all_data.privacy_score_three_attacker >= PRIVACY_FLOOR)
                ].sort_values("enhanced_balanced_score", ascending=False)
                if fallback.empty:
                    fallback = all_data[
                        (all_data.relative_path == item.relative_path)
                        & (all_data.method == "solid_mask_black")
                    ]
                actual = fallback.iloc[0]
                selected = str(actual.method)
                source = "runtime_failure_fallback"
            decisions.append({"fold": fold, "relative_path": item.relative_path, "policy_category": item.policy_category, "selected_method": selected, "score": actual.enhanced_balanced_score, "privacy": actual.privacy_score_three_attacker, "utility": actual.enhanced_utility_score, "selection_source": source})
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(OUT / "04_grouped_heldout_policy_routes.csv", index=False)
    summary = all_data.groupby("method").agg(
        n_images=("relative_path", "size"),
        success_rate=("enhanced_success_score", "mean"),
        privacy_score=("privacy_score_three_attacker", "mean"),
        utility_score=("enhanced_utility_score", "mean"),
        mean_runtime_seconds=("runtime_for_score", "mean"),
        balanced_score=("enhanced_balanced_score", "mean"),
        high_compute_score=("enhanced_high_compute_score", "mean"),
    ).reset_index()
    policy_row = pd.DataFrame([{
        "method": "grouped_heldout_policy_with_group2",
        "n_images": len(decision_frame),
        "success_rate": float((decision_frame.score > 0).mean()),
        "privacy_score": decision_frame.privacy.mean(),
        "utility_score": decision_frame.utility.mean(),
        "mean_runtime_seconds": np.nan,
        "balanced_score": decision_frame.score.mean(),
        "high_compute_score": np.nan,
    }])
    summary = pd.concat([summary, policy_row], ignore_index=True)
    summary.to_csv(OUT / "03_all_method_and_policy_summary.csv", index=False)
    return decision_frame, summary


def paired_statistics(decisions: pd.DataFrame, all_data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260712)
    rows = []
    for method in sorted(all_data.method.unique()):
        fixed = all_data[all_data.method.eq(method)][["relative_path", "enhanced_balanced_score"]]
        pair = decisions[["relative_path", "score"]].merge(fixed, on="relative_path", validate="one_to_one")
        difference = (pair.score - pair.enhanced_balanced_score).to_numpy()
        indices = rng.integers(0, len(difference), size=(10_000, len(difference)))
        low, high = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
        rows.append({
            "fixed_method": method,
            "policy_mean": pair.score.mean(),
            "fixed_mean": pair.enhanced_balanced_score.mean(),
            "mean_difference": difference.mean(),
            "ci_low": low,
            "ci_high": high,
            "policy_wins": int((difference > 1e-12).sum()),
            "fixed_wins": int((difference < -1e-12).sum()),
            "ties": int((np.abs(difference) <= 1e-12).sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "05_paired_policy_statistics.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-face-region", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    missing = [str(path) for path in METHOD_DIRS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing completed method directories: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.concat([load_base_rows(method, directory) for method, directory in METHOD_DIRS.items()], ignore_index=True)
    face_region = evaluate_face_region(base, args.reuse_face_region)
    advanced = score_candidates(base, face_region)
    existing = pd.read_csv(EXISTING)
    missing_local_utility = existing.enhanced_utility_score.isna() & existing.utility_score.notna()
    existing.loc[missing_local_utility, "enhanced_utility_score"] = existing.loc[missing_local_utility, "utility_score"]
    existing.loc[missing_local_utility, "enhanced_balanced_score"] = (
        0.50 * existing.loc[missing_local_utility, "privacy_score_three_attacker"]
        + 0.30 * existing.loc[missing_local_utility, "enhanced_utility_score"]
        + 0.10 * existing.loc[missing_local_utility, "runtime_score_constrained"]
        + 0.10 * existing.loc[missing_local_utility, "enhanced_success_score"]
    )
    existing.loc[missing_local_utility, "enhanced_high_compute_score"] = (
        0.50 * existing.loc[missing_local_utility, "privacy_score_three_attacker"]
        + 0.35 * existing.loc[missing_local_utility, "enhanced_utility_score"]
        + 0.05 * existing.loc[missing_local_utility, "runtime_score_high_compute"]
        + 0.10 * existing.loc[missing_local_utility, "enhanced_success_score"]
    )
    columns = [
        "relative_path", "method", "runtime_for_score", "privacy_score_three_attacker",
        "enhanced_success_score", "enhanced_utility_score", "enhanced_balanced_score",
        "enhanced_high_compute_score",
    ]
    combined = pd.concat([existing[columns], advanced[columns]], ignore_index=True)
    decisions, summary = heldout_policy(combined)
    statistics = paired_statistics(decisions, combined)
    distribution = decisions.groupby("selected_method").size().rename("image_count").reset_index()
    distribution.to_csv(OUT / "06_policy_method_distribution.csv", index=False)
    lines = [
        "# Group 2 Comparable Evaluation",
        "",
        "RiDDLE and FALCO were evaluated on the same reviewed 500-frame protocol and under the same three-attacker, face-region utility, runtime, success, held-out grouping, and privacy-floor rules used for the established methods.",
        "",
        "## Method and policy summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Paired policy comparisons",
        "",
        statistics.to_markdown(index=False),
        "",
        "## Policy distribution",
        "",
        distribution.to_markdown(index=False),
        "",
    ]
    (OUT / "07_group2_final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
