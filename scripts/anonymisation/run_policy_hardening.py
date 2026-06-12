#!/usr/bin/env python3
"""Harden anonymisation policy evidence using retained comparable outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import face_alignment
import lpips
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from skimage.metrics import structural_similarity
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/castle2024/raw"
MANIFEST = ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
BASE_METRICS = ROOT / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv"
ROUTES = ROOT / "outputs/06_end_to_end_thesis_validation/01_integrated_routing_log.csv"
OUT = ROOT / "outputs/03_anonymisation/11_policy_hardening"
TEMP = Path("/tmp/practicum_work/policy_hardening")
METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
]
STANDARD_METHODS = {
    "blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"
}
FACE_NET_THRESHOLD = 0.60
PRIVACY_FLOOR = 0.95


def image_crop(image: Image.Image, box: dict[str, int], size: int = 256) -> np.ndarray:
    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = min(image.width, int(box["x2"]))
    y2 = min(image.height, int(box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.asarray(
        image.crop((x1, y1, x2, y2)).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    ).copy()


def facenet_embeddings(crops: list[np.ndarray], model: InceptionResnetV1) -> np.ndarray:
    if not crops:
        return np.empty((0, 512), dtype=np.float32)
    tensor = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().cuda()
    tensor = torch.nn.functional.interpolate(tensor, size=(160, 160), mode="bilinear", align_corners=False)
    tensor = fixed_image_standardization(tensor)
    values = []
    with torch.inference_mode():
        for chunk in tensor.split(64):
            values.append(torch.nn.functional.normalize(model(chunk), dim=1).cpu())
    return torch.cat(values).numpy()


def lpips_values(original: list[np.ndarray], output: list[np.ndarray], model: lpips.LPIPS) -> np.ndarray:
    if not original:
        return np.empty(0, dtype=np.float32)
    a = torch.from_numpy(np.stack(original)).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1.0
    b = torch.from_numpy(np.stack(output)).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1.0
    values = []
    with torch.inference_mode():
        for x, y in zip(a.split(32), b.split(32), strict=True):
            values.append(model(x, y).flatten().cpu())
    return torch.cat(values).numpy()


def normalized_landmarks(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points -= points.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(points**2, axis=1)))
    return points / max(scale, 1e-8)


def procrustes_error(original: np.ndarray, output: np.ndarray) -> float:
    a = normalized_landmarks(original)
    b = normalized_landmarks(output)
    u, _, vt = np.linalg.svd(b.T @ a)
    rotation = u @ vt
    aligned = b @ rotation
    return float(np.sqrt(np.mean(np.sum((a - aligned) ** 2, axis=1))))


def background_score(original: Image.Image, output: Image.Image, boxes: list[dict[str, int]]) -> float:
    size = (640, 360)
    a = np.asarray(original.resize(size, Image.Resampling.BILINEAR)).astype(np.float32)
    b = np.asarray(output.resize(size, Image.Resampling.BILINEAR)).astype(np.float32)
    mask = np.ones((size[1], size[0]), dtype=bool)
    sx, sy = size[0] / original.width, size[1] / original.height
    for box in boxes:
        x1 = max(0, int(box["x1"] * sx))
        y1 = max(0, int(box["y1"] * sy))
        x2 = min(size[0], int(math.ceil(box["x2"] * sx)))
        y2 = min(size[1], int(math.ceil(box["y2"] * sy)))
        mask[y1:y2, x1:x2] = False
    if not mask.any():
        return float("nan")
    mae = np.abs(a - b).mean(axis=2)[mask].mean()
    return float(np.clip(1.0 - mae / 255.0, 0.0, 1.0))


def evaluate_face_region_metrics() -> pd.DataFrame:
    TEMP.mkdir(parents=True, exist_ok=True)
    part_file = TEMP / "local_metric_rows.csv"
    completed: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []
    if part_file.exists():
        existing = pd.read_csv(part_file)
        rows = existing.to_dict("records")
        completed = set(zip(existing["relative_path"], existing["method"], strict=False))

    manifest = pd.read_csv(MANIFEST)
    metrics = pd.read_csv(BASE_METRICS)
    lookup = metrics.set_index(["relative_path", "method"], drop=False)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().cuda()
    perceptual = lpips.LPIPS(net="alex").eval().cuda()
    landmarks = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device="cuda",
        face_detector="sfd",
        flip_input=False,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()

    for image_index, item in enumerate(manifest.itertuples(index=False), start=1):
        boxes = json.loads(item.reviewed_face_boxes_json)
        original_image = Image.open(RAW / item.relative_path).convert("RGB")
        original_crops = [image_crop(original_image, box) for box in boxes]
        original_embeddings = facenet_embeddings(original_crops, facenet)
        original_landmarks = []
        for crop in original_crops:
            detected = landmarks.get_landmarks_from_image(crop)
            original_landmarks.append(None if not detected else detected[0])

        for method in METHODS:
            key = (item.relative_path, method)
            if key in completed:
                continue
            base = lookup.loc[key]
            output_path = Path(str(base.output_path)) if pd.notna(base.output_path) else Path()
            if not output_path.is_file():
                rows.append({
                    "relative_path": item.relative_path,
                    "method": method,
                    "face_crop_count": len(boxes),
                    "metric_status": "missing_output",
                })
                continue
            output_image = Image.open(output_path).convert("RGB")
            output_crops = [image_crop(output_image, box) for box in boxes]
            if not boxes:
                rows.append({
                    "relative_path": item.relative_path,
                    "method": method,
                    "face_crop_count": 0,
                    "crop_ssim_mean": np.nan,
                    "crop_lpips_mean": np.nan,
                    "facenet_cosine_mean": np.nan,
                    "facenet_reid_rate_050": np.nan,
                    "facenet_reid_rate_060": np.nan,
                    "facenet_reid_rate_070": np.nan,
                    "landmark_pair_rate": np.nan,
                    "landmark_error_mean": np.nan,
                    "landmark_geometry_score": np.nan,
                    "background_preservation_score": background_score(original_image, output_image, boxes),
                    "face_region_utility_score": np.nan,
                    "metric_status": "no_face",
                })
                continue

            output_embeddings = facenet_embeddings(output_crops, facenet)
            cosine = np.sum(original_embeddings * output_embeddings, axis=1)
            crop_ssim = [
                structural_similarity(a, b, channel_axis=2, data_range=255)
                for a, b in zip(original_crops, output_crops, strict=True)
            ]
            crop_lpips = lpips_values(original_crops, output_crops, perceptual)
            errors = []
            detected_pairs = 0
            for source_landmarks, crop in zip(original_landmarks, output_crops, strict=True):
                detected = landmarks.get_landmarks_from_image(crop)
                target_landmarks = None if not detected else detected[0]
                if source_landmarks is not None and target_landmarks is not None:
                    detected_pairs += 1
                    errors.append(procrustes_error(source_landmarks, target_landmarks))
            pair_rate = detected_pairs / len(boxes)
            error_mean = float(np.mean(errors)) if errors else np.nan
            geometry = float(np.clip(1.0 - error_mean / 0.35, 0.0, 1.0)) if errors else 0.0
            lpips_utility = float(np.mean(np.clip(1.0 - crop_lpips / 0.50, 0.0, 1.0)))
            local_utility = (
                0.35 * float(np.mean(crop_ssim))
                + 0.25 * lpips_utility
                + 0.25 * geometry
                + 0.15 * pair_rate
            )
            rows.append({
                "relative_path": item.relative_path,
                "method": method,
                "face_crop_count": len(boxes),
                "crop_ssim_mean": float(np.mean(crop_ssim)),
                "crop_lpips_mean": float(np.mean(crop_lpips)),
                "facenet_cosine_mean": float(np.mean(cosine)),
                "facenet_reid_rate_050": float(np.mean(cosine >= 0.50)),
                "facenet_reid_rate_060": float(np.mean(cosine >= FACE_NET_THRESHOLD)),
                "facenet_reid_rate_070": float(np.mean(cosine >= 0.70)),
                "landmark_pair_rate": pair_rate,
                "landmark_error_mean": error_mean,
                "landmark_geometry_score": geometry,
                "background_preservation_score": background_score(original_image, output_image, boxes),
                "face_region_utility_score": local_utility,
                "metric_status": "ok",
            })

        if image_index % 10 == 0:
            pd.DataFrame(rows).to_csv(part_file, index=False)
            elapsed = time.perf_counter() - started
            eta = elapsed / image_index * (len(manifest) - image_index)
            print(f"metrics {image_index}/{len(manifest)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "01_face_region_and_third_attacker_metrics.csv", index=False)
    return result


def score_table(face_region: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_METRICS)
    data = base.merge(face_region, on=["relative_path", "method"], how="left", validate="one_to_one")
    data["third_attacker_reid_rate"] = data["facenet_reid_rate_060"]
    rates = data[["AdaFace_reid_rate", "ArcFace_reid_rate", "third_attacker_reid_rate"]]
    data["privacy_score_three_attacker"] = 1.0 - rates.mean(axis=1, skipna=True)
    face_positive = data["box_count"].fillna(0).gt(0)
    data.loc[~face_positive, "privacy_score_three_attacker"] = 1.0
    data["enhanced_success_score"] = data["metric_status"].isin(["ok", "no_face"]).astype(float)
    data["enhanced_utility_score"] = data["utility_score"]
    data.loc[face_positive, "enhanced_utility_score"] = (
        0.40 * data.loc[face_positive, "utility_score"]
        + 0.50 * data.loc[face_positive, "face_region_utility_score"]
        + 0.10 * data.loc[face_positive, "background_preservation_score"]
    )
    missing_output = data["enhanced_success_score"].eq(0)
    data.loc[missing_output, ["privacy_score_three_attacker", "enhanced_utility_score"]] = 0.0
    runtime = data["runtime_for_score"].fillna(data["runtime_seconds"])
    data["runtime_score_constrained"] = np.exp(-runtime.fillna(150.0) / 5.0)
    data["runtime_score_high_compute"] = 1.0 - np.log1p(runtime.fillna(150.0)) / np.log1p(150.0)
    data["runtime_score_high_compute"] = data["runtime_score_high_compute"].clip(0.0, 1.0)
    data["enhanced_balanced_score"] = (
        0.50 * data["privacy_score_three_attacker"]
        + 0.30 * data["enhanced_utility_score"]
        + 0.10 * data["runtime_score_constrained"]
        + 0.10 * data["enhanced_success_score"]
    )
    data["enhanced_high_compute_score"] = (
        0.50 * data["privacy_score_three_attacker"]
        + 0.35 * data["enhanced_utility_score"]
        + 0.05 * data["runtime_score_high_compute"]
        + 0.10 * data["enhanced_success_score"]
    )
    data.to_csv(OUT / "02_enhanced_per_image_policy_metrics.csv", index=False)
    summary = data.groupby("method", sort=False).agg(
        n_images=("relative_path", "size"),
        n_success=("enhanced_success_score", "sum"),
        privacy_score_three_attacker=("privacy_score_three_attacker", "mean"),
        enhanced_utility_score=("enhanced_utility_score", "mean"),
        face_region_utility_score=("face_region_utility_score", "mean"),
        landmark_pair_rate=("landmark_pair_rate", "mean"),
        background_preservation_score=("background_preservation_score", "mean"),
        runtime_seconds=("runtime_for_score", "mean"),
        constrained_score=("enhanced_balanced_score", "mean"),
        high_compute_score=("enhanced_high_compute_score", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "03_enhanced_method_summary.csv", index=False)
    return data


def route_category(row: pd.Series) -> str:
    if row.get("route_reason") == "confident_no_face":
        return "no_face"
    priority = [
        "single_face", "very_small_or_distant_face", "large_face",
        "motion_blur_or_low_sharpness", "small_face", "medium_face",
        "mixed_scale_face", "edge_or_partial_face", "profile_or_occluded_face",
        "low_light_or_dim", "high_clutter", "multi_face", "downward_egocentric_view",
    ]
    for category in priority:
        if int(row.get(f"pred_{category}", 0)) == 1:
            return category
    return "multi_face"


def heldout_policy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_csv(ROUTES)
    routes["policy_category"] = routes.apply(route_category, axis=1)
    image_rows = routes[["relative_path", "policy_category", "route_reason"]].copy()
    image_rows["participant"] = image_rows["relative_path"].str.split("/").str[2]
    groups = image_rows["participant"].to_numpy()
    splitter = GroupKFold(n_splits=5)
    decisions = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(image_rows, groups=groups), start=1):
        train_paths = set(image_rows.iloc[train_idx].relative_path)
        test = image_rows.iloc[test_idx]
        train = data[data.relative_path.isin(train_paths) & data.method.isin(STANDARD_METHODS)]
        category_scores = train.groupby(["policy_category", "method"], dropna=False)["enhanced_balanced_score"].mean() if "policy_category" in train else None
        if category_scores is None:
            train = train.merge(image_rows[["relative_path", "policy_category"]], on="relative_path", how="left")
            category_scores = train.groupby(["policy_category", "method"])["enhanced_balanced_score"].mean()
        global_scores = train.groupby("method")["enhanced_balanced_score"].mean()
        for item in test.itertuples(index=False):
            if item.policy_category == "no_face" and item.route_reason == "confident_no_face":
                decisions.append({"fold": fold, "relative_path": item.relative_path, "policy_category": item.policy_category, "selected_method": "no_action_copy", "selection_source": "verified_no_face", "score": 1.0, "privacy": 1.0, "utility": 1.0})
                continue
            try:
                values = category_scores.loc[item.policy_category].dropna()
                if values.empty:
                    raise KeyError(item.policy_category)
                selected = str(values.idxmax())
                source = "train_fold_category"
            except KeyError:
                selected = str(global_scores.idxmax())
                source = "train_fold_global_fallback"
            actual = data[(data.relative_path == item.relative_path) & (data.method == selected)]
            score = float(actual.enhanced_balanced_score.iloc[0]) if not actual.empty else 0.0
            privacy = float(actual.privacy_score_three_attacker.iloc[0]) if not actual.empty else 0.0
            utility = float(actual.enhanced_utility_score.iloc[0]) if not actual.empty else 0.0
            decisions.append({"fold": fold, "relative_path": item.relative_path, "policy_category": item.policy_category, "selected_method": selected, "selection_source": source, "score": score, "privacy": privacy, "utility": utility})
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(OUT / "04_grouped_heldout_policy_routes.csv", index=False)
    summary = pd.DataFrame([{
        "policy": "grouped_heldout_category_policy",
        "n_images": len(decision_frame),
        "mean_score": decision_frame.score.mean(),
        "n_failures": int((decision_frame.score <= 0).sum()),
    }])
    return decision_frame, summary


def pareto_and_cascade(data: pd.DataFrame, heldout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_csv(ROUTES)[["relative_path", "route_reason", "method"]]
    rows = []
    cascade_order = ["pixelate", "blur", "layered_blur_downscale_noise", "diffusion_low_step", "nullface"]
    for relative_path, group in data.groupby("relative_path", sort=False):
        route_reason = routes.loc[routes.relative_path.eq(relative_path), "route_reason"].iloc[0]
        if route_reason == "confident_no_face":
            rows.append({"relative_path": relative_path, "policy": "current_condition_policy", "selected_method": "no_action_copy", "score": 1.0, "privacy": 1.0, "utility": 1.0, "selection_reason": "verified_no_face"})
            rows.append({"relative_path": relative_path, "policy": "privacy_constrained_pareto", "selected_method": "no_action_copy", "score": 1.0, "privacy": 1.0, "utility": 1.0, "selection_reason": "verified_no_face"})
            rows.append({"relative_path": relative_path, "policy": "privacy_verifying_cascade", "selected_method": "no_action_copy", "score": 1.0, "privacy": 1.0, "utility": 1.0, "selection_reason": "verified_no_face"})
            continue
        current_method = routes.loc[routes.relative_path.eq(relative_path), "method"].iloc[0]
        current = group[group.method.eq(current_method)]
        if current.empty:
            current = group[group.method.eq("layered_blur_downscale_noise")]
        current = current.iloc[0]
        rows.append({"relative_path": relative_path, "policy": "current_condition_policy", "selected_method": current.method, "score": current.enhanced_balanced_score, "privacy": current.privacy_score_three_attacker, "utility": current.enhanced_utility_score, "selection_reason": "existing_integrated_route"})
        valid = group[(group.enhanced_success_score.eq(1)) & group.privacy_score_three_attacker.ge(PRIVACY_FLOOR)].copy()
        quality = valid[valid.enhanced_utility_score.ge(0.55)]
        candidates = quality if not quality.empty else valid
        if candidates.empty:
            candidates = group[group.method.eq("solid_mask_black")]
        best = candidates.sort_values(["enhanced_utility_score", "runtime_score_high_compute"], ascending=False).iloc[0]
        rows.append({"relative_path": relative_path, "policy": "privacy_constrained_pareto", "selected_method": best.method, "score": best.enhanced_balanced_score, "privacy": best.privacy_score_three_attacker, "utility": best.enhanced_utility_score, "selection_reason": "maximum_utility_after_privacy_quality_gates"})

        chosen = None
        for method in cascade_order:
            item = group[group.method.eq(method)]
            if item.empty:
                continue
            item = item.iloc[0]
            if item.enhanced_success_score == 1 and item.privacy_score_three_attacker >= PRIVACY_FLOOR and item.enhanced_utility_score >= 0.55:
                chosen = item
                break
        if chosen is None:
            chosen = group[group.method.eq("solid_mask_black")].iloc[0]
            reason = "terminal_solid_mask_fallback"
        else:
            reason = "first_candidate_passing_privacy_and_quality"
        rows.append({"relative_path": relative_path, "policy": "privacy_verifying_cascade", "selected_method": chosen.method, "score": chosen.enhanced_balanced_score, "privacy": chosen.privacy_score_three_attacker, "utility": chosen.enhanced_utility_score, "selection_reason": reason})

    policy_rows = pd.DataFrame(rows)
    heldout_rows = heldout.rename(columns={"selected_method": "selected_method", "score": "score"}).copy()
    heldout_rows["policy"] = "grouped_heldout_category_policy"
    heldout_rows["selection_reason"] = heldout_rows["selection_source"]
    policy_rows = pd.concat([policy_rows, heldout_rows[policy_rows.columns]], ignore_index=True)
    policy_rows.to_csv(OUT / "05_final_policy_routes.csv", index=False)
    summary = policy_rows.groupby("policy").agg(
        n_images=("relative_path", "size"),
        mean_score=("score", "mean"),
        privacy_score=("privacy", "mean"),
        utility_score=("utility", "mean"),
        n_methods=("selected_method", "nunique"),
    ).reset_index()
    distributions = policy_rows.groupby(["policy", "selected_method"]).size().rename("image_count").reset_index()
    summary = summary.merge(
        distributions.groupby("policy").apply(
            lambda x: json.dumps(dict(zip(x.selected_method, x.image_count, strict=False)), sort_keys=True),
            include_groups=False,
        ).rename("method_distribution"),
        on="policy",
    )
    summary.to_csv(OUT / "06_final_policy_summary.csv", index=False)
    return policy_rows, summary


def paired_statistics(policy_rows: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260712)
    rows = []
    fixed = data.groupby("method")["enhanced_balanced_score"].mean()
    for policy, group in policy_rows.groupby("policy"):
        if len(group) != 500:
            continue
        for method in METHODS:
            comparator = data[data.method.eq(method)][["relative_path", "enhanced_balanced_score"]].rename(columns={"enhanced_balanced_score": "fixed_score"})
            pair = group[["relative_path", "score"]].merge(comparator, on="relative_path", how="left")
            difference = (pair.score - pair.fixed_score.fillna(0.0)).to_numpy()
            samples = rng.integers(0, len(difference), size=(10_000, len(difference)))
            means = difference[samples].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            rows.append({
                "policy": policy,
                "fixed_method": method,
                "policy_mean": group.score.mean(),
                "fixed_mean": float(fixed[method]),
                "mean_difference": difference.mean(),
                "ci_low": low,
                "ci_high": high,
                "policy_win_count": int((difference > 1e-12).sum()),
                "fixed_win_count": int((difference < -1e-12).sum()),
                "tie_count": int((np.abs(difference) <= 1e-12).sum()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "07_paired_policy_statistics.csv", index=False)
    return result


def write_summary(methods: pd.DataFrame, policies: pd.DataFrame, statistics: pd.DataFrame) -> None:
    lines = [
        "# Anonymisation Policy Hardening",
        "",
        "All model inference used CUDA. Policy evaluation uses the manually reviewed 500-image egocentric-stress protocol.",
        "",
        "## Enhanced method metrics",
        "",
        methods.to_markdown(index=False),
        "",
        "## Policy comparison",
        "",
        policies.to_markdown(index=False),
        "",
        "## Statistical comparison",
        "",
        statistics.to_markdown(index=False),
        "",
        "face-region utility combines crop SSIM, crop LPIPS, landmark geometry, landmark detectability, and background preservation. The independent privacy attacker is FaceNet/VGGFace2 at cosine threshold 0.60. Results apply to the retained output set, declared thresholds, and reviewed 500-image protocol.",
        "",
    ]
    (OUT / "08_policy_hardening_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-metrics", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse_metrics and (OUT / "01_face_region_and_third_attacker_metrics.csv").exists():
        face_region = pd.read_csv(OUT / "01_face_region_and_third_attacker_metrics.csv")
    else:
        face_region = evaluate_face_region_metrics()
    data = score_table(face_region)
    routes = pd.read_csv(ROUTES)
    routes["policy_category"] = routes.apply(route_category, axis=1)
    data = data.merge(routes[["relative_path", "policy_category"]], on="relative_path", how="left", validate="many_to_one")
    heldout_routes, _ = heldout_policy(data)
    policy_rows, policy_summary = pareto_and_cascade(data, heldout_routes)
    stats = paired_statistics(policy_rows, data)
    method_summary = pd.read_csv(OUT / "03_enhanced_method_summary.csv")
    write_summary(method_summary, policy_summary, stats)
    shutil.rmtree(TEMP, ignore_errors=True)
    print(policy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
