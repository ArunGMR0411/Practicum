#!/usr/bin/env python3
"""Materialise the final held-out anonymisation policy with artifact fallback."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/03_anonymisation/15_materialised_adaptive_policy"
MODEL_ID = "google/siglip2-base-patch16-224"
REVIEW = ROOT / "outputs/03_anonymisation/14_group2_comparison/08_group2_visual_review.csv"
BOXES = ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv"
BASE_ROUTES = ROOT / "outputs/03_anonymisation/14_group2_comparison/04_grouped_heldout_policy_routes.csv"
RIDDLE_MANIFEST = ROOT / "outputs/03_anonymisation/12_riddle/01_riddle_500_manifest.csv"
DETERMINISTIC_MANIFEST = ROOT / "outputs/03_anonymisation/02_deterministic_baselines/01_output_manifest.csv"
EXISTING_SCORES = ROOT / "outputs/03_anonymisation/11_policy_hardening/02_enhanced_per_image_policy_metrics.csv"
ADVANCED_SCORES = ROOT / "outputs/03_anonymisation/14_group2_comparison/02_advanced_per_image_scores.csv"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    return torch.device("cuda")


def pooled_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if getattr(value, "pooler_output", None) is not None:
        return value.pooler_output
    if getattr(value, "last_hidden_state", None) is not None:
        return value.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unsupported model output: {type(value)!r}")


def crop_image(path: Path, row: pd.Series) -> Image.Image:
    image = Image.open(path).convert("RGB")
    padding = max(row.x2 - row.x1, row.y2 - row.y1) * 0.45
    return image.crop((
        max(0, row.x1 - padding),
        max(0, row.y1 - padding),
        min(image.width, row.x2 + padding),
        min(image.height, row.y2 + padding),
    ))


def extract_features(
    paths: list[str],
    box_lookup: pd.DataFrame,
    processor,
    model,
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    rows = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for relative_path in batch_paths:
            box = box_lookup.loc[relative_path]
            images.append(crop_image(
                ROOT / "outputs/03_anonymisation/12_riddle/images" / relative_path,
                box,
            ))
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            features = pooled_tensor(model.get_image_features(**inputs))
            features = torch.nn.functional.normalize(features.float(), dim=1)
        rows.append(features.cpu().numpy())
    return np.concatenate(rows), time.perf_counter() - started


def estimator() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LinearSVC(C=0.01, class_weight="balanced")),
    ])


def validate_gate(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    predictions = np.zeros(len(labels), dtype=np.int64)
    folds = np.zeros(len(labels), dtype=np.int64)
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(features, labels, groups), start=1):
        model = estimator().fit(features[train], labels[train])
        predictions[test] = model.predict(features[test])
        folds[test] = fold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    summary = pd.DataFrame([{
        "model": "siglip2_linear_svm",
        "validation": "five_fold_participant_grouped",
        "sample_size": len(labels),
        "artifact_count": int(labels.sum()),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "f2_privacy_utility_weighted": fbeta_score(labels, predictions, beta=2, zero_division=0),
        "specificity": tn / (tn + fp),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "fallback_rate": predictions.mean(),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }])
    return summary, folds, predictions


def main() -> None:
    device = require_cuda()
    OUT.mkdir(parents=True, exist_ok=True)
    staging_images = OUT / "images_staging"
    final_images = OUT / "images"
    shutil.rmtree(staging_images, ignore_errors=True)
    staging_images.mkdir(parents=True)

    review = pd.read_csv(REVIEW)
    boxes = pd.read_csv(BOXES).groupby("image_id", sort=False).first()
    routes = pd.read_csv(BASE_ROUTES)
    riddle = pd.read_csv(RIDDLE_MANIFEST).set_index("relative_path")
    deterministic = pd.read_csv(DETERMINISTIC_MANIFEST).set_index(["relative_path", "method"])

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    feature_model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    review_paths = review.relative_path.tolist()
    review_features, review_runtime = extract_features(
        review_paths, boxes, processor, feature_model, device
    )
    labels = review.riddle_obvious_artifact.to_numpy(dtype=np.int64)
    groups = review.relative_path.str.split("/").str[2].to_numpy()
    validation, folds, out_of_fold_predictions = validate_gate(review_features, labels, groups)
    validation["review_feature_runtime_seconds"] = review_runtime
    validation.to_csv(OUT / "01_artifact_gate_validation.csv", index=False)

    review_predictions = estimator().fit(review_features, labels).predict(review_features)
    reviewed_rows = review[["sample_index", "relative_path", "riddle_obvious_artifact"]].copy()
    reviewed_rows["validation_fold"] = folds
    reviewed_rows["grouped_out_of_fold_prediction"] = out_of_fold_predictions
    reviewed_rows["full_fit_prediction"] = review_predictions

    candidate_paths = routes.loc[routes.selected_method.eq("riddle"), "relative_path"].tolist()
    paths_with_boxes = [path for path in candidate_paths if path in boxes.index]
    candidate_features, candidate_runtime = extract_features(
        paths_with_boxes, boxes, processor, feature_model, device
    )
    final_gate = estimator().fit(review_features, labels)
    candidate_prediction = final_gate.predict(candidate_features)
    candidate_score = final_gate.decision_function(candidate_features)
    prediction_lookup = dict(zip(paths_with_boxes, candidate_prediction, strict=True))
    score_lookup = dict(zip(paths_with_boxes, candidate_score, strict=True))
    per_candidate_gate_runtime = candidate_runtime / max(len(paths_with_boxes), 1)
    del feature_model
    torch.cuda.empty_cache()

    reviewed_rows.to_csv(OUT / "02_artifact_gate_review_predictions.csv", index=False)
    gate_rows = []
    for path in candidate_paths:
        has_box = path in boxes.index
        gate_rows.append({
            "relative_path": path,
            "has_operational_box": int(has_box),
            "artifact_decision_score": score_lookup.get(path, np.nan),
            "predicted_artifact": int(prediction_lookup.get(path, 1)),
            "gate_action": "fallback_layered" if prediction_lookup.get(path, 1) == 1 else "retain_riddle",
            "gate_runtime_seconds": per_candidate_gate_runtime if has_box else 0.0,
        })
    gate_frame = pd.DataFrame(gate_rows)
    gate_frame.to_csv(OUT / "03_artifact_gate_predictions.csv", index=False)
    gate_lookup = gate_frame.set_index("relative_path")

    existing = pd.read_csv(EXISTING_SCORES)
    advanced = pd.read_csv(ADVANCED_SCORES)
    score_columns = [
        "relative_path", "method", "privacy_score_three_attacker",
        "enhanced_utility_score", "enhanced_balanced_score",
    ]
    score_table = pd.concat([existing[score_columns], advanced[score_columns]], ignore_index=True)
    score_table = score_table.set_index(["relative_path", "method"])

    route_rows = []
    manifest_rows = []
    materialise_started = time.perf_counter()
    for route in routes.itertuples(index=False):
        attempted = route.selected_method
        gate_action = "not_required"
        gate_seconds = 0.0
        if attempted == "riddle":
            gate = gate_lookup.loc[route.relative_path]
            gate_action = gate.gate_action
            gate_seconds = float(gate.gate_runtime_seconds)
            final_method = "layered_blur_downscale_noise" if gate.predicted_artifact == 1 else "riddle"
        else:
            final_method = attempted

        if final_method == "no_action_copy":
            source = ROOT / "data/castle2024/raw" / route.relative_path
            component_runtime = 0.0
            privacy = utility = score = 1.0
            box_count = int(riddle.loc[route.relative_path].box_count)
        elif final_method == "riddle":
            item = riddle.loc[route.relative_path]
            source = ROOT / item.output_path
            component_runtime = float(item.runtime_seconds) + gate_seconds
            metric = score_table.loc[(route.relative_path, final_method)]
            privacy = float(metric.privacy_score_three_attacker)
            utility = float(metric.enhanced_utility_score)
            score = float(metric.enhanced_balanced_score)
            box_count = int(item.box_count)
        else:
            item = deterministic.loc[(route.relative_path, final_method)]
            source = ROOT / item.output_path
            component_runtime = float(item.runtime_seconds)
            if attempted == "riddle":
                component_runtime += float(riddle.loc[route.relative_path].runtime_seconds) + gate_seconds
            metric = score_table.loc[(route.relative_path, final_method)]
            privacy = float(metric.privacy_score_three_attacker)
            utility = float(metric.enhanced_utility_score)
            score = float(metric.enhanced_balanced_score)
            box_count = int(item.boxes_processed)

        destination = staging_images / route.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        relative_output = (final_images / route.relative_path).relative_to(ROOT).as_posix()
        route_rows.append({
            "relative_path": route.relative_path,
            "policy_category": route.policy_category,
            "attempted_method": attempted,
            "artifact_gate_action": gate_action,
            "final_method": final_method,
            "selection_source": route.selection_source,
            "privacy_score": privacy,
            "utility_score": utility,
            "balanced_score": score,
            "component_runtime_seconds": component_runtime,
            "output_path": relative_output,
            "status": "ok",
        })
        manifest_rows.append({
            "relative_path": route.relative_path,
            "method": "adaptive_artifact_gated_policy",
            "output_path": relative_output,
            "status": "ok",
            "box_count": box_count,
            "runtime_seconds": component_runtime,
        })

    materialisation_runtime = time.perf_counter() - materialise_started
    route_frame = pd.DataFrame(route_rows)
    manifest = pd.DataFrame(manifest_rows)
    staged_count = sum(1 for path in staging_images.rglob("*") if path.is_file())
    if staged_count != len(routes):
        raise RuntimeError(
            f"Refusing to replace final outputs: staged {staged_count}/{len(routes)} files"
        )
    shutil.rmtree(final_images, ignore_errors=True)
    staging_images.rename(final_images)
    route_frame.to_csv(OUT / "04_final_policy_routing_log.csv", index=False)
    manifest.to_csv(OUT / "05_final_policy_manifest.csv", index=False)
    distribution = route_frame.groupby(["attempted_method", "artifact_gate_action", "final_method"]).size().rename("image_count").reset_index()
    distribution.to_csv(OUT / "06_final_method_distribution.csv", index=False)
    runtime = pd.DataFrame([{
        "method": "adaptive_artifact_gated_policy",
        "input_frames": len(manifest),
        "successful_frames": int(manifest.status.eq("ok").sum()),
        "failed_frames": int(manifest.status.ne("ok").sum()),
        "artifact_gate_candidates": len(candidate_paths),
        "artifact_gate_fallbacks": int(gate_frame.predicted_artifact.sum()),
        "gate_total_runtime_seconds": candidate_runtime,
        "gate_mean_candidate_runtime_seconds": per_candidate_gate_runtime,
        "component_runtime_total_seconds": manifest.runtime_seconds.sum(),
        "component_runtime_mean_seconds": manifest.runtime_seconds.mean(),
        "materialisation_copy_runtime_seconds": materialisation_runtime,
        "device": "cuda",
    }])
    runtime.to_csv(OUT / "07_final_runtime_summary.csv", index=False)
    print(validation.to_string(index=False))
    print(distribution.to_string(index=False))
    print(runtime.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Materialise the final artifact-gated adaptive policy."
    )
    parser.parse_args()
    main()
