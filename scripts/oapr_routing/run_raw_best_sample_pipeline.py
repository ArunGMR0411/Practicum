#!/usr/bin/env python3
"""Run the current evidence-supported detector and anonymisation policy on raw samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.src.privacy_pipeline_app.pipeline_demo import apply_layered, apply_solid
from scripts.detection.run_face_detector_hardening_experiment import ImageRecord, nms_fusion
from scripts.detection.run_low_compute_detector_policy_experiment import (
    cluster_predictions,
    cluster_to_features,
)
from scripts.detection.run_sliced_and_rfdetr_detector_experiment import run_rfdetr
from src.data.subset_building import detect_text_score, resize_for_analysis
from src.detection.scrfd_detector import SCRFDDetector
from src.evaluation.detection_metrics import ScoredBox

SOURCES = [
    "rfdetr_medium_face_030",
    "yolo11s_widerface_1280",
    "scrfd_10g_current_640",
    "yolo8s_widerface_repo_640",
    "yolo11n_pose_widerface_640",
]
CHECKPOINT = PROJECT_ROOT / "data/models/face_detection_candidates/rfdetr_hf_cache/models--Herojayjay--RFDETR-Face-Detection/snapshots/597fcce941997900080ce8127b53a5d24e330225/rfdetr_medium_face.pth"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_exclusions() -> set[str]:
    excluded: set[str] = set()
    files = [
        "outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv",
        "outputs/01_protocol/thesis_manifests/final_face_detection_500.csv",
        "outputs/01_protocol/thesis_manifests/final_multimodal_250.csv",
        "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv",
    ]
    for name in files:
        path = PROJECT_ROOT / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8")):
            for key in ("relative_path", "image_id", "image_path"):
                value = row.get(key, "")
                if value.endswith((".webp", ".jpg", ".jpeg", ".png")):
                    excluded.add(value.replace("data/castle2024/raw/", ""))
    return excluded


def sample_sources(count: int, seed: int) -> list[Path]:
    raw = PROJECT_ROOT / "data/castle2024/raw"
    excluded = canonical_exclusions()
    candidates = sorted(
        path for path in raw.rglob("*")
        if path.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}
        and str(path.relative_to(raw)) not in excluded
    )
    return random.Random(seed).sample(candidates, count)


def condition_map(image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> tuple[dict[str, int], dict[str, object]]:
    preview = image.convert("RGB").resize((512, 288), Image.Resampling.BILINEAR)
    array = np.asarray(preview)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clutter = float((cv2.Canny(gray, 80, 160) > 0).mean())
    text_score = int(detect_text_score(cv2.cvtColor(resize_for_analysis(cv2.cvtColor(array, cv2.COLOR_RGB2BGR), analysis_width=192), cv2.COLOR_BGR2GRAY)))
    width, height = image.size
    scales = []
    edge = 0
    for x1, y1, x2, y2 in boxes:
        ratio = (y2 - y1) / max(height, 1)
        scales.append("very_small_or_distant_face" if ratio < .065 else "small_face" if ratio < .13 else "medium_face" if ratio < .25 else "large_face")
        if x1 <= .03 * width or y1 <= .03 * height or x2 >= .97 * width or y2 >= .97 * height:
            edge = 1
    if len(set(scales)) > 1:
        scale = "mixed_scale_face"
    else:
        scale = scales[0] if scales else "none"
    conditions = {
        "no_face": int(not boxes),
        "single_face": int(len(boxes) == 1),
        "multi_face": int(len(boxes) > 1),
        "small_face": int(scale == "small_face"),
        "medium_face": int(scale == "medium_face"),
        "large_face": int(scale == "large_face"),
        "mixed_scale_face": int(scale == "mixed_scale_face"),
        "very_small_or_distant_face": int(scale == "very_small_or_distant_face"),
        "edge_or_partial_face": edge,
        "profile_or_occluded_face": 0,
        "downward_egocentric_view": int(any(((y1 + y2) / 2 / max(height, 1)) >= .65 for x1, y1, x2, y2 in boxes)),
        "motion_blur_or_low_sharpness": int(sharpness < 300),
        "low_light_or_dim": int(brightness < 60),
        "high_clutter": int(clutter > .13),
        "outdoor_or_vehicle_scene": 0,
    }
    return conditions, {
        "sharpness_score": round(sharpness, 3),
        "brightness_mean": round(brightness, 3),
        "edge_density": round(clutter, 5),
        "text_signal_score": text_score,
        "text_risk_signal": bool(text_score >= 12),
        "face_scale_signal": scale,
    }


def train_reranker() -> object:
    records: list[ImageRecord] = []
    for protocol, manifest in [
        ("baseline_500", "outputs/01_protocol/annotations/face_detection/01_baseline_500/manifest.csv"),
        ("egocentric_stress_500", "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"),
    ]:
        data = pd.read_csv(PROJECT_ROOT / manifest)
        for row in data.to_dict("records"):
            boxes = tuple(
                (int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))
                for b in json.loads(row.get("reviewed_face_boxes_json", "[]"))
            )
            records.append(ImageRecord(protocol, row["relative_path"], row["relative_path"], PROJECT_ROOT / "data/castle2024/raw" / row["relative_path"], boxes, row))
    candidates = pd.read_csv(PROJECT_ROOT / "outputs/02_face_detection/11_detector_candidate_box_telemetry/detector_candidate_boxes.csv")
    predictions: dict[str, list[ScoredBox]] = {name: [] for name in SOURCES}
    for row in candidates.to_dict("records"):
        name = row["detector_name"]
        if name not in predictions:
            continue
        protocol = {
            "01_baseline_500": "baseline_500",
            "02_egocentric_stress_500": "egocentric_stress_500",
        }.get(row["protocol"], row["protocol"])
        scoped = f"{protocol}::{row['relative_path']}"
        predictions[name].append(ScoredBox(scoped, (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])), float(row["score"]), {"source_detector": name}))
    records_by_id = {record.scoped_id: record for record in records}
    clusters = cluster_predictions(predictions, SOURCES, records_by_id)
    scene = {}
    for record in records:
        scene[record.scoped_id] = {name: int(record.attributes.get(name, "0") in {"1", "True", "true", "yes"}) for name in []}
    X = np.asarray([cluster_to_features(c, records_by_id[c.image_id], SOURCES, scene, True) for c in clusters], dtype=float)
    y = np.asarray([c.label for c in clusters], dtype=int)
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight={0: 1.0, 1: 2.5}, max_iter=1000, solver="liblinear", random_state=42))
    clf.fit(X, y)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "sample_test")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = sample_sources(args.count, args.seed)
    records = [ImageRecord("unseen_raw_sample", str(path.relative_to(PROJECT_ROOT / "data/castle2024/raw")), str(path.relative_to(PROJECT_ROOT / "data/castle2024/raw")), path, (), {}) for path in selected]

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"RF-DETR checkpoint missing: {CHECKPOINT}")
    reranker = train_reranker()
    rfdetr, _ = run_rfdetr(records, str(CHECKPOINT), "rfdetr_medium_face_030", threshold=.30)
    scrfd = SCRFDDetector(model_path="/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx", input_size=(640, 640))
    scrfd_predictions: list[ScoredBox] = []
    rows: list[dict[str, object]] = []
    box_trace: list[dict[str, object]] = []
    for record in records:
        with Image.open(record.path).convert("RGB") as image:
            started = time.perf_counter()
            result = scrfd.detect(image)
            for detection in result.detections:
                scrfd_predictions.append(ScoredBox(record.scoped_id, tuple(map(int, detection.box)), float(detection.confidence), {"source_detector": "scrfd_10g_current_640"}))
    predictions = {"rfdetr_medium_face_030": rfdetr, "scrfd_10g_current_640": scrfd_predictions}
    fused = nms_fusion(predictions, ["rfdetr_medium_face_030", "scrfd_10g_current_640"], "fusion_rfdetr_scrfd10g", .50, .20, .08, .90)
    by_image: dict[str, list[ScoredBox]] = {record.scoped_id: [] for record in records}
    for item in fused:
        by_image[item.image_id].append(item)

    for index, (record, source) in enumerate(zip(records, selected, strict=True), start=1):
        sample_id = f"sample_{index:02d}_{source.stem}"
        with Image.open(source).convert("RGB") as image:
            candidates_for_image = by_image[record.scoped_id]
            raw_boxes = [item.box for item in candidates_for_image]
            conditions, signals = condition_map(image, raw_boxes)
            cluster_list = cluster_predictions(
                predictions,
                ["rfdetr_medium_face_030", "scrfd_10g_current_640"],
                {item.scoped_id: item for item in records},
            )
            cluster_list = [item for item in cluster_list if item.image_id == record.scoped_id]
            feature_rows = [cluster_to_features(c, record, SOURCES, {record.scoped_id: conditions}, False) for c in cluster_list]
            probabilities = reranker.predict_proba(np.asarray(feature_rows, dtype=float))[:, 1] if feature_rows else np.asarray([])
            selected_boxes = [cluster.box for cluster, probability in zip(cluster_list, probabilities, strict=True) if probability >= .425]
            safety_candidates = [item for item in candidates_for_image if item.score >= .70]
            if not selected_boxes and safety_candidates:
                selected_boxes = [item.box for item in safety_candidates]
                safety_action = "high_confidence_candidate_override"
            else:
                safety_action = "no_override"
            for box_index, (cluster, probability) in enumerate(zip(cluster_list, probabilities, strict=True), start=1):
                box_trace.append({
                    "sample_id": sample_id,
                    "box_index": box_index,
                    "box": json.dumps(cluster.box),
                    "fused_score": round(float(cluster.score), 6),
                    "detector_sources": "|".join(sorted(cluster.sources)),
                    "rfdetr_source_score": round(float(cluster.source_scores.get("rfdetr_medium_face_030", 0.0)), 6),
                    "scrfd_source_score": round(float(cluster.source_scores.get("scrfd_10g_current_640", 0.0)), 6),
                    "reranker_probability": round(float(probability), 6),
                    "reranker_threshold": 0.425,
                    "selected_before_safety_gate": bool(probability >= .425),
                    "selected_after_safety_gate": bool(cluster.box in selected_boxes),
                })
            if not selected_boxes:
                method, action = "no_action_copy", "verified_no_face_copy"
                output = args.output / "outputs" / f"{sample_id}.webp"
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output, format="WEBP", quality=95, method=6)
            elif len(selected_boxes) == 1:
                method, action = "solid_mask_black", "single_face_policy"
                output = args.output / "outputs" / f"{sample_id}.webp"
                output.parent.mkdir(parents=True, exist_ok=True)
                apply_solid(image, selected_boxes).save(output, format="WEBP", quality=95, method=6)
            else:
                method, action = "layered_blur_downscale_noise", "face_positive_fallback_policy"
                output = args.output / "outputs" / f"{sample_id}.webp"
                output.parent.mkdir(parents=True, exist_ok=True)
                apply_layered(image, selected_boxes).save(output, format="WEBP", quality=95, method=6)
        rows.append({
            "sample_id": sample_id, "source_relative_path": record.relative_path,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_width": image.width, "source_height": image.height,
            "stage_1_input": "raw_castle_image_outside_canonical_manifests",
            "stage_2_detector": "rfdetr_medium_face_030+scrfd_10g_current_640",
            "stage_2_detector_device": "cuda:0",
            "stage_2_fused_candidate_count": len(candidates_for_image),
            "stage_3_reranker": "cv_box_reranker_with_rfdetr_predicted_conditions",
            "stage_3_threshold": .425,
            "stage_3_selected_box_count": len(selected_boxes),
            "stage_3_safety_action": safety_action,
            "stage_4_condition_flags_json": json.dumps(conditions, sort_keys=True),
            "stage_4_telemetry_json": json.dumps(signals, sort_keys=True),
            "stage_5_multimodal_status": "unseen_sample_image_level_signal_only_not_reviewed_ground_truth",
            "stage_6_selected_method": method, "stage_6_action": action,
            "output_path": str(output.relative_to(PROJECT_ROOT)), "output_exists": output.exists(), "status": "ok",
        })
    write_csv(args.output / "routing_log.csv", rows)
    write_csv(args.output / "box_decision_trace.csv", box_trace)
    write_csv(args.output / "output_manifest.csv", [{k: row[k] for k in ("sample_id", "source_relative_path", "stage_6_selected_method", "output_path", "status")} for row in rows])
    (args.output / "README.md").write_text("\n".join([
        "# Unseen Raw CASTLE Sample Run", "",
        "This run uses five raw CASTLE images outside the canonical evaluation manifests.",
        "", "Pipeline: raw input → RF-DETR/SCRFD candidate fusion → condition-aware box reranker → safety gate → current anonymisation policy.",
        "", "RF-DETR and SCRFD ran on CUDA. The reranker was fitted from the retained reviewed detector evidence and applied to the unseen samples.",
        "", "The sample has no human ground truth, so no accuracy score is claimed. The text signal is a screening signal, not reviewed multimodal ground truth.", "",
    ]) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["sample_id", "stage_2_fused_candidate_count", "stage_3_selected_box_count", "stage_3_safety_action", "stage_6_selected_method"]].to_string(index=False))


if __name__ == "__main__":
    main()
