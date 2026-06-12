#!/usr/bin/env python3
"""Benchmark hybrid scene-condition routing methods.

This script compares the previous raw handcrafted router with hybrid feature
sets that add SigLIP2, DINOv3, open-vocabulary SigLIP prompt cues, and optional
first-pass YOLO telemetry. The detector telemetry is explicitly a second-stage
feature group: it is available only after a cheap first detector pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_raw_scene_models import (  # noqa: E402
    DINOV3_MODEL,
    EXCLUDED_LABELS,
    SUPPORTED_MIN_SUPPORT,
    handcrafted_features,
)


SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
YOLO_FIRST_PASS_MODEL = ROOT / "data" / "models" / "yolov8n-face-lindevs.pt"

OPEN_VOCAB_PROMPTS = {
    "low_light_or_dim": "a low light dim indoor egocentric image",
    "motion_blur_or_low_sharpness": "a blurry motion blurred low sharpness image",
    "high_clutter": "a cluttered crowded indoor scene with many objects",
    "multi_face": "an image with multiple visible people or faces",
    "single_face": "an image with one visible face",
    "no_face": "an egocentric image with no visible face",
    "small_or_distant_face": "an image with small distant faces",
    "large_face": "a close up large visible face",
    "profile_or_occluded_face": "a profile face or partially occluded face",
    "edge_or_partial_face": "a face cut off near the edge of the frame",
    "downward_egocentric_view": "a downward first person egocentric view",
    "outdoor_or_vehicle_scene": "an outdoor or vehicle scene",
    "text_or_screen_risk": "an image with visible text signs labels documents or a screen",
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    feature_set: str
    estimator: str


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, str]:
        row = self.df.iloc[idx]
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["relative_path"]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark; refusing CPU fallback.")
    return torch.device("cuda")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_df(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    parts = []
    per_protocol = max(1, sample_size // max(df["protocol"].nunique(), 1))
    for _, group in df.groupby("protocol", sort=True):
        take = min(per_protocol, len(group))
        parts.append(group.sample(n=take, random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) < sample_size:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if not remaining.empty:
            extra_idx = rng.choice(remaining.index.to_numpy(), size=min(sample_size - len(sampled), len(remaining)), replace=False)
            sampled = pd.concat([sampled, remaining.loc[extra_idx]], ignore_index=True)
    return sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def extract_dinov3_embeddings(df: pd.DataFrame, batch_size: int, device: torch.device) -> pd.DataFrame:
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0)
    model.eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    loader = DataLoader(ImagePathDataset(df, transform), batch_size=batch_size, shuffle=False, num_workers=2)
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for images, rels in loader:
            feats = model(images.to(device, non_blocking=True))
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=tuple(range(2, feats.ndim)))
            feats = torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()
            for rel, vec in zip(rels, feats, strict=True):
                rec: dict[str, float | str] = {"relative_path": rel}
                rec.update({f"dinov3_{i:04d}": float(value) for i, value in enumerate(vec)})
                rows.append(rec)
    del model
    cleanup_cuda()
    return pd.DataFrame(rows)


def extract_siglip2_features(df: pd.DataFrame, batch_size: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL)
    model = AutoModel.from_pretrained(SIGLIP2_MODEL)
    model.eval().to(device)

    prompts = list(OPEN_VOCAB_PROMPTS.values())
    prompt_keys = list(OPEN_VOCAB_PROMPTS.keys())
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        text_features = pooled_tensor(model.get_text_features(**text_inputs))
        text_features = torch.nn.functional.normalize(text_features.float(), dim=1)

    embed_rows: list[dict[str, float | str]] = []
    prompt_rows: list[dict[str, float | str]] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        images = [Image.open(ROOT / row["image_path"]).convert("RGB") for _, row in batch.iterrows()]
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            image_features = pooled_tensor(model.get_image_features(**image_inputs))
            image_features = torch.nn.functional.normalize(image_features.float(), dim=1)
            similarities = (image_features @ text_features.T).detach().cpu().numpy()
            image_np = image_features.detach().cpu().numpy()
        for rel, vec, sims in zip(batch["relative_path"], image_np, similarities, strict=True):
            embed: dict[str, float | str] = {"relative_path": rel}
            embed.update({f"siglip2_{i:04d}": float(value) for i, value in enumerate(vec)})
            embed_rows.append(embed)

            cue: dict[str, float | str] = {"relative_path": rel}
            cue.update({f"open_vocab_{key}": float(value) for key, value in zip(prompt_keys, sims, strict=True)})
            prompt_rows.append(cue)
    del model, processor, text_features
    cleanup_cuda()
    return pd.DataFrame(embed_rows), pd.DataFrame(prompt_rows)


def pooled_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state.mean(dim=1)
    raise TypeError(f"unsupported model feature output type: {type(output)!r}")


def extract_yolo_telemetry(
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    confidence_threshold: float,
    image_size: int,
) -> pd.DataFrame:
    from ultralytics import YOLO

    if not YOLO_FIRST_PASS_MODEL.exists():
        raise FileNotFoundError(f"missing YOLO first-pass model: {YOLO_FIRST_PASS_MODEL}")
    model = YOLO(str(YOLO_FIRST_PASS_MODEL))
    rows: list[dict[str, float | str]] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        images = [Image.open(ROOT / row["image_path"]).convert("RGB") for _, row in batch.iterrows()]
        results = model.predict(
            source=images,
            conf=confidence_threshold,
            iou=0.5,
            imgsz=image_size,
            device=str(device.index or 0),
            verbose=False,
        )
        for (_, row), result in zip(batch.iterrows(), results, strict=True):
            width = float(result.orig_shape[1])
            height = float(result.orig_shape[0])
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                rows.append(empty_detector_row(row["relative_path"]))
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
            scores = boxes.conf.detach().cpu().numpy().astype(float)
            areas = ((xyxy[:, 2] - xyxy[:, 0]).clip(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clip(min=0)) / max(width * height, 1.0)
            heights = (xyxy[:, 3] - xyxy[:, 1]).clip(min=0) / max(height, 1.0)
            widths = (xyxy[:, 2] - xyxy[:, 0]).clip(min=0) / max(width, 1.0)
            centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0 / max(width, 1.0)
            centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0 / max(height, 1.0)
            edge = ((xyxy[:, 0] <= 0.02 * width) | (xyxy[:, 1] <= 0.02 * height) | (xyxy[:, 2] >= 0.98 * width) | (xyxy[:, 3] >= 0.98 * height))
            rec: dict[str, float | str] = {
                "relative_path": row["relative_path"],
                "det_yolo_any": 1.0,
                "det_yolo_count": float(len(xyxy)),
                "det_yolo_score_max": float(scores.max()),
                "det_yolo_score_mean": float(scores.mean()),
                "det_yolo_score_min": float(scores.min()),
                "det_yolo_score_std": float(scores.std()) if len(scores) > 1 else 0.0,
                "det_yolo_area_max": float(areas.max()),
                "det_yolo_area_mean": float(areas.mean()),
                "det_yolo_area_min": float(areas.min()),
                "det_yolo_area_std": float(areas.std()) if len(areas) > 1 else 0.0,
                "det_yolo_height_max": float(heights.max()),
                "det_yolo_height_mean": float(heights.mean()),
                "det_yolo_width_max": float(widths.max()),
                "det_yolo_width_mean": float(widths.mean()),
                "det_yolo_edge_count": float(edge.sum()),
                "det_yolo_lower_frame_count": float((centers_y > 0.65).sum()),
                "det_yolo_upper_frame_count": float((centers_y < 0.35).sum()),
                "det_yolo_center_count": float(((centers_x > 0.33) & (centers_x < 0.67) & (centers_y > 0.33) & (centers_y < 0.67)).sum()),
                "det_yolo_small_count": float((areas < 0.0025).sum()),
                "det_yolo_large_count": float((areas > 0.04).sum()),
            }
            rows.append(rec)
    del model
    cleanup_cuda()
    return pd.DataFrame(rows)


def empty_detector_row(relative_path: str) -> dict[str, float | str]:
    return {
        "relative_path": relative_path,
        "det_yolo_any": 0.0,
        "det_yolo_count": 0.0,
        "det_yolo_score_max": 0.0,
        "det_yolo_score_mean": 0.0,
        "det_yolo_score_min": 0.0,
        "det_yolo_score_std": 0.0,
        "det_yolo_area_max": 0.0,
        "det_yolo_area_mean": 0.0,
        "det_yolo_area_min": 0.0,
        "det_yolo_area_std": 0.0,
        "det_yolo_height_max": 0.0,
        "det_yolo_height_mean": 0.0,
        "det_yolo_width_max": 0.0,
        "det_yolo_width_mean": 0.0,
        "det_yolo_edge_count": 0.0,
        "det_yolo_lower_frame_count": 0.0,
        "det_yolo_upper_frame_count": 0.0,
        "det_yolo_center_count": 0.0,
        "det_yolo_small_count": 0.0,
        "det_yolo_large_count": 0.0,
    }


def build_estimators(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": OneVsRestClassifier(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            solver="liblinear",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            )
        ),
        "linear_svm": OneVsRestClassifier(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scale", StandardScaler()),
                    ("clf", LinearSVC(class_weight="balanced", random_state=random_state, max_iter=5000)),
                ]
            )
        ),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], label_support: pd.Series) -> dict[str, float]:
    supported_idx = [
        i
        for i, label in enumerate(labels)
        if label_support[label] >= SUPPORTED_MIN_SUPPORT and label not in EXCLUDED_LABELS
    ]
    if not supported_idx:
        supported_idx = [i for i, label in enumerate(labels) if label not in EXCLUDED_LABELS]
    yt = y_true[:, supported_idx]
    yp = y_pred[:, supported_idx]
    macro_f2 = fbeta_score(yt, yp, beta=2, average="macro", zero_division=0)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    micro_f2 = fbeta_score(yt, yp, beta=2, average="micro", zero_division=0)
    intersection = np.logical_and(yt == 1, yp == 1).sum(axis=1)
    union = np.logical_or(yt == 1, yp == 1).sum(axis=1)
    jaccard = float(np.mean(np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)))
    exact_match = float((yt == yp).all(axis=1).mean())
    score = 0.50 * macro_f2 + 0.25 * macro_f1 + 0.15 * micro_f2 + 0.10 * jaccard
    return {
        "oapr_scene_condition_score": float(score),
        "macro_f2_supported": float(macro_f2),
        "macro_f1_supported": float(macro_f1),
        "micro_f2_supported": float(micro_f2),
        "sample_jaccard_supported": float(jaccard),
        "exact_match_supported": float(exact_match),
    }


def per_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], label_support: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        support = int(label_support[label])
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        f2 = fbeta_score(y_true[:, i], y_pred[:, i], beta=2, zero_division=0)
        rows.append(
            {
                "label": label.removeprefix("label_"),
                "support": support,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
                "route_eligible": bool(support >= SUPPORTED_MIN_SUPPORT and f1 >= 0.70 and recall >= 0.70),
                "included_in_primary_score": bool(label not in EXCLUDED_LABELS and support >= SUPPORTED_MIN_SUPPORT),
            }
        )
    return rows


def evaluate_methods(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    method_estimators: dict[str, list[str]],
    random_state: int,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_cols = [c for c in df.columns if c.startswith("label_")]
    label_support = df[label_cols].sum()
    y = df[label_cols].astype(int).to_numpy()
    groups = df["source_group"].fillna("unknown").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError(f"not enough source groups for group validation: {len(unique_groups)}")
    gkf = GroupKFold(n_splits=n_splits)
    estimators = build_estimators(random_state)
    summary_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for feature_set, feature_cols in feature_sets.items():
        x = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        for estimator_name in method_estimators[feature_set]:
            method_id = f"{feature_set}__{estimator_name}"
            all_pred = np.zeros_like(y)
            fold_scores: list[dict[str, Any]] = []
            started = time.time()
            for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
                estimator = estimators[estimator_name]
                estimator.fit(x[train_idx], y[train_idx])
                pred = estimator.predict(x[test_idx])
                all_pred[test_idx] = pred
                metric = calculate_metrics(y[test_idx], pred, label_cols, label_support)
                metric["fold"] = fold
                fold_scores.append(metric)
            aggregate = calculate_metrics(y, all_pred, label_cols, label_support)
            route_rows = per_label_metrics(y, all_pred, label_cols, label_support)
            route_count = sum(1 for row in route_rows if row["route_eligible"])
            summary_rows.append(
                {
                    "method_id": method_id,
                    "feature_set": feature_set,
                    "estimator": estimator_name,
                    "feature_count": len(feature_cols),
                    "folds": n_splits,
                    "runtime_seconds": round(time.time() - started, 3),
                    "route_eligible_label_count": route_count,
                    **aggregate,
                    "fold_scores_json": json.dumps(fold_scores),
                }
            )
            for row in route_rows:
                row.update({"method_id": method_id, "feature_set": feature_set, "estimator": estimator_name})
                label_rows.append(row)
            for idx, rel in enumerate(df["relative_path"]):
                rec: dict[str, Any] = {"relative_path": rel, "protocol": df.iloc[idx]["protocol"], "method_id": method_id}
                for label_idx, label in enumerate(label_cols):
                    name = label.removeprefix("label_")
                    rec[f"true_{name}"] = int(y[idx, label_idx])
                    rec[f"pred_{name}"] = int(all_pred[idx, label_idx])
                pred_rows.append(rec)
    return pd.DataFrame(summary_rows), pd.DataFrame(label_rows), pd.DataFrame(pred_rows)


def write_report(summary: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> None:
    ranked = summary.sort_values("oapr_scene_condition_score", ascending=False).copy()
    previous = ranked[ranked["method_id"] == "previous_handcrafted_only__logistic_regression"]
    best = ranked.iloc[0]
    previous_score = float(previous.iloc[0]["oapr_scene_condition_score"]) if not previous.empty else math.nan
    delta = float(best["oapr_scene_condition_score"]) - previous_score if not math.isnan(previous_score) else math.nan
    best_labels = labels[labels["method_id"] == best["method_id"]].sort_values("label")
    lines = [
        "# Hybrid Scene-Condition Router Benchmark",
        "",
        "This benchmark compares the previous handcrafted raw-image router with hybrid feature sets using SigLIP2, DINOv3, open-vocabulary prompt cues, and first-pass YOLO telemetry.",
        "",
        "## Evidence Boundary",
        "",
        "- `previous_handcrafted_only` is a true raw-image pre-detection feature set.",
        "- `hybrid_raw_siglip_dinov3_openvocab` is also pre-detection.",
        "- `hybrid_with_firstpass_detector_telemetry` is a two-stage router because detector telemetry is available only after a cheap first detector pass.",
        "- No reviewed ground-truth face-box geometry is used as an input feature.",
        "",
        "## Primary Metric",
        "",
        "`OAPR Scene-Condition Score = 0.50*macro_F2 + 0.25*macro_F1 + 0.15*micro_F2 + 0.10*sample_Jaccard`",
        "",
        "## Best Method",
        "",
        f"- Method: `{best['method_id']}`",
        f"- OAPR Scene-Condition Score: `{best['oapr_scene_condition_score']:.4f}`",
        f"- Delta vs previous handcrafted logistic: `{delta:.4f}`",
        f"- Macro F2: `{best['macro_f2_supported']:.4f}`",
        f"- Macro F1: `{best['macro_f1_supported']:.4f}`",
        f"- Micro F2: `{best['micro_f2_supported']:.4f}`",
        f"- Route-eligible labels: `{int(best['route_eligible_label_count'])}`",
        "",
        "## Best-to-Worst Ranking",
        "",
        ranked[
            [
                "method_id",
                "oapr_scene_condition_score",
                "macro_f2_supported",
                "macro_f1_supported",
                "micro_f2_supported",
                "sample_jaccard_supported",
                "exact_match_supported",
                "route_eligible_label_count",
                "runtime_seconds",
            ]
        ].to_markdown(index=False),
        "",
        "## Best Method Per-Label Metrics",
        "",
        best_labels[
            [
                "label",
                "support",
                "precision",
                "recall",
                "f1",
                "f2",
                "route_eligible",
                "included_in_primary_score",
            ]
        ].to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--yolo-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    args = parser.parse_args()

    device = require_cuda()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_df = pd.read_csv(args.dataset)
    df = sample_df(all_df, args.sample_size, args.random_state)
    timings: list[dict[str, Any]] = [{"step": "input_rows", "rows": len(df), "device": str(device)}]
    started_all = time.time()

    started = time.time()
    hand = handcrafted_features(df)
    timings.append({"step": "handcrafted_features", "seconds": round(time.time() - started, 3)})

    started = time.time()
    dino = extract_dinov3_embeddings(df, args.batch_size, device)
    timings.append({"step": "dinov3_embeddings", "seconds": round(time.time() - started, 3), "device": str(device)})

    started = time.time()
    siglip, open_vocab = extract_siglip2_features(df, args.batch_size, device)
    timings.append({"step": "siglip2_embeddings_and_open_vocab", "seconds": round(time.time() - started, 3), "device": str(device)})

    started = time.time()
    telemetry = extract_yolo_telemetry(
        df,
        batch_size=args.batch_size,
        device=device,
        confidence_threshold=args.yolo_confidence_threshold,
        image_size=args.yolo_image_size,
    )
    timings.append({"step": "firstpass_yolo_telemetry", "seconds": round(time.time() - started, 3), "device": str(device)})

    merged = df.copy()
    for feat in [hand, dino, siglip, open_vocab, telemetry]:
        overlap = [c for c in feat.columns if c != "relative_path" and c in merged.columns]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(feat, on="relative_path", how="left", validate="one_to_one")

    handcrafted_cols = [c for c in hand.columns if c != "relative_path"]
    dino_cols = [c for c in dino.columns if c != "relative_path"]
    siglip_cols = [c for c in siglip.columns if c != "relative_path"]
    open_vocab_cols = [c for c in open_vocab.columns if c != "relative_path"]
    telemetry_cols = [c for c in telemetry.columns if c != "relative_path"]
    feature_sets = {
        "previous_handcrafted_only": handcrafted_cols,
        "hybrid_raw_siglip_dinov3_openvocab": handcrafted_cols + dino_cols + siglip_cols + open_vocab_cols,
        "hybrid_with_firstpass_detector_telemetry": handcrafted_cols
        + dino_cols
        + siglip_cols
        + open_vocab_cols
        + telemetry_cols,
        "firstpass_detector_telemetry_only": telemetry_cols,
        "handcrafted_plus_firstpass_detector_telemetry": handcrafted_cols + telemetry_cols,
    }
    method_estimators = {
        "previous_handcrafted_only": ["logistic_regression"],
        "hybrid_raw_siglip_dinov3_openvocab": ["logistic_regression", "linear_svm"],
        "hybrid_with_firstpass_detector_telemetry": ["logistic_regression", "linear_svm"],
        "firstpass_detector_telemetry_only": ["logistic_regression"],
        "handcrafted_plus_firstpass_detector_telemetry": ["logistic_regression"],
    }

    summary, per_label, predictions = evaluate_methods(
        merged,
        feature_sets=feature_sets,
        method_estimators=method_estimators,
        random_state=args.random_state,
        folds=args.folds,
    )
    summary = summary.sort_values("oapr_scene_condition_score", ascending=False)
    per_label = per_label.sort_values(["method_id", "label"])
    timings.append({"step": "total", "seconds": round(time.time() - started_all, 3)})

    summary_path = args.output_dir / "hybrid_scene_model_benchmark.csv"
    label_path = args.output_dir / "hybrid_scene_model_per_label_metrics.csv"
    pred_path = args.output_dir / "hybrid_scene_model_predictions.csv"
    runtime_path = args.output_dir / "hybrid_scene_model_runtime.csv"
    feature_path = args.output_dir / "hybrid_scene_feature_inventory.csv"
    report_path = args.output_dir / "hybrid_scene_model_benchmark.md"

    summary.to_csv(summary_path, index=False)
    per_label.to_csv(label_path, index=False)
    predictions.to_csv(pred_path, index=False)
    pd.DataFrame(timings).to_csv(runtime_path, index=False)
    pd.DataFrame(
        [
            {"feature_set": name, "feature_count": len(cols), "feature_stage": "post_firstpass_detector" if "detector_telemetry" in name else "raw_image"}
            for name, cols in feature_sets.items()
        ]
    ).to_csv(feature_path, index=False)
    write_report(summary, per_label, report_path)

    print(json.dumps({"rows": len(df), "output_dir": str(args.output_dir), "best": summary.iloc[0].to_dict()}, indent=2))


if __name__ == "__main__":
    main()
