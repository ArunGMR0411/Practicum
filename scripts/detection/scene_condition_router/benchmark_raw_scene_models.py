#!/usr/bin/env python3
"""Benchmark raw-image scene-condition router methods.

This script evaluates only features available before face detection:

- deterministic image-quality, colour, and texture features;
- frozen visual embeddings from DINOv3 via timm;
- frozen visual embeddings from torchvision ResNet-50;
- synergies between frozen embeddings and handcrafted raw-image features.

It does not use reviewed face-box geometry because this router is the first
pipeline stage for arbitrary raw WebP/JPG/PNG inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, fbeta_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path("/tmp/practicum_router_runs/raw_feature_benchmark")
RAW_ROOT = ROOT / "data" / "castle2024" / "raw"
DINOV3_MODEL = "convnext_base.dinov3_lvd1689m"

EXCLUDED_LABELS = {"label_text_or_screen_risk_in_face_protocol"}
SUPPORTED_MIN_SUPPORT = 30


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    feature_set: str
    estimator: str


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Any) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Any, str]:
        row = self.df.iloc[idx]
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        return self.transform(image), row["relative_path"]


def handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        image_path = ROOT / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {image_path}")
        small = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        edges = cv2.Canny(gray, 80, 160)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hist_h = hist_h / max(hist_h.sum(), 1.0)
        hist_s = hist_s / max(hist_s.sum(), 1.0)
        hist_v = hist_v / max(hist_v.sum(), 1.0)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        rec: dict[str, float | str] = {
            "relative_path": row["relative_path"],
            "img_brightness_mean": float(gray.mean()),
            "img_brightness_std": float(gray.std()),
            "img_sharpness_laplacian_var": float(lap.var()),
            "img_edge_density": float((edges > 0).mean()),
            "img_saturation_mean": float(hsv[:, :, 1].mean()),
            "img_hue_mean": float(hsv[:, :, 0].mean()),
            "img_value_mean": float(hsv[:, :, 2].mean()),
            "img_lab_l_mean": float(lab[:, :, 0].mean()),
            "img_lab_a_mean": float(lab[:, :, 1].mean()),
            "img_lab_b_mean": float(lab[:, :, 2].mean()),
            "img_laplacian_abs_mean": float(np.abs(lap).mean()),
            "img_dark_pixel_ratio": float((gray < 45).mean()),
            "img_bright_pixel_ratio": float((gray > 220).mean()),
        }
        for i, value in enumerate(hist_h):
            rec[f"hist_h_{i:02d}"] = float(value)
        for i, value in enumerate(hist_s):
            rec[f"hist_s_{i:02d}"] = float(value)
        for i, value in enumerate(hist_v):
            rec[f"hist_v_{i:02d}"] = float(value)
        rows.append(rec)
    return pd.DataFrame(rows)


def extract_dinov3(df: pd.DataFrame, batch_size: int, device: torch.device) -> pd.DataFrame:
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model = timm.create_model(DINOV3_MODEL, pretrained=True, num_classes=0)
    model.eval().to(device)
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    return extract_embeddings(df, model, transform, batch_size, device, "dinov3")


def extract_resnet50(df: pd.DataFrame, batch_size: int, device: torch.device) -> pd.DataFrame:
    from torchvision import models, transforms

    weights = models.ResNet50_Weights.DEFAULT
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return extract_embeddings(df, model, transform, batch_size, device, "resnet50")


def extract_embeddings(
    df: pd.DataFrame, model: Any, transform: Any, batch_size: int, device: torch.device, prefix: str
) -> pd.DataFrame:
    loader = DataLoader(ImageDataset(df, transform), batch_size=batch_size, shuffle=False, num_workers=2)
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for images, rels in loader:
            feats = model(images.to(device))
            if isinstance(feats, (tuple, list)):
                feats = feats[-1]
            if feats.ndim > 2:
                feats = feats.mean(dim=tuple(range(2, feats.ndim)))
            feats = torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()
            for rel, vec in zip(rels, feats, strict=True):
                rec: dict[str, float | str] = {"relative_path": rel}
                rec.update({f"{prefix}_{i:04d}": float(v) for i, v in enumerate(vec)})
                rows.append(rec)
    return pd.DataFrame(rows)


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
        "random_forest": OneVsRestClassifier(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=400,
                            min_samples_leaf=3,
                            class_weight="balanced_subsample",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        ),
        "extra_trees": OneVsRestClassifier(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        ),
    }


def method_specs(feature_sets: dict[str, list[str]]) -> list[MethodSpec]:
    specs: list[MethodSpec] = []
    for feature_set in feature_sets:
        if feature_set in {"handcrafted_only"}:
            estimators = ["logistic_regression", "linear_svm", "random_forest", "extra_trees"]
        elif feature_set in {"dinov3_only", "resnet50_only"}:
            estimators = ["logistic_regression", "linear_svm"]
        elif feature_set in {"dinov3_handcrafted", "resnet50_handcrafted"}:
            estimators = ["logistic_regression", "linear_svm"]
        else:
            estimators = ["logistic_regression"]
        for estimator in estimators:
            specs.append(MethodSpec(f"{feature_set}__{estimator}", feature_set, estimator))
    return specs


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
    jaccard = jaccard_score(yt, yp, average="samples", zero_division=0)
    exact_match = float((yt == yp).all(axis=1).mean())
    score = 0.50 * macro_f2 + 0.25 * macro_f1 + 0.15 * micro_f2 + 0.10 * jaccard
    return {
        "oapr_scene_condition_score": score,
        "macro_f2_supported": macro_f2,
        "macro_f1_supported": macro_f1,
        "micro_f2_supported": micro_f2,
        "sample_jaccard_supported": jaccard,
        "exact_match_supported": exact_match,
    }


def per_label_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], label_support: pd.Series
) -> list[dict[str, Any]]:
    rows = []
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
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2,
                "route_eligible": bool(support >= SUPPORTED_MIN_SUPPORT and f1 >= 0.70 and recall >= 0.70),
                "included_in_primary_score": bool(label not in EXCLUDED_LABELS and support >= SUPPORTED_MIN_SUPPORT),
            }
        )
    return rows


def evaluate_methods(df: pd.DataFrame, feature_sets: dict[str, list[str]], random_state: int, folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [c for c in df.columns if c.startswith("label_")]
    label_support = df[label_cols].sum()
    groups = df["source_group"].fillna("unknown").astype(str).to_numpy()
    y = df[label_cols].astype(int).to_numpy()
    estimators = build_estimators(random_state)
    gkf = GroupKFold(n_splits=folds)
    summary_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for spec in method_specs(feature_sets):
        x = df[feature_sets[spec.feature_set]].apply(pd.to_numeric, errors="coerce").to_numpy()
        all_pred = np.zeros_like(y)
        fold_scores: list[dict[str, float]] = []
        start = time.time()
        for fold, (train_idx, test_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
            estimator = estimators[spec.estimator]
            estimator.fit(x[train_idx], y[train_idx])
            pred = estimator.predict(x[test_idx])
            all_pred[test_idx] = pred
            fold_metric = calculate_metrics(y[test_idx], pred, label_cols, label_support)
            fold_metric["fold"] = fold
            fold_scores.append(fold_metric)
        aggregate = calculate_metrics(y, all_pred, label_cols, label_support)
        route_rows = per_label_metrics(y, all_pred, label_cols, label_support)
        route_count = sum(1 for row in route_rows if row["route_eligible"])
        summary_rows.append(
            {
                "method_id": spec.method_id,
                "feature_set": spec.feature_set,
                "estimator": spec.estimator,
                "feature_count": len(feature_sets[spec.feature_set]),
                "folds": folds,
                "runtime_seconds": round(time.time() - start, 3),
                "route_eligible_label_count": route_count,
                **aggregate,
                "fold_scores_json": json.dumps(fold_scores),
            }
        )
        for row in route_rows:
            row.update({"method_id": spec.method_id, "feature_set": spec.feature_set, "estimator": spec.estimator})
            label_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(label_rows)


def write_summary(summary: pd.DataFrame, labels: pd.DataFrame, output_path: Path) -> None:
    ranked = summary.sort_values("oapr_scene_condition_score", ascending=False).copy()
    top = ranked.iloc[0]
    top_labels = labels[labels["method_id"] == top["method_id"]].sort_values("label")
    lines = [
        "# Raw Scene-Condition Router Model Benchmark",
        "",
        "This benchmark uses only raw-image-compatible signals available before face detection.",
        "",
        "## Primary Metric",
        "",
        "`OAPR Scene-Condition Score = 0.50*macro_F2 + 0.25*macro_F1 + 0.15*micro_F2 + 0.10*sample_Jaccard`",
        "",
        "Rationale: macro F2 is weighted highest because missing a privacy-relevant condition is worse than over-flagging it; macro averaging prevents multi-face/no-face dominant labels from hiding weaker categories.",
        "",
        "## Best Model",
        "",
        f"- Method: `{top['method_id']}`",
        f"- OAPR Scene-Condition Score: `{top['oapr_scene_condition_score']:.4f}`",
        f"- Macro F2: `{top['macro_f2_supported']:.4f}`",
        f"- Macro F1: `{top['macro_f1_supported']:.4f}`",
        f"- Micro F2: `{top['micro_f2_supported']:.4f}`",
        f"- Route-eligible labels: `{int(top['route_eligible_label_count'])}`",
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
        "## Best Model Per-Label Metrics",
        "",
        top_labels[
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
        "## Fallback Rule",
        "",
        "Images that do not cross a route-eligible label threshold should be assigned to `uncategorized_or_low_confidence` and sent to the safe global detector/anonymisation fallback.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    df = pd.read_csv(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timings: list[dict[str, Any]] = []

    start = time.time()
    hand = handcrafted_features(df)
    timings.append({"step": "handcrafted_features", "seconds": round(time.time() - start, 3)})

    start = time.time()
    dino = extract_dinov3(df, args.batch_size, device)
    timings.append({"step": "dinov3_embeddings", "seconds": round(time.time() - start, 3), "device": str(device)})

    start = time.time()
    resnet = extract_resnet50(df, args.batch_size, device)
    timings.append({"step": "resnet50_embeddings", "seconds": round(time.time() - start, 3), "device": str(device)})

    overlap = [c for c in hand.columns if c != "relative_path" and c in df.columns]
    base_df = df.drop(columns=overlap)
    merged = base_df.merge(hand, on="relative_path", how="left", validate="one_to_one")
    merged = merged.merge(dino, on="relative_path", how="left", validate="one_to_one")
    merged = merged.merge(resnet, on="relative_path", how="left", validate="one_to_one")

    handcrafted_cols = [c for c in hand.columns if c != "relative_path"]
    dinov3_cols = [c for c in dino.columns if c != "relative_path"]
    resnet_cols = [c for c in resnet.columns if c != "relative_path"]
    feature_sets = {
        "handcrafted_only": handcrafted_cols,
        "dinov3_only": dinov3_cols,
        "resnet50_only": resnet_cols,
        "dinov3_handcrafted": dinov3_cols + handcrafted_cols,
        "resnet50_handcrafted": resnet_cols + handcrafted_cols,
    }

    summary, per_label = evaluate_methods(merged, feature_sets, args.random_state, args.folds)
    summary = summary.sort_values("oapr_scene_condition_score", ascending=False)
    summary_path = args.output_dir / "raw_scene_model_benchmark.csv"
    per_label_path = args.output_dir / "raw_scene_model_per_label_metrics.csv"
    report_path = args.output_dir / "raw_scene_model_benchmark.md"
    timing_path = args.output_dir / "raw_scene_model_benchmark_runtime.csv"
    summary.to_csv(summary_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    pd.DataFrame(timings + [{"step": "total", "seconds": round(time.time() - start_all, 3)}]).to_csv(
        timing_path, index=False
    )
    write_summary(summary, per_label, report_path)
    print(f"wrote {summary_path} rows={len(summary)}")
    print(f"wrote {per_label_path} rows={len(per_label)}")
    print(f"wrote {report_path}")
    print(f"wrote {timing_path}")


if __name__ == "__main__":
    main()
