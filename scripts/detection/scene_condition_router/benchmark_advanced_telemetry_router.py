#!/usr/bin/env python3
"""Benchmark advanced detector-telemetry scene-condition routing.

This follows the strongest finding from the hybrid benchmark: direct first-pass
detector behaviour is more predictive than raw embeddings for face-condition
routing. It adds multi-scale YOLO, SCRFD, and detector-disagreement features.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
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
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_raw_scene_models import EXCLUDED_LABELS, SUPPORTED_MIN_SUPPORT, handcrafted_features  # noqa: E402
from benchmark_hybrid_scene_router import extract_yolo_telemetry, require_cuda, sample_df  # noqa: E402

YOLO_FIRST_PASS_MODEL = ROOT / "data" / "models" / "yolov8n-face-lindevs.pt"
SCRFD_MODEL = Path("/home/arun-gmr/.insightface/models/buffalo_l/det_10g.onnx")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rename_feature_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "relative_path"})
    return out


def box_telemetry(relative_path: str, width: float, height: float, xyxy: np.ndarray, scores: np.ndarray, prefix: str) -> dict[str, float | str]:
    if xyxy.size == 0:
        return {
            "relative_path": relative_path,
            f"{prefix}_any": 0.0,
            f"{prefix}_count": 0.0,
            f"{prefix}_score_max": 0.0,
            f"{prefix}_score_mean": 0.0,
            f"{prefix}_score_min": 0.0,
            f"{prefix}_score_std": 0.0,
            f"{prefix}_area_max": 0.0,
            f"{prefix}_area_mean": 0.0,
            f"{prefix}_area_min": 0.0,
            f"{prefix}_area_std": 0.0,
            f"{prefix}_height_max": 0.0,
            f"{prefix}_height_mean": 0.0,
            f"{prefix}_width_max": 0.0,
            f"{prefix}_width_mean": 0.0,
            f"{prefix}_edge_count": 0.0,
            f"{prefix}_lower_frame_count": 0.0,
            f"{prefix}_upper_frame_count": 0.0,
            f"{prefix}_center_count": 0.0,
            f"{prefix}_small_count": 0.0,
            f"{prefix}_large_count": 0.0,
        }
    areas = ((xyxy[:, 2] - xyxy[:, 0]).clip(min=0) * (xyxy[:, 3] - xyxy[:, 1]).clip(min=0)) / max(
        width * height, 1.0
    )
    heights = (xyxy[:, 3] - xyxy[:, 1]).clip(min=0) / max(height, 1.0)
    widths = (xyxy[:, 2] - xyxy[:, 0]).clip(min=0) / max(width, 1.0)
    centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0 / max(width, 1.0)
    centers_y = (xyxy[:, 1] + xyxy[:, 3]) / 2.0 / max(height, 1.0)
    edge = (
        (xyxy[:, 0] <= 0.02 * width)
        | (xyxy[:, 1] <= 0.02 * height)
        | (xyxy[:, 2] >= 0.98 * width)
        | (xyxy[:, 3] >= 0.98 * height)
    )
    return {
        "relative_path": relative_path,
        f"{prefix}_any": 1.0,
        f"{prefix}_count": float(len(xyxy)),
        f"{prefix}_score_max": float(scores.max()) if len(scores) else 0.0,
        f"{prefix}_score_mean": float(scores.mean()) if len(scores) else 0.0,
        f"{prefix}_score_min": float(scores.min()) if len(scores) else 0.0,
        f"{prefix}_score_std": float(scores.std()) if len(scores) > 1 else 0.0,
        f"{prefix}_area_max": float(areas.max()),
        f"{prefix}_area_mean": float(areas.mean()),
        f"{prefix}_area_min": float(areas.min()),
        f"{prefix}_area_std": float(areas.std()) if len(areas) > 1 else 0.0,
        f"{prefix}_height_max": float(heights.max()),
        f"{prefix}_height_mean": float(heights.mean()),
        f"{prefix}_width_max": float(widths.max()),
        f"{prefix}_width_mean": float(widths.mean()),
        f"{prefix}_edge_count": float(edge.sum()),
        f"{prefix}_lower_frame_count": float((centers_y > 0.65).sum()),
        f"{prefix}_upper_frame_count": float((centers_y < 0.35).sum()),
        f"{prefix}_center_count": float(((centers_x > 0.33) & (centers_x < 0.67) & (centers_y > 0.33) & (centers_y < 0.67)).sum()),
        f"{prefix}_small_count": float((areas < 0.0025).sum()),
        f"{prefix}_large_count": float((areas > 0.04).sum()),
    }


def extract_scrfd_telemetry(
    df: pd.DataFrame,
    confidence_threshold: float,
    input_size: int,
) -> pd.DataFrame:
    import onnxruntime as ort
    from insightface.model_zoo import get_model

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(f"SCRFD CUDAExecutionProvider unavailable; providers={providers}")
    if not SCRFD_MODEL.exists():
        raise FileNotFoundError(f"missing SCRFD model: {SCRFD_MODEL}")
    detector = get_model(str(SCRFD_MODEL), providers=["CUDAExecutionProvider"])
    detector.prepare(ctx_id=0, input_size=(input_size, input_size))
    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        width, height = image.size
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces, _ = detector.detect(bgr, input_size=(input_size, input_size), max_num=0, metric="default")
        if faces is None or len(faces) == 0:
            rows.append(box_telemetry(row["relative_path"], width, height, np.empty((0, 4)), np.empty((0,)), "scrfd"))
            continue
        faces = np.asarray(faces, dtype=float)
        keep = faces[:, 4] >= confidence_threshold
        faces = faces[keep]
        if len(faces) == 0:
            rows.append(box_telemetry(row["relative_path"], width, height, np.empty((0, 4)), np.empty((0,)), "scrfd"))
            continue
        rows.append(box_telemetry(row["relative_path"], width, height, faces[:, :4], faces[:, 4], "scrfd"))
    del detector
    cleanup_cuda()
    return pd.DataFrame(rows)


def build_disagreement_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        y640_count = float(row.get("yolo640_det_yolo_count", 0.0))
        y1280_count = float(row.get("yolo1280_det_yolo_count", 0.0))
        scrfd_count = float(row.get("scrfd_count", 0.0))
        y640_area = float(row.get("yolo640_det_yolo_area_max", 0.0))
        y1280_area = float(row.get("yolo1280_det_yolo_area_max", 0.0))
        scrfd_area = float(row.get("scrfd_area_max", 0.0))
        y640_edge = float(row.get("yolo640_det_yolo_edge_count", 0.0))
        y1280_edge = float(row.get("yolo1280_det_yolo_edge_count", 0.0))
        scrfd_edge = float(row.get("scrfd_edge_count", 0.0))
        counts = np.array([y640_count, y1280_count, scrfd_count], dtype=float)
        rec = {
            "relative_path": row["relative_path"],
            "agree_count_min": float(counts.min()),
            "agree_count_max": float(counts.max()),
            "agree_count_range": float(counts.max() - counts.min()),
            "agree_count_std": float(counts.std()),
            "yolo_scale_count_gain": float(y1280_count - y640_count),
            "scrfd_vs_yolo640_count_gain": float(scrfd_count - y640_count),
            "scrfd_vs_yolo1280_count_gain": float(scrfd_count - y1280_count),
            "any_detector_found_face": float(max(counts) > 0),
            "all_detectors_found_no_face": float(max(counts) == 0),
            "detector_count_disagreement_flag": float((counts.max() - counts.min()) >= 2),
            "detector_any_disagreement_flag": float((counts.max() > 0) and (counts.min() == 0)),
            "yolo_scale_area_gain": float(y1280_area - y640_area),
            "scrfd_vs_yolo640_area_gain": float(scrfd_area - y640_area),
            "scrfd_vs_yolo1280_area_gain": float(scrfd_area - y1280_area),
            "detector_edge_count_max": float(max(y640_edge, y1280_edge, scrfd_edge)),
            "detector_edge_count_range": float(max(y640_edge, y1280_edge, scrfd_edge) - min(y640_edge, y1280_edge, scrfd_edge)),
        }
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
        "random_forest": OneVsRestClassifier(
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=500,
                            min_samples_leaf=2,
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
                            n_estimators=700,
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], label_support: pd.Series) -> dict[str, float]:
    supported_idx = [
        i
        for i, label in enumerate(labels)
        if label_support[label] >= SUPPORTED_MIN_SUPPORT and label not in EXCLUDED_LABELS
    ]
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
    folds: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_cols = [c for c in df.columns if c.startswith("label_")]
    label_support = df[label_cols].sum()
    y = df[label_cols].astype(int).to_numpy()
    groups = df["source_group"].fillna("unknown").astype(str).to_numpy()
    n_splits = min(folds, len(np.unique(groups)))
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
            fold_scores = []
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
            summary_rows.append(
                {
                    "method_id": method_id,
                    "feature_set": feature_set,
                    "estimator": estimator_name,
                    "feature_count": len(feature_cols),
                    "folds": n_splits,
                    "runtime_seconds": round(time.time() - started, 3),
                    "route_eligible_label_count": sum(1 for row in route_rows if row["route_eligible"]),
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
    ranked = summary.sort_values("oapr_scene_condition_score", ascending=False)
    current_best = ranked[ranked["method_id"] == "current_best_handcrafted_yolo640__logistic_regression"]
    current_score = float(current_best.iloc[0]["oapr_scene_condition_score"]) if not current_best.empty else math.nan
    best = ranked.iloc[0]
    delta = float(best["oapr_scene_condition_score"]) - current_score if not math.isnan(current_score) else math.nan
    best_labels = labels[labels["method_id"] == best["method_id"]].sort_values("label")
    lines = [
        "# Advanced Telemetry Scene-Condition Router Benchmark",
        "",
        "This benchmark tests the detector-telemetry hypothesis: multi-scale YOLO, SCRFD, and detector-disagreement features should improve scene-condition routing over the previous handcrafted + YOLO640 telemetry model.",
        "",
        "## Primary Metric",
        "",
        "`OAPR Scene-Condition Score = 0.50*macro_F2 + 0.25*macro_F1 + 0.15*micro_F2 + 0.10*sample_Jaccard`",
        "",
        "## Best Method",
        "",
        f"- Method: `{best['method_id']}`",
        f"- OAPR Scene-Condition Score: `{best['oapr_scene_condition_score']:.4f}`",
        f"- Delta vs current best handcrafted + YOLO640: `{delta:.4f}`",
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
            ["label", "support", "precision", "recall", "f1", "f2", "route_eligible", "included_in_primary_score"]
        ].to_markdown(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/02_face_detection/04_scene_condition_router/01_condition_dataset.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--yolo-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--scrfd-confidence-threshold", type=float, default=0.20)
    parser.add_argument(
        "--estimator-mode",
        choices=["logistic_only", "full_grid"],
        default="logistic_only",
        help="Use logistic_only for the main comparable run; full_grid is slower exploratory analysis.",
    )
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
    yolo640 = rename_feature_columns(
        extract_yolo_telemetry(df, args.batch_size, device, args.yolo_confidence_threshold, 640),
        "yolo640",
    )
    timings.append({"step": "yolo640_telemetry", "seconds": round(time.time() - started, 3), "device": str(device)})

    started = time.time()
    yolo1280 = rename_feature_columns(
        extract_yolo_telemetry(df, args.batch_size, device, args.yolo_confidence_threshold, 1280),
        "yolo1280",
    )
    timings.append({"step": "yolo1280_telemetry", "seconds": round(time.time() - started, 3), "device": str(device)})

    started = time.time()
    scrfd = extract_scrfd_telemetry(df, args.scrfd_confidence_threshold, 640)
    timings.append({"step": "scrfd_telemetry", "seconds": round(time.time() - started, 3), "device": str(device)})

    merged = df.copy()
    for feat in [hand, yolo640, yolo1280, scrfd]:
        overlap = [c for c in feat.columns if c != "relative_path" and c in merged.columns]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(feat, on="relative_path", how="left", validate="one_to_one")

    disagreement = build_disagreement_features(merged)
    merged = merged.merge(disagreement, on="relative_path", how="left", validate="one_to_one")

    hand_cols = [c for c in hand.columns if c != "relative_path"]
    yolo640_cols = [c for c in yolo640.columns if c != "relative_path"]
    yolo1280_cols = [c for c in yolo1280.columns if c != "relative_path"]
    scrfd_cols = [c for c in scrfd.columns if c != "relative_path"]
    disagree_cols = [c for c in disagreement.columns if c != "relative_path"]
    yolo_multiscale_cols = yolo640_cols + yolo1280_cols
    all_telemetry_cols = yolo640_cols + yolo1280_cols + scrfd_cols + disagree_cols

    feature_sets = {
        "current_best_handcrafted_yolo640": hand_cols + yolo640_cols,
        "yolo640_only": yolo640_cols,
        "yolo_multiscale_only": yolo_multiscale_cols,
        "scrfd_only": scrfd_cols,
        "detector_disagreement_only": disagree_cols,
        "yolo_scrfd_disagreement": all_telemetry_cols,
        "handcrafted_yolo_multiscale": hand_cols + yolo_multiscale_cols,
        "handcrafted_scrfd": hand_cols + scrfd_cols,
        "handcrafted_yolo_scrfd_disagreement": hand_cols + all_telemetry_cols,
    }
    if args.estimator_mode == "full_grid":
        low_dim_estimators = ["logistic_regression", "random_forest", "extra_trees"]
    else:
        low_dim_estimators = ["logistic_regression"]
    method_estimators = {
        "current_best_handcrafted_yolo640": ["logistic_regression"],
        "yolo640_only": ["logistic_regression"],
        "yolo_multiscale_only": low_dim_estimators,
        "scrfd_only": low_dim_estimators,
        "detector_disagreement_only": ["logistic_regression"],
        "yolo_scrfd_disagreement": low_dim_estimators,
        "handcrafted_yolo_multiscale": low_dim_estimators,
        "handcrafted_scrfd": low_dim_estimators,
        "handcrafted_yolo_scrfd_disagreement": low_dim_estimators,
    }
    summary, per_label, predictions = evaluate_methods(merged, feature_sets, method_estimators, args.folds, args.random_state)
    summary = summary.sort_values("oapr_scene_condition_score", ascending=False)
    per_label = per_label.sort_values(["method_id", "label"])
    timings.append({"step": "total", "seconds": round(time.time() - started_all, 3)})

    summary.to_csv(args.output_dir / "advanced_telemetry_model_benchmark.csv", index=False)
    per_label.to_csv(args.output_dir / "advanced_telemetry_per_label_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "advanced_telemetry_model_predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(args.output_dir / "advanced_telemetry_runtime.csv", index=False)
    pd.DataFrame(
        [{"feature_set": name, "feature_count": len(cols)} for name, cols in feature_sets.items()]
    ).to_csv(args.output_dir / "advanced_telemetry_feature_inventory.csv", index=False)
    write_report(summary, per_label, args.output_dir / "advanced_telemetry_model_benchmark.md")
    print(json.dumps({"rows": len(df), "output_dir": str(args.output_dir), "best": summary.iloc[0].to_dict()}, indent=2))


if __name__ == "__main__":
    main()
