#!/usr/bin/env python3

"""Train the learned routing baseline on calibration-set proxy winners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_NAMES = [
    "blur_score",
    "face_size_px",
    "occlusion_ratio",
    "webp_artifact_score",
    "face_box_count",
]


def load_perceptual_by_frame(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], dict[str, float]] = {}
    for entry in payload["detailed"]:
        result[(str(entry["relative_path"]), str(entry["method"]))] = {
            "ssim": float(entry["ssim"]),
            "lpips": float(entry["lpips"]),
        }
    return result


def load_reid_by_frame(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    grouped = df.groupby(["image_id", "method"], dropna=False).agg(
        adaface_hit_rate=("adaface_hit", "mean"),
        adaface_cosine_mean=("adaface_cosine_sim", "mean"),
        face_crop_count=("box_idx", "count"),
    )
    result: dict[tuple[str, str], dict[str, float]] = {}
    for (image_id, method), row in grouped.iterrows():
        result[(str(image_id), str(method))] = {
            "adaface_hit_rate": float(row["adaface_hit_rate"]),
            "adaface_cosine_mean": float(row["adaface_cosine_mean"]),
            "face_crop_count": int(row["face_crop_count"]),
        }
    return result


def build_frame_table(
    quality_df: pd.DataFrame,
    perceptual_by_frame: dict[tuple[str, str], dict[str, float]],
    reid_by_frame: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in quality_df.itertuples(index=False):
        relative_path = str(row.relative_path)
        blur_perceptual = perceptual_by_frame.get((relative_path, "blur"))
        pixel_perceptual = perceptual_by_frame.get((relative_path, "pixelate"))
        blur_reid = reid_by_frame.get((relative_path, "blur"))
        pixel_reid = reid_by_frame.get((relative_path, "pixelate"))
        if not all([blur_perceptual, pixel_perceptual, blur_reid, pixel_reid]):
            continue

        blur_proxy_score = float(blur_perceptual["ssim"]) * float(1.0 - blur_reid["adaface_hit_rate"])
        pixel_proxy_score = float(pixel_perceptual["ssim"]) * float(1.0 - pixel_reid["adaface_hit_rate"])
        winner = "blur" if blur_proxy_score >= pixel_proxy_score else "pixelate"

        rows.append(
            {
                "relative_path": relative_path,
                "blur_score": float(row.blur_score),
                "face_size_px": float(row.face_size_px),
                "occlusion_ratio": float(row.occlusion_ratio),
                "webp_artifact_score": float(row.webp_artifact_score),
                "face_box_count": int(row.face_box_count),
                "blur_proxy_score": blur_proxy_score,
                "pixelate_proxy_score": pixel_proxy_score,
                "winner": winner,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", default="outputs/01_protocol/05_calibration_quality_signals.csv")
    parser.add_argument("--perceptual-json", default="outputs/calibration_perceptual_results.json")
    parser.add_argument("--reid-json", default="outputs/calibration_reid_results.json")
    parser.add_argument("--routing-calibration-json", default="outputs/runs/routing_baseline/routing_calibration.json")
    parser.add_argument("--model-output", default="outputs/runs/routing_baseline/learned_router.joblib")
    parser.add_argument("--summary-output", default="outputs/learned_router_summary.json")
    args = parser.parse_args()

    quality_df = pd.read_csv(PROJECT_ROOT / args.quality_csv)
    perceptual_by_frame = load_perceptual_by_frame(PROJECT_ROOT / args.perceptual_json)
    reid_by_frame = load_reid_by_frame(PROJECT_ROOT / args.reid_json)
    frame_df = build_frame_table(quality_df, perceptual_by_frame, reid_by_frame)
    if frame_df.empty:
        raise ValueError("No complete calibration rows available for learned-router training")

    X = frame_df[FEATURE_NAMES]
    y = frame_df["winner"]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = cross_val_predict(model, X, y, cv=cv, method="predict")
    cv_accuracy = float(accuracy_score(y, oof_predictions))

    learned_proxy_scores = [
        float(frame_df.iloc[index][f"{label}_proxy_score"])
        for index, label in enumerate(oof_predictions.tolist())
    ]
    learned_average_proxy_score = float(sum(learned_proxy_scores) / len(learned_proxy_scores))

    routing_payload = json.loads((PROJECT_ROOT / args.routing_calibration_json).read_text(encoding="utf-8"))
    rule_based_average_proxy_score = float(routing_payload["best_thresholds"]["average_proxy_score"])
    baseline_average_proxy_scores = {
        key: float(value)
        for key, value in routing_payload["baseline_average_proxy_scores"].items()
    }

    model.fit(X, y)
    model_output_path = PROJECT_ROOT / args.model_output
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "classes": list(model.named_steps["classifier"].classes_),
            "model_version": "learned_router",
        },
        model_output_path,
    )

    summary = {
        "quality_csv": args.quality_csv,
        "perceptual_json": args.perceptual_json,
        "reid_json": args.reid_json,
        "routing_calibration_json": args.routing_calibration_json,
        "frame_count": int(len(frame_df)),
        "feature_names": FEATURE_NAMES,
        "label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "cross_validated_accuracy": cv_accuracy,
        "cross_validated_average_proxy_score": learned_average_proxy_score,
        "rule_based_average_proxy_score": rule_based_average_proxy_score,
        "baseline_average_proxy_scores": baseline_average_proxy_scores,
        "confusion_matrix": {
            "labels": ["blur", "pixelate"],
            "matrix": confusion_matrix(y, oof_predictions, labels=["blur", "pixelate"]).tolist(),
        },
        "comparison": {
            "beats_rule_based": bool(learned_average_proxy_score > rule_based_average_proxy_score),
            "beats_fixed_blur": bool(learned_average_proxy_score > baseline_average_proxy_scores["blur"]),
            "beats_fixed_pixelate": bool(learned_average_proxy_score > baseline_average_proxy_scores["pixelate"]),
        },
        "model_output": args.model_output,
    }

    summary_output_path = PROJECT_ROOT / args.summary_output
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
