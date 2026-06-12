#!/usr/bin/env python3

"""Sweep blur and face-size routing thresholds on calibration-set proxy scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        arcface_hit_rate=("arcface_hit", "mean"),
        arcface_cosine_mean=("arcface_cosine_sim", "mean"),
        face_crop_count=("box_idx", "count"),
    )
    result: dict[tuple[str, str], dict[str, float]] = {}
    for (image_id, method), row in grouped.iterrows():
        result[(str(image_id), str(method))] = {
            "adaface_hit_rate": float(row["adaface_hit_rate"]),
            "adaface_cosine_mean": float(row["adaface_cosine_mean"]),
            "arcface_hit_rate": float(row["arcface_hit_rate"]),
            "arcface_cosine_mean": float(row["arcface_cosine_mean"]),
            "face_crop_count": int(row["face_crop_count"]),
        }
    return result


def build_frame_score_table(
    quality_df: pd.DataFrame,
    perceptual_by_frame: dict[tuple[str, str], dict[str, float]],
    reid_by_frame: dict[tuple[str, str], dict[str, float]],
    methods: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in quality_df.itertuples(index=False):
        relative_path = str(row.relative_path)
        base = {
            "relative_path": relative_path,
            "blur_score": float(row.blur_score),
            "face_size_px": float(row.face_size_px),
            "occlusion_ratio": float(row.occlusion_ratio),
            "webp_artifact_score": float(row.webp_artifact_score),
            "face_box_count": int(row.face_box_count),
        }
        complete = True
        for method in methods:
            perceptual = perceptual_by_frame.get((relative_path, method))
            reid = reid_by_frame.get((relative_path, method))
            if perceptual is None or reid is None:
                complete = False
                break
            # Primary developmental proxy already used in project decisions.
            proxy_score = float(perceptual["ssim"]) * float(1.0 - reid["adaface_hit_rate"])
            base[f"{method}_proxy_score"] = proxy_score
            base[f"{method}_ssim"] = float(perceptual["ssim"])
            base[f"{method}_lpips"] = float(perceptual["lpips"])
            base[f"{method}_adaface_hit_rate"] = float(reid["adaface_hit_rate"])
            base[f"{method}_adaface_cosine_mean"] = float(reid["adaface_cosine_mean"])
            base[f"{method}_face_crop_count"] = int(reid["face_crop_count"])
        if complete:
            rows.append(base)
    return pd.DataFrame(rows)


def choose_method(face_size_px: float, blur_score: float, size_threshold: float, blur_threshold: float) -> str:
    if face_size_px <= size_threshold:
        return "blur"
    if blur_score >= blur_threshold:
        return "blur"
    return "pixelate"


def build_threshold_grid(series: pd.Series, quantiles: list[float]) -> list[float]:
    values = sorted({round(float(series.quantile(q)), 6) for q in quantiles})
    return values


def evaluate_grid(frame_df: pd.DataFrame, size_grid: list[float], blur_grid: list[float]) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    for size_threshold in size_grid:
        for blur_threshold in blur_grid:
            assigned = frame_df.apply(
                lambda row: choose_method(
                    face_size_px=float(row["face_size_px"]),
                    blur_score=float(row["blur_score"]),
                    size_threshold=size_threshold,
                    blur_threshold=blur_threshold,
                ),
                axis=1,
            )
            chosen_scores = [
                float(frame_df.iloc[index][f"{method}_proxy_score"])
                for index, method in enumerate(assigned.tolist())
            ]
            average_score = float(sum(chosen_scores) / len(chosen_scores)) if chosen_scores else 0.0
            blur_share = float((assigned == "blur").mean()) if len(assigned) else 0.0
            pixelate_share = float((assigned == "pixelate").mean()) if len(assigned) else 0.0
            row = {
                "face_size_threshold_px": size_threshold,
                "blur_threshold": blur_threshold,
                "average_proxy_score": average_score,
                "blur_share": blur_share,
                "pixelate_share": pixelate_share,
            }
            rows.append(row)
            if best_row is None or average_score > float(best_row["average_proxy_score"]):
                best_row = row
    if best_row is None:
        raise ValueError("Threshold grid evaluation produced no candidates")
    return rows, best_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", default="outputs/01_protocol/05_calibration_quality_signals.csv")
    parser.add_argument("--perceptual-json", default="outputs/calibration_perceptual_results.json")
    parser.add_argument("--reid-json", default="outputs/calibration_reid_results.json")
    parser.add_argument("--output-json", default="outputs/runs/routing_baseline/routing_calibration.json")
    args = parser.parse_args()

    quality_df = pd.read_csv(PROJECT_ROOT / args.quality_csv)
    perceptual_by_frame = load_perceptual_by_frame(PROJECT_ROOT / args.perceptual_json)
    reid_by_frame = load_reid_by_frame(PROJECT_ROOT / args.reid_json)
    methods = ["blur", "pixelate"]

    frame_df = build_frame_score_table(
        quality_df=quality_df,
        perceptual_by_frame=perceptual_by_frame,
        reid_by_frame=reid_by_frame,
        methods=methods,
    )
    if frame_df.empty:
        raise ValueError("No complete calibration frames were available for routing-threshold sweep")

    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    size_grid = build_threshold_grid(frame_df["face_size_px"], quantiles)
    blur_grid = build_threshold_grid(frame_df["blur_score"], quantiles)
    grid_rows, best_row = evaluate_grid(frame_df, size_grid=size_grid, blur_grid=blur_grid)

    baseline_scores = {
        method: float(frame_df[f"{method}_proxy_score"].mean())
        for method in methods
    }
    top_rows = sorted(grid_rows, key=lambda row: float(row["average_proxy_score"]), reverse=True)[:10]

    payload = {
        "quality_csv": args.quality_csv,
        "perceptual_json": args.perceptual_json,
        "reid_json": args.reid_json,
        "frame_count": int(len(frame_df)),
        "methods": methods,
        "proxy_objective": "frame_ssim * (1 - frame_adaface_hit_rate)",
        "baseline_average_proxy_scores": baseline_scores,
        "search_space": {
            "face_size_threshold_px": size_grid,
            "blur_threshold": blur_grid,
        },
        "best_thresholds": best_row,
        "top_candidates": top_rows,
        "grid_results": grid_rows,
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["best_thresholds"], indent=2))


if __name__ == "__main__":
    main()
