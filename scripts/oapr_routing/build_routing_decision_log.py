#!/usr/bin/env python3

"""Build a per-frame routing decision log with proxy metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.routing import LearnedRouter, QualityAssessment, QualitySignals, RuleBasedRouter, save_routing_decisions


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


def method_metrics(
    relative_path: str,
    method_name: str,
    perceptual_by_frame: dict[tuple[str, str], dict[str, float]],
    reid_by_frame: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    perceptual = perceptual_by_frame[(relative_path, method_name)]
    reid = reid_by_frame.get(
        (relative_path, method_name),
        {
            "adaface_hit_rate": 0.0,
            "adaface_cosine_mean": 0.0,
            "arcface_hit_rate": 0.0,
            "arcface_cosine_mean": 0.0,
            "face_crop_count": 0,
        },
    )
    return {
        "ssim": float(perceptual["ssim"]),
        "lpips": float(perceptual["lpips"]),
        "adaface_hit_rate": float(reid["adaface_hit_rate"]),
        "adaface_cosine_mean": float(reid["adaface_cosine_mean"]),
        "arcface_hit_rate": float(reid["arcface_hit_rate"]),
        "arcface_cosine_mean": float(reid["arcface_cosine_mean"]),
        "face_crop_count": int(reid["face_crop_count"]),
        "proxy_score": float(perceptual["ssim"]) * float(1.0 - reid["adaface_hit_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-csv", default="outputs/01_protocol/05_calibration_quality_signals.csv")
    parser.add_argument("--perceptual-json", default="outputs/calibration_perceptual_results.json")
    parser.add_argument("--reid-json", default="outputs/calibration_reid_results.json")
    parser.add_argument("--learned-model", default="outputs/runs/routing_baseline/learned_router.joblib")
    parser.add_argument("--output-csv", default="outputs/runs/routing_baseline/routing_decision_log_calibration.csv")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Record router selections without requiring perceptual or re-identification metrics.",
    )
    args = parser.parse_args()

    quality_df = pd.read_csv(PROJECT_ROOT / args.quality_csv)
    perceptual_by_frame = (
        {}
        if args.selection_only
        else load_perceptual_by_frame(PROJECT_ROOT / args.perceptual_json)
    )
    reid_by_frame = (
        {}
        if args.selection_only
        else load_reid_by_frame(PROJECT_ROOT / args.reid_json)
    )

    rule_router = RuleBasedRouter()
    learned_router = LearnedRouter(PROJECT_ROOT / args.learned_model)

    rows: list[dict[str, object]] = []
    for row in quality_df.itertuples(index=False):
        relative_path = str(row.relative_path)
        assessment = QualityAssessment(
            signals=QualitySignals(
                blur_score=float(row.blur_score),
                face_size_px=float(row.face_size_px),
                occlusion_ratio=float(row.occlusion_ratio),
                webp_artifact_score=float(row.webp_artifact_score),
            ),
            metadata={"face_box_count": int(row.face_box_count)},
        )
        rule_decision = rule_router.decide(assessment)
        learned_decision = learned_router.decide(assessment)

        base_row = {
            "relative_path": relative_path,
            "blur_score": float(row.blur_score),
            "face_size_px": float(row.face_size_px),
            "occlusion_ratio": float(row.occlusion_ratio),
            "webp_artifact_score": float(row.webp_artifact_score),
            "face_box_count": int(row.face_box_count),
            "rule_method": rule_decision.method_name,
            "rule_reason": str(rule_decision.metadata.get("route_reason", "")),
            "learned_method": learned_decision.method_name,
        }

        if args.selection_only:
            rows.append(base_row)
            continue

        available_methods = [m for (p, m) in perceptual_by_frame.keys() if p == relative_path]
        if not available_methods or "blur" not in available_methods:
            continue

        frame_metrics = {
            m: method_metrics(relative_path, m, perceptual_by_frame, reid_by_frame)
            for m in available_methods
        }

        best_proxy_method = max(frame_metrics.keys(), key=lambda m: frame_metrics[m]["proxy_score"])
        best_proxy_score = frame_metrics[best_proxy_method]["proxy_score"]

        selected_rule_metrics = frame_metrics.get(rule_decision.method_name, frame_metrics["blur"])
        selected_learned_metrics = frame_metrics.get(learned_decision.method_name, frame_metrics["blur"])

        record = {
            **base_row,
            "rule_proxy_score": float(selected_rule_metrics["proxy_score"]),
            "rule_ssim": float(selected_rule_metrics["ssim"]),
            "rule_lpips": float(selected_rule_metrics["lpips"]),
            "rule_adaface_hit_rate": float(selected_rule_metrics["adaface_hit_rate"]),
            "learned_proxy_score": float(selected_learned_metrics["proxy_score"]),
            "learned_ssim": float(selected_learned_metrics["ssim"]),
            "learned_lpips": float(selected_learned_metrics["lpips"]),
            "learned_adaface_hit_rate": float(selected_learned_metrics["adaface_hit_rate"]),
            "best_proxy_method": best_proxy_method,
            "best_proxy_score": float(best_proxy_score),
        }

        for m in ["blur", "pixelate", "diffusion"]:
            if m in frame_metrics:
                record[f"{m}_proxy_score"] = float(frame_metrics[m]["proxy_score"])

        rows.append(record)

    output_path = PROJECT_ROOT / args.output_csv
    save_routing_decisions(rows, output_path)
    print(json.dumps({"rows_written": len(rows), "output_csv": args.output_csv}, indent=2))


if __name__ == "__main__":
    main()
