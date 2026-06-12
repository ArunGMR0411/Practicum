#!/usr/bin/env python3

"""Evaluate simple blur-vs-diffusion routing rules on the current dev set."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIGGER_FLAGS = ["motion_blur_flag", "visible_screen_flag", "visible_text_flag"]


def load_perceptual(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(entry["relative_path"]), str(entry["method"])): {
            "ssim": float(entry["ssim"]),
            "lpips": float(entry["lpips"]),
        }
        for entry in payload["detailed"]
    }


def load_reid(path: Path) -> dict[tuple[str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    grouped = df.groupby(["image_id", "method"], dropna=False).agg(
        adaface_hit_rate=("adaface_hit", "mean"),
    )
    return {
        (str(image_id), str(method)): float(row["adaface_hit_rate"])
        for (image_id, method), row in grouped.iterrows()
    }


def proxy_score(
    relative_path: str,
    method: str,
    perceptual: dict[tuple[str, str], dict[str, float]],
    reid: dict[tuple[str, str], float],
) -> float:
    ssim = perceptual[(relative_path, method)]["ssim"]
    hit_rate = reid.get((relative_path, method), 0.0)
    return float(ssim * (1.0 - hit_rate))


def safe_paired_ttest(a: list[float], b: list[float]) -> dict[str, Any]:
    result = ttest_rel(np.array(a), np.array(b))
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "n_pairs": len(a),
    }


def build_candidates(face_sizes: pd.Series) -> list[dict[str, Any]]:
    thresholds = sorted(
        {
            0.0,
            float(face_sizes.quantile(0.50)),
            float(face_sizes.quantile(0.67)),
            float(face_sizes.quantile(0.80)),
            250.0,
            400.0,
        }
    )
    candidates: list[dict[str, Any]] = []
    for r in range(1, len(TRIGGER_FLAGS) + 1):
        for combo in itertools.combinations(TRIGGER_FLAGS, r):
            for min_true in range(1, len(combo) + 1):
                for threshold in thresholds:
                    candidates.append(
                        {
                            "trigger_flags": list(combo),
                            "min_true_flags": min_true,
                            "min_face_size_px": float(threshold),
                        }
                    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument("--quality-csv", default="outputs/runs/routing/dev_quality_signals_mtcnn.csv")
    parser.add_argument("--perceptual-json", default="outputs/perceptual_results.json")
    parser.add_argument("--reid-json", default="outputs/experimental_runs/classical_baselines/reid_results.json")
    parser.add_argument("--output-json", default="outputs/blur_diffusion_router_experiment.json")
    parser.add_argument("--output-csv", default="outputs/blur_diffusion_router_experiment.csv")
    args = parser.parse_args()

    manifest_df = pd.read_csv(PROJECT_ROOT / args.manifest)
    quality_df = pd.read_csv(PROJECT_ROOT / args.quality_csv)
    perceptual = load_perceptual(PROJECT_ROOT / args.perceptual_json)
    reid = load_reid(PROJECT_ROOT / args.reid_json)

    merged = manifest_df.merge(
        quality_df[["relative_path", "face_size_px"]],
        on="relative_path",
        how="inner",
        validate="one_to_one",
    ).copy()
    for flag in TRIGGER_FLAGS:
        merged[flag] = merged[flag].fillna(False).astype(bool)

    merged["blur_proxy"] = merged["relative_path"].map(lambda rp: proxy_score(str(rp), "blur", perceptual, reid))
    merged["diffusion_proxy"] = merged["relative_path"].map(
        lambda rp: proxy_score(str(rp), "diffusion", perceptual, reid)
    )

    blur_scores = merged["blur_proxy"].tolist()
    diffusion_scores = merged["diffusion_proxy"].tolist()
    candidates = build_candidates(merged["face_size_px"])

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        trigger_flags = candidate["trigger_flags"]
        min_true = int(candidate["min_true_flags"])
        min_face_size = float(candidate["min_face_size_px"])
        trigger_count = merged[trigger_flags].sum(axis=1)
        choose_diffusion = (trigger_count >= min_true) & (merged["face_size_px"] >= min_face_size)
        selected_scores = np.where(choose_diffusion, merged["diffusion_proxy"], merged["blur_proxy"])
        results.append(
            {
                **candidate,
                "diffusion_selections": int(choose_diffusion.sum()),
                "blur_selections": int((~choose_diffusion).sum()),
                "mean_proxy_score": float(np.mean(selected_scores)),
                "mean_proxy_delta_vs_blur": float(np.mean(selected_scores) - np.mean(blur_scores)),
                "paired_ttest_vs_blur": safe_paired_ttest(selected_scores.tolist(), blur_scores),
            }
        )

    results.sort(key=lambda row: row["mean_proxy_score"], reverse=True)
    best = results[0]
    output_payload = {
        "n_frames": int(len(merged)),
        "fixed_blur_mean_proxy_score": float(np.mean(blur_scores)),
        "fixed_diffusion_mean_proxy_score": float(np.mean(diffusion_scores)),
        "best_candidate": best,
        "top_candidates": results[:10],
        "interpretation": (
            "Candidates route to diffusion only when the configured multi-flag trigger count and "
            "minimum face-size gate are both satisfied. Comparison is against fixed blur on the "
            "same dev frames using the current SSIM x (1 - AdaFace hit-rate) proxy."
        ),
    }

    output_json_path = PROJECT_ROOT / args.output_json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")

    csv_df = pd.DataFrame(
        [
            {
                "trigger_flags": "|".join(row["trigger_flags"]),
                "min_true_flags": row["min_true_flags"],
                "min_face_size_px": row["min_face_size_px"],
                "diffusion_selections": row["diffusion_selections"],
                "blur_selections": row["blur_selections"],
                "mean_proxy_score": row["mean_proxy_score"],
                "mean_proxy_delta_vs_blur": row["mean_proxy_delta_vs_blur"],
                "p_value_vs_blur": row["paired_ttest_vs_blur"]["p_value"],
            }
            for row in results
        ]
    )
    csv_df.to_csv(PROJECT_ROOT / args.output_csv, index=False)

    print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
