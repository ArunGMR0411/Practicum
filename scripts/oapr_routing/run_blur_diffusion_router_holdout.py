#!/usr/bin/env python3

"""Evaluate blur-vs-diffusion routing rules under day-group holdout splits."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import GroupKFold

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
            177.59599947660413,
            233.53492022024525,
            250.0,
            299.78103428892473,
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


def apply_candidate(df: pd.DataFrame, candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trigger_flags = candidate["trigger_flags"]
    min_true = int(candidate["min_true_flags"])
    min_face_size = float(candidate["min_face_size_px"])
    trigger_count = df[trigger_flags].sum(axis=1)
    choose_diffusion = (trigger_count >= min_true) & (df["face_size_px"] >= min_face_size)
    selected_scores = np.where(choose_diffusion, df["diffusion_proxy"], df["blur_proxy"])
    return choose_diffusion.to_numpy(), selected_scores


def score_candidate(df: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
    choose_diffusion, selected_scores = apply_candidate(df, candidate)
    blur_scores = df["blur_proxy"].to_numpy()
    return {
        **candidate,
        "diffusion_selections": int(choose_diffusion.sum()),
        "blur_selections": int((~choose_diffusion).sum()),
        "mean_proxy_score": float(np.mean(selected_scores)),
        "mean_proxy_delta_vs_blur": float(np.mean(selected_scores) - np.mean(blur_scores)),
        "paired_ttest_vs_blur": safe_paired_ttest(selected_scores.tolist(), blur_scores.tolist()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument("--quality-csv", default="outputs/runs/routing/dev_quality_signals_mtcnn.csv")
    parser.add_argument("--perceptual-json", default="outputs/perceptual_results.json")
    parser.add_argument("--reid-json", default="outputs/experimental_runs/classical_baselines/reid_results.json")
    parser.add_argument("--output-json", default="outputs/blur_diffusion_router_holdout.json")
    parser.add_argument("--output-csv", default="outputs/blur_diffusion_router_holdout.csv")
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

    candidates = build_candidates(merged["face_size_px"])
    gkf = GroupKFold(n_splits=4)
    groups = merged["day_id"].astype(str)

    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    pooled_selected_scores: list[float] = []
    pooled_blur_scores: list[float] = []
    pooled_diffusion_scores: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(gkf.split(merged, groups=groups), start=1):
        train_df = merged.iloc[train_idx].reset_index(drop=True)
        test_df = merged.iloc[test_idx].reset_index(drop=True)

        train_results = [score_candidate(train_df, candidate) for candidate in candidates]
        train_results.sort(key=lambda row: row["mean_proxy_score"], reverse=True)
        best_candidate = train_results[0]

        test_scored = score_candidate(test_df, best_candidate)
        test_choose_diffusion, test_selected_scores = apply_candidate(test_df, best_candidate)
        test_blur_scores = test_df["blur_proxy"].to_numpy()
        test_diffusion_scores = test_df["diffusion_proxy"].to_numpy()

        pooled_selected_scores.extend(test_selected_scores.tolist())
        pooled_blur_scores.extend(test_blur_scores.tolist())
        pooled_diffusion_scores.extend(test_diffusion_scores.tolist())

        held_out_days = sorted(test_df["day_id"].astype(str).unique().tolist())
        fold_rows.append(
            {
                "fold_index": fold_index,
                "held_out_days": held_out_days,
                "n_test_frames": int(len(test_df)),
                "train_best_candidate": best_candidate,
                "test_result": test_scored,
                "test_fixed_blur_mean_proxy_score": float(np.mean(test_blur_scores)),
                "test_fixed_diffusion_mean_proxy_score": float(np.mean(test_diffusion_scores)),
            }
        )

        for row, choose_diff in zip(test_df.itertuples(index=False), test_choose_diffusion.tolist(), strict=False):
            selection_rows.append(
                {
                    "fold_index": fold_index,
                    "relative_path": str(row.relative_path),
                    "day_id": str(row.day_id),
                    "selected_method": "diffusion" if choose_diff else "blur",
                    "blur_proxy_score": float(row.blur_proxy),
                    "diffusion_proxy_score": float(row.diffusion_proxy),
                    "selected_proxy_score": float(row.diffusion_proxy if choose_diff else row.blur_proxy),
                    "motion_blur_flag": bool(row.motion_blur_flag),
                    "visible_screen_flag": bool(row.visible_screen_flag),
                    "visible_text_flag": bool(row.visible_text_flag),
                    "face_size_px": float(row.face_size_px),
                }
            )

    pooled_selected_scores_np = np.array(pooled_selected_scores)
    pooled_blur_scores_np = np.array(pooled_blur_scores)
    pooled_diffusion_scores_np = np.array(pooled_diffusion_scores)

    payload = {
        "n_frames": int(len(merged)),
        "holdout_scheme": "GroupKFold by day_id (4 folds)",
        "trigger_flags_considered": TRIGGER_FLAGS,
        "pooled_fixed_blur_mean_proxy_score": float(np.mean(pooled_blur_scores_np)),
        "pooled_fixed_diffusion_mean_proxy_score": float(np.mean(pooled_diffusion_scores_np)),
        "pooled_selected_mean_proxy_score": float(np.mean(pooled_selected_scores_np)),
        "pooled_delta_vs_blur": float(np.mean(pooled_selected_scores_np) - np.mean(pooled_blur_scores_np)),
        "pooled_paired_ttest_vs_blur": safe_paired_ttest(
            pooled_selected_scores_np.tolist(),
            pooled_blur_scores_np.tolist(),
        ),
        "fold_results": fold_rows,
        "interpretation": (
            "Each fold searches the best blur-vs-diffusion trigger rule on the training days only, "
            "then evaluates that rule on the held-out day group. Candidate rules use only "
            "motion_blur_flag, visible_screen_flag, visible_text_flag, and face_size_px."
        ),
    }

    output_json_path = PROJECT_ROOT / args.output_json
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pd.DataFrame(selection_rows).to_csv(PROJECT_ROOT / args.output_csv, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
