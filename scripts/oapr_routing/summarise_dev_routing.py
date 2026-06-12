#!/usr/bin/env python3

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_perceptual(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for entry in payload["detailed"]:
        result[(entry["relative_path"], entry["method"])] = {
            "ssim": entry["ssim"],
            "lpips": entry["lpips"],
        }
    return result

def load_reid(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    grouped = df.groupby(["image_id", "method"]).agg(
        adaface_hit_rate=("adaface_hit", "mean"),
    )
    result = {}
    for (image_id, method), row in grouped.iterrows():
        result[(image_id, method)] = {
            "adaface_hit_rate": row["adaface_hit_rate"],
        }
    return result

def main():
    log_path = PROJECT_ROOT / "outputs/runs/routing_baseline/routing_decision_log_dev.csv"
    perceptual_path = PROJECT_ROOT / "outputs/perceptual_results.json"
    reid_path = PROJECT_ROOT / "outputs/experimental_runs/classical_baselines/reid_results.json"

    if not all(p.exists() for p in [log_path, perceptual_path, reid_path]):
        print("Required evaluation files are missing.")
        sys.exit(1)

    log_df = pd.read_csv(log_path)
    perceptual = load_perceptual(perceptual_path)
    reid = load_reid(reid_path)

    # Compile per-frame metrics for each method
    frames = log_df["relative_path"].tolist()
    
    # We want to measure: ssim, lpips, reid_rate, proxy_score (ssim * (1 - reid_rate))
    methods = ["blur", "pixelate"]
    
    frame_metrics = {}
    for frame in frames:
        frame_metrics[frame] = {}
        for m in methods:
            p = perceptual.get((frame, m), {"ssim": 0.0, "lpips": 0.0})
            r = reid.get((frame, m), {"adaface_hit_rate": 0.0})
            ssim = p["ssim"]
            lpips = p["lpips"]
            reid_rate = r["adaface_hit_rate"]
            proxy = ssim * (1.0 - reid_rate)
            
            frame_metrics[frame][m] = {
                "ssim": ssim,
                "lpips": lpips,
                "reid_rate": reid_rate,
                "proxy": proxy
            }

    # Evaluate strategies
    strategies = {
        "fixed_blur": lambda f: "blur",
        "fixed_pixelate": lambda f: "pixelate",
        "rule_based_router": lambda f: log_df.loc[log_df["relative_path"] == f, "rule_method"].values[0],
        "learned_router": lambda f: log_df.loc[log_df["relative_path"] == f, "learned_method"].values[0],
        "oracle_best": lambda f: "blur" if frame_metrics[f]["blur"]["proxy"] >= frame_metrics[f]["pixelate"]["proxy"] else "pixelate"
    }

    summary = {}
    for name, strategy_fn in strategies.items():
        ssims, lpipss, reids, proxies = [], [], [], []
        choices = {"blur": 0, "pixelate": 0}
        for f in frames:
            choice = strategy_fn(f)
            choices[choice] += 1
            metrics = frame_metrics[f][choice]
            ssims.append(metrics["ssim"])
            lpipss.append(metrics["lpips"])
            reids.append(metrics["reid_rate"])
            proxies.append(metrics["proxy"])
        
        summary[name] = {
            "blur_selections": choices["blur"],
            "pixelate_selections": choices["pixelate"],
            "ssim_mean": float(np.mean(ssims)),
            "lpips_mean": float(np.mean(lpipss)),
            "reid_rate_mean": float(np.mean(reids)),
            "proxy_score_mean": float(np.mean(proxies))
        }

    # Output JSON summary
    output_json = PROJECT_ROOT / "outputs/runs/routing_baseline/routing_dev_results.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    # Print beautiful table
    print("\n" + "="*95)
    print("Dev Set Routing Strategies Comparison".center(95))
    print("="*95)
    print(f"{'Strategy':<22} | {'Blur/Pixel':<12} | {'Mean SSIM':<11} | {'Mean LPIPS':<11} | {'Mean Re-ID':<11} | {'Mean Proxy':<11}")
    print("-"*95)
    for name, metrics in summary.items():
        selections = f"{metrics['blur_selections']}/{metrics['pixelate_selections']}"
        print(f"{name:<22} | "
              f"{selections:<12} | "
              f"{metrics['ssim_mean']:<11.4f} | "
              f"{metrics['lpips_mean']:<11.4f} | "
              f"{metrics['reid_rate_mean']:<11.4f} | "
              f"{metrics['proxy_score_mean']:<11.4f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
