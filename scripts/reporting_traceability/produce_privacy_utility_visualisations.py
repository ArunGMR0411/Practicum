"""Produce privacy-utility visualisations from executed project artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs" / "figures"

ANON_RESULTS = PROJECT_ROOT / "outputs" / "submission_evidence" / "classical_baselines" / "anonymisation_full_results_yolo_scrfd_fallback.json"
ROUTING_RESULTS = PROJECT_ROOT / "outputs" / "submission_evidence" / "routing_analysis" / "routing_dev_results.json"
ROUTING_RUNTIME = PROJECT_ROOT / "outputs" / "submission_evidence" / "routing_analysis" / "routing_runtime_benchmark_dev.json"
FID_BASELINE = PROJECT_ROOT / "outputs" / "submission_evidence" / "baseline_metrics" / "fid_webp_baseline.json"
UTILITY_BUNDLE = PROJECT_ROOT / "outputs" / "submission_evidence" / "classical_baselines" / "utility_preservation_bundle.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_privacy_utility_scatter(anon: dict) -> Path:
    rows = []
    for method in anon["methods_evaluated"]:
        rows.append(
            {
                "method": method,
                "ssim": anon["perceptual_summary"][method]["ssim_mean"],
                "adaface_reid_rate": anon["reid_summary"][method]["adaface_reid_rate"],
            }
        )
    df = pd.DataFrame(rows)

    path = OUTPUT_DIR / "privacy_utility_ssim_vs_adaface_reid.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(df["ssim"], df["adaface_reid_rate"] * 100, s=140, color=["#2d6a4f", "#bc4749"])
    for _, row in df.iterrows():
        plt.annotate(row["method"], (row["ssim"], row["adaface_reid_rate"] * 100), xytext=(8, 6), textcoords="offset points")
    plt.xlabel("Mean SSIM against original WebP frames")
    plt.ylabel("AdaFace closed-set re-ID rate (%)")
    plt.title("Privacy-Utility Trade-Off on 500-Frame Anonymisation Subset")
    plt.grid(alpha=0.25)
    _save_figure(path)
    return path


def plot_router_proxy_vs_time(routing: dict, runtime: dict) -> Path:
    rows = [
        {
            "strategy": "fixed_blur",
            "proxy": routing["fixed_strategy_proxy_means"]["blur"],
            "time_ms": runtime["strategies"]["fixed_blur"]["mean_per_frame_ms"],
        },
        {
            "strategy": "fixed_pixelate",
            "proxy": routing["fixed_strategy_proxy_means"]["pixelate"],
            "time_ms": runtime["strategies"]["fixed_pixelate"]["mean_per_frame_ms"],
        },
        {
            "strategy": "rule_based_router",
            "proxy": routing["rule_based_router"]["mean_proxy_score"],
            "time_ms": runtime["strategies"]["rule_based_router"]["mean_per_frame_ms"],
        },
        {
            "strategy": "learned_router",
            "proxy": routing["learned_router"]["mean_proxy_score"],
            "time_ms": runtime["strategies"]["learned_router"]["mean_per_frame_ms"],
        },
    ]
    df = pd.DataFrame(rows)

    path = OUTPUT_DIR / "router_proxy_score_vs_runtime.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(df["time_ms"], df["proxy"], s=130, color="#31572c")
    for _, row in df.iterrows():
        plt.annotate(row["strategy"], (row["time_ms"], row["proxy"]), xytext=(7, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Mean router decision time per frame (ms, log scale)")
    plt.ylabel("Mean proxy score")
    plt.title("Routing Quality Versus Decision Runtime")
    plt.grid(alpha=0.25, which="both")
    _save_figure(path)
    return path


def plot_reid_rate_reduction(anon: dict) -> Path:
    methods = anon["methods_evaluated"]
    original = [100.0 for _ in methods]
    after = [anon["reid_summary"][method]["adaface_reid_rate"] * 100 for method in methods]

    path = OUTPUT_DIR / "adaface_reid_rate_pre_post_anonymisation.png"
    plt.figure(figsize=(7, 5))
    x_positions = range(len(methods))
    plt.plot(x_positions, original, marker="o", label="Original paired crop")
    plt.plot(x_positions, after, marker="o", label="After anonymisation")
    plt.xticks(list(x_positions), methods)
    plt.ylabel("AdaFace closed-set re-ID rate (%)")
    plt.title("Residual Privacy Attack Rate Before and After Anonymisation")
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(path)
    return path


def plot_fid_lpips_status(anon: dict, fid: dict) -> Path:
    methods = anon["methods_evaluated"]
    lpips_values = [anon["perceptual_summary"][method]["lpips_mean"] for method in methods]

    path = OUTPUT_DIR / "fid_vs_lpips_status.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(lpips_values, [0 for _ in methods], s=120, color="#6c757d", label="FID pending for non-generative baselines")
    for method, lpips in zip(methods, lpips_values):
        plt.annotate(method, (lpips, 0), xytext=(8, 6), textcoords="offset points")
    plt.axhline(0, color="#adb5bd", linewidth=1)
    plt.text(
        min(lpips_values),
        0.12,
        f"WebP self-FID baseline: {fid['fid_value']:.4f}\nAnonymised generative FID queued",
        ha="left",
        va="bottom",
    )
    plt.xlabel("Mean LPIPS against original WebP frames")
    plt.ylabel("FID delta relative to WebP baseline")
    plt.title("FID Reporting Status Relative to WebP Baseline")
    plt.yticks([0], ["pending"])
    plt.ylim(-0.2, 0.8)
    plt.grid(alpha=0.25, axis="x")
    plt.legend(loc="upper right")
    _save_figure(path)
    return path


def plot_bounded_nullface_tradeoff(bundle: dict) -> Path:
    methods = ["blur", "pixelate", "nullface"]
    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "ssim": bundle["perceptual_summary"][method]["ssim_mean"],
                "adaface_reid_rate": bundle["reid_summary"][method]["adaface_reid_rate"],
            }
        )
    df = pd.DataFrame(rows)

    path = OUTPUT_DIR / "bounded_nullface_hard_case_tradeoff.png"
    plt.figure(figsize=(7, 5))
    colors = {"blur": "#2d6a4f", "pixelate": "#bc4749", "nullface": "#1d3557"}
    plt.scatter(
        df["ssim"],
        df["adaface_reid_rate"] * 100,
        s=140,
        color=[colors[m] for m in df["method"]],
    )
    for _, row in df.iterrows():
        plt.annotate(row["method"], (row["ssim"], row["adaface_reid_rate"] * 100), xytext=(8, 6), textcoords="offset points")
    plt.xlabel("Mean SSIM against original WebP frames")
    plt.ylabel("AdaFace closed-set re-ID rate (%)")
    plt.title("Bounded Hard-Case Slice: NullFace vs Baselines")
    plt.grid(alpha=0.25)
    _save_figure(path)
    return path


def plot_utility_preservation(bundle: dict) -> Path:
    rows = bundle["method_summary"]
    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "utility_preservation_scores.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["utility_visual_score_mean"],
        df["demographic_consistency_score_mean"],
        s=140,
        color="#264653",
    )
    for _, row in df.iterrows():
        plt.annotate(
            row["method"],
            (row["utility_visual_score_mean"], row["demographic_consistency_score_mean"]),
            xytext=(8, 6),
            textcoords="offset points",
        )
    plt.xlabel("Reviewed utility visual score mean")
    plt.ylabel("Reviewed demographic consistency score mean")
    plt.title("Utility Preservation Beyond Perceptual Similarity")
    plt.xlim(0.5, 5.2)
    plt.ylim(0.5, 5.2)
    plt.grid(alpha=0.25)
    _save_figure(path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anon = _load_json(ANON_RESULTS)
    routing = _load_json(ROUTING_RESULTS)
    runtime = _load_json(ROUTING_RUNTIME)
    fid = _load_json(FID_BASELINE)
    utility_bundle = _load_json(UTILITY_BUNDLE)

    figures = [
        plot_privacy_utility_scatter(anon),
        plot_router_proxy_vs_time(routing, runtime),
        plot_reid_rate_reduction(anon),
        plot_fid_lpips_status(anon, fid),
        plot_utility_preservation(utility_bundle),
    ]
    manifest = {
        "version": "privacy_utility_visualisations",
        "source_artifacts": {
            "anonymisation": str(ANON_RESULTS.relative_to(PROJECT_ROOT)),
            "routing": str(ROUTING_RESULTS.relative_to(PROJECT_ROOT)),
            "routing_runtime": str(ROUTING_RUNTIME.relative_to(PROJECT_ROOT)),
            "fid_baseline": str(FID_BASELINE.relative_to(PROJECT_ROOT)),
            "nullface_bounded": str(NULLFACE_BOUNDED.relative_to(PROJECT_ROOT)),
            "utility_bundle": str(UTILITY_BUNDLE.relative_to(PROJECT_ROOT)),
        },
        "figures": [str(path.relative_to(PROJECT_ROOT)) for path in figures],
        "fid_note": "Anonymised generative FID values remain queued; current FID figure records the WebP baseline rule and pending status.",
    }
    manifest_path = OUTPUT_DIR / "privacy_utility_visualisations.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
