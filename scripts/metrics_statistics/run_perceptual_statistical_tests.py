#!/usr/bin/env python3

"""Paired statistical tests on perceptual anonymisation metrics.

Reads per-frame SSIM and LPIPS scores from the evaluation output JSON,
pairs them by frame (blur vs pixelate), and runs a paired t-test for each
metric. Reports t-statistics, two-sided p-values, and 95% confidence
intervals for the mean difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_paired_scores(
    results_path: Path,
) -> dict[str, dict[str, list[float]]]:
    """Load per-frame scores and group them by relative_path and method.

    Returns a dict mapping relative_path -> {method: {ssim: ..., lpips: ...}}.
    """
    with results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    by_path: dict[str, dict[str, dict[str, float]]] = {}
    for entry in data["detailed"]:
        rel = entry["relative_path"]
        method = entry["method"]
        by_path.setdefault(rel, {})[method] = {
            "ssim": entry["ssim"],
            "lpips": entry["lpips"],
        }

    return by_path


def paired_ttest_with_ci(
    scores_a: list[float],
    scores_b: list[float],
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Run a paired t-test and compute a confidence interval for mean diff.

    The test is two-sided: H0 is that the mean difference is zero.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b
    n = len(diff)

    result = ttest_rel(a, b)
    t_stat = float(result.statistic)
    p_value = float(result.pvalue)

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se_diff = std_diff / np.sqrt(n)

    from scipy.stats import t as t_dist

    alpha = 1 - confidence
    t_crit = float(t_dist.ppf(1 - alpha / 2, df=n - 1))
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "se_difference": float(se_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence,
        "n_pairs": n,
    }


def run_statistical_tests(
    results_path: Path,
    method_a: str = "blur",
    method_b: str = "pixelate",
) -> dict[str, Any]:
    """Load paired scores and run t-tests for SSIM and LPIPS."""
    by_path = load_paired_scores(results_path)

    ssim_a: list[float] = []
    ssim_b: list[float] = []
    lpips_a: list[float] = []
    lpips_b: list[float] = []

    for rel_path, methods in sorted(by_path.items()):
        if method_a not in methods or method_b not in methods:
            continue
        ssim_a.append(methods[method_a]["ssim"])
        ssim_b.append(methods[method_b]["ssim"])
        lpips_a.append(methods[method_a]["lpips"])
        lpips_b.append(methods[method_b]["lpips"])

    if not ssim_a:
        raise ValueError(
            f"No paired frames found for methods '{method_a}' and '{method_b}'"
        )

    ssim_test = paired_ttest_with_ci(ssim_a, ssim_b)
    lpips_test = paired_ttest_with_ci(lpips_a, lpips_b)

    return {
        "comparison": f"{method_a}_vs_{method_b}",
        "n_paired_frames": len(ssim_a),
        "ssim_paired_ttest": ssim_test,
        "lpips_paired_ttest": lpips_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired t-test on SSIM and LPIPS between anonymisation methods."
    )
    parser.add_argument(
        "--results",
        default="outputs/perceptual_results.json",
        help="Path to per-frame perceptual results JSON.",
    )
    parser.add_argument(
        "--method-a", default="blur", help="First method name."
    )
    parser.add_argument(
        "--method-b", default="pixelate", help="Second method name."
    )
    parser.add_argument(
        "--output",
        default="outputs/perceptual_statistical_tests.json",
        help="Path to save the statistical test results JSON.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    payload = run_statistical_tests(
        results_path, method_a=args.method_a, method_b=args.method_b
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(json.dumps(payload, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
