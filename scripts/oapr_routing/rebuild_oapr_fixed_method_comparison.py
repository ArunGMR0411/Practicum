#!/usr/bin/env python3
"""Rebuild OAPR fixed-method comparisons from canonical RQ2 metrics."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OAPR_DIR = ROOT / "outputs" / "submission_evidence" / "05_oapr"
OAPR_SUMMARY = OAPR_DIR / "12_oapr_full_metric_summary.csv"
RQ2_TABLE = ROOT / "outputs" / "submission_evidence" / "03_anonymisation" / "01_all_methods_comparison.csv"

COMPARABLE_METHODS = [
    "blur",
    "pixelate",
    "solid_mask_black",
    "layered_blur_downscale_noise",
    "nullface",
    "diffusion_low_step",
    "reverse_personalization",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", ""]
    if rows:
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    else:
        lines.append("No rows available.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(value: Any) -> float | None:
    try:
        if value in {None, "", "not_available"}:
            return None
        return float(value)
    except Exception:
        return None


def cmp_lower_better(oapr: float | None, fixed: float | None, eps: float = 1e-12) -> str:
    if oapr is None or fixed is None:
        return "not_available"
    if oapr < fixed - eps:
        return "oapr_better"
    if oapr > fixed + eps:
        return "fixed_better"
    return "tie"


def cmp_higher_better(oapr: float | None, fixed: float | None, eps: float = 1e-12) -> str:
    if oapr is None or fixed is None:
        return "not_available"
    if oapr > fixed + eps:
        return "oapr_better"
    if oapr < fixed - eps:
        return "fixed_better"
    return "tie"


def main() -> None:
    rq2_rows = {row["method"]: row for row in read_csv(RQ2_TABLE)}
    missing = [method for method in COMPARABLE_METHODS if method not in rq2_rows]
    if missing:
        raise SystemExit(f"Missing required comparable methods in {RQ2_TABLE}: {missing}")

    oapr_rows = read_csv(OAPR_SUMMARY)
    comparisons: list[dict[str, Any]] = []
    wins: list[dict[str, Any]] = []

    for oapr in oapr_rows:
        objective = oapr["objective_mode"]
        counter: Counter[str] = Counter()
        for method in COMPARABLE_METHODS:
            fixed = rq2_rows[method]
            o_ssim = f(oapr.get("SSIM_mean"))
            x_ssim = f(fixed.get("SSIM_mean"))
            o_lpips = f(oapr.get("LPIPS_mean"))
            x_lpips = f(fixed.get("LPIPS_mean"))
            o_ada = f(oapr.get("AdaFace_reid_rate"))
            x_ada = f(fixed.get("AdaFace_reid_rate"))
            o_arc = f(oapr.get("ArcFace_reid_rate"))
            x_arc = f(fixed.get("ArcFace_reid_rate"))

            ssim_cmp = cmp_higher_better(o_ssim, x_ssim)
            lpips_cmp = cmp_lower_better(o_lpips, x_lpips)
            ada_cmp = cmp_lower_better(o_ada, x_ada)
            arc_cmp = cmp_lower_better(o_arc, x_arc)
            for metric_name, outcome in {
                "SSIM": ssim_cmp,
                "LPIPS": lpips_cmp,
                "AdaFace": ada_cmp,
                "ArcFace": arc_cmp,
            }.items():
                counter[f"{metric_name}_{outcome}"] += 1

            comparisons.append(
                {
                    "objective_mode": objective,
                    "fixed_method": method,
                    "OAPR_SSIM_mean": o_ssim if o_ssim is not None else "not_available",
                    "fixed_SSIM_mean": x_ssim if x_ssim is not None else "not_available",
                    "SSIM_delta_OAPR_minus_fixed": (o_ssim - x_ssim) if o_ssim is not None and x_ssim is not None else "not_available",
                    "SSIM_result": ssim_cmp,
                    "OAPR_LPIPS_mean": o_lpips if o_lpips is not None else "not_available",
                    "fixed_LPIPS_mean": x_lpips if x_lpips is not None else "not_available",
                    "LPIPS_delta_OAPR_minus_fixed": (o_lpips - x_lpips) if o_lpips is not None and x_lpips is not None else "not_available",
                    "LPIPS_result": lpips_cmp,
                    "OAPR_AdaFace_reid_rate": o_ada if o_ada is not None else "not_available",
                    "fixed_AdaFace_reid_rate": x_ada if x_ada is not None else "not_available",
                    "AdaFace_delta_OAPR_minus_fixed_negative_means_OAPR_better": (o_ada - x_ada) if o_ada is not None and x_ada is not None else "not_available",
                    "AdaFace_result": ada_cmp,
                    "OAPR_ArcFace_reid_rate": o_arc if o_arc is not None else "not_available",
                    "fixed_ArcFace_reid_rate": x_arc if x_arc is not None else "not_available",
                    "ArcFace_delta_OAPR_minus_fixed_negative_means_OAPR_better": (o_arc - x_arc) if o_arc is not None and x_arc is not None else "not_available",
                    "ArcFace_result": arc_cmp,
                    "fixed_method_evidence_level": fixed.get("evidence_level", ""),
                    "fixed_method_limitation": fixed.get("limitation", ""),
                    "comparison_basis": "oapr_actual_routed_outputs_vs_canonical_rq2_metrics",
                    "report_safe_claim": "Objective-specific comparison only; OAPR is not claimed to dominate fixed blur globally.",
                }
            )

        wins.append(
            {
                "objective_mode": objective,
                "privacy_wins_AdaFace_vs_fixed_methods": counter["AdaFace_oapr_better"],
                "privacy_losses_AdaFace_vs_fixed_methods": counter["AdaFace_fixed_better"],
                "privacy_ties_AdaFace_vs_fixed_methods": counter["AdaFace_tie"],
                "privacy_wins_ArcFace_vs_fixed_methods": counter["ArcFace_oapr_better"],
                "privacy_losses_ArcFace_vs_fixed_methods": counter["ArcFace_fixed_better"],
                "privacy_ties_ArcFace_vs_fixed_methods": counter["ArcFace_tie"],
                "utility_wins_SSIM_vs_fixed_methods": counter["SSIM_oapr_better"],
                "utility_losses_SSIM_vs_fixed_methods": counter["SSIM_fixed_better"],
                "utility_ties_SSIM_vs_fixed_methods": counter["SSIM_tie"],
                "utility_wins_LPIPS_vs_fixed_methods": counter["LPIPS_oapr_better"],
                "utility_losses_LPIPS_vs_fixed_methods": counter["LPIPS_fixed_better"],
                "utility_ties_LPIPS_vs_fixed_methods": counter["LPIPS_tie"],
                "interpretation": "Objective-specific routed-output evidence; use to support bounded routing claims only.",
            }
        )

    comparison_fields = list(comparisons[0])
    win_fields = list(wins[0])
    write_csv(OAPR_DIR / "13_oapr_vs_fixed_methods_metric_comparison.csv", comparisons, comparison_fields)
    write_csv(OAPR_DIR / "14_oapr_objective_specific_wins.csv", wins, win_fields)

    privacy_first = next(row for row in oapr_rows if row["objective_mode"] == "privacy_first")
    failure_avoidance = next(row for row in oapr_rows if row["objective_mode"] == "failure_avoidance")
    decision = f"""# OAPR Final Decision

## Status

`BOUNDED_ACTUAL_ROUTED_OUTPUT_EVIDENCE`

## Evidence Added

- Objective modes evaluated: `{len(oapr_rows)}`
- Frames per objective: `500`
- Fixed methods compared: `{', '.join(COMPARABLE_METHODS)}`
- Comparison source: `outputs/03_anonymisation/01_all_methods_comparison.csv`
- OAPR metric source: `outputs/05_oapr/12_oapr_full_metric_summary.csv`

## Key Result

OAPR is not globally dominant over fixed blur or every fixed method. Its value is objective-specific routing:

- `privacy_first` reaches AdaFace Re-ID `{privacy_first['AdaFace_reid_rate']}` and ArcFace Re-ID `{privacy_first['ArcFace_reid_rate']}` by selecting solid masking for face-positive cases.
- `failure_avoidance` reaches AdaFace Re-ID `{failure_avoidance['AdaFace_reid_rate']}` and ArcFace Re-ID `{failure_avoidance['ArcFace_reid_rate']}` while preserving higher SSIM than privacy-first routing.
- Utility-oriented modes remain close to blur because they mostly select blur/copy and do not force stronger obfuscation where not needed.

## Bounded Claim

Use OAPR as a bounded, auditable policy layer. It selects among evidence-supported deterministic methods, avoids quality-limited advanced methods, and preserves fixed blur as the practical default when objective evidence does not justify a stronger action. Do not claim global superiority over fixed blur.
"""


if __name__ == "__main__":
    main()
