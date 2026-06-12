#!/usr/bin/env python3
"""Summarise stronger classical baseline execution and available metrics.

This script is intentionally conservative: it computes runtime/failure coverage
from the generated manifest and only computes SSIM when the execution environment
has the required dependency. LPIPS and Re-ID are not fabricated.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "runs" / "stronger_baselines"

try:
    import numpy as np
    from skimage.metrics import structural_similarity

    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False
    np = None  # type: ignore[assignment]
    structural_similarity = None  # type: ignore[assignment]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], title: str) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows available.")
    else:
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_ssim_if_available(original_path: Path, output_path: Path, eval_scale: float) -> float | None:
    if not HAS_SKIMAGE:
        return None
    with Image.open(original_path) as original, Image.open(output_path) as anonymised:
        original = original.convert("RGB")
        anonymised = anonymised.convert("RGB")
        if anonymised.size != original.size:
            anonymised = anonymised.resize(original.size, Image.Resampling.LANCZOS)
        if eval_scale != 1.0:
            size = (
                max(1, int(round(original.width * eval_scale))),
                max(1, int(round(original.height * eval_scale))),
            )
            original = original.resize(size, Image.Resampling.LANCZOS)
            anonymised = anonymised.resize(size, Image.Resampling.LANCZOS)
        original_np = np.array(original)  # type: ignore[union-attr]
        anonymised_np = np.array(anonymised)  # type: ignore[union-attr]
        min_dim = min(original_np.shape[0], original_np.shape[1])
        win_size = 7 if min_dim >= 7 else min_dim if min_dim % 2 else min_dim - 1
        if win_size < 3:
            return 1.0 if (original_np == anonymised_np).all() else 0.0
        return float(
            structural_similarity(  # type: ignore[misc]
                original_np,
                anonymised_np,
                channel_axis=-1,
                data_range=255,
                win_size=win_size,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/runs/stronger_baselines/manifest.csv")
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--eval-scale", type=float, default=0.25)
    parser.add_argument("--output-dir", default="outputs/runs/stronger_baselines")
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    raw_root = PROJECT_ROOT / args.raw_root
    output_dir = PROJECT_ROOT / args.output_dir
    rows = read_rows(manifest_path) if manifest_path.exists() else []

    execution_fields = [
        "relative_path",
        "method",
        "output_path",
        "boxes_processed",
        "runtime_seconds",
        "status",
        "output_exists",
        "raw_exists",
    ]
    execution_rows: list[dict[str, Any]] = []
    ssim_by_method: dict[str, list[float]] = defaultdict(list)
    runtime_by_method: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "success": 0, "failure": 0})

    for row in rows:
        method = row.get("method", "")
        relative_path = row.get("relative_path", "")
        output_path = PROJECT_ROOT / row.get("output_path", "")
        raw_path = raw_root / relative_path
        output_exists = output_path.exists()
        raw_exists = raw_path.exists()
        status = row.get("status", "")
        counts[method]["input"] += 1
        if status == "ok" and output_exists and raw_exists:
            counts[method]["success"] += 1
            runtime_by_method[method].append(float(row.get("runtime_seconds") or 0.0))
            ssim = compute_ssim_if_available(raw_path, output_path, args.eval_scale)
            if ssim is not None:
                ssim_by_method[method].append(ssim)
        else:
            counts[method]["failure"] += 1
        execution_rows.append(
            {
                **row,
                "output_exists": output_exists,
                "raw_exists": raw_exists,
            }
        )

    write_csv(output_dir / "stronger_baseline_execution_manifest.csv", execution_rows, execution_fields)

    runtime_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for method in sorted(counts):
        runtimes = runtime_by_method.get(method, [])
        ssims = ssim_by_method.get(method, [])
        runtime_rows.append(
            {
                "method": method,
                "n_input_frames": counts[method]["input"],
                "n_success": counts[method]["success"],
                "n_failure": counts[method]["failure"],
                "runtime_mean": round(statistics.mean(runtimes), 6) if runtimes else "not_available",
                "runtime_total": round(sum(runtimes), 6) if runtimes else "not_available",
                "evidence_level": "bounded_runtime" if counts[method]["success"] else "incomplete",
            }
        )
        metric_rows.append(
            {
                "method": method,
                "subset": "anonymisation_eval_subset",
                "n_input_frames": counts[method]["input"],
                "n_success": counts[method]["success"],
                "n_failure": counts[method]["failure"],
                "SSIM_mean": round(statistics.mean(ssims), 12) if ssims else "not_available",
                "LPIPS_mean": "not_available",
                "AdaFace_metric": "not_available",
                "ArcFace_metric": "not_available",
                "runtime_mean": round(statistics.mean(runtimes), 6) if runtimes else "not_available",
                "runtime_total": round(sum(runtimes), 6) if runtimes else "not_available",
                "evidence_level": "partial_comparable_local" if ssims else "execution_only",
                "limitation": "LPIPS and Re-ID require constrained-compute metric pass; execution environment lacks skimage/lpips/torch"
                if not HAS_SKIMAGE
                else "LPIPS and Re-ID require constrained-compute metric pass",
                "report_safe_claim": "Generated on the locked 500-frame subset; full privacy comparison is pending LPIPS and Re-ID metrics.",
            }
        )

    runtime_fields = [
        "method",
        "n_input_frames",
        "n_success",
        "n_failure",
        "runtime_mean",
        "runtime_total",
        "evidence_level",
    ]
    metric_fields = [
        "method",
        "subset",
        "n_input_frames",
        "n_success",
        "n_failure",
        "SSIM_mean",
        "LPIPS_mean",
        "AdaFace_metric",
        "ArcFace_metric",
        "runtime_mean",
        "runtime_total",
        "evidence_level",
        "limitation",
        "report_safe_claim",
    ]
    write_csv(output_dir / "stronger_baseline_runtime_summary.csv", runtime_rows, runtime_fields)
    write_csv(output_dir / "stronger_baseline_metric_summary.csv", metric_rows, metric_fields)


if __name__ == "__main__":
    main()
