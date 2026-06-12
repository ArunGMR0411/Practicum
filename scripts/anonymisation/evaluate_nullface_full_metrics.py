#!/usr/bin/env python3
"""Summarise existing full-500 NullFace outputs and available metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    import numpy as np
    from skimage.metrics import structural_similarity

    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False
    np = None  # type: ignore[assignment]
    structural_similarity = None  # type: ignore[assignment]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    lines = ["# NullFace Full 500 Metric Summary", ""]
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


def load_runtime_total(runtime_path: Path) -> tuple[str, str]:
    rows = read_csv(runtime_path)
    for row in rows:
        if row.get("segment") == "compute_constrained_remaining_298":
            return row.get("mean_runtime_seconds", "not_available"), row.get("total_runtime_seconds", "not_available")
    return "not_available", "not_available"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/03_anonymisation/03_nullface/01_nullface_full_500_manifest.csv")
    parser.add_argument("--raw-root", default="data/castle2024/raw")
    parser.add_argument("--runtime-csv", default="outputs/03_anonymisation/03_nullface/03_nullface_runtime_summary.csv")
    parser.add_argument("--failure-csv", default="outputs/03_anonymisation/03_nullface/04_nullface_failure_log.csv")
    parser.add_argument("--summary-json", default="outputs/03_anonymisation/03_nullface/02_nullface_full_500_summary.json")
    parser.add_argument("--output-dir", default="outputs/runs/nullface_metric_refresh")
    parser.add_argument("--eval-scale", type=float, default=0.25)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    raw_root = PROJECT_ROOT / args.raw_root
    runtime_path = PROJECT_ROOT / args.runtime_csv
    failure_path = PROJECT_ROOT / args.failure_csv
    summary_path = PROJECT_ROOT / args.summary_json
    output_dir = PROJECT_ROOT / args.output_dir

    rows = read_csv(manifest_path)
    failure_rows = read_csv(failure_path)
    ssims: list[float] = []
    n_success = 0
    n_output_missing = 0
    n_raw_missing = 0
    n_zero_box = 0

    for row in rows:
        relative_path = row.get("relative_path", "")
        output_path = PROJECT_ROOT / row.get("output_path", "")
        raw_path = raw_root / relative_path
        if int(row.get("boxes_processed") or 0) == 0:
            n_zero_box += 1
        if not output_path.exists():
            n_output_missing += 1
            continue
        if not raw_path.exists():
            n_raw_missing += 1
            continue
        n_success += 1
        ssim = compute_ssim_if_available(raw_path, output_path, args.eval_scale)
        if ssim is not None:
            ssims.append(ssim)

    runtime_mean, runtime_total = load_runtime_total(runtime_path)
    summary_json: dict[str, Any] = {}
    if summary_path.exists():
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))

    n_input = len(rows)
    explicit_failures = [
        row
        for row in failure_rows
        if row.get("failure_label") not in {"", "no_auto_failure_flag", "no_face_processed_copy_through"}
    ]
    n_failure = len(explicit_failures) + n_output_missing + n_raw_missing
    row = {
        "method": "nullface",
        "subset": "anonymisation_eval_subset",
        "n_input_frames": n_input,
        "n_success": n_success,
        "n_failure": n_failure,
        "frames_with_zero_boxes": n_zero_box or summary_json.get("frames_with_zero_boxes", "not_available"),
        "SSIM_mean": round(statistics.mean(ssims), 12) if ssims else "not_available",
        "LPIPS_mean": "not_available",
        "AdaFace_metric": "not_available",
        "ArcFace_metric": "not_available",
        "runtime_mean": runtime_mean,
        "runtime_total": runtime_total,
        "evidence_level": "partial_comparable_local" if ssims else "execution_only",
        "limitation": "LPIPS and Re-ID require constrained-compute metric pass; execution environment lacks skimage/lpips/torch"
        if not HAS_SKIMAGE
        else "LPIPS and Re-ID require a constrained-compute metric pass; runtime is partial because retained shards lack full per-frame timing",
        "report_safe_claim": "NullFace has full 500-frame output coverage, but remains close-to-full-comparable until LPIPS and Re-ID are computed.",
    }
    fields = [
        "method",
        "subset",
        "n_input_frames",
        "n_success",
        "n_failure",
        "frames_with_zero_boxes",
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
    write_csv(output_dir / "nullface_full_500_metric_summary.csv", [row], fields)


if __name__ == "__main__":
    main()
