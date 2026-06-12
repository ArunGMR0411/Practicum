#!/usr/bin/env python3
"""Build one canonical metric summary for a completed comparable anonymiser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    method = args.method
    directory = args.directory
    runtime = pd.read_csv(directory / f"02_{method}_runtime_summary.csv").iloc[0].to_dict()
    perceptual = json.loads((directory / f"04_{method}_perceptual.json").read_text())["summary"][method]
    reid = json.loads((directory / f"06_{method}_reid_summary.json").read_text())
    row = {
        "method": method,
        "n_input_frames": int(runtime["input_frames"]),
        "n_success": int(runtime["successful_frames"]),
        "n_failure": int(runtime["failed_frames"]),
        "face_crops": int(reid["face_crop_count"]),
        "SSIM_mean": perceptual["ssim_mean"],
        "LPIPS_mean": perceptual["lpips_mean"],
        "AdaFace_cosine_mean": reid["adaface_cosine_sim_mean"],
        "AdaFace_reid_rate": reid["adaface_reid_rate"],
        "ArcFace_cosine_mean": reid["arcface_cosine_sim_mean"],
        "ArcFace_reid_rate": reid["arcface_reid_rate"],
        "runtime_mean_seconds": runtime["mean_runtime_seconds"],
        "runtime_total_seconds": runtime["total_runtime_seconds"],
        "peak_vram_gib": runtime["peak_vram_gib"],
        "gpu_name": runtime["gpu_name"],
        "metric_scope": "reviewed_500_frames_full_resolution_1279_face_protocol",
    }
    target_csv = directory / f"07_{method}_metric_summary.csv"
    target_md = directory / f"08_{method}_comparable_summary.md"
    frame = pd.DataFrame([row])
    frame.to_csv(target_csv, index=False)
    target_md.write_text(
        "\n".join([
            f"# {method.upper()} Comparable Summary",
            "",
            "The method completed the reviewed 500-frame protocol. Full-frame perceptual metrics and face-crop AdaFace/ArcFace metrics use the same definitions as the established comparison methods.",
            "",
            frame.to_markdown(index=False),
            "",
        ]),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
