#!/usr/bin/env python3

"""Compute the Phase 3 unanonymised WebP self-FID baseline from exported face crops."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.fid_evaluator import (
    InceptionFeatureExtractor,
    compute_activation_statistics,
    frechet_distance,
)
from src.utils.compute_policy import build_compute_policy


DEFAULT_METADATA = PROJECT_ROOT / "outputs" / "fid_webp_baseline_crops.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fid_webp_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    parser.add_argument("--batch-size", type=int, default=0, help="0 means auto from compute policy")
    parser.add_argument("--device", default="", help="Empty means auto from compute policy")
    return parser.parse_args()


def load_split_paths(metadata_path: Path) -> tuple[list[Path], list[Path]]:
    """Load reference and comparison crop paths from exported metadata."""
    reference_paths: list[Path] = []
    comparison_paths: list[Path] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            crop_path = PROJECT_ROOT / row["crop_path"]
            if row["split"] == "reference":
                reference_paths.append(crop_path)
            elif row["split"] == "comparison":
                comparison_paths.append(crop_path)
    if len(reference_paths) < 2 or len(comparison_paths) < 2:
        raise ValueError("Need at least two crops in both reference and comparison splits")
    return reference_paths, comparison_paths


def main() -> None:
    args = parse_args()
    policy = build_compute_policy()
    metadata_path = PROJECT_ROOT / args.metadata
    output_path = PROJECT_ROOT / args.output
    reference_paths, comparison_paths = load_split_paths(metadata_path)
    device = args.device or policy.device
    batch_size = args.batch_size or policy.fid_batch_size

    extractor = InceptionFeatureExtractor(device=device)
    reference_features = extractor.extract(reference_paths, batch_size=batch_size)
    comparison_features = extractor.extract(comparison_paths, batch_size=batch_size)
    mu_ref, sigma_ref = compute_activation_statistics(reference_features)
    mu_cmp, sigma_cmp = compute_activation_statistics(comparison_features)
    fid_value = frechet_distance(mu_ref, sigma_ref, mu_cmp, sigma_cmp)

    payload = {
        "metric": "fid",
        "baseline_type": "webp_self_fid_split_half",
        "interpretation": (
            "Split-half FID across disjoint unanonymised CASTLE WebP face crops. "
            "Use later anonymised FID values relative to this baseline."
        ),
        "reference_crop_count": len(reference_paths),
        "comparison_crop_count": len(comparison_paths),
        "total_crop_count": len(reference_paths) + len(comparison_paths),
        "fid_value": float(fid_value),
        "metadata_source": str(metadata_path.relative_to(PROJECT_ROOT)),
        "device": device,
        "batch_size": int(batch_size),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
