#!/usr/bin/env python3

"""Materialise the locked Phase 3 FID subset manifest from the master manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subset_definitions import FID_SUBSET, no_overlap_check, resolve_subset


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "03_fid_source_50000.csv"
DEV_SET_PATH = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"
CALIBRATION_SET_PATH = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "02_calibration_200.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def build_fid_subset(manifest_path: Path) -> pd.DataFrame:
    """Resolve the FID subset and remove overlaps with locked earlier subsets."""
    manifest_df = pd.read_csv(manifest_path)
    blocked_paths: set[str] = set()
    for path in (DEV_SET_PATH, CALIBRATION_SET_PATH):
        if path.exists():
            blocked_df = pd.read_csv(path)
            blocked_paths.update(blocked_df["relative_path"].astype(str).tolist())
    filtered_df = manifest_df[~manifest_df["relative_path"].astype(str).isin(blocked_paths)].reset_index(drop=True)
    fid_df = resolve_subset(FID_SUBSET, filtered_df)
    subset_frames: dict[str, pd.DataFrame] = {"FID_SUBSET": fid_df}
    if DEV_SET_PATH.exists():
        subset_frames["DEV_SET"] = pd.read_csv(DEV_SET_PATH)
    if CALIBRATION_SET_PATH.exists():
        subset_frames["CALIBRATION_SET"] = pd.read_csv(CALIBRATION_SET_PATH)
    no_overlap_check(subset_frames)
    fid_df["selection_seed"] = int(FID_SUBSET["seed"])
    return fid_df


def main() -> None:
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    fid_df = build_fid_subset(PROJECT_ROOT / args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fid_df.to_csv(output_path, index=False)
    print(f"Saved FID subset to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
