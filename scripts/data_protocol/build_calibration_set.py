#!/usr/bin/env python3

"""Build a 200-frame calibration set with semantic quality-dimension coverage."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subset_building import (
    QUALITY_DIMENSIONS,
    load_manifest,
    merge_with_analysis_cache,
    stream_day_sample,
)
from src.data.subset_definitions import CALIBRATION_SET, no_overlap_check


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "02_calibration_200.csv"
DEV_SET_PATH = PROJECT_ROOT / "outputs" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"
RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "castle2024_selection_support" / "subset_analysis_cache.csv"
SEED = int(CALIBRATION_SET["seed"])
TARGET_COUNT = int(CALIBRATION_SET["n_samples"])
INITIAL_POOL_SIZE = 700
POOL_INCREMENT = 300
ANALYSIS_WIDTH = 192
FACE_ANALYSIS_WIDTH = 640


def is_missing(value: object) -> bool:
    """Return True when a value is empty or NaN-like."""
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan" or text.lower() == "null"


def derive_metadata_from_relative_path(relative_path: str) -> dict[str, str]:
    """Recover stable CASTLE metadata from a relative frame path."""
    parts = Path(relative_path).parts
    if len(parts) < 4:
        return {
            "day_id": "",
            "camera_stream_id": "",
            "participant_id": "",
            "view_type": "",
        }
    day_id = parts[0]
    group = parts[1]
    stream_id = parts[2]
    view_type = "egocentric" if group == "members" else "exocentric"
    participant_id = stream_id if group == "members" else "none"
    return {
        "day_id": day_id,
        "camera_stream_id": stream_id,
        "participant_id": participant_id,
        "view_type": view_type,
    }


def normalise_metadata_columns(calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Repair missing metadata fields from relative paths before saving."""
    calibration_df = calibration_df.copy()
    derived_rows = calibration_df["relative_path"].map(derive_metadata_from_relative_path).tolist()
    for column in ["day_id", "camera_stream_id", "participant_id", "view_type"]:
        if column not in calibration_df.columns:
            calibration_df[column] = [row[column] for row in derived_rows]
            continue
        repaired = []
        for original, derived in zip(calibration_df[column].tolist(), derived_rows, strict=False):
            repaired.append(derived[column] if is_missing(original) else original)
        calibration_df[column] = repaired
    return calibration_df


def parse_args() -> argparse.Namespace:
    """Parse input and output paths for calibration-set generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def stable_hash(value: str) -> int:
    """Return a deterministic integer hash derived from SHA-256."""
    digest = hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def exclude_dev_overlap(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows already assigned to the development set."""
    if not DEV_SET_PATH.exists():
        return manifest_df
    dev_df = pd.read_csv(DEV_SET_PATH)
    if "relative_path" not in dev_df.columns:
        return manifest_df
    return manifest_df[~manifest_df["relative_path"].isin(dev_df["relative_path"])].reset_index(drop=True)


def choose_candidates(manifest_df: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Choose a diverse calibration candidate pool anchored across stream-day groups."""
    anchors = stream_day_sample(manifest_df, seed=SEED)
    remaining = manifest_df[~manifest_df["relative_path"].isin(anchors["relative_path"])].copy()
    ordered_remaining = remaining.sort_values(
        by="relative_path",
        key=lambda s: s.map(lambda value: (stable_hash(value), value)),
    ).reset_index(drop=True)
    extra_count = max(pool_size - len(anchors), 0)
    extras = ordered_remaining.head(extra_count)
    return pd.concat([anchors, extras], ignore_index=True).drop_duplicates("relative_path").reset_index(drop=True)


def sample_calibration_rows(analysed_df: pd.DataFrame) -> pd.DataFrame:
    """Sample a fixed-size calibration set while representing all quality dimensions."""
    face_present_df = analysed_df[analysed_df["face_size"].astype(str).isin(["small", "medium", "large"])].copy()
    if len(face_present_df) < TARGET_COUNT:
        raise ValueError(
            f"Insufficient face-present rows for calibration sampling: {len(face_present_df)} available"
        )

    ordered_df = face_present_df.sort_values(
        by="relative_path",
        key=lambda series: series.map(lambda value: (stable_hash(value), value)),
    ).reset_index(drop=True)

    grouped = (
        ordered_df.groupby(
            QUALITY_DIMENSIONS,
            dropna=False,
        )
        .head(3)
        .drop_duplicates("relative_path")
        .reset_index(drop=True)
    )
    remaining = ordered_df[~ordered_df["relative_path"].isin(grouped["relative_path"])].reset_index(drop=True)
    filler_count = TARGET_COUNT - len(grouped)
    if filler_count < 0:
        selected_df = grouped.head(TARGET_COUNT).copy()
    else:
        selected_df = pd.concat([grouped, remaining.head(filler_count)], ignore_index=True)

    selected_df = selected_df.drop_duplicates("relative_path").reset_index(drop=True)
    if len(selected_df) != TARGET_COUNT:
        raise ValueError(f"Failed to build a {TARGET_COUNT}-frame calibration set; got {len(selected_df)}")
    selected_df["selection_seed"] = SEED
    return selected_df


def verify_quality_representation(calibration_df: pd.DataFrame) -> None:
    """Reject calibration sets that fail to represent any required quality dimension."""
    for column in QUALITY_DIMENSIONS:
        values = calibration_df[column].dropna().astype(str).unique().tolist()
        if len(values) < 3:
            raise ValueError(f"Calibration set does not represent {column} sufficiently: {values}")


def generate_calibration_set(manifest_path: Path) -> pd.DataFrame:
    """Build the full calibration set DataFrame from the project manifest."""
    manifest_df = exclude_dev_overlap(load_manifest(manifest_path))
    pool_size = INITIAL_POOL_SIZE
    while True:
        candidates = choose_candidates(manifest_df, pool_size=pool_size)
        analysed_df = merge_with_analysis_cache(
            candidates,
            cache_path=CACHE_PATH,
            raw_root=RAW_ROOT,
            analysis_width=ANALYSIS_WIDTH,
            max_workers=6,
            face_backend="mtcnn",
            face_analysis_width=FACE_ANALYSIS_WIDTH,
        )
        calibration_df = sample_calibration_rows(analysed_df)
        try:
            verify_quality_representation(calibration_df)
            return calibration_df
        except ValueError:
            pool_size += POOL_INCREMENT
            if pool_size > len(manifest_df):
                raise


def main() -> None:
    """Generate the calibration set with fixed-seed stratification."""
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    calibration_df = normalise_metadata_columns(generate_calibration_set(PROJECT_ROOT / args.manifest))
    subset_frames = {"CALIBRATION_SET": calibration_df}
    if DEV_SET_PATH.exists():
        subset_frames["DEV_SET"] = pd.read_csv(DEV_SET_PATH)
    no_overlap_check(subset_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(output_path, index=False)
    print(f"Saved calibration set to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
