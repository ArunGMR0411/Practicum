#!/usr/bin/env python3

"""Build the CASTLE 300-frame development set with semantic condition coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subset_building import (
    CONDITION_LABELS,
    build_face_condition_candidate_banks,
    build_condition_candidate_bank,
    build_screen_candidate_bank,
    choose_condition_frame,
    load_manifest,
    merge_with_analysis_cache,
    stream_day_sample,
)
from src.data.subset_definitions import DEV_SET, no_overlap_check


DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "01_development_300.csv"
DEFAULT_INSPECTION_NOTE = PROJECT_ROOT / "outputs" / "dev_set_inspection_note.txt"
CALIBRATION_SET_PATH = PROJECT_ROOT / "outputs" / "submission_evidence" / "01_protocol" / "supporting_protocols" / "02_calibration_200.csv"
RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"
CACHE_PATH = PROJECT_ROOT / "outputs" / "cache" / "castle2024_selection_support" / "subset_analysis_cache.csv"
REFINEMENT_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "castle2024_selection_support" / "dev_face_refinement_cache"
DEV_BANK_DIR = PROJECT_ROOT / "outputs" / "cache" / "castle2024_selection_support" / "dev_candidate_banks"
SEED = int(DEV_SET["seed"])
TARGET_COUNT = int(DEV_SET["n_samples"])
MIN_PER_CONDITION = 30
INITIAL_POOL_SIZE = 700
POOL_INCREMENT = 300
ANALYSIS_WIDTH = 192
FACE_ANALYSIS_WIDTH = 640
FACE_CONDITIONS = ["small_face", "extreme_pose", "downward_view", "multiple_faces", "no_face"]
FACE_BANK_TOP_K = 90
GENERIC_BANK_TOP_K = 120


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


def normalise_metadata_columns(dev_df: pd.DataFrame) -> pd.DataFrame:
    """Repair missing metadata fields from relative paths before saving."""
    dev_df = dev_df.copy()
    derived_rows = dev_df["relative_path"].map(derive_metadata_from_relative_path).tolist()
    for column in ["day_id", "camera_stream_id", "participant_id", "view_type"]:
        if column not in dev_df.columns:
            dev_df[column] = [row[column] for row in derived_rows]
            continue
        repaired = []
        for original, derived in zip(dev_df[column].tolist(), derived_rows, strict=False):
            repaired.append(derived[column] if is_missing(original) else original)
        dev_df[column] = repaired
    return dev_df


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for development-set generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)))
    parser.add_argument(
        "--inspection-note",
        default=str(DEFAULT_INSPECTION_NOTE.relative_to(PROJECT_ROOT)),
    )
    return parser.parse_args()


def stable_order_key(value: str) -> tuple[int, str]:
    """Return a deterministic sort key derived from SHA-256."""
    digest = hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16), value


def exclude_existing_overlap(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are already assigned to the calibration set if it exists."""
    if not CALIBRATION_SET_PATH.exists():
        return manifest_df
    calibration_df = pd.read_csv(CALIBRATION_SET_PATH)
    if "relative_path" not in calibration_df.columns:
        return manifest_df
    return manifest_df[~manifest_df["relative_path"].isin(calibration_df["relative_path"])].reset_index(drop=True)


def choose_candidates(manifest_df: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Choose a reproducible candidate pool with stream-day anchors and diverse extras."""
    anchors = stream_day_sample(manifest_df, seed=SEED)
    remaining = manifest_df[~manifest_df["relative_path"].isin(anchors["relative_path"])].copy()
    ordered_remaining = remaining.sort_values(
        by="relative_path",
        key=lambda s: s.map(stable_order_key),
    ).reset_index(drop=True)
    extra_count = max(pool_size - len(anchors), 0)
    extras = ordered_remaining.head(extra_count)
    return pd.concat([anchors, extras], ignore_index=True).drop_duplicates("relative_path").reset_index(drop=True)


def build_dev_set(analysed_df: pd.DataFrame) -> pd.DataFrame:
    """Select 300 frames with semantic condition coverage and stream-day anchors."""
    selected_rows: list[pd.Series] = []
    selected_paths: set[str] = set()

    def add_row(row: pd.Series, condition_label: str) -> None:
        if row["relative_path"] in selected_paths:
            return
        row_copy = row.copy()
        row_copy["condition_label"] = condition_label
        row_copy["selection_seed"] = SEED
        selected_rows.append(row_copy)
        selected_paths.add(row["relative_path"])

    for condition in CONDITION_LABELS:
        eligible = analysed_df[
            analysed_df.apply(lambda row: choose_condition_frame(row, condition), axis=1)
        ].copy()
        eligible = eligible.sort_values(
            by="relative_path",
            key=lambda s: s.map(stable_order_key),
        ).reset_index(drop=True)
        if len(eligible) < MIN_PER_CONDITION:
            raise ValueError(f"Condition {condition} has only {len(eligible)} eligible frames")
        for _, row in eligible.iterrows():
            if sum(item["condition_label"] == condition for item in selected_rows) >= MIN_PER_CONDITION:
                break
            add_row(row, condition)

    anchors = stream_day_sample(analysed_df, seed=SEED)
    for _, row in anchors.iterrows():
        if len(selected_rows) >= TARGET_COUNT:
            break
        if row["relative_path"] in selected_paths:
            continue
        fallback_label = row["condition_matches"][0] if row["condition_matches"] else "no_face"
        add_row(row, fallback_label)

    diverse_remainder = analysed_df[~analysed_df["relative_path"].isin(selected_paths)].copy()
    diverse_remainder = diverse_remainder.sort_values(
        by="relative_path",
        key=lambda s: s.map(stable_order_key),
    ).reset_index(drop=True)
    for _, row in diverse_remainder.iterrows():
        if len(selected_rows) >= TARGET_COUNT:
            break
        if not row["condition_matches"]:
            continue
        add_row(row, row["condition_matches"][0])

    if len(selected_rows) < TARGET_COUNT:
        remaining = analysed_df[~analysed_df["relative_path"].isin(selected_paths)].copy()
        remaining = remaining.sort_values(
            by="relative_path",
            key=lambda s: s.map(stable_order_key),
        ).reset_index(drop=True)
        for _, row in remaining.iterrows():
            if len(selected_rows) >= TARGET_COUNT:
                break
            add_row(row, "no_face" if row["no_face_flag"] else (row["condition_matches"][0] if row["condition_matches"] else "motion_blur"))

    if len(selected_rows) != TARGET_COUNT:
        raise ValueError(f"Failed to build a {TARGET_COUNT}-frame development set; got {len(selected_rows)}")

    dev_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    verify_condition_counts(dev_df)
    return dev_df


def build_dev_set_from_banks(
    analysed_df: pd.DataFrame,
    face_banks: dict[str, pd.DataFrame],
    generic_banks: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Assemble the development set from precomputed condition-specific banks."""
    selected_rows: list[pd.Series] = []
    selected_paths: set[str] = set()

    def add_from_bank(condition_label: str, bank_df: pd.DataFrame) -> None:
        if bank_df.empty:
            return
        current = sum(1 for row in selected_rows if row["condition_label"] == condition_label)
        needed = max(MIN_PER_CONDITION - current, 0)
        if needed <= 0:
            return
        ordered = bank_df.sort_values(
            by="relative_path",
            key=lambda s: s.map(stable_order_key),
        ).reset_index(drop=True)
        for _, row in ordered.iterrows():
            if row["relative_path"] in selected_paths:
                continue
            row_copy = row.copy()
            row_copy["condition_label"] = condition_label
            row_copy["selection_seed"] = SEED
            selected_rows.append(row_copy)
            selected_paths.add(str(row["relative_path"]))
            needed -= 1
            if needed <= 0:
                break

    for condition in CONDITION_LABELS:
        bank_df = face_banks.get(condition)
        if bank_df is None or bank_df.empty:
            bank_df = generic_banks.get(condition, pd.DataFrame())
        add_from_bank(condition, bank_df)

    dev_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    verify_condition_counts(dev_df)

    anchors = stream_day_sample(analysed_df, seed=SEED)
    anchor_rows = []
    for _, row in anchors.iterrows():
        if row["relative_path"] in selected_paths:
            continue
        fallback_label = row["condition_matches"][0] if row["condition_matches"] else "no_face"
        row_copy = row.copy()
        row_copy["condition_label"] = fallback_label
        row_copy["selection_seed"] = SEED
        anchor_rows.append(row_copy)
        selected_paths.add(str(row["relative_path"]))
    if anchor_rows:
        dev_df = pd.concat([dev_df, pd.DataFrame(anchor_rows)], ignore_index=True)

    diverse_remainder = analysed_df[~analysed_df["relative_path"].isin(selected_paths)].copy()
    diverse_remainder = diverse_remainder.sort_values(
        by="relative_path",
        key=lambda s: s.map(stable_order_key),
    ).reset_index(drop=True)
    filler_rows = []
    for _, row in diverse_remainder.iterrows():
        if len(dev_df) + len(filler_rows) >= TARGET_COUNT:
            break
        if not row["condition_matches"]:
            continue
        row_copy = row.copy()
        row_copy["condition_label"] = row["condition_matches"][0]
        row_copy["selection_seed"] = SEED
        filler_rows.append(row_copy)
    if filler_rows:
        dev_df = pd.concat([dev_df, pd.DataFrame(filler_rows)], ignore_index=True)

    if len(dev_df) < TARGET_COUNT:
        remaining = analysed_df[~analysed_df["relative_path"].isin(dev_df["relative_path"])].copy()
        remaining = remaining.sort_values(
            by="relative_path",
            key=lambda s: s.map(stable_order_key),
        ).reset_index(drop=True)
        fallback_rows = []
        for _, row in remaining.iterrows():
            if len(dev_df) + len(fallback_rows) >= TARGET_COUNT:
                break
            row_copy = row.copy()
            row_copy["condition_label"] = "no_face" if row["no_face_flag"] else (row["condition_matches"][0] if row["condition_matches"] else "motion_blur")
            row_copy["selection_seed"] = SEED
            fallback_rows.append(row_copy)
        if fallback_rows:
            dev_df = pd.concat([dev_df, pd.DataFrame(fallback_rows)], ignore_index=True)

    if len(dev_df) != TARGET_COUNT:
        raise ValueError(f"Failed to build a {TARGET_COUNT}-frame development set; got {len(dev_df)}")

    return dev_df.reset_index(drop=True)


def load_saved_candidate_banks() -> dict[str, pd.DataFrame]:
    """Load persisted dev candidate banks when they have been materialized already."""
    banks: dict[str, pd.DataFrame] = {}
    for condition in CONDITION_LABELS:
        bank_path = DEV_BANK_DIR / f"{condition}_bank.csv"
        if not bank_path.exists():
            return {}
        banks[condition] = pd.read_csv(bank_path)
    return banks


def combine_candidate_banks(primary_df: pd.DataFrame, fallback_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer persisted bank rows but top up from fallback candidates when needed."""
    if primary_df.empty:
        return fallback_df.reset_index(drop=True)
    if fallback_df.empty:
        return primary_df.reset_index(drop=True)
    combined = pd.concat([primary_df, fallback_df], ignore_index=True)
    return combined.drop_duplicates("relative_path", keep="first").reset_index(drop=True)


def overlay_refined_face_rows(
    analysed_df: pd.DataFrame,
    raw_root: Path,
) -> pd.DataFrame:
    """Replace only face-condition candidate rows with targeted MTCNN-refined analysis."""
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "seed": SEED,
                "top_k": FACE_BANK_TOP_K,
                "analysis_width": FACE_ANALYSIS_WIDTH,
                "conditions": FACE_CONDITIONS,
                "relative_paths": sorted(analysed_df["relative_path"].astype(str).tolist()),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = REFINEMENT_CACHE_DIR / f"{cache_key}.csv"
    if cache_path.exists():
        refined_df = pd.read_csv(cache_path)
    else:
        face_banks = build_face_condition_candidate_banks(
            analysed_df,
            raw_root=raw_root,
            conditions=FACE_CONDITIONS,
            top_k=FACE_BANK_TOP_K,
            seed=SEED,
            analysis_width=FACE_ANALYSIS_WIDTH,
        )
        refined_parts = [bank for bank in face_banks.values() if not bank.empty]
        if not refined_parts:
            return analysed_df
        refined_df = pd.concat(refined_parts, ignore_index=True).drop_duplicates("relative_path", keep="last")
        REFINEMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        refined_df.to_csv(cache_path, index=False)
    base_df = analysed_df[~analysed_df["relative_path"].isin(refined_df["relative_path"])].copy()
    combined_df = pd.concat([base_df, refined_df], ignore_index=True)
    return combined_df.drop_duplicates("relative_path", keep="last").reset_index(drop=True)


def verify_condition_counts(dev_df: pd.DataFrame) -> None:
    """Reject any development set that underrepresents a required condition."""
    counts = dev_df["condition_label"].value_counts().to_dict()
    underrepresented = {
        label: counts.get(label, 0)
        for label in CONDITION_LABELS
        if counts.get(label, 0) < MIN_PER_CONDITION
    }
    if underrepresented:
        raise ValueError(f"Underrepresented hard conditions remain unresolved: {underrepresented}")


def write_inspection_note(dev_df: pd.DataFrame, note_path: Path) -> None:
    """Record the locked development-set inspection note."""
    shared_streams = sorted(set(dev_df["camera_stream_id"]))
    shared_days = sorted(set(dev_df["day_or_session_id"]))
    note_lines = [
        "Development set inspection note",
        f"Seed: {SEED}",
        "The active data surface records locked development-set coverage directly from the generated development set.",
        f"Shared stream IDs: {', '.join(shared_streams)}",
        f"Shared day IDs: {', '.join(shared_days)}",
        "The development set enforces the required minimum of 30 frames per condition label.",
        json.dumps(dev_df["condition_label"].value_counts().sort_index().to_dict(), indent=2),
    ]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")


def generate_dev_set(manifest_path: Path) -> pd.DataFrame:
    """Build the full development set DataFrame from the project manifest."""
    manifest_df = exclude_existing_overlap(load_manifest(manifest_path))
    saved_banks = load_saved_candidate_banks()
    pool_size = INITIAL_POOL_SIZE
    while True:
        candidates = choose_candidates(manifest_df, pool_size=pool_size)
        analysed_df = merge_with_analysis_cache(
            candidates,
            cache_path=CACHE_PATH,
            raw_root=RAW_ROOT,
            analysis_width=ANALYSIS_WIDTH,
            max_workers=6,
        )
        if saved_banks:
            face_banks = {
                condition: saved_banks.get(condition, pd.DataFrame())
                for condition in FACE_CONDITIONS
            }
        else:
            analysed_df = overlay_refined_face_rows(analysed_df, raw_root=RAW_ROOT)
            face_banks = build_face_condition_candidate_banks(
                analysed_df,
                raw_root=RAW_ROOT,
                conditions=FACE_CONDITIONS,
                top_k=FACE_BANK_TOP_K,
                seed=SEED,
                analysis_width=FACE_ANALYSIS_WIDTH,
            )
        generic_banks = {
            "motion_blur": build_condition_candidate_bank(
                analysed_df,
                "motion_blur",
                top_k=GENERIC_BANK_TOP_K,
                seed=SEED,
            ),
            "visible_text": build_condition_candidate_bank(
                analysed_df,
                "visible_text",
                top_k=GENERIC_BANK_TOP_K,
                seed=SEED,
            ),
            "visible_screen": build_screen_candidate_bank(
                analysed_df,
                top_k=GENERIC_BANK_TOP_K,
                seed=SEED,
            ),
        }
        if saved_banks:
            face_banks = {
                condition: combine_candidate_banks(saved_banks.get(condition, pd.DataFrame()), face_banks.get(condition, pd.DataFrame()))
                for condition in FACE_CONDITIONS
            }
            generic_banks = {
                condition: combine_candidate_banks(saved_banks.get(condition, pd.DataFrame()), generic_banks.get(condition, pd.DataFrame()))
                for condition in generic_banks
            }
        try:
            return build_dev_set_from_banks(analysed_df, face_banks=face_banks, generic_banks=generic_banks)
        except ValueError:
            pool_size += POOL_INCREMENT
            if pool_size > len(manifest_df):
                raise


def main() -> None:
    """Generate, validate, and save the reproducible 300-frame dev set."""
    args = parse_args()
    output_path = PROJECT_ROOT / args.output
    note_path = PROJECT_ROOT / args.inspection_note
    dev_df = normalise_metadata_columns(generate_dev_set(PROJECT_ROOT / args.manifest))
    no_overlap_check({"DEV_SET": dev_df})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_df.to_csv(output_path, index=False)
    write_inspection_note(dev_df, note_path)
    print(f"Saved development set to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
