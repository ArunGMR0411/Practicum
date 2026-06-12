#!/usr/bin/env python3

"""Build a manually reviewable text-presence validation subset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score_row(
    row: pd.Series,
    day_counter: Counter[str],
    stream_counter: Counter[str],
    view_counter: Counter[str],
) -> tuple[int, int, int, str]:
    """Prefer underrepresented day, stream, and view combinations."""
    day_id = str(row["day_id"])
    stream_id = str(row["camera_stream_id"])
    view_type = str(row["view_type"])
    return (
        day_counter[day_id],
        stream_counter[stream_id],
        view_counter[view_type],
        str(row["relative_path"]),
    )


def select_balanced_rows(pool: pd.DataFrame, target_count: int, seed: int) -> pd.DataFrame:
    """Select rows with simple diversity-aware greedy balancing."""
    if len(pool) < target_count:
        raise ValueError(
            f"Requested {target_count} rows from pool of size {len(pool)} for text validation."
        )

    shuffled = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    day_counter: Counter[str] = Counter()
    stream_counter: Counter[str] = Counter()
    view_counter: Counter[str] = Counter()
    selected_indices: list[int] = []

    available = shuffled.copy()
    while len(selected_indices) < target_count:
        scored = available.apply(
            score_row,
            axis=1,
            day_counter=day_counter,
            stream_counter=stream_counter,
            view_counter=view_counter,
        )
        best_index = min(scored.items(), key=lambda item: item[1])[0]
        best_row = available.loc[best_index]
        selected_indices.append(int(best_row["__row_id"]))
        day_counter[str(best_row["day_id"])] += 1
        stream_counter[str(best_row["camera_stream_id"])] += 1
        view_counter[str(best_row["view_type"])] += 1
        available = available.drop(index=best_index)

    return shuffled[shuffled["__row_id"].isin(selected_indices)].copy()


def build_review_template(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Create the manual review template with empty annotation columns."""
    template = selected_df.copy()
    template["manual_text_present"] = ""
    template["manual_contains_legible_text"] = ""
    template["manual_notes"] = ""
    template["reviewer_id"] = ""
    template["review_status"] = "pending"
    return template[
        [
            "relative_path",
            "day_id",
            "camera_stream_id",
            "view_type",
            "condition_label",
            "visible_text_flag",
            "sample_bucket",
            "manual_text_present",
            "manual_contains_legible_text",
            "review_status",
            "reviewer_id",
            "manual_notes",
        ]
    ]


def materialise_review_pack(
    selected_df: pd.DataFrame,
    raw_root: Path,
    pack_root: Path,
) -> None:
    """Create a image pack with symlinks for faster manual review."""
    images_root = pack_root / "images"
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True, exist_ok=True)

    for row in selected_df.itertuples(index=False):
        src = raw_root / row.relative_path
        dest = images_root / row.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(src, dest)


def write_readme(pack_root: Path, total_count: int, positive_count: int, negative_count: int) -> None:
    """Write concise review instructions for the validation pack."""
    readme_path = pack_root / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                "Manual text validation subset review instructions",
                "",
                "Purpose:",
                "Review a small trusted subset for binary text-presence validation before further text-detector comparisons.",
                "",
                "How to review:",
                "1. Open each frame from images/ using the relative_path in text_validation_review_template.csv.",
                "2. Set manual_text_present to yes if any visible text is present anywhere in the frame, otherwise no.",
                "3. Set manual_contains_legible_text to yes only if at least some visible text is human-readable, otherwise no.",
                "4. Leave short notes only for ambiguous or difficult frames.",
                "5. Change review_status from pending to reviewed when the row is completed.",
                "",
                f"Subset size: {total_count}",
                f"Positive proxy rows: {positive_count}",
                f"Negative proxy rows: {negative_count}",
                "",
                "This pack is for validation only. It does not replace the main dev manifest.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument(
        "--output-root",
        default="outputs/01_protocol/annotations/multimodal_validation_subset",
    )
    parser.add_argument("--total-count", type=int, default=80)
    parser.add_argument("--positive-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.positive_count <= 0 or args.total_count <= 0:
        raise ValueError("Counts must be positive.")
    if args.positive_count >= args.total_count:
        raise ValueError("positive-count must be smaller than total-count.")

    manifest_path = PROJECT_ROOT / args.manifest
    output_root = PROJECT_ROOT / args.output_root
    dataset_root = manifest_path.parent.parent
    raw_root = dataset_root / "raw"

    df = pd.read_csv(manifest_path)
    if "visible_text_flag" not in df.columns:
        raise KeyError("Manifest is missing visible_text_flag.")

    df = df.copy().reset_index(drop=True)
    df["__row_id"] = df.index

    positive_pool = df[df["visible_text_flag"].astype(bool)].copy()
    negative_pool = df[~df["visible_text_flag"].astype(bool)].copy()

    effective_positive_count = min(args.positive_count, len(positive_pool))
    effective_negative_count = min(args.total_count - effective_positive_count, len(negative_pool))
    effective_total_count = effective_positive_count + effective_negative_count

    if effective_positive_count == 0 or effective_negative_count == 0:
        raise ValueError(
            "Text validation subset requires at least one positive and one negative candidate."
        )

    positive_df = select_balanced_rows(
        positive_pool,
        target_count=effective_positive_count,
        seed=args.seed,
    )
    negative_df = select_balanced_rows(
        negative_pool,
        target_count=effective_negative_count,
        seed=args.seed + 1,
    )

    positive_df["sample_bucket"] = "proxy_positive"
    negative_df["sample_bucket"] = "proxy_negative"
    selected_df = (
        pd.concat([positive_df, negative_df], ignore_index=True)
        .sort_values(["sample_bucket", "day_id", "camera_stream_id", "relative_path"])
        .reset_index(drop=True)
    )

    manifest_output = output_root / "text_validation_subset_manifest.csv"
    review_output = output_root / "text_validation_review_template.csv"
    summary_output = output_root / "summary.json"

    save_csv(
        selected_df[
            [
                "relative_path",
                "day_id",
                "camera_stream_id",
                "view_type",
                "condition_label",
                "visible_text_flag",
                "sample_bucket",
            ]
        ],
        manifest_output,
    )
    save_csv(build_review_template(selected_df), review_output)
    materialise_review_pack(selected_df, raw_root=raw_root, pack_root=output_root)
    write_readme(
        output_root,
        total_count=len(selected_df),
        positive_count=int((selected_df["sample_bucket"] == "proxy_positive").sum()),
        negative_count=int((selected_df["sample_bucket"] == "proxy_negative").sum()),
    )

    summary = {
        "version": "1.0",
        "manifest": str(manifest_output.relative_to(PROJECT_ROOT)),
        "review_template": str(review_output.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "requested_total_count": args.total_count,
        "requested_positive_count": args.positive_count,
        "requested_negative_count": args.total_count - args.positive_count,
        "total_count": int(len(selected_df)),
        "positive_count": int((selected_df["sample_bucket"] == "proxy_positive").sum()),
        "negative_count": int((selected_df["sample_bucket"] == "proxy_negative").sum()),
        "unique_days": int(selected_df["day_id"].nunique()),
        "unique_streams": int(selected_df["camera_stream_id"].nunique()),
        "view_type_counts": selected_df["view_type"].value_counts().to_dict(),
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
