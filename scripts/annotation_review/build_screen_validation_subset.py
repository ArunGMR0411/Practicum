#!/usr/bin/env python3

"""Build a manually reviewable screen-presence validation subset."""

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
    return (
        day_counter[str(row["day_id"])],
        stream_counter[str(row["camera_stream_id"])],
        view_counter[str(row["view_type"])],
        str(row["relative_path"]),
    )


def select_balanced_rows(pool: pd.DataFrame, target_count: int, seed: int) -> pd.DataFrame:
    if len(pool) < target_count:
        raise ValueError(f"Requested {target_count} rows from pool of size {len(pool)}.")
    shuffled = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    day_counter: Counter[str] = Counter()
    stream_counter: Counter[str] = Counter()
    view_counter: Counter[str] = Counter()
    selected_ids: list[int] = []
    available = shuffled.copy()

    while len(selected_ids) < target_count:
        scored = available.apply(
            score_row,
            axis=1,
            day_counter=day_counter,
            stream_counter=stream_counter,
            view_counter=view_counter,
        )
        best_index = min(scored.items(), key=lambda item: item[1])[0]
        best_row = available.loc[best_index]
        selected_ids.append(int(best_row["__row_id"]))
        day_counter[str(best_row["day_id"])] += 1
        stream_counter[str(best_row["camera_stream_id"])] += 1
        view_counter[str(best_row["view_type"])] += 1
        available = available.drop(index=best_index)

    return shuffled[shuffled["__row_id"].isin(selected_ids)].copy()


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)


def materialise_review_pack(selected_df: pd.DataFrame, raw_root: Path, pack_root: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/01_protocol/supporting_protocols/01_development_300.csv")
    parser.add_argument(
        "--output-root",
        default="outputs/01_protocol/annotations/multimodal_validation_subset",
    )
    parser.add_argument("--total-count", type=int, default=60)
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    output_root = PROJECT_ROOT / args.output_root
    dataset_root = manifest_path.parent.parent
    raw_root = dataset_root / "raw"

    df = pd.read_csv(manifest_path).reset_index(drop=True)
    df["__row_id"] = df.index

    positive_pool = df[df["visible_screen_flag"].astype(bool)].copy()
    negative_pool = df[~df["visible_screen_flag"].astype(bool)].copy()

    effective_positive_count = min(args.positive_count, len(positive_pool))
    effective_negative_count = min(args.total_count - effective_positive_count, len(negative_pool))
    if effective_positive_count == 0 or effective_negative_count == 0:
        raise ValueError("Need at least one positive and one negative row for screen validation.")

    positive_df = select_balanced_rows(positive_pool, effective_positive_count, args.seed)
    negative_df = select_balanced_rows(negative_pool, effective_negative_count, args.seed + 1)
    positive_df["sample_bucket"] = "proxy_positive"
    negative_df["sample_bucket"] = "proxy_negative"
    selected_df = (
        pd.concat([positive_df, negative_df], ignore_index=True)
        .sort_values(["sample_bucket", "day_id", "camera_stream_id", "relative_path"])
        .reset_index(drop=True)
    )

    manifest_output = output_root / "screen_validation_subset_manifest.csv"
    review_output = output_root / "screen_validation_review_template.csv"
    summary_output = output_root / "summary.json"
    readme_output = output_root / "README.txt"

    save_csv(
        selected_df[
            [
                "relative_path",
                "day_id",
                "camera_stream_id",
                "view_type",
                "condition_label",
                "visible_screen_flag",
                "sample_bucket",
            ]
        ],
        manifest_output,
    )

    review_df = selected_df[
        [
            "relative_path",
            "day_id",
            "camera_stream_id",
            "view_type",
            "condition_label",
            "visible_screen_flag",
            "sample_bucket",
        ]
    ].copy()
    review_df["manual_screen_present"] = ""
    review_df["manual_screen_contains_sensitive_content"] = ""
    review_df["review_status"] = "pending"
    review_df["reviewer_id"] = ""
    review_df["manual_notes"] = ""
    save_csv(review_df, review_output)

    materialise_review_pack(selected_df, raw_root=raw_root, pack_root=output_root)

    readme_output.write_text(
        "\n".join(
            [
                "Manual screen validation subset review instructions",
                "",
                "1. Open each frame from images/ using screen_validation_review_template.csv.",
                "2. Set manual_screen_present to yes if any screen, monitor, laptop display, TV, phone screen, or tablet screen is visible.",
                "3. Set manual_screen_contains_sensitive_content to yes only if the visible screen clearly exposes readable or sensitive content.",
                "4. Change review_status to reviewed when complete.",
                "5. Use manual_notes only for ambiguous cases.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
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
