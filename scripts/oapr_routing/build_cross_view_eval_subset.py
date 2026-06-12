#!/usr/bin/env python3

"""Build a deterministic cross-view evaluation subset from the CASTLE manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.subset_definitions import resolve_cross_view_subset


def sample_cross_view_pairs(
    manifest_df: pd.DataFrame,
    target_pairs: int,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Sample a balanced cross-view pair subset across day and exocentric stream."""
    if target_pairs <= 0:
        raise ValueError("target_pairs must be positive")

    pair_df = resolve_cross_view_subset(manifest_df)
    if pair_df.empty:
        raise ValueError("No cross-view pairs were resolved from the manifest")

    pair_df = pair_df.sort_values(
        ["day_id", "exocentric_stream_id", "egocentric_stream_id", "timestamp_id"]
    ).reset_index(drop=True)
    pair_df["pair_id"] = (
        pair_df["day_id"].astype(str)
        + "::"
        + pair_df["timestamp_id"].astype(str)
        + "::"
        + pair_df["egocentric_stream_id"].astype(str)
        + "::"
        + pair_df["exocentric_stream_id"].astype(str)
    )

    group_keys = ["day_id", "exocentric_stream_id"]
    groups = list(pair_df.groupby(group_keys, sort=True))
    if target_pairs < len(groups):
        raise ValueError(
            f"target_pairs={target_pairs} is smaller than the number of strata={len(groups)}"
        )

    counts = [len(group_df) for _, group_df in groups]
    total = sum(counts)
    allocations = [max(1, int(target_pairs * count / total)) for count in counts]

    current_total = sum(allocations)
    while current_total > target_pairs:
        adjusted = False
        for idx in sorted(range(len(groups)), key=lambda i: allocations[i], reverse=True):
            if allocations[idx] > 1:
                allocations[idx] -= 1
                current_total -= 1
                adjusted = True
                if current_total == target_pairs:
                    break
        if not adjusted:
            break
    while current_total < target_pairs:
        for idx in sorted(range(len(groups)), key=lambda i: counts[i] - allocations[i], reverse=True):
            if allocations[idx] < counts[idx]:
                allocations[idx] += 1
                current_total += 1
                if current_total == target_pairs:
                    break
        else:
            break

    sampled_frames: list[pd.DataFrame] = []
    for (group_key, group_df), allocation in zip(groups, allocations, strict=True):
        sample_n = min(allocation, len(group_df))
        sampled_frames.append(
            group_df.sample(n=sample_n, random_state=random_seed).sort_values(
                ["timestamp_id", "egocentric_stream_id"]
            )
        )

    sampled_df = (
        pd.concat(sampled_frames, ignore_index=True)
        .sort_values(["day_id", "exocentric_stream_id", "timestamp_id", "egocentric_stream_id"])
        .reset_index(drop=True)
    )
    return sampled_df


def build_summary(sampled_df: pd.DataFrame, target_pairs: int, random_seed: int) -> dict[str, object]:
    """Create a compact manifest summary."""
    by_group = (
        sampled_df.groupby(["day_id", "exocentric_stream_id"])
        .size()
        .reset_index(name="pair_count")
        .sort_values(["day_id", "exocentric_stream_id"])
    )
    return {
        "target_pairs": int(target_pairs),
        "sampled_pairs": int(len(sampled_df)),
        "random_seed": int(random_seed),
        "days": sorted(sampled_df["day_id"].dropna().unique().tolist()),
        "egocentric_stream_count": int(sampled_df["egocentric_stream_id"].nunique()),
        "exocentric_stream_count": int(sampled_df["exocentric_stream_id"].nunique()),
        "pairs_by_day": {
            str(key): int(value) for key, value in sampled_df["day_id"].value_counts().sort_index().items()
        },
        "pairs_by_exocentric_stream": {
            str(key): int(value)
            for key, value in sampled_df["exocentric_stream_id"].value_counts().sort_index().items()
        },
        "pairs_by_day_and_exocentric_stream": by_group.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/castle2024/raw_dataset_index.csv",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=200,
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="outputs/01_protocol/supporting_protocols/04_cross_view_pairs_200.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/runs/cross_view/cross_view_eval_subset_summary.json",
    )
    args = parser.parse_args()

    manifest_path = PROJECT_ROOT / args.manifest
    output_path = PROJECT_ROOT / args.output
    summary_path = PROJECT_ROOT / args.summary_output

    manifest_df = pd.read_csv(manifest_path)
    sampled_df = sample_cross_view_pairs(
        manifest_df=manifest_df,
        target_pairs=args.target_pairs,
        random_seed=args.random_seed,
    )
    summary = build_summary(sampled_df, args.target_pairs, args.random_seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"manifest": str(output_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
