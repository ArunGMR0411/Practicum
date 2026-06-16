"""Tests for deterministic cross-view evaluation subset sampling."""

from __future__ import annotations

import pandas as pd

from scripts.oapr_routing.build_cross_view_eval_subset import sample_cross_view_pairs


def build_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in ["day1", "day2"]:
        for timestamp_idx in range(3):
            timestamp = f"08_{timestamp_idx:04d}"
            for ego in ["alice", "bob"]:
                rows.append(
                    {
                        "relative_path": f"{day}/members/{ego}/{timestamp}.webp",
                        "file_name": f"{timestamp}.webp",
                        "camera_stream_id": ego,
                        "view_type": "egocentric",
                        "participant_id": ego,
                        "day_or_session_id": day,
                        "integrity_status": "valid",
                    }
                )
            for exo in ["kitchen", "meeting"]:
                rows.append(
                    {
                        "relative_path": f"{day}/fixed/{exo}/{timestamp}.webp",
                        "file_name": f"{timestamp}.webp",
                        "camera_stream_id": exo,
                        "view_type": "exocentric",
                        "participant_id": "none",
                        "day_or_session_id": day,
                        "integrity_status": "valid",
                    }
                )
    return pd.DataFrame(rows)


def test_cross_view_sampling_is_deterministic_and_stratified() -> None:
    manifest_df = build_manifest()
    sampled_a = sample_cross_view_pairs(manifest_df, target_pairs=8, random_seed=42)
    sampled_b = sample_cross_view_pairs(manifest_df, target_pairs=8, random_seed=42)

    assert sampled_a["pair_id"].tolist() == sampled_b["pair_id"].tolist()
    assert len(sampled_a) == 8
    assert set(sampled_a["day_id"]) == {"day1", "day2"}
    assert set(sampled_a["exocentric_stream_id"]) == {"kitchen", "meeting"}
