"""CASTLE subset definitions and manifest sampling helpers."""

from __future__ import annotations

from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    pd = Any  # type: ignore[assignment]

# NOTE: Re-identification evaluation must always stratify by day_id
# to enforce the closed-set 10-participant constraint.
# bao is a valid participant and must NOT be excluded from
# detection or anonymisation subsets.

DEV_SET = {
    "integrity_status": "valid",
    "n_samples": 300,
    "stratify_by": ["day_id", "view_type", "camera_stream_id"],
    "condition_labels": [
        "small_face",
        "motion_blur",
        "extreme_pose",
        "downward_view",
        "visible_screen",
        "visible_text",
        "multiple_faces",
        "no_face",
    ],
    "seed": 42,
    "no_overlap_with": ["CALIBRATION_SET"],
}

CALIBRATION_SET = {
    "integrity_status": "valid",
    "n_samples": 200,
    "stratify_by": ["blur_level", "face_size", "occlusion_ratio", "webp_artefact_severity"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET"],
}

DETECTION_EVAL_SUBSET = {
    "integrity_status": "valid",
    "view_type": "egocentric",
    "n_samples": 500,
    "stratify_by": ["day_id", "camera_stream_id"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

ANONYMISATION_EVAL_SUBSET = {
    "integrity_status": "valid",
    "view_type": "egocentric",
    "n_samples": 500,
    "stratify_by": ["day_id", "participant_id"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

ADAPTIVE_EVAL_SUBSET = {
    "integrity_status": "valid",
    "view_type": "egocentric",
    "n_samples": "all",
    "stratify_by": ["day_id", "camera_stream_id"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

FID_SUBSET = {
    "integrity_status": "valid",
    "n_samples": 50000,
    "stratify_by": ["camera_stream_id"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

INSPECTION_SUBSET = {
    "integrity_status": "valid",
    "n_samples": 50,
    "stratify_by": ["view_type", "camera_stream_id"],
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

CROSS_VIEW_EVAL_SUBSET = {
    "integrity_status": "valid",
    "match_on": ["day_id", "timestamp_id"],
    "view_types": ["egocentric", "exocentric"],
    "pairing_rule": "identical_HHNNNN",
    "time_alignment_rule": "t = (NNNN - 1) * 5 seconds",
    "selection_scope": "matched egocentric and exocentric frame pairs at identical HHNNNN timestamps",
    "seed": 42,
    "no_overlap_with": ["DEV_SET", "CALIBRATION_SET"],
}

SUBSET_DEFINITIONS = {
    "DEV_SET": DEV_SET,
    "CALIBRATION_SET": CALIBRATION_SET,
    "DETECTION_EVAL_SUBSET": DETECTION_EVAL_SUBSET,
    "ANONYMISATION_EVAL_SUBSET": ANONYMISATION_EVAL_SUBSET,
    "ADAPTIVE_EVAL_SUBSET": ADAPTIVE_EVAL_SUBSET,
    "FID_SUBSET": FID_SUBSET,
    "INSPECTION_SUBSET": INSPECTION_SUBSET,
    "CROSS_VIEW_EVAL_SUBSET": CROSS_VIEW_EVAL_SUBSET,
}


def resolve_subset(subset_def: dict[str, Any], manifest_df: Any) -> Any:
    """Resolve a subset definition into a sampled manifest DataFrame."""
    filtered_df = manifest_df.copy()
    filtered_df = _apply_column_aliases(filtered_df)
    for key in ("integrity_status", "view_type"):
        if key in subset_def and subset_def[key] is not None:
            filtered_df = filtered_df[filtered_df[key] == subset_def[key]]

    n_samples = subset_def["n_samples"]
    if n_samples == "all":
        return filtered_df.reset_index(drop=True)

    n_samples = int(n_samples)
    if n_samples > len(filtered_df):
        raise ValueError(
            f"Requested n_samples={n_samples} exceeds available rows after filtering: {len(filtered_df)}"
        )

    stratify_by = [_canonical_column_name(name) for name in (subset_def.get("stratify_by") or [])]
    seed = int(subset_def.get("seed", 42))
    if not stratify_by:
        return filtered_df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    group_sizes = (
        filtered_df.groupby(stratify_by, dropna=False)
        .size()
        .reset_index(name="group_size")
        .sort_values(stratify_by)
        .reset_index(drop=True)
    )
    total_available = int(group_sizes["group_size"].sum())
    group_sizes["target_float"] = group_sizes["group_size"] * n_samples / total_available
    group_sizes["target_count"] = group_sizes["target_float"].astype(int)
    remainder = n_samples - int(group_sizes["target_count"].sum())

    if remainder > 0:
        group_sizes["fractional"] = group_sizes["target_float"] - group_sizes["target_count"]
        group_sizes = group_sizes.sort_values(
            ["fractional", "group_size"], ascending=[False, False]
        ).reset_index(drop=True)
        for idx in range(remainder):
            group_sizes.loc[idx, "target_count"] += 1
        group_sizes = group_sizes.drop(columns=["fractional"])

    sampled_parts: list[pd.DataFrame] = []
    keyed_targets = {
        _normalise_group_key(tuple(row[col] for col in stratify_by)): int(row["target_count"])
        for _, row in group_sizes.iterrows()
    }

    for group_key, group_df in filtered_df.groupby(stratify_by, dropna=False, sort=True):
        target_count = keyed_targets.get(_normalise_group_key(group_key), 0)
        if target_count > 0:
            sampled_parts.append(group_df.sample(n=target_count, random_state=seed))

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def no_overlap_check(subset_frames: dict[str, Any]) -> bool:
    """Return True when all provided subsets are pairwise disjoint by relative path."""
    seen: dict[str, str] = {}
    for subset_name, subset_df in subset_frames.items():
        if "relative_path" not in subset_df.columns:
            raise ValueError(f"{subset_name} is missing required column 'relative_path'")
        for relative_path in subset_df["relative_path"].tolist():
            owner = seen.get(relative_path)
            if owner is not None and owner != subset_name:
                raise ValueError(
                    f"Subset overlap detected for relative_path={relative_path} between {owner} and {subset_name}"
                )
            seen[relative_path] = subset_name
    return True


def resolve_cross_view_subset(manifest_df: Any) -> Any:
    """Resolve candidate cross-view pairs using identical day and HHNNNN timestamps."""
    filtered_df = manifest_df.copy()
    filtered_df = _apply_column_aliases(filtered_df)
    filtered_df = filtered_df[filtered_df["integrity_status"] == "valid"]
    egocentric_df = filtered_df[filtered_df["view_type"] == "egocentric"].copy()
    exocentric_df = filtered_df[filtered_df["view_type"] == "exocentric"].copy()

    ego_pairs = egocentric_df.rename(
        columns={
            "relative_path": "egocentric_relative_path",
            "camera_stream_id": "egocentric_stream_id",
            "participant_id": "egocentric_participant_id",
        }
    )
    exo_pairs = exocentric_df.rename(
        columns={
            "relative_path": "exocentric_relative_path",
            "camera_stream_id": "exocentric_stream_id",
        }
    )
    return ego_pairs.merge(
        exo_pairs,
        on=["day_id", "timestamp_id"],
        suffixes=("_ego", "_exo"),
        how="inner",
    )


def _normalise_group_key(group_key: Any) -> tuple[Any, ...]:
    """Normalise pandas group keys to a tuple for stable dictionary lookup."""
    if isinstance(group_key, tuple):
        return group_key
    return (group_key,)


def _canonical_column_name(name: str) -> str:
    """Map subset-friendly aliases onto manifest column names."""
    alias_map = {
        "day_id": "day_or_session_id",
        "timestamp_id": "timestamp_id",
    }
    return alias_map.get(name, name)


def _apply_column_aliases(manifest_df: Any) -> Any:
    """Expose convenience aliases expected by subset definitions."""
    if "day_id" not in manifest_df.columns and "day_or_session_id" in manifest_df.columns:
        manifest_df = manifest_df.copy()
        manifest_df["day_id"] = manifest_df["day_or_session_id"]
    if "timestamp_id" not in manifest_df.columns and "file_name" in manifest_df.columns:
        manifest_df = manifest_df.copy()
        manifest_df["timestamp_id"] = manifest_df["file_name"].str.rsplit(".", n=1).str[0]
    return manifest_df
