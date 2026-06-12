#!/usr/bin/env python3
"""Verify the locked 500-frame and 250-frame evaluation protocols.

The thesis does **not** re-sample protocols at regeneration time. Membership is
fixed by the reviewed manifests under outputs/01_protocol/thesis_manifests/.
This script checks that:

1. Face 500 manifests contain exactly 500 unique relative paths.
2. Multimodal 250 manifest contains exactly 250 unique protocol rows.
3. Paths resolve against data/castle2024/raw_dataset_index.csv when present.
4. Raw files exist under data/castle2024/raw/ when the mount is available.

Exit code 0 on success; 1 if validation fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "outputs/01_protocol/thesis_manifests"
RAW_INDEX = ROOT / "data/castle2024/raw_dataset_index.csv"
RAW_ROOT = ROOT / "data/castle2024/raw"

FACE_500 = {
    "detection": MANIFESTS / "final_face_detection_500.csv",
    "anonymisation": MANIFESTS / "final_face_anonymisation_500.csv",
    "advanced_methods": MANIFESTS / "final_advanced_methods_500.csv",
}
MM_250 = MANIFESTS / "final_multimodal_250.csv"
MM_ANN = (
    ROOT
    / "outputs/01_protocol/annotations/multimodal_250/reviewed_multimodal_250_with_boxes.csv"
)

# Documented selection seeds used when subsets were originally materialised.
# Multimodal membership is reviewer-curated (not random re-sample).
PROTOCOL_SEEDS = {
    "face_detection_500_selection_seed_field": "global_500_from_anonymisation_oapr_protocol",
    "face_anonymisation_500_selection_seed_field": "global_500_from_anonymisation_oapr_protocol",
    "multimodal_250": "reviewer_curated_locked_protocol (not random)",
    "supporting_dev_300_seed": 42,
    "supporting_calibration_200_seed": 42,
}


def _rel_col(df: pd.DataFrame) -> str:
    for c in ("relative_path", "image_id", "image_path"):
        if c in df.columns:
            return c
    raise KeyError(f"No path column in {list(df.columns)}")


def _paths(df: pd.DataFrame) -> list[str]:
    col = _rel_col(df)
    vals = df[col].astype(str)
    # strip data/castle2024/raw/ prefix if present
    cleaned = []
    for v in vals:
        v = v.replace("\\", "/")
        prefix = "data/castle2024/raw/"
        if v.startswith(prefix):
            v = v[len(prefix) :]
        cleaned.append(v)
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-raw",
        action="store_true",
        help="Fail if raw frames are missing (default: warn only).",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "protocol_seeds": PROTOCOL_SEEDS,
        "checks": {},
        "ok": True,
    }
    errors: list[str] = []
    warnings: list[str] = []

    index_paths: set[str] | None = None
    if RAW_INDEX.exists():
        idx = pd.read_csv(RAW_INDEX)
        index_paths = set(idx["relative_path"].astype(str))
        report["raw_index_rows"] = int(len(idx))
    else:
        warnings.append(f"Missing raw index: {RAW_INDEX}")

    for name, path in FACE_500.items():
        if not path.exists():
            errors.append(f"Missing face protocol: {path}")
            continue
        df = pd.read_csv(path)
        paths = _paths(df)
        n = len(paths)
        n_unique = len(set(paths))
        check = {
            "path": str(path.relative_to(ROOT)),
            "rows": n,
            "unique_paths": n_unique,
            "expected": 500,
        }
        if n != 500 or n_unique != 500:
            errors.append(f"{name}: expected 500 unique paths, got rows={n} unique={n_unique}")
            check["ok"] = False
        else:
            check["ok"] = True
        if index_paths is not None:
            missing_idx = sorted(set(paths) - index_paths)
            check["missing_from_index"] = len(missing_idx)
            if missing_idx:
                errors.append(f"{name}: {len(missing_idx)} paths missing from raw index")
                check["ok"] = False
        missing_files = []
        if RAW_ROOT.exists():
            for rel in paths:
                if not (RAW_ROOT / rel).exists():
                    missing_files.append(rel)
            check["missing_raw_files"] = len(missing_files)
            if missing_files:
                msg = f"{name}: {len(missing_files)} raw files missing under {RAW_ROOT}"
                if args.require_raw:
                    errors.append(msg)
                    check["ok"] = False
                else:
                    warnings.append(msg)
        report["checks"][f"face_500_{name}"] = check

    if not MM_250.exists():
        errors.append(f"Missing multimodal protocol: {MM_250}")
    else:
        mm = pd.read_csv(MM_250)
        paths = _paths(mm)
        check = {
            "path": str(MM_250.relative_to(ROOT)),
            "rows": len(paths),
            "unique_paths": len(set(paths)),
            "expected": 250,
            "ok": len(paths) == 250 and len(set(paths)) == 250,
        }
        if not check["ok"]:
            errors.append(
                f"multimodal: expected 250 unique paths, got rows={len(paths)} unique={len(set(paths))}"
            )
        if MM_ANN.exists():
            ann = pd.read_csv(MM_ANN)
            check["annotation_rows"] = int(len(ann))
            if len(ann) != 250:
                warnings.append(f"multimodal annotations rows={len(ann)} (expected 250)")
        report["checks"]["multimodal_250"] = check

    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = len(errors) == 0

    out = ROOT / "outputs/01_protocol/02_locked_protocol_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")
    if errors:
        print(f"FAILED with {len(errors)} error(s)", file=sys.stderr)
        return 1
    print("OK: locked 500 / 250 protocols verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
