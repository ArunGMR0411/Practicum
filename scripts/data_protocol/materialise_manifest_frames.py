#!/usr/bin/env python3

"""Copy raw CASTLE frames referenced by manifest CSVs into a transfer directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "castle2024" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=Path,
        help="Manifest CSV containing a relative_path column. Pass multiple times for a union.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Source raw CASTLE frame root.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Destination root, e.g. transfer/constrained-compute_seed. Frames are copied under data/castle2024/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the transfer plan without copying files.",
    )
    return parser.parse_args()


def read_manifest_paths(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "relative_path" not in (reader.fieldnames or []):
            raise ValueError(f"{manifest_path} is missing required column: relative_path")
        return [str(row["relative_path"]) for row in reader if str(row.get("relative_path", "")).strip()]


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    destination_raw_root = output_root / "data" / "castle2024" / "raw"

    requested_paths: set[str] = set()
    manifest_summaries = []
    for manifest_path in args.manifest:
        paths = read_manifest_paths(manifest_path)
        requested_paths.update(paths)
        manifest_summaries.append(
            {
                "manifest": str(manifest_path),
                "rows_with_relative_path": len(paths),
                "unique_paths": len(set(paths)),
            }
        )

    copied = 0
    skipped_existing = 0
    missing = []
    total_bytes = 0
    for relative_path in sorted(requested_paths):
        source = raw_root / relative_path
        destination = destination_raw_root / relative_path
        if not source.is_file():
            missing.append(relative_path)
            continue
        size = source.stat().st_size
        total_bytes += size
        if args.dry_run:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == size:
            skipped_existing += 1
            continue
        shutil.copy2(source, destination)
        copied += 1

    summary = {
        "dry_run": bool(args.dry_run),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "destination_raw_root": str(destination_raw_root),
        "manifest_summaries": manifest_summaries,
        "unique_requested_paths": len(requested_paths),
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / 1024 / 1024, 3),
        "total_gib": round(total_bytes / 1024 / 1024 / 1024, 3),
        "copied": copied,
        "skipped_existing": skipped_existing,
    }
    print(json.dumps(summary, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
