#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "castle2024"
RAW_ROOT = DATASET_ROOT / "raw"
MANIFEST_PATH = DATASET_ROOT / "raw_dataset_index.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "submission_evidence" / "09_traceability" / "07_raw_dataset_integrity.txt"
MAX_WORKERS = min(8, os.cpu_count() or 1)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_4k(width: int, height: int) -> bool:
    return (width >= 3840 and height >= 2160) or (width >= 2160 and height >= 3840)


def validate_row(row: dict[str, str]) -> dict[str, object]:
    relative_path = row["relative_path"]
    path = RAW_ROOT / relative_path
    result: dict[str, object] = {
        "relative_path": relative_path,
        "file_ext": row["file_ext"],
        "file_size_bytes": int(row["file_size_bytes"]),
        "manifest_width": int(row["image_width"]) if row["image_width"] != "null" else None,
        "manifest_height": int(row["image_height"]) if row["image_height"] != "null" else None,
        "manifest_checksum": row["checksum_sha256"],
        "manifest_read_status": row["read_status"],
        "manifest_integrity_status": row["integrity_status"],
        "open_ok": False,
        "checksum_match": False,
        "dimension_match": False,
        "native_4k": False,
        "actual_width": None,
        "actual_height": None,
        "actual_checksum": None,
        "issues": [],
    }

    if not path.exists():
        result["issues"].append("missing_file")
        return result

    actual_checksum = sha256_file(path)
    result["actual_checksum"] = actual_checksum
    result["checksum_match"] = actual_checksum == row["checksum_sha256"]
    if not result["checksum_match"]:
        result["issues"].append("checksum_mismatch")

    try:
        with Image.open(path) as image:
            width = image.width
            height = image.height
            image.verify()
        result["open_ok"] = True
        result["actual_width"] = width
        result["actual_height"] = height
        result["dimension_match"] = (
            row["image_width"] != "null"
            and row["image_height"] != "null"
            and width == int(row["image_width"])
            and height == int(row["image_height"])
        )
        if not result["dimension_match"]:
            result["issues"].append("dimension_mismatch")
        result["native_4k"] = classify_4k(width, height)
        if not result["native_4k"]:
            result["issues"].append("unexpected_dimensions")
    except (UnidentifiedImageError, OSError):
        result["issues"].append("failed_open")

    if row["read_status"] == "ok" and not result["open_ok"]:
        result["issues"].append("manifest_ok_but_failed_validation")

    return result


def format_counter(counter: Counter[str]) -> list[str]:
    lines: list[str] = []
    for key in sorted(counter):
        lines.append(f"{key}: {counter[key]}")
    return lines


def main() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")

    validated_rows = 0
    ext_counter: Counter[str] = Counter()
    resolution_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    flagged_files: list[str] = []
    size_sum = 0
    min_size: int | None = None
    max_size: int | None = None
    native_4k_count = 0

    total_rows = 0
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for result in executor.map(validate_row, reader, chunksize=32):
                total_rows += 1
                ext = str(result["file_ext"])
                size = int(result["file_size_bytes"])
                ext_counter[ext] += 1
                size_sum += size
                min_size = size if min_size is None else min(min_size, size)
                max_size = size if max_size is None else max(max_size, size)

                if result["open_ok"]:
                    validated_rows += 1
                    resolution_counter[f'{result["actual_width"]}x{result["actual_height"]}'] += 1
                    if result["native_4k"]:
                        native_4k_count += 1

                for issue in result["issues"]:
                    issue_counter[str(issue)] += 1

                if result["issues"]:
                    flagged_files.append(
                        f'{result["relative_path"]} | issues={",".join(result["issues"])} | '
                        f'manifest_checksum={result["manifest_checksum"]} | '
                        f'actual_checksum={result["actual_checksum"]} | '
                        f'manifest_dimensions={result["manifest_width"]}x{result["manifest_height"]} | '
                        f'actual_dimensions={result["actual_width"]}x{result["actual_height"]}'
                    )

    mean_size = size_sum / total_rows if total_rows else math.nan
    native_4k_proportion = native_4k_count / validated_rows if validated_rows else math.nan

    report_lines = [
        "CASTLE 2024 Integrity Report",
        f"Manifest: {MANIFEST_PATH}",
        f"Raw dataset root: {RAW_ROOT}",
        "",
        "Summary",
        f"Total manifest rows: {total_rows}",
        f"Validated image opens: {validated_rows}",
        f"Files flagged: {len(flagged_files)}",
        "",
        "File statistics",
        f"Mean file size bytes: {mean_size:.2f}",
        f"Min file size bytes: {min_size}",
        f"Max file size bytes: {max_size}",
        "",
        "Counts by extension",
        *format_counter(ext_counter),
        "",
        "Resolution distribution",
        *format_counter(resolution_counter),
        "",
        "Native 4K proportion",
        f"Native 4K files: {native_4k_count}",
        f"Native 4K proportion among successfully opened files: {native_4k_proportion:.6f}",
        "",
        "Issue counts",
        *(format_counter(issue_counter) if issue_counter else ["none"]),
        "",
        "Flagged files",
        *(flagged_files if flagged_files else ["none"]),
    ]

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
