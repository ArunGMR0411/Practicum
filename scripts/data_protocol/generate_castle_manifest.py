#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Iterable
from concurrent.futures import ProcessPoolExecutor

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data" / "castle2024"
RAW_ROOT = DATASET_ROOT / "raw"
MANIFEST_PATH = DATASET_ROOT / "raw_dataset_index.csv"
MAX_WORKERS = min(8, os.cpu_count() or 1)

FIELDNAMES = [
    "relative_path",
    "file_name",
    "file_ext",
    "file_size_bytes",
    "checksum_sha256",
    "image_width",
    "image_height",
    "camera_stream_id",
    "view_type",
    "participant_id",
    "day_or_session_id",
    "read_status",
    "integrity_status",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_metadata(relative_path: Path) -> tuple[str, str, str, str]:
    parts = relative_path.parts
    day_or_session_id = parts[0] if len(parts) > 0 else ""
    top_level_group = parts[1] if len(parts) > 1 else ""
    leaf_folder = parts[2] if len(parts) > 2 else ""

    if top_level_group == "members":
        return leaf_folder, "egocentric", leaf_folder, day_or_session_id
    if top_level_group == "fixed":
        return leaf_folder, "exocentric", "", day_or_session_id
    return "", "unknown", "", day_or_session_id


def null_if_empty(value: str) -> str:
    return value if value else "null"


def iter_files(root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        for filename in filenames:
            yield Path(current_root) / filename


def build_row(path: Path) -> dict[str, str | int]:
    relative_path = path.relative_to(RAW_ROOT)
    file_ext = path.suffix.lower().lstrip(".")
    camera_stream_id, view_type, participant_id, day_or_session_id = infer_metadata(relative_path)

    row: dict[str, str | int] = {
        "relative_path": relative_path.as_posix(),
        "file_name": path.name,
        "file_ext": file_ext if file_ext else "null",
        "file_size_bytes": path.stat().st_size,
        "checksum_sha256": "null",
        "image_width": "null",
        "image_height": "null",
        "camera_stream_id": null_if_empty(camera_stream_id),
        "view_type": view_type,
        "participant_id": null_if_empty(participant_id),
        "day_or_session_id": null_if_empty(day_or_session_id),
        "read_status": "failed",
        "integrity_status": "unsupported",
    }

    try:
        row["checksum_sha256"] = sha256_file(path)
        with Image.open(path) as image:
            row["image_width"] = image.width
            row["image_height"] = image.height
        row["read_status"] = "ok"
        row["integrity_status"] = "valid"
    except FileNotFoundError:
        row["integrity_status"] = "missing"
    except UnidentifiedImageError:
        row["integrity_status"] = "unsupported"
    except OSError:
        row["integrity_status"] = "corrupt"

    return row


def ensure_contract_dirs() -> None:
    for dirname in ["derived", "manifests", "samples", "annotations", "docs"]:
        (DATASET_ROOT / dirname).mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not RAW_ROOT.exists():
        raise SystemExit(f"Raw dataset root not found: {RAW_ROOT}")

    ensure_contract_dirs()

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row in executor.map(build_row, iter_files(RAW_ROOT), chunksize=32):
                writer.writerow(row)

    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
