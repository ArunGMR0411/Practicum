#!/usr/bin/env python3

"""Summarise CASTLE manifest metadata into a compact JSON report."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        """Fallback passthrough when tqdm is unavailable."""
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "castle2024" / "raw_dataset_index.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "castle_metadata.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for metadata summarisation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)),
        help="Output JSON path relative to the project root.",
    )
    return parser.parse_args()


def count_rows(path: Path) -> int:
    """Count manifest data rows without loading the entire file into memory."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def null_to_unknown(value: str) -> str:
    """Map manifest null placeholders to a stable 'unknown' bucket."""
    return "unknown" if value in {"", "null"} else value


def build_summary(manifest_path: Path) -> dict[str, object]:
    """Stream the manifest and compute summary statistics for the dataset."""
    total_rows = count_rows(manifest_path)
    totals = {"total_frames": 0, "valid_frames": 0, "invalid_frames": 0}
    by_stream: dict[str, dict[str, float | int]] = defaultdict(lambda: {"frame_count": 0, "storage_mb": 0.0})
    by_view_type: Counter[str] = Counter()
    by_participant: Counter[str] = Counter()
    by_session: Counter[str] = Counter()
    resolution_distribution: Counter[str] = Counter()
    file_sizes: list[int] = []

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, total=total_rows, desc="Summarising CASTLE manifest", unit="frame"):
            totals["total_frames"] += 1
            if row["integrity_status"] == "valid":
                totals["valid_frames"] += 1
            else:
                totals["invalid_frames"] += 1

            stream_id = null_to_unknown(row["camera_stream_id"])
            by_stream[stream_id]["frame_count"] += 1
            by_stream[stream_id]["storage_mb"] += int(row["file_size_bytes"]) / (1024 * 1024)

            by_view_type[null_to_unknown(row["view_type"])] += 1
            by_participant[null_to_unknown(row["participant_id"])] += 1
            by_session[null_to_unknown(row["day_or_session_id"])] += 1

            width = row["image_width"]
            height = row["image_height"]
            resolution_distribution[f"{width}x{height}"] += 1

            file_sizes.append(int(row["file_size_bytes"]))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "totals": totals,
        "by_stream": {
            key: {
                "frame_count": int(value["frame_count"]),
                "storage_mb": round(float(value["storage_mb"]), 4),
            }
            for key, value in sorted(by_stream.items())
        },
        "by_view_type": dict(sorted(by_view_type.items())),
        "by_participant": dict(sorted(by_participant.items())),
        "by_session": dict(sorted(by_session.items())),
        "resolution_distribution": dict(sorted(resolution_distribution.items())),
        "file_size_bytes": {
            "mean": round(sum(file_sizes) / len(file_sizes), 4) if file_sizes else 0.0,
            "min": min(file_sizes) if file_sizes else 0,
            "max": max(file_sizes) if file_sizes else 0,
        },
    }
    return payload


def write_json_atomic(payload: dict[str, object], output_path: Path) -> None:
    """Write JSON atomically by renaming a temporary file into place."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(output_path)


def main() -> None:
    """Generate the CASTLE metadata summary JSON and print a brief summary."""
    args = parse_args()
    manifest_path = DEFAULT_MANIFEST
    output_path = PROJECT_ROOT / args.output
    summary = build_summary(manifest_path)
    write_json_atomic(summary, output_path)

    totals = summary["totals"]
    print(f"Saved metadata summary to {output_path.relative_to(PROJECT_ROOT)}")
    print(
        f"Frames: total={totals['total_frames']} valid={totals['valid_frames']} "
        f"invalid={totals['invalid_frames']}"
    )
    print(f"Streams: {len(summary['by_stream'])} | Resolutions: {len(summary['resolution_distribution'])}")


if __name__ == "__main__":
    main()
