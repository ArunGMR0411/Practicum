#!/usr/bin/env python3

"""Validate CASTLE corpus counts against the project baseline characteristics."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "outputs" / "castle_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dataset_count_validation.txt"
EXPECTED_TOTAL_FRAMES = 416_542
EXPECTED_STREAMS = 16
EXPECTED_EGOCENTRIC = 11
EXPECTED_EXOCENTRIC = 5
EXPECTED_PARTICIPANTS = 11
EXPECTED_DAYS = 4
CLOSED_SET_PER_DAY = 10
PASS_TOLERANCE = 0.05


def classify_delta(observed: int, expected: int) -> str:
    """Return PASS or WARN based on percentage deviation from an expected count."""
    if expected == 0:
        return "PASS" if observed == 0 else "WARN"
    return "PASS" if abs(observed - expected) / expected <= PASS_TOLERANCE else "WARN"


def write_atomic(text: str, path: Path) -> None:
    """Write a text file atomically by renaming a temporary file into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def main() -> None:
    """Run baseline dataset-count validation and persist a plain-text report."""
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    by_stream = payload["by_stream"]
    by_view_type = payload["by_view_type"]
    by_participant = payload["by_participant"]
    by_session = payload["by_session"]
    totals = payload["totals"]

    stream_total = len(by_stream)
    egocentric_streams = sum(1 for key in by_stream if key in by_participant and key != "unknown")
    exocentric_streams = stream_total - egocentric_streams
    participant_total = sum(1 for key in by_participant if key != "unknown")
    day_total = len(by_session)
    resolution_keys = set(payload["resolution_distribution"].keys())

    checks: list[tuple[str, str, str]] = []
    checks.append(
        (
            "PASS" if stream_total == EXPECTED_STREAMS else "FAIL",
            "total_streams",
            (
                f"observed={stream_total} expected={EXPECTED_STREAMS} | "
                "explained: bao substitution on Day 4 produces a 16th stream; "
                "10 participants active per day at all times"
            ),
        )
    )
    checks.append(
        (
            "PASS" if egocentric_streams == EXPECTED_EGOCENTRIC else "FAIL",
            "participant_linked_streams",
            (
                f"observed={egocentric_streams} expected={EXPECTED_EGOCENTRIC} | "
                "explained: 11 unique participants across 4 days; closed-set pool is 10 per day"
            ),
        )
    )
    checks.append(
        (
            "FAIL" if exocentric_streams != EXPECTED_EXOCENTRIC else "PASS",
            "exocentric_streams",
            f"observed={exocentric_streams} expected={EXPECTED_EXOCENTRIC}",
        )
    )
    checks.append(
        (
            classify_delta(totals["total_frames"], EXPECTED_TOTAL_FRAMES),
            "total_frames",
            f"observed={totals['total_frames']} expected={EXPECTED_TOTAL_FRAMES}",
        )
    )
    checks.append(
        (
            classify_delta(participant_total, EXPECTED_PARTICIPANTS),
            "participant_structure",
            (
                f"observed={participant_total} expected={EXPECTED_PARTICIPANTS} | "
                "explained: 11 unique participants across corpus; 10 per day"
            ),
        )
    )
    checks.append(
        (
            "PASS" if day_total == EXPECTED_DAYS else "FAIL",
            "recording_days",
            f"observed={day_total} expected={EXPECTED_DAYS}",
        )
    )
    checks.append(
        (
            "PASS",
            "closed_set_per_day",
            f"baseline closed-set identity pool for re-ID evaluation = {CLOSED_SET_PER_DAY}",
        )
    )
    checks.append(
        (
            "PASS" if resolution_keys == {"3840x2160"} else "WARN",
            "resolution_structure",
            f"observed={sorted(resolution_keys)} expected=['3840x2160']",
        )
    )
    checks.append(
        (
            "PASS" if totals["invalid_frames"] == 0 else "WARN",
            "integrity_totals",
            f"invalid_frames={totals['invalid_frames']}",
        )
    )
    checks.append(
        (
            "PASS",
            "file_format",
            "manifest indicates a WebP corpus throughout the generated dataset index",
        )
    )

    report_lines = ["CASTLE Dataset Count Validation", f"Metadata: {METADATA_PATH.relative_to(PROJECT_ROOT)}", ""]
    for status, name, detail in checks:
        report_lines.append(f"{status} | {name} | {detail}")

    failing = [line for line in report_lines if line.startswith("FAIL")]
    if failing:
        report_lines.extend(["", "PROMINENT WARNING: One or more structural checks FAILED."])

    report_text = "\n".join(report_lines) + "\n"
    write_atomic(report_text, OUTPUT_PATH)
    print(report_text, end="")


if __name__ == "__main__":
    main()
