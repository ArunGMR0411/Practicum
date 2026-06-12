#!/usr/bin/env python3
"""Summarize Reverse Personalization failed-frame retry evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "submission_evidence"
    / "03_anonymisation"
    / "05_reverse_personalization"
)
BEFORE = {
    "n_input_frames": 500,
    "n_success": 444,
    "n_failure": 56,
    "AdaFace_reid_rate": "0.13570741097208855",
    "ArcFace_reid_rate": "0.1530317613089509",
    "runtime_total_wall_seconds": "70918.933639",
    "evidence_level": "partial_comparable_with_method_failures",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_output_path(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    try:
        return str(Path(text).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        marker = "/Practicum/"
        if marker in text:
            return text.split(marker, 1)[1]
        return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=DEFAULT_ROOT / "logs" / "rp_retry_padding075_det005_results.jsonl",
    )
    parser.add_argument("--variant-label", default="padding075_det005_face1024")
    args = parser.parse_args()

    rows = read_jsonl(args.results_jsonl)
    normalized: list[dict[str, object]] = []
    for row in rows:
        status = str(row.get("status", ""))
        retry_status = {
            "ok": "RECOVERED",
            "existing": "RECOVERED_EXISTING_OUTPUT",
            "error": "FAILED_AFTER_RETRY",
            "failed": "FAILED_AFTER_RETRY",
        }.get(status, "UNKNOWN")
        normalized.append(
            {
                "relative_path": row.get("relative_path", ""),
                "retry_status": retry_status,
                "raw_status": status,
                "box_count": row.get("box_count", ""),
                "runtime_seconds": row.get("runtime_seconds", ""),
                "output_path": normalize_output_path(row.get("output_path", "")),
                "error": row.get("error", ""),
                "retry_variant": args.variant_label,
            }
        )

    recovered = [r for r in normalized if str(r["retry_status"]).startswith("RECOVERED")]
    failed = [r for r in normalized if r["retry_status"] == "FAILED_AFTER_RETRY"]
    runtimes = [
        float(r["runtime_seconds"])
        for r in normalized
        if r["runtime_seconds"] not in {"", None} and r["raw_status"] != "existing"
    ]
    runtime_total = sum(runtimes)
    after_success = BEFORE["n_success"] + len(recovered)
    after_failure = max(BEFORE["n_failure"] - len(recovered), 0)

    comparison_rows = [
        {
            "stage": "before_retry",
            **BEFORE,
        },
        {
            "stage": "after_failed_frame_retry",
            "n_input_frames": BEFORE["n_input_frames"],
            "n_success": after_success,
            "n_failure": after_failure,
            "AdaFace_reid_rate": "not_recomputed_for_retry_outputs",
            "ArcFace_reid_rate": "not_recomputed_for_retry_outputs",
            "runtime_total_wall_seconds": BEFORE["runtime_total_wall_seconds"],
            "retry_frames_attempted": len(normalized),
            "retry_frames_recovered": len(recovered),
            "retry_frames_still_failed": len(failed),
            "retry_runtime_total_seconds": round(runtime_total, 3),
            "retry_runtime_median_seconds": round(median(runtimes), 3) if runtimes else "not_available",
            "evidence_level": (
                "improved_partial_comparable_with_failed_frame_retry"
                if failed
                else "full_frame_coverage_after_failed_frame_retry"
            ),
        },
    ]

    write_csv(
        args.root / "02_rp_retry_results.csv",
        normalized,
        [
            "relative_path",
            "retry_status",
            "raw_status",
            "box_count",
            "runtime_seconds",
            "output_path",
            "error",
            "retry_variant",
        ],
    )
    write_csv(
        args.root / "03_rp_before_after_fix_comparison.csv",
        comparison_rows,
        [
            "stage",
            "n_input_frames",
            "n_success",
            "n_failure",
            "AdaFace_reid_rate",
            "ArcFace_reid_rate",
            "runtime_total_wall_seconds",
            "retry_frames_attempted",
            "retry_frames_recovered",
            "retry_frames_still_failed",
            "retry_runtime_total_seconds",
            "retry_runtime_median_seconds",
            "evidence_level",
        ],
    )

    try:
        results_log = str(args.results_jsonl.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        results_log = str(args.results_jsonl)

    summary_lines = [
        "# Reverse Personalization Failed-Frame Retry Summary",
        "",
        "## Retry Variant",
        "",
        f"- Variant: `{args.variant_label}`",
        f"- Results log: `{results_log}`",
        f"- Frames attempted: {len(normalized)}",
        f"- Frames recovered: {len(recovered)}",
        f"- Frames still failed: {len(failed)}",
        f"- Retry runtime total seconds: {runtime_total:.3f}",
        "",
        "## Before/After Status",
        "",
        f"- Before retry: 444/500 successes, 56/500 failures.",
        f"- After retry: {after_success}/500 successes, {after_failure}/500 failures.",
        "",
        "## Interpretation",
        "",
        "The retry evidence is consolidated from restartable shard records. Existing outputs produced by the same retry configuration are counted as recovered when execution resumes.",
    ]
    if failed:
        summary_lines.extend(
            [
                "",
                "Remaining failures are retained as method-level failures rather than external blockers. They should be reported as persistent RP crop/embedding failures after targeted retry.",
            ]
        )
    else:
        summary_lines.extend(
            [
                "",
                "The failed-frame retry recovered all previously failed RP frames for output coverage. Metric recomputation is still required before claiming full metric comparability for the recovered outputs.",
            ]
        )
    print(json.dumps({"attempted": len(normalized), "recovered": len(recovered), "failed": len(failed)}, indent=2))


if __name__ == "__main__":
    main()
