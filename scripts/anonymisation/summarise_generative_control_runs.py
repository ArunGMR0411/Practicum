#!/usr/bin/env python3

"""Summarise and gate generative control-pack method runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "submission_evidence" / "generative_control_runs"
DEFAULT_METHODS = ("stylegan", "reverse_personalization", "reface")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method", action="append", dest="methods", default=[])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "generative_control_run_summary.json",
    )
    parser.add_argument("--allow-missing-methods", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarise_method(output_root: Path, method: str) -> dict[str, Any]:
    manifest_path = output_root / f"{method}_control_manifest.csv"
    summary_path = output_root / f"{method}_control_summary.json"
    if not manifest_path.is_file():
        return {
            "method": method,
            "present": False,
            "ready_for_larger_run": False,
            "reason": f"missing manifest: {manifest_path}",
        }

    rows = read_manifest(manifest_path)
    missing_outputs = [
        row.get("output_path", "")
        for row in rows
        if row.get("status") == "ok" and not Path(row.get("output_path", "")).is_file()
    ]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    runtimes = [
        float(row["runtime_seconds"])
        for row in rows
        if row.get("runtime_seconds") not in {"", None} and row.get("status") == "ok"
    ]
    summary_payload: dict[str, Any] = {}
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    ready = bool(rows) and not error_rows and not missing_outputs
    return {
        "method": method,
        "present": True,
        "ready_for_larger_run": ready,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "row_count": len(rows),
        "ok_count": len(rows) - len(error_rows),
        "error_count": len(error_rows),
        "missing_output_count": len(missing_outputs),
        "mean_runtime_seconds": round(sum(runtimes) / len(runtimes), 6) if runtimes else None,
        "max_runtime_seconds": round(max(runtimes), 6) if runtimes else None,
        "errors": [
            {
                "relative_path": row.get("relative_path", ""),
                "error": row.get("error", ""),
            }
            for row in error_rows
        ],
        "missing_outputs": missing_outputs,
        "source_summary": summary_payload,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    methods = tuple(args.methods) if args.methods else DEFAULT_METHODS
    method_summaries = [summarise_method(output_root, method) for method in methods]
    missing_methods = [item["method"] for item in method_summaries if not item["present"]]
    failed_methods = [item["method"] for item in method_summaries if not item["ready_for_larger_run"]]
    ready = not failed_methods and (args.allow_missing_methods or not missing_methods)
    if args.allow_missing_methods:
        ready = all(
            item["ready_for_larger_run"] for item in method_summaries if item["present"]
        ) and any(item["present"] for item in method_summaries)
    return {
        "output_root": str(output_root),
        "methods": list(methods),
        "ready_for_larger_run": ready,
        "missing_methods": missing_methods,
        "failed_methods": failed_methods,
        "method_summaries": method_summaries,
    }


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["ready_for_larger_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
