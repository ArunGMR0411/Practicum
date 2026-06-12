#!/usr/bin/env python3

"""Run a deterministic control-pack batch through one anonymiser method."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.registry import build_anonymiser_registry
from src.utils.compute_policy import build_compute_policy

DEFAULT_PACK_ROOT = PROJECT_ROOT / "outputs" / "submission_evidence" / "generative_control_package"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "submission_evidence" / "generative_control_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    parser.add_argument("--method", required=True, help="Anonymiser method name from the registry.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-workers", type=int, default=0, help="0 means auto from compute policy")
    return parser.parse_args()


def load_pack_summary(pack_root: Path) -> dict[str, Any]:
    summary_path = pack_root / "control_pack_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Control-pack summary missing: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_boxes(pack_root: Path, boxes_json: str) -> list[tuple[int, int, int, int]]:
    payload = json.loads((pack_root / boxes_json).read_text(encoding="utf-8"))
    return [tuple(map(int, box)) for box in payload.get("boxes", [])]


def output_path_for(output_root: Path, method: str, relative_path: str) -> Path:
    return output_root / method / relative_path


def process_one_item(
    anonymiser: Any,
    pack_root: Path,
    output_root: Path,
    method: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    relative_path = str(item["relative_path"])
    input_path = pack_root / "data" / "castle2024" / "raw" / relative_path
    output_path = output_path_for(output_root, method, relative_path)
    row: dict[str, Any] = {
        "relative_path": relative_path,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "method": method,
        "status": "pending",
        "box_count": int(item.get("box_count", 0)),
        "runtime_seconds": "",
        "error": "",
        "metadata_json": "",
    }
    image = Image.open(input_path).convert("RGB")
    boxes = load_boxes(pack_root, str(item["boxes_json"]))
    image.load()
    frame_start = time.perf_counter()
    result = anonymiser.anonymise(image, boxes)
    runtime = time.perf_counter() - frame_start
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path)
    row.update(
        {
            "status": "ok",
            "runtime_seconds": round(runtime, 6),
            "metadata_json": json.dumps(result.metadata, sort_keys=True),
        }
    )
    return row


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    pack_root = args.pack_root.resolve()
    output_root = args.output_root.resolve()
    summary = load_pack_summary(pack_root)
    policy = build_compute_policy()
    registry = build_anonymiser_registry()
    if args.method not in registry:
        raise ValueError(f"Unknown method {args.method!r}; available={sorted(registry)}")

    anonymiser = registry[args.method]
    box_files = list(summary.get("box_files", []))
    if args.max_images is not None:
        box_files = box_files[: args.max_images]

    manifest_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    max_workers = args.max_workers or policy.generative_control_max_workers

    if max_workers <= 1:
        for item in box_files:
            try:
                row = process_one_item(anonymiser, pack_root, output_root, args.method, item)
            except Exception as exc:
                row = {
                    "relative_path": str(item["relative_path"]),
                    "input_path": str(pack_root / "data" / "castle2024" / "raw" / str(item["relative_path"])),
                    "output_path": str(output_path_for(output_root, args.method, str(item["relative_path"]))),
                    "method": args.method,
                    "status": "error",
                    "box_count": int(item.get("box_count", 0)),
                    "runtime_seconds": "",
                    "error": str(exc),
                    "metadata_json": "",
                }
                errors.append({"relative_path": str(item["relative_path"]), "error": str(exc)})
                manifest_rows.append(row)
                if not args.continue_on_error:
                    break
            else:
                manifest_rows.append(row)
    else:
        indexed_rows: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(process_one_item, anonymiser, pack_root, output_root, args.method, item): index
                for index, item in enumerate(box_files)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                item = box_files[index]
                try:
                    indexed_rows[index] = future.result()
                except Exception as exc:
                    indexed_rows[index] = {
                        "relative_path": str(item["relative_path"]),
                        "input_path": str(pack_root / "data" / "castle2024" / "raw" / str(item["relative_path"])),
                        "output_path": str(output_path_for(output_root, args.method, str(item["relative_path"]))),
                        "method": args.method,
                        "status": "error",
                        "box_count": int(item.get("box_count", 0)),
                        "runtime_seconds": "",
                        "error": str(exc),
                        "metadata_json": "",
                    }
                    errors.append({"relative_path": str(item["relative_path"]), "error": str(exc)})
        manifest_rows = [indexed_rows[index] for index in sorted(indexed_rows)]

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"{args.method}_control_manifest.csv"
    summary_path = output_root / f"{args.method}_control_summary.json"
    fieldnames = [
        "relative_path",
        "input_path",
        "output_path",
        "method",
        "status",
        "box_count",
        "runtime_seconds",
        "error",
        "metadata_json",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    result_summary = {
        "method": args.method,
        "pack_root": str(pack_root),
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "requested_frames": len(box_files),
        "processed_rows": len(manifest_rows),
        "ok_count": ok_count,
        "error_count": len(errors),
        "max_workers": int(max_workers),
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
        "errors": errors,
    }
    summary_path.write_text(json.dumps(result_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result_summary, indent=2))
    if errors:
        raise SystemExit(1)
    return result_summary


def main() -> None:
    args = parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
