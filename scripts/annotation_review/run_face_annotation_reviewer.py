#!/usr/bin/env python3

"""Run the face-annotation reviewer app."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_SRC = PROJECT_ROOT / "app" / "src"
if str(APP_SRC) not in sys.path:
    sys.path.insert(0, str(APP_SRC))

def main() -> None:
    """Parse CLI args and start the reviewer server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--mode",
        choices=["face_boxes", "multimodal_presence"],
        default="face_boxes",
        help="Review mode to launch.",
    )
    parser.add_argument(
        "--review-csv",
        default="",
        help="Optional shared CSV for multimodal_presence mode.",
    )
    parser.add_argument(
        "--face-pack-root",
        default="",
        help="Optional face annotation pack root override for face_boxes mode.",
    )
    parser.add_argument(
        "--face-tasks-csv",
        default="",
        help="Optional face task CSV override for face_boxes mode.",
    )
    args = parser.parse_args()
    if args.review_csv:
        os.environ["CASTLE_TEXT_REVIEW_CSV"] = args.review_csv
        os.environ["CASTLE_MULTIMODAL_REVIEW_CSV"] = args.review_csv
    if args.face_pack_root:
        os.environ["CASTLE_FACE_PACK_ROOT"] = args.face_pack_root
    if args.face_tasks_csv:
        os.environ["CASTLE_FACE_TASKS_CSV"] = args.face_tasks_csv

    from privacy_pipeline_app.face_annotation_reviewer import run_server

    run_server(host=args.host, port=args.port, mode=args.mode)


if __name__ == "__main__":
    main()
