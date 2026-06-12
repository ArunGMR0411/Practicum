#!/usr/bin/env python3

"""Backend bridge for the official G2Face reversible anonymiser."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "g2face"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--donor-strategy", default="nearest_non_self")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    os.chdir(BACKEND_ROOT)

    from option import options
    from test import process_image

    payload = json.loads(args.boxes_json.read_text(encoding="utf-8"))
    if not payload.get("boxes"):
        Image.open(args.input).convert("RGB").save(args.output)
        return

    required = [
        args.model_path,
        BACKEND_ROOT / "weights" / "epoch_20.pth",
        BACKEND_ROOT / "pretrain" / "ms1mv3_arcface_r50.pth",
        BACKEND_ROOT / "model" / "d3dfr" / "BFM" / "BFM_model_front.mat",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"G2Face assets missing: {missing}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_dir = tmp / "input"
        output_dir = tmp / "output"
        input_dir.mkdir()
        source_name = "input.png"
        shutil.copyfile(args.input, input_dir / source_name)

        old_argv = sys.argv
        sys.argv = ["run_g2face_backend.py", "--celebahq_path", str(input_dir), "--batch_size", "1"]
        try:
            opt = options()
        finally:
            sys.argv = old_argv
        opt.device = "cuda"
        process_image(
            opt=opt,
            image_path=str(input_dir),
            save_path=str(output_dir),
            checkpoint_path=str(args.model_path),
            batch_size=1,
        )
        candidate = output_dir / "any" / source_name
        if not candidate.is_file():
            raise RuntimeError(f"G2Face did not produce expected output at {candidate}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate, args.output)


if __name__ == "__main__":
    main()
