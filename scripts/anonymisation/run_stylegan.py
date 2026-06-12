#!/usr/bin/env python3

"""Run the StyleGAN anonymiser adapter on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.stylegan_anonymiser import StyleGANAnonymiser


def parse_boxes(path: Path | None) -> list[tuple[int, int, int, int]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(map(int, box)) for box in payload.get("boxes", [])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path)
    parser.add_argument("--backend-root", type=Path)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kwargs = {"truncation_psi": args.truncation_psi, "seed": args.seed}
    if args.backend_root is not None:
        kwargs["backend_root"] = args.backend_root
    if args.runner is not None:
        kwargs["runner_path"] = args.runner
    if args.model_path is not None:
        kwargs["model_path"] = args.model_path

    anonymiser = StyleGANAnonymiser(**kwargs)
    image = Image.open(args.input).convert("RGB")
    boxes = parse_boxes(args.boxes_json)
    result = anonymiser.anonymise(image, boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
