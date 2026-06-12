#!/usr/bin/env python3

"""Backend bridge for the official StyleID StyleGAN anonymiser."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

if str(PROJECT_ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_ROOT = PROJECT_ROOT / "third_party" / "styleid"

from src.anonymisation.stylegan_backend_utils import StyleGANComposeConfig, anonymise_styleid_faces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_boxes(path: Path) -> list[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(map(int, box)) for box in payload.get("boxes", [])]


def main() -> None:
    args = parse_args()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    os.chdir(BACKEND_ROOT)

    required = [
        args.model_path,
        BACKEND_ROOT / "pretrained_models" / "psp_celebs_seg_to_face.pt",
        BACKEND_ROOT / "pretrained_models" / "CurricularFace_Backbone.pth",
        BACKEND_ROOT / "pretrained_models" / "mobilenet_celeba.pth",
        BACKEND_ROOT / "pretrained_models" / "unet_model.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"StyleID assets missing: {missing}")

    from styleid import StyleID

    source = Image.open(args.input).convert("RGB")
    boxes = load_boxes(args.boxes_json)
    if not boxes:
        source.save(args.output)
        return

    model = StyleID(checkpoint=str(args.model_path))
    output = anonymise_styleid_faces(model, source, boxes, StyleGANComposeConfig())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)


if __name__ == "__main__":
    main()
