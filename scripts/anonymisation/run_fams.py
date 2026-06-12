#!/usr/bin/env python3

"""Run the FAMS anonymiser adapter on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = PROJECT_ROOT / "data" / "models" / "stable-diffusion-inpainting"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.fams_anonymiser import FAMSAnonymiser


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
    parser.add_argument("--model-id", default="hkung/face-anon-simple")
    parser.add_argument("--base-model-id", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--face-image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--anonymization-degree", type=float, default=1.25)
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-model-cpu-offload", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "model_id": args.model_id,
        "base_model_id": args.base_model_id,
        "clip_model_id": args.clip_model_id,
        "face_image_size": args.face_image_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "anonymization_degree": args.anonymization_degree,
        "overlap_iou_threshold": args.overlap_iou_threshold,
        "seed": args.seed,
        "enable_model_cpu_offload": not args.disable_model_cpu_offload,
    }
    if args.backend_root is not None:
        kwargs["backend_root"] = args.backend_root
    if args.runner is not None:
        kwargs["runner_path"] = args.runner

    anonymiser = FAMSAnonymiser(**kwargs)
    image = Image.open(args.input).convert("RGB")
    boxes = parse_boxes(args.boxes_json)
    result = anonymiser.anonymise(image, boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
