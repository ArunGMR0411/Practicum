#!/usr/bin/env python3

"""Run the NullFace anonymiser adapter on one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.nullface_anonymiser import NullFaceAnonymiser


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
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--insightface-model-path", default=str(Path.home() / ".insightface"))
    parser.add_argument("--crop-padding-ratio", type=float, default=0.9)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--num-diffusion-steps", type=int, default=60)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--skip", type=int, default=40)
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--id-emb-scale", type=float, default=1.0)
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-delay-steps", type=int, default=10)
    args = parser.parse_args()

    kwargs = {
        "model_id": args.model_id,
        "insightface_model_path": args.insightface_model_path,
        "crop_padding_ratio": args.crop_padding_ratio,
        "guidance_scale": args.guidance_scale,
        "num_diffusion_steps": args.num_diffusion_steps,
        "eta": args.eta,
        "skip": args.skip,
        "ip_adapter_scale": args.ip_adapter_scale,
        "id_emb_scale": args.id_emb_scale,
        "det_thresh": args.det_thresh,
        "det_size": args.det_size,
        "seed": args.seed,
        "mask_delay_steps": args.mask_delay_steps,
    }
    if args.backend_root is not None:
        kwargs["backend_root"] = args.backend_root
    if args.runner is not None:
        kwargs["runner_path"] = args.runner

    anonymiser = NullFaceAnonymiser(**kwargs)
    image = Image.open(args.input).convert("RGB")
    boxes = parse_boxes(args.boxes_json)
    result = anonymiser.anonymise(image, boxes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
