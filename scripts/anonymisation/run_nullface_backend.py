#!/usr/bin/env python3

"""Backend bridge for NullFace."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "nullface"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from anonymize_face import anonymize_face


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, required=True)
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
    return parser.parse_args()


def load_boxes(path: Path) -> list[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(map(int, box)) for box in payload.get("boxes", [])]


def expand_crop(image_size: tuple[int, int], box: tuple[int, int, int, int], padding_ratio: float) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    side = max(box_w, box_h)
    padded_side = int(math.ceil(side * (1.0 + max(0.0, padding_ratio) * 2.0)))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, int(round(center_x - padded_side / 2.0)))
    top = max(0, int(round(center_y - padded_side / 2.0)))
    right = min(width, left + padded_side)
    bottom = min(height, top + padded_side)
    if right - left < padded_side:
        left = max(0, right - padded_side)
    if bottom - top < padded_side:
        top = max(0, bottom - padded_side)
    return left, top, right, bottom


def build_mask(crop_size: tuple[int, int], local_box: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("RGB", crop_size, "black")
    draw = ImageDraw.Draw(mask)
    draw.ellipse(local_box, fill="white")
    return mask


def main() -> None:
    args = parse_args()
    source = Image.open(args.input).convert("RGB")
    boxes = load_boxes(args.boxes_json)
    if not boxes:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source.save(args.output)
        return

    output = source.copy()
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        for index, box in enumerate(boxes):
            crop_window = expand_crop(output.size, box, args.crop_padding_ratio)
            left, top, right, bottom = crop_window
            crop = output.crop(crop_window).convert("RGB")
            local_box = (box[0] - left, box[1] - top, box[2] - left, box[3] - top)
            crop_path = temp_dir / f"crop_{index}.png"
            mask_path = temp_dir / f"mask_{index}.png"
            log_path = temp_dir / f"log_{index}.txt"
            crop.save(crop_path)
            build_mask(crop.size, local_box).save(mask_path)
            anonymised = anonymize_face(
                image_path=str(crop_path),
                mask_image_path=str(mask_path),
                sd_model_path=args.model_id,
                insightface_model_path=args.insightface_model_path,
                device_num=0,
                guidance_scale=args.guidance_scale,
                num_diffusion_steps=args.num_diffusion_steps,
                eta=args.eta,
                skip=args.skip,
                ip_adapter_scale=args.ip_adapter_scale,
                id_emb_scale=args.id_emb_scale,
                output_log_file=str(log_path),
                det_thresh=args.det_thresh,
                det_size=args.det_size,
                seed=args.seed,
                mask_delay_steps=args.mask_delay_steps,
            )
            if anonymised is None:
                continue
            anonymised = anonymised.resize(crop.size, Image.Resampling.LANCZOS).convert("RGB")
            output.paste(anonymised, crop_window)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)


if __name__ == "__main__":
    main()
