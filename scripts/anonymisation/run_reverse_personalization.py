#!/usr/bin/env python3

"""Thin CLI wrapper around the official Reverse Personalization implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "reverse_personalization"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from anonymize_faces_in_image import anonymize_faces_in_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--attribute-prompt", default=None, help="Optional attribute-control prompt.")
    parser.add_argument("--sd-model-path", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--insightface-model-path", default="~/.insightface")
    parser.add_argument("--device-num", type=int, default=0)
    parser.add_argument("--skip", type=float, default=0.7)
    parser.add_argument("--id-emb-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=-10.0)
    parser.add_argument("--num-inversion-steps", type=int, default=100)
    parser.add_argument("--face-image-size", type=int, default=1024)
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--use-model-cpu-offload",
        action="store_true",
        help="Enable Diffusers model CPU offload for lower-VRAM systems.",
    )
    parser.add_argument(
        "--enable-face-detection",
        action="store_true",
        help="Use the backend's own face detection and alignment path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = anonymize_faces_in_image(
        input_image=args.input,
        attribute_prompt=args.attribute_prompt,
        sd_model_path=args.sd_model_path,
        insightface_model_path=args.insightface_model_path,
        device_num=args.device_num,
        skip=args.skip,
        id_emb_scale=args.id_emb_scale,
        guidance_scale=args.guidance_scale,
        num_inversion_steps=args.num_inversion_steps,
        face_image_size=args.face_image_size,
        det_thresh=args.det_thresh,
        ip_adapter_scale=args.ip_adapter_scale,
        det_size=args.det_size,
        seed=args.seed,
        enable_face_detection=args.enable_face_detection,
        use_model_cpu_offload=args.use_model_cpu_offload,
    )
    result.save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
