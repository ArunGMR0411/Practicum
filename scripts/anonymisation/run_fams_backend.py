#!/usr/bin/env python3

"""Backend bridge for Face Anonymization Made Simple."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "face_anon_simple"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "data" / "models" / "stable-diffusion-inpainting"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

_hf_hub_download = huggingface_hub.hf_hub_download

def _hf_hub_download_compat(*args, **kwargs):
    kwargs.pop("proxies", None)
    kwargs.pop("resume_download", None)
    return _hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = _hf_hub_download_compat

vendored_diffusers = importlib.import_module("src.diffusers")
sys.modules["diffusers"] = vendored_diffusers

from src.diffusers import AutoencoderKL, DDPMScheduler
from src.anonymisation.fams_backend_utils import (
    build_fams_detections,
    build_fams_detections_from_reviewed_boxes,
    select_detections_for_boxes,
)
from src.utils.compute_policy import build_compute_policy
from src.utils.runtime_tuning import configure_torch_runtime, runtime_device_from_config

from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline
from utils.extractor import FaceType, get_transform_mat
from utils.merger import paste_foreground_onto_background

import face_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boxes-json", type=Path, required=True)
    parser.add_argument("--model-id", default="hkung/face-anon-simple")
    parser.add_argument("--base-model-id", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--face-image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--anonymization-degree", type=float, default=1.25)
    parser.add_argument("--overlap-iou-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    return parser.parse_args()


def load_boxes(path: Path) -> list[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [tuple(map(int, box)) for box in payload.get("boxes", [])]


def base_model_kwargs(base_model_id: str) -> dict[str, object]:
    base_path = Path(base_model_id)
    if base_path.is_dir():
        fp16_vae = base_path / "vae" / "diffusion_pytorch_model.fp16.safetensors"
        fp16_scheduler = base_path / "scheduler" / "scheduler_config.json"
        if fp16_vae.is_file() and fp16_scheduler.is_file():
            return {"variant": "fp16"}
    return {}


def apply_low_vram_profile(args: argparse.Namespace) -> None:
    policy = build_compute_policy()
    if policy.device != "cuda" or not policy.use_low_vram_mode:
        return
    args.face_image_size = min(args.face_image_size, 128)
    args.num_inference_steps = min(args.num_inference_steps, 8)
    args.guidance_scale = min(args.guidance_scale, 2.5)
    args.enable_model_cpu_offload = True


def build_pipeline(args: argparse.Namespace) -> StableDiffusionReferenceNetPipeline:
    device = runtime_device_from_config()
    configure_torch_runtime(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_kwargs = base_model_kwargs(args.base_model_id)

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", use_safetensors=True, torch_dtype=dtype)
    referencenet = ReferenceNetModel.from_pretrained(
        args.model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=dtype
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        args.model_id, subfolder="conditioning_referencenet", use_safetensors=True, torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.base_model_id,
        subfolder="vae",
        use_safetensors=True,
        torch_dtype=dtype,
        **base_kwargs,
    )
    scheduler = DDPMScheduler.from_pretrained(
        args.base_model_id,
        subfolder="scheduler",
        use_safetensors=True,
        **base_kwargs,
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(args.clip_model_id, use_safetensors=True)
    image_encoder = CLIPVisionModel.from_pretrained(args.clip_model_id, use_safetensors=True, torch_dtype=dtype)
    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe.enable_attention_slicing()
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    else:
        pipe = pipe.to(device)
    return pipe


def main() -> None:
    args = parse_args()
    apply_low_vram_profile(args)
    os.chdir(BACKEND_ROOT)
    source = Image.open(args.input).convert("RGB")
    boxes = load_boxes(args.boxes_json)
    if not boxes:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source.save(args.output)
        return

    device = runtime_device_from_config()
    face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd", device=device)
    detections = build_fams_detections_from_reviewed_boxes(
        source,
        boxes,
        face_detector=face_detector,
        get_transform_mat=get_transform_mat,
        face_image_size=args.face_image_size,
        face_type=FaceType.WHOLE_FACE,
    )
    if detections:
        selected = detections
    else:
        landmarks_list = face_detector.get_landmarks(np.array(source)[:, :, :3])
        detections = build_fams_detections(
            source,
            landmarks_list,
            get_transform_mat=get_transform_mat,
            face_image_size=args.face_image_size,
            face_type=FaceType.WHOLE_FACE,
        )
        selected = select_detections_for_boxes(detections, boxes, args.overlap_iou_threshold)
    if not selected:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        source.save(args.output)
        return

    pipe = build_pipeline(args)
    generator = torch.manual_seed(args.seed)
    anon_image = source
    for detection in selected:
        anon_face = pipe(
            source_image=detection.face_image,
            conditioning_image=detection.face_image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            anonymization_degree=args.anonymization_degree,
            width=args.face_image_size,
            height=args.face_image_size,
        ).images[0]
        anon_image = paste_foreground_onto_background(anon_face, anon_image, detection.image_to_face_mat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    anon_image.save(args.output)


if __name__ == "__main__":
    import numpy as np

    main()
