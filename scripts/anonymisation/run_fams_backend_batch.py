#!/usr/bin/env python3

"""Run Face Anonymization Made Simple over a manifest while keeping models resident."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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

from src.anonymisation.fams_backend_utils import build_fams_detections, select_detections_for_boxes
from src.diffusers import AutoencoderKL, DDPMScheduler
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
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_job_shard(jobs: list[dict[str, Any]], shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count <= 1:
        return jobs
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard selection: index={shard_index}, count={shard_count}")
    return [job for idx, job in enumerate(jobs) if idx % shard_count == shard_index]


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


class FAMSSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = runtime_device_from_config()
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("FAMS requires CUDA; CPU fallback is disabled")
        process_share = 1.0 / max(1, int(args.shard_count))
        self.tuning = configure_torch_runtime(self.device, process_share=process_share)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base_kwargs = base_model_kwargs(args.base_model_id)
        self.face_detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            face_detector="sfd",
            device=self.device,
        )

        self.unet = UNet2DConditionModel.from_pretrained(
            args.model_id, subfolder="unet", use_safetensors=True, torch_dtype=self.dtype
        )
        self.referencenet = ReferenceNetModel.from_pretrained(
            args.model_id, subfolder="referencenet", use_safetensors=True, torch_dtype=self.dtype
        )
        self.conditioning_referencenet = ReferenceNetModel.from_pretrained(
            args.model_id, subfolder="conditioning_referencenet", use_safetensors=True, torch_dtype=self.dtype
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.base_model_id, subfolder="vae", use_safetensors=True, torch_dtype=self.dtype, **self.base_kwargs
        )
        self.scheduler = DDPMScheduler.from_pretrained(
            args.base_model_id, subfolder="scheduler", use_safetensors=True, **self.base_kwargs
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(args.clip_model_id, use_safetensors=True)
        self.image_encoder = CLIPVisionModel.from_pretrained(args.clip_model_id, use_safetensors=True, torch_dtype=self.dtype)
        self.pipe = StableDiffusionReferenceNetPipeline(
            unet=self.unet,
            referencenet=self.referencenet,
            conditioning_referencenet=self.conditioning_referencenet,
            vae=self.vae,
            feature_extractor=self.feature_extractor,
            image_encoder=self.image_encoder,
            scheduler=self.scheduler,
        )
        self.pipe.enable_attention_slicing()
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            if args.enable_model_cpu_offload:
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe = self.pipe.to(self.device)
        else:
            self.pipe = self.pipe.to(self.device)
        self.generator = torch.manual_seed(args.seed)

    def process(self, input_path: Path, output_path: Path, reviewed_boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
        source = Image.open(input_path).convert("RGB")
        if not reviewed_boxes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path)
            return {"status": "copied", "box_count": 0}

        landmarks_list = self.face_detector.get_landmarks(np.array(source)[:, :, :3])
        detections = build_fams_detections(
            source,
            landmarks_list,
            get_transform_mat=get_transform_mat,
            face_image_size=self.args.face_image_size,
            face_type=FaceType.WHOLE_FACE,
        )
        selected = select_detections_for_boxes(detections, reviewed_boxes, self.args.overlap_iou_threshold)
        if not selected:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path)
            return {"status": "copied", "box_count": 0}

        anon_image = source
        for detection in selected:
            anon_face = self.pipe(
                source_image=detection.face_image,
                conditioning_image=detection.face_image,
                num_inference_steps=self.args.num_inference_steps,
                guidance_scale=self.args.guidance_scale,
                generator=self.generator,
                anonymization_degree=self.args.anonymization_degree,
                width=self.args.face_image_size,
                height=self.args.face_image_size,
            ).images[0]
            anon_image = paste_foreground_onto_background(anon_face, anon_image, detection.image_to_face_mat)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        anon_image.save(output_path)
        return {"status": "ok", "box_count": len(selected)}


def main() -> None:
    args = parse_args()
    apply_low_vram_profile(args)
    os.chdir(BACKEND_ROOT)
    jobs = select_job_shard(load_jobs(args.manifest_path), args.shard_index, args.shard_count)
    session = FAMSSession(args)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for job in jobs:
        input_path = Path(job["input_path"]).resolve()
        output_path = Path(job["output_path"]).resolve()
        frame_start = time.perf_counter()
        payload = {
            "relative_path": job.get("relative_path", ""),
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
        try:
            result = session.process(
                input_path,
                output_path,
                reviewed_boxes=[tuple(map(int, box)) for box in job.get("boxes", [])],
            )
            payload.update(result)
        except Exception as exc:
            payload.update({"status": "error", "error": str(exc), "box_count": len(job.get("boxes", []))})
        payload["runtime_seconds"] = round(time.perf_counter() - frame_start, 6)
        results.append(payload)

    summary = {
        "jobs_requested": len(jobs),
        "jobs_completed": len(results),
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "runtime_tuning": {
            "cpu_threads": session.tuning.cpu_threads,
            "interop_threads": session.tuning.interop_threads,
            "tf32_enabled": session.tuning.tf32_enabled,
            "cudnn_benchmark": session.tuning.cudnn_benchmark,
            "float32_matmul_precision": session.tuning.float32_matmul_precision,
        },
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
