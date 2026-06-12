#!/usr/bin/env python3

"""Run NullFace over a manifest while keeping the heavy model stack resident."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import autocast, inference_mode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "nullface"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from diffusers.utils import load_image

from ddm_inversion.inversion_utils import inversion_forward_process, inversion_reverse_process
from ddm_inversion.utils import image_grid
from prompt_to_prompt.ptp_classes import load_512
from src.utils.runtime_tuning import configure_torch_runtime
from utils.face_embedding import FaceEmbeddingExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
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


def expand_crop(image_size: tuple[int, int], box: tuple[int, int, int, int], padding_ratio: float) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    side = max(box_w, box_h)
    padded_side = int(round(side * (1.0 + max(0.0, padding_ratio) * 2.0)))
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


class NullFaceSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = f"cuda:{args.device_index}"
        process_share = 1.0 / max(1, int(args.shard_count))
        self.tuning = configure_torch_runtime("cuda", process_share=process_share)
        os.chdir(BACKEND_ROOT)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sd15.bin",
            image_encoder_folder=None,
        )
        self.pipe.set_ip_adapter_scale(args.ip_adapter_scale)
        self.dtype = self.pipe.dtype
        self.pipe.scheduler = DDIMScheduler.from_config(args.model_id, subfolder="scheduler")
        self.extractor = FaceEmbeddingExtractor(
            ctx_id=args.device_index,
            det_thresh=args.det_thresh,
            det_size=(args.det_size, args.det_size),
            model_path=args.insightface_model_path,
        )

    def anonymise_crop(self, crop_path: Path, mask_path: Path, *, seed: int) -> Image.Image | None:
        try:
            id_embs_inv, id_embs = self.extractor.get_face_embeddings(
                image_path=str(crop_path),
                is_opposite=True,
                seed=seed,
                scale_factor=self.args.id_emb_scale,
                dtype=self.dtype,
                device=self.device,
            )
        except ValueError:
            return None

        self.pipe.scheduler.set_timesteps(self.args.num_diffusion_steps)
        x0 = load_512(str(crop_path), 0, 0, 0, 0, self.device).to(dtype=self.dtype)
        mask_image = load_image(str(mask_path))

        with autocast("cuda"), inference_mode():
            w0 = (self.pipe.vae.encode(x0).latent_dist.mode() * 0.18215).to(dtype=self.dtype)

        _, zs, wts = inversion_forward_process(
            self.pipe,
            w0,
            etas=self.args.eta,
            prompt="",
            cfg_scale=self.args.guidance_scale,
            prog_bar=False,
            num_inference_steps=self.args.num_diffusion_steps,
            ip_adapter_image_embeds=[id_embs_inv],
        )

        generator = torch.manual_seed(seed)
        w0_out, _ = inversion_reverse_process(
            self.pipe,
            xT=wts[self.args.num_diffusion_steps - self.args.skip],
            etas=self.args.eta,
            prompts=[""],
            cfg_scales=[self.args.guidance_scale],
            prog_bar=False,
            zs=zs[: (self.args.num_diffusion_steps - self.args.skip)],
            controller=None,
            ip_adapter_image_embeds=[id_embs],
            init_image=x0,
            mask_image=mask_image,
            generator=generator,
            mask_delay_steps=self.args.mask_delay_steps,
        )

        with autocast("cuda"), inference_mode():
            decoded = self.pipe.vae.decode(1 / 0.18215 * w0_out).sample
        if decoded.dim() < 4:
            decoded = decoded[None, :, :, :]
        return image_grid(decoded)

    def process(self, input_path: Path, output_path: Path, reviewed_boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
        source = Image.open(input_path).convert("RGB")
        if not reviewed_boxes:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path)
            return {"status": "copied", "box_count": 0}

        output = source.copy()
        processed = 0
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for index, box in enumerate(reviewed_boxes):
                crop_window = expand_crop(output.size, box, self.args.crop_padding_ratio)
                left, top, right, bottom = crop_window
                crop = output.crop(crop_window).convert("RGB")
                local_box = (box[0] - left, box[1] - top, box[2] - left, box[3] - top)
                crop_path = tmp / f"crop_{index}.png"
                mask_path = tmp / f"mask_{index}.png"
                crop.save(crop_path)
                build_mask(crop.size, local_box).save(mask_path)
                anon = self.anonymise_crop(crop_path, mask_path, seed=self.args.seed + index)
                if anon is None:
                    continue
                anon = anon.resize(crop.size, Image.Resampling.LANCZOS).convert("RGB")
                output.paste(anon, crop_window)
                processed += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)
        return {"status": "ok", "box_count": processed}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NullFace batch backend requires CUDA")

    jobs = select_job_shard(load_jobs(args.manifest_path), args.shard_index, args.shard_count)
    session = NullFaceSession(args)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for job in jobs:
        input_path = Path(job["input_path"]).resolve()
        output_path = Path(job["output_path"]).resolve()
        reviewed_boxes = [tuple(map(int, box)) for box in job.get("boxes", [])]
        frame_start = time.perf_counter()
        payload = {
            "relative_path": job.get("relative_path", ""),
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
        try:
            result = session.process(input_path, output_path, reviewed_boxes)
            payload.update(result)
        except Exception as exc:
            payload.update({"status": "error", "error": str(exc), "box_count": len(reviewed_boxes)})
        payload["runtime_seconds"] = round(time.perf_counter() - frame_start, 6)
        results.append(payload)

    summary = {
        "jobs_requested": len(jobs),
        "jobs_completed": len(results),
        "total_runtime_seconds": round(time.perf_counter() - started, 6),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "device_index": int(args.device_index),
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
