#!/usr/bin/env python3

"""Run Reverse Personalization over a manifest while keeping models resident."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "third_party" / "reverse_personalization"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch
import face_alignment
from transformers import CLIPVisionModelWithProjection

from src.utils.runtime_tuning import configure_torch_runtime
from sdxl.leditspp.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as StableDiffusionPipelineXL_LEDITS
from sdxl.leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from utils.extractor import extract_faces
from utils.face_embedding import FaceEmbeddingExtractor
from utils.merger import paste_foreground_onto_background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--attribute-prompt", default=None)
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
    parser.add_argument("--enable-face-detection", action="store_true")
    parser.add_argument("--use-manifest-boxes", action="store_true")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.75)
    parser.add_argument("--mask-expansion-ratio", type=float, default=0.2)
    parser.add_argument("--mask-feather-px", type=int, default=18)
    parser.add_argument("--use-model-cpu-offload", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--process-share", type=float, default=None)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--results-jsonl", type=Path, default=None)
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


def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class ReversePersonalizationSession:
    def __init__(self, args: argparse.Namespace) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Reverse Personalization batch backend requires CUDA")

        self.args = args
        self.dtype = torch.float16
        self.device = f"cuda:{args.device_num}"
        process_share = args.process_share if args.process_share is not None else 1.0 / max(1, int(args.shard_count))
        process_share = max(0.05, min(1.0, float(process_share)))
        self.tuning = configure_torch_runtime("cuda", process_share=process_share)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=self.dtype,
        )
        self.pipe = StableDiffusionPipelineXL_LEDITS.from_pretrained(
            args.sd_model_path,
            image_encoder=self.image_encoder,
            torch_dtype=self.dtype,
        )
        self.pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
            args.sd_model_path,
            subfolder="scheduler",
            algorithm_type="sde-dpmsolver++",
            solver_order=2,
        )
        if args.use_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter-FaceID",
            subfolder=None,
            weight_name="ip-adapter-faceid_sdxl.bin",
            image_encoder_folder=None,
        )
        self.pipe.set_ip_adapter_scale(args.ip_adapter_scale)
        self.extractor = FaceEmbeddingExtractor(
            ctx_id=args.device_num,
            det_thresh=args.det_thresh,
            det_size=(args.det_size, args.det_size),
            model_path=args.insightface_model_path,
        )
        self.fa = None
        if args.enable_face_detection:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                face_detector="sfd",
            )

    def _valid_boxes(self, image: Image.Image, boxes: list[Any] | None) -> list[tuple[int, int, int, int]]:
        if not boxes:
            return []
        width, height = image.size
        valid: list[tuple[int, int, int, int]] = []
        for item in boxes:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in item)
            left = max(0, min(x1, width))
            top = max(0, min(y1, height))
            right = max(0, min(x2, width))
            bottom = max(0, min(y2, height))
            if right > left and bottom > top:
                valid.append((left, top, right, bottom))
        return valid

    def _crop_window(self, image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        width, height = image.size
        x1, y1, x2, y2 = box
        face_width = max(1, x2 - x1)
        face_height = max(1, y2 - y1)
        pad_x = int(round(face_width * self.args.crop_padding_ratio))
        pad_y = int(round(face_height * self.args.crop_padding_ratio))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    def _soft_face_mask(
        self,
        crop_size: tuple[int, int],
        box: tuple[int, int, int, int],
        crop_window: tuple[int, int, int, int],
    ) -> Image.Image:
        crop_left, crop_top, _crop_right, _crop_bottom = crop_window
        x1, y1, x2, y2 = box
        crop_box = [x1 - crop_left, y1 - crop_top, x2 - crop_left, y2 - crop_top]
        face_width = max(1, crop_box[2] - crop_box[0])
        face_height = max(1, crop_box[3] - crop_box[1])
        expand_x = int(round(face_width * self.args.mask_expansion_ratio))
        expand_y = int(round(face_height * self.args.mask_expansion_ratio))
        ellipse = (
            max(0, crop_box[0] - expand_x),
            max(0, crop_box[1] - expand_y),
            min(crop_size[0], crop_box[2] + expand_x),
            min(crop_size[1], crop_box[3] + expand_y),
        )
        mask = Image.new("L", crop_size, 0)
        ImageDraw.Draw(mask).ellipse(ellipse, fill=255)
        if self.args.mask_feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.args.mask_feather_px))
        return mask

    def _anonymise_face_image(self, face_image: Image.Image) -> Image.Image:
        prepared_face = face_image.convert("RGB").resize(
            (self.args.face_image_size, self.args.face_image_size),
            Image.Resampling.LANCZOS,
        )
        id_embs_inv, id_embs = self.extractor.get_face_embeddings(
            image_path=prepared_face,
            seed=self.args.seed,
            scale_factor=self.args.id_emb_scale,
            dtype=self.dtype,
            device=self.device,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.args.seed)
        with torch.inference_mode():
            _ = self.pipe.invert(
                image=prepared_face,
                num_inversion_steps=self.args.num_inversion_steps,
                skip=self.args.skip,
                source_guidance_scale=self.args.guidance_scale,
                ip_adapter_image_embeds=[id_embs_inv],
                generator=generator,
            )
            return self.pipe(
                prompt="",
                negative_prompt=self.args.attribute_prompt,
                ip_adapter_image_embeds=[id_embs],
                num_images_per_prompt=1,
                generator=generator,
                guidance_scale=self.args.guidance_scale,
                timesteps=self.pipe.scheduler.timesteps,
                latents=self.pipe.init_latents,
                num_inference_steps=self.args.num_inversion_steps,
            ).images[0].convert("RGB")

    def _process_manifest_boxes(self, source: Image.Image, boxes: list[Any] | None) -> tuple[Image.Image, int]:
        valid_boxes = self._valid_boxes(source, boxes)
        anon_image = source.copy()
        for box in valid_boxes:
            crop_window = self._crop_window(source, box)
            crop = anon_image.crop(crop_window).convert("RGB")
            anon_crop = self._anonymise_face_image(crop).resize(crop.size, Image.Resampling.LANCZOS)
            mask = self._soft_face_mask(crop.size, box, crop_window)
            composited = Image.composite(anon_crop, crop, mask)
            anon_image.paste(composited, crop_window[:2])
        return anon_image, len(valid_boxes)

    def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        should_process: bool,
        boxes: list[Any] | None = None,
    ) -> dict[str, Any]:
        source = Image.open(input_path).convert("RGB")
        if not should_process:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source.save(output_path)
            return {"status": "copied", "box_count": 0}

        if self.args.use_manifest_boxes:
            anon_image, faces_processed = self._process_manifest_boxes(source, boxes)
        elif self.fa is not None:
            anon_image = source
            face_images, image_to_face_matrices = extract_faces(self.fa, source, self.args.face_image_size)
            faces_processed = 0
            for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
                anon_face_image = self._anonymise_face_image(face_image)
                if image_to_face_mat is not None:
                    anon_image = paste_foreground_onto_background(anon_face_image, anon_image, image_to_face_mat)
                else:
                    anon_image = anon_face_image
                faces_processed += 1
        else:
            anon_image = source
            face_images = [source]
            image_to_face_matrices = [None]
            faces_processed = 0
            for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
                anon_face_image = self._anonymise_face_image(face_image)
                if image_to_face_mat is not None:
                    anon_image = paste_foreground_onto_background(anon_face_image, anon_image, image_to_face_mat)
                else:
                    anon_image = anon_face_image
                faces_processed += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        anon_image.save(output_path)
        return {"status": "ok", "box_count": faces_processed}


def main() -> None:
    args = parse_args()
    jobs = select_job_shard(load_jobs(args.manifest_path), args.shard_index, args.shard_count)
    session = ReversePersonalizationSession(args)
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
            if args.resume_existing and output_path.is_file():
                result = {"status": "existing", "box_count": len(job.get("boxes", []))}
            else:
                result = session.process(
                    input_path,
                    output_path,
                    should_process=bool(job.get("boxes", [])),
                    boxes=job.get("boxes", []),
                )
            payload.update(result)
        except Exception as exc:
            payload.update({"status": "error", "error": str(exc), "box_count": len(job.get("boxes", []))})
        payload["runtime_seconds"] = round(time.perf_counter() - frame_start, 6)
        results.append(payload)
        append_jsonl(args.results_jsonl, payload)

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
