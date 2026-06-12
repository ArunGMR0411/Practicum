#!/usr/bin/env python3
"""Run RiDDLE on the reviewed 500-frame anonymisation protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
DEFAULT_BOXES = ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv"
DEFAULT_OUTPUT = ROOT / "outputs/03_anonymisation/12_riddle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--face-batch-size", type=int, default=64)
    parser.add_argument("--image-chunk-size", type=int, default=50)
    parser.add_argument("--padding", type=float, default=0.40)
    parser.add_argument("--save-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_models(source_root: Path, asset_root: Path):
    sys.path.insert(0, str(source_root))
    from mapper.latent_id_mappers import TransformerMapperSplit
    from models.psp import pSp
    from models.stylegan2.model import Generator

    e4e_path = asset_root / "e4e_ffhq_encode_256.pt"
    stylegan_path = asset_root / "stylegan2-ffhq-256.pt"
    mapper_path = asset_root / "iteration_90000.pt"

    e4e_checkpoint = torch.load(e4e_path, map_location="cpu", weights_only=False)
    e4e_options = dict(e4e_checkpoint["opts"])
    e4e_options["checkpoint_path"] = str(e4e_path)
    e4e_options["stylegan_weights"] = str(stylegan_path)
    encoder = pSp(Namespace(**e4e_options)).eval().cuda()

    generator = Generator(256, 512, 8).eval().cuda()
    generator.load_state_dict(
        torch.load(stylegan_path, map_location="cpu", weights_only=False)["g_ema"],
        strict=False,
    )
    mapper = TransformerMapperSplit(
        split_list=[4, 4, 6],
        normalize_type="layernorm",
        add_linear=True,
        add_pos_embedding=True,
    ).eval().cuda()
    mapper.load_state_dict(
        torch.load(mapper_path, map_location="cpu", weights_only=False)["mapper_state_dict"]
    )
    return encoder, generator, mapper


def square_crop(box: tuple[int, int, int, int], width: int, height: int, padding: float):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(width, int(round(cx + side / 2)))
    bottom = min(height, int(round(cy + side / 2)))
    return left, top, right, bottom


def to_tensor(crop: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor


def deterministic_passwords(generator, keys: list[str], seed: int) -> torch.Tensor:
    latents = []
    for key in keys:
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        item_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        cpu_generator = torch.Generator(device="cpu").manual_seed(item_seed)
        latents.append(torch.randn(512, generator=cpu_generator))
    z = torch.stack(latents).cuda(non_blocking=True)
    return generator.style(z).unsqueeze(1).repeat(1, 14, 1)


@torch.inference_mode()
def encrypt_batch(encoder, generator, mapper, batch: torch.Tensor, keys: list[str], seed: int) -> np.ndarray:
    batch = batch.cuda(non_blocking=True)
    codes = encoder.encoder(batch)
    if encoder.opts.start_from_latent_avg:
        codes = codes + encoder.latent_avg.repeat(codes.shape[0], 1, 1)
    passwords = deterministic_passwords(generator, keys, seed)
    encrypted = mapper(torch.cat([codes, passwords], dim=-1))
    images, _ = generator(
        [encrypted], input_is_latent=True, randomize_noise=False, truncation=1
    )
    images = images.add(1.0).mul(127.5).clamp(0, 255).byte()
    return images.permute(0, 2, 3, 1).cpu().numpy()


def composite_face(
    image: np.ndarray,
    generated: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    face_box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = crop_rect
    target_h, target_w = bottom - top, right - left
    generated = cv2.resize(generated, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    x1, y1, x2, y2 = face_box
    center = (int((x1 + x2) / 2 - left), int((y1 + y2) / 2 - top))
    axes = (max(2, int((x2 - x1) * 0.62)), max(2, int((y2 - y1) * 0.68)))
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    sigma = max(2.0, 0.06 * max(axes))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = np.clip(mask[..., None], 0.0, 1.0)
    original = image[top:bottom, left:right].astype(np.float32)
    image[top:bottom, left:right] = np.clip(
        original * (1.0 - mask) + generated.astype(np.float32) * mask, 0, 255
    ).astype(np.uint8)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def recorded_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="WEBP", quality=90, method=4)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    manifest = pd.read_csv(args.manifest)
    box_rows = pd.read_csv(args.boxes)
    grouped_boxes = {
        key: [tuple(map(int, values)) for values in group[["x1", "y1", "x2", "y2"]].values]
        for key, group in box_rows.groupby("image_id", sort=False)
    }
    output_images = args.output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "01_riddle_500_manifest.csv"
    failure_path = args.output_dir / "03_riddle_failure_log.csv"
    summary_path = args.output_dir / "02_riddle_runtime_summary.csv"

    encoder, generator, mapper = load_models(args.source_root, args.asset_root)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result_rows: list[dict] = []
    failures: list[dict] = []

    for chunk_start in range(0, len(manifest), args.image_chunk_size):
        chunk_started = time.perf_counter()
        chunk = manifest.iloc[chunk_start : chunk_start + args.image_chunk_size]
        images: dict[str, np.ndarray] = {}
        tasks: list[dict] = []
        existing_rows: list[dict] = []
        for row in chunk.itertuples(index=False):
            relative_path = str(row.relative_path)
            raw_path = ROOT / str(row.image_path)
            output_path = output_images / relative_path
            boxes = grouped_boxes.get(relative_path, [])
            if output_path.exists() and not args.overwrite:
                existing_rows.append(
                    {
                        "relative_path": relative_path,
                        "method": "riddle",
                        "output_path": recorded_path(output_path),
                        "status": "existing",
                        "box_count": len(boxes),
                        "runtime_seconds": 0.0,
                    }
                )
                continue
            try:
                image = np.asarray(Image.open(raw_path).convert("RGB")).copy()
                images[relative_path] = image
                for box_index, box in enumerate(boxes):
                    rect = square_crop(box, image.shape[1], image.shape[0], args.padding)
                    left, top, right, bottom = rect
                    if right <= left or bottom <= top:
                        raise ValueError(f"invalid crop {rect}")
                    tasks.append(
                        {
                            "relative_path": relative_path,
                            "box_index": box_index,
                            "box": box,
                            "rect": rect,
                            "tensor": to_tensor(image[top:bottom, left:right]),
                        }
                    )
            except Exception as exc:
                failures.append({"relative_path": relative_path, "stage": "load", "reason": str(exc)})

        for batch_start in range(0, len(tasks), args.face_batch_size):
            batch_tasks = tasks[batch_start : batch_start + args.face_batch_size]
            try:
                batch = torch.stack([task["tensor"] for task in batch_tasks])
                keys = [f"{task['relative_path']}:{task['box_index']}" for task in batch_tasks]
                generated = encrypt_batch(encoder, generator, mapper, batch, keys, args.seed)
                for task, output_crop in zip(batch_tasks, generated, strict=True):
                    composite_face(
                        images[task["relative_path"]], output_crop, task["rect"], task["box"]
                    )
            except Exception as exc:
                for task in batch_tasks:
                    failures.append(
                        {"relative_path": task["relative_path"], "stage": "inference", "reason": str(exc)}
                    )
                raise

        chunk_elapsed = time.perf_counter() - chunk_started
        weights = {key: max(1, len(grouped_boxes.get(key, []))) for key in images}
        total_weight = sum(weights.values()) or 1
        with ThreadPoolExecutor(max_workers=args.save_workers) as executor:
            future_paths = {
                executor.submit(save_image, output_images / relative_path, image): relative_path
                for relative_path, image in images.items()
            }
            for future in as_completed(future_paths):
                relative_path = future_paths[future]
                output_path = output_images / relative_path
                try:
                    future.result()
                    result_rows.append(
                        {
                            "relative_path": relative_path,
                            "method": "riddle",
                            "output_path": recorded_path(output_path),
                            "status": "ok" if grouped_boxes.get(relative_path) else "copied_no_face",
                            "box_count": len(grouped_boxes.get(relative_path, [])),
                            "runtime_seconds": chunk_elapsed * weights[relative_path] / total_weight,
                        }
                    )
                except Exception as exc:
                    failures.append({"relative_path": relative_path, "stage": "save", "reason": str(exc)})
        result_rows.extend(existing_rows)
        write_csv(
            manifest_path,
            sorted(result_rows, key=lambda item: item["relative_path"]),
            ["relative_path", "method", "output_path", "status", "box_count", "runtime_seconds"],
        )
        write_csv(failure_path, failures, ["relative_path", "stage", "reason"])
        elapsed = time.perf_counter() - started
        completed = min(chunk_start + len(chunk), len(manifest))
        eta = elapsed / completed * (len(manifest) - completed)
        print(
            f"frames={completed}/{len(manifest)} faces={len(tasks)} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
            f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    successful = [row for row in result_rows if row["status"] in {"ok", "copied_no_face", "existing"}]
    summary = [
        {
            "method": "riddle",
            "input_frames": len(manifest),
            "successful_frames": len(successful),
            "failed_frames": len(manifest) - len(successful),
            "face_boxes": sum(len(grouped_boxes.get(str(path), [])) for path in manifest.relative_path),
            "total_runtime_seconds": elapsed,
            "mean_runtime_seconds": elapsed / len(manifest),
            "face_batch_size": args.face_batch_size,
            "image_chunk_size": args.image_chunk_size,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_name": f"cuda_accelerator_{round(torch.cuda.get_device_properties(0).total_memory / 1024**3)}gb",
            "runtime_note": "End-to-end generation and compositing runtime; per-image rows allocate shared batch time by face count.",
        }
    ]
    write_csv(summary_path, summary, list(summary[0]))
    print(json.dumps(summary[0], indent=2))


if __name__ == "__main__":
    main()
