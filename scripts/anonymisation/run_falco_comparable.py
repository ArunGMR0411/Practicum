#!/usr/bin/env python3
"""Run a faithful FALCO crop protocol on the reviewed 500-frame surface."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
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
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
DEFAULT_BOXES = ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv"
DEFAULT_OUTPUT = ROOT / "outputs/03_anonymisation/13_falco"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/falco_castle_work"))
    parser.add_argument("--pool-size", type=int, default=60000)
    parser.add_argument("--pool-batch-size", type=int, default=64)
    parser.add_argument("--inversion-batch-size", type=int, default=64)
    parser.add_argument("--optimisation-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--id-margin", type=float, default=0.0)
    parser.add_argument("--lambda-id", type=float, default=10.0)
    parser.add_argument("--lambda-attr", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--padding", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--save-workers", type=int, default=10)
    parser.add_argument("--keep-work-cache", action="store_true")
    return parser.parse_args()


def square_crop(box: tuple[int, int, int, int], width: int, height: int, padding: float):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (
        max(0, int(round(cx - side / 2))),
        max(0, int(round(cy - side / 2))),
        min(width, int(round(cx + side / 2))),
        min(height, int(round(cy + side / 2))),
    )


def to_tensor(crop: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    return torch.from_numpy(resized).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def composite_face(image, generated, rect, box) -> None:
    left, top, right, bottom = rect
    target_h, target_w = bottom - top, right - left
    generated = cv2.resize(generated, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    x1, y1, x2, y2 = box
    center = (int((x1 + x2) / 2 - left), int((y1 + y2) / 2 - top))
    axes = (max(2, int((x2 - x1) * 0.62)), max(2, int((y2 - y1) * 0.68)))
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, 0.06 * max(axes)))
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


def load_models(source_root: Path, id_margin: float):
    sys.path.insert(0, str(source_root))
    os.chdir(source_root)
    import clip
    from lib.config import FARL_PRETRAIN_MODEL
    from lib.id_loss import IDLoss
    from lib.attr_loss import AttrLoss
    from models.load_generator import load_generator
    from models.psp import pSp

    checkpoint = source_root / "models/pretrained/e4e/e4e_ffhq_encode.pt"
    e4e_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    options = dict(e4e_data["opts"])
    options["checkpoint_path"] = str(checkpoint)
    options["device"] = "cuda"
    encoder = pSp(Namespace(**options)).eval().cuda()
    generator = load_generator(
        "stylegan2_ffhq1024",
        latent_is_w=True,
        CHECKPOINT_DIR=str(source_root / "models/pretrained/genforce"),
    ).eval().cuda()
    identity = IDLoss(id_margin=id_margin).eval().cuda()
    attribute = AttrLoss(feat_ext="farl").eval().cuda()

    farl, _ = clip.load("ViT-B/16", device="cuda", jit=False)
    farl_state = torch.load(
        source_root / "models/pretrained/farl" / FARL_PRETRAIN_MODEL,
        map_location="cpu",
        weights_only=False,
    )
    farl.load_state_dict(farl_state["state_dict"], strict=False)
    farl.eval().float()
    farl_transform = transforms.Compose(
        [
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return encoder, generator, identity, attribute, farl, farl_transform


@torch.inference_mode()
def build_fake_pool(generator, farl, transform, args):
    latent_path = args.work_dir / "fake_latents.pt"
    feature_path = args.work_dir / "fake_farl_features.pt"
    if latent_path.exists() and feature_path.exists():
        return torch.load(latent_path), torch.load(feature_path)
    latents, features = [], []
    cpu_rng = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    for start in range(0, args.pool_size, args.pool_batch_size):
        count = min(args.pool_batch_size, args.pool_size - start)
        z = torch.randn(count, generator.dim_z, generator=cpu_rng).cuda(non_blocking=True)
        wp = generator.get_w(z, truncation=0.7)
        images = generator(wp)
        encoded = farl.encode_image(transform(images)).float()
        encoded = torch.nn.functional.normalize(encoded, dim=1)
        latents.append(wp.cpu())
        features.append(encoded.cpu())
        if start == 0 or (start + count) % 2048 == 0 or start + count == args.pool_size:
            elapsed = time.perf_counter() - started
            done = start + count
            eta = elapsed / done * (args.pool_size - done)
            print(f"fake_pool={done}/{args.pool_size} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    latent_tensor = torch.cat(latents)
    feature_tensor = torch.cat(features)
    torch.save(latent_tensor, latent_path)
    torch.save(feature_tensor, feature_path)
    return latent_tensor, feature_tensor


def prepare_faces(manifest, grouped_boxes, padding):
    images: dict[str, np.ndarray] = {}
    tasks: list[dict] = []
    for row in manifest.itertuples(index=False):
        relative_path = str(row.relative_path)
        image = np.asarray(Image.open(ROOT / str(row.image_path)).convert("RGB")).copy()
        images[relative_path] = image
        for box_index, box in enumerate(grouped_boxes.get(relative_path, [])):
            rect = square_crop(box, image.shape[1], image.shape[0], padding)
            left, top, right, bottom = rect
            tasks.append(
                {
                    "relative_path": relative_path,
                    "box_index": box_index,
                    "box": box,
                    "rect": rect,
                    "source": to_tensor(image[top:bottom, left:right]),
                }
            )
    return images, tasks


@torch.inference_mode()
def invert_and_pair(tasks, encoder, generator, farl, transform, fake_features, batch_size, work_dir):
    real_path = work_dir / "real_latents.pt"
    pair_path = work_dir / "fake_pair_indices.pt"
    if real_path.exists() and pair_path.exists():
        return torch.load(real_path), torch.load(pair_path)
    real_latents, real_features = [], []
    for start in range(0, len(tasks), batch_size):
        batch = torch.stack([task["source"] for task in tasks[start : start + batch_size]]).cuda()
        codes = encoder.encoder(batch)
        if encoder.opts.start_from_latent_avg:
            codes = codes + encoder.latent_avg.repeat(codes.shape[0], 1, 1)
        feats = torch.nn.functional.normalize(farl.encode_image(transform(batch)).float(), dim=1)
        real_latents.append(codes.cpu())
        real_features.append(feats.cpu())
        print(f"inversion={min(start+batch_size,len(tasks))}/{len(tasks)}", flush=True)
    latents = torch.cat(real_latents)
    real_features = torch.cat(real_features).cuda()
    fake_features = fake_features.cuda()
    pair_indices = []
    for start in range(0, len(real_features), 256):
        similarity = real_features[start : start + 256] @ fake_features.T
        pair_indices.append(similarity.argmax(dim=1).cpu())
    pairs = torch.cat(pair_indices)
    torch.save(latents, real_path)
    torch.save(pairs, pair_path)
    return latents, pairs


def optimise_faces(tasks, real_latents, fake_latents, pairs, generator, identity, attribute, args, images):
    started = time.perf_counter()
    for start in range(0, len(tasks), args.optimisation_batch_size):
        end = min(start + args.optimisation_batch_size, len(tasks))
        current_tasks = tasks[start:end]
        real = real_latents[start:end].cuda()
        fake = fake_latents[pairs[start:end]].cuda()
        source = torch.stack([task["source"] for task in current_tasks]).cuda()
        trainable = torch.nn.Parameter(fake[:, 3:8, :].clone())
        optimiser = torch.optim.Adam([trainable], lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimiser, milestones=[int(0.75 * args.epochs), int(0.9 * args.epochs)], gamma=0.8
        )
        for _ in range(args.epochs):
            optimiser.zero_grad(set_to_none=True)
            latent = torch.cat([real[:, :3, :], trainable, real[:, 8:, :]], dim=1)
            generated = generator(latent)
            loss = args.lambda_id * identity(generated, source) + args.lambda_attr * attribute(source, generated)
            loss.backward()
            optimiser.step()
            scheduler.step()
        with torch.inference_mode():
            latent = torch.cat([real[:, :3, :], trainable, real[:, 8:, :]], dim=1)
            generated = generator(latent).add(1).mul(127.5).clamp(0, 255).byte()
            generated = generated.permute(0, 2, 3, 1).cpu().numpy()
        for task, output_crop in zip(current_tasks, generated, strict=True):
            composite_face(images[task["relative_path"]], output_crop, task["rect"], task["box"])
        elapsed = time.perf_counter() - started
        done = end
        eta = elapsed / done * (len(tasks) - done)
        print(
            f"optimised_faces={done}/{len(tasks)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m "
            f"vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
            flush=True,
        )


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="WEBP", quality=90, method=4)


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.manifest = args.manifest.resolve()
    args.boxes = args.boxes.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is disabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_images = args.output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    box_rows = pd.read_csv(args.boxes)
    grouped_boxes = {
        key: [tuple(map(int, values)) for values in group[["x1", "y1", "x2", "y2"]].values]
        for key, group in box_rows.groupby("image_id", sort=False)
    }
    run_started = time.perf_counter()
    encoder, generator, identity, attribute, farl, farl_transform = load_models(
        args.source_root, args.id_margin
    )
    torch.cuda.reset_peak_memory_stats()
    fake_latents, fake_features = build_fake_pool(generator, farl, farl_transform, args)
    images, tasks = prepare_faces(manifest, grouped_boxes, args.padding)
    real_latents, pairs = invert_and_pair(
        tasks,
        encoder,
        generator,
        farl,
        farl_transform,
        fake_features,
        args.inversion_batch_size,
        args.work_dir,
    )
    optimise_faces(
        tasks,
        real_latents,
        fake_latents,
        pairs,
        generator,
        identity,
        attribute,
        args,
        images,
    )

    with ThreadPoolExecutor(max_workers=args.save_workers) as executor:
        futures = {
            executor.submit(save_image, output_images / relative_path, image): relative_path
            for relative_path, image in images.items()
        }
        for future in as_completed(futures):
            future.result()

    elapsed = time.perf_counter() - run_started
    mean_runtime = elapsed / len(manifest)
    rows = []
    for relative_path in manifest.relative_path.astype(str):
        output_path = output_images / relative_path
        rows.append(
            {
                "relative_path": relative_path,
                "method": "falco",
                "output_path": recorded_path(output_path),
                "status": "ok" if grouped_boxes.get(relative_path) else "copied_no_face",
                "box_count": len(grouped_boxes.get(relative_path, [])),
                "runtime_seconds": mean_runtime,
            }
        )
    write_csv(
        args.output_dir / "01_falco_500_manifest.csv",
        rows,
        ["relative_path", "method", "output_path", "status", "box_count", "runtime_seconds"],
    )
    summary = [
        {
            "method": "falco",
            "input_frames": len(manifest),
            "successful_frames": len(rows),
            "failed_frames": 0,
            "face_boxes": len(tasks),
            "fake_pool_size": args.pool_size,
            "epochs": args.epochs,
            "id_margin": args.id_margin,
            "total_runtime_seconds": elapsed,
            "mean_runtime_seconds": mean_runtime,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_name": f"cuda_accelerator_{round(torch.cuda.get_device_properties(0).total_memory / 1024**3)}gb",
        }
    ]
    write_csv(args.output_dir / "02_falco_runtime_summary.csv", summary, list(summary[0]))
    write_csv(args.output_dir / "03_falco_failure_log.csv", [], ["relative_path", "stage", "reason"])
    if not args.keep_work_cache:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    print(summary[0])


if __name__ == "__main__":
    main()
