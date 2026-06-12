#!/usr/bin/env python3
"""Run RiDDLE with landmark alignment and conservative visual fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv"
DEFAULT_BOXES = ROOT / "outputs/02_face_detection/13_anonymisation_protocol_face_boxes.csv"

# ArcFace five-point template, scaled from 112 to the RiDDLE 256-pixel input.
TEMPLATE = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
) * (256.0 / 112.0)


@dataclass
class AlignedFace:
    box: tuple[int, int, int, int]
    transform: np.ndarray
    aligned: np.ndarray
    landmark_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--boxes", type=Path, default=DEFAULT_BOXES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--minimum-face-size", type=int, default=48)
    parser.add_argument("--landmark-threshold", type=float, default=0.90)
    parser.add_argument("--mask-scale", type=float, default=0.88)
    parser.add_argument("--mask-feather", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_riddle(source_root: Path, asset_root: Path):
    sys.path.insert(0, str(source_root))
    from argparse import Namespace
    from mapper.latent_id_mappers import TransformerMapperSplit
    from models.psp import pSp
    from models.stylegan2.model import Generator

    e4e_path = asset_root / "e4e_ffhq_encode_256.pt"
    stylegan_path = asset_root / "stylegan2-ffhq-256.pt"
    mapper_path = asset_root / "iteration_90000.pt"
    for path in (e4e_path, stylegan_path, mapper_path):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint = torch.load(e4e_path, map_location="cpu", weights_only=False)
    options = dict(checkpoint["opts"])
    options["checkpoint_path"] = str(e4e_path)
    options["stylegan_weights"] = str(stylegan_path)
    encoder = pSp(Namespace(**options)).eval().cuda()

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


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def password_latents(generator, keys: list[str], seed: int) -> torch.Tensor:
    latents = []
    for key in keys:
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        item_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        cpu_generator = torch.Generator(device="cpu").manual_seed(item_seed)
        latents.append(torch.randn(512, generator=cpu_generator))
    z = torch.stack(latents).cuda(non_blocking=True)
    return generator.style(z).unsqueeze(1).repeat(1, 14, 1)


@torch.inference_mode()
def generate_candidates(encoder, generator, mapper, aligned: np.ndarray, key: str, count: int, seed: int) -> np.ndarray:
    tensor = to_tensor(aligned).unsqueeze(0).cuda(non_blocking=True)
    codes = encoder.encoder(tensor)
    if encoder.opts.start_from_latent_avg:
        codes = codes + encoder.latent_avg.repeat(codes.shape[0], 1, 1)
    codes = codes.repeat(count, 1, 1)
    keys = [f"{key}:candidate-{index}" for index in range(count)]
    encrypted = mapper(torch.cat([codes, password_latents(generator, keys, seed)], dim=-1))
    images, _ = generator([encrypted], input_is_latent=True, randomize_noise=False, truncation=1)
    images = images.add(1.0).mul(127.5).clamp(0, 255).byte()
    return images.permute(0, 2, 3, 1).cpu().numpy()


def padded_crop(image: np.ndarray, box: tuple[int, int, int, int], ratio: float = 0.65):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    pad = side * ratio
    return (
        max(0, int(x1 - pad)),
        max(0, int(y1 - pad)),
        min(image.shape[1], int(x2 + pad)),
        min(image.shape[0], int(y2 + pad)),
    )


def find_alignment(mtcnn: MTCNN, image: np.ndarray, box: tuple[int, int, int, int], threshold: float) -> AlignedFace | None:
    left, top, right, bottom = padded_crop(image, box)
    crop = Image.fromarray(image[top:bottom, left:right])
    detected, probabilities, landmarks = mtcnn.detect(crop, landmarks=True)
    if detected is None or landmarks is None or probabilities is None:
        return None

    detected = np.asarray(detected, dtype=np.float32)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    target_center = np.asarray(
        [(box[0] + box[2]) / 2 - left, (box[1] + box[3]) / 2 - top],
        dtype=np.float32,
    )
    centers = np.column_stack(((detected[:, 0] + detected[:, 2]) / 2, (detected[:, 1] + detected[:, 3]) / 2))
    index = int(np.argmin(np.linalg.norm(centers - target_center, axis=1)))
    probability = float(probabilities[index])
    if probability < threshold:
        return None

    source = landmarks[index].astype(np.float32)
    source[:, 0] += left
    source[:, 1] += top
    transform, _ = cv2.estimateAffinePartial2D(source, TEMPLATE, method=cv2.LMEDS)
    if transform is None:
        return None
    aligned = cv2.warpAffine(
        image,
        transform,
        (256, 256),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return AlignedFace(box=box, transform=transform, aligned=aligned, landmark_probability=probability)


def candidate_quality(mtcnn: MTCNN, candidate: np.ndarray) -> float:
    boxes, probabilities, landmarks = mtcnn.detect(Image.fromarray(candidate), landmarks=True)
    if boxes is None or probabilities is None or landmarks is None:
        return -1.0
    index = int(np.argmax(probabilities))
    box = boxes[index]
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (256.0 * 256.0)
    points = landmarks[index]
    symmetry = 1.0 - min(1.0, abs((points[0, 1] - points[1, 1])) / 32.0)
    return float(probabilities[index]) + 0.15 * min(area / 0.35, 1.0) + 0.10 * symmetry


def colour_match(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source = cv2.cvtColor(candidate, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    for channel in range(3):
        source_mean, source_std = source[..., channel].mean(), source[..., channel].std()
        target_mean, target_std = target[..., channel].mean(), target[..., channel].std()
        source[..., channel] = (source[..., channel] - source_mean) * (
            max(target_std, 1.0) / max(source_std, 1.0)
        ) + target_mean
    return cv2.cvtColor(np.clip(source, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def composite_aligned(
    image: np.ndarray,
    face: AlignedFace,
    generated: np.ndarray,
    mask_scale: float,
    feather: float,
) -> None:
    generated = colour_match(generated, face.aligned)
    inverse = cv2.invertAffineTransform(face.transform)
    height, width = image.shape[:2]
    warped = cv2.warpAffine(
        generated,
        inverse,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.zeros((256, 256), dtype=np.float32)
    axes = (int(86 * mask_scale), int(106 * mask_scale))
    cv2.ellipse(mask, (128, 132), axes, 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    mask = cv2.warpAffine(mask, inverse, (width, height), flags=cv2.INTER_LINEAR)
    mask = np.clip(mask[..., None], 0.0, 1.0)
    image[:] = np.clip(image * (1.0 - mask) + warped * mask, 0, 255).astype(np.uint8)


def layered_fallback(image: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    crop = Image.fromarray(image[y1:y2, x1:x2])
    width, height = crop.size
    if width <= 0 or height <= 0:
        return
    tiny = crop.resize((max(1, width // 24), max(1, height // 24)), Image.Resampling.BILINEAR)
    transformed = tiny.resize((width, height), Image.Resampling.NEAREST)
    transformed = transformed.filter(ImageFilter.GaussianBlur(radius=max(4, min(width, height) / 18)))
    image[y1:y2, x1:x2] = np.asarray(transformed)


def frame_generation_eligibility(item) -> tuple[bool, list[str]]:
    """Restrict portrait generation to conditions supported by the source model."""
    reasons: list[str] = []
    if str(item.face_count_category) != "single_face":
        reasons.append("not_single_face")
    if str(item.face_scale_category) not in {"medium", "large"}:
        reasons.append("unsupported_face_scale")
    if str(item.edge_partial_face) == "yes":
        reasons.append("edge_or_partial_face")
    if str(item.profile_occluded_face) == "yes":
        reasons.append("profile_or_occluded_face")
    if str(item.blur_low_sharpness) == "yes":
        reasons.append("motion_blur_or_low_sharpness")
    return not reasons, reasons


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    mtcnn = MTCNN(keep_all=True, device="cuda", min_face_size=20)
    encoder, generator, mapper = load_riddle(args.source_root, args.asset_root)
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()
    for frame_index, item in enumerate(manifest.itertuples(index=False), start=1):
        relative_path = str(item.relative_path)
        output_path = output_images / relative_path
        if output_path.exists() and not args.overwrite:
            continue
        frame_started = time.perf_counter()
        try:
            raw_path = ROOT / str(item.image_path)
            image = np.asarray(Image.open(raw_path).convert("RGB")).copy()
            boxes = grouped_boxes.get(relative_path, [])
            generation_eligible, eligibility_reasons = frame_generation_eligibility(item)
            generated_faces = fallback_faces = 0
            reasons: list[str] = list(eligibility_reasons)
            for box_index, box in enumerate(boxes):
                size = min(box[2] - box[0], box[3] - box[1])
                if not generation_eligible:
                    layered_fallback(image, box)
                    fallback_faces += 1
                    continue
                if size < args.minimum_face_size:
                    layered_fallback(image, box)
                    fallback_faces += 1
                    reasons.append("small_face")
                    continue
                face = find_alignment(mtcnn, image, box, args.landmark_threshold)
                if face is None:
                    layered_fallback(image, box)
                    fallback_faces += 1
                    reasons.append("landmark_failure")
                    continue
                candidates = generate_candidates(
                    encoder,
                    generator,
                    mapper,
                    face.aligned,
                    f"{relative_path}:{box_index}",
                    args.candidate_count,
                    args.seed,
                )
                scores = [candidate_quality(mtcnn, candidate) for candidate in candidates]
                best = int(np.argmax(scores))
                if scores[best] < 0.90:
                    layered_fallback(image, box)
                    fallback_faces += 1
                    reasons.append("candidate_quality_failure")
                    continue
                composite_aligned(image, face, candidates[best], args.mask_scale, args.mask_feather)
                generated_faces += 1

            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(output_path, format="WEBP", quality=92, method=4)
            rows.append(
                {
                    "relative_path": relative_path,
                    "method": "riddle_landmark_hardened",
                    "output_path": (
                        output_path.relative_to(ROOT).as_posix()
                        if output_path.is_relative_to(ROOT)
                        else output_path.as_posix()
                    ),
                    "status": "ok" if boxes else "copied_no_face",
                    "box_count": len(boxes),
                    "generation_eligible": generation_eligible,
                    "eligibility_reasons": "|".join(eligibility_reasons),
                    "generated_faces": generated_faces,
                    "fallback_faces": fallback_faces,
                    "fallback_reasons": "|".join(sorted(set(reasons))),
                    "runtime_seconds": time.perf_counter() - frame_started,
                }
            )
        except Exception as exc:
            failures.append({"relative_path": relative_path, "reason": f"{type(exc).__name__}: {exc}"})
        if frame_index % 10 == 0 or frame_index == len(manifest):
            elapsed = time.perf_counter() - started
            eta = elapsed / frame_index * (len(manifest) - frame_index)
            print(
                f"frames={frame_index}/{len(manifest)} elapsed={elapsed/60:.1f}m "
                f"eta={eta/60:.1f}m vram={torch.cuda.max_memory_allocated()/2**30:.1f}GiB",
                flush=True,
            )
            write_csv(
                args.output_dir / "manifest.csv",
                rows,
                [
                    "relative_path", "method", "output_path", "status", "box_count",
                    "generation_eligible", "eligibility_reasons",
                    "generated_faces", "fallback_faces", "fallback_reasons", "runtime_seconds",
                ],
            )
            write_csv(args.output_dir / "failure_log.csv", failures, ["relative_path", "reason"])

    elapsed = time.perf_counter() - started
    summary = [{
        "method": "riddle_landmark_hardened",
        "input_frames": len(manifest),
        "successful_frames": len(rows),
        "failed_frames": len(failures),
        "generated_faces": sum(row["generated_faces"] for row in rows),
        "fallback_faces": sum(row["fallback_faces"] for row in rows),
        "runtime_total_seconds": elapsed,
        "runtime_mean_seconds": elapsed / max(len(manifest), 1),
        "candidate_count": args.candidate_count,
        "minimum_face_size": args.minimum_face_size,
        "landmark_threshold": args.landmark_threshold,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }]
    write_csv(args.output_dir / "runtime_summary.csv", summary, list(summary[0]))
    print(summary[0])


if __name__ == "__main__":
    main()
