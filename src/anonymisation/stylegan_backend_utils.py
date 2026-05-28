"""Shared StyleID backend helpers with improved compositing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


@dataclass(frozen=True)
class StyleGANComposeConfig:
    crop_context_ratio: float = 0.6
    mask_expansion_ratio: float = 0.22
    mask_feather_px: int = 20
    crop_resolution: int = 1024
    segmentation_threshold: float = 0.35
    generated_face_padding_ratio: float = 0.12
    target_face_expansion_ratio: float = 0.18
    use_seamless_clone: bool = True


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0).clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    return to_pil_image(tensor)


def build_crop_window(
    image: Image.Image,
    box: tuple[int, int, int, int],
    context_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = image.size
    left, top, right, bottom = box
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    pad_x = int(round(box_w * context_ratio))
    pad_y = int(round(box_h * context_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def build_soft_face_mask(
    crop_size: tuple[int, int],
    local_face_box: tuple[int, int, int, int],
    expansion_ratio: float,
    feather_px: int,
) -> Image.Image:
    crop_w, crop_h = crop_size
    mask = Image.new("L", (crop_w, crop_h), 0)
    draw = ImageDraw.Draw(mask)
    left, top, right, bottom = local_face_box
    face_w = max(1, right - left)
    face_h = max(1, bottom - top)
    expand_x = int(round(face_w * expansion_ratio))
    expand_y = int(round(face_h * expansion_ratio))
    ellipse_box = (
        max(0, left - expand_x),
        max(0, top - expand_y),
        min(crop_w, right + expand_x),
        min(crop_h, bottom + expand_y),
    )
    draw.ellipse(ellipse_box, fill=255)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return mask


def tensor_to_mask_image(
    tensor: torch.Tensor | np.ndarray,
    target_size: tuple[int, int],
    threshold: float,
    feather_px: int,
) -> Image.Image:
    if isinstance(tensor, np.ndarray):
        mask_np = np.array(tensor, copy=False)
    else:
        mask_np = tensor.detach().cpu().squeeze(0).float().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np.mean(axis=0)
    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_bin = (mask_np >= threshold).astype(np.uint8) * 255
    mask = Image.fromarray(mask_bin, mode="L").resize(target_size, Image.Resampling.BICUBIC)
    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather_px // 2)))
    return mask


def combine_masks(segmentation_mask: Image.Image, face_prior_mask: Image.Image) -> Image.Image:
    seg = np.asarray(segmentation_mask, dtype=np.float32) / 255.0
    prior = np.asarray(face_prior_mask, dtype=np.float32) / 255.0
    combined = np.clip(seg * prior, 0.0, 1.0)
    return Image.fromarray((combined * 255.0).astype(np.uint8), mode="L")


def mask_to_bbox(mask: Image.Image, padding_ratio: float = 0.0) -> tuple[int, int, int, int] | None:
    mask_np = np.asarray(mask.convert("L"))
    ys, xs = np.nonzero(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    pad_x = int(round(box_w * padding_ratio))
    pad_y = int(round(box_h * padding_ratio))
    width, height = mask.size
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def expand_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    expansion_ratio: float,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_w = max(1, right - left)
    box_h = max(1, bottom - top)
    expand_x = int(round(box_w * expansion_ratio))
    expand_y = int(round(box_h * expansion_ratio))
    width, height = image_size
    return (
        max(0, left - expand_x),
        max(0, top - expand_y),
        min(width, right + expand_x),
        min(height, bottom + expand_y),
    )


def place_generated_face(
    generated_crop: Image.Image,
    generated_mask: Image.Image,
    source_crop: Image.Image,
    target_box: tuple[int, int, int, int],
    padding_ratio: float,
) -> tuple[Image.Image, Image.Image]:
    generated_bbox = mask_to_bbox(generated_mask, padding_ratio=padding_ratio)
    if generated_bbox is None:
        return source_crop, Image.new("L", source_crop.size, 0)

    generated_face = generated_crop.crop(generated_bbox).convert("RGB")
    generated_face_mask = generated_mask.crop(generated_bbox).convert("L")
    target_w = max(1, target_box[2] - target_box[0])
    target_h = max(1, target_box[3] - target_box[1])
    if generated_face.size[0] < 2 or generated_face.size[1] < 2 or target_w < 2 or target_h < 2:
        return source_crop, Image.new("L", source_crop.size, 0)

    resized_face = generated_face.resize((target_w, target_h), Image.BICUBIC)
    resized_face_mask = generated_face_mask.resize((target_w, target_h), Image.BICUBIC)

    placed = source_crop.copy()
    face_layer = Image.new("RGB", source_crop.size, (0, 0, 0))
    face_layer.paste(resized_face, target_box[:2])

    ellipse_mask = build_soft_face_mask(
        source_crop.size,
        target_box,
        expansion_ratio=0.0,
        feather_px=max(8, int(round(target_h * 0.08))),
    )
    placed_mask = Image.new("L", source_crop.size, 0)
    placed_mask.paste(resized_face_mask, target_box[:2])
    mask = combine_masks(placed_mask, ellipse_mask)
    placed = Image.composite(face_layer, placed, mask)
    return placed, mask


def match_local_tone(
    anonymised_crop: Image.Image,
    source_crop: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    anon = np.asarray(anonymised_crop).astype(np.float32)
    src = np.asarray(source_crop).astype(np.float32)
    alpha = np.asarray(mask).astype(np.float32) / 255.0
    active = alpha > 0.05
    if not np.any(active):
        return anonymised_crop

    anon_flat = anon[active]
    src_flat = src[active]
    anon_mean = anon_flat.mean(axis=0)
    src_mean = src_flat.mean(axis=0)
    anon_std = anon_flat.std(axis=0)
    src_std = src_flat.std(axis=0)
    adjusted = (anon - anon_mean) * (src_std / np.maximum(anon_std, 1.0)) + src_mean
    blended = anon * (1.0 - alpha[..., None]) + adjusted * alpha[..., None]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, "RGB")


def seamless_composite(
    source_crop: Image.Image,
    anonymised_crop: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    src_np = np.asarray(source_crop.convert("RGB"))
    anon_np = np.asarray(anonymised_crop.convert("RGB"))
    mask_np = np.asarray(mask.convert("L"))
    if np.count_nonzero(mask_np) == 0:
        return anonymised_crop
    ys, xs = np.nonzero(mask_np)
    center = (int(round(xs.mean())), int(round(ys.mean())))
    blended = cv2.seamlessClone(
        anon_np[:, :, ::-1],
        src_np[:, :, ::-1],
        mask_np,
        center,
        cv2.NORMAL_CLONE,
    )
    return Image.fromarray(blended[:, :, ::-1], mode="RGB")


def anonymise_styleid_faces(
    model: object,
    source: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    config: StyleGANComposeConfig,
) -> Image.Image:
    output = source.copy().convert("RGB")
    to_tensor = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    for box in boxes:
        crop_window = build_crop_window(output, box, config.crop_context_ratio)
        crop = source.crop(crop_window).convert("RGB")
        crop_up = crop.resize((config.crop_resolution, config.crop_resolution), Image.BICUBIC)
        crop_tensor = to_tensor(crop_up).unsqueeze(0).cuda(non_blocking=True)
        with torch.inference_mode():
            segmentation_mask_tensor, _mask_color = model.get_mask(crop_tensor)
            anonymised_tensor = model.rand_face_from_mask(segmentation_mask_tensor, latent_mask=[5, 6, 7, 8, 9])
        anonymised_up = tensor_to_image(anonymised_tensor)

        local_face_box = (
            box[0] - crop_window[0],
            box[1] - crop_window[1],
            box[2] - crop_window[0],
            box[3] - crop_window[1],
        )
        target_face_box = expand_box(
            local_face_box,
            crop.size,
            config.target_face_expansion_ratio,
        )
        segmentation_mask_up = tensor_to_mask_image(
            segmentation_mask_tensor,
            (config.crop_resolution, config.crop_resolution),
            config.segmentation_threshold,
            config.mask_feather_px,
        )
        placed_face_crop, soft_mask = place_generated_face(
            anonymised_up,
            segmentation_mask_up,
            crop,
            target_face_box,
            config.generated_face_padding_ratio,
        )
        if np.count_nonzero(np.asarray(soft_mask)) == 0:
            face_prior_mask = build_soft_face_mask(
                crop.size,
                local_face_box,
                config.mask_expansion_ratio,
                config.mask_feather_px,
            )
            fallback_face = anonymised_up.resize(crop.size, Image.BICUBIC)
            soft_mask = face_prior_mask
            placed_face_crop = Image.composite(fallback_face, crop, soft_mask)
        tone_matched = match_local_tone(placed_face_crop, crop, soft_mask)
        composited = Image.composite(tone_matched, crop, soft_mask)
        if config.use_seamless_clone:
            try:
                composited = seamless_composite(crop, composited, soft_mask)
            except cv2.error:
                pass
        output.paste(composited, crop_window)

    return ImageOps.exif_transpose(output).convert("RGB")
