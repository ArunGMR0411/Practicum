"""Diffusion-based face anonymiser using crop-level inpainting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser, BoundingBox
from src.utils.system_config import resolve_torch_device

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    torch = None  # type: ignore[assignment]

try:
    from diffusers import StableDiffusionInpaintPipeline
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    StableDiffusionInpaintPipeline = None  # type: ignore[assignment]


DEFAULT_MODEL_ID = "data/models/stable-diffusion-inpainting"
DEFAULT_PROMPT = "portrait photo of a different anonymous person, natural face, preserve lighting and pose"
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, deformed, duplicate face, extra eyes, extra mouth, low quality, artifacts"


@dataclass(frozen=True)
class CropWindow:
    left: int
    top: int
    right: int
    bottom: int
    face_box: BoundingBox


class DiffusionAnonymiser(BaseAnonymiser):
    """Replace each face crop with an inpainted synthetic alternative."""

    method_name = "diffusion"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        inference_steps: int = 5,
        guidance_scale: float = 7.0,
        strength_padding_ratio: float = 0.75,
        mask_expansion_ratio: float = 0.0,
        mask_feather_px: int = 0,
        target_resolution: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.inference_steps = int(inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.padding_ratio = float(strength_padding_ratio)
        self.mask_expansion_ratio = float(mask_expansion_ratio)
        self.mask_feather_px = int(mask_feather_px)
        self.target_resolution = int(target_resolution)
        self.device = device or resolve_torch_device()
        self._pipeline: Any | None = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if StableDiffusionInpaintPipeline is None:
            raise NotImplementedError("diffusion unavailable: diffusers is not installed")

        kwargs: dict[str, Any] = {}
        if self.device == "cuda" and torch is not None:
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True

        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            safety_checker=None,
            feature_extractor=None,
            **kwargs
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_attention_slicing()

        if self.device == "cuda":
            try:
                pipeline = pipeline.to(self.device)
            except Exception:
                pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(self.device)

        self._pipeline = pipeline
        return pipeline

    def _expand_crop(self, image: Image.Image, box: BoundingBox) -> CropWindow:
        width, height = image.size
        x1, y1, x2, y2 = box
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = int(round(box_w * self.padding_ratio))
        pad_y = int(round(box_h * self.padding_ratio))
        left = max(0, x1 - pad_x)
        top = max(0, y1 - pad_y)
        right = min(width, x2 + pad_x)
        bottom = min(height, y2 + pad_y)
        return CropWindow(left=left, top=top, right=right, bottom=bottom, face_box=box)

    def _make_mask(self, crop_window: CropWindow) -> Image.Image:
        crop_width = crop_window.right - crop_window.left
        crop_height = crop_window.bottom - crop_window.top
        mask = Image.new("L", (crop_width, crop_height), 0)
        draw = ImageDraw.Draw(mask)
        x1, y1, x2, y2 = crop_window.face_box
        face_width = max(1, x2 - x1)
        face_height = max(1, y2 - y1)
        expand_x = int(round(face_width * self.mask_expansion_ratio))
        expand_y = int(round(face_height * self.mask_expansion_ratio))
        local_box = (
            max(0, x1 - crop_window.left - expand_x),
            max(0, y1 - crop_window.top - expand_y),
            min(crop_width, x2 - crop_window.left + expand_x),
            min(crop_height, y2 - crop_window.top + expand_y),
        )
        draw.ellipse(local_box, fill=255)
        if self.mask_feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.mask_feather_px))
        return mask

    def _prepare_pipeline_inputs(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
        original_size = image.size
        resized_image = image.resize((self.target_resolution, self.target_resolution), Image.Resampling.LANCZOS)
        resized_mask = mask.resize((self.target_resolution, self.target_resolution), Image.Resampling.LANCZOS)
        return resized_image, resized_mask, original_size

    def anonymise(self, image: Image.Image, boxes: list[BoundingBox]) -> AnonymiserResult:
        output = image.copy().convert("RGB")
        valid_boxes = self.validate_boxes(output, boxes)
        if not valid_boxes:
            return AnonymiserResult(
                image=output,
                metadata={
                    "method": self.method_name,
                    "boxes_processed": 0,
                    "tiling_required": False,
                    "model_id": self.model_id,
                },
            )

        pipeline = self._get_pipeline()
        for box in valid_boxes:
            crop_window = self._expand_crop(output, box)
            crop = output.crop((crop_window.left, crop_window.top, crop_window.right, crop_window.bottom)).convert("RGB")
            mask = self._make_mask(crop_window)
            resized_crop, resized_mask, original_size = self._prepare_pipeline_inputs(crop, mask)

            result = pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=resized_crop,
                mask_image=resized_mask,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
            )
            inpainted = result.images[0].resize(original_size, Image.Resampling.LANCZOS)
            composited = Image.composite(inpainted, crop, mask)
            output.paste(composited, (crop_window.left, crop_window.top))

        return AnonymiserResult(
            image=output,
            metadata={
                "method": self.method_name,
                "boxes_processed": len(valid_boxes),
                "tiling_required": False,
                "model_id": self.model_id,
                "device": self.device,
                "target_resolution": self.target_resolution,
                "inference_steps": self.inference_steps,
                "crop_padding_ratio": self.padding_ratio,
                "mask_expansion_ratio": self.mask_expansion_ratio,
                "mask_feather_px": self.mask_feather_px,
            },
        )
