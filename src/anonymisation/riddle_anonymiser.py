"""RiDDLE single-image adapter for the research demonstrator App.

Requires CUDA plus RiDDLE source and assets configured via environment variables
or default project paths. When unavailable, ``reason`` is non-empty and the App
must treat apply as a failed advanced method (honest fallback).
"""

from __future__ import annotations

import hashlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser, BoundingBox

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    os.environ.get("RIDDLE_SOURCE_ROOT", str(PROJECT_ROOT / "third_party" / "riddle"))
)
DEFAULT_ASSETS = Path(
    os.environ.get("RIDDLE_ASSET_ROOT", str(PROJECT_ROOT / "data" / "models" / "riddle"))
)


class RiddleAnonymiser(BaseAnonymiser):
    method_name = "riddle"

    def __init__(
        self,
        *,
        source_root: Path = DEFAULT_SOURCE,
        asset_root: Path = DEFAULT_ASSETS,
        padding: float = 0.40,
        seed: int = 20260712,
    ) -> None:
        self.source_root = Path(source_root)
        self.asset_root = Path(asset_root)
        self.padding = float(padding)
        self.seed = int(seed)
        self._models = None
        self.reason = self._availability_reason()

    def _availability_reason(self) -> str:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            return f"riddle requires torch ({exc})"
        if not torch.cuda.is_available():
            return "riddle requires CUDA"
        e4e = self.asset_root / "e4e_ffhq_encode_256.pt"
        stylegan = self.asset_root / "stylegan2-ffhq-256.pt"
        mapper = self.asset_root / "iteration_90000.pt"
        if not self.source_root.is_dir():
            return f"riddle source missing at {self.source_root} (set RIDDLE_SOURCE_ROOT)"
        missing = [str(p) for p in (e4e, stylegan, mapper) if not p.is_file()]
        if missing:
            return f"riddle assets missing (set RIDDLE_ASSET_ROOT): {missing}"
        return ""

    def _load_models(self):
        if self._models is not None:
            return self._models
        import torch

        root = str(self.source_root.resolve())
        sys.path = [root] + [p for p in sys.path if p != root]
        for key in list(sys.modules):
            if key == "models" or key.startswith("models.") or key == "mapper" or key.startswith("mapper."):
                del sys.modules[key]
        from models.psp import pSp  # type: ignore
        from models.stylegan2.model import Generator  # type: ignore
        from mapper.latent_id_mappers import TransformerMapperSplit  # type: ignore

        e4e_path = self.asset_root / "e4e_ffhq_encode_256.pt"
        stylegan_path = self.asset_root / "stylegan2-ffhq-256.pt"
        mapper_path = self.asset_root / "iteration_90000.pt"
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
        self._models = (encoder, generator, mapper)
        return self._models

    @staticmethod
    def _square_crop(box, width, height, padding):
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        left = max(0, int(round(cx - side / 2)))
        top = max(0, int(round(cy - side / 2)))
        right = min(width, int(round(cx + side / 2)))
        bottom = min(height, int(round(cy + side / 2)))
        return left, top, right, bottom

    def anonymise(self, image: Image.Image, boxes: list[BoundingBox]) -> AnonymiserResult:
        if self.reason:
            raise RuntimeError(self.reason)
        import cv2
        import torch

        encoder, generator, mapper = self._load_models()
        valid = self.validate_boxes(image, boxes)
        arr = np.asarray(image.convert("RGB")).copy()
        h, w = arr.shape[:2]
        for index, box in enumerate(valid):
            rect = self._square_crop(box, w, h, self.padding)
            left, top, right, bottom = rect
            if right <= left or bottom <= top:
                continue
            crop = arr[top:bottom, left:right]
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            batch = (
                torch.from_numpy(resized)
                .permute(2, 0, 1)
                .float()
                .div_(127.5)
                .sub_(1.0)
                .unsqueeze(0)
                .cuda()
            )
            digest = hashlib.sha256(f"{self.seed}:riddle:{index}:{box}".encode()).digest()
            item_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            cpu_gen = torch.Generator(device="cpu").manual_seed(item_seed)
            z = torch.randn(1, 512, generator=cpu_gen).cuda()
            with torch.inference_mode():
                codes = encoder.encoder(batch)
                if getattr(encoder.opts, "start_from_latent_avg", False):
                    codes = codes + encoder.latent_avg.repeat(codes.shape[0], 1, 1)
                passwords = generator.style(z).unsqueeze(1).repeat(1, 14, 1)
                encrypted = mapper(torch.cat([codes, passwords], dim=-1))
                images, _ = generator(
                    [encrypted], input_is_latent=True, randomize_noise=False, truncation=1
                )
                generated = (
                    images.add(1.0).mul(127.5).clamp(0, 255).byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()[0]
                )
            target_h, target_w = bottom - top, right - left
            generated = cv2.resize(generated, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            x1, y1, x2, y2 = box
            center = (int((x1 + x2) / 2 - left), int((y1 + y2) / 2 - top))
            axes = (max(2, int((x2 - x1) * 0.62)), max(2, int((y2 - y1) * 0.68)))
            mask = np.zeros((target_h, target_w), dtype=np.float32)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(2.0, 0.06 * max(axes)))
            mask = np.clip(mask[..., None], 0.0, 1.0)
            original = arr[top:bottom, left:right].astype(np.float32)
            arr[top:bottom, left:right] = np.clip(
                original * (1.0 - mask) + generated.astype(np.float32) * mask, 0, 255
            ).astype(np.uint8)
        return AnonymiserResult(
            image=Image.fromarray(arr),
            metadata={"method": "riddle", "boxes": len(valid)},
        )
