"""FALCO single-image adapter for the research demonstrator App.

Requires CUDA and the FALCO source tree (set FALCO_SOURCE_ROOT). When unavailable,
``reason`` is non-empty so the App falls back honestly.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser, BoundingBox

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(
    os.environ.get("FALCO_SOURCE_ROOT", str(PROJECT_ROOT / "third_party" / "falco"))
)


class FalcoAnonymiser(BaseAnonymiser):
    method_name = "falco"

    def __init__(
        self,
        *,
        source_root: Path = DEFAULT_SOURCE,
        padding: float = 0.40,
        seed: int = 20260712,
        epochs: int = 10,
    ) -> None:
        self.source_root = Path(source_root)
        self.padding = float(padding)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self._models = None
        self.reason = self._availability_reason()

    def _availability_reason(self) -> str:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            return f"falco requires torch ({exc})"
        if not torch.cuda.is_available():
            return "falco requires CUDA"
        if not self.source_root.is_dir():
            return f"falco source missing at {self.source_root} (set FALCO_SOURCE_ROOT)"
        checkpoint = self.source_root / "models/pretrained/e4e/e4e_ffhq_encode.pt"
        if not checkpoint.is_file():
            return f"falco checkpoint missing at {checkpoint}"
        return ""

    def _load_models(self):
        if self._models is not None:
            return self._models
        import torch

        # GenForce fused CUDA ops need ninja + a full CUDA toolkit (not cusparselt).
        try:
            from privacy_pipeline_app.runtime_env import configure_app_runtime

            configure_app_runtime(force=True)
        except Exception:
            venv_bin = PROJECT_ROOT / ".venv" / "bin"
            if venv_bin.is_dir():
                os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
            for cand in sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages/nvidia/cu*")):
                if (cand / "bin" / "nvcc").exists() and (cand / "include").is_dir():
                    os.environ["CUDA_HOME"] = str(cand)
                    os.environ["PATH"] = str(cand / "bin") + os.pathsep + os.environ.get("PATH", "")
                    break

        root = Path(self.source_root).resolve()
        # Isolate FALCO imports from other research trees that also ship a models package.
        drop_prefixes = ("models", "lib", "mapper", "criteria", "configs")
        for key in list(sys.modules):
            if any(key == pref or key.startswith(pref + ".") for pref in drop_prefixes):
                del sys.modules[key]
        # Keep only this source root ahead of site-packages for local packages.
        cleaned = []
        for p in sys.path:
            try:
                rp = str(Path(p).resolve())
            except Exception:
                cleaned.append(p)
                continue
            if rp.endswith("/third_party/riddle") or rp.endswith("/third_party/styleid"):
                continue
            if rp != str(root):
                cleaned.append(p)
        sys.path = [str(root)] + cleaned
        os.chdir(root)
        import importlib
        importlib.invalidate_caches()

        from models.psp import pSp  # type: ignore
        from models.load_generator import load_generator  # type: ignore

        checkpoint = root / "models/pretrained/e4e/e4e_ffhq_encode.pt"
        e4e_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        options = dict(e4e_data["opts"])
        options["checkpoint_path"] = str(checkpoint)
        options["device"] = "cuda"
        encoder = pSp(Namespace(**options)).eval().cuda()
        genforce_dir = str(root / "models/pretrained/genforce")
        generator = load_generator(
            "stylegan2_ffhq1024",
            latent_is_w=True,
            CHECKPOINT_DIR=genforce_dir,
        ).eval().cuda()
        self._models = (encoder, generator)
        return self._models

    @staticmethod
    def _square_crop(box, width, height, padding):
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return (
            max(0, int(round(cx - side / 2))),
            max(0, int(round(cy - side / 2))),
            min(width, int(round(cx + side / 2))),
            min(height, int(round(cy + side / 2))),
        )

    def anonymise(self, image: Image.Image, boxes: list[BoundingBox]) -> AnonymiserResult:
        if self.reason:
            raise RuntimeError(self.reason)
        import cv2
        import torch

        # GenForce fused ops need ninja on PATH (venv bin).
        venv_bin = PROJECT_ROOT / ".venv" / "bin"
        if venv_bin.is_dir():
            os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")

        encoder, generator = self._load_models()
        valid = self.validate_boxes(image, boxes)
        arr = np.asarray(image.convert("RGB")).copy()
        h, w = arr.shape[:2]
        torch.manual_seed(self.seed)
        for box in valid:
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
            with torch.inference_mode():
                codes = encoder.encoder(batch)
                if getattr(encoder.opts, "start_from_latent_avg", False):
                    codes = codes + encoder.latent_avg.repeat(codes.shape[0], 1, 1)
                # Lightweight interactive path: latent jitter in W+ (full FALCO
                # optim needs FaRL/ID losses and is too slow for live UI).
                noise = torch.zeros_like(codes)
                noise[:, 3:8, :] = 0.25 * torch.randn_like(codes[:, 3:8, :])
                latents = codes + noise
                images = generator(latents)
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
            metadata={"method": "falco", "boxes": len(valid), "mode": "latent_jitter"},
        )
