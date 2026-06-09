import os
from pathlib import Path
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.evaluation.adaface_net import load_pretrained_model


def _prepend_venv_cuda_libs() -> None:
    """Expose pip-installed CUDA runtime libraries to ONNXRuntime if present."""
    venv = os.environ.get("VIRTUAL_ENV")
    if not venv and ".venv" in Path(sys.executable).parts:
        parts = Path(sys.executable).parts
        venv = str(Path(*parts[: parts.index(".venv") + 1]))
    if not venv:
        return
    py_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = Path(venv) / "lib" / py_dir / "site-packages" / "nvidia"
    if not nvidia_root.exists():
        return
    lib_dirs = [str(path) for path in nvidia_root.rglob("lib") if path.is_dir()]
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = lib_dirs + [part for part in existing.split(":") if part]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(merged))


_prepend_venv_cuda_libs()

try:
    from insightface.model_zoo import get_model
except ImportError:
    get_model = None

class ReIDEvaluator:
    def __init__(self, adaface_ckpt_path, arcface_onnx_path, device='cpu', require_arcface_gpu=False):
        self.device = torch.device(device)
        
        # Load AdaFace model
        if not os.path.exists(adaface_ckpt_path):
            raise FileNotFoundError(f"AdaFace checkpoint not found at: {adaface_ckpt_path}")
        self.adaface_model = load_pretrained_model('ir_50', adaface_ckpt_path).to(self.device)
        self.adaface_model.eval()

        # Load ArcFace model
        self.arcface_model = None
        if get_model is not None and os.path.exists(arcface_onnx_path):
            providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
            self.arcface_model = get_model(arcface_onnx_path, providers=providers)
            # For insightface, prepare expects ctx_id (negative for CPU)
            ctx_id = -1 if device == 'cpu' else 0
            self.arcface_model.prepare(ctx_id=ctx_id)
            session = getattr(self.arcface_model, "session", None)
            providers_used = session.get_providers() if session is not None else []
            if require_arcface_gpu and "CUDAExecutionProvider" not in providers_used:
                raise RuntimeError(
                    "ArcFace CUDAExecutionProvider did not initialize; refusing CPU fallback for a GPU-required metric."
                )

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def extract_embeddings_adaface(self, crops: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            tensors = []
            for crop in batch_crops:
                resized = crop.resize((112, 112), Image.Resampling.BILINEAR)
                rgb_crop = resized.convert("RGB")
                tensor = TF.to_tensor(rgb_crop)
                tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                tensors.append(tensor)
            
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                embeddings, _ = self.adaface_model(batch)
            all_embeddings.append(embeddings.cpu().numpy())
            
        embeddings_np = np.concatenate(all_embeddings, axis=0)
        return self._normalize_embeddings(embeddings_np)

    def extract_embeddings_arcface(self, crops: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if self.arcface_model is None:
            return np.zeros((len(crops), 512), dtype=np.float32)
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        
        all_embeddings = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            bgr_imgs = []
            for crop in batch_crops:
                resized = crop.resize((112, 112), Image.Resampling.BILINEAR)
                rgb_crop = resized.convert("RGB")
                bgr_img = np.array(rgb_crop)[:, :, ::-1]
                bgr_imgs.append(bgr_img)
            
            feats = self.arcface_model.get_feat(bgr_imgs)
            if len(bgr_imgs) == 1 and feats.ndim == 1:
                feats = np.expand_dims(feats, axis=0)
            all_embeddings.append(feats)
            
        embeddings_np = np.concatenate(all_embeddings, axis=0)
        return self._normalize_embeddings(embeddings_np)

    def compute_reid_metrics(self, gallery_embeddings: np.ndarray, query_embeddings: np.ndarray) -> dict[str, float]:
        if len(gallery_embeddings) == 0 or len(query_embeddings) == 0:
            return {"cosine_similarity": 0.0, "reid_rate": 0.0}
        
        # Unit-normalized embeddings dot product computes cosine similarity
        sim_matrix = np.dot(query_embeddings, gallery_embeddings.T)
        
        # 1. Average Cosine Similarity of matched pairs (diagonal elements)
        matched_similarities = np.diag(sim_matrix)
        avg_cosine_sim = float(np.mean(matched_similarities))
        
        # 2. Rank-1 Re-ID Rate (Rank-1 Identification accuracy)
        predicted_indices = np.argmax(sim_matrix, axis=1)
        correct_predictions = (predicted_indices == np.arange(len(query_embeddings)))
        reid_rate = float(np.mean(correct_predictions))
        
        return {
            "cosine_similarity": avg_cosine_sim,
            "reid_rate": reid_rate
        }
