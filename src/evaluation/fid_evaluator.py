"""Helpers for FID-style feature extraction and Frechet distance computation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor


def compute_activation_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and covariance for one feature matrix."""
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features array, got shape {features.shape}")
    if len(features) < 2:
        raise ValueError("At least two feature vectors are required to compute covariance")
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Frechet distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    if mu1.shape != mu2.shape:
        raise ValueError(f"Mean vectors have different shapes: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"Covariance matrices have different shapes: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm((sigma1 @ sigma2).astype(np.float64), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(max(fid, 0.0))


class InceptionFeatureExtractor:
    """Extract 2048D Inception-v3 pool features for FID-like evaluation."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False).eval()
        self.extractor = create_feature_extractor(model, return_nodes={"avgpool": "avgpool"}).to(self.device)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def extract(self, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        """Extract pooled inception features for the supplied image paths."""
        if not image_paths:
            raise ValueError("At least one image path is required")

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                tensors = []
                for path in batch_paths:
                    with Image.open(path) as image:
                        tensors.append(self.preprocess(image.convert("RGB")))
                batch = torch.stack(tensors, dim=0).to(self.device)
                outputs = self.extractor(batch)["avgpool"]
                features = torch.flatten(outputs, start_dim=1).cpu().numpy()
                batches.append(features)
        return np.concatenate(batches, axis=0)
