"""Unit tests for perceptual evaluation metrics (SSIM and LPIPS)."""

from __future__ import annotations

import unittest
import numpy as np
from PIL import Image, ImageFilter

from src.evaluation.perceptual_metrics import compute_ssim, LPIPSEvaluator


class TestPerceptualMetrics(unittest.TestCase):
    """Test calculations of SSIM and LPIPS on synthetic image pairs."""

    def setUp(self) -> None:
        # Create a simple synthetic image (RGB)
        np_img = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add some structure
        np_img[32:96, 32:96, 0] = 255  # red square
        np_img[48:80, 48:80, 1] = 255  # green square
        self.image1 = Image.fromarray(np_img)

        # Identity image (copy)
        self.image_identity = self.image1.copy()

        # Degraded image (heavily blurred version of image1)
        self.image_blurred = self.image1.filter(ImageFilter.GaussianBlur(radius=5.0))

        # Different image (noise)
        self.image_noise = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

        # Initialise LPIPS evaluator on CPU to keep unit test fast and lightweight
        self.lpips_evaluator = LPIPSEvaluator(net="alex", use_gpu=False)

    def test_ssim_identity(self) -> None:
        """SSIM of identical images should be 1.0."""
        score = compute_ssim(self.image1, self.image_identity)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_ssim_degraded(self) -> None:
        """SSIM of blurred image should be strictly less than 1.0."""
        score = compute_ssim(self.image1, self.image_blurred)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_lpips_identity(self) -> None:
        """LPIPS distance of identical images should be 0.0."""
        score = self.lpips_evaluator.compute_lpips(self.image1, self.image_identity)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_lpips_degraded(self) -> None:
        """LPIPS of blurred/noisy image should be strictly greater than 0.0."""
        score_blurred = self.lpips_evaluator.compute_lpips(self.image1, self.image_blurred)
        score_noise = self.lpips_evaluator.compute_lpips(self.image1, self.image_noise)
        self.assertGreater(score_blurred, 0.0)
        self.assertGreater(score_noise, score_blurred)  # Noise is perceptually more different than blur


if __name__ == "__main__":
    unittest.main()
