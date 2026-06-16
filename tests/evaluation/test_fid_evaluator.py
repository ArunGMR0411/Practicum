"""Tests for FID evaluator helpers."""

from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.fid_evaluator import compute_activation_statistics, frechet_distance


class FIDEvaluatorHelpersTest(unittest.TestCase):
    """Validate the numerical helper layer without running heavy feature extraction."""

    def test_compute_activation_statistics_shapes(self) -> None:
        features = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
        mu, sigma = compute_activation_statistics(features)
        self.assertEqual(mu.shape, (2,))
        self.assertEqual(sigma.shape, (2, 2))

    def test_frechet_distance_zero_for_identical_statistics(self) -> None:
        mu = np.array([0.5, -1.25], dtype=np.float64)
        sigma = np.array([[1.0, 0.2], [0.2, 0.5]], dtype=np.float64)
        value = frechet_distance(mu, sigma, mu, sigma)
        self.assertAlmostEqual(value, 0.0, places=6)

    def test_frechet_distance_is_symmetric(self) -> None:
        mu1 = np.array([0.0, 1.0], dtype=np.float64)
        sigma1 = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        mu2 = np.array([1.0, -1.0], dtype=np.float64)
        sigma2 = np.array([[2.0, 0.3], [0.3, 1.0]], dtype=np.float64)
        forward = frechet_distance(mu1, sigma1, mu2, sigma2)
        backward = frechet_distance(mu2, sigma2, mu1, sigma1)
        self.assertAlmostEqual(forward, backward, places=6)
