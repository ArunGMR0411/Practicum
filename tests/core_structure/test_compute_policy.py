"""Tests for automatic runtime compute-policy selection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.utils.compute_policy import build_compute_policy


class ComputePolicyTest(unittest.TestCase):
    @patch("src.utils.compute_policy.resolve_torch_device", return_value="cuda")
    @patch("src.utils.compute_policy.read_accelerator_memory_gb", return_value=(80.0, 70.0))
    @patch("src.utils.compute_policy.read_host_memory_gb", return_value=(128.0, 120.0))
    @patch("src.utils.compute_policy.read_safe_batch_size", side_effect=[32, 64, 1])
    @patch("src.utils.compute_policy.load_system_config", return_value={"runtime_label": "cuda"})
    def test_large_gpu_scales_batches(self, *_mocks) -> None:
        policy = build_compute_policy()
        self.assertEqual(policy.device, "cuda")
        self.assertGreaterEqual(policy.fid_batch_size, 128)
        self.assertGreaterEqual(policy.reid_batch_size, 256)
        self.assertEqual(policy.generative_control_max_workers, 4)
        self.assertTrue(policy.use_mixed_precision)
        self.assertFalse(policy.use_low_vram_mode)

    @patch("src.utils.compute_policy.resolve_torch_device", return_value="cpu")
    @patch("src.utils.compute_policy.read_accelerator_memory_gb", return_value=(0.0, 0.0))
    @patch("src.utils.compute_policy.read_safe_batch_size", side_effect=[8, 16, 1])
    @patch("src.utils.compute_policy.load_system_config", return_value={"runtime_label": "cpu"})
    def test_cpu_policy_stays_conservative(self, *_mocks) -> None:
        policy = build_compute_policy()
        self.assertEqual(policy.device, "cpu")
        self.assertEqual(policy.generative_control_max_workers, 1)
        self.assertFalse(policy.use_mixed_precision)


if __name__ == "__main__":
    unittest.main()
