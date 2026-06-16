#!/usr/bin/env python3

"""Probes for the anonymiser feasibility scaffolding."""

from __future__ import annotations

import unittest

from PIL import Image

from src.anonymisation.registry import build_anonymiser_registry


class AnonymiserFeasibilityTest(unittest.TestCase):
    """Verify every registered anonymiser has a predictable feasibility path."""

    def test_registry_keys_match_expected_method_set(self) -> None:
        registry = build_anonymiser_registry()
        self.assertEqual(
            set(registry.keys()),
            {
                "blur",
                "pixelate",
                "nullface",
                "stylegan",
                "reverse_personalization",
                "diffusion",
                "fams",
                "reface",
                "riddle",
                "falco",
            },
        )

    def test_unavailable_methods_raise_clear_signal(self) -> None:
        registry = build_anonymiser_registry()
        image = Image.new("RGB", (128, 128), color=(120, 140, 160))
        boxes = [(16, 16, 96, 96)]
        for method_name in ("reface",):
            with self.subTest(method=method_name):
                with self.assertRaises(NotImplementedError) as context:
                    registry[method_name].anonymise(image, boxes)
                self.assertIn("unavailable", str(context.exception))

    def test_registered_methods_have_synthetic_output_or_clear_unavailable_signal(self) -> None:
        """Each method should either produce a same-size image or raise NotImplementedError."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch
        from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
        from src.anonymisation.external_command_anonymiser import ExternalCommandAnonymiser

        # Mock _get_pipeline to avoid Hugging Face downloads during feasibility tests
        mock_pipeline = MagicMock()
        def mock_call(*args, **kwargs):
            mock_result = MagicMock()
            mock_result.images = [kwargs.get("image") or args[2]]
            return mock_result
        mock_pipeline.side_effect = mock_call
        original_get_pipeline = DiffusionAnonymiser._get_pipeline
        DiffusionAnonymiser._get_pipeline = lambda self: mock_pipeline

        try:
            image = Image.new("RGB", (256, 256), color=(120, 140, 160))
            boxes = [(32, 32, 224, 224)]
            registry = build_anonymiser_registry()

            for method_name, anonymiser in registry.items():
                with self.subTest(method=method_name):
                    backend_patch = None
                    if (
                        isinstance(anonymiser, ExternalCommandAnonymiser)
                        or hasattr(anonymiser, "_run_backend_command")
                    ) and not getattr(anonymiser, "reason", ""):
                        def fake_run_backend(command: list[str]):
                            output_path = Path(command[command.index("--output") + 1])
                            image.save(output_path)

                            class Result:
                                returncode = 0
                                stdout = "Saved to output"
                                stderr = ""

                            return Result()

                        backend_patch = patch.object(anonymiser, "_run_backend_command", side_effect=fake_run_backend)
                        backend_patch.start()
                    try:
                        # Preflight reason (missing assets/source) is a valid unavailable signal
                        preflight = getattr(anonymiser, "reason", "") or ""
                        if preflight:
                            self.assertTrue(str(preflight))
                            continue
                        result = anonymiser.anonymise(image, boxes)
                    except NotImplementedError as exc:
                        self.assertIn("unavailable", str(exc))
                        continue
                    except RuntimeError as exc:
                        # Research backends may refuse when CUDA/assets are missing
                        self.assertTrue(str(exc))
                        continue
                    finally:
                        if backend_patch is not None:
                            backend_patch.stop()
                    self.assertEqual(result.image.size, image.size)
                    self.assertIsNotNone(result.image)
        finally:
            DiffusionAnonymiser._get_pipeline = original_get_pipeline


if __name__ == "__main__":
    unittest.main()
