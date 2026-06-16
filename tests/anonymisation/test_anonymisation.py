"""Anonymisation unit tests covering smoke, edge cases, and unavailable methods."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.anonymisation.base_anonymiser import BaseAnonymiser
from src.anonymisation.blur_anonymiser import BlurAnonymiser
from src.anonymisation.pixelate_anonymiser import PixelateAnonymiser
from src.anonymisation.stylegan_anonymiser import StyleGANAnonymiser
from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
from src.anonymisation.fams_anonymiser import FAMSAnonymiser
from src.anonymisation.reface_anonymiser import RefaceAnonymiser
from src.anonymisation.reverse_personalization_anonymiser import ReversePersonalizationAnonymiser


class AnonymisationSmokeTest(unittest.TestCase):
    """Confirm the baseline anonymisers run and preserve dimensions."""

    def setUp(self) -> None:
        self.image = Image.new("RGB", (256, 256), color=(200, 200, 200))
        self.boxes = [(32, 32, 224, 224)]

    def test_blur_anonymiser(self) -> None:
        result = BlurAnonymiser().anonymise(self.image, self.boxes)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.metadata["boxes_processed"], 1)

    def test_pixelate_anonymiser(self) -> None:
        result = PixelateAnonymiser().anonymise(self.image, self.boxes)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.metadata["boxes_processed"], 1)


class NoFaceFrameTest(unittest.TestCase):
    """Anonymisers should handle frames with no detected faces gracefully."""

    def setUp(self) -> None:
        self.image = Image.new("RGB", (512, 512), color=(100, 150, 200))
        self.empty_boxes: list[tuple[int, int, int, int]] = []

    def test_blur_no_face(self) -> None:
        """Blur anonymiser on zero-face frame should return image unchanged."""
        result = BlurAnonymiser().anonymise(self.image, self.empty_boxes)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.metadata["boxes_processed"], 0)
        # Pixel-exact identity check: no modification when there are no boxes.
        np.testing.assert_array_equal(
            np.array(result.image), np.array(self.image)
        )

    def test_pixelate_no_face(self) -> None:
        """Pixelate anonymiser on zero-face frame should return image unchanged."""
        result = PixelateAnonymiser().anonymise(self.image, self.empty_boxes)
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.metadata["boxes_processed"], 0)
        np.testing.assert_array_equal(
            np.array(result.image), np.array(self.image)
        )


class PlannedAnonymiserTest(unittest.TestCase):
    """Planned anonymisers should either expose a real adapter or fail explicitly."""

    def test_stylegan_loads(self) -> None:
        anon = StyleGANAnonymiser()
        self.assertIsInstance(anon, BaseAnonymiser)

    def test_stylegan_raises_when_backend_unavailable(self) -> None:
        anon = StyleGANAnonymiser(backend_root=Path("/tmp/missing-stylegan-backend-for-test"))
        with self.assertRaises(NotImplementedError):
            anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])

    def test_stylegan_runs_adapter_when_backend_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = root / "runner.py"
            model = root / "model.pkl"
            pretrained = root / "pretrained_models"
            pretrained.mkdir()
            model.write_text("stub", encoding="utf-8")
            for name in (
                "psp_celebs_seg_to_face.pt",
                "CurricularFace_Backbone.pth",
                "mobilenet_celeba.pth",
                "unet_model.pth",
            ):
                (pretrained / name).write_text("stub", encoding="utf-8")
            runner.write_text(
                "import argparse\n"
                "from PIL import Image\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output')\n"
                "p.add_argument('--boxes-json')\n"
                "p.add_argument('--model-path')\n"
                "p.add_argument('--truncation-psi')\n"
                "p.add_argument('--seed')\n"
                "a=p.parse_args()\n"
                "Image.new('RGB', Image.open(a.input).size, (1, 2, 3)).save(a.output)\n",
                encoding="utf-8",
            )
            anon = StyleGANAnonymiser(backend_root=root, runner_path=runner, model_path=model)
            result = anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])
        self.assertEqual(result.image.size, (64, 64))
        self.assertEqual(result.metadata["method"], "stylegan")
        self.assertEqual(result.metadata["boxes_processed"], 1)

    def test_reverse_personalization_loads(self) -> None:
        anon = ReversePersonalizationAnonymiser()
        self.assertIsInstance(anon, BaseAnonymiser)

    def test_fams_loads(self) -> None:
        anon = FAMSAnonymiser()
        self.assertIsInstance(anon, BaseAnonymiser)

    def test_fams_raises_when_backend_unavailable(self) -> None:
        anon = FAMSAnonymiser(backend_root=Path("/tmp/missing-fams-backend-for-test"))
        with self.assertRaises(NotImplementedError):
            anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])

    def test_fams_runs_adapter_when_backend_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = root / "runner.py"
            (root / "utils").mkdir(parents=True)
            (root / "src" / "diffusers" / "pipelines" / "referencenet").mkdir(parents=True)
            (root / "utils" / "anonymize_faces_in_image.py").write_text("stub", encoding="utf-8")
            (root / "utils" / "extractor.py").write_text("stub", encoding="utf-8")
            (root / "utils" / "merger.py").write_text("stub", encoding="utf-8")
            (root / "src" / "diffusers" / "pipelines" / "referencenet" / "pipeline_referencenet.py").write_text(
                "stub",
                encoding="utf-8",
            )
            runner.write_text(
                "import argparse\n"
                "from PIL import Image\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output')\n"
                "p.add_argument('--boxes-json')\n"
                "p.add_argument('--model-id')\n"
                "p.add_argument('--base-model-id')\n"
                "p.add_argument('--clip-model-id')\n"
                "p.add_argument('--face-image-size')\n"
                "p.add_argument('--num-inference-steps')\n"
                "p.add_argument('--guidance-scale')\n"
                "p.add_argument('--anonymization-degree')\n"
                "p.add_argument('--overlap-iou-threshold')\n"
                "p.add_argument('--seed')\n"
                "p.add_argument('--enable-model-cpu-offload', action='store_true')\n"
                "a=p.parse_args()\n"
                "Image.new('RGB', Image.open(a.input).size, (7, 8, 9)).save(a.output)\n",
                encoding="utf-8",
            )
            with patch.object(FAMSAnonymiser, "_dependency_reason", return_value=""):
                anon = FAMSAnonymiser(backend_root=root, runner_path=runner)
            result = anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])
        self.assertEqual(result.image.size, (64, 64))
        self.assertEqual(result.metadata["method"], "fams")
        self.assertEqual(result.metadata["boxes_processed"], 1)

    @patch("src.anonymisation.reverse_personalization_anonymiser.read_accelerator_memory_gb", return_value=(7.5, 7.5))
    @patch.object(ReversePersonalizationAnonymiser, "_availability_reason", return_value="")
    def test_reverse_personalization_auto_enables_low_vram_settings(self, _mock_reason, _mock_memory) -> None:
        anon = ReversePersonalizationAnonymiser(
            num_inversion_steps=100,
            face_image_size=1024,
            det_size=640,
        )
        self.assertTrue(anon.use_model_cpu_offload)
        self.assertEqual(anon.num_inversion_steps, 5)
        self.assertEqual(anon.face_image_size, 512)
        self.assertEqual(anon.det_size, 320)

    @patch("src.anonymisation.reverse_personalization_anonymiser.read_accelerator_memory_gb", return_value=(7.5, 7.5))
    @patch.object(ReversePersonalizationAnonymiser, "_availability_reason", return_value="")
    def test_reverse_personalization_respects_explicit_offload_override(self, _mock_reason, _mock_memory) -> None:
        anon = ReversePersonalizationAnonymiser(
            num_inversion_steps=9,
            face_image_size=768,
            det_size=512,
            use_model_cpu_offload=False,
        )
        self.assertFalse(anon.use_model_cpu_offload)
        self.assertEqual(anon.num_inversion_steps, 9)
        self.assertEqual(anon.face_image_size, 768)
        self.assertEqual(anon.det_size, 512)

    @patch.object(ReversePersonalizationAnonymiser, "_availability_reason", return_value="")
    def test_reverse_personalization_runs_adapter_when_backend_ready(self, _mock_reason) -> None:
        anon = ReversePersonalizationAnonymiser()
        source = Image.new("RGB", (64, 64), color=(120, 120, 120))
        expected = Image.new("RGB", (64, 64), color=(10, 20, 30))

        def fake_run_backend(_command: list[str]):
            output_path = Path(_command[_command.index("--output") + 1])
            expected.save(output_path)

            class Result:
                returncode = 0
                stdout = "Saved to output"
                stderr = ""

            return Result()

        with patch.object(anon, "_run_backend_command", side_effect=fake_run_backend):
            result = anon.anonymise(source, [(0, 0, 32, 32)])

        self.assertEqual(result.image.size, source.size)
        self.assertEqual(result.metadata["boxes_processed"], 1)
        self.assertEqual(result.metadata["method"], "reverse_personalization")
        self.assertTrue(result.metadata["backend_uses_internal_detection"])
        np.testing.assert_array_equal(np.array(result.image), np.array(expected))

    @patch.object(
        ReversePersonalizationAnonymiser,
        "_availability_reason",
        return_value="Reverse Personalization dependencies unavailable: face_alignment,peft",
    )
    def test_reverse_personalization_raises_when_backend_unavailable(self, _mock_reason) -> None:
        anon = ReversePersonalizationAnonymiser()
        with self.assertRaises(NotImplementedError):
            anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])

    def test_reface_loads(self) -> None:
        anon = RefaceAnonymiser()
        self.assertIsInstance(anon, BaseAnonymiser)

    def test_reface_raises_when_backend_unavailable(self) -> None:
        anon = RefaceAnonymiser()
        with self.assertRaises(NotImplementedError):
            anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])

    def test_reface_runs_adapter_when_backend_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner = root / "runner.py"
            model = root / "model.pt"
            (root / "weights").mkdir()
            (root / "pretrain").mkdir()
            (root / "model" / "d3dfr" / "BFM").mkdir(parents=True)
            model.write_text("stub", encoding="utf-8")
            (root / "weights" / "epoch_20.pth").write_text("stub", encoding="utf-8")
            (root / "pretrain" / "ms1mv3_arcface_r50.pth").write_text("stub", encoding="utf-8")
            (root / "model" / "d3dfr" / "BFM" / "BFM_model_front.mat").write_text("stub", encoding="utf-8")
            runner.write_text(
                "import argparse\n"
                "from PIL import Image\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--input')\n"
                "p.add_argument('--output')\n"
                "p.add_argument('--boxes-json')\n"
                "p.add_argument('--model-path')\n"
                "p.add_argument('--donor-strategy')\n"
                "p.add_argument('--seed')\n"
                "a=p.parse_args()\n"
                "Image.new('RGB', Image.open(a.input).size, (4, 5, 6)).save(a.output)\n",
                encoding="utf-8",
            )
            with patch.object(RefaceAnonymiser, "_dependency_reason", return_value=""):
                anon = RefaceAnonymiser(backend_root=root, runner_path=runner, model_path=model)
            result = anon.anonymise(Image.new("RGB", (64, 64)), [(0, 0, 32, 32)])
        self.assertEqual(result.image.size, (64, 64))
        self.assertEqual(result.metadata["method"], "reface")
        self.assertEqual(result.metadata["boxes_processed"], 1)


class DiffusionAnonymiserTest(unittest.TestCase):
    """Test the Diffusion anonymiser using a mocked pipeline to avoid downloads."""

    def test_diffusion_loads(self) -> None:
        anon = DiffusionAnonymiser()
        self.assertIsInstance(anon, BaseAnonymiser)

    def test_diffusion_anonymise_mocked(self) -> None:
        from unittest.mock import MagicMock
        anon = DiffusionAnonymiser()
        
        # Create a mock pipeline that returns the input image
        mock_pipeline = MagicMock()
        def mock_call(prompt, negative_prompt, image, mask_image, num_inference_steps, guidance_scale):
            mock_result = MagicMock()
            mock_result.images = [image]
            return mock_result
        mock_pipeline.side_effect = mock_call
        
        anon._pipeline = mock_pipeline
        
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        boxes = [(32, 32, 224, 224)]
        result = anon.anonymise(img, boxes)
        
        self.assertEqual(result.image.size, img.size)
        self.assertEqual(result.metadata["boxes_processed"], 1)
        self.assertEqual(result.metadata["method"], "diffusion")


class ValidateBoxesTest(unittest.TestCase):
    """Edge cases for BaseAnonymiser.validate_boxes."""

    def setUp(self) -> None:
        self.anonymiser = BlurAnonymiser()
        self.image = Image.new("RGB", (200, 200))

    def test_valid_box_passes(self) -> None:
        boxes = [(10, 10, 100, 100)]
        result = self.anonymiser.validate_boxes(self.image, boxes)
        self.assertEqual(result, [(10, 10, 100, 100)])

    def test_out_of_bounds_clamped(self) -> None:
        """Boxes extending beyond image bounds should be clamped."""
        boxes = [(-10, -20, 300, 400)]
        result = self.anonymiser.validate_boxes(self.image, boxes)
        self.assertEqual(len(result), 1)
        left, top, right, bottom = result[0]
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 200)
        self.assertLessEqual(bottom, 200)

    def test_degenerate_box_discarded(self) -> None:
        """Zero-area or inverted boxes should be discarded."""
        boxes = [
            (50, 50, 50, 50),   # zero width and height
            (100, 100, 50, 50), # inverted
        ]
        result = self.anonymiser.validate_boxes(self.image, boxes)
        self.assertEqual(result, [])

    def test_mixed_valid_and_degenerate(self) -> None:
        """Only valid boxes should survive from a mixed input."""
        boxes = [
            (10, 10, 50, 50),     # valid
            (100, 100, 50, 50),   # inverted → discarded
            (20, 20, 80, 80),     # valid
        ]
        result = self.anonymiser.validate_boxes(self.image, boxes)
        self.assertEqual(len(result), 2)

    def test_empty_boxes_returns_empty(self) -> None:
        result = self.anonymiser.validate_boxes(self.image, [])
        self.assertEqual(result, [])

    def test_fully_outside_image_discarded(self) -> None:
        """Box completely outside image bounds → clamped to zero area → discarded."""
        boxes = [(300, 300, 400, 400)]
        result = self.anonymiser.validate_boxes(self.image, boxes)
        self.assertEqual(result, [])


class ImageDimensionPreservationTest(unittest.TestCase):
    """Confirm output dimensions match input for various image sizes."""

    def test_blur_various_sizes(self) -> None:
        for w, h in [(64, 64), (1920, 1080), (100, 300)]:
            img = Image.new("RGB", (w, h))
            boxes = [(0, 0, w // 2, h // 2)]
            result = BlurAnonymiser().anonymise(img, boxes)
            self.assertEqual(result.image.size, (w, h))

    def test_pixelate_various_sizes(self) -> None:
        for w, h in [(64, 64), (1920, 1080), (100, 300)]:
            img = Image.new("RGB", (w, h))
            boxes = [(0, 0, w // 2, h // 2)]
            result = PixelateAnonymiser().anonymise(img, boxes)
            self.assertEqual(result.image.size, (w, h))


if __name__ == "__main__":
    unittest.main()
