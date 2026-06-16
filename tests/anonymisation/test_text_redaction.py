"""Tests for text and screen redaction helpers."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anonymisation.screen_redactor import ScreenRedactor
from src.anonymisation.text_redactor import TextRedactor


class TestTextRedaction(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (200, 120), color="white")
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((20, 20, 120, 70), fill="black")
        self.boxes = [(20, 20, 120, 70)]

    def test_text_redactor_fill_covers_region(self) -> None:
        redactor = TextRedactor(mode="fill", fill_color=(255, 0, 0))
        result = redactor.anonymise(self.image, self.boxes)
        region = np.array(result.image.crop((20, 20, 120, 70)))
        self.assertTrue(np.all(region == np.array([255, 0, 0], dtype=np.uint8)))
        self.assertEqual(result.metadata["boxes_processed"], 1)

    def test_screen_redactor_fill_covers_region(self) -> None:
        redactor = ScreenRedactor(mode="fill", fill_color=(0, 255, 0))
        result = redactor.anonymise(self.image, self.boxes)
        region = np.array(result.image.crop((20, 20, 120, 70)))
        self.assertTrue(np.all(region == np.array([0, 255, 0], dtype=np.uint8)))
        self.assertEqual(result.metadata["boxes_processed"], 1)

    def test_text_redactor_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            TextRedactor(mode="invalid")

    def test_screen_redactor_no_boxes_returns_zero_processed(self) -> None:
        redactor = ScreenRedactor(mode="blur")
        result = redactor.anonymise(self.image, [])
        self.assertEqual(result.metadata["boxes_processed"], 0)
        self.assertEqual(result.image.size, self.image.size)


if __name__ == "__main__":
    unittest.main()
