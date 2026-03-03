"""Read-only CASTLE 2024 loader backed by the dataset manifest."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image


class CASTLEDataset:
    """Load CASTLE frames from a manifest without applying preprocessing by default.

    The loader reads the manifest once to build a lightweight byte-offset index for
    rows that match the active filters. Image files are decoded on demand one item
    at a time, always from the source WebP frame. Optional evaluation-time scaling
    can be applied after decode, but no cropping, normalisation, augmentation, or
    other preprocessing is performed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        return_format: str = "pil",
        eval_scale: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> None:
        """Initialise the loader from a manifest path and optional row filters."""
        self.manifest_path = Path(manifest_path)
        self.return_format = return_format
        self.eval_scale = eval_scale
        self.filters = dict(filters or {})
        self.filters.setdefault("integrity_status", "valid")
        self.dataset_root = self.manifest_path.parent.parent
        self.raw_root = self._resolve_raw_root()
        self._validate_arguments()
        self._fieldnames, self._row_offsets = self._build_index()

    def __len__(self) -> int:
        """Return the number of manifest rows visible through the active filters."""
        return len(self._row_offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one manifest-backed dataset item as path, image, and metadata."""
        if index < 0:
            index += len(self._row_offsets)
        if index < 0 or index >= len(self._row_offsets):
            raise IndexError("dataset index out of range")

        row = self._read_row_at_offset(self._row_offsets[index])
        image_path = self.raw_root / row["relative_path"]

        with Image.open(image_path) as image:
            loaded_image = image.copy()

        loaded_image = self._apply_eval_scale(loaded_image)
        if self.return_format == "numpy":
            image_value: Image.Image | np.ndarray = np.array(loaded_image)
        else:
            image_value = loaded_image

        return {
            "path": image_path,
            "image": image_value,
            "metadata": row,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield dataset items one at a time in manifest order."""
        for index in range(len(self)):
            yield self[index]

    def _validate_arguments(self) -> None:
        """Validate constructor arguments before building the manifest index."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if not self.raw_root.exists():
            raise FileNotFoundError(f"CASTLE raw image root not found: {self.raw_root}")
        if self.return_format not in {"pil", "numpy"}:
            raise ValueError("return_format must be 'pil' or 'numpy'")
        if self.eval_scale is not None and self.eval_scale <= 0:
            raise ValueError("eval_scale must be greater than 0 when provided")

    def _resolve_raw_root(self) -> Path:
        """Resolve the raw frame directory for root-level or thesis manifest files."""
        raw_root = self.dataset_root / "raw"
        if raw_root.exists():
            return raw_root
        castle_raw_root = self.dataset_root / "castle2024" / "raw"
        if castle_raw_root.exists():
            return castle_raw_root
        return raw_root

    def _build_index(self) -> tuple[list[str], list[int]]:
        """Build a lightweight byte-offset index for filtered manifest rows."""
        offsets: list[int] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            header_line = handle.readline()
            if not header_line:
                raise ValueError(f"Manifest is empty: {self.manifest_path}")
            fieldnames = next(csv.reader([header_line.rstrip("\n")]))

            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = self._parse_manifest_line(fieldnames, line)
                if self._matches_filters(row):
                    offsets.append(offset)

        return fieldnames, offsets

    def _parse_manifest_line(self, fieldnames: list[str], line: str) -> dict[str, str]:
        """Parse a single CSV row string into a metadata dictionary."""
        values = next(csv.reader([line.rstrip("\n")]))
        return dict(zip(fieldnames, values, strict=True))

    def _matches_filters(self, row: dict[str, str]) -> bool:
        """Return True when a manifest row satisfies every active filter."""
        for key, expected in self.filters.items():
            if row.get(key) != expected:
                return False
        return True

    def _read_row_at_offset(self, offset: int) -> dict[str, str]:
        """Read and parse a manifest row at a previously indexed byte offset."""
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(offset)
            line = handle.readline()
        return self._parse_manifest_line(self._fieldnames, line)

    def _apply_eval_scale(self, image: Image.Image) -> Image.Image:
        """Optionally downscale a decoded image for evaluation-time loading only."""
        if self.eval_scale is None or self.eval_scale == 1.0:
            return image

        width = max(1, int(round(image.width * self.eval_scale)))
        height = max(1, int(round(image.height * self.eval_scale)))
        return image.resize((width, height), Image.Resampling.LANCZOS)
