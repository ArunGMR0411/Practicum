"""Feasibility-safe placeholder wrappers for heavy anonymisation methods."""

from __future__ import annotations

from PIL import Image

from src.anonymisation.base_anonymiser import AnonymiserResult, BaseAnonymiser


class UnavailableAnonymiser(BaseAnonymiser):
    """Represent a planned anonymiser that is not implemented in the current repo."""

    def __init__(self, method_name: str, reason: str) -> None:
        self.method_name = method_name
        self.reason = reason

    def anonymise(self, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> AnonymiserResult:
        """Raise a clear error when an unavailable method is invoked."""
        raise NotImplementedError(f"{self.method_name} unavailable: {self.reason}")
