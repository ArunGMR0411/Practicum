"""Factory helpers for anonymiser feasibility and runtime selection."""

from __future__ import annotations

from src.anonymisation.blur_anonymiser import BlurAnonymiser
from src.anonymisation.diffusion_anonymiser import DiffusionAnonymiser
from src.anonymisation.falco_anonymiser import FalcoAnonymiser
from src.anonymisation.fams_anonymiser import FAMSAnonymiser
from src.anonymisation.nullface_anonymiser import NullFaceAnonymiser
from src.anonymisation.pixelate_anonymiser import PixelateAnonymiser
from src.anonymisation.reface_anonymiser import RefaceAnonymiser
from src.anonymisation.reverse_personalization_anonymiser import ReversePersonalizationAnonymiser
from src.anonymisation.riddle_anonymiser import RiddleAnonymiser
from src.anonymisation.stylegan_anonymiser import StyleGANAnonymiser


def build_anonymiser_registry() -> dict[str, object]:
    """Return anonymiser instances keyed by method name (including research methods)."""
    return {
        "blur": BlurAnonymiser(),
        "pixelate": PixelateAnonymiser(),
        "nullface": NullFaceAnonymiser(),
        "stylegan": StyleGANAnonymiser(),
        "reverse_personalization": ReversePersonalizationAnonymiser(),
        "diffusion": DiffusionAnonymiser(),
        "fams": FAMSAnonymiser(),
        "reface": RefaceAnonymiser(),
        "riddle": RiddleAnonymiser(),
        "falco": FalcoAnonymiser(),
    }
