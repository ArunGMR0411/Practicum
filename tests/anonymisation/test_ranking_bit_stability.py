"""Recompute ranking-relevant scores from published per-image CSVs (bit-stable freeze check)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PER_IMAGE = ROOT / "outputs/03_anonymisation/09_policy_scoring/anonymisation_policy_per_image_metrics.csv"
ALL_METHODS = ROOT / "outputs/03_anonymisation/01_all_methods_comparison.csv"


def _three_attacker_privacy(ada_rate: float, arc_rate: float) -> float:
    """Simple privacy transform used in exploratory tables: 1 - mean Re-ID rates."""
    return float(1.0 - 0.5 * (ada_rate + arc_rate))


@pytest.mark.public
def test_per_image_metrics_exist() -> None:
    assert PER_IMAGE.is_file(), "published per-image metric CSV missing"
    assert ALL_METHODS.is_file()


@pytest.mark.public
def test_recompute_method_success_counts_match_all_methods_table() -> None:
    per = pd.read_csv(PER_IMAGE)
    am = pd.read_csv(ALL_METHODS)
    # deterministic methods present in both
    for method in ["blur", "pixelate", "solid_mask_black", "layered_blur_downscale_noise"]:
        g = per[per["method"].astype(str) == method]
        assert len(g) == 500, method
        n_success = int((pd.to_numeric(g["success"], errors="coerce").fillna(0) >= 1).sum())
        row = am[am["method"].astype(str) == method]
        assert not row.empty, method
        assert int(row.iloc[0]["n_success"]) == n_success
        assert int(row.iloc[0]["n_input_frames"]) == 500


@pytest.mark.public
def test_recompute_mean_reid_privacy_ordering_stable() -> None:
    """Privacy ranking among visual-safe methods stays solid_mask <= layered <= blur on mean max-ReID."""
    per = pd.read_csv(PER_IMAGE)
    stats = {}
    for method in ["blur", "solid_mask_black", "layered_blur_downscale_noise"]:
        g = per[per["method"].astype(str) == method]
        g = g[pd.to_numeric(g["success"], errors="coerce").fillna(0) >= 1]
        ada = pd.to_numeric(g["AdaFace_reid_rate"], errors="coerce")
        arc = pd.to_numeric(g["ArcFace_reid_rate"], errors="coerce")
        max_reid = pd.concat([ada, arc], axis=1).max(axis=1)
        stats[method] = float(max_reid.mean())
    # lower max-ReID = stronger privacy
    assert stats["solid_mask_black"] <= stats["layered_blur_downscale_noise"] + 1e-9
    assert stats["layered_blur_downscale_noise"] <= stats["blur"] + 1e-9


@pytest.mark.public
def test_all_methods_table_ranks_are_consistent_with_n_success() -> None:
    am = pd.read_csv(ALL_METHODS)
    # RP has 482 successful outputs and 18 failures.
    rp = am[am["method"].astype(str).str.contains("reverse_personalization", case=False, na=False)]
    assert not rp.empty
    assert int(rp.iloc[0]["n_success"]) == 482
    assert int(rp.iloc[0]["n_failure"]) == 18
