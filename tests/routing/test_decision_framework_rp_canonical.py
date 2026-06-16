"""Canonical RP 482/500 must win over older 444 per-image aggregates."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_rp_final_summary_is_482() -> None:
    import pandas as pd

    path = (
        ROOT
        / "outputs/03_anonymisation/05_reverse_personalization/09_rp_final_metric_summary.csv"
    )
    assert path.is_file()
    row = pd.read_csv(path).iloc[0]
    assert int(row["n_success"]) == 482
    assert int(row["n_input_frames"]) == 500


def test_decision_framework_atomic_uses_rp_482() -> None:
    import pandas as pd

    path = (
        ROOT
        / "outputs/05_oapr/decision_framework/01_atomic_metrics/05_face_anonymisation_atomic.csv"
    )
    assert path.is_file()
    df = pd.read_csv(path)
    rp = df[df["method"].astype(str).str.lower().eq("reverse_personalization")]
    assert not rp.empty
    assert int(rp.iloc[0]["n_success"]) == 482


def test_apply_canonical_override_function() -> None:
    import importlib.util

    script = ROOT / "scripts/oapr_routing/run_decision_framework_evaluation.py"
    spec = importlib.util.spec_from_file_location("df_eval", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [
        {
            "method": "reverse_personalization",
            "n_images": 500,
            "n_success": 444,
            "failure_rate": 0.112,
            "n_nonsensitive": 444,
            "non_sensitive_utility": 0.99,
            "non_sensitive_source": "test",
        }
    ]
    out = mod.apply_canonical_face_method_overrides(rows)
    assert int(out[0]["n_success"]) == 482
    assert float(out[0]["failure_rate"]) == pytest.approx(18 / 500)
