"""Policy, replay, and review metadata checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from privacy_pipeline_app.production_runner import ALL_MODES
from privacy_pipeline_app.wizard_workflow import create_run
from src.policy.registry import get_app_policy_semantics, get_profile_defaults, load_policy_registry

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ID = "runtime_3_source_all_raw_rf_approximation"


def test_objective_profiles_are_fixed_app_presets_not_scientific_oapr() -> None:
    expected = {"privacy": "solid_mask", "balanced": "layered", "utility": "blur"}
    semantics = get_app_policy_semantics()
    assert semantics["app_policy_id"] == "objective_profile"
    assert semantics["scientific_policy_id"] == "oapr_visual_safe_balanced_500"
    assert semantics["app_policy_id"] != semantics["scientific_policy_id"]
    assert {focus: get_profile_defaults(focus)["face_anonymisation"] for focus in expected} == expected
    assert "oapr" not in ALL_MODES


def test_legacy_wizard_alias_is_explicitly_recorded(tmp_path, monkeypatch) -> None:
    source = tmp_path / "images"
    source.mkdir()
    monkeypatch.setattr("privacy_pipeline_app.wizard_workflow.APP_RUNS", tmp_path / "runs")
    state = create_run(str(source), False, "oapr", "layered", "balanced", True)
    assert state.strategy == "objective_profile"
    assert state.plan["legacy_alias_requested"] is True
    assert state.plan["legacy_alias_note"] == "Strategy alias resolved to objective_profile."


def test_app_detector_has_exact_source_bank_and_no_scientific_score() -> None:
    registry = load_policy_registry()
    runtime = registry["app_runtime_detector"]
    assert runtime["policy_id"] == RUNTIME_ID
    assert runtime["candidate_sources"] == [
        "rfdetr_medium_face_030",
        "yolo11s_widerface_1280",
        "scrfd_10g_current_640",
    ]
    assert runtime["scientific_score_assigned"] is False
    assert runtime["runtime_score"] is None
    assert registry["scientific_policies"]["face_detector_primary"]["policy_id"] != RUNTIME_ID


def test_route_replay_records_applied_difference() -> None:
    path = ROOT / "outputs/10_final_enhancement_evaluation/06_frozen_scientific_oapr_route_replay/metadata/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["execution_mode"] == "frozen_scientific_oapr_route_replay"
    assert summary["fresh_routing"] is False
    assert summary["scientific_route_counts"] != summary["app_applied_counts"]
    assert summary["detector_application_policy_id"] == RUNTIME_ID


def test_review_records_classify_automation_and_do_not_report_agreement() -> None:
    review = ROOT / "outputs/03_anonymisation/14_group2_comparison/13_expanded_structured_visual_review.csv"
    rows = list(csv.DictReader(review.open(encoding="utf-8")))
    assert rows
    assert all(row["provenance_class"] for row in rows)
    assert any(row["provenance_class"] == "heuristic_or_automatically_generated" for row in rows)
    summary_path = ROOT / "outputs/03_anonymisation/14_group2_comparison/15_expanded_review_provenance_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["agreement_statistics"] == "not_computed_mixed_non_independent_sources"
    assert not any("kappa" in key.lower() for key in summary)


def test_copy_is_selected_only_when_no_face_operator_is_required() -> None:
    decisions = ROOT / "outputs/01_protocol/smoke_protocol_public/runs/latest/metadata/decisions.csv"
    if not decisions.is_file():
        return
    for row in csv.DictReader(decisions.open(encoding="utf-8")):
        if row["selected_method"] == "copy":
            assert int(row["n_faces_detected"]) == 0
