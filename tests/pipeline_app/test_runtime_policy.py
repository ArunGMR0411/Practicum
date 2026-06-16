from __future__ import annotations

import json

from privacy_pipeline_app.runtime_policy import estimate_runtime, select_runtime_policy


def test_runtime_policy_uses_efficient_evidence_tier_for_eight_gb_gpu() -> None:
    policy = select_runtime_policy({"cuda_available": True, "vram_total_mb": 7705})
    assert policy.policy_id == "accelerated_efficient"
    assert policy.face_policy_id == "fusion_rfdetr_scrfd10g"
    assert policy.multimodal_image_size == 1280


def test_runtime_policy_uses_full_fusion_for_larger_accelerator() -> None:
    policy = select_runtime_policy({"cuda_available": True, "vram_total_mb": 24 * 1024})
    assert policy.policy_id == "accelerated_full"
    assert policy.face_policy_id == "runtime_3_source_all_raw_rf_approximation"


def test_eta_uses_completed_same_policy_measurement(tmp_path) -> None:
    policy = select_runtime_policy({"cuda_available": True, "vram_total_mb": 7705})
    run = tmp_path / "run"
    (run / "metadata").mkdir(parents=True)
    (run / "state.json").write_text(
        json.dumps(
            {
                "plan": {"runtime_policy_id": policy.policy_id},
                "detect_summary": {"n_images": 10, "runtime_seconds": 120, "n_errors": 0},
            }
        ),
        encoding="utf-8",
    )
    (run / "metadata" / "system_profile.json").write_text(
        json.dumps({"gpu_name": "Example GPU"}), encoding="utf-8"
    )
    estimate = estimate_runtime(
        35,
        policy,
        {"gpu_name": "Example GPU"},
        tmp_path,
    )
    assert estimate.seconds_per_image == 12
    assert estimate.total_seconds == 420
    assert estimate.sample_count == 1


def test_eta_uses_latest_substantial_same_system_run(tmp_path) -> None:
    policy = select_runtime_policy({"cuda_available": True, "vram_total_mb": 7705})
    for run_id, count, seconds in (
        ("20260716T120000Z", 35, 17.5),
        ("20260716T130000Z", 6, 150.0),
        ("20260717T010000Z", 35, 812.0),
    ):
        run = tmp_path / run_id
        (run / "metadata").mkdir(parents=True)
        (run / "state.json").write_text(
            json.dumps(
                {
                    "plan": {},
                    "detect_summary": {
                        "n_images": count,
                        "runtime_seconds": seconds,
                        "n_errors": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run / "metadata" / "system_profile.json").write_text(
            json.dumps({"gpu_name": "Example GPU"}), encoding="utf-8"
        )

    estimate = estimate_runtime(35, policy, {"gpu_name": "Example GPU"}, tmp_path)

    assert estimate.seconds_per_image == 812 / 35
    assert estimate.total_seconds == 812
    assert estimate.sample_count == 2
