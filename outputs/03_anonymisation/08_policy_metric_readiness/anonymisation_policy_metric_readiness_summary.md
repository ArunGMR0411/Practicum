# Anonymisation Policy Metric Readiness

Purpose: decide whether category-aware OAPR anonymisation scoring can run from retained evidence, or whether method generation/metric recomputation is needed.

Inputs:

- 500-frame anonymisation manifest: `outputs/01_protocol/thesis_manifests/final_face_anonymisation_500.csv`.
- Manual condition profile: `outputs/01_protocol/annotations/face_detection/02_egocentric_stress_500/manifest.csv`.
- Final OAPR condition profile: `outputs/02_face_detection/10_post_detection_condition_annotation/post_detection_condition_predictions.csv` filtered to `fixed_policy_detector_telemetry_multiclass_scale_hybrid_available_handoff_boxes`.
- Multimodal risk policy: `outputs/04_multimodal_privacy/01_multimodal_250_evidence/04_multimodal_risk_policy.csv`.
- Consolidated method metrics: `outputs/03_anonymisation/01_all_methods_comparison.csv`.

Findings:

- Policy-ready per-image metric methods now: `blur, pixelate, solid_mask_black, layered_blur_downscale_noise, nullface, diffusion_low_step`.
- Methods with output images but missing per-image metric detail: `none`.
- Partial/bounded candidates requiring failure penalty: `reverse_personalization`.
- Quality-limited/non-policy methods: `styleid_stylegan, fams`.

Compute decision:

- Additional high-memory compute is not needed for the policy-readiness audit or scoring logic.
- Deterministic and NullFace output regeneration is not needed because those outputs already exist.
- Per-image perceptual/ReID details are now complete for deterministic baselines and NullFace.
- Diffusion already had full per-image perceptual and per-face ReID detail.
- Reverse Personalization has detailed metrics for the successful 482 outputs and must be scored with an explicit 18-frame failure penalty.

Boundary:

- Do not choose category-specific anonymisation winners from aggregate method means alone.
- The next step is category-aware OAPR anonymisation score computation from the completed per-image evidence.
