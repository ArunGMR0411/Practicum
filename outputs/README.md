# Thesis Evidence

This directory contains the canonical evidence retained for the thesis. Read
the numbered folders in order: each stage consumes the preceding protocol or
predictions and produces the evidence needed by the next stage. Raw CASTLE
frames are not distributed here.

The shared evidence package includes metric files, manifests, figures, contact
sheets, source maps, and traceability records. Bulk generated anonymised image
folders are treated as regenerable review artifacts and are not redistributed
with the compact evidence package.

## Numbering Convention

- Numbered directories follow the experimental pipeline.
- Numbered metadata files follow the order in which they are produced or read.
- Generated images retain their CASTLE-relative names. This preserves a stable
  link between an input frame, its manifest row, and every method output.
- `outputs/09_traceability/01_evidence_index.csv` is the canonical interpretation ledger. Files here retain
  measurements, provenance, validation, and reproducibility evidence.

Foundational evidence directories under 01_protocol/ (reviewed data and
protocol manifests, as they are part of the initial protocol definition stage):

- `01_protocol/annotations/` - Reviewer-app verified face and multimodal ground truth.
- `01_protocol/thesis_manifests/` - Locked canonical manifests for all evaluation protocols.

See `outputs/09_traceability/` for full traceability records.

These were relocated from `data/` as they represent evidence rather than raw
inputs.

## Pipeline Story

### 01 Protocol

The project first fixes what will be evaluated.

1. `01_locked_500_input_manifest.csv` defines the 500 unique face-method
   frames.
2. `02_locked_500_protocol_validation.json` confirms 500 inputs, 354 frames
   with reviewed face boxes, 146 without boxes, 1,279 reviewed boxes, and no
   missing inputs at validation.
3. `03_metric_contract.csv` defines required privacy, utility, runtime, and
   failure fields.
4. `04_output_contract.csv` defines the retained evidence expected from each
   method.
5. Files `05`–`09` contain disjoint calibration signals and calibration
   anonymisation, perceptual, and re-identification results.
6. `supporting_protocols/` contains the numbered development, calibration,
   FID, cross-view, and missed-face audit protocols.

Comparable methods must use aligned input rows, explicit output/failure
accounting, SSIM, LPIPS, AdaFace, ArcFace, and runtime evidence. Protocol
completion establishes metric comparability, not visual realism, demographic
consistency, production suitability, or universal superiority.

### 02 Face Detection

Detection is evaluated before anonymisation because an anonymiser cannot
protect a face it does not receive.

1. `01_baseline_detector_development_500/detector_scores.csv` contains the
   compact detector score table for the original 500-frame detector-development
   surface.
2. `02_global_egocentric_stress_500/detector_scores.csv` contains the compact
   detector score table for the harder 500-frame egocentric stress surface.
3. `03_detector_two_protocol_comparison.csv` compares all retained detectors
   across both 500-frame protocols.
4. `04_yolo_root_cause_and_detector_sources.md` records why the invalid
   generic-object YOLO evidence was replaced by face-specific detector evidence.
5. `04_scene_condition_router/` contains the reviewed condition dataset and
   the baseline, hybrid, and final telemetry benchmarks for the evidence-
   supported Scene-Condition Router with fallback.
6. `05_condition_subgroup_analysis/` records which detector performs best for
   each face/scene subgroup and supports detector-aware routing.
7. `13_anonymisation_protocol_face_boxes.csv` is the 1,279-box handoff used by
   the anonymisation protocol.
8. `14_final_detector_policy/` records the primary privacy-weighted detector
   selection and recall-first alternative.
9. `15_deployment_selection/` mirrors detector gates and deployment scores from
   the in-body OAPR decision framework (`outputs/05_oapr/decision_framework/`).

### 03 Anonymisation

This stage compares every retained face anonymisation branch.

1. `01_all_methods_comparison.csv` is the single canonical method table. It
   includes deterministic, advanced, quality-limited, dependency-limited, and
   inaccessible methods with explicit evidence boundaries.
2. `02_deterministic_baselines/` contains the blur, pixelate, solid-mask, and
   layered-obfuscation manifest for the same 500-frame protocol. Bulk rendered
   outputs are regenerable from the retained manifests and code.
3. `03_nullface/` contains the 500-frame manifest, summary, runtime, failure,
   and quality-review evidence. Bulk rendered outputs are regenerable from the
   retained manifests and code.
4. `04_diffusion/` contains generation accounting, runtime, perceptual,
   AdaFace/ArcFace, final metrics, failures, and the two-pass quality review.
   Bulk rendered outputs are regenerable from the retained manifests and code.
5. `05_reverse_personalization/` contains final 482/500 coverage, retry,
   before/after, failure-root-cause, perceptual, Re-ID, and runtime evidence.
   Bulk rendered outputs are regenerable from the retained manifests and code.
6. `06_styleid_fams/` contains the defended pilot manifest, summary, contact
   sheet, and retained StyleID/FAMS images. These methods failed the qualitative
   promotion gate and were not advanced to the locked 500-frame protocol.
7. `07_method_accessibility.csv` records why non-executed branches were
   excluded from comparable evaluation.
8. `08_policy_metric_readiness/` and `09_policy_scoring/` contain the joined
   per-image metric surface and the first condition-aware policy scores.
9. `10_full_policy/` contains the executable six-mode policy and complete
   500-frame routing log.
10. `11_policy_hardening/` contains the stricter three-attacker evaluation,
    face-region utility metrics, grouped held-out routes, privacy-verifying
    cascade comparison, paired bootstrap statistics, and advanced-method
    feasibility boundary.
11. `12_riddle/` contains the complete 500-frame RiDDLE generation manifest,
    full-resolution perceptual metrics, dual-attacker Re-ID evidence, runtime,
    failures, and canonical summary.
12. `13_falco/` contains the corresponding complete FALCO evidence using its
    official 60,000-reference FaRL pairing and 50-step optimisation protocol.
13. `14_group2_comparison/` applies the same third-attacker, face-region utility,
    grouped held-out policy, bootstrap, and visual-review rules to both methods
    and the established comparison set.
14. `15_materialised_adaptive_policy/` contains the 500-image RiDDLE-heavy
    policy ablation, participant-grouped artifact gate, deterministic layered
    fallbacks, direct SSIM/LPIPS and three-attacker Re-ID metrics, runtime, and
    paired bootstrap comparisons. It is quantitative evidence, not the final
    runtime policy.
15. `16_visual_quality_hardening/` contains the uniform advanced-method review,
    targeted RiDDLE/FAMS tuning results, final method eligibility, and the
    500-image visual-safe adaptive policy. This is the current deployment-facing
    policy evidence.
16. `17_author_recipe_visual_audit/` checks seven generative branches against
    their author or author-aligned preprocessing and parameters on five hard
    egocentric cases. The configuration table, per-case gate, and zoomed review
    sheets record why none is eligible for default routing despite targeted
    improvements.
17. `18_deployment_selection/` mirrors eligibility ranking and deployment scores
    from the in-body OAPR decision framework.

### 04 Multimodal Privacy

Face protection alone does not address names, messages, documents, or screens.

`01_multimodal_250_evidence/` is the canonical RQ3 evidence. Read its numbered
files in order:

1. Protocol evolution and author-recipe settings.
2. Development, held-out test, and complete-protocol detector comparisons.
3. Final selected text/screen localization predictions.
4. Text-only, screen-only, combined-risk, and no-risk routing decisions.
5. Held-out combined multimodal-risk metrics.
6. Per-image fixed redaction measurements.
7. Adaptive-versus-fixed redaction comparisons and paired intervals.
8. Development-selected adaptive redaction policy.
9. Final adaptive per-image measurements.
10. Residual detection, readability, obscuration, and utility flags.
11. Compact RQ3 result summary.
12. Detection improvement campaign and residual operator ablations as numbered
    support packages.
13. `15_deployment_selection/` mirrors localisation-oriented detection and
    redaction deployment scores from the in-body OAPR decision framework.

The protocol contains 250 manually reviewed egocentric images, 116 text boxes,
and 139 screen boxes. Localization and redaction are evaluated separately;
neither the high combined-risk recall nor measured region obscuration is a
claim that all visible PII or screen semantics are removed. Presence and
localisation-oriented scores are dual-reported under progressive evaluation.

### 05 OAPR

The Objective-Aware Privacy Router uses the established method evidence rather
than assuming one anonymiser is universally best.

1. `01_selector_criteria.csv` defines criterion coverage.
2. Files `02`–`07` retain routing/materialisation manifests, logs, runtime, and
   failures.
3. Files `08`–`11` contain per-frame, perceptual, and Re-ID evidence.
4. `12_oapr_full_metric_summary.csv` is the canonical routed-output result.
5. Files `13`–`14` compare OAPR with fixed methods and record
   objective-specific wins.
6. Files `15`–`16` contain ablation and oracle-upper-bound evidence.
7. `routing_support/` and `cross_view_analysis/` retain interpretability,
   paired-view, failure-example, and residual-risk support.
8. `decision_framework/` is the **in-body progressive evaluation** package:
   atomic metrics, safety gates, stage deployment scores, sensitivity, Pareto,
   and exploratory-vs-deployment comparison. Stage mirrors live under
   `02_face_detection/15_deployment_selection/`,
   `03_anonymisation/18_deployment_selection/`, and multimodal
   `15_deployment_selection/`. Runner:
   `scripts/oapr_routing/run_decision_framework_evaluation.py`.

Evidence labels used by this project include full comparable, bounded,
exploratory, execution-probe, non-promoted, runtime-limited,
dependency-limited, quality-limited, and literature-only. OAPR is an auditable,
objective-aware policy layer; it is not claimed to globally outperform blur.
Exploratory composites support comparison tables; gates govern deployable
defaults.

### 06 Statistics and Review

1. `01_bootstrap_confidence_intervals.csv` records computable and explicitly
   bounded confidence intervals.
2. `02_failure_mode_taxonomy.csv` consolidates detector, privacy, utility,
   visual, runtime, dependency, and routing failures.
3. `03_fid_webp_baseline.json` is a WebP self-FID pipeline sanity check, not an
   anonymiser comparison.
4. `visual_review/` contains the numbered review sample, two independent
   passes, disagreements, adjudicated labels, aggregate results, and contact
   sheets.

### 07 App Validation

This stage records demonstrator validation evidence.

1. `01_app_validation_manifest.csv` links every validation mode to its logs.
2. `02_app_runtime_summary.csv` records seven validation modes and 700
   processing events.
3. `03_app_failure_log.csv` records failures; the retained validation completed
   with zero processing failures.
4. Bulk per-run logs and rendered demonstrator images are not retained in this
   compact package; their aggregate accounting remains in the manifest,
   runtime summary, and failure log.

The app evidence supports a bounded research demonstrator, not production
deployment or complete anonymisation.

### 07 Adaptive Full Comparison

`07_adaptive_full_comparison/` is the cross-stage comparison surface. It joins
the final face detector, multimodal detector, face-anonymisation policy
ablations, and adaptive multimodal policy against their fixed comparators. The
RiDDLE-heavy tables retain the measured 500-output comparison; stage
`03_anonymisation/16_visual_quality_hardening/` records why the visual-safe
deterministic policy supersedes it as the runtime default.

### 08 Figures

`08_figures/` is reserved for reproducible figures generated from the retained
tables. Figures used by the final paper are maintained in
`docs/thesis_papers/images/`.
Their source tables are mapped in stage `09`.

### 09 Traceability

1. `01_evidence_index.csv` identifies the source artifact for each claim.
2. `02_claim_to_evidence_map.csv` maps claims to evidence and limitations.
3. `03_table_to_source_map.csv` maps report tables to machine-readable sources.
4. `07_core_weight_sha256.csv` records hashes for the main local weights.
5. `07_raw_dataset_integrity.txt` records aggregate dataset integrity.
6. `08_rq_to_evidence_map.csv` maps each research question to its evidence.
7. `09_missing_excluded_file_ledger.csv` records excluded local assets.
8. `10_final_stage_comparison.md` summarises stage results.

### 10 Final evaluation package

`10_final_enhancement_evaluation/` holds final policy-support tables:

1. Visual-safety gate effect (generative research-only support).
2. Adversarial residual-leakage (Re-ID, OCR, screen, VLM-proxy questions).
3. Progressive decision framework final policy tables.
4. Adaptive component ablations.

## Pipeline order (thesis flow)

`01` protocol → `02` face detection → `03` face anonymisation → `04` multimodal
→ `05` OAPR (+ decision framework) → `06` validation/statistics → `07`
comparisons/app validation → `08` figures → `09` traceability → `10` final
evaluation package.

## Reproducibility

Use Python 3.10 and one project virtual environment. Install
`requirements.txt`; use `requirements-core.txt` for lightweight table and
manifest work and `requirements-advanced.txt` for advanced
methods. App dependencies remain under `app/requirements.txt`.

Long-running methods use manifest-driven, restartable shards:

1. build the locked manifest;
2. split it into bounded shards;
3. validate dependencies, memory, throughput, and output format;
4. derive concurrency from available CPU, memory, and accelerator capacity;
5. retain per-shard logs and manifests;
6. consolidate outputs against the locked rows;
7. compute metrics only after materialisation and alignment.

Sharding is an engineering mechanism for reproducibility. It does not change
the scientific claim boundary. GPU metrics must not silently fall back to CPU
when the metric contract requires accelerator execution.

## Start Here

1. `01_protocol/01_locked_500_input_manifest.csv`
2. `02_face_detection/03_detector_two_protocol_comparison.csv`
3. `03_anonymisation/01_all_methods_comparison.csv`
4. `03_anonymisation/16_visual_quality_hardening/08_visual_quality_hardening_summary.md`
5. `04_multimodal_privacy/01_multimodal_250_evidence/11_rq3_final_summary.md`
6. `05_oapr/12_oapr_full_metric_summary.csv`
7. `09_traceability/01_evidence_index.csv`
8. `outputs/09_traceability/01_evidence_index.csv`
