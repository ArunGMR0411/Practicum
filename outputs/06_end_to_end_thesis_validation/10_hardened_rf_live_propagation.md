# Frozen OAPR route replay with detector-derived application boxes

**Status:** retained execution evidence; terminology corrected

## Canonical frozen-route execution path

| Field | Value |
| --- | --- |
| Detector for application boxes | `runtime_3_source_all_raw_rf_approximation` (historical run used the same three live sources; id corrected at freeze) |
| Pipeline | frozen scientific route decision → detector-derived application boxes → App anonymisation modules |
| Artefact | `outputs/10_final_enhancement_evaluation/06_frozen_scientific_oapr_route_replay/` |
| Summary | `.../metadata/summary.json` |
| Scientific route counts | layered **286** / solid_mask **81** / copy **133** |
| Match | `matches_expected_286_81_133: true` |
| Failures | 0 / 500 |

Script: `scripts/oapr_routing/run_frozen_scientific_oapr_route_replay.py --use-offline-routes`

## Relationship to this folder (`06_end_to_end_thesis_validation/`)

Historical integrated validation under this directory materialised the **same 286/81/133 visual-safe policy** by joining the condition profiler with **precomputed deterministic outputs** (handoff boxes / post-detection conditions). It remains valid metric evidence for policy comparison tables.

The retained 500-frame execution is not fresh scientific routing: it reads the frozen route table, then runs the three-source detector path to obtain application boxes. Scientific route counts are **286/81/133**; actual applied operators are **285 layered / 72 solid mask / 143 copy** because ten face-positive scientific routes received zero live boxes and copied safely. The difference is retained and reported rather than hidden.

## App binding

- Scientific policy id: `oapr_visual_safe_balanced_500` (286/81/133 condition-aware)
- App policy id: `objective_profile` (fixed per-focus face operators; not bit-identical)
- Binding registry: `configs/policy_registry.json` (`scientific_policy_id` + `simplification` notes)

## Decision tables

Face detector deployment ranking is score-led (no force-rank of adopted primary). See:

- `configs/scoring_definitions.json`
- `outputs/05_oapr/decision_framework/03_stage_scores/01_face_detection_deployment.csv`
