# Egocentric Visual Privacy Pipeline

This MSc practicum develops and evaluates a visual multimodal privacy pipeline
for egocentric images. It detects faces, readable text, and screens; applies
privacy protection; measures privacy, utility, visual quality, runtime, and
failures; and uses an Objective-Aware Privacy Router (OAPR) to select eligible
methods.

CASTLE is the experimental image source. The repository retains the reviewed
protocols, annotations, evaluation code, metric tables, and final IEEE paper.
Raw CASTLE frames and most model weights are local assets and are not
redistributed.

App installation and operation are documented separately in
[`app/README.md`](app/README.md).

## Project structure

| Path | Purpose |
| --- | --- |
| `src/` | Shared detection, anonymisation, evaluation, policy, and routing code |
| `scripts/` | Protocol construction, experiments, metrics, and reporting commands |
| `configs/` | Policy registry, scoring definitions, and compute profile |
| `outputs/01_protocol/` | Reviewed annotations and locked evaluation manifests |
| `outputs/02_face_detection/` | Face-detection and scene-condition evidence |
| `outputs/03_anonymisation/` | Deterministic and advanced anonymisation evidence |
| `outputs/04_multimodal_privacy/` | Text, screen, and redaction evidence |
| `outputs/05_oapr/` | OAPR routing and policy-selection evidence |
| `outputs/06_statistics_and_review/` | Statistical and visual-review evidence |
| `outputs/07_adaptive_full_comparison/` | Cross-stage comparisons |
| `outputs/09_traceability/` | Claim, table, and evidence maps |
| `outputs/10_final_enhancement_evaluation/` | Final policy and residual-risk evaluation |
| `docs/thesis_papers/` | Final paper, bibliography, figures, and 24 source papers |
| `tests/` | Unit, integration, public-fixture, and local-asset tests |
| `third_party/` | Source adapters for evaluated research methods |

## Evaluated protocols

Protocol membership is fixed. Reproduction must use the retained manifests and
must not draw new random samples.

| Protocol | Size | Manifest or annotation source |
| --- | ---: | --- |
| Baseline face detection | 500 images | `baseline_face_detection_500.csv` |
| Egocentric-stress face detection | 500 images | `final_face_detection_500.csv` |
| Face anonymisation | 500 images | `final_face_anonymisation_500.csv` |
| Advanced-method comparison | 500 images | `final_advanced_methods_500.csv` |
| Multimodal privacy | 250 images | `final_multimodal_250.csv` |
| Reviewed multimodal boxes | 250 images | `reviewed_multimodal_250_with_boxes.csv` |

The manifests are under `outputs/01_protocol/thesis_manifests/`. Reviewed face
and multimodal annotations are under `outputs/01_protocol/annotations/`.

## Environment

Python 3.10 is recommended. Run commands from the repository root.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/app/src"
```

Dependency sets:

- `requirements-core.txt`: manifests, tables, statistics, and lightweight tests.
- `requirements.txt`: main experimental environment.
- `requirements-advanced.txt`: optional generative-method dependencies.
- `app/requirements.txt`: App-only dependencies, documented in `app/README.md`.

CUDA is required for the GPU detector stack and advanced generative methods.
Protocol validation, retained-result inspection, table generation, and most
tests can run without those methods.

## Private data

CASTLE must be placed locally using this layout:

```text
data/castle2024/
  raw/
    <day>/members/<participant>/<frame>.webp
  raw_dataset_index.csv
```

The raw frames contain identifiable personal information. Do not commit or
redistribute them. Reviewed annotations and relative paths are retained as
project evidence under `outputs/01_protocol/`.

## Model assets

Model weights are excluded from version control. The principal local paths are:

| Asset | Expected path |
| --- | --- |
| YOLO11Face | `data/models/face_detection_candidates/yolo11s_widerface.pt` |
| RF-DETR face | `data/models/face_detection_candidates/rfdetr_medium_face.pth` |
| SCRFD 10G | `~/.insightface/models/buffalo_l/det_10g.onnx` |
| Screen YOLO11s | `app/models/multimodal_screen_yolo11s.pt` |
| AdaFace IR-50 | `data/models/adaface_ir50_ms1mv2.ckpt` |
| Stable Diffusion inpainting | `data/models/stable-diffusion-inpainting/` |
| RiDDLE assets | `data/models/riddle/` |

Additional NullFace, FAMS, StyleID, Reverse Personalization, RiDDLE, FALCO,
and G2Face source adapters are under `third_party/`. Their readiness commands
are under `scripts/anonymisation/check_*_readiness.py` and
`scripts/anonymisation/cache_*_assets.py`.

## Reproduction workflow

### 1. Validate the locked protocols

This check validates manifest sizes and uniqueness. It also checks the local
CASTLE index when available.

```bash
.venv/bin/python scripts/data_protocol/verify_locked_protocols.py
```

Require every raw frame to exist:

```bash
.venv/bin/python scripts/data_protocol/verify_locked_protocols.py --require-raw
```

Validate the CASTLE inventory:

```bash
.venv/bin/python scripts/data_protocol/validate_castle_manifest.py
```

### 2. Review annotations

```bash
.venv/bin/python scripts/annotation_review/validate_face_annotations.py
```

The annotation reviewer can be launched with:

```bash
.venv/bin/python scripts/annotation_review/run_face_annotation_reviewer.py
```

### 3. Face detection and condition profiling

The RQ1 scripts are grouped under `scripts/detection/`. The main experimental
families include detector inference, two-protocol evaluation, hardening,
condition slices, runtime-source validation, and scene-condition routing.

```bash
.venv/bin/python scripts/detection/run_face_detector_hardening_experiment.py
.venv/bin/python scripts/detection/run_detector_error_hardening.py
.venv/bin/python scripts/detection/evaluate_runtime_source_detector.py
```

Canonical results are retained under `outputs/02_face_detection/`. Running GPU
inference again is unnecessary when only auditing the reported tables.

### 4. Face anonymisation

RQ2 runners are under `scripts/anonymisation/`. Deterministic methods can be
evaluated independently. Advanced methods require their local source trees,
weights, and compatible CUDA environments.

Key aggregation commands include:

```bash
.venv/bin/python scripts/anonymisation/build_comparable_method_summary.py
.venv/bin/python scripts/anonymisation/build_anonymisation_policy_scores.py
.venv/bin/python scripts/anonymisation/build_canonical_method_eligibility.py
```

Per-method manifests, metrics, failure records, runtimes, and visual reviews
are retained under `outputs/03_anonymisation/`.

### 5. Multimodal privacy

RQ3 evaluates text and screen localisation separately from protection and
residual readability.

```bash
.venv/bin/python scripts/multimodal/run_multimodal_region_evaluation.py
.venv/bin/python scripts/multimodal/run_multimodal_region_redaction_evaluation.py
```

The main evidence package is
`outputs/04_multimodal_privacy/01_multimodal_250_evidence/`.

### 6. OAPR and final policy evaluation

RQ4 combines eligible stage evidence through objective-aware routing and
explicit privacy, utility, runtime, failure, and visual-quality gates.

```bash
.venv/bin/python scripts/oapr_routing/run_decision_framework_evaluation.py
.venv/bin/python scripts/oapr_routing/run_frozen_scientific_oapr_route_replay.py --use-offline-routes
```

Retained routing tables are under `outputs/05_oapr/`. Final policy and residual
analyses are under `outputs/10_final_enhancement_evaluation/`.

### 7. Public smoke protocol

The synthetic smoke protocol exercises the downstream workflow without CASTLE
or detector inference:

```bash
.venv/bin/python scripts/smoke/run_public_smoke_protocol.py
```

## Tests

Run tests that do not require private frames or local model assets:

```bash
.venv/bin/python -m pytest -m "not e2e_assets"
```

Run the complete local suite when CASTLE and model assets are installed:

```bash
.venv/bin/python -m pytest
```

Synthetic public fixtures are under `tests/fixtures/public_micro/`. Tests that
require private data or weights use the `e2e_assets` marker.

## Retained evidence

Re-running every model is not required to inspect or reproduce the conclusions.
The repository retains machine-readable manifests, predictions, metrics,
runtime records, failures, visual-review summaries, statistical comparisons,
and policy decisions.

Start with:

- `outputs/09_traceability/01_evidence_index.csv`
- `outputs/09_traceability/02_claim_to_evidence_map.csv`
- `outputs/09_traceability/03_table_to_source_map.csv`
- `outputs/09_traceability/08_rq_to_evidence_map.csv`
- `outputs/README.md`

Exact numerical values remain tied to the reviewed protocols, model versions,
and execution conditions recorded by the project.

## Thesis paper

The final ten-page IEEE paper is `docs/thesis_papers/main.pdf`. Rebuild it with:

```bash
cd docs/thesis_papers
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

The bibliography contains 24 verified references. Corresponding paper copies
are retained under `docs/thesis_papers/source_papers/`.

## Scope

The framework reduces observed face, text, and screen privacy risk while
retaining useful scene information. It does not claim that one method is best
for every condition or that all contextual privacy risk is removed. Results
should be interpreted using the recorded failure, visual-review, protocol, and
hardware boundaries.

## Licence and external software

See [`LICENSE`](LICENSE) for the repository terms and `app/LICENSE` for the App
terms. Third-party methods retain their own licences. CASTLE usage remains
subject to dataset permission and must not be inferred from the presence of
manifests or derived metrics in this repository.
