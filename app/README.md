# Egocentric Privacy Pipeline App

Local demonstrator for detecting and protecting faces, text, and screens.

```text
Setup -> Preflight -> Detect -> Review -> Anonymise -> Report -> Done
```

## Install

Use Python 3.10 from the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r app/requirements.txt
```

Advanced generative methods also require:

```bash
python -m pip install -r requirements-advanced.txt
python -m pip install ninja
```

## Core assets

| Asset | Path |
| --- | --- |
| Screen YOLO11s | `app/models/multimodal_screen_yolo11s.pt` |
| YOLO11Face | `data/models/face_detection_candidates/yolo11s_widerface.pt` |
| RF-DETR face | `data/models/face_detection_candidates/rfdetr_medium_face.pth` |
| SCRFD 10G | `~/.insightface/models/buffalo_l/det_10g.onnx` |
| EasyOCR | User cache, downloaded on first use |

Download RF-DETR:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Herojayjay/RFDETR-Face-Detection",
    filename="rfdetr_medium_face.pth",
    local_dir="data/models/face_detection_candidates",
)
PY
```

Install SCRFD:

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0, det_size=(640,640))"
```

The screen model and evaluated YOLO11Face weight must retain the exact filenames shown above. Missing detector tiers are replaced by the next available tier.

## Advanced backends

The Preflight selector includes NullFace, diffusion, RiDDLE, FALCO, FAMS, Reverse Personalization, and StyleID. Their adapters use these locations:

| Method | Source or asset path |
| --- | --- |
| NullFace | `third_party/nullface/` |
| Diffusion | `data/models/stable-diffusion-inpainting/` |
| FAMS | `third_party/face_anon_simple/` |
| StyleID | `third_party/styleid/pretrained_models/` |
| RiDDLE | `third_party/riddle/`, `data/models/riddle/` |
| FALCO | `third_party/falco/models/pretrained/` |
| Reverse Personalization | `third_party/reverse_personalization/` |

Availability and compute suitability are checked before execution. If a selected backend cannot run, the configured deterministic fallback is applied and both method IDs are recorded.

## Run

Place permitted images in `app/inputs/`, then start the web interface:

```bash
source .venv/bin/activate
python app/run_web.py
```

The interface opens at `http://127.0.0.1:7860` by default.

Workflow:

1. Select a protection profile and input folder.
2. Review the Preflight plan and compute recommendations.
3. Run face, text, and screen detection.
4. Review or edit detected regions.
5. Apply protection.
6. Inspect the report and protected previews.

Runs are stored under `app/outputs/runs/<run_id>/`:

```text
state.json
input_manifest.csv
detections/
anonymised/
side_by_side/
metadata/
report/success_report.md
```

Input images and run outputs are ignored by Git.

## Compute recommendations

Recommendations use available CUDA memory. The operator may override them.

| Environment | Default face policy |
| --- | --- |
| CUDA, 12 GB or more | RF-DETR + YOLO11Face + SCRFD |
| CUDA, 6 GB or more | RF-DETR + SCRFD |
| CUDA, lower memory | YOLO11Face + SCRFD |
| CPU | YOLO11Face |

Deterministic face protection has negligible GPU demand. Advanced methods show their expected memory requirement in Preflight.

## CLI

```bash
python app/run_cli.py --input-dir app/inputs --mode layered --output-dir app/outputs/manual_run
python app/run_cli.py --input image.webp --mode blur --output-dir app/outputs/manual_run
python app/run_cli.py --manifest manifest.csv --mode objective_profile --objective privacy_first --output-dir app/outputs/manual_run
```

Fixed modes are `blur`, `pixelate`, `solid_mask`, and `layered`.

## Tests

```bash
export PYTHONPATH="$PWD:$PWD/app/src"
python -m pytest tests/pipeline_app -q
```

## Data handling

Processing is local. Use only images you are authorised to process. Do not commit private images, model weights, or generated runs.
