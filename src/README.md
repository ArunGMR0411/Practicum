# Research Source Code

Reusable implementation shared by the experimental scripts and application:

- `data/`: CASTLE loading and fixed-subset utilities.
- `detection/`: face, text, and screen detectors and combined policies.
- `anonymisation/`: deterministic redaction and advanced-method adapters.
- `evaluation/`: detection, perceptual, OCR, FID, and Re-ID metrics.
- `policy/`: policy-registry access.
- `routing/`: objective-aware routing and decision logging.
- `utils/`: compute profiles, configuration, logging, timing, and run IDs.
- `pipeline.py`: shared pipeline entry point.

Experiment entry points are grouped under `../scripts/`.
