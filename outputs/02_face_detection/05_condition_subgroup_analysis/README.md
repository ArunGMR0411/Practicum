# Face-Condition Subgroup Detector Analysis

This folder contains subgroup-specific detector evidence for the baseline 500 and global egocentric 500 face-detection protocols. It supports a stronger OAPR design by showing that detector choice can be conditioned on scene/face categories rather than fixed globally.

## Structure

- `01_baseline_500/`: subgroup detector metrics for the baseline reviewed 500-image protocol.
- `02_egocentric_stress_500/`: subgroup detector metrics for the current global egocentric 500-image protocol.
- `combined_best_detector_by_subgroup.csv`: side-by-side best detector recommendations by subgroup.
- `combined_best_detector_by_subgroup.md`: readable version of the same recommendations.

## Selection interpretation

F1 is a useful headline metric when a subgroup has ground-truth faces because it balances precision and recall. For privacy routing, the OAPR detector choice should weight recall more heavily in face-positive subgroups because missed faces are privacy failures. For no-face subgroups, F1 is not meaningful because there are no ground-truth boxes; the useful signal is false-positive control/specificity.

## Current high-level result

| Protocol | All-images F1 winner | OAPR detector choice |
|---|---|---|
| 01_baseline_500 | yolo_face_scrfd_fallback (`0.9037`) | yolo_face_scrfd_fallback (`0.8736`) |
| 02_egocentric_stress_500 | scrfd (`0.9357`) | scrfd (`0.9262`) |
