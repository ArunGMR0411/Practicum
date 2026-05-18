# Detector Candidate Box Telemetry

This folder retains per-image candidate boxes from the detector families used by the face-detection policy evidence.

Purpose:

- Preserve true per-detector candidate boxes for detector-disagreement telemetry.
- Support the Step 4 condition-profile model before anonymisation routing.
- Avoid losing evidence by keeping only aggregate detector scores.

Main files:

- `detector_candidate_boxes.csv`: per-detector and policy boxes.
- `detector_candidate_runtime.csv`: runtime/failure evidence.
- `detector_candidate_box_export_summary.csv`: export summary.
