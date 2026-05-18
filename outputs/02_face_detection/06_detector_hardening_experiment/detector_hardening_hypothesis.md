# Face Detector Hardening Hypothesis

The best path is not a single replacement detector. The tested hypothesis is that CASTLE face detection improves when face-specific YOLO and SCRFD candidates are combined with model-size/multiscale variants and agreement-aware fusion.

The score used here is the project detector-routing score: no-face subgroups use specificity, while face-positive subgroups use `0.65*recall + 0.25*F1 + 0.10*precision`.

## Result

The hypothesis is supported. The strongest combined 1,000-image method is:

- `fusion_yolo11s1280_scrfd10g_agreement`
- OAPR detector score: `0.9072`
- F1: `0.9002`
- Precision: `0.8877`
- Recall: `0.9129`
- TP/FP/FN: `2380 / 301 / 227`

This improves over:

- `scrfd_10g_current_640`: OAPR detector score `0.8941`, recall `0.8761`
- `yolo11s_widerface_640`: OAPR detector score `0.8996`, recall `0.8972`
- `yolo8s_lindevs_640`: OAPR detector score `0.8943`, recall `0.8960`

The improvement comes from recall recovery. Agreement-aware fusion accepts extra YOLO11/SCRFD boxes where the privacy-risk cost of a missed face is higher than a moderate false-positive increase.

## Category-level interpretation

The fusion is strongest for broad face-positive categories: all images, downward/egocentric views, edge/partial faces, clutter, mixed-scale faces, multi-face scenes, small faces, and very small/distant faces.

Specialised exceptions remain useful:

- `scrfd_10g_current_640` remains best for no-face specificity.
- `yolo11n_pose_widerface_640` is best for large-face and low-light categories.
- `fusion_yolo11s_scrfd10g_agreement` is best for medium-face and profile/occluded categories.
- `yolo8s_widerface_repo_640` is best for motion-blur/low-sharpness and single-face categories.
- `yolo8s_lindevs_640` remains best on the small outdoor/vehicle subgroup.

## Candidate outcomes

- YOLO11Face WIDERFace weights are runnable and useful.
- DSFD, RetinaFace-MobileNet and YuNet are runnable, but do not beat the best fusion on the combined protocol.
- RetinaFace-ResNet hit an external checkpoint HTTP 502 during smoke testing and was excluded.
- SCRFD HF ONNX mirror variants were attempted, but they were not plug-compatible with the existing InsightFace wrapper and returned broadcast-shape errors. They were excluded from ranking.
- RF-DETR face remains a possible separate setup, but the current fusion already improves the detector score without pulling in a heavier transformer detector.

Final interpretation should be based on `detector_hardening_best_by_subgroup.csv` and `detector_hardening_overall_scores.csv`.
