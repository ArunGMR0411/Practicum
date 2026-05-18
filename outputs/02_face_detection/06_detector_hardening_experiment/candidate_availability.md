# Detector Candidate Availability

- `YOLO11Face yolo11s_widerface`: `RUNNABLE`. Tested as Ultralytics WIDERFace detector. Source: https://github.com/zjykzj/YOLO11Face/releases/tag/v1.0.0
- `YOLO11Face yolo11n-pose_widerface`: `RUNNABLE`. Tested because pose-trained face boxes may improve partial/profile cases. Source: https://github.com/zjykzj/YOLO11Face/releases/tag/v1.0.0
- `SCRFD current 10G InsightFace ONNX`: `RUNNABLE`. Tested as the canonical SCRFD implementation. Source: InsightFace buffalo_l/det_10g.onnx
- `SCRFD HF ONNX 500M/1G/2.5G/34G mirrors`: `ATTEMPTED_NOT_PLUG_COMPATIBLE`. Attempted through existing InsightFace/SCRFD path; variants returned broadcast-shape errors and are excluded from ranking. Source: https://huggingface.co/RuteNL/SCRFD-face-detection-ONNX
- `RF-DETR face checkpoint`: `SEPARATE_SETUP_RECOMMENDED`. Checkpoint exists, but requires RF-DETR runtime and is medium transformer detector; not pulled into the constrained run unless final fusion still underperforms. Source: https://huggingface.co/Herojayjay/RFDETR-Face-Detection
- `SFE-DETR`: `NOT_RUNNABLE_FROM_CLEAN_ASSET_SEARCH`. Keep as literature candidate only until reproducible weights are identified. Source: repository/checkpoint not found in HF/GitHub search
- `YOLOv12-face`: `NOT_RUNNABLE_FROM_CLEAN_ASSET_SEARCH`. Do not include without reproducible face-trained weights. Source: no clean WIDERFace face-specific weights found in HF/GitHub search
- `DSFD 1920`: `RUNNABLE`. Tested as a slower but feasible detector family; did not beat fusion overall. Source: installed face_detection package
- `RetinaFace-MobileNet 1920`: `RUNNABLE`. Tested as a lightweight RetinaFace-family candidate; did not beat YOLO/SCRFD fusion. Source: installed face_detection package
- `RetinaFace-ResNet50 1920`: `EXTERNAL_ASSET_FETCH_FAILED`. Smoke test attempted; checkpoint fetch returned HTTP 502 and was excluded from ranking. Source: installed face_detection package external checkpoint URL
- `YuNet 2026 1280`: `RUNNABLE`. Tested as lightweight OpenCV Zoo detector; high recall but too many false positives for top ranking. Source: data/models/face_detection_yunet_2026may.onnx
