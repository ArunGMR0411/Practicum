# YOLO Root Cause and Detector Source Note

## Root cause corrected

The broken global-500 YOLO result was not treated as a valid detector conclusion. The run used COCO YOLO weights (`data/models/yolov8n.pt` / `data/models/yolov9c.pt`) whose class `0` is `person`, not `face`, and the wrapper previously wrote every predicted class as a face box. That produced thousands of false positives and made YOLO appear unusable.

The corrected evidence uses face-specific YOLOv8-face weights (`data/models/yolov8s-face-lindevs.pt`) where class `0` is `face`, and the detector wrapper now supports explicit class filtering.

## Detector families evaluated on both 500-image protocols

- YOLOv8-Face, public release weights from `https://github.com/lindevs/yolov8-face`.
- SCRFD, InsightFace ONNX detector from the `buffalo_l/det_10g.onnx` asset.
- RetinaFace-MobileNet, GPU-backed PyTorch checkpoint exposed by the installed `face_detection` package.
- YuNet 2026, OpenCV Zoo ONNX model from `https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet`.
- DSFD, GPU-backed PyTorch detector exposed by the installed `face_detection` package.
- MTCNN, facenet-pytorch detector.
- YOLO-Face+SCRFD operational hybrid, included because it reflects the fallback-style detector composition used by the pipeline.

## Evidence files

- `outputs/02_face_detection/01_baseline_detector_development_500/`
- `outputs/02_face_detection/02_global_egocentric_stress_500/`
- `outputs/02_face_detection/03_detector_two_protocol_comparison.csv`
- `outputs/02_face_detection/03_detector_two_protocol_comparison.md`
