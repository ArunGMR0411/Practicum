# Face Detector Two-Protocol Comparison

This table compares the restored baseline 500-image face-detection protocol against the current global egocentric 500-image protocol. Both protocols use manually reviewed face boxes from the project reviewer app.

| Protocol | Method | AP | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_detector_development_500 | yolo_face_s | 0.8317 | 0.9761 | 0.8331 | 0.8989 | 1103 | 27 | 221 |
| baseline_detector_development_500 | scrfd | 0.8322 | 0.9625 | 0.8346 | 0.8940 | 1105 | 43 | 219 |
| baseline_detector_development_500 | yolo_face_scrfd_fallback | 0.8457 | 0.9681 | 0.8474 | 0.9037 | 1122 | 37 | 202 |
| baseline_detector_development_500 | retinaface_mobilenet | 0.7521 | 0.9079 | 0.7591 | 0.8268 | 1005 | 102 | 319 |
| baseline_detector_development_500 | yunet_2026 | 0.7578 | 0.9720 | 0.7591 | 0.8524 | 1005 | 29 | 319 |
| baseline_detector_development_500 | dsfd | 0.8492 | 0.9393 | 0.8535 | 0.8943 | 1130 | 73 | 194 |
| baseline_detector_development_500 | mtcnn | 0.6678 | 0.8372 | 0.6760 | 0.7480 | 895 | 174 | 429 |
| global_egocentric_stress_500 | yolo_face_s | 0.8765 | 0.9390 | 0.8879 | 0.9127 | 1140 | 74 | 144 |
| global_egocentric_stress_500 | scrfd | 0.9119 | 0.9539 | 0.9182 | 0.9357 | 1179 | 57 | 105 |
| global_egocentric_stress_500 | yolo_face_scrfd_fallback | 0.9029 | 0.9378 | 0.9159 | 0.9267 | 1176 | 78 | 108 |
| global_egocentric_stress_500 | retinaface_mobilenet | 0.8099 | 0.9035 | 0.8170 | 0.8581 | 1049 | 112 | 235 |
| global_egocentric_stress_500 | yunet_2026 | 0.8115 | 0.9527 | 0.8154 | 0.8787 | 1047 | 52 | 237 |
| global_egocentric_stress_500 | dsfd | 0.8768 | 0.8832 | 0.8894 | 0.8863 | 1142 | 151 | 142 |
| global_egocentric_stress_500 | mtcnn | 0.7264 | 0.8378 | 0.7360 | 0.7836 | 945 | 183 | 339 |

## Interpretation

- The earlier low YOLO score on the global protocol was caused by evaluating COCO object-detection YOLO weights as if every predicted class were a face. The corrected YOLO evidence uses face-specific YOLOv8-face weights and filters class `0=face`.
- The baseline protocol remains useful as a detector-development surface, but the global egocentric protocol is the stronger stress test because it aligns with the 500-frame anonymisation/OAPR protocol and includes reviewed condition categories.
- SCRFD is the strongest standalone method on the global protocol by F1; corrected YOLO-Face is close; YOLO-Face+SCRFD provides an operational hybrid boundary.
- The detector story should not claim perfect localisation. All methods still have false negatives, especially under egocentric clutter, occlusion, and scale variation.
