# FALCO Comparable Summary

The method completed the reviewed 500-frame protocol. Full-frame perceptual metrics and face-crop AdaFace/ArcFace metrics use the same definitions as the established comparison methods.

| method   |   n_input_frames |   n_success |   n_failure |   face_crops |   SSIM_mean |   LPIPS_mean |   AdaFace_cosine_mean |   AdaFace_reid_rate |   ArcFace_cosine_mean |   ArcFace_reid_rate |   runtime_mean_seconds |   runtime_total_seconds |   peak_vram_gib | gpu_name              | metric_scope                                           |
|:---------|-----------------:|------------:|------------:|-------------:|------------:|-------------:|----------------------:|--------------------:|----------------------:|--------------------:|-----------------------:|------------------------:|----------------:|:----------------------|:-------------------------------------------------------|
| falco    |              500 |         500 |           0 |         1279 |    0.982634 |    0.0203222 |             0.0819824 |          0.00156372 |             0.0722763 |          0.00156372 |                16.6231 |                 8311.53 |         47.7404 | cuda_accelerator_80gb | reviewed_500_frames_full_resolution_1279_face_protocol |
