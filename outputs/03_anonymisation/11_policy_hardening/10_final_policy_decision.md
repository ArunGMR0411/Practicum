# Final Anonymisation Policy Hardening Decision

## Evaluation basis

The manually reviewed 500-image egocentric-stress protocol was rescored using three independent identity attackers, face-region utility, landmark geometry, background preservation, measured runtime, and output success. All neural inference used CUDA. Grouped held-out routing prevents category winners from being selected and evaluated on the same participant groups.

The enhanced balanced score is:

`0.50 * three-attacker privacy + 0.30 * enhanced utility + 0.10 * constrained runtime + 0.10 * success`

Privacy is `1 - mean(AdaFace Re-ID, ArcFace Re-ID, FaceNet/VGGFace2 Re-ID)`. Enhanced utility combines full-frame perceptual utility with face-crop SSIM/LPIPS, landmark geometry and background preservation. FaceNet uses cosine threshold `0.60`, with `0.50` and `0.70` sensitivity columns retained in the per-image evidence.

## Findings

- The previous category policy scores `0.888033` under the harder metric surface and no longer beats fixed layered obfuscation (`0.889789`). This supersedes any assumption that adding categories alone guarantees improvement.
- Grouped held-out category routing scores `0.894409`, improving on fixed layered obfuscation by `+0.004620` with 95% bootstrap interval `[0.002673, 0.006261]`. Its 500 routes are 175 blur, 186 layered, 6 pixelate and 133 verified no-action frames.
- The privacy-verifying cascade scores `0.900863`, with privacy `0.995111` and utility `0.715002`. It improves on every fixed comparator under the same enhanced score. Against fixed layered obfuscation the mean gain is `+0.011074`, 95% CI `[0.008609, 0.013819]`.
- The cascade routes 159 blur, 84 pixelate, 42 layered, 36 solid mask, 35 diffusion, 11 NullFace and 133 verified no-action frames. This demonstrates that advanced methods can be selected when per-image privacy and quality gates justify them; they are not excluded by the policy.
- The privacy-constrained Pareto oracle reaches the highest measured utility (`0.798385`) and privacy (`0.994971`) but scores `0.883620` under the balanced objective. It remains an upper-bound trade-off analysis, not the default policy.

## Decision

The grouped held-out category policy is the strongest directly generalised category router. The privacy-verifying cascade is the strongest enhanced policy simulation and the recommended next materialisation target because it verifies residual identity before escalating to a stronger method. It must be described as a retained-output cascade evaluation until its sequential generation-and-verification implementation is materialised end to end.

This feasibility-stage decision was superseded by complete 500-frame FALCO and RiDDLE execution. Both methods now have 500/500 output accounting, 1,279-face AdaFace/ArcFace evidence, full-resolution SSIM/LPIPS, runtime, independent FaceNet and face-region utility metrics, and a deterministic 100-crop visual review. The updated policy evidence is under `12_riddle/`, `13_falco/`, and `14_group2_comparison/`.

RiDDLE reaches hardened balanced score `0.903056` and enters the comparable candidate pool with explicit artifact fallback. FALCO reaches `0.810677`; its strong identity suppression is offset by `16.623` seconds/frame and lower visual plausibility. The updated grouped held-out policy reaches `0.909798`, a `+0.020009` gain over fixed layered obfuscation with 95% CI `[0.017595, 0.022615]`.

Canonical numeric sources are `03_enhanced_method_summary.csv`, `04_grouped_heldout_policy_routes.csv`, `05_final_policy_routes.csv`, `06_final_policy_summary.csv`, `07_paired_policy_statistics.csv`, and `09_advanced_method_feasibility.csv` in this directory.
