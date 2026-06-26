# No-Action Copy Safety Audit

**Total no-action copy frames in final policy:** 133

**Safety gate results:**
- The RF-DETR candidate safety gate was applied to all predicted no-face frames.
- It overrode 13 frames that had high-confidence face candidates (preventing unsafe copies).
- After the gate, 0 unsafe false `no_face` decisions remained in the final policy.

**Audit of the 133 final copy-through frames:**
- Frames with safety_candidate_count == 0: 133/133
- Max safety_candidate_count among final copies: 0

**Conclusion:** All 133 no-action copy frames in the final visual-safe policy have no residual high-confidence face candidates after the safety gate. The no-face copy decision is safe on this evidence.

**Source:** outputs/06_end_to_end_thesis_validation/01_integrated_routing_log.csv (safety_candidate_count column)
**Related:** outputs/06_end_to_end_thesis_validation/07_end_to_end_validation_summary.md
