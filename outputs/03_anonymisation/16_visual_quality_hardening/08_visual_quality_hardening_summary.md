# Visual Quality Investigation and Policy Hardening

## Result

The investigation found a domain-transfer failure shared by the evaluated generative face methods: aligned-portrait generators can suppress identity strongly while inserting a face whose pose, gaze, expression, geometry, or blend is implausible in a 4K egocentric frame. Pixel metrics and Re-ID scores did not reliably expose this failure. RiDDLE therefore cannot remain the dominant adaptive route.

FAMS was re-executed on five deliberately different archetypes using the authors' 512-pixel, 25-step unaligned multi-face recipe and anonymisation degrees 1.20, 1.25, and 1.40. A separate 512-pixel, 200-step control used the author-supplied aligned sample. Severe coloured corruption persisted in both controls with the obtainable SD2.1 mirror. The original Stability AI model identifier was inaccessible, so exact checkpoint reproduction remains externally constrained; the failure cannot be attributed to CASTLE or crop alignment alone.

RiDDLE was tested with five-point landmark alignment, four deterministic identity candidates, colour matching, inverse warping, feathered blending, and condition-aware fallback. Alignment improved placement, but the one generation-eligible large face still changed pose and gaze unnaturally. The robust improvement came from refusing unsupported cases, not from making generation universally reliable.

## Root causes

- **Domain mismatch:** RiDDLE/FALCO were built around aligned portrait distributions; CASTLE contains profile, partial, distant, blurred, edge, and multi-face views.
- **Compositing mismatch:** square crop generation and broad masks can produce visible patch boundaries or a mask-like face.
- **Objective blind spot:** Re-ID, SSIM, and LPIPS can reward privacy/background preservation while missing facial naturalness and pose consistency.
- **Detection interaction:** false-positive boxes can cause a generative method to insert a face where none exists.
- **FAMS incompatibility:** the documented model configuration and pinned dependency control both failed on the targeted egocentric archetypes.

## Final policy

The canonical default route is now visual-safe and deterministic. It uses no-action copy-through only on reviewed no-face frames, layered blur/downscale/noise as the balanced default, and solid mask where the privacy-weighted policy requires stronger concealment. Blur and pixelation remain eligible candidates, although the balanced 500-frame policy does not select them after scoring.

| Final action | Frames | Share |
|---|---:|---:|
| `layered_blur_downscale_noise` | 286 | 57.2% |
| `no_action_copy` | 133 | 26.6% |
| `solid_mask_black` | 81 | 16.2% |

Across 500 frames the policy completed 500/500 outputs. Its component means are privacy `0.9905`, utility `0.8914`, runtime `0.9350`, success `1.0000`, and objective `0.9562` under the existing balanced-standard scoring definition.

Generative methods remain valuable comparable research evidence, but none is a default runtime action until an independently validated naturalness/pose gate can demonstrate adequate recall on egocentric failures.

The subsequent author-recipe audit extended the same five-case gate to RiDDLE,
FAMS, NullFace, StyleID, low-step diffusion, Reverse Personalization, and the
canonical full-pool FALCO outputs. Alignment, localized masking, guidance
sweeps, and tighter diffusion blending improved isolated failure modes, but no
method achieved complete face coverage, adequate privacy change, and stable
pose/expression preservation across all five cases. The full audit is retained
under `outputs/03_anonymisation/17_author_recipe_visual_audit/`.

## Evidence boundary

This hardening establishes a defensible deployment policy for the reviewed CASTLE protocol. It does not claim that deterministic obfuscation preserves facial aesthetics, that every generative artifact has been enumerated, or that the policy generalises beyond the evaluated domain without validation.
