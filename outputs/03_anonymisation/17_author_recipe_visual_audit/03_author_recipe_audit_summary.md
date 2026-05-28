# Author-Recipe Visual Audit

## Purpose

This audit tests whether the advanced anonymisers' visual failures were caused
by avoidable implementation choices. Five deliberately difficult CASTLE frames
cover large frontal, profile/occluded, small distant, edge/partial, and
multi-face conditions. They contain six manually reviewed faces. This is a
targeted stress gate, not a prevalence estimate for the 500-frame protocol.

The gate requires complete reviewed-face coverage, plausible compositing,
observable privacy change, and preservation of pose/expression-relevant visual
structure. A method is not default-route eligible when it misses a reviewed
face, leaves identity change uncertain, or introduces conspicuous geometry,
pose, expression, or blending failures.

## Recipe Findings

- **RiDDLE:** FFHQ landmark alignment, four password candidates, colour
  matching, inverse warping, and feathered blending improve placement. They do
  not prevent pose/expression drift, implausible small-face synthesis, or
  incomplete multi-face coverage.
- **FAMS:** the authors specify 25 denoising steps for unaligned multi-face
  use, whereas 200 steps apply to an already aligned single-face example. Both
  the 25-step CASTLE sweep and a 200-step author-sample control produced severe
  coloured corruption with the obtainable SD2.1 mirror. The original
  `stabilityai/stable-diffusion-2-1` identifier was inaccessible, so exact
  checkpoint reproduction is externally constrained; the failure is not
  attributed to CASTLE alone.
- **NullFace:** author-style alignment and a localized mask that retains the
  eyes and mouth reduce broad patch artifacts. Large/edge/multi-face geometry
  remains malformed, while profile and distant cases show weak privacy change.
- **StyleID:** the author-selected latent layers, alignment, tone matching, and
  soft compositing do not solve pasted-face appearance, pose mismatch, or
  incomplete multi-face alignment.
- **Low-step diffusion:** reducing guidance from 7.0 to 4.5, removing mask
  expansion, and tightening feathering removes the earlier mask-like seams and
  catastrophic 50-step artifacts. The result remains conditional because
  geometry changes on larger/profile faces and privacy change on the
  multi-face case is uncertain.
- **Reverse Personalization:** the author detector failed four of five stress
  frames and over-detected the remaining frame. Reviewed-box execution at
  guidance `-5`, `-7.5`, and the author default `-10` produced four frames but
  failed the multi-face case at every setting. The successful outputs remain
  appearance/expression-changing rather than attribute-stable.
- **FALCO:** the retained canonical outputs use the full author-recommended
  60,000-reference FaRL pool and 50-epoch W+ optimization. The five hard cases
  still show expression/geometry drift and severe edge/multi-face artifacts;
  reduced-pool evidence was therefore unnecessary.

## Decision

No audited generative method passes the strict default-routing gate across all
five cases. RiDDLE and FALCO retain full comparable research status because
their 500-frame metrics remain valid. NullFace, low-step diffusion, and Reverse
Personalization retain bounded research evidence. StyleID and FAMS remain
quality-limited after systematic tuning. These outcomes do not invalidate the
quantitative comparisons; they show why metric-only promotion is unsafe.

The visual-safe adaptive default therefore remains deterministic: reviewed
no-face copy-through, layered blur/downscale/noise for balanced protection, and
solid masking where stronger concealment is required. Generative methods may
only be reconsidered after an independently validated visual-quality gate
demonstrates complete face coverage and adequate recall of the failure modes
shown here.

## Evidence Boundary

The audit does not claim that every possible parameter combination was tested,
that all generative outputs are defective, or that the five stress cases
estimate population-level failure rates. It establishes that exact or
author-aligned recipes do not remove the observed deployment-critical failures
on the tested egocentric conditions.
