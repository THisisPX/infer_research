# Task Plan: QSNR Framework for Rotation-Quantization Analysis

## Goal
Produce a rigorous theoretical framework document (`qsnr_framework.md`) that formalizes why Hadamard/Cayley rotation methods provide significantly less quantization error reduction for FP4/NVFP4 than for INT4, with mathematically grounded claims, experimental protocols, and identification of the highest-impact possible finding.

## Phases
- [x] Phase 1: Research background (NVFP4 format, E2M1 codebook, rotation methods papers)
- [x] Phase 2: Flesh out mathematical formalism (QSNR decomposition, block-structured error analysis)
- [x] Phase 3: Classify claims as analytically provable vs. empirically testable
- [x] Phase 4: Design experimental protocol to distinguish Claims 1, 2, 3
- [x] Phase 5: Identify most surprising possible finding
- [x] Phase 6: Write and review deliverable

## Key Research Findings
1. E2M1 codebook confirmed: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} — non-uniform, dense near zero
2. NVFP4 scaling chain: global FP32 × block E4M3 × E2M1 value, B=16, two-level
3. Claim 1 partially provable via extreme value theory (bound G_B ≤ 1 + O(1/B))
4. Claim 2 analytically provable for scale error existence, but G < 1 requires empirical verification
5. Triple decomposition experimental design separates block size, scale precision, and codebook effects

## Status
**All phases complete** - Deliverable written to qsnr_framework.md (611 lines, 29.7 KB)

### Verification
- Document structure: 10 sections (Problem → Model → Decomposition → Rotation Effects → Claims → Classification → Protocol → Findings → Decisions → Appendix)
- Mathematical content: 2 theorems, 2 propositions, 3 sub-claims each for Claims 1-3
- Experimental design: 5 experiments (A-E) with control variables, metrics, and power analysis
- Key output: Finding S2 (scale precision dominance) identified as most surprising possible finding
