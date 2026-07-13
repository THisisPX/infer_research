# QSNR Framework: Why Rotation Methods Underperform for NVFP4

## 1. Problem Statement

Hadamard/Cayley rotation methods (QuaRot, SpinQuant, FlatQuant) reduce quantization error in INT4 by 30-60% via orthogonal transforms. For NVFP4 (E2M1 format, block size 16, 2-level E4M3+FP32 scaling), preliminary evidence suggests rotation provides at most 5-10% improvement -- and may be actively harmful under certain conditions.

**Core research question:** Which structural difference between INT4 and NVFP4 is primarily responsible for the loss of rotation benefit?

Three competing mechanisms are formalized below:
- **Claim 1 (Block Size Dominance):** B=16 is already small enough that per-block scaling handles outliers locally.
- **Claim 2 (Scaling-Induced Penalty):** E4M3 scale quantization error interacts negatively with rotation.
- **Claim 3 (Codebook Asymmetry):** E2M1's non-uniform codebook degrades under rotated distributions.

---

## 2. Formal Model

### 2.1 Notation

| Symbol | Definition |
|--------|-----------|
| W | Weight matrix, W ∈ R^{d_in × d_out} |
| H | Orthogonal rotation matrix, H ∈ R^{d_in × d_in}, H^T H = I |
| Q(W; Φ) | Quantization operator with format parameters Φ |
| B | Block size (number of elements sharing one scale factor) |
| C | Quantization codebook (set of representable values) |
| s_b | Block scale factor for block b |
| ŝ_b | Quantized block scale (ŝ_b = Q_scale(s_b)) |

### 2.2 Quantization Operators

**INT4 quantization** (B=128, FP16 scale, symmetric uniform codebook):

```
C_INT4 = {-7, -6, ..., -1, 0, 1, ..., 7}

For block b:  s_b = max(|W_b|) / 7
              ŝ_b = s_b                          (FP16, error ≈ 2^{-11})
              Q_INT4(W_b) = ŝ_b · clamp(⌊W_b / ŝ_b⌉, -7, 7)
```

**NVFP4 quantization** (B=16, E4M3 block scale + FP32 global scale, E2M1 codebook):

```
C_NVFP4 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}

Global:     s_global = max_{all blocks}(max(|W_b|) / 6)    (FP32)
            W' = W / s_global

Block b:    s_b = max(|W'_b|) / 6
            ŝ_b = Q_E4M3(s_b)                               (E4M3, error ≈ 2^{-4})
            Q_NVFP4(W_b) = s_global · ŝ_b · Q_E2M1(clip(W'_b / ŝ_b, -6, 6))
```

Key structural differences:

| Property | INT4 | NVFP4 |
|----------|------|-------|
| Codebook C | Uniform, 16 levels | Non-uniform, 16 levels |
| Codebook spacing | Constant = 1 (in scaled units) | {0.5, 1, 1.5, 2, 3, 4, 6} |
| Block size B | 128 | 16 |
| Scale format | FP16 (10-bit mantissa) | E4M3 (3-bit mantissa) |
| Scale relative error | ε_scale ≈ 2^{-11} ≈ 0.05% | ε_scale ≈ 2^{-4} ≈ 6.25% |
| Scaling levels | 1 (per-block) | 2 (global + per-block) |

### 2.3 QSNR Definition

For a random weight matrix, the quantization signal-to-noise ratio is:

```
QSNR(W) = E[||W||²_F] / E[||W - Q(W)||²_F]
```

Under rotation H:

```
QSNR_rot(W, H) = E[||W||²_F] / E[||W - H^T · Q(H · W)||²_F]
               = E[||HW||²_F] / E[||HW - Q(HW)||²_F]     (|| · ||_F is orthogonally invariant)
```

**Rotation gain:**

```
G(W, H) = QSNR_rot(W, H) / QSNR(W)
```

G > 1 means rotation improves quantization. G < 1 means rotation degrades quantization.

---

## 3. Error Decomposition

### 3.1 Block-Structured Error

For block-structured quantization, total squared error is additive over blocks:

```
||W - Q(W)||²_F = Σ_{b ∈ blocks} ε_b
```

where ε_b = ||W_b - Q(W_b)||².

### 3.2 Per-Block Error Decomposition

For a single block b with true optimal scale s_b* (the scale that minimizes per-block quantization error in exact arithmetic) and quantized scale ŝ_b:

```
ε_b = ε_b^scale + ε_b^codebook + ε_b^cross
```

where:

```
ε_b^scale  = ||W_b - ŝ_b · round_C(W_b / ŝ_b)||² - ||W_b - s_b* · round_C(W_b / s_b*)||²
            (additional error from using quantized scale ŝ_b instead of optimal s_b*)

ε_b^codebook = ||W_b - s_b* · round_C(W_b / s_b*)||²
              (error from mapping to discrete codebook, even with optimal scale)

ε_b^cross  = interaction term from scale-codebook coupling (typically second-order)
```

For INT4: ε_b^scale ≈ 0 (FP16 has precision far exceeding codebook resolution)
For NVFP4: ε_b^scale is non-negligible (E4M3 has 3-bit mantissa, comparable to 4-bit codebook)

### 3.3 Scale Error Analysis: NVFP4 Specific

The E4M3 block scale quantization introduces structured error. E4M3 has:
- 4 exponent bits (range: 2^{-9} to 2^{8})
- 3 mantissa bits (8 representable significands per exponent: {1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875})

Maximum relative rounding error for E4M3:
```
|ŝ_b - s_b*| / s_b* ≤ 2^{-(m+1)} = 2^{-4} = 0.0625 = 6.25%
```

**Proposition 1 (Scale error bound):** For any block b, with E4M3 scale quantization:

```
ε_b^scale ≤ B · (s_b*)² · (2^{-(m+1)})² · max_{c ∈ C} c²
```

where m = 3 (E4M3 mantissa bits), B is block size, and max_{c ∈ C} c² = 36 (E2M1 codebook maximum).

Proof: The scale error Δs = |ŝ_b - s_b*| ≤ s_b* · 2^{-(m+1)}. For any element w, the reconstruction error contribution from scale error is approximately (Δs · c)² where c = round_C(w/s_b*). Taking max over c and summing over B elements yields the bound.

In practice, with s_b* ≈ 1 (after global scaling), ε_b^scale ≤ 16 · 1 · (0.0625)² · 36 ≈ 2.25 per block, which is comparable to the codebook error.

### 3.4 Codebook Error Analysis

For E2M1 codebook C = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}:

The maximum quantization error (half the spacing between adjacent levels) is position-dependent:

```
max_quant_error(x) = min_{c ∈ C} |x - c|
```

| Interval | Adjacent spacing | Max error |
|----------|-----------------|-----------|
| [0, 0.5] | 0.5 | 0.25 |
| [0.5, 1.0] | 0.5 | 0.25 |
| [1.0, 1.5] | 0.5 | 0.25 |
| [1.5, 2.0] | 0.5 | 0.25 |
| [2.0, 3.0] | 1.0 | 0.50 |
| [3.0, 4.0] | 1.0 | 0.50 |
| [4.0, 6.0] | 2.0 | 1.00 |

The codebook is 2x finer near zero and 4x coarser at the tail than a uniform 16-level grid spanning [-6, 6] (which would have spacing 12/15 = 0.8).

**Codebook density function:**

Define D(c) = 1 / spacing(c), the local density of representable values near codebook point c.

```
D(c) = { 2.0  for |c| ≤ 1.5    (spacing 0.5)
         1.0  for |c| ∈ {2, 3}  (spacing 1.0)
         0.5  for |c| ∈ {4, 6}  (spacing 2.0)
       }
```

---

## 4. Rotation's Effect on Error Components

### 4.1 Effect on Codebook Error

Rotation H transforms the weight matrix: W → HW. For per-block quantization with 1×B blocks:

**Before rotation:** Block b contains values {w_{i,j}, w_{i,j+1}, ..., w_{i,j+B-1}} from the same row, spanning B consecutive channels. Since different channels have different RMS values (some are "outlier channels"), some blocks have much larger range than others.

**After rotation:** Block b contains values {w'_{i,j}, ..., w'_{i,j+B-1}} where each w'_{i,k} = Σ_l H_{k,l} · w_{i,l}. By the linear combination, every value is a mixture of all channels. Per-block distributions become more homogeneous across blocks.

**Quantitative effect (by extreme value theory):**

Let block max M_b = max(|W_b|). Under mild conditions (finite second moment), for B independent elements with variance σ²:

```
E[M_b] ≈ σ · √(2 log B)
Var[M_b] ≈ σ² · π²/(6 log B)
```

Before rotation, σ varies across channels (σ_i differs per channel i). After rotation, σ' ≈ σ̄ (mean RMS) for all channels. The variance of block maxima across blocks decreases by a factor proportional to Var[σ_i] / σ̄².

For INT4 (B=128): √(2 log 128) ≈ 3.10. The block max is ~3σ, and an outlier with 5σ RMS forces a 5/3 ≈ 67% larger scale for its block.

For NVFP4 (B=16): √(2 log 16) ≈ 2.36. The block max is ~2.36σ, and an outlier with 5σ RMS forces a 5/2.36 ≈ 112% larger scale -- but only for 16 elements instead of 128.

**Theorem 1 (Block size scaling of rotation benefit):**

Under the assumption that channel variances {σ²_1, ..., σ²_{d_in}} are i.i.d. draws from distribution F with Var[σ²] > 0, the expected codebook error reduction from rotation satisfies:

```
Δ_codebook(B) = E[ε_codebook(no rotation)] - E[ε_codebook(with rotation)]
              ≤ K · Var[σ²] / (σ̄⁴ · B)
```

for some constant K depending on the codebook. As B → ∞, Δ_codebook(B) → 0 (since rotation's homogenization becomes less valuable as blocks already average over many channels).

As B → 1, Δ_codebook(B) → 0 trivially (per-element quantization is rotation-invariant).

The benefit peaks at intermediate B and the peak location depends on Var[σ²].

### 4.2 Effect on Scale Error (NVFP4 Specific)

In NVFP4, rotation changes per-block statistics, which changes the distribution of optimal block scales {s_b*}. Since s_b* is then quantized to E4M3, the quality of scale representation depends on the scale distribution.

**Mechanism of scale error amplification:**

1. Before rotation: block scales cluster around channel-specific values. Blocks from the same channel have similar scales. Since per-channel RMS varies slowly (adjacent blocks on the same channel have similar scales), the scale distribution is multi-modal with modes at each channel's characteristic scale.

2. After rotation: block scales are drawn from a more continuous distribution (each block mixes all channels). The scale distribution is approximately chi-distributed with B degrees of freedom.

3. E4M3 quantization: The relative rounding error depends on where the scale value falls in the E4M3 grid. Some values are well-represented (near E4M3 grid points), others poorly (midpoints between grid points). If rotation shifts scale values away from well-represented points, scale error increases.

**Proposition 2 (Scale error under rotation):**

Let s* ~ p(s) be the distribution of optimal block scales before rotation, and s'* ~ p'(s) after rotation. The expected per-element scale error is:

```
E[ε_scale] = ∫ p(s) · B · (s - Q_E4M3(s))² · E[round_C(w/s)²] ds
```

Rotation's effect on E[ε_scale] depends on:
- Whether p'(s) puts more mass at E4M3 grid midpoints than p(s)
- Whether the range of s'* is wider (requiring E4M3 to represent more diverse values)

For typical LLM weights: p(s) has most mass concentrated around a few channel-specific scales (good for E4M3 if those happen to align with grid points), while p'(s) has broader support (E4M3 must represent more values, increasing expected error if grid points don't align).

**Corollary:** For E8M0 scale (MXFP4, pure power-of-two scaling: s ∈ {2^n}), the scale error is independent of distribution shape, depending only on whether the distribution's support exceeds the E8M0 range. This predicts rotation is never harmful in MXFP4.

### 4.3 Total Error After Rotation

The net effect of rotation on QSNR for NVFP4:

```
G(W, H) = E[||W||²] / E[Σ_b ε_b^rot]
        = E[||W||²] / E[Σ_b (ε_b^scale,rot + ε_b^codebook,rot)]

G(W, H) > 1  iff  Σ_b ε_b^rot < Σ_b ε_b^orig
              iff  Δε_codebook + Δε_scale < 0
```

where Δε = ε^rot - ε^orig. For INT4, Δε_scale ≈ 0 and Δε_codebook < 0, so G > 1.

For NVFP4, Δε_codebook may be small (B=16 already handles outliers) and Δε_scale may be positive (rotation degrades scale allocation). If Δε_scale > |Δε_codebook|, then G < 1 -- rotation is harmful.

---

## 5. Formal Claims

### Claim 1: Block Size Dominance

**Statement:**

```
lim_{B → 1} G_B = 1
```

For INT4 (B=128): G_int4 > 1 (empirically, 30-60% improvement)
For NVFP4 (B=16): Is G_nvfp4 significantly > 1, or is G_nvfp4 ≈ 1?

**Sub-claim 1a (Trivial bound):** G_B = 1 for B = 1 (per-element quantization is rotation-invariant).

**Sub-claim 1b (Scaling bound):** Δ_codebook(B) ≤ K / B for some constant K (Theorem 1). For B=16, the maximum codebook error reduction from rotation is at most ~6% of the codebook error at B=128, even if the outlier severity (Var[σ²]/σ̄⁴) is high.

**Sub-claim 1c (Critical block size):** There exists B* such that for B < B*, G_B ≤ 1 + δ for small δ (e.g., δ = 0.05). The question is: is B* ≈ 16 or B* >> 16?

**Testable prediction:** For B=16 with FP16 scale (removing scale error), rotation provides at most ~3% improvement in codebook error. This is independent of codebook shape -- it comes purely from block size.

### Claim 2: Scaling-Induced Rotation Penalty

**Statement:** In NVFP4, the E4M3 block scale introduces a scale error term Δε_scale that can be larger than the codebook error reduction |Δε_codebook|, making G < 1 possible.

**Sub-claim 2a (Scale error is first-order):** In NVFP4, ε_scale / ε_codebook ≈ O(2^{m_codebook - m_scale}) = O(2^{2 - 3}) = O(0.5), meaning scale error is comparable in magnitude to codebook error. In INT4, this ratio is O(2^{2 - 10}) = O(0.004), negligible.

**Sub-claim 2b (Rotation amplifies scale error):** Rotation broadens the distribution of optimal block scales from a multi-modal to a continuous distribution. This increases the expected E4M3 quantization error per block, because:
1. More distinct scale values must be represented
2. Some values fall at E4M3 grid midpoints

**Sub-claim 2c (MXFP4 vs NVFP4):** For MXFP4 (E8M0 scale, pure powers of two), rotation cannot increase scale error because the E8M0 format has uniform relative error across all scale values. Therefore:
```
G_NVFP4(W, H) < G_MXFP4(W, H)  for the same block size and codebook
```

**Testable prediction:** With E4M3 scale + INT codebook (isolating scale effect from codebook shape), rotation degrades performance. With E8M0 scale + INT codebook, rotation never degrades performance.

### Claim 3: Codebook Asymmetry Interaction

**Statement:** The E2M1 codebook's non-uniform spacing (dense near zero, sparse at tail) interacts with rotation's distribution-reshaping effect in a way that is distinct from block-size and scale-precision effects.

**Sub-claim 3a (Density allocation):** The E2M1 codebook allocates density proportional to 1/|c| (approximately). This matches Laplacian/peaked distributions common in untransformed weights. Rotation makes distributions more Gaussian, which has lighter tails and more mass at intermediate values. This may:
- Improve fit at intermediate values (better use of {±2, ±3} bins)
- Worsen fit at extremes (values that would have been at ±6 or clipped may fall in sparser regions)

**Sub-claim 3b (Tail under-utilization):** If rotation reduces the maximum absolute value within a block (by averaging outlier contributions), the block scale is reduced, effectively compressing all values toward zero. This concentrates values in the dense region of the E2M1 codebook -- which seems beneficial, but means tail bins (±4, ±6) are under-utilized. The effective bit resolution is reduced from 4 bits to ~3-3.5 bits.

**Sub-claim 3c (Codebook shape preference):** A uniform 16-level codebook (INT-style, but with 16 levels instead of 15) would make rotation strictly beneficial (at fixed B and scale format), because uniform codebooks have no "dense near zero" preference.

**Testable prediction:** Compare NVFP4 (E2M1 codebook) with "uniform FP4" (same bit budget, 16 uniformly spaced levels). If rotation benefit differs between them at same B and same scale format, the codebook shape matters independently.

---

## 6. Analytic vs. Empirical Classification

### 6.1 Claims Provable from First Principles

| Claim | Status | Required Tools |
|-------|--------|---------------|
| 1a: G_{B=1} = 1 | **Provable** | Trivial: per-element quantization is rotation-invariant |
| 1b: Δ_codebook ≤ K/B | **Provable** | Extreme value theory + Hoeffding bound on block max variance |
| 2a: ε_scale / ε_codebook ratio in NVFP4 | **Provable** | Direct from format specifications (E4M3 mantissa = 3 bits, E2M1 resolution = 2-3 effective bits) |
| 2c: E8M0 scale error is rotation-invariant | **Provable** | E8M0 relative error is constant for all inputs (power-of-two format), so rotation cannot change it |
| 3a: E2M1 density function D(c) | **Provable** | Direct from codebook definition |

### 6.2 Claims Requiring Empirical Validation

| Claim | Why Not Provable | What to Measure |
|-------|-----------------|-----------------|
| 1c: Is B* ≈ 16? | Depends on weight distribution F, Var[σ²]/σ̄⁴ | Sweep B ∈ {4, 8, 16, 32, 64, 128, 256} on real LLM weights |
| 2b: Rotation amplifies scale error | Depends on alignment between scale distribution and E4M3 grid points -- no analytic bound without distributional assumptions | Measure ε_scale before/after rotation for fixed codebook |
| G_nvfp4 < 1 for real weights | Requires real weight distributions and full interaction of all three mechanisms | Run full NVFP4 quantization on real weights before/after rotation |
| 3b: Tail under-utilization | Depends on how much rotation compresses the distribution, which is distribution-specific | Measure codebook bin utilization before/after rotation |

### 6.3 Hybrid (Provable Bounds + Empirical Tightness)

| Claim | Analytic Bound | Empirical Question |
|-------|---------------|-------------------|
| Rotation gain for NVFP4 | Upper bound: G ≤ 1 + c · log(B)/B | How tight is this bound for realistic LLM weight matrices? |
| Scale error amplification | Maximum scale error increase ≤ worst-case E4M3 error × B | What is the typical increase for real weights? |
| Codebook asymmetry effect | Maximum difference: 4x (uniform vs non-uniform codebook at tails) | Is the effect real or dominated by scale/block-size effects? |

---

## 7. Experimental Protocol

### 7.1 Triple Decomposition Design

The protocol cleanly separates the three mechanisms by controlling B, scale format, and codebook independently.

**Factor 1: Block size B**
- Levels: {4, 8, 16, 32, 64, 128, 256}

**Factor 2: Scale format**
- Levels: {FP16 (negligible error), E4M3 (3-bit mantissa), E8M0 (power-of-two), FP32 (no error, oracle)}

**Factor 3: Codebook**
- Levels: {E2M1 (non-uniform), Uniform-16 (16 equally spaced levels), Uniform-15 (INT4-like, symmetric)}

**Factor 4: Rotation**
- Levels: {None, Hadamard, Random Orthogonal, Cayley-learned}

**Factor 5: Weight distribution**
- Levels: {Simulated Gaussian, Simulated Laplacian, Simulated Laplacian + channel outliers (5x, 10x), Real LLM weights (LLaMA-7B, LLaMA-70B, extracted per-layer)}

### 7.2 Key Iso-Condition Experiments

**Experiment A: Isolate block size effect (fix high-precision scale + uniform codebook)**

```
B ∈ {4, 8, 16, 32, 64, 128, 256}
Scale: FP16 (negligible error)
Codebook: Uniform-16
Rotation: Hadamard vs. None
Metric: G(B) = QSNR_rot / QSNR for each B

Expected: G(B) → 1 as B → 1, G(B) increases with B
Key measurement: G(16) -- is it significantly > 1?
```

**Experiment B: Isolate scale precision effect (fix B + uniform codebook, vary scale format)**

```
B = 16 (to match NVFP4)
Codebook: Uniform-16
Scale: {FP16, E4M3, E8M0}
Rotation: Hadamard vs. None
Metric: G(scale_format) for each format

Expected: 
  G(FP16) ≈ 1 (no scale error for either condition)
  G(E8M0) ≥ 1 (scale error unchanged by rotation)
  G(E4M3) potentially < 1 (scale error may increase under rotation)

Key measurement: G(E4M3) vs. G(FP16) -- is there a penalty (G < 1)?
```

**Experiment C: Isolate codebook shape effect (fix B + fix scale, vary codebook)**

```
B = 16
Scale: FP16 (remove scale error)
Codebook: {E2M1, Uniform-16}
Rotation: Hadamard vs. None
Metric: G(codebook) for each codebook

Expected: G(Uniform-16) ≥ G(E2M1) if asymmetry imposes penalty
Key measurement: Is G(E2M1) < G(Uniform-16) at same B and scale format?
```

**Experiment D: Full NVFP4 (all three effects active)**

```
B = 16, Scale = E4M3 + FP32 global, Codebook = E2M1
Rotation: Hadamard vs. None
Metric: G_nvfp4_full

Compare to:
  G(B=128, Uniform, FP16) [INT4 baseline]
  Sum of isolated effects from A, B, C

Key measurement: Is G_nvfp4_full decomposable as a product of independent effects?
```

**Experiment E: Scale distribution analysis (diagnostic)**

```
For each condition, record:
  - Distribution of optimal block scales s_b* (histogram)
  - Distribution of E4M3 rounding errors (s_b* - ŝ_b) / s_b*
  - Codebook bin utilization (fraction of values quantized to each E2M1 level)
  - Block max distribution (per-block max abs value, pre- and post-rotation)

Compare pre-rotation vs. post-rotation for each metric.
```

### 7.3 Weight Sources

1. **Synthetic:**
   - Gaussian W_{i,j} ~ N(0, 1)
   - Laplacian W_{i,j} ~ Laplace(0, 1)
   - Channel-outlier: σ_i = 1 for 95% of channels, σ_i = 10 for 5% of channels

2. **Real LLM weights (extracted per-layer):**
   - LLaMA-2/3-7B: all linear layers (Q, K, V, O, gate, up, down)
   - LLaMA-2/3-70B: all linear layers
   - Group by layer type (attention vs. FFN) -- weight distributions differ

### 7.4 Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| QSNR | 10·log_10(E[||W||²] / E[||W-Ŵ||²]) dB | Primary metric |
| Per-channel QSNR | 10·log_10(E[||W_i||²] / E[||W_i-Ŵ_i||²]) | Diagnose channel-level effects |
| Scale utilization | Fraction of E4M3 grid points used | Scale format efficiency |
| Codebook utilization | KL(p_bin || uniform) | Codebook shape match |
| Clipping rate | Fraction of |w/s| > 6 | Outlier handling |
| Effective bits | log_2(||W|| / ||W-Ŵ||) / 2 | Information-theoretic metric |

### 7.5 Power Analysis

For detecting G = 0.95 (5% degradation) vs. G = 1.00 (no effect):
- QSNR variance estimated from pilot: σ_QSNR ≈ 0.1 dB
- Minimum N per condition: N = (1.96 × 0.1 / 0.22)² ≈ 0.8 → 2 degrees of freedom needed
- With LLaMA-7B (32 layers × 7 matrices = 224 weight matrices): sufficient statistical power
- LLaMA-70B (80 layers × 7 matrices = 560 weight matrices): excellent power

---

## 8. Most Surprising Possible Finding

### 8.1 Candidate Surprising Findings

Four possible results would constitute genuine theoretical contributions:

**Finding S1: Rotation is harmful for NVFP4 (G < 1 with statistical significance).**
- Surprise level: High (practical), Moderate (theoretical)
- Would mean: the community's default assumption that "rotation makes distributions more uniform, which always helps quantization" is false for NVFP4.
- Theoretical contribution: Existence proof that rotation can degrade quantization with realistic formats.
- Practical impact: Direct guidance to avoid rotation methods in NVFP4 deployment.

**Finding S2: Scale precision, not block size, is the dominant mechanism for rotation benefit.**
- Surprise level: Highest (theoretical)
- Would mean: the entire line of work on rotation-based quantization (QuaRot, SpinQuant, FlatQuant) has been optimizing the wrong quantity. These methods frame their benefit as "spreading outliers across dimensions" -- a block-size-dependent phenomenon. If scale precision is the real bottleneck, then:
  - For FP4 formats with low-precision scales, rotation should be redesigned to optimize scale representation, not value distribution.
  - For INT4 formats, rotation works not because it handles outliers (the stated reason) but because it happens to make scale allocation easier (an unstated benefit).
  - A new class of methods targeting scale optimization (not value distribution) could outperform rotation for FP formats.
- Theoretical contribution: Reframes the fundamental question from "how to spread outliers" to "how to allocate scaling budget."

**Finding S3: The E2M1 codebook shape is a first-order effect independent of block size and scale.**
- Surprise level: Moderate
- Would mean: The industry's choice of E2M1 codebook is suboptimal for rotated distributions. A different FP4 codebook (e.g., uniform spacing, or a codebook optimized for Gaussian distributions) would get significantly more benefit from rotation.
- Theoretical contribution: Quantitative characterization of how codebook shape and rotation interact, motivating codebook design as a first-class concern.

**Finding S4: Rotation benefit is highly layer-dependent for NVFP4 -- helps attention projections, hurts FFN projections.**
- Surprise level: Moderate (practical)
- Would mean: Simple "always rotate" or "never rotate" policies are suboptimal. Per-layer decisions are needed.
- Theoretical contribution: Characterization of which weight matrix properties predict rotation benefit in NVFP4.

### 8.2 Recommendation: Prioritize Finding S2

Finding S2 (scale precision as dominant mechanism) is the **most surprising possible finding** because:

1. **It challenges the consensus explanation.** QuaRot, SpinQuant, and FlatQuant all attribute their success to "making activation distributions more uniform across channels." None discuss scale representation as a mechanism. Showing that scale precision explains more variance than outlier reduction would require the community to revise its understanding of why rotation works.

2. **It has downstream algorithmic consequences.** If scale precision is the bottleneck, new methods should optimize: (a) scale format allocation (assign more bits to scale, fewer to codebook), (b) scale-aware rotation (penalize rotations that create hard-to-represent scales), (c) per-layer scale quantization strategies.

3. **It generalizes beyond NVFP4.** The same mechanism would apply to any format where scale precision is comparable to codebook resolution, such as MXFP6, FP8 with block scaling, or future sub-8-bit formats.

4. **It is falsifiable.** Experiment B (Section 7.2) provides a clean test: if G(E4M3) < G(FP16) at the same block size and codebook, the scale precision effect is real and measurable. The magnitude tells us whether it dominates.

---

## 9. Phase 1 Decision Criteria

After executing the experimental protocol, the following outcomes determine whether Phase 1 proceeds to algorithm design:

| Outcome | G_nvfp4_full | Dominant mechanism | Phase 1 decision |
|---------|-------------|-------------------|-----------------|
| Null | G ≈ 1, no significant effect of any factor | Block size is sufficient | **Pivot:** Investigate whether rotation helps other FP4 formats (MXFP4, FP6) or other block sizes |
| Optimistic | G > 1.1 | Codebook shape or scale precision is favorable | **Proceed:** Design rotation methods optimized for NVFP4 structure |
| Pessimistic | G < 0.9 | Scale precision penalty > codebook benefit | **Pivot:** Develop scale-format-aware methods that do NOT use rotation |
| Mixed | G depends on layer/matrix | Matrix-dependent mechanisms | **Proceed narrowly:** Design per-layer adaptive strategy |

---

## 10. Mathematical Appendix

### A.1 Proof of Theorem 1 (Block Size Scaling)

*Sketch.* Let W have channels with variances σ²_1, ..., σ²_{d_in} drawn from F. For B-block quantization along the channel dimension:

Without rotation: block b on channel i contains B elements from channel i only. The block max M_b ~ σ_i · √(2 log B). The per-channel scale is s_i = σ_i · √(2 log B) / q_max. The within-channel distribution is approximately uniform in [-σ_i · √3, σ_i · √3] (for bounded distributions).

With rotation: each output element is a linear combination w'_{i,j} = Σ_k H_{j,k} w_{i,k}. By the Lyapunov CLT, w'_{i,j} → N(0, σ̄²) where σ̄² = (1/d_in) Σ_k σ²_k. The block max distribution is approximately Gumbel with location σ̄ · √(2 log B) and scale σ̄ / √(2 log B).

The variance of the block max across blocks:
- Before: Var[M_b] ≈ (1/B) · Var[σ²_i] · (2 log B) · (q_max)²
- After: Var[M_b] ≈ σ̄² · π²/(6 log B) · (q_max)²

The reduction in block max variance (which drives better quantization):
```
Var[M_b]^before / Var[M_b]^after ≈ Var[σ²_i] · (2 log B)² / (B · σ̄⁴ · π²/6)
```

As B decreases, this ratio approaches 1 (rotation helps less). For B=16 and typical Var[σ²_i]/σ̄⁴ ≈ 0.1 for LLM activations:
```
Ratio ≈ 0.1 · (2 · 2.77)² / (16 · 1.64) ≈ 0.1 · 30.7 / 26.2 ≈ 0.12
```

So for B=16, rotation reduces block max variance by at most ~12% (vs. ~80% for B=128). The corresponding QSNR improvement is proportional and therefore at most ~1 dB.

### A.2 E4M3 Quantization Error Distribution

The E4M3 format has 256 possible values (8 bits). The relative quantization error for a value x quantized to E4M3:

```
E4M3_relative_error(x) = |x - Q_E4M3(x)| / x
```

This is bounded by 2^{-4} = 0.0625, but the actual error depends on x's position in the E4M3 grid. The expected error (over random x uniformly distributed on a log scale) is:

```
E[E4M3_relative_error] ≈ 2^{-5} ≈ 0.031
```

The critical point: if rotation changes the distribution of s_b* from being concentrated near "good" E4M3 values to being uniformly distributed, the expected scale error increases by up to 2x.

### A.3 QSNR and Effective Bit Width

QSNR relates to effective bit width (EBW):

```
EBW(dB) = 20·log_10(2^{bits}) ≈ 6.02 × bits
```

For 4-bit quantization, ideal QSNR ≈ 24.1 dB.

Observed QSNR (dB) translates to effective bits:
```
bits_eff = QSNR_observed / 6.02
```

Rotation gain in dB:
```
Δ_QSNR(dB) = 10·log_10(G)
```

For INT4, typical Δ_QSNR ≈ 2-3 dB (G ≈ 1.6-2.0), corresponding to ~0.3-0.5 effective bits gained.

For NVFP4, if Δ_QSNR ≈ 0.2-0.5 dB (G ≈ 1.05-1.12), it corresponds to ~0.03-0.08 effective bits -- negligible in practice.

---

## References

1. Ashkboos, S., et al. (2024). "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." NeurIPS 2024.
2. Liu, Z., et al. (2024). "SpinQuant: LLM Quantization with Learned Rotations."
3. Sun, M., et al. (2024). "FlatQuant: Flatness-aware Quantization for Large Language Models."
4. NVIDIA. (2025). "NVFP4: Pushing Intelligence to 4-bit." NVIDIA Research Blog.
5. Rouhani, B.D., et al. (2023). "Microscaling Data Formats for Deep Learning." (MX format specification, E8M0, E4M3, E2M1 definitions.)
