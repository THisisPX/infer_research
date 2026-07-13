# Notes: QSNR Framework Research

## Sources

### NVFP4 / E2M1 Format
- E2M1: 2 exponent bits, 1 mantissa bit, 1 sign bit = 4 bits total
- Exponent bias: 1 (E2M1), range: 2^(-1) to 2^1 = 0.5 to 2.0
- Mantissa: 1 bit means 2 representable significands: 1.0 and 1.5 (binary: 1.0, 1.1)
- Normal values: ±(1 + m/2) × 2^(e - bias) for e > 0
  - e=1: scale 2^0=1, mantissas [1.0, 1.5] → ±1, ±1.5
  - e=2: scale 2^1=2, mantissas [1.0, 1.5] → ±2, ±3
  - e=3: scale 2^2=4, mantissas [1.0, 1.5] → ±4, ±6
- Subnormal (e=0): ±m/2 × 2^(1-bias) = ±0, ±0.5
- Complete codebook: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- Key property: non-uniform spacing — dense near zero, sparse at tails

### NVFP4 Block Structure
- Block size B=16 (1×16 row-wise blocks)
- Per-block scale: E4M3 format (4 exponent, 3 mantissa bits)
  - E4M3 range: ~2^-9 to 2^8, 3 mantissa bits → 8 sub-levels per exponent
- Secondary global scale: FP32 per row
- Scaling chain: W_block ≈ global_scale_fp32 × block_scale_e4m3 × e2m1_codebook_value

### INT4 Block Structure (for comparison)
- Block size B=128 (usually per-channel or per-group)
- Uniform codebook: {-7, -6, ..., -1, 0, 1, ..., 7} or {0, ..., 15} with zero-point
- Per-block scale: FP16
- Key difference: scale is high-precision (FP16) and codebook is uniform

### Rotation Methods
- QuaRot (Ashkboos et al., 2024): Hadamard rotation applied to activations and weights
- SpinQuant (Liu et al., 2024): Learned rotation matrix (Cayley-parameterized)
- FlatQuant (Sun et al., 2024): Optimized rotation + per-channel smoothing
- Common structure: Y = XW = (XH)(H^T W), where H is orthogonal

## Synthesized Findings

### Key Quantitative Differences
| Property | INT4 | NVFP4 |
|----------|------|-------|
| Block size B | 128 | 16 |
| Scale precision | FP16 (11-bit mantissa) | E4M3 (3-bit mantissa) |
| Codebook | Uniform, 7 bits effective | Non-uniform, dense near zero |
| Scale error σ_s | ~2^-11 relative | ~2^-3 relative |
| Codebook resolution Δ | 1/7 × range | Variable (0.5 at center, 2 at tail) |

### Critical Insight: Scale Error Matters in NVFP4
In INT4, the FP16 scale contributes negligible error (~2^-11) relative to the 3-bit quantization error (~2^-3). But in NVFP4, the E4M3 scale also has only ~3-bit precision, so scale error and codebook error are of comparable magnitude.

This is the core of Claim 2: rotation changes the distribution of values within each block, changing which E4M3 scale value is chosen, potentially degrading precision.
