# Direction C: Weight-Activation Joint Quantization for LLMs -- Literature Scan (2024-2026)

**Date:** 2026-07-09
**Scope:** Systematic literature scan across 30+ papers, 4 sub-questions
**Hardware Context:** 4x B300 (FP4 native) + 8x A100 (INT8/FP16)

---

## Table of Contents

1. [C1: W4A4/W4A8/W8A8 Joint Quantization SOTA](#c1-w4a4w4a8w8a8-joint-quantization-sota)
2. [C2: Optimal Bit-width Combinations](#c2-optimal-bit-width-combinations)
3. [C3: Activation Quantization Challenges](#c3-activation-quantization-challenges)
4. [C4: MOE Model Joint Quantization](#c4-moe-model-joint-quantization)
5. [Cross-Cutting Gap Analysis](#cross-cutting-gap-analysis)

---

## C1: W4A4/W4A8/W8A8 Joint Quantization SOTA

### C1.1 Key Papers (8 papers)

| # | Title | Authors | Venue/Year | arXiv ID | One-Sentence Contribution | Best Result |
|---|-------|---------|------------|----------|---------------------------|-------------|
| 1 | **FlatQuant: Flatness Matters for LLM Quantization** | Sun, Liu, Bai et al. (Huawei/Tsinghua) | ICML 2025 | 2410.09426 | Learnable per-layer affine transforms with Kronecker decomposition to flatten weight/activation distributions | LLaMA-3-70B W4A4: 79.01% zero-shot (-0.94% vs FP16); surpasses SpinQuant by 7.5% |
| 2 | **SpinQuant: LLM Quantization with Learned Rotations** | Liu, Zhao, Fedorov et al. (Meta FAIR) | ICLR 2025 | 2405.16406 | Cayley-optimized learned rotation matrices replacing QuaRot's random Hadamard rotations | LLaMA-3 8B W4A4KV4: 45.1% gap reduction vs QuaRot; beats LLM-QAT by 19.1 pts |
| 3 | **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** | Ashkboos et al. (ETH/Microsoft/ISTA) | NeurIPS 2024 | 2404.00456 | First end-to-end W4A4KV4 via random Hadamard rotations; no higher-precision fallback channels | LLaMA-2 70B W4A4KV4: 0.47 PPL loss, 99% zero-shot retained |
| 4 | **PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers** | Chen, Liu, Wang et al. | arXiv, Oct 2024 | 2410.05265 | First static per-tensor quantization outperforming dynamic per-token via outlier token prefixing in KV cache | LLaMA-3 8B W4A4KV4: PPL 7.43 (vs QuaRot 8.41); +5.98 reasoning accuracy |
| 5 | **CBQ: Cross-Block Quantization for LLMs** | Ding, Liu, Tu et al. (USTC/Huawei) | ICLR 2025 Spotlight | 2312.07950 | Cross-block reconstruction with LoRA-Rounding captures inter-block dependencies | >99% perf retention across OPT/LLaMA; W4A4/W4A8/W6A6; 4.3h for LLaMA-65B |
| 6 | **MergeQuant: Accurate 4-bit Static Quantization by Channel-wise Calibration** | Wang et al. (BUPT/PengCheng) | arXiv, Mar 2025 | 2503.07654 | Per-channel static quantization with Quantization Step Migration (QSM) fusing into normalization | LLaMA-2 70B W4A4: -1.3 pts zero-shot; 2.06x speedup |
| 7 | **LO-BCQ: Block Clustered Quantization for 4-bit (W4A4) LLM Inference** | Elangovan et al. (NVIDIA/Purdue) | TMLR 2025 | 2502.05376 | Block decomposition + clustering + per-cluster Lloyd-Max codebooks | <0.1 PPL loss on LLaMA-2 7B/70B; <1% MMLU loss; no fine-tuning |
| 8 | **Atom: Low-bit Quantization for Efficient and Accurate LLM Serving** | Zhao et al. | MLSys 2024 | 2310.19102 | W4A4 via mixed-precision (outlier channels INT8, rest INT4) + custom fused CUDA kernels | LLaMA-2 70B W4A4: PPL 3.68 (vs FP16 3.32); 7.7x throughput |

### C1.2 SOTA Landscape

#### Comparison Table: Methods by Bit-Width

| Method | Type | W-bits | A-bits | KV-bits | Best Model Tested | Accuracy Loss | Key Technique |
|--------|------|--------|--------|---------|-------------------|---------------|---------------|
| **FlatQuant** | PTQ | 4 | 4 | -- | LLaMA-3-70B | -0.94% zero-shot | Learnable affine + Kronecker |
| **SpinQuant** | PTQ | 4 | 4 | 4 | LLaMA-2 7B | -2.9 pts zero-shot | Learned Cayley rotations |
| **QuaRot** | PTQ | 4 | 4 | 4 | LLaMA-2 70B | PPL +0.47 | Random Hadamard rotations |
| **PrefixQuant** | PTQ | 4 | 4 | 4 | LLaMA-3 8B | PPL 7.43 | Static outlier prefixing |
| **CBQ** | PTQ | 4 | 4 | -- | LLaMA-65B | <1% loss | Cross-block + LoRA-Rounding |
| **MergeQuant** | PTQ | 4 | 4 | -- | LLaMA-2 70B | -1.3 pts zero-shot | QSM static quantization |
| **LO-BCQ** | PTQ | 4 | 4 | -- | LLaMA-2 70B | <0.1 PPL | Block clustering + codebooks |
| **SmoothQuant** | PTQ | 8 | 8 | -- | LLaMA-2 70B | -0.04 PPL | Channel-wise smoothing |
| **QServe/QoQ** | PTQ | 4 | 8 | 4 | LLaMA-3 8B | Near-FP16 | Progressive group quant |
| **AWQ** | PTQ | 4 | 16 | -- | LLaMA-3.1 70B | >99% MT-Bench | Activation-aware scaling |
| **QuEST** | QAT | 4 | 4 | -- | 800M-param | Pareto-optimal at 4-bit | Hadamard + trust gradient |
| **LLM-QAT** | QAT | 4 | 4 | -- | LLaMA-7B | Competitive w/ PTQ | Full QAT + data distillation |
| **Metis** | QAT | 4 | 4 | -- | LLaMA-3 8B | 0.1% degradation | FP4 training (W4A4G4) |

#### W4A4 Methods Achieving >95% FP16 Accuracy

Multiple methods achieve this on 70B-class models:
1. **FlatQuant**: LLaMA-3-70B retains 98.8% of FP16 zero-shot accuracy
2. **QuaRot**: LLaMA-2-70B retains 99% of FP16 zero-shot accuracy
3. **CBQ**: >99% performance retention across LLaMA family
4. **LO-BCQ**: <1% loss on downstream tasks (>99% retained)
5. **MergeQuant**: LLaMA-2-70B gap of 1.3 points (~98.3% retained)

On 7B-class models, the gap is larger (2-5% loss) even for the best methods. Larger models quantize better -- a consistent finding.

#### PTQ vs QAT at Each Bit-Width

| Bit-Width | PTQ Capability | QAT Capability | Verdict |
|-----------|---------------|----------------|---------|
| **W8A8** | Near-lossless, ~99% FP16 | Unnecessary | PTQ dominates |
| **W4A8** | PTQ dominates (QServe, PrefixQuant) | Cost rarely justified | PTQ preferred |
| **W4A4** | <1% loss on 70B+, 2-3% on 7B | QAT wins by 1-2 pts on small models | PTQ for large, QAT for small |
| **W2A4 / sub-3-bit** | PTQ collapses | QAT essential | QAT only |

### C1.3 Bottlenecks and Limitations

1. **FC2/down_proj Layer Dominates Error**: The SwiGLU FFN's down-projection input is a bilinear product (`silu(w1) * w3`) that amplifies channel-wise activation outliers. Kurtosis of w2 input: ~1921 vs ~2.85 for other layers. Multiple 2025 papers independently identify this as the dominant residual bottleneck.

2. **Small Models Quantize Worse**: Scaling exponent for quantization error: gamma_N = -0.2186 (error decreases with model size). W4A4 on 7B: 2-5% loss; on 70B: <1% loss.

3. **Architecture Sensitivity**: LLaMA-3 family (especially 8B) is harder to quantize than LLaMA-2. LLaMA-3-70B uniquely suffers under per-channel W8A8 (accuracy drops to 45.5% without mixed-granularity fix).

4. **Weight Error Overtakes Activation Error at Scale**: At high data-to-parameter ratios, weight quantization error surpasses activation error (scaling exponent gamma_D = +0.0745).

5. **Reasoning Degradation**: Math reasoning (AIME, MATH) can degrade up to 69.81% for extreme quantization. W4A4 on reasoning models shows larger gaps than on chat models.

6. **Production Kernel Gap**: Only SmoothQuant W8A8 and AWQ/GPTQ W4A16 have mature kernel support in TensorRT-LLM and vLLM. All W4A4 methods require custom kernels not yet upstream.

### C1.4 Hardware Match

| Method | B300 (FP4 Native) | A100 (INT8/FP16) |
|--------|-------------------|-------------------|
| **NVFP4 (NVIDIA native)** | Full native support. E2M1 format, 2-level microblock scaling. ~2x throughput vs BF16. TensorRT-LLM + vLLM support. | N/A (requires Blackwell) |
| **SmoothQuant W8A8** | Works via FP8 path | **Best W8A8 path**: Near-lossless, 1.47x speedup, native TensorRT-LLM |
| **QServe W4A8KV4** | Works with adaptation | **Best W4A8 path**: 1.2-2.4x vs TensorRT-LLM, keeps GEMM on INT8 Tensor Cores |
| **AWQ/GPTQ W4A16** | Memory benefit only | Mature ecosystem; memory-bound speedup only |
| **QuaRot/SpinQuant W4A4KV4** | Requires adaptation to NVFP4 block format | Custom FHT kernels needed; no framework support |
| **MergeQuant W4A4 static** | Compatible via NVFP4 | No dynamic quantization overhead; INT4 GEMM acceleration |

**Critical Note**: A100 W4A4 suffers from high rho ratio (Tensor Core:CUDA Core = 64:1), meaning pure W4A4 kernels incur significant CUDA core dequantization overhead. W4A8 (QServe style) keeping all computation on INT8 Tensor Cores is often faster in practice.

---

## C2: Optimal Bit-width Combinations

### C2.1 Key Papers (7 papers)

| # | Title | Authors | Venue/Year | arXiv ID | Key Finding |
|---|-------|---------|------------|----------|-------------|
| 1 | **Scaling Laws for Precision** | Kumar, Ankner et al. | arXiv 2024 | 2411.04330 | Compute-optimal training precision is ~7-8 bits; validated on 465+ runs up to 1.7B params |
| 2 | **Low-Bit Quantization Favors Undertrained LLMs** | Ouyang et al. | ACL 2025 | 2411.17691 | QiD scaling law: error decreases with model size (~1/N^0.23), increases with training tokens (~D^0.53). Future 100T-token models may degrade severely |
| 3 | **QuaRot** | Ashkboos et al. | NeurIPS 2024 | 2404.00456 | W4A4KV4 with <0.47 PPL loss on LLaMA-2 70B; first true end-to-end 4-bit |
| 4 | **SpinQuant** | Liu et al. (Meta) | ICLR 2025 | 2405.16406 | W4A4KV4 gap to FP16: 2.9 pts on LLaMA-2 7B; 30.2% gap reduction vs QuaRot |
| 5 | **ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization** | Liu et al. (Meta) | NeurIPS 2025 | 2502.02631 | Sub-4-bit (ternary, 2-bit, 3-bit) outperforms 4-bit on the Pareto frontier. Learning transition at 2-3 bits |
| 6 | **Task-Circuit Quantization (TaCQ)** | Xiao, Sung et al. | COLM 2025 | 2504.07389 | Task-specific weight circuits for mixed-precision; at 3.1 bits recovers 96% of MMLU |
| 7 | **Quantitative Analysis of DeepSeek Model Quantization** | China Unicom AI | arXiv 2025 | 2505.02390 | Q4_K_M: 0% avg drop vs FP8; Q2_K_L: 8.91% drop; math collapses (AIME 39.2 to 15.41) |

### C2.2 Degradation Ladder: W16A16 to W4A4

| Configuration | Throughput vs FP16 | Memory vs FP16 | PPL Increase (7B) | PPL Increase (70B) | Zero-shot Drop | Status |
|---------------|-------------------|----------------|-------------------|--------------------|-------------------|--------|
| **W16A16** | 1.0x | 1.0x | 0.00 | 0.00 | 0% | Baseline |
| **W8A16** (GPTQ/AWQ) | ~1.1-1.4x | ~1.8-1.9x | 0.02-0.05 | 0.01-0.03 | <1% | Production-ready |
| **W8A8** (SmoothQuant) | ~1.3-1.56x | ~2.0x | 0.05-0.15 | 0.03-0.10 | 1-3% | Mostly safe |
| **W4A16** (AWQ/GPTQ) | ~1.4-1.8x | ~3.5-3.8x | 0.3-1.0 | 0.1-0.5 | 2-5% | Viable |
| **W4A8** | ~2.0x | ~3.8x+ | 0.8-2.5 | 0.4-1.2 | 5-10% | NOT lossless, no PTQ matches FP16 |
| **W4A4** (QuaRot/SpinQuant) | ~2.5-3.4x | ~4.0x | 0.5-1.5 | 0.29-0.47 | 1-5% | Emerging; rotation-based methods viable |

### C2.3 Catastrophic Degradation Thresholds

Three independent findings converge:
- **Emergent abilities** (Liu et al., LREC 2024): Well-preserved at 4-bit, **2-bit causes catastrophic collapse** (GSM8K near 0%)
- **General LM quality** (Yao et al., AAAI 2024): PTQ degradation accelerates sharply **below W4A8**
- **DeepSeek-V3** (Unicom, 2025): Q4 (4.82 bits): 0% drop; Q3_K_M: 0.52% drop; **Q2_K_L (2.91 bits): 8.91% drop**; AIME math: 39.2 to 15.41

**Practical safe floor**: W4A4 via rotation-based methods (QuaRot/SpinQuant) or W4A16 weight-only for most use cases.

### C2.4 Scaling with Model Size

| Model Scale | Quantization Robustness | Key Evidence |
|-------------|------------------------|--------------|
| **7-8B** | Moderate | LLaMA-3 8B W4A4 gap ~2.9 pts (SpinQuant); smaller models more sensitive |
| **70B** | High (with LLaMA-3 exception) | LLaMA-2 70B W4A4KV4: 0.29-0.47 PPL loss, 99% retained. LLaMA-3 70B: UNIQUE vulnerability |
| **405B** | Very high | Behaves like 8B models in robustness; W8A8 safe |
| **671B MoE (DeepSeek-V3)** | High down to ~3.5 bits | Q4_K_M: ~0% drop; Dynamic Q3: ~0% drop; Q2: 8.9% drop |

### C2.5 Task-Dependent Optimal Configurations

| Task Type | Quantization Sensitivity | Optimal W-A Config | Notes |
|-----------|------------------------|--------------------|-------|
| **Knowledge Recall** (MMLU, TriviaQA) | LOW | W4A16 or W4A4 (rotated) | Most robust; well-preserved at 4-bit |
| **Language Generation** (WikiText, C4) | LOW-MODERATE | W4A4 (rotated) or W8A8 | PPL degrades gracefully |
| **Math Reasoning** (GSM8K, MATH) | HIGH | W8A8 minimum | Multi-step chains amplify per-step quant error |
| **Code Generation** (HumanEval, MBPP) | HIGHEST | W8A8 minimum | Activation quantization especially harmful |
| **Long-Context** (>128K) | HIGH (KV dominates) | W4A4 + KV4 (rotated) or W8A8 + KV~3.25 | KV cache becomes dominant memory consumer |

### C2.6 The Pareto Frontier

```
Quality Loss (% from FP16)
     ^
     |  Binary (1b)     -- ParetoQ: ~6.8pt gap
     |       \
     |        Ternary    -- ParetoQ: ~5.6pt gap
     |              \
     |               2-bit -- ParetoQ/AQLM: ~3.4pt gap
     |                     \
     |                      3-bit -- AQLM/KVTuner: ~1-2pt gap
     |                            \
     |  * SWEET SPOT -- 4-bit ---- QuaRot/SpinQuant: <1pt gap
     |                                   \
     |  8-bit ----------------------------- Near-lossless
     +----------------------------------------------> Bits/Param
```

**Sweet spots by scenario**:
- Cloud API serving (latency-sensitive): **W8A8** or FP8
- Cloud batch inference (throughput): **W4A16** weight-only
- Single-GPU 70B deployment: **W4A16** or **W4A4** (rotated)
- On-device/edge: **2-bit** (ParetoQ)
- Long-context (>128K): **W4A4 + KV4** (SpinQuant) or **W8A8 + KV~3.25** (KVTuner)

### C2.7 Hardware Match

**B300 NVFP4**: Natural match for W4A4 (E2M1 format on native FP4 tensor cores). Performance: DeepSeek-R1 671B on DGX B200 >3x vs H200 FP8, >30,000 tok/s. Accuracy: <1% degradation.

**A100 INT8**: For INT8 tensor core utilization, use **W8A8 via SmoothQuant**. For pure memory savings, **AWQ-4bit (W4A16) with vLLM** is the most deployed configuration.

**Critical caveat**: Native FP4 tensor core execution only works on datacenter Blackwell (B200/B300, SM 10.x). Consumer Blackwell (RTX 5090, SM 12.0+) falls back to BF16 dequant kernels with no FP4 compute advantage.

---

## C3: Activation Quantization Challenges

### C3.1 Key Papers (7 papers)

| # | Title | Authors | Venue/Year | arXiv ID | Core Contribution |
|---|-------|---------|------------|----------|-------------------|
| 1 | **SmoothQuant** | Xiao, Lin, Seznec et al. (MIT) | ICML 2023 | 2211.10438 | Per-channel scaling migrates quantization difficulty from activations to weights; W8A8 with negligible loss |
| 2 | **QuaRot** | Ashkboos et al. (ETH/Microsoft/ISTA) | NeurIPS 2024 | 2404.00456 | First true end-to-end W4A4KV4 via randomized Hadamard rotations |
| 3 | **SpinQuant** | Liu, Zhao et al. (Meta) | ICLR 2025 | 2405.16406 | Learned rotation matrices via Cayley SGD; 2.9 pt gap to FP16 on W4A4KV4 LLaMA2-7B |
| 4 | **DuQuant: Distributing Outliers via Dual Transformation** | Lin, Xu, Wu et al. (UCAS/Tsinghua) | NeurIPS 2024 Oral | 2406.01721 | Dual transformation (rotation + zigzag permutation) targeting both Normal and Massive outliers |
| 5 | **KVQuant: Towards 10M Context Length LLM Inference** | Hooper et al. (Berkeley) | NeurIPS 2024 | 2401.18079 | 3-bit KV cache via per-channel Key quant, Pre-RoPE quant, non-uniform datatypes |
| 6 | **FlatQuant** | Sun, Liu et al. (Huawei/Tsinghua) | ICML 2025 | 2410.09426 | Learnable Kronecker-factored affine transforms; <1% accuracy drop on LLaMA3-70B W4A4 |
| 7 | **Outlier-Safe Pre-Training (OSP)** | Park et al. (Korea Univ.) | ACL 2025 | 2506.19697 | Pre-training prevention: Muon optimizer + Single-Scale RMSNorm eliminates outliers (kurtosis 0.04 vs 1818.56) |

### C3.2 Why Activation Quantization Is Harder

| Aspect | Weights | Activations |
|--------|---------|-------------|
| Distribution shape | Near-Gaussian, smooth | Long-tailed, extreme outliers (100-1000x normal) |
| Outlier presence | Rare, small magnitude | Systematic: persistent across specific channels |
| Dynamic range | Static (known after training) | Input-dependent: varies per token, per sequence |
| Inter/intra-channel variance | Low inter-channel variance | High inter-channel, low intra-channel variance |

**Outlier Channel Phenomenon**: Outliers appear in fixed, persistent input channels across ALL tokens. They emerge early in training, concentrate in residual stream layers, and affect <1% of channels. Two types identified:
1. **Channel-wise (Normal) Outliers**: Large-magnitude values persistent across all tokens in specific input channels
2. **Spike (Massive) Outliers**: ~1400x median values occurring in only a few tokens, first discovered in FFN down_proj layers

**Root cause**: Diagonal preconditioners (Adam) and channel-wise normalization (RMSNorm) amplify privileged bases. OSP (ACL 2025) proved that replacing Adam with Muon + single-scale RMSNorm eliminates outliers entirely.

### C3.3 Outlier Mitigation Methods

**Category 1 - SmoothQuant-Style Magnitude Migration**: Apply per-channel scaling s_j to divide activations, multiply weights. Mathematically equivalent. Limitation: makes weight distributions steeper, problematic at sub-4-bit.

**Category 2 - Rotation-Based Methods (Dominant 2024-2025 Paradigm)**:

| Method | Rotation Type | Optimization | Key Result |
|--------|--------------|--------------|------------|
| QuaRot | Random Hadamard | None | W4A4KV4, 0.47 PPL loss (70B) |
| SpinQuant | Learned Hadamard | Cayley SGD (~1.3h) | 2.9 pt gap to FP16 (7B) |
| DuQuant | Block-diagonal + zigzag | Greedy, outlier-dim-guided | SOTA W4A4, NeurIPS Oral |
| FlatQuant | Kronecker-factored affine | Per-layer, lightweight | <1% loss (70B) |
| DFRot | Refined Hadamard | Alternating optimization | +0.98 PPL over QuaRot |

**Why rotations work**: Random orthogonal transforms increase "incoherence" -- spreading outlier magnitudes across dimensions. Hadamard is preferred because: O(n log n) via FWHT, entries are +-1, and DFRot proved they reduce error on rare "massive activation" tokens.

**Category 3 - Pre-Training Prevention (Emerging)**: OSP (ACL 2025) uses Muon optimizer + Single-Scale RMSNorm to train models that never develop outliers. This is a fundamental alternative to post-hoc mitigation but requires training from scratch.

### C3.4 Quantization Granularity Tradeoffs

| Granularity | Overhead | Accuracy | GPU Compatibility |
|-------------|----------|----------|-------------------|
| **Per-Tensor** | Minimal | Fails with outliers | Standard GEMM path |
| **Per-Channel** | C FP32 values | Best for weights | **Breaks INT8 GEMM** (scale cannot be factored from sum) |
| **Per-Token** | T FP32 values + runtime reduction | **Standard for activations** | Extra reduction kernel; 5-10% latency overhead |
| **Per-Group** (g=128) | (N/g) FP32 values | **Best for low-bit** | Complex GEMM kernels needed (CUTLASS, custom) |

**For activations specifically**: Per-token dynamic quantization is the de facto standard. Per-channel is ideal for addressing systematic outlier channels but **breaks INT8 GEMM kernel compatibility**. Per-group provides highest accuracy for extreme low-bit (W4A4) but requires custom kernels.

### C3.5 Small Batch Inference Challenge

**Why small batch degrades activation quantization**:
1. Stronger outlier contamination (single outlier token has proportionally larger impact)
2. Dynamic range miscalibration (actual ranges deviate more from calibration mean)
3. Memory-bandwidth bound (dynamic scaling overhead proportionally worse)
4. Sequence-length effects (early tokens have noisier statistics)

**Solutions**: Online/test-time calibration (TTQ, 2025), Sequence-Length-Aware Calibration (SLAC), Prefix-based stabilization (CushionCache, EMNLP 2024), Pre-training prevention (OSP, ACL 2025).

### C3.6 Hardware Match

**B300 (NVFP4)**: B300's NVFP4 inherently provides per-group scaling via block-level FP8 scales, aligning naturally with the granularity trends. Rotation-based methods remain valuable for outlier suppression even with NVFP4's native scaling. **NVFP4 is the primary inference format on Blackwell**, not INT8.

**A100 (INT8)**: INT8 Tensor Cores (mma.m16n8k32) support per-channel weights + per-tensor activations. FP8 is NOT supported (H100+ feature only). At small batch sizes (M=16), kernel launch overhead can make INT8 GEMM slower than FP16.

---

## C4: MOE Model Joint Quantization

### C4.1 Key Papers (8 papers)

| # | Title | Authors | Venue/Year | arXiv ID | Contribution | MoE Model Tested |
|---|-------|---------|------------|----------|--------------|------------------|
| 1 | **DeepSeek-V2 Technical Report** | DeepSeek-AI | arXiv, May 2024 | 2405.04434 | MLA + DeepSeekMoE architecture; INT4/INT8 inference serving | DeepSeek-V2 (236B/21B active) |
| 2 | **DeepSeek-V3 Technical Report** | DeepSeek-AI | arXiv, Dec 2024 | 2412.19437 | First 671B MoE FP8 mixed-precision training; 1x128 activation scaling, 128x128 weight scaling | DeepSeek-V3 (671B/37B active) |
| 3 | **QuantMoE-Bench** | Li, Jin, Cheng, Chen | arXiv, Jun 2024 | 2406.08155 | First systematic MoE PTQ benchmark; data-driven mixed-precision allocation per component type | Mixtral-8x7B, DeepSeek-MoE-16B |
| 4 | **MC-MoE: Mixture Compressor for MoE LLMs** | He et al. | ICLR 2025 | 2410.06270 | Training-free compression: ILP-based mixed-precision + online dynamic pruning; 2.54-bit average | Mixtral-8x7B, Mixtral-8x22B |
| 5 | **EAC-MoE: Expert-Selection Aware Compressor** | Multiple authors | ACL 2025 | (ACL 2025 long) | Identifies "expert shift" problem; TopK-MSE router calibration + frequency-based expert pruning | Mixtral-8x7B, DeepSeek-MoE-16B, Qwen1.5-MoE |
| 6 | **MoEQuant** | Hu et al. | arXiv, May 2025 | 2505.03804 | Expert-Balanced Self-Sampling + Affinity-Guided Quantization for inter/intra-expert imbalance | DeepSeekMoE-16B, Mixtral-8x7B |
| 7 | **A Survey on MoE Inference Optimization** | Liu et al. | arXiv, Dec 2024 | 2412.14219 | Comprehensive MoE inference survey covering model-level, system-level, hardware-level techniques | Survey across Mixtral, DeepSeek, Qwen |
| 8 | **MoE-I^2: Inter-Expert Pruning + Intra-Expert Low-Rank Decomposition** | Multiple authors | arXiv, Nov 2024 | 2411.01016 | Genetic search-based non-uniform expert pruning + low-rank decomposition; joint pruning+quantization | Qwen1.5-MoE, DeepSeek-V2-Lite, Mixtral-8x7B |

### C4.2 How MOE Changes the Quantization Landscape

**Three structural differences from dense models**:

1. **Parameter distribution is uneven**: Router (<0.03% of params) controls which FFN experts participate. Router quantization error propagates fundamentally differently from expert weight error.

2. **Total vs activated parameters**: DeepSeek-V3 has 18x more total params than activated. Weight quantization primarily reduces memory capacity (fitting model in GPU); activation quantization accelerates per-token computation.

3. **Three MoE component types with different sensitivity** (from QuantMoE-Bench):

| Parameter Type | Recommended Bit-Width | Sensitivity Rationale |
|---------------|----------------------|----------------------|
| Attention layers (MHSA/QKV) | 4-bit | Weight magnitude outliers are more prominent in attention |
| Shared experts (always active) | 4-bit | Affects all tokens; error accumulates per layer |
| Routed active experts (selective) | 2-bit | Affects only a fraction of tokens; error is localized |
| Early MoE layers (blocks 1-3) | 4-bit | Error propagates and amplifies through depth |
| Late MoE layers (deep blocks) | 2-bit | Less impact on model output |

**Sparsity: helpful and harmful**:
- **Helpful**: Error isolation -- each token only sees k experts, so expert quantization error is contained
- **Harmful**: Quantization distorts router logits, causing **expert shift** -- choosing different experts than FP16 model

EAC-MoE showed that router distortion alone (with perfect expert weights) raised Mixtral-8x7B PPL from 3.84 to 4.17.

### C4.3 Expert Heterogeneity

**Evidence for heterogeneous expert sensitivity**:
1. **Activation amplitude heterogeneity** (ExpertQuant): Different experts produce different activation ranges, requiring per-expert per-channel scaling
2. **Varying activation frequency alone is insufficient** for allocation (QuantMoE-Bench)
3. **Hessian trace is more reliable** than frequency (MoPEQ, ICCV 2025 workshop)
4. **Counterintuitive finding** (2604.06515): Experts learning low-frequency tokens, which had smaller router norm changes during training, are paradoxically **more** sensitive to quantization

**Three bit-width allocation paradigms**:

| Method | Metric | Granularity | Key Result |
|--------|--------|-------------|------------|
| MC-MoE PMQ | Access frequency + reconstruction error | Per-expert | 2.54-bit avg, 76.6% params, 3.8% loss |
| MoPEQ | Hessian trace x activation frequency | Per-expert, per-projection | 1.5x size reduction, <5% accuracy drop |
| MxMoE | ILP-based block-level | Per linear block (Gate/Down/Up) | Importance varies within same expert |

### C4.4 Router Quantization Sensitivity

**Router is the single most sensitive MoE component.** Evidence:
- EAC-MoE: At 2.54-bit average (with MHSA at 4-bit, router FP16): ~3% loss. At 3.03-bit: <0.5% loss.
- ExpertQuant: "Router accuracy is the dominant factor in MoE quantization"
- QuantMoE-Bench: MHSA below 4-bit causes significant expert selection changes, large PPL loss

**Router degradation by bit-width** (Mixtral-8x7B):
| MHSA/Attention Bits | Expert Shift Rate | PPL Impact |
|---------------------|-------------------|------------|
| FP16 | 0% (reference) | 3.84 (baseline) |
| 4-bit | Low | Minimal |
| 3-bit | Moderate | Noticeable |
| 2-bit | High | Severe (3.84 to 4.17, weights perfect) |

**Router-specific strategies**: TopK-MSE calibration (EAC-MoE), Rank-Aware Jaccard Loss + Gap Hinge Loss (ExpertQuant), Value-Structure Alignment (VSRAQ, 2606.05688), Post-quantization router fine-tuning (GEMQ, 2605.23078).

### C4.5 MOE-Specific Quantization Methods (Not Naive Dense Adaptation)

1. **MC-MoE PMQ** (ICLR 2025): ILP per-expert bit allocation using access frequency + reconstruction error + routing scores. Combined with online dynamic pruning.

2. **QuantMoE-Bench** (2024): Multi-granularity allocation (MoE block, expert, linear layer). Outlier-aware linear layer scorer + MoE block importance predictor.

3. **MoEQuant EBSS** (2025): Expert-Balanced Self-Sampling fixes calibration imbalance (high-frequency experts dominate calibration). Affinity-Guided Quantization handles sample-expert affinity variation.

4. **PuzzleMoE** (ICML 2026, 2511.04805): Sparse expert merging via similarity + saliency masks, combined with 3-bit group quantization. 4.8x total compression, ~1.7% loss.

5. **DeepSeek-V3 FP8 Training** (production-scale): 2.788M H800 GPU hours. Mixed precision: embeddings/attention/norm in BF16, GEMM (FFN/experts) in FP8. 1x128 tile-level activation scaling, 128x128 block-level weight scaling. Open-source DeepGEMM + DeepEP libraries.

### C4.6 Hardware Match

**B300 NVFP4 for MoE**:
- Native FP4 tensor cores accelerate routed FFN expert matmul (~2x vs BF16)
- Mixed quantization scheme proven: NVFP4 for expert FFN, FP8 for attention, BF16 for shared layers
- DeepSeek-V4-Flash NVFP4 benchmark (4x B300, TP=4, 172GB): AIME 2024 pass@1 96.00% (vs BF16 96.15%, within noise), 2.95x wall-clock speedup
- Expert-level compute scheduling: fusion of gating + top-k selection + quantization + reorder into 1-2 kernels

**A100 for MoE**:
- 80GB HBM2e cannot fit unquantized Mixtral-8x7B (~90GB+). INT4 quantization reduces to ~23GB.
- Decode is memory-bandwidth bound; INT4/FP4 weight quantization proportionally reduces latency
- **MLA caveat**: MLA reconstruction weights (~16MB each, fit in L2 cache) benefit disproportionately less from INT4; FP8/FP16 may be better for MLA layers
- Expert offloading + quantization synergy (DAOP, MoE-Infinity, HOBBIT, KTRANSFORMERS)

**Consumer Blackwell (RTX 5090) caveat**: No native FP4 compute acceleration -- falls back to Marlin BF16 dequant kernels with no throughput advantage, only memory savings.

---

## Cross-Cutting Gap Analysis

### Gap 1: W4A4 Production-Ready Kernels Are Missing
**Evidence**: Only SmoothQuant W8A8 and AWQ/GPTQ W4A16 are in TensorRT-LLM/vLLM. No W4A4 method (QuaRot, SpinQuant, FlatQuant, PrefixQuant) has upstream kernel support. The Hadamard/FHT kernel requirement is the blocker.
**Hardware match**: B300 could accelerate NVFP4-W4A4 but needs custom kernel work.
**2-4 month feasibility**: High for a specific model family on B300.

### Gap 2: Small-Batch Activation Quantization Degradation
**Evidence**: Static calibration methods all degrade at batch=1. Dynamic calibration adds overhead that is proportionally worse at small batch. No method simultaneously achieves high accuracy AND low overhead at batch=1.
**Hardware match**: Relevant to A100 interactive serving. B300's NVFP4 block scaling may help.
**2-4 month feasibility**: Medium-high. A test-time calibration approach for B300 is novel.

### Gap 3: MOE-Specific Joint W-A Quantization Is Under-Explored
**Evidence**: Only QuantMoE-Bench and DeepSeek-V3's FP8 training address W-A jointly for MoE. Most MOE quantization papers focus on weight-only or are training-free. No rotation-based method has been specifically adapted for MoE architectures (expert-specific rotations, router-aware calibration).
**Hardware match**: Directly matches B300 NVFP4 for expert computation + A100 for attention/shared layers.
**2-4 month feasibility**: High. This is the largest open gap with strongest hardware alignment.

### Gap 4: Task-Adaptive Bit-Width for MoE Experts
**Evidence**: Multiple papers confirm expert heterogeneity but no work combines task-adaptive bit-width allocation with MOE. TaCQ (task-circuit quantization) exists for dense models but has no MoE extension.
**Hardware match**: Mixed precision across experts maps naturally to B300 + A100 hybrid deployment.
**2-4 month feasibility**: Medium. Requires developing both a method and evaluation.

### Gap 5: Router Quantization Below 4-Bit
**Evidence**: The router is the single most sensitive MoE component. No method achieves effective router quantization below 4-bit without significant expert shift. Router-specific quantization (TopK-MSE, Rank-Aware Jaccard) exists but is early-stage.
**Hardware match**: Router is tiny (<0.03% params) so this matters more for end-to-end W4A4 than for memory.
**2-4 month feasibility**: Medium-low. May require fundamentally new approaches.

### Direction C Priority Ranking

| Priority | Gap | Rationale |
|----------|-----|-----------|
| **P0** | Gap 3: MOE-specific W-A joint quantization | Largest open gap, strongest hardware alignment (B300+A100), 2-4 month feasible, no strong competitor yet |
| **P1** | Gap 2: Small-batch activation quantization | Practical importance for interactive serving, B300 NVFP4 may naturally help |
| **P2** | Gap 1: Production kernel work | Infrastructure-heavy, less novelty |
| **P3** | Gap 4: Task-adaptive MoE bit allocation | Requires method + evaluation development, higher risk |
| **P4** | Gap 5: Router quantization below 4-bit | Fundamental difficulty, may require new approaches |

---

## References (Consolidated)

1. SmoothQuant (Xiao et al., ICML 2023) -- arXiv:2211.10438
2. QuaRot (Ashkboos et al., NeurIPS 2024) -- arXiv:2404.00456
3. SpinQuant (Liu et al., ICLR 2025) -- arXiv:2405.16406
4. FlatQuant (Sun et al., ICML 2025) -- arXiv:2410.09426
5. DuQuant (Lin et al., NeurIPS 2024 Oral) -- arXiv:2406.01721
6. PrefixQuant (Chen et al., arXiv 2024) -- arXiv:2410.05265
7. CBQ (Ding et al., ICLR 2025 Spotlight) -- arXiv:2312.07950
8. MergeQuant (Wang et al., arXiv 2025) -- arXiv:2503.07654
9. LO-BCQ (Elangovan et al., TMLR 2025) -- arXiv:2502.05376
10. Atom (Zhao et al., MLSys 2024) -- arXiv:2310.19102
11. Scaling Laws for Precision (Kumar et al., arXiv 2024) -- arXiv:2411.04330
12. Low-Bit Quantization Favors Undertrained LLMs (Ouyang et al., ACL 2025) -- arXiv:2411.17691
13. ParetoQ (Liu et al., NeurIPS 2025) -- arXiv:2502.02631
14. TaCQ (Xiao et al., COLM 2025) -- arXiv:2504.07389
15. DeepSeek Quantization Analysis (China Unicom, 2025) -- arXiv:2505.02390
16. KVQuant (Hooper et al., NeurIPS 2024) -- arXiv:2401.18079
17. OSP (Park et al., ACL 2025) -- arXiv:2506.19697
18. DeepSeek-V2 (DeepSeek-AI, 2024) -- arXiv:2405.04434
19. DeepSeek-V3 (DeepSeek-AI, 2024) -- arXiv:2412.19437
20. QuantMoE-Bench (Li et al., arXiv 2024) -- arXiv:2406.08155
21. MC-MoE (He et al., ICLR 2025) -- arXiv:2410.06270
22. EAC-MoE (ACL 2025)
23. MoEQuant (Hu et al., arXiv 2025) -- arXiv:2505.03804
24. MoE Inference Survey (Liu et al., arXiv 2024) -- arXiv:2412.14219
25. MoE-I^2 (arXiv 2024) -- arXiv:2411.01016
26. PuzzleMoE (ICML 2026) -- arXiv:2511.04805
27. AQLM (Egiazarian et al., ICML 2024) -- arXiv:2401.06118
28. KVTuner (Li et al., ICML 2025) -- arXiv:2502.04420
29. QServe -- vLLM integration, 2025
30. DFRot (Xiang & Zhang, COLM 2025) -- arXiv:2412.00648
