# Notes: LLM 量化推理文献扫描

## Research Context
- 主线：LLM 量化推理
- 目标场景：MOE 模型
- 硬件：4×B300 (原生 FP4) + 8×A100
- 时间线：2-4 个月快速产出
- 覆盖：低比特精度-效率 + 权重激活联合量化 + 系统优化

---

## 方向 A: 低比特精度-效率与混合精度策略 ✅ (completed)

### A1: 低比特量化格式对比

**Key papers:**
- **INT vs FP: A Comprehensive Study** (Chen/Wu, HKU & ByteDance, arXiv:2510.25602, Oct 2025) — MXINT8 > MXFP8 at 8-bit, but NVFP4 > NVINT4 at 4-bit. The "crossover phenomenon"
- **MX+** (Lee et al., MICRO 2025, arXiv:2510.14557) — Repurposes block-max exponent as extra mantissa, +42.15% accuracy over MXFP4 with negligible overhead
- **AMXFP4** (Lee et al., ACL 2025 Findings, arXiv:2411.09909) — Asymmetric microscaling suppresses outliers, calibration-free, outperforms MXFP4 by 3%
- **VPTQ** (Liu et al., EMNLP 2024, arXiv:2409.17066) — Vector quantization to 2-bit, SOTA accuracy with 10-18% of prior methods' quantization time

**Format comparison at 4-bit:**

| Format | Block Size | Scale | Native HW | Accuracy Ranking |
|--------|-----------|-------|-----------|-----------------|
| NVFP4 | 1×16 | 2-level (E4M3+FP32) | B200/B300 | Top tier (with QAT) |
| MXFP4 | 1×32 | 1-level (E8M0) | Blackwell, Trainium | Strong baseline |
| MX+/AMXFP4 | 1×32 | Asymmetric/Extended | Blackwell (native) | Near-FP8 accuracy |
| AWQ/GPTQ INT4 | group 128 | FP16 per-group | Universal (CUDA) | Good (1-3% drop) |
| BNB-nf4 | block 64 | FP32 per-block | Universal (LUT) | **Worst** — up to 59% drop on long-context |

**Key insight:** NVFP4 and MXFP4 are ISA-level incompatible. NVFP4's 1×16 blocks give ~88% lower quantization error than MXFP4's 1×32 blocks, but at 4.5 vs 4.25 effective bits. **No universal format exists — NVIDIA bet on proprietary NVFP4.**

**B300 match:** ★★★★★ Native NVFP4 + MXFP4 Tensor Cores. Full FP4 stack.
**A100 match:** ★★☆☆☆ No native FP4/FP8. INT4 via CUDA emulation (~85% dequant overhead).

### A2: 混合精度量化策略

**Key papers:**
- **MicroMix** (Liu et al., ICLR 2026, arXiv:2508.02343) — Per-channel MXFP4/6/8 within a single layer, co-designed kernel, 2.29-3.38x over TensorRT-FP16
- **AMQ** (Lee et al., arXiv:2509.12019) — AutoML for layer-wise bit allocation, reaches Pareto frontier
- **KVTuner** (Li et al., ICML 2025, arXiv:2502.04420) — Near-lossless 3.25-bit mixed KV cache for Llama-3.1-8B
- **ScaleBITS** (arXiv:2602.17698) — Block-wise bit allocation via bi-directional channel sensitivity, +36% over uniform quantization
- **SFMP** (arXiv:2602.01027) — Fractional bit-widths, unified GEMM kernel for arbitrary average bit-width

**Sensitivity hierarchy (consistent across papers):**
1. FFN layers >> Attention layers (quantization tolerance)
2. Down-projection layers = most sensitive
3. Early & late layers > middle layers (need higher precision)
4. Gating scores correlate 0.99 Pearson with expert importance → precision allocation is lightweight
5. Bit-width and LoRA rank interact non-trivially (AutoQRA)

**Gaps:** Search cost is massive (>10^100 configurations). No runtime precision adaptation. Interaction with speculative decoding and MoE routing underexplored.
**B300 match:** ★★★★☆ Native MXFP4/6/8 in Tensor Cores enables MicroMix-type approaches. W4A8 recipe (DeepSeek R1) is production-validated.
**A100 match:** ★★☆☆☆ Only INT8+FP16 mixed, no fine-grained mixed precision.

### A3: 低比特对 emergent ability 的影响

**Key papers:**
- **Do Emergent Abilities Exist in Quantized LLMs** (Liu et al., LREC-COLING 2024, arXiv:2307.08072) — First systematic study: 4-bit retains ICL/CoT/IF; 2-bit catastrophic
- **Can Compressed LLMs Truly Act?** (Dong et al., ICML 2025, arXiv:2505.19433) — ACBench for agentic evaluation; 4-bit drops 10-15% on real-world tasks; **DeepSeek-R1 is particularly compression-sensitive**
- **Does Quantization Affect Long-Context?** (Mekala et al., EMNLP 2025, arXiv:2505.20276) — 5 models × 5 methods × 26 languages × 128K tokens; 8-bit safe, 4-bit degrades progressively with length
- **Evaluating Quantized LLMs up to 405B** (Li et al., IJCAI 2025, arXiv:2409.11055) — FP8 most robust; AWQ > GPTQ; larger models tolerate better

**Degradation hierarchy (4-bit, most→least sensitive):**
1. Long-context reasoning (>32K): 10-59% drop
2. Multilingual / low-resource: up to 5× English drop
3. Multi-step math reasoning: 7-15% drop
4. Real-world agentic tasks: 10-15% drop
5. Instruction following: 1-3%
6. In-context learning: 1-3%

**Key findings:** Error mode → Incorrect Logic (~50%), Calculation Error (~20%). Qwen-2.5 significantly more robust than Llama-3.1. Distilled reasoning models (DeepSeek-R1) more sensitive than base models.

**B300 match:** ★★★★☆ 288GB VRAM enables full-precision baselines at scale for systematic evaluation.
**A100 match:** ★★★☆☆ VRAM-constrained for large-model comparison studies.

### A4: B300 native FP4 软硬件协同

**Key papers/reports:**
- **SageAttention3** (Zhang et al., NeurIPS 2025, arXiv:2505.11594) — First FP4 attention kernel on Blackwell, 1,038 TOPS on RTX 5090, ~5× FlashAttention2
- **LMSYS GB200 NVL72 Report** (Sep 2025) — 3.8× prefill, 4.8× decode vs H100 using NVFP4 MoE + FP8 attention
- **NVIDIA "Pushing Intelligence to 4-bit"** (2025) — NVFP4 format design, 88% lower error than power-of-two scaling
- **Silicon Showdown** (arXiv:2605.00519, May 2026) — RTX 5090 NVFP4 falls back to Marlin BF16; no native FP4 compute on consumer Blackwell

**Full FP4 inference stack on B300:**
| Component | Precision | Hardware Path | Speedup vs FP8 |
|-----------|-----------|---------------|----------------|
| Weights GEMM | NVFP4 | tcgen05.mma native | ~1.9× |
| Activations GEMM | NVFP4/FP8 | tcgen05.mma | ~1.9× (FP4) |
| KV Cache | NVFP4 | HW FP4→FP8 dequant | ~50% memory |
| Attention (QK,PV) | NVFP4 | SageAttention3 | ~5× vs FA2 |
| MoE dispatch | NVFP4 | DeepEP fused quantize | ~50% comm reduction |

**Production benchmarks:**
- Llama-3.3-70B, B300 FP4: 8,196 tok/s/GPU (3.1× vs H200 FP8)
- DeepSeek-R1, B300 FP4: 6,235 tok/s/GPU (4.7× vs H200 FP8)
- DeepSeek V4, 4×B300 NVFP4: AIME pass@1=96.0% vs BF16 96.15%

**B300 match:** ★★★★★ Primary platform for FP4 research.
**A100 match:** ★☆☆☆☆ No native FP4 compute. Baseline comparison only.

---

## 方向 C: 权重激活联合量化 ✅ (completed)

### C1: W4A4/W4A8/W8A8 联合量化 SOTA

**Key papers (8):**
- **FlatQuant** (Sun et al., ICML 2025, arXiv:2410.09426) — Learnable Kronecker-factored affine transforms; LLaMA-3-70B W4A4: -0.94% vs FP16; beats SpinQuant by 7.5%
- **SpinQuant** (Liu et al., Meta, ICLR 2025, arXiv:2405.16406) — Learned Cayley rotations; 45.1% gap reduction vs QuaRot
- **QuaRot** (Ashkboos et al., NeurIPS 2024, arXiv:2404.00456) — First end-to-end W4A4KV4 via random Hadamard rotations; 99% zero-shot retained on LLaMA-2 70B
- **PrefixQuant** (Chen et al., arXiv:2410.05265) — First static per-tensor quant beating dynamic per-token; outlier token prefixing in KV cache
- **CBQ** (Ding et al., ICLR 2025 Spotlight, arXiv:2312.07950) — Cross-block reconstruction with LoRA-Rounding; >99% perf retention
- **MergeQuant** (Wang et al., arXiv:2503.07654) — Per-channel static quantization with QSM; 2.06× speedup
- **LO-BCQ** (Elangovan et al., NVIDIA, TMLR 2025, arXiv:2502.05376) — Block clustering + Lloyd-Max codebooks; <0.1 PPL loss
- **Atom** (Zhao et al., MLSys 2024, arXiv:2310.19102) — W4A4 mixed-precision + fused CUDA kernels; 7.7× throughput

**SOTA summary by bit-width:**
| Bit-Width | PTQ Status | QAT Status |
|-----------|-----------|------------|
| W8A8 | Near-lossless, PTQ dominates | Unnecessary |
| W4A8 | PTQ preferred (QServe, PrefixQuant) | Cost rarely justified |
| W4A4 | <1% loss on 70B+, 2-5% on 7B | QAT wins by 1-2 pts on small models |
| Sub-3-bit | PTQ collapses | QAT essential |

**Key insight:** On 70B+ models, rotation-based W4A4 PTQ achieves >99% FP16 accuracy. The FC2/down_proj layer in SwiGLU FFN dominates quantization error (kurtosis ~1921 vs ~2.85 for other layers).

**B300 match:** ★★★★☆ NVFP4 native support enables W4A4 but needs custom kernels (none upstream yet).
**A100 match:** ★★★☆☆ W4A4 suffers from CUDA core dequant overhead (rho=64:1). W4A8 (QServe) often faster in practice.

### C2: 最优位宽组合

**Key papers (7):**
- **Scaling Laws for Precision** (Kumar et al., arXiv:2411.04330) — Compute-optimal training precision ≈ 7-8 bits
- **Low-Bit Quantization Favors Undertrained LLMs** (Ouyang et al., ACL 2025, arXiv:2411.17691) — Error scales ~1/N^0.23 but **increases** with training tokens (~D^0.53). Future 100T-token models may degrade MORE
- **ParetoQ** (Liu et al., Meta, NeurIPS 2025, arXiv:2502.02631) — Sub-4-bit can outperform 4-bit on Pareto frontier. Learning transition at 2-3 bits
- **TaCQ** (Xiao et al., COLM 2025, arXiv:2504.07389) — Task-specific weight circuits; 3.1 bits recovers 96% MMLU
- **DeepSeek Quant Analysis** (China Unicom, arXiv:2505.02390) — Q4_K_M: 0% drop; Q2_K_L (2.91 bits): **8.91% drop**; AIME math: 39.2→15.41

**Catastrophic degradation thresholds (3 independent findings converge):**
- Emergent abilities preserved at 4-bit, **2-bit = catastrophic** (GSM8K near 0%)
- PTQ degrades sharply below W4A8
- DeepSeek-V3: safe floor ≈ 3.5 bits, Q2_K_L (2.91 bits) = 8.91% drop

**Task sensitivity ranking (most→least):**
1. Code Generation — activation quantization especially harmful
2. Math Reasoning — multi-step chains amplify per-step error
3. Long-Context — KV cache becomes dominant memory consumer
4. Knowledge Recall — most robust, well-preserved at 4-bit

**Sweet spots:** Cloud API = W8A8, Cloud batch = W4A16, Single-GPU 70B = W4A4 (rotated), Edge = 2-bit (ParetoQ), Long-context = W4A4+KV4

**B300 match:** ★★★★★ Natural match for NVFP4 W4A4. DeepSeek-R1 on DGX B200 >3× vs H200 FP8.
**A100 match:** ★★★☆☆ AWQ-4bit (W4A16) is most deployed. W8A8 via SmoothQuant recommended.

### C3: 激活量化挑战

**Key papers (7):**
- **SmoothQuant** (Xiao et al., ICML 2023, arXiv:2211.10438) — Per-channel scaling migrates quantization difficulty from activations to weights; W8A8 negligible loss
- **DuQuant** (Lin et al., NeurIPS 2024 Oral, arXiv:2406.01721) — Dual transformation (rotation + zigzag permutation) for both Normal and Massive outliers
- **KVQuant** (Hooper et al., NeurIPS 2024, arXiv:2401.18079) — 3-bit KV cache via per-channel Key quant + Pre-RoPE quant + non-uniform datatypes
- **OSP** (Park et al., ACL 2025, arXiv:2506.19697) — **Pre-training prevention**: Muon optimizer + Single-Scale RMSNorm eliminates outliers entirely (kurtosis 0.04 vs 1818.56)

**Why activations are harder:**

| Aspect | Weights | Activations |
|--------|---------|-------------|
| Distribution | Near-Gaussian, smooth | Long-tailed, outliers 100-1000× normal |
| Outliers | Rare, small | Systematic, persistent channels |
| Dynamic range | Static | Input-dependent, per-token variance |

**Two outlier types identified:**
1. Channel-wise (Normal): Large magnitude in specific channels, persistent across ALL tokens
2. Spike (Massive): ~1400× median, in few tokens only, discovered in FFN down_proj

**Root cause:** Diagonal preconditioners (Adam) + channel-wise normalization (RMSNorm) amplify privileged bases. OSP proves replacing Adam with Muon eliminates outliers entirely.

**Dominant paradigm 2024-2025:** Rotation-based methods (QuaRot→SpinQuant→FlatQuant→DuQuant). Hadamard transforms increase "incoherence" — spreading outlier magnitudes across dimensions.

**Small batch challenge:** Stronger outlier contamination + dynamic range miscalibration + memory-bandwidth bound + sequence-length effects. No method simultaneously achieves high accuracy AND low overhead at batch=1.

**B300 match:** ★★★★☆ NVFP4 block-level FP8 scaling aligns with per-group granularity trends.
**A100 match:** ★★★☆☆ INT8 Tensor Cores require per-channel weights + per-tensor activations; no FP8.

### C4: MOE 联合量化

**Key papers (8):**
- **DeepSeek-V2** (arXiv:2405.04434) — MLA + DeepSeekMoE; INT4/INT8 serving
- **DeepSeek-V3** (arXiv:2412.19437) — **First 671B MoE FP8 mixed-precision training**; 1×128 activation scaling, 128×128 weight scaling; 2.788M H800 GPU hours
- **QuantMoE-Bench** (Li et al., arXiv:2406.08155) — First systematic MoE PTQ benchmark; data-driven mixed-precision per component type
- **MC-MoE** (He et al., ICLR 2025, arXiv:2410.06270) — ILP-based mixed-precision + online dynamic pruning; **2.54-bit average, only 3.8% loss**
- **EAC-MoE** (ACL 2025) — Identifies "expert shift" problem; TopK-MSE router calibration; router distortion alone raises PPL from 3.84 to 4.17
- **MoEQuant** (Hu et al., arXiv:2505.03804) — Expert-Balanced Self-Sampling + Affinity-Guided Quantization
- **PuzzleMoE** (ICML 2026, arXiv:2511.04805) — Sparse expert merging + 3-bit group quantization; 4.8× compression, ~1.7% loss

**MOE-specific quantization challenges:**

| Component | % of Params | Sensitivity | Rec. Bit-Width |
|-----------|------------|-------------|----------------|
| Router (gate) | <0.03% | **EXTREME** — controls expert selection | FP16 or 8-bit minimum |
| Attention (MHSA) | ~30-40% | High — weight outliers prominent | 4-bit |
| Shared experts | Varies | High — affects ALL tokens | 4-bit |
| Routed experts | ~60-70% | Variable — error is localized to k tokens | 2-4 bit |
| Early MoE layers | — | High — error propagates through depth | 4-bit |
| Late MoE layers | — | Lower — less impact on output | 2-bit |

**Expert heterogeneity evidence:** Different experts produce different activation ranges, activation frequency alone is insufficient for bit allocation, Hessian trace is more reliable than frequency, and counterintuitively — experts learning low-frequency tokens are MORE sensitive to quantization.

**Expert shift problem:** Quantization distorts router logits → different experts selected vs FP16. EAC-MoE: router distortion alone (with perfect expert weights) raises PPL from 3.84→4.17.

**B300 match:** ★★★★★ NVFP4 for expert FFN + FP8 for attention + BF16 for shared layers. DeepSeek-V4-Flash on 4×B300: AIME pass@1=96.00% (BF16=96.15%), 2.95× wall-clock speedup.
**A100 match:** ★★★☆☆ 80GB can't fit unquantized Mixtral-8x7B (~90GB+). INT4 quantization essential. Expert offloading + quantization synergy (HOBBIT, DAOP).

---

## 方向 D: 量化推理系统优化 ✅ (completed)

### D1: 低比特推理 kernel

**Key papers/projects:**
- **MARLIN** (Frantar et al., PPoPP 2025, arXiv:2408.11743) — FP16×INT4 GEMM kernel, 2.8x e2e speedup in vLLM at batch 16-32
- **FlashInfer** (UW/CMU, arXiv:2501.01005) — FP8/NVFP4/MXFP4/INT4 attention kernels, 1.2-1.3 PFLOPs/s on H100
- **BitBLAS/Ladder** (Microsoft, OSDI 2024) — Hardware-aware low-bit GEMM, up to 8x vs cuBLAS for INT4/INT2/INT1
- **Alpha-MoE** (Aleph Alpha, 2025) — Fused FP8 W8A8 MoE kernel, 200% speedup vs Triton kernels
- **CUTLASS SM100** — Native FP4 via `tcgen05.mma` with hardware block scaling (NVFP4: 16-element blocks, MXFP4: 32-element)

**Key insight:** W4A4 on Hopper is slower than FP16 due to CUDA-core dequant overhead. Blackwell's native FP4 tensor cores make true W4A4 faster than W4A16 for the first time.

**B300 match:** ★★★★★ Native FP4 UMMA acceleration. B300 delivers 5-10× throughput vs A100 at same precision.
**A100 match:** ★★★☆☆ Software dequant only, Marlin works (CC≥8.0), no FP4 path.

### D2: 量化推理显存管理

**Key papers:**
- **KIVI** (Liu et al., ICML 2024, arXiv:2402.02750) — Tuning-free 2-bit KV cache quantization, per-channel key + per-token value
- **CacheGen** (Liu et al., SIGCOMM 2024, arXiv:2310.07240) — 3.5-4.3× KV compression with adaptive bandwidth encoding
- **TurboQuant** (2025) — Calibration-free random rotation + optimal scalar quantization, 4-7× KV compression
- **KV Pareto** (Gokhale et al., 2025, arXiv:2512.01953) — Joint AWQ weights + KV quantization + prefill chunking optimization, 68-78% total memory reduction
- **PMPD** (Chen et al., ICLR 2025) — Progressive mixed-precision decoding: higher precision for prefill, gradually reduce during decode

**SOTA summary:** INT4 per-channel KV quantization is near-lossless baseline (≈75% KV reduction). 2-bit KV (KIVI) achieves 87.5% reduction with minor degradation. Combined weight+KV quantization achieves 68-78% total memory reduction with 1-3% accuracy drop.

**B300 match:** ★★★★☆ 288GB HBM3e reduces urgency, but KV cache still dominates at 128K+ context. FP8 attention halves decode memory access.
**A100 match:** ★★★★★ KV quantization is practically mandatory for >32K context on 80GB.

### D3: 多卡量化推理分发

**Key papers:**
- **TP-Aware Dequantization** (Hoque et al., IBM, 2024, arXiv:2402.04925) — Optimize GPTQ+TP deployment, 1.81× speedup on Llama-70B
- **Communication Compression for TP LLM Inference** (Hansen-Palmus et al., 2024, arXiv:2411.09510) — FP4/FP5/INT4 block quantization before AllReduce, 1.2-2× TTFT reduction
- **SplitQuant** (Zhao et al., CLUSTER 2025) — Phase-aware joint optimization of mixed precision + TP/PP + micro-batch on heterogeneous GPUs
- **vLLM-Ascend RFC #3012** (2025) — Quantize activations **before** All2All, ~50% communication payload reduction for MoE

**Key insight:** Communication compression is most valuable on PCIe/IB, marginal on NVLink. TP-aware dequant eliminates AllGather by maintaining GPU-local data. No automated "quantization-aware 3D parallelism" compiler exists yet.

**B300 match:** ★★★☆☆ NVLink 5 (1.8 TB/s) makes communication less bottlenecked. 288GB means fewer GPUs needed — PP becomes less relevant.
**A100 match:** ★★★★☆ Communication compression provides max value on NVLink 3 (600 GB/s) + PCIe.

### D4: MOE 量化推理调度

**Key papers:**
- **HOBBIT** (Tang et al., 2024, arXiv:2411.01433) — 3-tier mixed-precision expert offloading, 9.93× decode speedup vs SOTA offloading
- **DyMoE** (2025, arXiv:2603.19172) — Runtime dynamic mixed-precision based on expert importance, 3.44-22.7× TTFT reduction
- **D2MoE** (Mobicom 2025, arXiv:2504.15299) — Matryoshka nested bit-width for MoE weights, token-adaptive selection, up to 53% memory reduction
- **MoE-SpeQ** (Wang/Liu et al., 2025, arXiv:2511.14102) — 4-bit quantized draft model for speculative expert prefetching, 90.9% fidelity, 2.34× speedup
- **Fate** (2025, arXiv:2502.12224) — Cross-layer gate prediction with 99% expert cache hit rate, 4.5× prefill and 4.1× decode speedup

**Production systems:**
- **SGLang**: W8A8 FP8, W4AFP8 (expert INT4 + dense FP8), NVFP4, DeepGEMM integration. GB200 NVL72: 26,156 input tok/s and 13,386 output tok/s per GPU.
- **vLLM**: FP8, AWQ, GPTQ, Marlin INT4, NVFP4 (experimental), DeepGEMM integration.

**Key insight:** I/O dominates MoE offloading (85-99% time). Quantization helps by reducing bytes per transfer. Gating scores correlate 0.99 Pearson with expert importance → lightweight runtime precision allocation is viable.

**B300 match:** ★★★★★ 288GB keeps more experts resident. NVFP4 halves MoE weights near-FP8 accuracy. NVLink 5 accelerates All2All. GB200 benchmarks show 3.8× prefilling and 4.8× decoding vs H100 for MoE.
**A100 match:** ★★★☆☆ Works for INT4 MoE with aggressive offloading but limited by 80GB + PCIe bandwidth.

---

## Synthesized Findings

### 10 Cross-Cutting Signals from All Three Directions

1. **B300 NVFP4 is the watershed.** Hopper W4A4 was slower than FP16 due to CUDA-core dequant overhead. Blackwell makes true W4A4 faster than W4A16 for the first time. This fundamentally changes the quantization design space.

2. **W4A4 is algorithmically solved for dense 70B+ models.** Rotation-based PTQ (QuaRot→SpinQuant→FlatQuant) achieves <1% accuracy loss. But W4A4 has NO production kernel support in any framework.

3. **MOE-specific joint quantization is the largest open gap.** All three scans independently point here. MOE architecture changes everything: router sensitivity, expert heterogeneity, sparsity patterns, All2All communication. Nobody has adapted rotation-based W4A4 for MOE.

4. **The format war is unresolved.** NVFP4 (NVIDIA proprietary) vs MXFP4 (OCP standard) are ISA-level incompatible. No universal 4-bit standard exists. This has practical consequences for model portability.

5. **The catastrophic threshold is ~2.9 bits.** Three independent studies converge: 4-bit safe, 2-bit catastrophic. DeepSeek-V3 at 2.91 bits drops 8.91% average + math collapses (AIME 39.2→15.41).

6. **Router is the most sensitive MoE component.** <0.03% of parameters but quantization distortion alone raises PPL from 3.84→4.17 (expert shift). Router quantization below 4-bit is unsolved.

7. **Mixed-precision is the consensus path forward.** Per-expert/per-layer/per-channel precision allocation is validated across all three directions. MicroMix (ICLR 2026), KVTuner (ICML 2025), MC-MoE (ICLR 2025) all point this way.

8. **Long-context degradation is the biggest quality risk.** 4-bit models degrade progressively with context length (up to 59% for BNB-nf4). Mechanism is poorly understood theoretically.

9. **No quantization-aware parallelism compiler exists.** Quantization plan + expert parallel topology + communication schedule are optimized independently. vLLM-Ascend's "quantize before All2All" is the closest but is hand-crafted.

10. **Production MOE quantization is rough.** SGLang W4AFP8 (expert INT4 + dense FP8) and DeepSeek-V3 native FP8 are the only production-validated recipes. No framework supports per-expert precision or rotation-based W4A4 for MOE.

---

## Gap Matrix

### Evaluation Dimensions
- **Gap Size**: How much unsolved research space exists?
- **Hardware Match**: How well does it leverage B300 (NVFP4) + A100?
- **Feasibility**: Can it be done in 2-4 months?
- **Novelty**: How likely to be scooped? How crowded is the space?
- **Impact**: If successful, how significant is the contribution?

### Gap Candidates (synthesized across A+C+D)

| # | Gap | Lines | Gap Size | HW Match | Feasibility | Novelty | Impact | Overall |
|---|-----|-------|----------|----------|-------------|---------|--------|---------|
| **G1** | **MOE-specific W4A4 joint quantization** (rotation-based, router-aware, expert-adaptive) | C3+C4+D4 | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ | **P0** |
| **G2** | **B300-native NVFP4 kernel + algorithm co-design for W4A4** | A4+D1+C1 | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | **P0** |
| **G3** | **Mixed-precision MoE with automated expert-aware allocation** | A2+C2+C4 | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | **P1** |
| **G4** | **Quantization-aware MoE parallelism compiler** (quant + EP + TP joint optimization) | D3+D4 | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★★★★ | **P1** |
| **G5** | **Long-context quantization robustness** (mechanism + mitigation) | A3+D2 | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | **P2** |
| **G6** | **Small-batch activation quantization for interactive serving** | C3+D2 | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | **P2** |
| **G7** | **FP4 speculative decoding** (quantized draft model + target model) | A4+D4 | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★☆ | **P2** |

### Detailed Gap Analysis

#### G1 — MOE-specific W4A4 Joint Quantization [P0 ★★★★★]

**What:** Adapt rotation-based W4A4 methods (QuaRot/SpinQuant/FlatQuant) for MoE architecture with:
- Expert-specific rotation matrices (experts have heterogeneous activation distributions)
- Router-aware calibration (TopK-MSE, Rank-Aware Jaccard)
- Expert-shift mitigation during quantization
- Potentially: per-expert bit-width allocation based on importance

**Evidence of gap:**
- No rotation-based W4A4 method has been adapted for MoE (all tested on dense LLaMA)
- QuantMoE-Bench (2024) only covers weight-only and W8A8, not W4A4
- MoEQuant (2025) addresses calibration imbalance but doesn't use rotations
- EAC-MoE shows router distortion alone kills performance
- DeepSeek-V3 uses FP8 training, not PTQ W4A4

**Hardware alignment:**
- B300: NVFP4 native for expert FFN matmul (~2× vs BF16)
- A100: INT4 for expert weights, can validate generalization across hardware generations
- Mixed deployment: NVFP4 experts (B300) + FP8 attention (A100) = natural experiment

**2-4 month scope:** Method design + implementation + evaluation on Mixtral/Qwen-MoE/DeepSeek-V2-Lite

**Risk:** Someone may publish MoE+rotation quantization during our work (field is hot)

---

#### G2 — B300-native NVFP4 Kernel + Algorithm Co-design [P0 ★★★★☆]

**What:** Design quantization algorithms that exploit NVFP4's unique hardware properties (1×16 blocks, 2-level scaling) rather than treating it as a generic 4-bit format. Co-design with custom CUDA kernels.

**Evidence of gap:**
- NVFP4's 1×16 blocks give ~88% lower quantization error than MXFP4's 1×32 blocks
- All existing W4A4 methods were designed for generic INT4/FP4, not NVFP4-specific
- SageAttention3 (NeurIPS 2025) shows what hardware-aware FP4 algorithm design can achieve
- No published work on NVFP4-optimized quantization algorithms (only NVIDIA internal)

**Hardware alignment:**
- B300: Primary platform. NVFP4 tensor cores (tcgen05.mma), FP4-to-FP8 HW dequant
- A100: Baseline comparisons, INT8 ceiling

**2-4 month scope:** Profile NVFP4 behavior → design block-aware quantization → implement custom kernel → benchmark vs generic methods

**Risk:** Requires low-level CUDA/CUTLASS work; hardware access during development

---

#### G3 — Automated Mixed-Precision for MoE [P1 ★★★★☆]

**What:** Build an automated framework that assigns per-expert bit-widths based on importance/routing frequency/sensitivity, co-optimized with the quantization method.

**Evidence of gap:**
- MC-MoE (ICLR 2025) uses ILP-based allocation but only weight-only, no activation quant
- MicroMix (ICLR 2026) does per-channel MXFP4/6/8 but for dense models only
- No work combines: per-expert precision + joint W-A quantization + automated search
- Expert heterogeneity is well-documented but not systematically exploited for quantization

**Hardware alignment:**
- B300: Native MXFP4/6/8 enables fine-grained mixed precision at hardware level
- A100: Coarser mixed precision (INT8+FP16) but validates generalization

**2-4 month scope:** Sensitivity analysis → search algorithm → mixed-precision deployment → evaluation

**Risk:** Medium — multiple groups are working on MoE compression

---

#### G4 — Quantization-Aware MoE Parallelism Compiler [P1 ★★★★★]

**What:** Jointly optimize quantization plan + expert parallel topology + communication schedule. Automate the "quantize before All2All" pattern discovered by vLLM-Ascend.

**Evidence of gap:**
- No automated framework exists (confirmed by all three scans)
- vLLM-Ascend RFC #3012: hand-crafted quantize-before-All2All saves ~50% communication
- SplitQuant is closest but only addresses heterogeneous GPU types, not MoE topology
- TP-aware dequantization (IBM, 2024) and communication compression (Recogni, 2024) are early signals

**Hardware alignment:**
- B300: NVLink 5 (1.8 TB/s) reduces but doesn't eliminate need for communication optimization
- A100: NVLink 3 (600 GB/s) — communication compression provides max value here

**2-4 month scope:** AMBITIOUS. May need to descope to a specific sub-problem (e.g., EP-aware quantization for DeepSeek-style models only).

**Risk:** Systems paper timeline is tight; requires integration with vLLM/SGLang

---

#### G5 — Long-Context Quantization Robustness [P2]

**What:** Theoretically understand WHY 4-bit degrades progressively with context length, and design context-length-aware quantization strategies.

**Gap:** EMNLP 2025 paper documents the phenomenon comprehensively but offers no mechanism-level explanation. Theory gap + practical mitigation gap.

**Risk:** May require fundamental theoretical work that doesn't fit the 2-4 month engineering timeline.

---

#### G6 — Small-Batch Activation Quantization [P2]

**What:** Achieve high-accuracy activation quantization at batch=1 without dynamic quantization overhead.

**Gap:** All static methods degrade at batch=1. Dynamic methods add 5-10% latency that is proportionally worse at small batch.

**Risk:** Incremental contribution. Crowded space (multiple 2025 papers address this).

---

#### G7 — FP4 Speculative Decoding [P2]

**What:** Use FP4-quantized draft model + FP8/FP16 target model for speculative decoding. NVFP4's speed advantage makes draft model extremely fast.

**Gap:** No published work combining FP4 inference with speculative decoding. Natural fit for B300.

**Risk:** Requires careful draft-target accuracy alignment. May not work if FP4 draft quality is too low.

---

## Phase 1 Research: Rotation-FP4 Interaction Deep-Dive (2026-07-12)

### Papers Analyzed (7+1 papers)

1. **MR-GPTQ** (Egiazarian et al., ICLR 2026, arXiv:2509.23202)
2. **BRQ** (Shao et al., arXiv:2511.04214, Nov 2025) -- emerged as highly relevant
3. **DuQuant++** (Lin et al., arXiv:2604.17789, Apr 2026)
4. **SOAR** (Bao et al., arXiv:2605.12245, May 2026)
5. **HiF4** (Luo et al., arXiv:2602.11287, Feb 2026)
6. **FAAR** (Lin et al., arXiv:2603.22370, Mar 2026)
7. **4/6** (Cook et al., arXiv:2512.02010, Dec 2025)
8. **INT vs FP Study** (Chen/Wu, arXiv:2510.25602, Oct 2025) -- crossover phenomenon

---

### Paper 1: MR-GPTQ (arXiv:2509.23202, ICLR 2026)

**Exact mechanism:** Formal proof (Lemma 1 + asymptotic analysis).

Lemma 1: Under i.i.d. Normal(0,1), after Hadamard + MFP quant, `MSE_top(G) = MSE(G)`. Without rotation (Laplace), `MSE_top(G) = 0` (absmax preserves outlier).

Preserved mass decay:
- Laplace: R_L(G) = Theta((log G)^2 * G^(-delta)), delta = q_min/2
- Normal (post-rotation): R_N(G) = Theta(sqrt(log G) * G^(-delta^2))
- Since delta^2 < delta, Normal wins at large G, loses at small G -> crossover exists.

At NVFP4 G=16, rotation INCREASES per-element MSE. "Provably" means asymptotic crossover is mathematically derived.

**Quantitative rotation effect (from MR-GPTQ Table):**

| Model | NVFP4 RTN | NVFP4 RTN+HT | HT Effect |
|-------|-----------|--------------|-----------|
| Llama3-8B | 94.8% | 93.8% | **-1.0% (hurts)** |
| Llama3-1B | 83.9% | 80.9% | -3.0% |
| Qwen3-8B | 98.9% | 96.0% | -2.9% |

| Model | MXFP4 RTN | MXFP4 RTN+HT | HT Effect |
|-------|-----------|--------------|-----------|
| Llama3-8B | 88.1% | 89.3% | **+1.2% (helps)** |
| Llama3-1B | 67.7% | 74.4% | +6.7% |

**Key insight:** Rotation HURTS NVFP4 (G=16) but HELPS MXFP4 (G=32) -- consistent with crossover theory. NVFP4's E4M3 scale already provides fine-grained per-block adaptation. MXFP4's E8M0 is too coarse, so rotation helps.

**Gap:** Asymptotic only. Real LLM distributions not exactly Laplace/Normal. Codebook non-uniformity not modeled. Exact crossover G* not solved analytically.

---

### Paper 2: BRQ (arXiv:2511.04214, Nov 2025)

**3 destructive mechanisms for global rotation under MXFP4:**

1. Outlier energy redistribution inflates >70% of normal block scales
2. MXFP4's PoT scale (E8M0) is too coarse to compensate (scale must jump 2x)
3. Regular blocks' inflated errors dominate cumulative loss

**Quantitative evidence:**

| Method | LLaMA-2 7B PPL | LLaMA-3 8B PPL |
|--------|---------------|----------------|
| FP16 | 5.47 | 6.14 |
| MXFP4 RTN | 7.08 | 8.23 |
| QuaRot (global rot + RTN) | **13.09 (catastrophic)** | 9.56 |
| SpinQuant (optimized global) | 5.99 | 7.62 |
| **BRQ (random block rot)** | -- | **7.14** |

BRQ with RANDOM block rotations already outperforms SpinQuant with OPTIMIZED global rotations. BRQ reduces rotation overhead ~40% vs global rotation.

**Gap:** Only studies MXFP4 (G=32). Does block rotation also benefit NVFP4 (G=16)?

---

### Paper 3: DuQuant++ (arXiv:2604.17789, Apr 2026)

**Mechanism:** Under MXFP4, each block of 32 has independent E8M0 scale -> cross-block variance problem (requiring dual rotation + zigzag in original DuQuant) becomes irrelevant. Single outlier-aware rotation suffices. Online rotation cost halved.

**Quantitative (LLaMA-3-8B W4A4 MXFP4):**

| Method | WikiText2 PPL | Avg QA Acc |
|--------|--------------|------------|
| FP16 | 6.14 | 69.1% |
| QuaRot | 9.46 | 62.9% |
| MR-GPTQ | 7.29 | 66.1% |
| DuQuant++ | 7.07 | 66.5% |
| DuQuant++ (w/ GPTQ) | 6.88 | 67.1% |

**Gap:** Empirical observation, not proof. No theoretical characterization of when rotation helps vs hurts. No exploration of block size < 32.

---

### Paper 4: SOAR (arXiv:2605.12245, May 2026)

**Mechanism:** Scale factor inaccuracies (joint global + block scale + scale quantization coupling) are the dominant error source in NVFP4. Two contributions: (1) CJSO: closed-form joint optimization of global + block scales. (2) DSS: decouple quantization scale from dequantization scale.

**Rotation interaction:** NOT tested. SOAR focuses on scale, not rotation. Implication: if scale error dominates, rotation (which addresses element-level distribution) should provide minimal additional benefit. But this is UNTESTED.

**Gap:** Does scale optimization make rotation redundant? Does the error decomposition hold post-rotation?

---

### Paper 5: HiF4 (arXiv:2602.11287, Feb 2026)

**Three-level hierarchy:** E6M2 (base, 64 elements) -> E1_8 (8 sub-blocks) -> E1_16 (16 micro-blocks). Element format: E1M2.

**Format comparison (MSE on Gaussian):** HiF4 : NVFP4 : MXFP4 = 1 : 1.32 : 1.89

**Hierarchy results (W4A4):**
- Weight fidelity: HiF4 > NVFP4 >> MXFP4 >> INT4
- Activation fidelity: NVFP4 > HiF4 >> MXFP4 >> INT4
- W4A4 RTN Qwen3-8B: HiF4 (10.30 PPL) vs NVFP4 (10.16 PPL) vs MXFP4 (11.21) -- NVFP4 and HiF4 very close

**Rotation: COMPLETELY UNTESTED.** HiF4 tests SmoothQuant and SVDQuant (scaling-based), not rotation.

**Gap:** Does HiF4's 3-level hierarchy already handle outliers well enough that rotation is unnecessary? Would rotation help or hurt at G=64 with fine-grained scaling? Is there a U-curve where both very small blocks (G=16, outliers already localized) and very large blocks with fine scaling (G=64, scale handles it) need less rotation, but intermediate blocks (G=32 with coarse scaling) need rotation most?

---

### Paper 6: FAAR (arXiv:2603.22370, Mar 2026)

**Mechanism:** Learnable rounding offsets that explicitly account for NVFP4's non-uniform E2M1 grid ({0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6} with step sizes 0.5/1.0/2.0). Temperature-scaled sigmoid for smooth optimization. 2FA fine-tuning aligns with BF16.

**Quantitative (WikiText-2 PPL):**

| Model | RTN | FAAR | Delta |
|-------|-----|------|-------|
| Llama3-1B | 14.28 | 12.60 | -1.68 |
| Qwen3-1.7B | 23.06 | 21.27 | -1.79 |

FAAR further improves over GPTQ by 1.06-1.41 PPL. Training: ~4 GPU hours on 1B.

**Rotation interaction: COMPLETELY UNTESTED.** FAAR compares against RTN and GPTQ, not rotation methods.

**Gap:** FAAR addresses rounding operator (where to round); rotation addresses distribution shape (input to rounding). These could be complementary -- both untested.

---

### Paper 7: 4/6 (arXiv:2512.02010, Dec 2025)

**Mechanism:** E2M1 codebook has a large 2.0 step between values 4 and 6. Standard scaling (max->6) creates a "near-maximal value gap" for values at 66.6%-100% of block max. 4/6 adaptively scales to 4 instead of 6 per-block, choosing lower MSE.

**Quantitative:** Prevents training divergence. PTQ: +19.9% gap closure vs BF16 with AWQ. Inference overhead <2%.

**Rotation interaction:** RHT used in training recipe but NOT ablated. 4/6 NOT compatible with GPTQ (34.6% degradation -- adaptive scaling disrupts GPTQ assumptions). Paper cites MR-GPTQ: "blockwise Hadamard rotations can increase top-element MSE at NVFP4's small block size (16)."

**Gap:** Direct ablation of 4/6 +/- rotation. Does 4/6's per-block adaptive scaling already handle what rotation would address?

---

### Paper 8 (Bonus): INT vs FP Study (arXiv:2510.25602, Oct 2025)

**Crossover phenomenon:** Hadamard rotation causes NVINT4 to overtake NVFP4 (12/12 models). Crest factor kappa = max/RMS. INT4 wins when kappa < 2.04; FP4 wins when kappa > 2.04. Rotation reduces kappa (spreads outliers), pushing below the crossover threshold.

---

### Claims Matrix

| Claim about rotation-FP4 | Source | Proof or Observation? | Strength |
|---|---|---|---|
| Block size 16 neutralizes rotation (NVFP4) | MR-GPTQ | Proof (Lemma 1 + asymptotic crossover) | **Strong** |
| Global rotation inflates regular block scales (MXFP4) | BRQ (2511.04214) | Empirical + mechanism analysis | **Strong** |
| MXFP4 PoT scale too coarse for post-rotation distributions | BRQ, MR-GPTQ | Mechanism analysis | **Strong** |
| Cross-block variance becomes irrelevant under MXFP4 | DuQuant++ | Empirical observation | Medium |
| Non-uniform codebook (E2M1) interacts with rounding, not rotation | FAAR, 4/6 | Empirical (rounding-specific) | Medium |
| Scale inaccuracy is dominant error source in NVFP4 | SOAR | Analytical (closed-form CJSO) | Medium |
| NVINT4 overtakes NVFP4 after rotation (crossover) | INT vs FP Study | Empirical (crest factor analysis) | **Strong** |
| 2-level scaling (E4M3+FP32) enables finer granularity than 1-level (E8M0) | MR-GPTQ, 4/6 | Mechanism analysis | **Strong** |
| Hierarchy (64->8->4) resolves outliers without rotation | HiF4 | Implicit (not rotation-tested) | Weak |
| Format-aware rounding + rotation interaction | FAAR (untested) | N/A -- gap | **Unknown** |
| Scale optimization makes rotation redundant | SOAR (untested) | N/A -- gap | **Unknown** |

---

### Key Research Gap: Factor Disentanglement

**No paper has independently varied block size, codebook, and scale format to isolate their contributions to rotation (in)effectiveness.**

| Study | Block size | Codebook | Scale format | Clean ablation? |
|-------|-----------|----------|-------------|-----------------|
| MR-GPTQ | Varies (16 vs 32) | Fixed (E2M1) | Varies (E4M3 vs E8M0) | **No** -- format changes all 3 |
| BRQ | Fixed (32) | Fixed (E2M1) | Fixed (E8M0) | Only within MXFP4 |
| INT vs FP | Varies | Varies | Varies | Compares INT4 vs FP4 |
| HiF4 | Fixed (64) | Different (E1M2) | Different (3-level) | No rotation test |
| FAAR | Fixed (16) | Fixed (E2M1) | Fixed (E4M3) | No rotation test |
| 4/6 | Fixed (16) | Fixed (E2M1) | Fixed (E4M3) | No rotation ablation |
| SOAR | Fixed (16) | Fixed (E2M1) | Fixed (E4M3) | No rotation ablation |

### Open Theoretical Question

**"Can we construct a unified error decomposition for block-scaled quantization that cleanly separates the contributions of (a) block size, (b) codebook non-uniformity, and (c) scale format precision to the net rotation benefit DeltaMSE(rotation) = MSE(no rotation) - MSE(with rotation)?"**

This is non-trivial because:
1. MR-GPTQ only models distribution shape (Laplace->Normal), not codebook non-uniformity or scale format
2. BRQ only models cross-block contamination, not within-block error
3. FAAR/4over6 only model rounding to non-uniform grids, not rotation
4. No paper combines all three factors into a single framework
5. The INT vs FP crossover is observed but its three-factor decomposition is not formalized

### Specific Testable Sub-Questions

1. **Q1:** Is there a U-curve for rotation benefit vs block size? Both very small blocks (G<8, already outlier-localized) and very large blocks with fine scaling (G>64, scale handles outliers) may need less rotation than intermediate blocks.

2. **Q2:** Is there a "codebook-limited regime" where rotation cannot reduce error below the E2M1 codebook's inherent quantization floor (determined by the 2.0 step between 4 and 6)?

3. **Q3:** Are block rotation (BRQ-style) + scale optimization (SOAR-style) + format-aware rounding (FAAR-style) complementary? Likely sub-additive, but the cross-terms are unexplored.

4. **Q4:** The INT vs FP crossover after rotation -- can it be predicted from first principles using crest factor analysis generalized to FP formats with non-uniform codebooks?
