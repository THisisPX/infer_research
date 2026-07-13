# Direction A: Low-bit Precision-Efficiency Tradeoffs and Mixed-Precision Strategies for LLM Quantization

## Literature Scan Report (2024-2026)

**Date:** 2026-07-09
**Scope:** arxiv, NeurIPS, ICML, ICLR, EMNLP, ACL, MICRO, DAC, and major lab technical blogs

---

## A1: Low-bit Quantization Format Comparison

### Key Papers (5)

| # | Title | Authors | Venue/Year | arxiv ID | 1-Sentence Contribution |
|---|-------|---------|------------|----------|------------------------|
| 1 | **INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats** | Chen, Wu et al. (HKU & ByteDance) | arxiv / Oct 2025 | 2510.25602 | Systematic head-to-head comparison of INT vs FP formats across granularities; finds MXINT8 > MXFP8 at 8-bit fine-grained, but NVFP4 > NVINT4 at 4-bit (reversible with Hadamard rotation) |
| 2 | **MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving** | Lee, Park, Cha, Cho, Sim (Seoul National Univ.) | MICRO 2025 | 2510.14557 | Repurposes the redundant exponent field of block-max outliers as extra mantissa bits, achieving +42.15% accuracy over MXFP4 with only 0.25 extra bits and near-MXFP4 throughput |
| 3 | **AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference** | Lee, Park, Kim et al. | ACL 2025 (Findings) | 2411.09909 | Asymmetric shared scales simultaneously suppress outliers and capture group-wise asymmetry; calibration-free, outperforms MXFP4 by 3% on VQA |
| 4 | **Post Training Quantization of Large Language Models with Microscaling Formats** | Sharify et al. (Meta / AMD) | NeurIPS ENLSP Workshop 2024 | 2405.07135 | Combines SmoothQuant, AWQ, GPTQ with MX formats; shows MXINT (W4A8) achieves negligible accuracy loss vs uncompressed baselines |
| 5 | **VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models** | Liu, Wen et al. (Microsoft & USTC) | EMNLP 2024 | 2409.17066 | Vector quantization with channel-independent second-order optimization pushing to 2-bit; SOTA accuracy with 10.4-18.6% of prior methods' quantization time |

### SOTA Landscape: Format Taxonomy and Ranking (2024-2026)

**Core format families at 4-bit and below:**

| Format | Element Bits | Block Size | Scale Type | Rep Values | Primary Hardware |
|--------|-------------|------------|------------|------------|-----------------|
| **NVFP4** | E2M1 | 1x16 | E4M3 + FP32 (two-level) | 15 (E2M1) | NVIDIA B200/B300 native |
| **MXFP4** | E2M1 | 1x32 | E8M0 (one-level) | 15 (E2M1) | Blackwell, Trainium native |
| **MXINT4** | INT4 | 1x32 | E8M0 | 16 (uniform) | Blackwell, Trainium native |
| **INT4 (AWQ/GPTQ)** | INT4 | group 128 | FP16 per-group | 16 | Universal (CUDA emulation) |
| **NF4 (BNB-nf4)** | 4-bit NF | per-block 64 | FP32 per-block | 16 (non-uniform) | Universal (lookup table) |
| **E2M3 (FP6 variant)** | E2M3 | 1x32 | E8M0 | 31 | Blackwell MXFP6 native |
| **FP8 (E4M3)** | E4M3 | per-tensor | FP32 | 255 | Hopper native, universal |
| **VPTQ (2-bit)** | 2-bit VQ | codebook | per-codebook | 4 per stage | Universal (LUT) |

**Accuracy ranking at 4-bit (best to worst, as of mid-2026):**

1. **MX+ / MXFP4+** -- extra mantissa for outliers, pushes 4-bit near-FP8 accuracy
2. **NVFP4 + QAT** -- DeepSeek-V4 trains expert weights natively in NVFP4; <1% gap on AIME
3. **AMXFP4** -- outperforms MXFP4 by 3%, rotation methods by 1.6%
4. **MXFP4 / NVFP4 (PTQ)** -- strong baseline with calibration; ~1-3% degradation
5. **AWQ-int4 / GPTQ-int4** -- weight-only; 1-3% degradation on standard tasks
6. **NVINT4 + Hadamard rotation** -- can surpass NVFP4 on certain models
7. **BNB-nf4** -- 6.9% average drop, up to 59% on long-context; use with caution

**Key 2024-2026 findings:**

- **The crossover phenomenon** (arxiv 2510.25602): At coarse granularity (per-tensor), FP formats win due to dynamic range. At fine granularity (MX, block size 32), INT8 wins at 8-bit (MXINT8 > MXFP8), but FP4 wins at 4-bit. This challenges NVIDIA's FP-centric trajectory.
- **Block size matters**: NVFP4's 1x16 blocks give ~88% lower quantization error than MXFP4's 1x32 blocks, at the cost of 4.5 vs 4.25 effective bits per value.
- **NVFP4 is not MXFP4**: NVFP4 uses two-level scaling (E4M3 + FP32), MXFP4 uses single-level (E8M0). They are different formats with different hardware paths.
- **Outlier mitigation is the critical path**: MX+ repurposes exponent bits; AMXFP4 uses asymmetric scales; rotation methods (QuaRot, SpinQuant) use Hadamard transforms. Each approach has different compute/accuracy tradeoffs.

### Bottlenecks and Limitations

1. **No universal 4-bit format**: NVFP4 and MXFP4 are incompatible at the ISA level. NVFP4 is NVIDIA-proprietary; MXFP4 is OCP-standard but has different precision characteristics. A model quantized for one cannot run natively on hardware designed for the other.
2. **4-bit quantization still requires calibration for best results**: While MX formats theoretically enable calibration-free RTN quantization, PTQ with calibration data consistently outperforms direct casting for all formats.
3. **Consumer GPU gap**: RTX 5090 stores NVFP4 weights but falls back to Marlin BF16 dequant kernels -- no native FP4 compute advantage on consumer Blackwell. Native FP4 requires B200/B300 datacenter GPUs.
4. **INT vs FP debate unresolved**: MXINT8 beats MXFP8 at 8-bit, challenging the FP-first hardware trajectory. The industry may be over-investing in FP formats when INT formats are more area/energy-efficient at matched throughput.

### Hardware Match Assessment

| Hardware | Best Native Format | Best Emulated Format | Notes |
|----------|-------------------|---------------------|-------|
| **B300 (Blackwell)** | NVFP4 (native), MXFP4 (native) | -- | Full FP4 stack: weights, activations, KV cache, attention |
| **A100 (Ampere)** | INT8 (native), FP16 (native) | INT4 (CUDA emulation, ~85% overhead in dequant) | No FP8/FP4 native support |
| **H100/H200 (Hopper)** | FP8 (native, 2x BF16 throughput) | INT4, FP4 (CUDA emulation) | FP8 max native precision |

---

## A2: Mixed-Precision Quantization Strategies

### Key Papers (5)

| # | Title | Authors | Venue/Year | arxiv ID | 1-Sentence Contribution |
|---|-------|---------|------------|----------|------------------------|
| 1 | **MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for LLMs** | Liu, Meng, Luo et al. (Tianjin Univ.) | ICLR 2026 | 2508.02343 | Co-designed algorithm + GEMM kernel supporting arbitrary MXFP4/6/8 channel combinations within a single layer; near-FP16 accuracy at ~5-bit average; 2.29-3.38x over TensorRT-FP16 |
| 2 | **AMQ: Enabling AutoML for Mixed-Precision Weight-Only Quantization of LLMs** | Lee et al. | arxiv / Sep 2025 | 2509.12019 | Automated layer-wise bit allocation with search space pruning, proxy-based quality prediction, and iterative search; reaches Pareto frontier of quality vs memory |
| 3 | **KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization** | Li, Xing et al. | ICML 2025 | 2502.04420 | Theoretically-grounded layer-wise attention pattern analysis for KV cache precision allocation; nearly lossless 3.25-bit mixed precision for Llama-3.1-8B |
| 4 | **ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs** | -- | arxiv / 2025 | 2602.17698 | Block-wise bit allocation via bi-directional channel sensitivity analysis and scalable greedy search; +36% over uniform quantization in ultra-low-bit regimes |
| 5 | **SFMP: Fine-Grained, Hardware-Friendly and Search-Free Mixed-Precision Quantization** | -- | arxiv / 2025 | 2602.01027 | Fractional bit-widths transform discrete precision allocation into continuous optimization; unified GEMM kernel supporting arbitrary average bit-width |

### Additional Notable Work

| # | Title | Venue/Year | arxiv ID | Contribution |
|---|-------|------------|----------|-------------|
| 6 | **AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-Rank Adapters** | arxiv / 2025 | 2602.22268 | Two-phase coarse-to-fine evolutionary + Bayesian optimization; shows bit-width and LoRA rank interact non-trivially |
| 7 | **TAP: Training-Free Automatic Proxy Discovery for Mixed-Precision Quantization via LLMs** | arxiv / 2025 | 2512.07419 | LLM-driven evolutionary search for superior training-free sensitivity proxies |
| 8 | **AutoMixQ: Self-Adjusting Quantization for High Performance Memory-Efficient Fine-Tuning** | arxiv / Nov 2024 | 2411.13814 | Joint per-layer quantization + pruning + LoRA; Pareto-optimal memory/performance tradeoffs |

### SOTA Landscape: How Mixed-Precision Decisions Are Made

**Precision allocation granularity (finest to coarsest):**

1. **Per-block (sub-channel)**: ScaleBITS, SFMP -- block-wise bit allocation within weight matrices
2. **Per-channel**: MicroMix -- channel-wise MXFP4/6/8 assignment within a single linear layer
3. **Per-expert**: DeepSeek-V4 -- MoE experts quantized to different precisions based on importance
4. **Per-layer**: AMQ, KVTuner -- each transformer layer gets its own bit-width
5. **Per-module**: Attention in FP8, FFN in INT4 -- manual heuristics

**Decision mechanisms:**

| Strategy | Representative Work | Mechanism |
|----------|-------------------|-----------|
| **Sensitivity-based search** | KVTuner, ScaleBITS | Analyze Hessian spectrum, activation magnitudes, or attention patterns to identify sensitive components |
| **Evolutionary / Bayesian optimization** | AMQ, AutoQRA | Treat as combinatorial optimization; use evolutionary search or Bayesian optimization |
| **Learned proxies** | TAP, EMQ | Train lightweight models to predict quantization sensitivity, replacing hand-crafted metrics |
| **Continuous relaxation** | SFMP | Fractional bit-widths + gradient-based optimization |
| **Hardware co-design** | MicroMix | Precision decisions co-optimized with GEMM kernel design for Blackwell |

**Key findings across papers:**

- **FFN layers are more quantization-tolerant** than attention layers (consistent across multiple studies)
- **Down-projection layers** are typically the most sensitive
- **Early and late layers** often need higher precision than middle layers
- **KV cache precision** is more critical for long-context tasks than for short-context; KVTuner shows 3.25-bit mixed KV cache is near-lossless
- **Bit-width and adapter rank interact**: AutoQRA shows joint optimization of quantization precision and LoRA rank yields better Pareto frontiers than independent optimization

### Bottlenecks and Limitations

1. **Search cost**: Full combinatorial search over all layers/channels is infeasible (AMQ reports >10^100 configurations). All methods rely on approximations.
2. **Hardware fragmentation**: Mixed-precision kernels need hardware-aware design. MicroMix works on Blackwell but not on Hopper/A100. SFMP's unified GEMM kernel is promising but not yet production-hardened.
3. **Calibration data dependency**: The optimal mixed-precision configuration depends on calibration data distribution; configurations may not generalize across domains.
4. **Dynamic precision switching**: No current method supports runtime precision adaptation based on input difficulty -- all decisions are static.
5. **Interaction with speculative decoding and MoE routing**: Mixed-precision effects on these inference optimizations are underexplored.

### Hardware Match Assessment

| Hardware | Mixed-Precision Support |
|----------|------------------------|
| **B300 (Blackwell)** | Native MXFP4/6/8 in Tensor Cores (MicroMix target); NVFP4 + FP8 mixed (DeepSeek R1 deployment: W4A8) |
| **A100 (Ampere)** | INT8 + FP16 mixed only; no FP8/FP4; limited mixed-precision granularity |
| **H100/H200 (Hopper)** | FP8 + FP16/BF16 mixed; W8A8 supported; no native 4-bit |

---

## A3: Low-bit Impact on Emergent Abilities

### Key Papers (5)

| # | Title | Authors | Venue/Year | arxiv ID | 1-Sentence Contribution |
|---|-------|---------|------------|----------|------------------------|
| 1 | **Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study** | Liu, Liu, Gao et al. (Renmin Univ. & Alibaba) | LREC-COLING 2024 | 2307.08072 | First systematic study: 4-bit retains emergent abilities (ICL, CoT, IF); 2-bit catastrophic; FFN down-projection most sensitive; fine-tuning partially compensates |
| 2 | **Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression** | Dong, Tang, Liu et al. | ICML 2025 | 2505.19433 | Introduces ACBench for agentic capability evaluation; 4-bit preserves planning/tool-use but drops 10-15% on real-world tasks; DeepSeek-R1 is particularly compression-sensitive |
| 3 | **Does Quantization Affect Models' Performance on Long-Context Tasks?** | Mekala, Atmakuru, Song, Karpinska, Iyyer | EMNLP 2025 | 2505.20276 | Large-scale study (5 models, 5 methods, 26 languages, up to 128K tokens): 8-bit safe; 4-bit degrades progressively with context length; BNB-nf4 worst (up to 59% drop) |
| 4 | **A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models: An Experimental Analysis up to 405B** | Li et al. | IJCAI 2025 | 2409.11055 | Evaluates GPTQ, AWQ, SmoothQuant, FP8 across 7B-405B on 13 benchmarks; FP8 most robust; AWQ > GPTQ; larger models tolerate quantization better |
| 5 | **Evaluating Quantized Large Language Models** | Li et al. (Tsinghua & Infinigence-AI) | arxiv / 2024 | 2406.12928 | Multi-step reasoning and self-calibration more sensitive than instruction-following; common error modes: Incorrect Logic (~50%), Calculation Error (~20%) |

### SOTA Landscape: What Degrades and When

**Capability degradation hierarchy (most-to-least sensitive at 4-bit):**

| Rank | Capability | 4-bit Impact | 8-bit Impact | Notes |
|------|-----------|-------------|-------------|-------|
| 1 | **Long-context reasoning (>32K)** | Severe (10-59% drop) | <1% | Progressive degradation with length; BNB-nf4 worst |
| 2 | **Multilingual / Low-resource languages** | Major (up to 5x English drop) | ~1-2% | Non-English accuracy disproportionately affected |
| 3 | **Multi-step mathematical reasoning** | Moderate (7-15%) | <1% | CoT chains accumulate quantization noise |
| 4 | **Real-world agentic tasks** | Moderate (10-15%) | <2% | Tool use, planning, embodied tasks |
| 5 | **Factuality / Hallucination** | Moderate | <1% | Higher abstention rates under quantization |
| 6 | **Instruction following** | Minor (1-3%) | Negligible | Relatively robust |
| 7 | **In-context learning** | Minor (1-3%) | Negligible | Few-shot demonstrations help compensate |
| 8 | **Standard QA / Classification** | Minimal (<2%) | Negligible | Well-preserved |

**Key degradation patterns (2024-2025 consensus):**

1. **Error mode analysis** (arxiv 2406.12928): Under quantization, LLM reasoning errors shift toward Incorrect Logic (~50% of errors), Calculation Error (~20%), Condition Missing, and Copy Mistakes. This suggests quantization primarily degrades reasoning fidelity, not factual recall.
2. **Model-specific robustness**: Qwen-2.5 family is significantly more quantization-robust than Llama-3.1. Qwen-2.5 72B under BNB-nf4 remains stable; Llama-3.1 70B drops 32% on the same task (arxiv 2505.20276).
3. **Scaling laws**: Larger models tolerate lower precision better. Quantized 405B often outperforms FP16 70B on most benchmarks except hallucination and instruction-following (arxiv 2409.11055).
4. **DeepSeek-R1 sensitivity** (arxiv 2505.19433): Distilled reasoning models are particularly sensitive to compression -- they perform worse than base models on agentic tasks post-compression despite stronger reasoning pre-compression.
5. **Fine-tuning as mitigation**: LoRA fine-tuning on 2-bit models can recover performance to near-4-bit levels (arxiv 2307.08072). Parameter-efficient fine-tuning post-quantization is an effective recovery strategy.

### Bottlenecks and Limitations

1. **No standardized evaluation benchmark for quantized LLM capabilities**: ACBench (ICML 2025) is the first comprehensive benchmark targeting agentic capabilities specifically. Most studies use ad-hoc task collections.
2. **Long-context degradation is poorly understood theoretically**: Empirical evidence is strong (EMNLP 2025), but the mechanism (attention score collapse? KV cache noise accumulation? positional encoding distortion?) lacks theoretical treatment.
3. **Multilingual quantization gap is underexplored**: Only one paper (Mekala et al., EMNLP 2025) systematically evaluates across 26 languages. Tokenizer effects, embedding sensitivity, and language-specific outlier patterns need investigation.
4. **Safety and alignment under quantization**: Limited work on whether quantized models exhibit different refusal behavior, toxicity, or jailbreak susceptibility.
5. **Interaction with inference optimizations**: Quantization effects when combined with speculative decoding, KV cache compression, or prompt compression are not well-studied.

### Hardware Match Assessment

| Evaluation Need | B300 Suitability | A100 Suitability |
|----------------|-----------------|-----------------|
| Long-context (>32K) quantized evaluation | Excellent -- B300's 288GB VRAM enables full-precision baselines at scale | Limited -- VRAM constraints restrict model size |
| Multi-format comparison | Excellent -- native FP4, FP8, INT8 support | Partial -- no FP8/FP4 native, INT8 only |
| Agentic capability benchmarking | Good -- high throughput enables large-scale ACBench evaluation | Adequate |

---

## A4: B300 Native FP4 Hardware-Software Co-design Opportunities

### Key Papers and Technical Reports (5)

| # | Title | Authors / Source | Venue/Year | ID | 1-Sentence Contribution |
|---|-------|---------|------------|-----|------------------------|
| 1 | **SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training** | Zhang, Wei, Zhang et al. (Tsinghua) | NeurIPS 2025 | arxiv 2505.11594 | First FP4 attention kernel on Blackwell; 1,038 TOPS on RTX 5090, ~5x FlashAttention2; two-level scaling for softmax; 2.4-3x end-to-end video generation speedup |
| 2 | **Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II)** | LMSYS | Technical Report, Sep 2025 | -- | 3.8x prefill, 4.8x decode vs H100 using NVFP4 MoE + FP8 attention; W4A8 deployment recipe for DeepSeek-R1/V3 on Blackwell |
| 3 | **Pushing Intelligence to 4-bit** | NVIDIA Research | Technical Blog, 2025 | -- | Comprehensive overview of NVFP4 format design: two-level scaling, 88% lower error than power-of-two scaling; covers QAT, PTQ, KV cache, attention |
| 4 | **Scaling NVFP4 Inference for FLUX.2 on NVIDIA Blackwell Data Center GPUs** | NVIDIA | Technical Blog, Jan 2026 | -- | 10.2x speedup over H200 using stacked NVFP4 + TeaCache + CUDA Graphs + torch.compile + multi-GPU; production recipe |
| 5 | **Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference** | -- | arxiv / May 2026 | 2605.00519 | Consumer Blackwell (RTX 5090) NVFP4 falls back to Marlin BF16; no native FP4 compute advantage; 1.6x over BF16 via PyTorch backend; Apple M3 Ultra 23x more energy-efficient |

### Additional Notable Resources

| # | Title | Source | Contribution |
|---|-------|--------|-------------|
| 6 | **How DeepInfra Built on NVIDIA's Inference Stack** | DeepInfra Blog, 2025 | 4x infrastructure reduction: workload requiring 4x H200 runs on single B300 at higher tok/s with NVFP4 |
| 7 | **NVFP4 on DGX B200 (Getting Started Guide)** | Macnica / NVIDIA | Technical walkthrough of NVFP4 PTQ pipeline: TransformerEngine + TensorRT Model Optimizer |
| 8 | **CUDA: native 4-bit float quant (Blackwell PP +40%)** | llama.cpp PR #23572 | Experimental native NVFP4 support in llama.cpp; +40% prompt processing speedup |
| 9 | **canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP** | Hugging Face | Community NVFP4 quantized DeepSeek-V4-Flash checkpoint |

### SOTA Landscape: What B300 Native FP4 Enables

**The full FP4 inference stack on Blackwell (B200/B300):**

```
Layer                | Precision   | Hardware Path              | Speedup vs FP8
---------------------|-------------|----------------------------|---------------
Weights (GEMM)       | NVFP4       | tcgen05.mma (native FP4)   | ~1.9x
Activations (GEMM)   | NVFP4/FP8   | tcgen05.mma (native)       | ~1.9x (FP4), ~1x (FP8)
KV Cache             | NVFP4       | FP4-to-FP8 HW dequant      | ~50% memory reduction
Attention (QK, PV)   | NVFP4       | SageAttention3             | ~5x vs FlashAttention2
MoE dispatch         | NVFP4       | DeepEP fused quantize      | ~50% communication reduction
```

**What was previously hard that B300 makes practical:**

| Capability | Pre-Blackwell Status | B300 Status |
|-----------|---------------------|-------------|
| W4A4 inference (weights + activations) | Required software emulation (QuaRot/SpinQuant with CUDA dequant, ~85% overhead) | Native Tensor Core execution with fused dequant in MMA instructions |
| FP4 KV cache | INT4/INT2 KV cache via software dequant, high overhead | Hardware FP4-to-FP8 dequant path with <2% overhead |
| FP4 attention | Not feasible -- attention matmuls require higher precision | SageAttention3 delivers 1,038 TOPS native |
| End-to-end FP4 training | MXFP4 diverges early without mixed precision | NVFP4 pretraining with 36% fewer tokens than MXFP4; DeepSeek-V4 trains expert weights natively in FP4 |
| Large-scale MoE FP4 deployment | Memory-bound, weight offloading required | DeepSeek V4 on 4x B300: 1,577 tok/s (c=16), within 0.4 points of BF16 on AIME |

**Production deployment performance data:**

| Configuration | Tokens/sec/GPU | Context |
|--------------|----------------|---------|
| Llama-3.3-70B, B200 TP1 FP4, 1024/1024 | 6,943 | 2.6x vs H200 FP8 (2,637) |
| Llama-3.3-70B, B300 TP1 FP4, 1024/1024 | 8,196 | 3.1x vs H200 FP8 |
| DeepSeek-R1, B200 FP4, 1024/1024 | 5,757 | 3.3x vs H200 FP8 (1,724) |
| DeepSeek-R1, B300 FP4, 1024/8192 | 6,235 | 4.7x vs H200 FP8 (1,335) |
| DeepSeek V4, 4x B300 NVFP4 chat (c=1) | 278.68 tok/s per request | AIME pass@1 = 96.0% vs BF16 96.15% |

**Cost economics (FP4 vs FP8 for 70B-class model):**

| GPU Config | Tokens/sec | Cost/1M tokens |
|-----------|-----------|----------------|
| H100 SXM, FP8 | 3,066 | $0.182 |
| H200 SXM, FP8 | 4,374 | $0.288 |
| B200, FP4 | 12,841 | $0.130 |

**Ecosystem maturity (as of mid-2026):**

| Framework | NVFP4 Support Level |
|-----------|-------------------|
| TensorRT-LLM | Production (B200/B300); most mature |
| vLLM (>=0.18.0) | Production for MoE + dense NVFP4; FlashInfer FP4 kernel |
| SGLang | Production; up to 4x throughput vs Hopper for DeepSeek-R1 |
| NVIDIA ModelOpt | PTQ calibration pipelines for NVFP4 |
| llama.cpp | Experimental (N4_0 quant, +40% prompt processing) |
| HuggingFace Transformers | Community checkpoints; no native integration |

### Bottlenecks and Limitations

1. **Better scale search algorithms**: Current block-wise scaling uses simple max-based or MSE-optimal approaches. Hadamard transforms and learned scaling could further reduce quantization error.
2. **KV-cache dequant efficiency in decode**: While B300 has hardware FP4-to-FP8 dequant, fusing this into the autoregressive decode loop with zero bubbles remains challenging.
3. **FP4 attention quality gap**: SageAttention3's two-level stretch quant narrows but does not close the accuracy gap. Attention remains higher-risk than GEMM for FP4.
4. **PTQ vs. QAT gap**: Post-training NVFP4 recipes still lag behind quantization-aware training. Cheaper PTQ that approaches QAT quality is an open problem.
5. **Consumer Blackwell gap**: RTX 5090 stores NVFP4 but cannot compute natively. This limits research and development to datacenter GPU owners.
6. **Software optimization lag**: RTX 4090 outperforms RTX 5090 by 2.2x in TTFT for some workloads due to Blackwell software immaturity (as of May 2026).
7. **Pre-quantized weight ecosystem**: Quantizing large models to NVFP4 can exceed RTX 5090 VRAM. Pre-quantized weight distribution (NVIDIA HuggingFace repos) is needed but not yet comprehensive.

### Hardware Match Assessment

| Aspect | B300 (Blackwell) | A100 (Ampere) |
|--------|-----------------|---------------|
| **Native FP4 compute** | Yes (tcgen05.mma, 4x FP16 throughput) | No |
| **FP4 attention** | SageAttention3, ~5x FlashAttention2 | Software emulation only |
| **FP4 KV cache** | HW-accelerated FP4-to-FP8 dequant | INT4 via CUDA dequant (~85% overhead) |
| **FP4 training** | Viable (DeepSeek-V4 QAT, LongLive-2.0 end-to-end) | Not practical |
| **MX format hardware** | Native MXFP4/6/8 in Tensor Cores | Emulation only |
| **VRAM** | 288 GB HBM3e | 40/80 GB HBM2e |
| **Ecosystem maturity** | Maturing rapidly (TRT-LLM, vLLM, SGLang) | Mature but legacy ceiling at INT8/FP16 |
| **Role in FP4 research** | Primary platform for FP4 HW-SW co-design | Baseline comparison (INT8 ceiling) |

---

## Cross-Cutting Themes and Research Opportunities

### 1. The Format War: NVFP4 vs MXFP4 vs INT4

NVIDIA is betting on NVFP4 as a proprietary format, while the OCP MX standard offers an open alternative. The INT vs FP study (2510.25602) challenges both by showing INT8 wins at 8-bit. The key open question: will the industry converge on a single 4-bit standard, or will fragmentation persist?

### 2. The Mixed-Precision Sweet Spot

The most impactful near-term direction appears to be **W4A8 with mixed-precision KV cache**:
- Weights: NVFP4 (B300 native)
- Activations: FP8 (B300 native via FP8 Tensor Cores)
- KV Cache: KVTuner-guided mixed precision (3-4 bits)
- Attention: SageAttention3 NVFP4 for long sequences, FP8 for short

This configuration is fully hardware-accelerated on B300 and achieves near-FP8 accuracy at ~4-bit weight memory.

### 3. The Long-Context Quantization Gap

EMNLP 2025 findings that 4-bit degrades progressively with context length are practically critical. Understanding the mechanism (attention score collapse? positional encoding distortion?) and designing context-length-aware quantization is an open research direction.

### 4. B300-Specific Research Opportunities

- **FP4-native training recipes**: DeepSeek-V4 shows FP4 QAT is viable. Generalizing this to arbitrary model architectures.
- **FP4 speculative decoding**: No published work on combining FP4 inference with speculative decoding.
- **Multi-node FP4 MoE**: Scaling FP4 MoE beyond single-node (NVLink domain) with FP4-reduced cross-node communication.
- **Dynamic precision inference**: Adapting precision per-token based on difficulty (easy tokens at 4-bit, hard tokens at 8-bit).

---

## References (Chronological by Sub-question)

### A1 References
- Chen & Wu et al. "INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats." arxiv:2510.25602, Oct 2025.
- Lee et al. "MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving." MICRO 2025, arxiv:2510.14557.
- Lee et al. "AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference." ACL 2025 Findings, arxiv:2411.09909.
- Sharify et al. "Post Training Quantization of LLMs with Microscaling Formats." NeurIPS ENLSP Workshop 2024, arxiv:2405.07135.
- Liu et al. "VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs." EMNLP 2024, arxiv:2409.17066.
- Ashkboos et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." NeurIPS 2024, arxiv:2404.00456.
- Liu et al. "SpinQuant: LLM Quantization with Learned Rotations." ICLR 2025, arxiv:2405.16406.

### A2 References
- Liu et al. "MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for LLMs." ICLR 2026, arxiv:2508.02343.
- Lee et al. "AMQ: Enabling AutoML for Mixed-Precision Weight-Only Quantization of LLMs." arxiv:2509.12019, Sep 2025.
- Li et al. "KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization." ICML 2025, arxiv:2502.04420.
- "ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs." arxiv:2602.17698, 2025.
- "SFMP: Fine-Grained, Hardware-Friendly and Search-Free Mixed-Precision Quantization." arxiv:2602.01027, 2025.
- "AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-Rank Adapters." arxiv:2602.22268, 2025.
- Kang et al. "TAP: Training-Free Automatic Proxy Discovery for Mixed-Precision Quantization via LLMs." arxiv:2512.07419, 2025.
- "AutoMixQ: Self-Adjusting Quantization for High Performance Memory-Efficient Fine-Tuning." arxiv:2411.13814, Nov 2024.
- Gong et al. "A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms." arxiv:2409.16694, Sep 2024.

### A3 References
- Liu et al. "Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study." LREC-COLING 2024, arxiv:2307.08072.
- Dong et al. "Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression." ICML 2025, arxiv:2505.19433.
- Mekala et al. "Does Quantization Affect Models' Performance on Long-Context Tasks?" EMNLP 2025, arxiv:2505.20276.
- Li et al. "A Comprehensive Evaluation of Quantized Instruction-Tuned LLMs: An Experimental Analysis up to 405B." IJCAI 2025, arxiv:2409.11055.
- Li et al. "Evaluating Quantized Large Language Models." arxiv:2406.12928, 2024.
- Yazan et al. "The Impact of Quantization on Retrieval-Augmented Generation: An Analysis of Small LLMs." SIGIR 2024 Workshop, arxiv:2406.10251.

### A4 References
- Zhang et al. "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training." NeurIPS 2025, arxiv:2505.11594.
- LMSYS. "Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II)." lmsys.org/blog/2025-09-25-gb200-part-2, Sep 2025.
- NVIDIA Research. "Pushing Intelligence to 4-bit." research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/, 2025.
- NVIDIA Technical Blog. "Scaling NVFP4 Inference for FLUX.2 on NVIDIA Blackwell Data Center GPUs." developer.nvidia.com, Jan 2026.
- "Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference." arxiv:2605.00519, May 2026.
- DeepInfra. "How DeepInfra Built on NVIDIA's Inference Stack and Why It Paid Off." deepinfra.com/blog, 2025.
- Scaleway. "Understanding the NVIDIA FP4 Format." scaleway.com/en/docs/gpu/reference-content/understanding-nvidia-fp4/, 2025.
- Lee et al. "Is Finer Better? The Limits of Microscaling Formats in Large Language Models." ICLR 2026, IBM Research.
