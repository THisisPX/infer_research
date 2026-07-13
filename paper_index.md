# 完整文献索引: LLM 量化推理 (2024-2026)

**整理日期:** 2026-07-10
**来源:** 三轮扫描 + 两轮深挖, 共 ~100 篇

---

## 1. 量化方法与格式 (Quantization Methods & Formats)

### 1.1 低比特格式

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 1 | **INT v.s. FP: A Comprehensive Study of Fine-Grained Low-bit Quantization Formats** (Chen, Wu et al., HKU & ByteDance) | arXiv, Oct 2025 | 2510.25602 | 系统性对比 INT vs FP 格式在不同粒度下的表现; 发现 MXINT8 > MXFP8 at 8-bit, 但 NVFP4 > NVINT4 at 4-bit ("crossover phenomenon") |
| 2 | **MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving** (Lee et al., SNU) | MICRO 2025 | 2510.14557 | 将 block-max 的冗余 exponent 位转用为额外 mantissa; 比 MXFP4 精度提升 42.15% |
| 3 | **AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference** (Lee et al.) | ACL 2025 Findings | 2411.09909 | 非对称 microscaling 同时抑制 outlier 和捕获 group-wise asymmetry; 无需 calibration |
| 4 | **Post Training Quantization of LLMs with Microscaling Formats** (Sharify et al., Meta/AMD) | NeurIPS ENLSP Workshop 2024 | 2405.07135 | SmoothQuant + AWQ + GPTQ 与 MX format 结合; MXINT W4A8 几乎无损 |
| 5 | **VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs** (Liu et al., Microsoft/USTC) | EMNLP 2024 | 2409.17066 | 向量量化推到 2-bit; channel-independent second-order optimization; SOTA 精度, 量化时间仅 10-18% |
| 6 | **AQLM: Extreme Compression of LLMs via Additive Quantization** (Egiazarian et al.) | ICML 2024 | 2401.06118 | Additive 量化; 将权重表示为多个 codebook 向量的和; 2-bit 下 SOTA |
| 7 | **ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization** (Liu et al., Meta) | NeurIPS 2025 | 2502.02631 | Sub-4-bit (ternary, 2-bit, 3-bit) 在 Pareto frontier 上可超越 4-bit; 发现 2-3 bit 处的 learning transition |

### 1.2 旋转基量化 (Rotation-Based)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 8 | **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (Ashkboos et al., ETH/Microsoft/ISTA) | NeurIPS 2024 | 2404.00456 | 首个端到端 W4A4KV4 via random Hadamard rotations; LLaMA-2 70B 仅 +0.47 PPL |
| 9 | **SpinQuant: LLM Quantization with Learned Rotations** (Liu et al., Meta FAIR) | ICLR 2025 | 2405.16406 | Cayley-optimized learned rotation matrices; 比 QuaRot 减少 45.1% gap |
| 10 | **FlatQuant: Flatness Matters for LLM Quantization** (Sun et al., Huawei/Tsinghua) | ICML 2025 | 2410.09426 | Learnable Kronecker 分解仿射变换摊平分布; LLaMA-3-70B W4A4 仅 -0.94% |
| 11 | **DuQuant: Distributing Outliers via Dual Transformation** (Lin et al., UCAS/Tsinghua) | NeurIPS 2024 Oral | 2406.01721 | Dual transformation (rotation + zigzag permutation) 针对 Normal 和 Massive 两种 outlier |
| 12 | **PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers** (Chen et al.) | arXiv, Oct 2024 | 2410.05265 | 首个 static per-tensor 量化超越 dynamic per-token; outlier token prefixing in KV cache |
| 13 | **DFRot: Refined Hadamard Rotation for LLM Quantization** (Xiang & Zhang) | COLM 2025 | 2412.00648 | 改进的 Hadamard 旋转; +0.98 PPL over QuaRot |
| 14 | **CBQ: Cross-Block Quantization for LLMs** (Ding et al., USTC/Huawei) | ICLR 2025 Spotlight | 2312.07950 | Cross-block reconstruction + LoRA-Rounding 捕获跨 block 依赖; >99% 性能保留 |
| 15 | **LoPRo: Low-rank Projection with Block-wise Hadamard Permutation** (Gu et al.) | arXiv, Jan 2026 | 2601.19675 | 块级 Walsh-Hadamard 置换 + 低秩; Mixtral 上 2-bit PPL 降低 0.4 |
| 16 | **SmoothRot: Training-time Channel Scaling for Rotation** (2026) | arXiv | 2606.09927 | 训练时学习 channel scaling 用于后续旋转量化; 密集 LLM |
| 17 | **ReSpinQuant: Revisiting SpinQuant** (2026) | arXiv | 2604.11080 | SpinQuant 的改进版; 密集 LLM |
| 18 | **OptRot** | arXiv, Dec 2025 | — | 发现 W4A4 中旋转对权重和激活的优化目标存在 tradeoff |

### 1.3 混合精度策略

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 19 | **MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for LLMs** (Liu et al., Tianjin Univ.) | ICLR 2026 | 2508.02343 | Co-designed algorithm + kernel 支持单层内任意 MXFP4/6/8 通道组合; 平均 ~5-bit 近 FP16 精度 |
| 20 | **AMQ: Enabling AutoML for Mixed-Precision Weight-Only Quantization** (Lee et al.) | arXiv, Sep 2025 | 2509.12019 | 自动化逐层 bit allocation; 搜索空间剪枝 + proxy quality prediction; 达 Pareto frontier |
| 21 | **ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs** | arXiv, 2025 | 2602.17698 | Block-wise bit allocation via bi-directional channel sensitivity; +36% over uniform |
| 22 | **SFMP: Fine-Grained, Hardware-Friendly and Search-Free Mixed-Precision Quantization** | arXiv, 2025 | 2602.01027 | Fractional bit-widths; 统一 GEMM kernel 支持任意平均位宽 |
| 23 | **AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-Rank Adapters** | arXiv, 2025 | 2602.22268 | 进化 + Bayesian optimization 联合优化 bit-width 和 LoRA rank |
| 24 | **TAP: Training-Free Automatic Proxy Discovery for Mixed-Precision Quantization via LLMs** (Kang et al.) | arXiv, 2025 | 2512.07419 | LLM 驱动的进化搜索发现更好的 training-free sensitivity proxy |
| 25 | **AutoMixQ: Self-Adjusting Quantization for High Performance Memory-Efficient Fine-Tuning** | arXiv, Nov 2024 | 2411.13814 | 联合 per-layer quantization + pruning + LoRA; Pareto-optimal |

### 1.4 权重+激活联合量化 (Joint W-A)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 26 | **SmoothQuant** (Xiao et al., MIT) | ICML 2023 | 2211.10438 | Per-channel scaling 将量化难度从 activation 迁移到 weight; W8A8 几乎无损 |
| 27 | **MergeQuant: Accurate 4-bit Static Quantization** (Wang et al., BUPT) | arXiv, Mar 2025 | 2503.07654 | Per-channel static quantization + QSM 融合进 normalization; 2.06× speedup |
| 28 | **LO-BCQ: Block Clustered Quantization for 4-bit LLM Inference** (Elangovan et al., NVIDIA/Purdue) | TMLR 2025 | 2502.05376 | Block 分解 + clustering + per-cluster Lloyd-Max codebooks; 无微调 |
| 29 | **Atom: Low-bit Quantization for Efficient LLM Serving** (Zhao et al.) | MLSys 2024 | 2310.19102 | W4A4 mixed-precision + fused CUDA kernels; 7.7× throughput |
| 30 | **TaCQ: Task-Circuit Quantization** (Xiao et al.) | COLM 2025 | 2504.07389 | Task-specific weight circuits for mixed-precision; 3.1 bits 恢复 96% MMLU |
| 31 | **QuEST: Quantized Efficient Supervised Training** | — | — | Hadamard + trust gradient for QAT W4A4; Pareto-optimal |
| 32 | **LLM-QAT** | — | — | Full QAT + data distillation for W4A4 |

### 1.5 训练相关 (Scaling Laws, Training Quantization)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 33 | **Scaling Laws for Precision** (Kumar et al.) | arXiv, 2024 | 2411.04330 | Compute-optimal training precision ≈ 7-8 bits; 465+ runs up to 1.7B params |
| 34 | **Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws** (Ouyang et al.) | ACL 2025 | 2411.17691 | 量化误差 ∝ 1/N^0.23 但 ∝ D^0.53; **训练更充分的模型更不耐量化** |
| 35 | **ECO: Error-Compensated Quantization-Aware Training** | ICML 2026 | — | 无 FP32 master copy 的量化训练; 已在 SMoE 模型上测试 |
| 36 | **OSP: Outlier-Safe Pre-Training** (Park et al., Korea Univ.) | ACL 2025 | 2506.19697 | Muon optimizer + Single-Scale RMSNorm 消除 training 中的 outlier (kurtosis 0.04 vs 1818.56) |

### 1.6 量化能力评估

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 37 | **Do Emergent Abilities Exist in Quantized LLMs** (Liu et al., Renmin & Alibaba) | LREC-COLING 2024 | 2307.08072 | 首个系统研究: 4-bit 保留 emergent abilities; 2-bit 灾难性 |
| 38 | **Can Compressed LLMs Truly Act? Agentic Capabilities in LLM Compression** (Dong et al.) | ICML 2025 | 2505.19433 | ACBench 评估; 4-bit 保留 planning/tool-use 但 real-world 降 10-15%; **DeepSeek-R1 特别敏感** |
| 39 | **Does Quantization Affect Long-Context Tasks?** (Mekala et al.) | EMNLP 2025 | 2505.20276 | 5 models × 5 methods × 26 languages × 128K tokens; 4-bit 随长度逐步退化; BNB-nf4 最差 (59% drop) |
| 40 | **A Comprehensive Evaluation of Quantized Instruction-Tuned LLMs up to 405B** (Li et al.) | IJCAI 2025 | 2409.11055 | FP8 最稳健; AWQ > GPTQ; 更大模型更耐量化 |
| 41 | **Evaluating Quantized Large Language Models** (Li et al., Tsinghua) | arXiv, 2024 | 2406.12928 | 多步推理和自我校准最敏感; 错误模式: Incorrect Logic (~50%), Calculation Error (~20%) |
| 42 | **Quantitative Analysis of DeepSeek Model Quantization** (China Unicom) | arXiv, 2025 | 2505.02390 | Q4_K_M: 0% drop; Q2_K_L (2.91 bits): 8.91% drop; AIME math 39.2→15.41 |

---

## 2. MOE 量化 (Mixture-of-Experts Quantization)

### 2.1 MOE 专用量化方法

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 43 | **QuantMoE-Bench** (Li et al.) | arXiv, Jun 2024 | 2406.08155 | 首个系统 MoE PTQ benchmark; data-driven mixed-precision allocation per component type |
| 44 | **MC-MoE: Mixture Compressor for MoE LLMs** (He et al.) | ICLR 2025 | 2410.06270 | ILP 基混合精度 + 在线动态 pruning; **2.54-bit 平均仅 3.8% loss** |
| 45 | **MC#: Quantization + Gumbel-Softmax Pruning for MoE** (Huang et al.) | IEEE TPAMI 2026 | — | 联合量化 + Gumbel-Softmax pruning; 6.2× weight reduction |
| 46 | **EAC-MoE: Expert-Selection Aware Compression for MoE** | ACL 2025 | 2508.01625 | **发现 "expert shift" 问题**; TopK-MSE router calibration + dynamic expert pruning; 2.06-3.03 bit |
| 47 | **MoEQuant: Expert-Balanced Self-Sampling + Affinity-Guided Quantization** (Hu et al.) | ICML 2025 | 2505.03804 | 解决校准不均衡: Expert-Balanced Self-Sampling + Affinity-Guided Quantization |
| 48 | **MoE-I^2: Inter-Expert Pruning + Intra-Expert Low-Rank Decomposition** | arXiv, Nov 2024 | 2411.01016 | Genetic search 基非均匀 expert pruning + 低秩分解; 联合 pruning+量化 |
| 49 | **PuzzleMoE: Sparse Expert Merging + Quantization** | ICML 2026 | 2511.04805 | Sparse expert merging via similarity + saliency masks + 3-bit group quantization; 4.8× compression |
| 50 | **MxMoE: Mix-precision Group-GEMM for MoE** (Duanmu et al.) | ICML 2025 | — | Linear block-level sensitivity + 自定义 Group-GEMM 内核; W2.25-W5A5 |
| 51 | **KBVQ-MoE: KLT-guided SVD + Vector Quantization for MoE** | ICLR 2026 | — | KLT 引导 SVD 提取共享组件 + VQ; 跨专家共享 dominant SVD |
| 52 | **CAMERA: Micro-Expert Compression** | AAAI 2026 | — | 微专家粒度 as 压缩单元; 跨矩阵混合精度量化 |
| 53 | **GEMQ: Global ILP Expert Bit-width Allocation + Router Fine-tuning** (Deng et al.) | ICML 2026 | 2605.23078 | 全局 ILP 专家位宽分配 + router fine-tuning; 1.5-4 bit; **测量: 1.5-bit 下 41.31% token 换专家** |
| 54 | **AlphaQ: Heavy-Tail Self-Regularization Theory** | arXiv, Jun 2026 | 2606.04980 | 基于重尾光谱理论的无 calibration 位宽分配; 3.5-bit 压缩 >4× |
| 55 | **BitsMoE: SVD Shared Basis + Expert-Specific Factors** | arXiv, Jun 2026 | 2606.00079 | SVD 分解为共享基 + 专家特定因子; ILP 混合精度; Qwen3-30B 2-bit 提升 27.83pp |
| 56 | **TileQ: 2D Tiled Low-Rank Quantization for MoE** | ICML 2026 | — | 2D 分块低秩量化; 跨输入/输出维度共享低秩因子; 内存减少 10× |
| 57 | **VSRAQ: Value-Structure Alignment for Router-Consistent PTQ** (Park et al.) | arXiv, Jun 2026 | 2606.05688 | Value + Structure alignment 保护路由一致性; TopK-MSE + Rank loss + Boundary margin |
| 58 | **MoBiE: First MoE Binarization Framework** | ACL 2026 Findings | — | 首个 MoE 二值化 (1-bit); 联合 SVD + input nullspace constraint; PPL 降低 52.2% |
| 59 | **REAP: Router-Weighted Expert Pruning** | ICLR 2026 | — | 路由器门值 + activation norm 作为 saliency; 可与 GGUF 量化结合 |
| 60 | **Attribution-Guided Pruning + Quantization for MoE** | arXiv, Jun 2026 | 2606.18304 | 专家内 channel 级剪枝 + 4-bit 量化; 覆盖最大化 formulation |
| 61 | **TD-MoE: Tensor Decomposition for MoE Compression** | ICLR 2026 | — | 张量分解 for MoE compression |
| 62 | **VEQ: Modality-Adaptive Quantization for MoE VLMs** | ICML 2026 | — | 模态自适应量化 for MoE VLM |
| 63 | **Nota AI INT4 MoE** (Workshop) | ICML Workshop 2026 | — | INT4 + 15% expert pruning; router-consistent + downstream-error-aware |

### 2.2 MOE 旋转/平滑量化 (Rotation/Smoothing for MoE)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 64 | **CodeQuant: Learned Cayley Rotation + Clustering for MoE W4A4** (Yin et al., NYU) | **ICLR 2026** | 2604.10496 | 可学习 Cayley 旋转 + 自适应权重聚类 + KL router loss; MoE-native W4A4; CPU 加速 4.15×; **最接近 P0-1 的工作** |
| 65 | **EAQuant: Expert-Aware Smoothing + Router KL Alignment** (Fu et al., Huawei/PKU) | arXiv, Jun 2025 | 2506.13329 | 专家感知平滑聚合 + router KL divergence 对齐 + 校准平衡; W4A4/W3A4/W3A3/W2A4 SOTA |
| 66 | **KurTail: Kurtosis-Optimized Learnable Rotation for 4-bit** (Akhondzadeh et al.) | EMNLP 2025 Findings | 2503.01483 | 基于峰度的可学习旋转; Mixtral 上测试; 比 QuaRot 好 13.3% |
| 67 | **HLWQ: Per-Expert Walsh-Hadamard + Lloyd-Max Quantization** | arXiv, 2025 | — | 每专家 Walsh-Hadamard + Lloyd-Max; 量化误差 -54%; A100 上 50 秒完成 |
| 68 | **Stay Rotated: Random Hadamard + Universal Codebook** | TMLR 2025 | — | Training-free; 计算保留在旋转域; Llama-4-Scout (109B MoE, 768 experts) |
| 69 | **ParoQuant: Pairwise Rotation Quantization for MoE** | ICLR 2026 | — | 成对旋转抑制权重 outlier; Qwen3/3.5/3.6 MoE 上测试; 已发布 checkpoint |
| 70 | **ButterflyMoE: Per-Expert Butterfly Rotation + Shared Ternary Basis** | arXiv, Jan 2026 | 2601.13563 | 每专家独立 butterfly 旋转 + 共享三元基; 专家压缩, 非传统量化 |
| 71 | **Spectral Influence Rotations** (MoE-aware absorption fix) | arXiv, 2026 | 2605.25203 | 修正 MoE 中旋转吸收的谱效应 |

### 2.3 ExpertQuant / RouteQuant (在审)

| # | 论文 | 状态 | 一句话 |
|---|------|------|--------|
| 72 | **ExpertQuant** | OpenReview (在审) | 专家感知 scale + Rank-Aware Jaccard Loss + Gap Hinge Loss; W4A4/W4A8; 最接近量化 router 的工作 |
| 73 | **RouteQuant** | TMLR (在审) | "Beyond Freezing the Router" — 量化 router 以补偿 expert distortion |

### 2.4 MOE 理论/分析

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 74 | **Efficient Quantization of MoE with Theoretical Guarantees** (Chowdhury et al.) | ICLR 2026 | — | 基于训练动态的专家位宽理论保证; 量化误差 ∝ router ℓ₂ 范数变化 |
| 75 | **Depth Registers** (2025) | arXiv, Apr 2025 | — | 理论证明 Hadamard 旋转无法约束 SwiGLU 双线性乘积的尾部行为 |
| 76 | **VQMoE: Discrete Representation Learning for Expert Routing** (Do et al.) | arXiv, Nov 2024 | 2411.19402 | VQ 学习离散输入表示指向 expert; bypass 连续 router; 理论证明最优性 |

---

## 3. 推理系统与内核 (Inference Systems & Kernels)

### 3.1 低比特推理内核

| # | 论文/项目 | 会议/年 | ID | 一句话 |
|---|----------|---------|-----|--------|
| 77 | **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference** (Frantar et al., ISTA/Neural Magic) | PPoPP 2025 | 2408.11743 | 高度优化 FP16×INT4 GEMM kernel; vLLM 端到端 2.8× 加速 |
| 78 | **FlashInfer** (UW/CMU) | 2025 | 2501.01005 | FP8/NVFP4/MXFP4/INT4 attention kernels; H100 上 1.2-1.3 PFLOPs/s |
| 79 | **BitBLAS/Ladder** (Microsoft) | OSDI 2024 | — | 硬件感知低比特 GEMM; INT4/INT2/INT1; vs cuBLAS 最高 8× 加速 |
| 80 | **Alpha-MoE: Megakernel for Tensor-Parallel MoE Inference** (Aleph Alpha) | Tech Report, 2025 | — | 融合 W8A8 MoE kernel; 6 个操作合为 1 个 persistent kernel; 200% vs Triton |
| 81 | **DeepGEMM** (SGLang/DeepSeek) | Open Source, 2025 | — | H100 FP8 GEMM for DeepSeek 的 128×128 block-wise FP8 scaling |
| 82 | **Cohere W4A8 GEMM Kernel** | vLLM, 2026 | — | INT4 权重 × FP8 激活 for Hopper; CUTLASS LUT; TTFT 快 58% |
| 83 | **DeepSeek Tile Kernels** | Open Source, 2025 | — | Per-token/per-block/per-channel FP8/FP4/E5M6 quantization; fused SwiGLU+quantize |

### 3.2 KV-Cache 量化

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 84 | **KIVI: Tuning-Free Asymmetric 2-bit Quantization for KV Cache** (Liu et al.) | ICML 2024 | 2402.02750 | 首个无微调 2-bit KV cache 量化; key 按 channel, value 按 token; 2.6× peak memory reduction |
| 85 | **KVQuant: Towards 10M Context Length LLM Inference** (Hooper et al., Berkeley) | NeurIPS 2024 | 2401.18079 | 3-bit KV cache via per-channel Key quant + Pre-RoPE quant + non-uniform datatypes |
| 86 | **CacheGen: KV Cache Compression and Streaming** (Liu et al.) | SIGCOMM 2024 | 2310.07240 | 3.5-4.3× KV compression with adaptive bandwidth encoding |
| 87 | **KV Cache is 1 Bit Per Channel** | NeurIPS 2024 | — | 耦合 KV channels 做联合熵量化; 1 bit/channel |
| 88 | **TurboQuant: Calibration-Free Random Rotation + Scalar Quantization** | 2025 | — | 无 calibration 随机旋转 + 标量量化; KV 压缩 4-7× |
| 89 | **KV Pareto: Joint AWQ Weights + KV Quantization + Prefill Chunking** (Gokhale et al., AMD) | 2025 | 2512.01953 | 联合优化权重 + KV 量化 + prefill chunking; 总内存减少 68-78% |
| 90 | **MoE-nD: Per-Layer Mixed KV Compression via Router** | arXiv, 2026 | 2604.17695 | MoE-style router 选择每层 KV 压缩策略; 14× compression; AIME +6-27 分 |
| 91 | **TurboQuant-MoE: K3/V2-bit with NashMoE Router** | ICLR 2026 | — | NashMoE router + Markov trajectory predictor + PID VRAM controller; 128K context 8.53× compression |

### 3.3 量化推理显存与调度

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 92 | **PMPD: Progressive Mixed-Precision Decoding** (Chen et al.) | ICLR 2025 | — | 预填充用高精度, 解码过程逐步降精度; 1.4-12.2× vs FP16 |
| 93 | **Palu: Low-Rank Projection + KV Quantization** | ICLR 2025 | 2407.21118 | KV 低秩投影 + 量化; RoPE 下 KV 压缩 50% + 1.89× 加速 |
| 94 | **HACK: Homomorphic Attention over Quantized KV** | 2025 | 2502.03589 | 量化 KV 上的同态注意力计算 — 无需反量化; 作业完成时间 -70.9% |
| 95 | **LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference** (Liu et al.) | 2024 | — | 模块化系统感知量化框架; 融合 CUDA kernel + NCCL sync; 单节点多 GPU + 多节点 + 边缘 |

### 3.4 MOE 推理系统 (卸载 / 调度 / Dynamic Precision)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 96 | **HOBBIT: Mixed-Precision Expert Offloading** (Tang et al.) | arXiv, Nov 2024 | 2411.01433 | 三级层次 (token/layer/sequence) 混合精度卸载; 9.93× decode speedup vs SOTA |
| 97 | **DyMoE: Dynamic Expert Orchestration with Mixed-Precision** | arXiv, 2026 | 2603.19172 | 运行时动态混合精度; 基于专家重要性和深度依赖; TTFT 减少 3.44-22.7× |
| 98 | **D2MoE: Matryoshka Nested Bit-width + Token-Adaptive Selection** | Mobicom 2025 | 2504.15299 | Matryoshka 嵌套位宽 + token-adaptive 选择; 峰值内存 -53% |
| 99 | **MoE-SpeQ: Speculative Quantized Decoding with Expert Prefetching** (Wang et al.) | 2025 | 2511.14102 | 4-bit draft model for speculative expert prefetching; 90.9% fidelity; 2.34× speedup |
| 100 | **Fate: Cross-Layer Gate Prediction** | 2025 | 2502.12224 | Cross-layer gate prediction 达 99% expert cache hit; prefill 4.5×, decode 4.1× |
| 101 | **MoE-APEX: Adaptive Precision Expert Offloading** | ASPLOS 2026 | — | 缓存未命中 expert → 低位变体; token-level loading + hierarchy prefetch; decode 1.34-9.75× |
| 102 | **DynaExq: Runtime Expert Precision Adjustment** | arXiv, 2025 | 2511.15015 | 热点→FP8, 温→INT4, 冷→INT2/CPU; Qwen3-235B on single H200 |
| 103 | **SliceMoE: Dynamic Bit-Sliced Caching** | arXiv, 2025 | 2512.24545 | MSB/LSB as Matryoshka quantization; 能耗 -2.8× |
| 104 | **vLLM Expert Offloading** | PR #37190, 2026 | — | CPU pinned memory 专家权重 + GPU LFRU cache |
| 105 | **MobileMoE: INT4 QAT for On-Device MoE** (Meta) | arXiv, 2026 | 2605.27358 | 四阶段训练配方; 手机上比 dense MobileLLM-Pro 快 1.8-3.8× |

---

## 4. 分布式推理与通信 (Distributed Inference & Communication)

| # | 论文/项目 | 会议/年 | ID | 一句话 |
|---|----------|---------|-----|--------|
| 106 | **TP-Aware Dequantization** (Hoque et al., IBM) | 2024 | 2402.04925 | 优化 GPTQ + TP deployment; GPU 本地保持数据; Llama-70B 上 1.81× |
| 107 | **Communication Compression for TP LLM Inference** (Hansen-Palmus et al.) | arXiv, Nov 2024 | 2411.09510 | AllReduce 前 FP4/FP5/INT4 块量化; PCIe 上 TTFT 减少 2× |
| 108 | **FlashCommunication V2** (Meituan) | 2025 | 2508.03760 | 2-8 bit 任意位宽 AllReduce/All2All; bit-splitting + spike-preserving; AllReduce 3.2× |
| 109 | **BirdMoE: Stochastic Quantization + Mixed-Precision for MoE All2All** | DAC 2025 | — | MoE All2All 随机量化 + 混合精度; 压缩 4-10× |
| 110 | **EQuARX: Dynamic Block Quantization AllReduce** (Ahmed et al.) | 2025 | 2506.17615 | XLA compiler-native; 1.8× speedup; TPU 适用 |
| 111 | **SplitQuant: Phase-Aware Distributed Service on Heterogeneous GPUs** (Zhao et al.) | CLUSTER 2025 | — | 联合优化混合精度 + TP/PP + micro-batch on heterogeneous GPUs |
| 112 | **LM-Offload: Quantization-Aware Tensor Offloading** | IPDPSW 2025 | — | 量化感知张量卸载 + 线程级并行控制; vs FlexGen 2.95× |
| 113 | **Alpa** (Zheng et al.) | OSDI 2022 | 2201.12023 | 算子内 + 算子间自动并行 via ILP; **不考虑量化** |
| 114 | **FlexFlow** (Jia et al.) | 2019 | 1807.05358 | MCMC on SOAP space for data + model parallelism; **不考虑量化** |
| 115 | **MASE** (Fouchard et al.) | 2023-25 | — | 量化搜索 + Alpa-style 自动分布式; **顺序执行, 非联合** |
| 116 | **DeepEP** (DeepSeek) | Open Source, Feb 2025 | — | 首个开源 EP 通信库; NVLink 域内 ~158 GB/s, IB 跨节点 ~47 GB/s; FP8/NVFP4 support |
| 117 | **LMSYS GB200 NVL72 Deployment Report** | Tech Report, Sep 2025 | — | DeepSeek on B200: 3.8× prefill, 4.8× decode vs H100; NVFP4 MoE + FP8 attention |

---

## 5. 硬件与格式 (Hardware & Formats)

| # | 论文/项目 | 来源 | ID/Date | 一句话 |
|---|----------|------|---------|--------|
| 118 | **SageAttention3: Microscaling FP4 Attention** (Zhang et al., Tsinghua) | NeurIPS 2025 | 2505.11594 | 首个 Blackwell FP4 attention kernel; 1,038 TOPS; ~5× FlashAttention2 |
| 119 | **Pushing Intelligence to 4-bit** | NVIDIA Research Blog, 2025 | — | NVFP4 format design: two-level scaling, 88% lower error than power-of-two |
| 120 | **Scaling NVFP4 Inference for FLUX.2 on Blackwell** | NVIDIA Tech Blog, Jan 2026 | — | 10.2× vs H200; NVFP4 + TeaCache + CUDA Graphs + torch.compile |
| 121 | **Silicon Showdown: Consumer-Grade LLM Inference** | arXiv, May 2026 | 2605.00519 | **RTX 5090 NVFP4 falls back to Marlin BF16; 无原生 FP4 compute**; Apple M3 Ultra 23× more efficient |
| 122 | **Is Finer Better? The Limits of Microscaling Formats** (Lee et al., IBM) | ICLR 2026 | — | 更细粒度的 MX format 不一定更好; 理解 block size 的极限 |
| 123 | **Boundary-Aware Quantization** (Kiselev) | arXiv, Jul 2026 | 2607.01478 | 量化改变 decision boundary 而非 accuracy; Decision Jaccard 在 8-bit 仅 0.428 |
| 124 | **Hash Layers** (Roller et al.) | arXiv, 2021 | 2106.04426 | 用固定 hash 函数替代 learned router; 零可学习参数 |
| 125 | **Curiosity-Driven QMoE** | arXiv, Nov 2025 | — | Bayesian uncertainty 引导 expert 选择; 4-bit experts |

---

## 6. 综述与调查 (Surveys)

| # | 论文 | 会议/年 | ID | 一句话 |
|---|------|---------|-----|--------|
| 126 | **A Survey of Low-bit Large Language Models** (Gong et al.) | arXiv, Sep 2024 | 2409.16694 | 低比特 LLM 综合综述: basics, systems, algorithms |
| 127 | **MoE Inference Optimization Survey** (Liu et al.) | arXiv, Dec 2024 | 2412.14219 | MoE 推理综合综述: model-level, system-level, hardware-level |

---

## 7. 核心生产系统与部署资料

| # | 资源 | 内容 |
|---|------|------|
| 128 | **DeepSeek-V2 Technical Report** (arXiv:2405.04434) | MLA + DeepSeekMoE architecture; INT4/INT8 serving |
| 129 | **DeepSeek-V3 Technical Report** (arXiv:2412.19437) | 首个 671B MoE FP8 混合精度训练; 2.788M H800 GPU hours |
| 130 | **DeepSeek-V4-Flash NVFP4** (HuggingFace) | Community NVFP4 quantized DeepSeek-V4 checkpoint |
| 131 | **Qwen3.6 W4A4 Hadamard64 MoE** (HuggingFace) | 实践部署: 256 routed experts W4A4; per-rank 16.5→5.5 GiB |
| 132 | **SGLang EP + Quantization Docs** | W8A8 FP8, W4AFP8, NVFP4, DeepGEMM integration; GB200 NVL72 benchmarks |
| 133 | **vLLM MoE Kernel Features** | FusedMoE, multiple All2All backends, NVFP4 experimental, Cohere W4A8 |
| 134 | **vLLM-Ascend RFC #3012 + PR #3420** | Quantize before All2All; 通信 payload ~50% 减少 |
| 135 | **CUDA native 4-bit float quant (Blackwell)** (llama.cpp #23572) | Experimental NVFP4 in llama.cpp; +40% prompt processing |
