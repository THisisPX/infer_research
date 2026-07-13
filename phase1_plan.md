# Phase 1 Plan: FP4 旋转失效的理论分析

## Goal
确定"旋转方法对 FP4 效果显著弱于 INT4"是否有非平凡的理论解释，还是 trivial consequence of block size。

**Success criterion:** 发现一个未被 MR-GPTQ/社区发表的机制性 insight。
**Stop criterion:** 结论为 "因为 block 太小所以 rotation 无效"（trivial），立即止损。

## Phases
- [x] P1.1: 深读关键论文 — MR-GPTQ, DuQuant++, SOAR, HiF4, FAAR, 4/6 (+ BRQ, INT vs FP) ✅
- [x] P1.2: QSNR 理论框架搭建 ✅
- [ ] P1.3: Pilot experiment — 合成权重 + LLM 权重上验证三因子分解
- [ ] P1.4: Go/No-go decision

## P1.1 + P1.2 交叉验证结果

### 三因子分解 (两个 agent 独立得到一致结论)

| 因子 | P1.1 (论文深读) | P1.2 (QSNR 框架) | 一致性 |
|------|---------------|-----------------|--------|
| **Block Size** | MR-GPTQ Lemma 1 证明 16 元素块可证明地中和旋转 | Theorem 1: Δ_codebook(B) ≤ K/B | ✅ 一致 — 可证明但需实验验证 B*≈16 |
| **Scale Precision** | BRQ 发现 MXFP4 的 E8M0 太粗糙; MR-GPTQ 发现 NVFP4 的 E4M3 细 8× | Proposition 2: E4M3 尺度误差 6.25% vs FP16 0.05% | ✅ 一致 — E4M3 误差被旋转放大而 E8M0 不变 |
| **Codebook Shape** | FAAR/4/6: E2M1 非均匀网格影响舍入; INT vs FP: NVINT4 旋转后超越 NVFP4 | Claim 3: E2M1 零点密集+尾部稀疏与旋转后高斯分布不匹配 | ✅ 一致 — 但两个 agent 都认为尺度精度效应可能更强 |

### P1.1 发现的额外论文

| 论文 | 日期 | 关键贡献 |
|------|------|---------|
| **BRQ** (Block Rotation Quantization) | 2025-11 | 发现 >70% 常规块尺度在旋转后被 inflate; MXFP4 E8M0 太粗糙 |
| **INT vs FP crossover** | 2025-10 | NVINT4 在 Hadamard 旋转后超越 NVFP4 — crest factor 分析 |

### 关键 Gap 确认

**没有任何论文独立变化 block size、codebook、scale format 三者来分离它们对旋转收益的贡献。** 所有现有工作都在切换格式时同时改变至少两个维度 (如 NVFP4: B=16+E4M3 vs MXFP4: B=32+E8M0)。

### 最可能产生 Surprising Finding 的方向

**Finding S2: Scale precision, not block size, is the dominant mechanism.**

两个 agent 独立收敛到这个结论：
- P1.1: "BRQ identifies cross-block contamination for MXFP4's E8M0, but NVFP4's E4M3 scale is 8x finer — so the same mechanism produces different severity. No formula connects scale format precision to rotation contamination magnitude."
- P1.2: "If scale precision is the bottleneck, new methods should optimize scale format allocation, not value distribution. This would mean the QuaRot→SpinQuant→FlatQuant line has been optimizing the wrong quantity."

## P1.3 Pilot Experiment Design

### 实验优先级

| 优先级 | 实验 | 目的 | 时间 |
|--------|------|------|------|
| **P0** | Exp A (B-sweep) + Exp B (scale format) | 验证 block size bound + 测试 S2 hypothesis | 2-3 days |
| P1 | Exp C (codebook shape) | 如果 Exp B 的 scale effect 不够显著 | 1 day |
| P2 | Exp D (full NVFP4) | 全量验证 + 与 MR-GPTQ baseline 对比 | 1 day |

### Exp A: Block Size Sweep
```
B ∈ {4, 8, 16, 32, 64, 128}
Scale: FP16 (no scale error)
Codebook: Uniform-16
Weights: synthetic (Gaussian, Laplacian, Laplacian+outliers) + LLaMA-7B layers
Rotation: Hadamard vs None
Metric: G(B) = QSNR_rot / QSNR_raw
Key question: Is G(16) < 1.03? (Theorem 1 predicts ~3% max at B=16)
```

### Exp B: Scale Format Isolation (MOST CRITICAL)
```
B = 16 (fixed)
Codebook: Uniform-16 (fixed)
Scale: {FP16 (oracle), E4M3 (NVFP4-like), E8M0 (MXFP4-like)}
Rotation: Hadamard vs None
Metric: G(scale_format)
Key question: Is G(E4M3) < G(FP16)? If yes → scale precision matters independently.
              Is G(E4M3) < 1? If yes → rotation is actively harmful in NVFP4.
```

## Go/No-go Criteria

| Exp B 结果 | 解释 | 决策 |
|-----------|------|------|
| G(E4M3) < 0.95 | 尺度精度是主导机制, 旋转有害 | **Go** — S2 confirmed, 设计 scale-aware rotation |
| 0.95 < G(E4M3) < 1.0 | 尺度精度有影响但不大 | **Go cautiously** — 需要 Exp C 确认 codebook 贡献 |
| G(E4M3) ≈ 1.0 | 尺度精度无关, 纯 block size 效应 | **No-go** — trivial explanation confirmed |

## Decisions Made
- 2026-07-11: P1.1 + P1.2 交叉验证完成, 两个 agent 独立确认三因子分解框架
- 2026-07-11: Finding S2 (scale precision dominance) 被两个 agent 独立标记为 "最令人惊讶的可能发现"
- 2026-07-11: Exp B 是最关键的 go/no-go 实验
- 2026-07-11: 新增 BRQ (2025-11) 和 INT vs FP crossover (2025-10) 到已知文献列表

## Status
**P1.3 Ready** — Waiting for B300 environment confirmation + experiment execution
