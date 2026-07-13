# Task Plan: LLM 量化推理三维宽扫 + Gap Analysis

## Goal
对 LLM 量化推理领域三大子方向做系统文献扫描，产出 evidence-labeled gap map，根据 gap 大小和硬件匹配度决议主攻方向。

## Phases
- [x] Phase 0: Setup — 创建计划文件、定义扫描范围
- [x] Phase 1: 方向 A 扫描 — 低比特精度-效率与混合精度策略（4 个子问题）
- [x] Phase 2: 方向 C 扫描 — 权重激活联合量化（4 个子问题）
- [x] Phase 3: 方向 D 扫描 — 量化推理系统优化（4 个子问题）
- [x] Phase 4: 交叉分析 — MOE × 量化交叉问题扫描（已包含在三条线中）
- [x] Phase 5: Gap Matrix 整合 — 三条线回收，产出 gap 评估矩阵 → saved to notes.md
- [ ] Phase 6: 主攻方向决议 — 等待用户确认

## Scan Summary

| 方向 | Papers | Status | Deliverable |
|------|--------|--------|-------------|
| A: 低比特精度-效率 | ~30 | ✅ | `Direction_A_Low_Bit_Quantization_Literature_Scan.md` |
| C: 权重激活联合量化 | ~30 | ✅ | `direction_C_joint_WA_quantization.md` |
| D: 量化推理系统优化 | ~20 | ✅ | Integrated into `notes.md` |
| **Total** | **~80** | | |

## Top 7 Gap Candidates (from Gap Matrix)

| Priority | Gap | Gap Size | HW Match | Feasibility | Novelty | Impact |
|----------|-----|----------|----------|-------------|---------|--------|
| **P0-1** | MOE-specific W4A4 joint quantization (rotation-based, router-aware) | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★★ |
| **P0-2** | B300-native NVFP4 kernel + algorithm co-design | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| **P1-1** | Automated mixed-precision for MoE (per-expert allocation) | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ |
| **P1-2** | Quantization-aware MoE parallelism compiler | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| **P2-1** | Long-context quantization robustness (mechanism + mitigation) | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ |
| **P2-2** | Small-batch activation quantization | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ |
| **P2-3** | FP4 speculative decoding | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★☆ |

## Recommendation (pending user confirmation)

**Primary: P0-1 — MOE-specific W4A4 Joint Quantization**

Adapt rotation-based W4A4 methods (QuaRot/SpinQuant/FlatQuant) for MoE architecture:
- Expert-specific rotation matrices (heterogeneous activation distributions)
- Router-aware calibration (TopK-MSE) to prevent expert shift
- Evaluation on Mixtral, Qwen-MoE, DeepSeek-V2-Lite

**Why this:**
1. Largest open gap confirmed by all 3 scan directions independently
2. Strongest hardware alignment (B300 NVFP4 for experts + A100 for attention)
3. 2-4 month feasible (method design + implementation + evaluation)
4. No strong competitor publication yet
5. Natural extension of mature dense W4A4 methods into the MoE domain

**Supplementary: P0-2 — NVFP4-Aware Algorithm Design**

Exploit NVFP4's 1×16 block + 2-level scaling as a first-class design target rather than a generic 4-bit backend. Can be integrated into P0-1 or pursued as a standalone contribution.

## Decisions Made
- 2026-07-09: 选定方式 2（宽扫 + Gap Analysis）
- 2026-07-09: 主线 = 量化推理，MOE 为目标场景
- 2026-07-09: 不确定 venue，按最快节奏推进
- 2026-07-09: 资源 = 4×B300 + 8×A100

## Errors Encountered
- (none yet)

## Status
**NVFP4 deep-dive complete** — 发现真实 gap, 但需仔细判断

**已完成轮次:**
- Phase 1-3: 三方向扫描 (~80 papers) ✅
- 深挖 1: Gap verification ✅
- 深挖 2-3: B, C killed ✅
- 深挖 4: NVFP4 algorithm-hardware co-design ✅
- paper_index.md: 135 entries ✅
