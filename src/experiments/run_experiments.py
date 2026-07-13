"""Phase 1.3 Pilot Experiments with Joint W-A Quantization.

Rotation benefit pathway:
  Y = X·W = (X·H)(H^T·W)
  Rotation spreads activation outlier energy → per-token quant of X·H is better
  Weight quantization of H^T·W is nearly unchanged (orthogonal transform)
  → Joint output Y_q = (X·H)_q · (H^T·W)_q has lower error

Usage:
    python run.py --exp ALL_WA           # Joint W-A experiments
    python run.py --exp A2               # Block size sweep (joint W-A)
    python run.py --exp B2               # Scale format isolation (joint W-A)
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from ..qsnr import JointWAQSNR, compute_wa_qsnr
from ..quantizers.codebooks import e2m1_codebook, uniform_16_codebook, int4_codebook
from ..quantizers.scales import SCALE_FORMATS
from ..rotation import pad_to_power_of_2
from ..weights.activations import ACTIVATION_GENERATORS
from ..weights.synthetic import SYNTHETIC_GENERATORS

OUTPUT_DIR = Path("results/phase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# Experiment A2: Block Size Sweep (Joint W-A)
# ═════════════════════════════════════════════════════════════════════

def run_experiment_a2(
    act_dist: str = "outlier",
    weight_dist: str = "channel_outlier",
    seq_len: int = 128,
    d_model: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> Dict:
    """Exp A2: Sweep weight block size, keep activation quantization fixed.

    Isolates: does W block size affect the OVERALL W4A4 rotation benefit?

    Config:
      - Activation: per-token, Uniform-16, FP16 scale (fixed across all B)
        → rotation benefit from outlier smoothing is CONSTANT here
      - Weight: block_size B ∈ {4,8,16,32,64,128}, Uniform-16, FP16 scale
      - Rotation: Hadamard on both X and W via the QuaRot identity

    Key question: Does smaller W block size reduce the JOINT rotation gain?
    If joint_gain(B=16) ≈ joint_gain(B=128) → W block size doesn't matter
    If joint_gain(B=16) < joint_gain(B=128) → W block size does matter
    """
    block_sizes = [4, 8, 16, 32, 64, 128]
    a_codebook = uniform_16_codebook(rmax=7.0)
    w_codebook = uniform_16_codebook(rmax=7.0)

    print(f"\n{'='*60}")
    print(f"Experiment A2: Block Size Sweep (Joint W-A)")
    print(f"  Act: {act_dist} ({seq_len}×{d_model}), per-token, Uniform-16, FP16")
    print(f"  Weight: {weight_dist} ({d_model}×{d_out}), Uniform-16, FP16")
    print(f"{'='*60}")

    act_gen = ACTIVATION_GENERATORS[act_dist]
    wgt_gen = SYNTHETIC_GENERATORS[weight_dist]

    x = act_gen(seq_len, d_model, seed=seed)
    w = wgt_gen(d_model, d_out, seed=seed)

    # Pad d_model to power-of-2 for Hadamard
    n_pad = 1
    while n_pad < d_model:
        n_pad <<= 1

    results = {"config": {"exp": "A2", "act_dist": act_dist,
                          "weight_dist": weight_dist,
                          "seq_len": seq_len, "d_model": d_model,
                          "d_model_padded": n_pad, "d_out": d_out},
               "block_results": {}}

    header = f"{'B':>4s}  {'W-gain':>8s}  {'A-gain':>8s}  {'Joint-gain':>10s}  {'Joint-raw':>10s}  {'Joint-rot':>10s}  {'Verdict'}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for B in block_sizes:
        r = compute_wa_qsnr(
            x=x, w=w,
            w_block_size=B, w_codebook=w_codebook, w_scale_format="FP16",
            a_codebook=a_codebook, a_scale_format="FP16",
            a_quant_mode="per_token",
        )

        verdict = ("BENEFICIAL" if r.joint_gain > 1.01
                   else "HARMFUL" if r.joint_gain < 0.99 else "NEUTRAL")

        results["block_results"][B] = {
            "w_gain": round(r.w_gain, 4),
            "a_gain": round(r.a_gain, 4),
            "joint_gain": round(r.joint_gain, 4),
            "joint_qsnr_raw": round(r.joint_qsnr_raw, 2),
            "joint_qsnr_rot": round(r.joint_qsnr_rot, 2),
            "w_qsnr_raw": round(r.w_qsnr_raw, 2),
            "w_qsnr_rot": round(r.w_qsnr_rot, 2),
            "a_qsnr_raw": round(r.a_qsnr_raw, 2),
            "a_qsnr_rot": round(r.a_qsnr_rot, 2),
        }

        print(f"  {B:>4d}  {r.w_gain:>8.4f}  {r.a_gain:>8.4f}  "
              f"{r.joint_gain:>10.4f}  {r.joint_qsnr_raw:>10.1f}  "
              f"{r.joint_qsnr_rot:>10.1f}  {verdict}")

    out_path = OUTPUT_DIR / f"expA2_jointWA_{act_dist}_{weight_dist}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved to {out_path}")
    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment B2: Scale Format Isolation (Joint W-A)
# ═════════════════════════════════════════════════════════════════════

def run_experiment_b2(
    act_dist: str = "outlier",
    weight_dist: str = "channel_outlier",
    seq_len: int = 128,
    d_model: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> Dict:
    """Exp B2: Vary WEIGHT scale format, keep everything else fixed.

    Since rotation benefit for ACTIVATIONS comes from smoothing X → X·H
    (independent of W's scale format), we expect A-gain ≈ constant.
    But W's scale format may affect the JOINT gain differently after rotation.

    Config:
      - Activation: per-token, Uniform-16, FP16 scale
      - Weight: B=16, Uniform-16, scale ∈ {FP32, FP16, E4M3, E8M0}
    """
    block_size = 16
    a_codebook = uniform_16_codebook(rmax=7.0)
    w_codebook = uniform_16_codebook(rmax=7.0)
    scale_formats = ["FP32", "FP16", "E4M3", "E8M0"]

    print(f"\n{'='*60}")
    print(f"Experiment B2: Scale Format Isolation (Joint W-A, B={block_size})")
    print(f"  Act: {act_dist} ({seq_len}×{d_model}), per-token, Uniform-16, FP16")
    print(f"  Weight: {weight_dist} ({d_model}×{d_out}), B={block_size}, Uniform-16")
    print(f"{'='*60}")

    act_gen = ACTIVATION_GENERATORS[act_dist]
    wgt_gen = SYNTHETIC_GENERATORS[weight_dist]

    x = act_gen(seq_len, d_model, seed=seed)
    w = wgt_gen(d_model, d_out, seed=seed)

    results = {"config": {"exp": "B2", "act_dist": act_dist,
                          "weight_dist": weight_dist,
                          "seq_len": seq_len, "d_model": d_model,
                          "d_out": d_out, "block_size": block_size},
               "scale_results": {}}

    header = (f"{'Scale':>8s}  {'W-gain':>8s}  {'A-gain':>8s}  "
              f"{'Joint-gain':>10s}  {'Joint-raw':>10s}  "
              f"{'Joint-rot':>10s}  {'Verdict'}")
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for sf in scale_formats:
        r = compute_wa_qsnr(
            x=x, w=w,
            w_block_size=block_size, w_codebook=w_codebook,
            w_scale_format=sf,
            a_codebook=a_codebook, a_scale_format="FP16",
            a_quant_mode="per_token",
        )

        scale_info = SCALE_FORMATS[sf]
        verdict = ("BENEFICIAL" if r.joint_gain > 1.01
                   else "HARMFUL" if r.joint_gain < 0.99 else "NEUTRAL")

        results["scale_results"][sf] = {
            "w_gain": round(r.w_gain, 4),
            "a_gain": round(r.a_gain, 4),
            "joint_gain": round(r.joint_gain, 4),
            "joint_qsnr_raw": round(r.joint_qsnr_raw, 2),
            "joint_qsnr_rot": round(r.joint_qsnr_rot, 2),
            "scale_err_bound": scale_info.relative_error_bound,
        }

        print(f"  {sf:>8s}  {r.w_gain:>8.4f}  {r.a_gain:>8.4f}  "
              f"{r.joint_gain:>10.4f}  {r.joint_qsnr_raw:>10.1f}  "
              f"{r.joint_qsnr_rot:>10.1f}  {verdict}")

    g_e4m3 = results["scale_results"]["E4M3"]["joint_gain"]
    g_fp16 = results["scale_results"]["FP16"]["joint_gain"]

    print(f"\n  {'─'*50}")
    print(f"  Joint G(E4M3) = {g_e4m3:.4f}  |  Joint G(FP16) = {g_fp16:.4f}")
    print(f"  ΔG = {g_e4m3 - g_fp16:+.4f} (scale penalty, joint W-A)")

    if g_e4m3 < 0.95:
        print(f"\n  >>> GO: Scale precision IS dominant for joint W-A.")
        results["_decision"] = "GO"
    elif g_e4m3 < 1.0:
        print(f"\n  >>> GO CAUTIOUSLY: Weak scale effect.")
        results["_decision"] = "GO_CAUTIOUS"
    else:
        print(f"\n  >>> RE-EVALUATE: Check if activation outlier scale is realistic.")
        results["_decision"] = "REEVALUATE"

    out_path = OUTPUT_DIR / f"expB2_jointWA_{act_dist}_{weight_dist}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved to {out_path}")
    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment C2: Activation Outlier Severity Sweep
# ═════════════════════════════════════════════════════════════════════

def run_experiment_c2(
    seq_len: int = 128,
    d_model: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> Dict:
    """Exp C2: Sweep activation outlier severity.

    Tests how the rotation benefit varies with outlier severity.
    This validates whether our synthetic activations are generating
    realistic rotation gains.

    Config:
      - Activation: per-token, Uniform-16, FP16
      - Weight: B=16, Uniform-16, FP16 (no scale confound)
      - Vary: activation outlier_scale ∈ {1, 5, 10, 20, 50, 100}
    """
    outlier_scales = [1, 5, 10, 20, 50, 100, 200]
    a_codebook = uniform_16_codebook(rmax=7.0)
    w_codebook = uniform_16_codebook(rmax=7.0)

    print(f"\n{'='*60}")
    print(f"Experiment C2: Outlier Severity Sweep")
    print(f"  Act: outlier ({seq_len}×{d_model}), per-token, Uniform-16, FP16")
    print(f"  Weight: channel_outlier ({d_model}×{d_out}), B=16, Uniform-16, FP16")
    print(f"{'='*60}")

    from ..weights.activations import activation_with_outliers
    from ..weights.synthetic import channel_outlier_weights

    w = channel_outlier_weights(d_model, d_out, seed=seed)

    results = {"config": {"exp": "C2", "seq_len": seq_len, "d_model": d_model,
                          "d_out": d_out},
               "outlier_results": {}}

    header = (f"{'Outlier':>8s}  {'A-gain':>8s}  {'Joint-gain':>10s}  "
              f"{'A-raw':>8s}  {'A-rot':>8s}  {'5th-tile':>8s}  {'1st-tile':>8s}")
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for os in outlier_scales:
        x = activation_with_outliers(seq_len, d_model, outlier_scale=os, seed=seed)
        r = compute_wa_qsnr(
            x=x, w=w,
            w_block_size=16, w_codebook=w_codebook, w_scale_format="FP16",
            a_codebook=a_codebook, a_scale_format="FP16",
            a_quant_mode="per_token",
        )

        # Show worst-token gain (rotation helps outliers the most)
        pt_gains_raw = r.a_qsnr_per_token_raw
        pt_gains_rot = r.a_qsnr_per_token_rot
        pt_gains = [rot - raw for raw, rot in zip(pt_gains_raw, pt_gains_rot)]
        pt_gains_sorted = sorted(pt_gains)

        # Bottom 5th and 1st percentile (most outlier-affected tokens)
        n = len(pt_gains_sorted)
        p5 = pt_gains_sorted[max(0, n // 20)]
        p1 = pt_gains_sorted[max(0, n // 100)]

        results["outlier_results"][os] = {
            "a_gain": round(r.a_gain, 4),
            "joint_gain": round(r.joint_gain, 4),
            "a_qsnr_raw": round(r.a_qsnr_raw, 2),
            "a_qsnr_rot": round(r.a_qsnr_rot, 2),
            "per_token_gain_worst_5pct": round(p5, 2),
            "per_token_gain_worst_1pct": round(p1, 2),
        }

        print(f"  {os:>8.0f}  {r.a_gain:>8.4f}  {r.joint_gain:>10.4f}  "
              f"{r.a_qsnr_raw:>8.1f}  {r.a_qsnr_rot:>8.1f}  "
              f"{p5:>8.1f}  {p1:>8.1f}")

    out_path = OUTPUT_DIR / f"expC2_outlier_sweep_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved to {out_path}")
    return results


# ═════════════════════════════════════════════════════════════════════
# Experiment D: Full NVFP4 vs INT4 W4A4 comparison
# ═════════════════════════════════════════════════════════════════════

def run_experiment_d(
    act_dist: str = "llm_like",
    weight_dist: str = "channel_outlier",
    seq_len: int = 128,
    d_model: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> Dict:
    """Exp D: Compare full format configurations (NVFP4 vs INT4 vs MXFP4).

    Each format uses its REAL specifications:
      - INT4: W: B=128, uniform-15 codebook, FP16 scale
              A: per-token, uniform-15 codebook, FP16 scale
      - NVFP4: W: B=16, E2M1 codebook, E4M3+FP32 scales
               A: per-token, E2M1 codebook, E4M3 scale
      - MXFP4: W: B=32, E2M1 codebook, E8M0 scale
               A: per-token, E2M1 codebook, E8M0 scale
    """
    configs = {
        "INT4": {
            "w_block_size": 128, "w_codebook": int4_codebook(),
            "w_scale": "FP16", "w_global": None,
            "a_codebook": int4_codebook(), "a_scale": "FP16",
        },
        "NVFP4": {
            "w_block_size": 16, "w_codebook": e2m1_codebook(),
            "w_scale": "E4M3", "w_global": "FP32",
            "a_codebook": e2m1_codebook(), "a_scale": "E4M3",
        },
        "MXFP4": {
            "w_block_size": 32, "w_codebook": e2m1_codebook(),
            "w_scale": "E8M0", "w_global": None,
            "a_codebook": e2m1_codebook(), "a_scale": "E8M0",
        },
    }

    print(f"\n{'='*60}")
    print(f"Experiment D: Full Format Comparison (INT4 vs NVFP4 vs MXFP4)")
    print(f"  Act: {act_dist} ({seq_len}×{d_model})")
    print(f"  Weight: {weight_dist} ({d_model}×{d_out})")
    print(f"{'='*60}")

    act_gen = ACTIVATION_GENERATORS[act_dist]
    wgt_gen = SYNTHETIC_GENERATORS[weight_dist]

    x = act_gen(seq_len, d_model, seed=seed)
    w = wgt_gen(d_model, d_out, seed=seed)

    results = {"config": {"exp": "D", "act_dist": act_dist,
                          "weight_dist": weight_dist,
                          "seq_len": seq_len, "d_model": d_model, "d_out": d_out},
               "format_results": {}}

    header = (f"{'Format':>8s}  {'W-gain':>8s}  {'A-gain':>8s}  "
              f"{'Joint-gain':>10s}  {'NoRot(dB)':>10s}  "
              f"{'Rot(dB)':>10s}  {'Verdict'}")
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for fmt_name, cfg in configs.items():
        r = compute_wa_qsnr(
            x=x, w=w,
            w_block_size=cfg["w_block_size"], w_codebook=cfg["w_codebook"],
            w_scale_format=cfg["w_scale"], w_global_scale=cfg["w_global"],
            a_codebook=cfg["a_codebook"], a_scale_format=cfg["a_scale"],
            a_quant_mode="per_token",
        )

        verdict = ("BENEFICIAL" if r.joint_gain > 1.01
                   else "HARMFUL" if r.joint_gain < 0.99 else "NEUTRAL")

        results["format_results"][fmt_name] = {
            "w_gain": round(r.w_gain, 4),
            "a_gain": round(r.a_gain, 4),
            "joint_gain": round(r.joint_gain, 4),
            "joint_no_rot": round(r.joint_qsnr_raw, 2),
            "joint_rot": round(r.joint_qsnr_rot, 2),
        }

        print(f"  {fmt_name:>8s}  {r.w_gain:>8.4f}  {r.a_gain:>8.4f}  "
              f"{r.joint_gain:>10.4f}  {r.joint_qsnr_raw:>10.1f}  "
              f"{r.joint_qsnr_rot:>10.1f}  {verdict}")

    out_path = OUTPUT_DIR / f"expD_formats_{act_dist}_{weight_dist}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved to {out_path}")
    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1.3 Joint W-A Experiments")
    parser.add_argument("--exp", type=str, default="ALL_WA",
                        choices=["A2", "B2", "C2", "D", "ALL_WA"],
                        help="Which experiment(s) to run")
    parser.add_argument("--act-dist", type=str, default="outlier",
                        choices=list(ACTIVATION_GENERATORS.keys()),
                        help="Activation distribution")
    parser.add_argument("--weight-dist", type=str, default="channel_outlier",
                        choices=list(SYNTHETIC_GENERATORS.keys()),
                        help="Weight distribution")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=4096)
    parser.add_argument("--d-out", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.exp in ("A2", "ALL_WA"):
        run_experiment_a2(args.act_dist, args.weight_dist,
                          args.seq_len, args.d_model, args.d_out, args.seed)
    if args.exp in ("B2", "ALL_WA"):
        run_experiment_b2(args.act_dist, args.weight_dist,
                          args.seq_len, args.d_model, args.d_out, args.seed)
    if args.exp in ("C2", "ALL_WA"):
        run_experiment_c2(args.seq_len, args.d_model, args.d_out, args.seed)
    if args.exp in ("D", "ALL_WA"):
        run_experiment_d(args.act_dist, args.weight_dist,
                         args.seq_len, args.d_model, args.d_out, args.seed)

if __name__ == "__main__":
    main()
