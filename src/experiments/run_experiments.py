"""Phase 1.3 Pilot Experiments: A (Block Size Sweep) and B (Scale Format Isolation).

Usage:
    python -m src.experiments.run_experiments --exp A    # Block size sweep
    python -m src.experiments.run_experiments --exp B    # Scale format isolation
    python -m src.experiments.run_experiments --exp ALL  # Both
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..qsnr import QSNRDecomposition, compute_qsnr_decomposition, compute_qsnr_matrix
from ..quantizers.block_quant import block_quantize
from ..quantizers.codebooks import get_codebook, e2m1_codebook, uniform_16_codebook, int4_codebook
from ..quantizers.scales import SCALE_FORMATS
from ..rotation import apply_hadamard_rotation, pad_to_power_of_2
from ..weights.synthetic import SYNTHETIC_GENERATORS


# ── Output ──────────────────────────────────────────────────────────

OUTPUT_DIR = Path("results/phase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Experiment A: Block Size Sweep ──────────────────────────────────

@dataclass
class ExpAResult:
    """Results from Experiment A."""
    config: Dict
    # Per block-size: (G_no_rot, G_rot, gain)
    block_results: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "config": self.config,
            "block_results": self.block_results,
        }


def run_experiment_a(
    weight_dist: str = "channel_outlier",
    d_in: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> ExpAResult:
    """Experiment A: Sweep block size, isolate block size effect.

    Config:
      - Codebook: Uniform-16 (fixed)
      - Scale format: FP16 (negligible error, fixed)
      - Block sizes: [4, 8, 16, 32, 64, 128, 256]
      - Rotation: Hadamard vs None

    Key question: At B=16, is rotation benefit >3%?
    """
    block_sizes = [4, 8, 16, 32, 64, 128, 256]
    codebook = uniform_16_codebook(rmax=7.0)

    print(f"\n{'='*60}")
    print(f"Experiment A: Block Size Sweep")
    print(f"  Distribution: {weight_dist}, shape: ({d_in}, {d_out})")
    print(f"  Codebook: Uniform-16, Scale: FP16")
    print(f"{'='*60}")

    # Generate weights (pad to power of 2 for Hadamard compatibility)
    gen = SYNTHETIC_GENERATORS[weight_dist]
    w = gen(d_in, d_out, seed=seed)
    w = pad_to_power_of_2(w, dim=1)

    results = ExpAResult(config={
        "exp": "A",
        "weight_dist": weight_dist,
        "d_in": d_in, "d_out": w.shape[1],
        "codebook": "Uniform-16",
        "scale_format": "FP16",
    })

    for B in block_sizes:
        # Without rotation
        r_raw = block_quantize(w, block_size=B, codebook=codebook, scale_format="FP16")

        # With Hadamard rotation
        w_rot = apply_hadamard_rotation(w, dim=1)
        r_rot = block_quantize(w_rot, block_size=B, codebook=codebook, scale_format="FP16")

        qsnr_raw = r_raw.qsnr_db
        qsnr_rot = r_rot.qsnr_db
        gain = 10 ** ((qsnr_rot - qsnr_raw) / 10) if qsnr_raw < float('inf') else 1.0

        results.block_results[B] = {
            "qsnr_raw_dB": round(qsnr_raw, 2),
            "qsnr_rot_dB": round(qsnr_rot, 2),
            "gain": round(gain, 4),
            "mse_raw": round(r_raw.mse, 6),
            "mse_rot": round(r_rot.mse, 6),
            "is_beneficial": gain > 1.01,
            "is_harmful": gain < 0.99,
        }

        print(f"  B={B:3d}: G={gain:.4f}  "
              f"(raw={qsnr_raw:.1f}dB → rot={qsnr_rot:.1f}dB)  "
              f"{'BENEFICIAL' if gain > 1 else 'HARMFUL' if gain < 1 else 'NEUTRAL'}")

    # Save
    out_path = OUTPUT_DIR / f"expA_{weight_dist}_{d_in}x{d_out}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\n  → Saved to {out_path}")

    return results


# ── Experiment B: Scale Format Isolation ────────────────────────────

@dataclass
class ExpBResult:
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"config": self.config, "results": self.results}


def run_experiment_b(
    weight_dist: str = "channel_outlier",
    d_in: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> ExpBResult:
    """Experiment B: Isolate scale format effect.

    Config:
      - Block size: 16 (fixed, NVFP4 block size)
      - Codebook: Uniform-16 (fixed, removes codebook shape confound)
      - Scale formats: [FP16, E4M3, E8M0, FP32]
      - Rotation: Hadamard vs None

    Key question: Is G(E4M3) < G(FP16)?
    If G(E4M3) < 0.95 → scale precision is dominant mechanism.
    If G(E4M3) ≈ 1.0 → block size is the full story.
    """
    block_size = 16
    codebook = uniform_16_codebook(rmax=7.0)
    scale_formats = ["FP32", "FP16", "E4M3", "E8M0"]

    print(f"\n{'='*60}")
    print(f"Experiment B: Scale Format Isolation (B={block_size})")
    print(f"  Distribution: {weight_dist}, shape: ({d_in}, {d_out})")
    print(f"  Codebook: Uniform-16 (removes codebook confound)")
    print(f"{'='*60}")

    gen = SYNTHETIC_GENERATORS[weight_dist]
    w = gen(d_in, d_out, seed=seed)
    w = pad_to_power_of_2(w, dim=1)
    w_rot = apply_hadamard_rotation(w, dim=1)

    results = ExpBResult(config={
        "exp": "B",
        "weight_dist": weight_dist,
        "d_in": d_in, "d_out": w.shape[1],
        "block_size": block_size,
        "codebook": "Uniform-16",
    })

    header = f"{'Scale':>8s}  {'QSNR_raw':>10s}  {'QSNR_rot':>10s}  {'Gain':>8s}  {'MSE_raw':>10s}  {'MSE_rot':>10s}  {'Verdict'}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")

    for sf in scale_formats:
        r_raw = block_quantize(w, block_size=block_size, codebook=codebook, scale_format=sf)
        r_rot = block_quantize(w_rot, block_size=block_size, codebook=codebook, scale_format=sf)

        qsnr_raw = r_raw.qsnr_db
        qsnr_rot = r_rot.qsnr_db
        gain = 10 ** ((qsnr_rot - qsnr_raw) / 10) if qsnr_raw < float('inf') else 1.0

        fmt_info = SCALE_FORMATS[sf]
        scale_err = fmt_info.relative_error_bound

        verdict = (
            "BENEFICIAL" if gain > 1.01
            else "HARMFUL" if gain < 0.99
            else "NEUTRAL"
        )

        results.results[sf] = {
            "qsnr_raw_dB": round(qsnr_raw, 2),
            "qsnr_rot_dB": round(qsnr_rot, 2),
            "gain": round(gain, 4),
            "mse_raw": round(r_raw.mse, 6),
            "mse_rot": round(r_rot.mse, 6),
            "scale_err_bound": scale_err,
            "n_clipped_raw": r_raw.n_clipped,
            "n_clipped_rot": r_rot.n_clipped,
        }

        print(f"  {sf:>8s}  {qsnr_raw:>10.2f}  {qsnr_rot:>10.2f}  "
              f"{gain:>8.4f}  {r_raw.mse:>10.6f}  {r_rot.mse:>10.6f}  {verdict}")

    # Go/no-go analysis
    g_e4m3 = results.results["E4M3"]["gain"]
    g_fp16 = results.results["FP16"]["gain"]

    print(f"\n  {'─'*50}")
    print(f"  G(E4M3) = {g_e4m3:.4f}  |  G(FP16) = {g_fp16:.4f}")
    print(f"  ΔG = {g_e4m3 - g_fp16:+.4f} (scale penalty)")

    if g_e4m3 < 0.95:
        print(f"\n  >>> GO: G(E4M3) < 0.95 — Scale precision IS dominant. Proceed to algorithm design.")
        results.results["_decision"] = "GO"
    elif g_e4m3 < 1.0:
        print(f"\n  >>> GO CAUTIOUSLY: 0.95 ≤ G(E4M3) < 1.0 — Scale precision matters but weakly.")
        print(f"      Run Exp C (codebook isolation) to complete the picture.")
        results.results["_decision"] = "GO_CAUTIOUS"
    else:
        print(f"\n  >>> NO-GO: G(E4M3) ≥ 1.0 — Trivial block size explanation confirmed.")
        results.results["_decision"] = "NO_GO"

    # Save
    out_path = OUTPUT_DIR / f"expB_{weight_dist}_{d_in}x{d_out}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"\n  → Saved to {out_path}")

    return results


# ── Experiment A+B Combined: Full Grid ──────────────────────────────

def run_experiment_ab_grid(
    weight_dist: str = "channel_outlier",
    d_in: int = 4096,
    d_out: int = 4096,
    seed: int = 42,
) -> Dict:
    """Run full A×B grid: all block sizes × all scale formats × with/without rotation.

    This produces the complete data for the three-factor decomposition analysis.
    """
    block_sizes = [4, 8, 16, 32, 64, 128, 256]
    scale_formats = ["FP32", "FP16", "E4M3", "E8M0"]
    codebook = uniform_16_codebook(rmax=7.0)

    print(f"\n{'='*60}")
    print(f"Experiment AB Grid: {len(block_sizes)} B × {len(scale_formats)} SF × 2 rot")
    print(f"{'='*60}")

    gen = SYNTHETIC_GENERATORS[weight_dist]
    w = gen(d_in, d_out, seed=seed)
    w = pad_to_power_of_2(w, dim=1)
    w_rot = apply_hadamard_rotation(w, dim=1)

    grid = {}
    for B in block_sizes:
        grid[B] = {}
        for sf in scale_formats:
            r_raw = block_quantize(w, block_size=B, codebook=codebook, scale_format=sf)
            r_rot = block_quantize(w_rot, block_size=B, codebook=codebook, scale_format=sf)

            qsnr_raw = r_raw.qsnr_db
            qsnr_rot = r_rot.qsnr_db
            gain = 10 ** ((qsnr_rot - qsnr_raw) / 10) if qsnr_raw < float('inf') else 1.0

            grid[B][sf] = {
                "qsnr_raw": round(qsnr_raw, 2),
                "qsnr_rot": round(qsnr_rot, 2),
                "gain": round(gain, 4),
            }

    # Save
    out_path = OUTPUT_DIR / f"expAB_grid_{weight_dist}_{d_in}x{d_out}_s{seed}.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"weight_dist": weight_dist, "d_in": d_in, "d_out": d_out}, "grid": grid}, f, indent=2)
    print(f"  → Saved to {out_path}")

    return grid


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1.3 Pilot Experiments")
    parser.add_argument("--exp", type=str, default="ALL",
                        choices=["A", "B", "ALL", "GRID"],
                        help="Which experiment(s) to run")
    parser.add_argument("--dist", type=str, default="channel_outlier",
                        choices=list(SYNTHETIC_GENERATORS.keys()),
                        help="Synthetic weight distribution")
    parser.add_argument("--d-in", type=int, default=4096)
    parser.add_argument("--d-out", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-dists", action="store_true",
                        help="Run on all synthetic distributions")
    args = parser.parse_args()

    dists = list(SYNTHETIC_GENERATORS.keys()) if args.all_dists else [args.dist]

    for dist in dists:
        if args.exp in ("A", "ALL"):
            run_experiment_a(dist, args.d_in, args.d_out, args.seed)

        if args.exp in ("B", "ALL"):
            run_experiment_b(dist, args.d_in, args.d_out, args.seed)

        if args.exp in ("GRID", "ALL"):
            run_experiment_ab_grid(dist, args.d_in, args.d_out, args.seed)


if __name__ == "__main__":
    main()
