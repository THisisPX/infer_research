"""Phase 1.3 Real-Model Experiments: Per-layer W4A4 rotation analysis.

Replaces synthetic simulation with actual LLM weights and activations.
Each experiment mirrors the paper methodology: load model, run calibration data,
apply rotation+quantization, measure per-layer metrics.

Usage:
    python run_real.py --model mistral-7b --exp per_layer
    python run_real.py --model llama-3-8b --exp format_compare
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..qsnr import JointWAQSNR, compute_wa_qsnr
from ..quantizers.codebooks import (
    Codebook, e2m1_codebook, uniform_16_codebook, int4_codebook,
)
from ..rotation import apply_hadamard_rotation
from ..weights.real_weights import (
    ModelData, LayerData, extract_model_data,
    load_model, get_wikitext_calibration,
)

OUTPUT_DIR = Path("results/phase1_real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# Experiment R1: Per-Layer Error Decomposition
# ═════════════════════════════════════════════════════════════════════

@dataclass
class LayerResult:
    name: str
    layer_type: str
    d_in: int
    d_out: int

    # Weight-only QSNR (no rotation)
    w_qsnr_raw: float = 0.0  # dB
    w_qsnr_rot: float = 0.0  # dB
    w_gain: float = 1.0

    # Activation-only QSNR
    a_qsnr_raw: float = 0.0
    a_qsnr_rot: float = 0.0
    a_gain: float = 1.0

    # Joint output Y = X·W reconstruction
    joint_qsnr_raw: float = 0.0
    joint_qsnr_rot: float = 0.0
    joint_gain: float = 1.0

    # Scale statistics for weights (diagnostic)
    scale_std_raw: float = 0.0
    scale_std_rot: float = 0.0

    # Activation outlier severity (diagnostic)
    act_outlier_ratio: float = 0.0  # max_channel_rms / median_channel_rms

    def to_dict(self) -> Dict:
        return {
            "name": self.name, "layer_type": self.layer_type,
            "d_in": self.d_in, "d_out": self.d_out,
            "w_qsnr_raw": round(self.w_qsnr_raw, 2),
            "w_qsnr_rot": round(self.w_qsnr_rot, 2),
            "w_gain": round(self.w_gain, 4),
            "a_qsnr_raw": round(self.a_qsnr_raw, 2),
            "a_qsnr_rot": round(self.a_qsnr_rot, 2),
            "a_gain": round(self.a_gain, 4),
            "joint_qsnr_raw": round(self.joint_qsnr_raw, 2),
            "joint_qsnr_rot": round(self.joint_qsnr_rot, 2),
            "joint_gain": round(self.joint_gain, 4),
            "scale_std_raw": round(self.scale_std_raw, 6),
            "scale_std_rot": round(self.scale_std_rot, 6),
            "act_outlier_ratio": round(self.act_outlier_ratio, 2),
        }


def analyze_layer(
    ld: LayerData,
    w_block_size: int,
    w_codebook: Codebook,
    w_scale_format: str,
    w_global_scale: Optional[str] = None,
    a_codebook: Optional[Codebook] = None,
    a_scale_format: str = "FP16",
    max_act_tokens: int = 2048,
    min_act_tokens: int = 256,
    device: str = "cuda",
) -> Optional[LayerResult]:
    """Analyze one layer with joint W-A quantization.

    Args:
        ld: LayerData with weight and activations
        max_act_tokens: cap activation tokens (RAM)
        min_act_tokens: skip layer if fewer tokens collected
    """
    if a_codebook is None:
        a_codebook = w_codebook

    if len(ld.activations) == 0 or ld.activations[0].shape[0] < min_act_tokens:
        return None

    # Weight: HF stores as (d_out, d_in), we want (d_in, d_out)
    w = ld.weight.T.contiguous().float().to(device)

    # Activation: (tokens, d_in)
    x = ld.activations[0].float().to(device)
    if x.shape[0] > max_act_tokens:
        idx = torch.randperm(x.shape[0])[:max_act_tokens]
        x = x[idx]

    d_in, d_out = w.shape
    result = LayerResult(
        name=ld.name,
        layer_type=ld.layer_type,
        d_in=d_in, d_out=d_out,
    )

    # ── Compute activation outlier ratio ──
    channel_rms = x.std(dim=0)
    result.act_outlier_ratio = float(
        (channel_rms.max() / channel_rms.median()).item()
    )

    # ── Joint W-A QSNR ──
    r = compute_wa_qsnr(
        x=x, w=w,
        w_block_size=w_block_size, w_codebook=w_codebook,
        w_scale_format=w_scale_format, w_global_scale=w_global_scale,
        a_codebook=a_codebook, a_scale_format=a_scale_format,
        a_quant_mode="per_token",
    )

    result.w_qsnr_raw = r.w_qsnr_raw
    result.w_qsnr_rot = r.w_qsnr_rot
    result.w_gain = r.w_gain
    result.a_qsnr_raw = r.a_qsnr_raw
    result.a_qsnr_rot = r.a_qsnr_rot
    result.a_gain = r.a_gain
    result.joint_qsnr_raw = r.joint_qsnr_raw
    result.joint_qsnr_rot = r.joint_qsnr_rot
    result.joint_gain = r.joint_gain

    return result


# ═════════════════════════════════════════════════════════════════════
# Experiment R1: Per-Layer Analysis
# ═════════════════════════════════════════════════════════════════════

def run_per_layer_experiment(
    model_name: str = "mistral-7b",
    num_calibration: int = 30,
    max_seq_len: int = 2048,
    max_act_tokens: int = 2048,
    device: str = "cuda",
) -> Dict:
    """Run per-layer error decomposition on real model.

    Config: INT4 baseline (B=128, uniform-15, FP16 scale)
            vs NVFP4 (B=16, E2M1, E4M3+FP32)
            with and without rotation
    """
    print(f"\n{'='*60}")
    print(f"Experiment R1: Per-Layer Error Decomposition")
    print(f"  Model: {model_name}")
    print(f"  Calibration samples: {num_calibration}")
    print(f"{'='*60}")

    # ── Load model ──
    t0 = time.time()
    model, tokenizer = load_model(model_name, device=device)
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    # ── Calibration data ──
    cal_texts = get_wikitext_calibration(tokenizer, num_samples=num_calibration)
    print(f"  Calibration: {len(cal_texts)} WikiText-2 samples")

    # ── Extract weights + activations ──
    t0 = time.time()
    model_data = extract_model_data(
        model, tokenizer, cal_texts, max_seq_len=max_seq_len, device=device,
    )
    print(f"  Extraction done in {time.time()-t0:.0f}s")

    # ── Quantization configs ──
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

    results = {"config": {"model": model_name, "num_calibration": num_calibration},
               "format_results": {}}

    for fmt_name, cfg in configs.items():
        print(f"\n  ── {fmt_name} ──")
        layer_results = []

        for name, ld in model_data.layers.items():
            lr = analyze_layer(
                ld,
                w_block_size=cfg["w_block_size"],
                w_codebook=cfg["w_codebook"],
                w_scale_format=cfg["w_scale"],
                w_global_scale=cfg["w_global"],
                a_codebook=cfg["a_codebook"],
                a_scale_format=cfg["a_scale"],
                max_act_tokens=max_act_tokens,
                device=device,
            )
            if lr is not None:
                layer_results.append(lr)

        if not layer_results:
            print(f"    No layers analyzed!")
            continue

        # ── Aggregate stats ──
        # Group by layer type
        by_type = {}
        for lr in layer_results:
            t = lr.layer_type
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(lr)

        print(f"    Analyzed {len(layer_results)} layers:")
        for ltype, lrs in sorted(by_type.items()):
            avg_w_gain = sum(l.w_gain for l in lrs) / len(lrs)
            avg_a_gain = sum(l.a_gain for l in lrs) / len(lrs)
            avg_joint_gain = sum(l.joint_gain for l in lrs) / len(lrs)
            avg_ol_ratio = sum(l.act_outlier_ratio for l in lrs) / len(lrs)
            print(f"      {ltype:>15s}: {len(lrs):>2d} layers  "
                  f"W-gain={avg_w_gain:.4f}  A-gain={avg_a_gain:.4f}  "
                  f"Joint-gain={avg_joint_gain:.4f}  outlier={avg_ol_ratio:.1f}x")

        # Overall averages
        avg_w = sum(l.w_gain for l in layer_results) / len(layer_results)
        avg_a = sum(l.a_gain for l in layer_results) / len(layer_results)
        avg_j = sum(l.joint_gain for l in layer_results) / len(layer_results)

        n_beneficial = sum(1 for l in layer_results if l.joint_gain > 1.01)
        n_harmful = sum(1 for l in layer_results if l.joint_gain < 0.99)
        n_neutral = len(layer_results) - n_beneficial - n_harmful

        print(f"      {'─'*45}")
        print(f"      OVERALL: W={avg_w:.4f}  A={avg_a:.4f}  Joint={avg_j:.4f}")
        print(f"      Beneficial={n_beneficial}  Harmful={n_harmful}  Neutral={n_neutral}")

        results["format_results"][fmt_name] = {
            "n_layers": len(layer_results),
            "avg_w_gain": round(avg_w, 4),
            "avg_a_gain": round(avg_a, 4),
            "avg_joint_gain": round(avg_j, 4),
            "n_beneficial": n_beneficial,
            "n_harmful": n_harmful,
            "n_neutral": n_neutral,
            "by_layer_type": {
                t: {
                    "n": len(lrs),
                    "avg_w_gain": round(sum(l.w_gain for l in lrs)/len(lrs), 4),
                    "avg_a_gain": round(sum(l.a_gain for l in lrs)/len(lrs), 4),
                    "avg_joint_gain": round(sum(l.joint_gain for l in lrs)/len(lrs), 4),
                    "avg_outlier_ratio": round(
                        sum(l.act_outlier_ratio for l in lrs)/len(lrs), 2),
                }
                for t, lrs in sorted(by_type.items())
            },
            "layers": [lr.to_dict() for lr in layer_results],
        }

    # ── Save ──
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    out_path = OUTPUT_DIR / f"r1_per_layer_{safe_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → Saved to {out_path}")
    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Real-model quantization experiments")
    parser.add_argument("--model", type=str, default="mistral-7b",
                        help="Model name or local path (e.g. /path/to/Qwen3-4B)")
    parser.add_argument("--exp", type=str, default="per_layer",
                        choices=["per_layer"],
                        help="Experiment to run")
    parser.add_argument("--num-calibration", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-act-tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.exp == "per_layer":
        run_per_layer_experiment(
            model_name=args.model,
            num_calibration=args.num_calibration,
            max_seq_len=args.max_seq_len,
            max_act_tokens=args.max_act_tokens,
            device=args.device,
        )


if __name__ == "__main__":
    main()
