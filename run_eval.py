"""Rotation + Quantization PPL Comparison on real LLMs.

Workflow:
  1. Load FP16 model (Qwen3-4B)
  2. For each variant (NoRot, Rot):
     a. Save original weights
     b. Apply Hadamard rotation to linear layers (Rot variant only)
     c. Quantize weights in-place (NVFP4, INT4, MXFP4)
     d. Run WikiText-2 PPL
     e. Restore original weights
  3. Compare: ΔPPL = PPL(Rot+Q) - PPL(Q)

No vLLM, no B300, no FP4 kernel needed.
Uses HF model.forward() with FP16 compute + quantized FP4 weights.

Usage:
    python run_eval.py --model /workspace/volume/distributed-training-softdata/models/Qwen3-4B
    python run_eval.py --model /workspace/volume/distributed-training-softdata/models/Qwen3-4B --formats NVFP4,INT4
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import torch
import torch.nn as nn

from src.quantizers.codebooks import Codebook, e2m1_codebook, int4_codebook, uniform_16_codebook
from src.quantizers.scales import quantize_scale
from src.rotation import apply_hadamard_rotation, _next_power_of_2

OUTPUT_DIR = Path("results/eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# In-place quantization
# ═════════════════════════════════════════════════════════════════════

def quantize_weight_nvfp4(w: torch.Tensor) -> torch.Tensor:
    """In-place quantize weight to NVFP4 format. Returns quantized tensor.

    NVFP4: B=16, E2M1 codebook, E4M3 block scale + FP32 global scale.
    """
    d_out, d_in = w.shape  # HF format: (out, in)
    B = 16
    codebook = e2m1_codebook()
    device = w.device

    # Global scale (FP32)
    max_abs = w.abs().max()
    global_scale = max_abs / codebook.max_abs
    if global_scale == 0:
        global_scale = torch.tensor(1.0, device=device)

    w_norm = w / global_scale

    # Pad to block multiple
    pad_out = 0
    if d_out % B != 0:
        pad_out = B - (d_out % B)
        w_norm = torch.nn.functional.pad(w_norm, (0, 0, 0, pad_out))

    n_blocks = w_norm.shape[0] // B
    w_blocks = w_norm.view(n_blocks, B, d_in)  # (n_blocks, B, d_in)

    # Per-block scales (optimal, then quantized to E4M3)
    block_max = w_blocks.abs().max(dim=1).values.max(dim=1).values  # (n_blocks,)
    scales_opt = block_max / codebook.max_abs
    scales_opt = scales_opt.clamp(min=1e-12)
    scales_quant = quantize_scale(scales_opt, "E4M3").clamp(min=1e-12)

    # Quantize values to E2M1
    wq_blocks = codebook.quantize_round(w_blocks, scales_quant.view(-1, 1, 1))
    wq = wq_blocks.view(n_blocks * B, d_in)

    # Strip padding, multiply back global scale
    if pad_out > 0:
        wq = wq[:d_out]

    return global_scale * wq


def quantize_weight_int4(w: torch.Tensor) -> torch.Tensor:
    """In-place quantize weight to INT4 format: B=128, FP16 scale, uniform codebook."""
    d_out, d_in = w.shape
    B = 128
    codebook = int4_codebook()

    pad_out = 0
    if d_out % B != 0:
        pad_out = B - (d_out % B)
        w = torch.nn.functional.pad(w, (0, 0, 0, pad_out))

    n_blocks = w.shape[0] // B
    w_blocks = w.view(n_blocks, B, d_in)

    block_max = w_blocks.abs().max(dim=1).values.max(dim=1).values
    scales = (block_max / codebook.max_abs).clamp(min=1e-12)
    scales_quant = quantize_scale(scales, "FP16").clamp(min=1e-12)

    wq_blocks = codebook.quantize_round(w_blocks, scales_quant.view(-1, 1, 1))
    wq = wq_blocks.view(n_blocks * B, d_in)

    if pad_out > 0:
        wq = wq[:d_out]
    return wq


def quantize_weight_mxfp4(w: torch.Tensor) -> torch.Tensor:
    """In-place quantize weight to MXFP4: B=32, E8M0 scale, E2M1 codebook."""
    d_out, d_in = w.shape
    B = 32
    codebook = e2m1_codebook()

    pad_out = 0
    if d_out % B != 0:
        pad_out = B - (d_out % B)
        w = torch.nn.functional.pad(w, (0, 0, 0, pad_out))

    n_blocks = w.shape[0] // B
    w_blocks = w.view(n_blocks, B, d_in)

    block_max = w_blocks.abs().max(dim=1).values.max(dim=1).values
    scales = (block_max / codebook.max_abs).clamp(min=1e-12)
    scales_quant = quantize_scale(scales, "E8M0").clamp(min=1e-12)

    wq_blocks = codebook.quantize_round(w_blocks, scales_quant.view(-1, 1, 1))
    wq = wq_blocks.view(n_blocks * B, d_in)

    if pad_out > 0:
        wq = wq[:d_out]
    return wq


QUANTIZERS = {
    "NVFP4": quantize_weight_nvfp4,
    "INT4": quantize_weight_int4,
    "MXFP4": quantize_weight_mxfp4,
}


# ═════════════════════════════════════════════════════════════════════
# PPL Computation
# ═════════════════════════════════════════════════════════════════════

def compute_ppl(
    model,
    tokenizer,
    texts: List[str],
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> Dict:
    """Compute perplexity on text samples."""
    model = model.to(device).eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0

    with torch.no_grad():
        for i, text in enumerate(texts):
            if len(text.strip()) < 20:
                continue

            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            if loss is not None and not torch.isnan(loss):
                seq_len = inputs["input_ids"].shape[1]
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
                n_samples += 1

            if (n_samples + 1) % 20 == 0 and n_samples > 0:
                ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                print(f"    [{n_samples}] PPL={ppl:.2f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss)
    return {"ppl": round(ppl, 2), "avg_loss": round(avg_loss, 4),
            "n_samples": n_samples, "total_tokens": total_tokens}


# ═════════════════════════════════════════════════════════════════════
# Experiment: Rotated vs Unrotated Quantization PPL
# ═════════════════════════════════════════════════════════════════════

def get_linear_layers(model) -> List[Tuple[str, nn.Linear]]:
    """Get all linear layers (name, module) in the model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))
    return layers


def apply_hadamard_to_model(model, device: str = "cuda"):
    """Apply Hadamard rotation to all linear layer weights.

    QuaRot identity: Y = X·W = (X·H)(H^T·W)
    Rotate on INPUT dim: W → H^T·W (dim=0 of HF weight format = rows/d_out in our convention)

    Wait — HF stores weights as (d_out, d_in). The linear operation is:
      y = x @ W^T   where x is (..., d_in), W is (d_out, d_in)
    QuaRot: y = (x @ H) @ (H^T @ W)^T = x @ H @ W^T @ H = ...
    Actually: y = x W^T. With rotation: (x @ H) (H^T @ W)^T = x @ H @ W^T @ H
    But QuaRot rotates ACTIVATIONS along the hidden dim (last dim of x, first dim of W^T).
    For the weight side: W_new = H^T @ W where H acts on d_in dimension.
    In HF format (d_out, d_in): rotate along dim=1 (input dim).

    Actually simpler: since Hadamard H = H^T, we just apply rotation
    to dim=1 of HF weight (the d_in / input dimension).
    """
    layers = get_linear_layers(model)
    for name, module in layers:
        w = module.weight.data.float()  # (d_out, d_in)
        # Pad d_in to power of 2
        n = _next_power_of_2(w.shape[1])
        if n > w.shape[1]:
            w_pad = torch.nn.functional.pad(w, (0, n - w.shape[1]))
        else:
            w_pad = w
        # Apply Hadamard along dim=1 (input dim): W_new = W @ H^T = W @ H (since H symmetric)
        w_rot = apply_hadamard_rotation(w_pad, dim=1)[:, :w.shape[1]]
        module.weight.data = w_rot.to(module.weight.dtype).to(module.weight.device)


def apply_quantization_to_model(
    model, quantizer_fn, skip_layers: Set[str] = None
) -> List[Tuple[str, float, float]]:
    """Quantize all linear layer weights in-place.

    Returns per-layer QSNR stats: [(name, qsnr_dB, mse)]
    """
    if skip_layers is None:
        skip_layers = set()
    stats = []
    layers = get_linear_layers(model)
    for name, module in layers:
        if any(skip_name in name for skip_name in skip_layers):
            continue
        w_orig = module.weight.data.float()
        w_quant = quantizer_fn(w_orig.clone())

        signal = (w_orig ** 2).mean().item()
        noise = ((w_quant - w_orig) ** 2).mean().item()
        qsnr = 10 * math.log10(signal / noise) if noise > 0 else float('inf')

        module.weight.data = w_quant.to(module.weight.dtype).to(module.weight.device)
        stats.append((name, round(qsnr, 1), round(noise, 8)))
    return stats


def run_rot_quant_ppl_experiment(
    model_path: str,
    formats: List[str] = None,
    num_ppl_samples: int = 100,
    device: str = "cuda",
) -> Dict:
    """Main experiment: compare PPL with and without rotation before quantization.

    For each format:
      1. Measure PPL of original FP16 model (baseline)
      2. Quantize WITHOUT rotation → measure PPL
      3. Restore, apply Hadamard → quantize → measure PPL
      4. Compute ΔPPL = PPL(rot+Q) - PPL(Q)
    """
    if formats is None:
        formats = ["NVFP4", "INT4"]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Rotation + Quantization PPL Comparison")
    print(f"  Model: {model_path}")
    print(f"  Formats: {formats}")
    print(f"{'='*60}")

    # ── Load ──
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    # ── Test data ──
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_texts = [t["text"] for t in dataset if len(t["text"].strip()) > 50]
    test_texts = test_texts[:num_ppl_samples]
    print(f"  Test: {len(test_texts)} WikiText-2 samples")

    # ── Save original weights ──
    linear_layers = get_linear_layers(model)
    original_weights = {
        name: module.weight.data.clone().cpu()
        for name, module in linear_layers
    }
    print(f"  Saved {len(original_weights)} layer weights")

    results = {"model": str(model_path), "formats": {}}

    # ── FP16 baseline ──
    print(f"\n  ── FP16 Baseline ──")
    ppl_fp16 = compute_ppl(model, tokenizer, test_texts, device=device)
    print(f"  FP16 PPL: {ppl_fp16['ppl']:.2f}")
    results["fp16_baseline"] = ppl_fp16

    for fmt_name in formats:
        quant_fn = QUANTIZERS[fmt_name]
        print(f"\n  {'─'*50}")
        print(f"  Format: {fmt_name}")
        print(f"  {'─'*50}")

        # ── Variant 1: Quantize only (no rotation) ──
        print(f"  [1/2] Quantize without rotation...")
        t0 = time.time()
        restore_weights(model, original_weights)
        q_stats_raw = apply_quantization_to_model(model, quant_fn)
        ppl_raw = compute_ppl(model, tokenizer, test_texts, device=device)
        avg_qsnr_raw = sum(s[1] for s in q_stats_raw) / len(q_stats_raw) if q_stats_raw else 0
        print(f"    PPL: {ppl_raw['ppl']:.2f}  (vs FP16={ppl_fp16['ppl']:.2f}, Δ={ppl_raw['ppl']-ppl_fp16['ppl']:+.2f})")
        print(f"    Avg weight QSNR: {avg_qsnr_raw:.1f} dB")
        print(f"    Quantize time: {time.time()-t0:.0f}s")

        # ── Variant 2: Rotate then quantize ──
        print(f"  [2/2] Rotate + Quantize...")
        t0 = time.time()
        restore_weights(model, original_weights)
        print(f"    Applying Hadamard rotation... ({time.time()-t0:.0f}s)")
        t_rot = time.time()
        apply_hadamard_to_model(model, device)
        print(f"    Rotation done in {time.time()-t_rot:.0f}s")
        q_stats_rot = apply_quantization_to_model(model, quant_fn)
        ppl_rot = compute_ppl(model, tokenizer, test_texts, device=device)
        avg_qsnr_rot = sum(s[1] for s in q_stats_rot) / len(q_stats_rot) if q_stats_rot else 0
        print(f"    PPL: {ppl_rot['ppl']:.2f}  (vs FP16={ppl_fp16['ppl']:.2f}, Δ={ppl_rot['ppl']-ppl_fp16['ppl']:+.2f})")
        print(f"    Avg weight QSNR: {avg_qsnr_rot:.1f} dB")
        print(f"    Total time: {time.time()-t0:.0f}s")

        # ── Comparison ──
        delta = ppl_rot["ppl"] - ppl_raw["ppl"]
        beneficial = delta < 0
        status = "✓ ROTATION HELPS" if beneficial else "✗ ROTATION HURTS"
        print(f"\n    ΔPPL = {delta:+.2f} — {status}")

        results["formats"][fmt_name] = {
            "ppl_fp16": ppl_fp16["ppl"],
            "ppl_no_rot": ppl_raw["ppl"],
            "ppl_rot": ppl_rot["ppl"],
            "delta_ppl": round(delta, 2),
            "rotation_beneficial": beneficial,
            "avg_qsnr_no_rot": round(avg_qsnr_raw, 1),
            "avg_qsnr_rot": round(avg_qsnr_rot, 1),
        }

    # ── Restore ──
    restore_weights(model, original_weights)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"{'Format':>10s}  {'PPL(noRot)':>12s}  {'PPL(Rot)':>12s}  {'ΔPPL':>8s}  {'Verdict'}")
    print(f"{'─'*60}")
    for fmt_name, res in results["formats"].items():
        verdict = "ROT HELPS" if res["rotation_beneficial"] else "ROT HURTS"
        print(f"  {fmt_name:>10s}  {res['ppl_no_rot']:>12.2f}  {res['ppl_rot']:>12.2f}  "
              f"{res['delta_ppl']:>+8.2f}  {verdict}")

    out_path = OUTPUT_DIR / "rot_quant_ppl.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Saved to {out_path}")

    return results


def restore_weights(model, saved_weights: Dict[str, torch.Tensor]):
    """Restore model weights from saved copies."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in saved_weights:
            w = saved_weights[name].to(module.weight.device).to(module.weight.dtype)
            module.weight.data.copy_(w)


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rotation + Quantization PPL Comparison")
    parser.add_argument("--model", type=str,
                        default="/workspace/volume/distributed-training-softdata/models/Qwen3-4B")
    parser.add_argument("--formats", type=str, default="NVFP4,INT4,MXFP4")
    parser.add_argument("--num-ppl-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",")]

    run_rot_quant_ppl_experiment(
        model_path=args.model,
        formats=formats,
        num_ppl_samples=args.num_ppl_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
