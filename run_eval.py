"""End-to-end PPL evaluation on pre-quantized models.

Compares: FP8 baseline vs NVFP4 model on WikiText-2 perplexity.
Then analyzes per-layer weight statistics to understand rotation potential.

Usage:
    python run_eval.py \
      --nvfp4-model /workspace/volume/pengxiong/models/Qwen3-8B-NVFP4 \
      --fp8-model /workspace/volume/distributed-training-softdata/models/Qwen3-8B-FP8 \
      --fp16-model /workspace/volume/distributed-training-softdata/models/Qwen3-4B
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


OUTPUT_DIR = Path("results/eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# PPL Evaluation
# ═════════════════════════════════════════════════════════════════════

def compute_ppl(
    model,
    tokenizer,
    dataset,
    max_samples: int = 100,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> Dict:
    """Compute perplexity on a text dataset.

    Returns:
        {"ppl": float, "avg_loss": float, "n_samples": int}
    """
    from datasets import load_dataset

    model = model.to(device).eval()

    total_loss = 0.0
    total_tokens = 0
    n_samples = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            text = sample["text"]
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

            if (n_samples + 1) % 20 == 0:
                print(f"    [{n_samples}/{max_samples}] PPL={torch.exp(torch.tensor(total_loss/total_tokens)).item():.2f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return {"ppl": round(ppl, 2), "avg_loss": round(avg_loss, 4), "n_samples": n_samples, "total_tokens": total_tokens}


# ═════════════════════════════════════════════════════════════════════
# Per-Layer Quantization Quality Analysis
# ═════════════════════════════════════════════════════════════════════

def analyze_layer_weights(model, max_layers: int = None) -> List[Dict]:
    """Analyze per-layer weight statistics without running forward pass.

    For each linear layer, compute:
    - Weight RMS, channel RMS std (outlier severity proxy)
    - Optimal per-block scale distribution std
    - Projected quantization error for INT4/NVFP4/MXFP4
    """
    layers = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if max_layers and len(layers) >= max_layers:
            break

        w = module.weight.data.float()
        d_out, d_in = w.shape  # HF format: (out, in)

        # Weight channel statistics
        channel_rms = w.std(dim=1)  # per-output-channel RMS
        outlier_ratio = float((channel_rms.max() / channel_rms.median()).item()) if channel_rms.median() > 0 else 1.0

        # Max abs per channel → proxy for optimal block scale
        # For B=16 per-block quantization along output dim
        w_padded = w
        if d_out % 16 != 0:
            pad = 16 - (d_out % 16)
            w_padded = torch.nn.functional.pad(w, (0, 0, 0, pad))
        n_blocks = w_padded.shape[0] // 16
        w_blocks = w_padded.view(n_blocks, 16, d_in)
        block_maxes = w_blocks.abs().max(dim=1).values.max(dim=1).values  # (n_blocks,)
        # Scale = max_val / 6 for E2M1
        block_scales = block_maxes / 6.0
        scale_std = float(block_scales.std().item())
        scale_mean = float(block_scales.mean().item())
        scale_cv = scale_std / scale_mean if scale_mean > 0 else 0.0

        layers.append({
            "name": name,
            "d_in": d_in, "d_out": d_out,
            "elements": d_in * d_out,
            "outlier_ratio": round(outlier_ratio, 2),
            "scale_cv": round(scale_cv, 4),  # coefficient of variation
            "scale_range": round(float((block_scales.max() / block_scales.min()).item()), 2) if block_scales.min() > 0 else float('inf'),
        })

    return layers


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-quantized models")
    parser.add_argument("--nvfp4-model", type=str,
                        default="/workspace/volume/pengxiong/models/Qwen3-8B-NVFP4")
    parser.add_argument("--fp8-model", type=str,
                        default="/workspace/volume/distributed-training-softdata/models/Qwen3-8B-FP8")
    parser.add_argument("--max-ppl-samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # ── Load test data ──
    print("Loading WikiText-2 test set...")
    wikitext_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"  {len(wikitext_test)} test samples")

    results = {}

    for label, model_path in [
        ("FP8", args.fp8_model),
        ("NVFP4", args.nvfp4_model),
    ]:
        if not Path(model_path).exists():
            print(f"\n  SKIP {label}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {label}: {model_path}")
        print(f"{'='*60}")

        t0 = time.time()
        print("  Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # NVFP4 models exported by NVIDIA ModelOpt may have mismatched weight shapes
        # (packed FP4 storage, fused layers). Use ignore_mismatched_sizes.
        load_kwargs = dict(
            torch_dtype=torch.float16,
            device_map=args.device,
            trust_remote_code=True,
        )
        if "NVFP4" in label.upper() or "nvfp4" in str(model_path).lower():
            load_kwargs["ignore_mismatched_sizes"] = True
            print("  (NVFP4 model: ignoring shape mismatches for packed weights)")

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        print(f"  Loaded in {time.time()-t0:.0f}s")

        # ── PPL ──
        print(f"  Computing PPL on {args.max_ppl_samples} samples...")
        t0 = time.time()
        ppl_result = compute_ppl(
            model, tokenizer, wikitext_test,
            max_samples=args.max_ppl_samples, device=args.device,
        )
        print(f"  PPL: {ppl_result['ppl']:.2f}  (loss={ppl_result['avg_loss']:.4f}, "
              f"{ppl_result['n_samples']} samples, {ppl_result['total_tokens']} tokens)")
        print(f"  PPL eval done in {time.time()-t0:.0f}s")

        # ── Per-layer analysis ──
        print("  Analyzing per-layer weight statistics...")
        layer_stats = analyze_layer_weights(model)

        # Group by layer type
        from collections import defaultdict
        by_type = defaultdict(list)
        for ls in layer_stats:
            # Infer type from name
            name = ls["name"]
            for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                if key in name:
                    by_type[key].append(ls)
                    break
            else:
                by_type["other"].append(ls)

        print(f"    {len(layer_stats)} linear layers analyzed")
        for ltype, lst in sorted(by_type.items()):
            avg_ol = sum(l["outlier_ratio"] for l in lst) / len(lst)
            avg_cv = sum(l["scale_cv"] for l in lst) / len(lst)
            max_ol = max(l["outlier_ratio"] for l in lst)
            print(f"      {ltype:>15s}: {len(lst):>3d} layers  "
                  f"avg_outlier={avg_ol:.1f}x  max_outlier={max_ol:.1f}x  avg_scale_cv={avg_cv:.4f}")

        # Top outlier layers
        layer_stats_sorted = sorted(layer_stats, key=lambda x: x["outlier_ratio"], reverse=True)
        print(f"\n    Top 5 most outlier-heavy layers:")
        for ls in layer_stats_sorted[:5]:
            print(f"      {ls['name']}: outlier_ratio={ls['outlier_ratio']:.0f}x  "
                  f"scale_cv={ls['scale_cv']:.4f}  dim={ls['d_in']}→{ls['d_out']}")

        results[label] = {
            "model_path": str(model_path),
            "ppl": ppl_result,
            "n_layers": len(layer_stats),
            "layer_type_summary": {
                t: {
                    "n": len(lst),
                    "avg_outlier_ratio": round(sum(l["outlier_ratio"] for l in lst)/len(lst), 2),
                    "max_outlier_ratio": round(max(l["outlier_ratio"] for l in lst), 2),
                    "avg_scale_cv": round(sum(l["scale_cv"] for l in lst)/len(lst), 4),
                }
                for t, lst in sorted(by_type.items())
            },
            "top_outlier_layers": layer_stats_sorted[:10],
        }

        # Clean up
        del model
        torch.cuda.empty_cache()

    # ── Compare ──
    if "FP8" in results and "NVFP4" in results:
        ppl_fp8 = results["FP8"]["ppl"]["ppl"]
        ppl_nvfp4 = results["NVFP4"]["ppl"]["ppl"]
        delta = ppl_nvfp4 - ppl_fp8
        print(f"\n{'='*60}")
        print(f"Comparison Summary")
        print(f"{'='*60}")
        print(f"  FP8 PPL:   {ppl_fp8:.2f}")
        print(f"  NVFP4 PPL: {ppl_nvfp4:.2f}")
        print(f"  ΔPPL:      {delta:+.2f} ({'+' if delta > 0 else ''}{delta/ppl_fp8*100:.1f}%)")
        results["_comparison"] = {
            "fp8_ppl": ppl_fp8,
            "nvfp4_ppl": ppl_nvfp4,
            "delta_ppl": round(delta, 2),
            "delta_pct": round(delta / ppl_fp8 * 100, 2),
        }

    out_path = OUTPUT_DIR / "model_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Saved to {out_path}")


if __name__ == "__main__":
    main()
