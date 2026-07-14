"""SpinQuant rotation on/off + NVFP4 quantization PPL comparison via llm-compressor.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_llmc.py --model /path/to/Qwen3-4B
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # must be before torch import

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier


# ═════════════════════════════════════════════════════════════════════
# PPL Evaluation
# ═════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_wikitext_ppl(
    model,
    tokenizer,
    max_samples: int = 100,
    max_seq_len: int = 2048,
    device: str = "cuda:0",
) -> Dict:
    """Compute WikiText-2 perplexity. Falls back to PTB if unavailable."""
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        print("    WikiText-2 unavailable, using PTB...")
        ds = load_dataset("ptb_text_only", split="test")
    texts = [t.get("text", t.get("sentence", "")) for t in ds
             if len(t.get("text", t.get("sentence", "")).strip()) > 50]
    texts = texts[:max_samples]

    model = model.to(device).eval()
    total_loss = 0.0
    total_tokens = 0
    n_samples = 0

    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq_len,
            ).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            if loss is not None and not torch.isnan(loss):
                seq_len = inputs["input_ids"].shape[1]
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
                n_samples += 1
            if (n_samples + 1) % 20 == 0 and n_samples > 0:
                ppl = math.exp(total_loss / total_tokens)
                print(f"    [{n_samples}] PPL={ppl:.2f}")

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss)
    return {"ppl": round(ppl, 2), "avg_loss": round(avg_loss, 4),
            "n_samples": n_samples, "total_tokens": total_tokens}


# ═════════════════════════════════════════════════════════════════════
# Quantization Recipe Factories
# ═════════════════════════════════════════════════════════════════════

def recipe_nvfp4_no_rotation() -> list:
    """NVFP4 weight quantization, no rotation."""
    return [
        QuantizationModifier(
            targets="Linear",
            scheme="NVFP4A16",       # weight-only FP4, activations remain FP16
            ignore=["lm_head"],
        ),
    ]


def recipe_nvfp4_spinquant_r1r2() -> list:
    """SpinQuant R1+R2 rotation (offline, zero inference overhead) + NVFP4."""
    return [
        SpinQuantModifier(
            rotations=["R1", "R2"],
            transform_type="hadamard",
            transform_block_size=128,  # block-wise: Qwen3 hidden_size=2560 not power-of-2
        ),
        QuantizationModifier(
            targets="Linear",
            scheme="NVFP4A16",
            ignore=["lm_head"],
        ),
    ]


def recipe_nvfp4_spinquant_full() -> list:
    """SpinQuant R1+R2+R3+R4 (online rotations included) + NVFP4."""
    return [
        SpinQuantModifier(
            rotations=["R1", "R2", "R3", "R4"],
            transform_type="hadamard",
            transform_block_size=128,
        ),
        QuantizationModifier(
            targets="Linear",
            scheme="NVFP4A16",
            ignore=["lm_head"],
        ),
    ]


def recipe_int4_no_rotation() -> list:
    """INT4 weight quantization, no rotation."""
    return [
        QuantizationModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
        ),
    ]


def recipe_int4_spinquant_r1r2() -> list:
    """SpinQuant + INT4."""
    return [
        SpinQuantModifier(rotations=["R1", "R2"], transform_type="hadamard",
                          transform_block_size=128),
        QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    ]


# ═════════════════════════════════════════════════════════════════════
# Main Experiment
# ═════════════════════════════════════════════════════════════════════

def run_comparison(
    model_path: str,
    output_dir: str = "results/llmc",
    ppl_samples: int = 100,
    skip_fp16_ppl: bool = False,
    device: str = "cuda",
) -> Dict:
    """Run: FP16 baseline → quantize with each recipe → PPL comparison.

    Recipes:
      1. FP16 baseline PPL
      2. NVFP4 no rotation
      3. NVFP4 + SpinQuant R1+R2 (offline only)
      4. NVFP4 + SpinQuant R1+R2+R3+R4 (full)
      5. INT4 no rotation
      6. INT4 + SpinQuant R1+R2
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"model": str(model_path), "recipes": {}}

    # ── Shared tokenizer (loaded once) ──
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ── 1. FP16 baseline (fresh load) ──
    if not skip_fp16_ppl:
        print(f"\n{'='*60}")
        print("Baseline: FP16 model PPL")
        print(f"{'='*60}")
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        fp16_result = compute_wikitext_ppl(
            model_fp16, tokenizer, max_samples=ppl_samples, device=device,
        )
        print(f"  FP16 PPL: {fp16_result['ppl']:.2f}")
        results["fp16_baseline"] = fp16_result
        del model_fp16
        torch.cuda.empty_cache()

    # ── 2-6. Quantized variants ──
    recipes = [
        ("NVFP4_noRot", recipe_nvfp4_no_rotation()),
        ("NVFP4_SpinR1R2", recipe_nvfp4_spinquant_r1r2()),
        ("NVFP4_SpinFull", recipe_nvfp4_spinquant_full()),
        ("INT4_noRot", recipe_int4_no_rotation()),
        ("INT4_SpinR1R2", recipe_int4_spinquant_r1r2()),
    ]

    for name, recipe in recipes:
        print(f"\n{'='*60}")
        print(f"Recipe: {name}")
        print(f"{'='*60}")

        # Fresh load
        print(f"  Loading model...")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        print(f"  Loaded in {time.time()-t0:.0f}s")

        # Apply recipe (SpinQuant + Quantize in one shot)
        print(f"  Applying recipe...")
        t0 = time.time()
        oneshot(model=model, recipe=recipe)
        print(f"  Recipe applied in {time.time()-t0:.0f}s")

        # Save quantized model
        save_name = Path(model_path).name + f"-{name}"
        save_path = out_dir / save_name
        print(f"  Saving to {save_path}...")
        tokenizer.save_pretrained(str(save_path))
        model.save_pretrained(str(save_path), save_compressed=True)

        # PPL
        print(f"  Computing PPL...")
        t0 = time.time()
        ppl_result = compute_wikitext_ppl(
            model, tokenizer, max_samples=ppl_samples, device=device,
        )
        print(f"  PPL: {ppl_result['ppl']:.2f}  (eval time: {time.time()-t0:.0f}s)")

        results["recipes"][name] = {
            "save_path": str(save_path),
            "ppl": ppl_result["ppl"],
            "avg_loss": ppl_result["avg_loss"],
        }

        if "fp16_baseline" in results:
            delta = ppl_result["ppl"] - results["fp16_baseline"]["ppl"]
            results["recipes"][name]["delta_vs_fp16"] = round(delta, 2)

        del model
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")

    fp16_ppl = results.get("fp16_baseline", {}).get("ppl", None)
    rows = []
    for name, r in results["recipes"].items():
        ppl = r["ppl"]
        delta = r.get("delta_vs_fp16", float('nan'))
        rows.append((name, ppl, delta))
    # Sort by PPL
    rows.sort(key=lambda x: x[1])

    header = f"{'Recipe':<25s}  {'PPL':>10s}  {'ΔFP16':>10s}"
    print(f"\n  {header}")
    print(f"  {'─'*len(header)}")
    if fp16_ppl is not None:
        print(f"  {'FP16 (baseline)':<25s}  {fp16_ppl:>10.2f}  {'—':>10s}")
    for name, ppl, delta in rows:
        print(f"  {name:<25s}  {ppl:>10.2f}  {delta:>+10.2f}")

    # Rotation benefit: NVFP4_noRot vs NVFP4_SpinR1R2
    no_rot = results["recipes"].get("NVFP4_noRot", {}).get("ppl")
    sp_r1 = results["recipes"].get("NVFP4_SpinR1R2", {}).get("ppl")
    sp_ful = results["recipes"].get("NVFP4_SpinFull", {}).get("ppl")
    if no_rot and sp_r1:
        benefit_r1 = sp_r1 - no_rot
        status = "✓ ROT HELPS" if benefit_r1 < 0 else "✗ ROT HURTS"
        print(f"\n  NVFP4 SpinQuant R1+R2 benefit: ΔPPL = {benefit_r1:+.2f}  {status}")
    if sp_r1 and sp_ful:
        benefit_full = sp_ful - sp_r1
        status = "✓ R3+R4 HELPS" if benefit_full < 0 else "✗ R3+R4 HURTS"
        print(f"  R3+R4 additional benefit:     ΔPPL = {benefit_full:+.2f}  {status}")

    # INT4 comparison
    int4_no = results["recipes"].get("INT4_noRot", {}).get("ppl")
    int4_sp = results["recipes"].get("INT4_SpinR1R2", {}).get("ppl")
    if int4_no and int4_sp:
        benefit_int4 = int4_sp - int4_no
        status = "✓ ROT HELPS" if benefit_int4 < 0 else "✗ ROT HURTS"
        print(f"\n  INT4 SpinQuant R1+R2 benefit:  ΔPPL = {benefit_int4:+.2f}  {status}")

    out_path = out_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  → Results saved to {out_path}")

    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SpinQuant rotation on/off PPL comparison")
    parser.add_argument("--model", type=str,
                        default="/workspace/volume/distributed-training-softdata/models/Qwen3-4B")
    parser.add_argument("--output-dir", type=str, default="results/llmc")
    parser.add_argument("--ppl-samples", type=int, default=100)
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_comparison(
        model_path=args.model,
        output_dir=args.output_dir,
        ppl_samples=args.ppl_samples,
        skip_fp16_ppl=args.skip_fp16,
        device=args.device,
    )


if __name__ == "__main__":
    main()
