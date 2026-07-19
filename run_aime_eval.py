"""Kill experiment: NVFP4 vs BF16 accuracy vs generation length on AIME 2024.

Usage:
    # NVFP4 model
    python3 run_aime_eval.py \
      --model /workspace/volume/pengxiong/models/Qwen3-8B-NVFP4 \
      --label NVFP4

    # BF16 baseline
    python3 run_aime_eval.py \
      --model /workspace/volume/distributed-training-softdata/models/Qwen3-8B \
      --label BF16

    # Compare results
    python3 run_aime_eval.py --compare results/aime_eval/ --plot
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


OUTPUT_DIR = Path("results/aime_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# AIME 2024 loader
# ═════════════════════════════════════════════════════════════════════

def load_aime(path: str = "/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.jsonl") -> List[Dict]:
    """Load AIME 2024 problems."""
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    print(f"Loaded {len(problems)} AIME 2024 problems")
    return problems


# ═════════════════════════════════════════════════════════════════════
# Answer extraction
# ═════════════════════════════════════════════════════════════════════

def extract_answer(text: str) -> Optional[int]:
    """Extract final integer answer from model output.

    AIME answers are integers 0-999.
    Strategy: find the last number in the text after the final \\boxed{} or answer marker.
    """
    # Try \boxed{...} first
    boxed_match = re.findall(r'\\boxed\{(\d+)\}', text)
    if boxed_match:
        return int(boxed_match[-1])

    # Try "answer is X" / "the answer is X"
    answer_patterns = [
        r'(?:answer|final answer|result)(?:\s+is)?\s*:?\s*(\d{1,4})',
        r'=?\s*(\d{1,4})\s*(?:$|\n)',
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return int(matches[-1])

    # Fallback: last standalone number
    all_numbers = re.findall(r'\b(\d{1,4})\b', text)
    if all_numbers:
        return int(all_numbers[-1])

    return None


# ═════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════

def format_prompt(problem: Dict) -> str:
    """Format AIME problem as chat template."""
    messages = problem["prompt"]
    # Build a simple chat prompt (works for Qwen3 chat template)
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def run_vllm_eval(
    model_path: str,
    problems: List[Dict],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    tensor_parallel: int = 1,
    label: str = "",
) -> Dict:
    """Run AIME evaluation using vLLM.

    Returns per-problem results with generation lengths.
    """
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"AIME 2024 Eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Max tokens: {max_tokens}, TP: {tensor_parallel}")
    print(f"{'='*60}")

    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        max_model_len=16384,
    )
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0 if temperature == 0.0 else 0.95,
    )

    # Format prompts
    prompts = [format_prompt(p) for p in problems]

    # Tokenize to get prompt lengths
    tokenizer = llm.get_tokenizer()
    prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
    print(f"  Prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, mean={np.mean(prompt_lengths):.0f}")

    # Generate
    print(f"  Generating {len(prompts)} responses...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"  Generation done in {gen_time:.0f}s ({gen_time/len(prompts):.1f}s/problem)")

    # Parse results
    results = []
    correct = 0
    total = 0
    total_output_tokens = 0

    for i, (problem, output) in enumerate(zip(problems, outputs)):
        response = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)
        total_output_tokens += output_tokens

        pred = extract_answer(response)
        label_val = int(problem["label"].strip())
        is_correct = (pred == label_val)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": i,
            "label": problem["label"],
            "prediction": pred,
            "correct": is_correct,
            "prompt_tokens": prompt_lengths[i],
            "output_tokens": output_tokens,
            "total_tokens": prompt_lengths[i] + output_tokens,
            "response": response,  # save full response for qualitative analysis
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.2%}")
    print(f"  Avg output tokens: {total_output_tokens/total:.0f}")
    print(f"  Total generation time: {gen_time:.0f}s")

    # ── Bucket by output length (fine granularity) ──
    buckets = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096),
               (4096, 5120), (5120, 6144), (6144, 7168), (7168, 8192),
               (8192, float('inf'))]
    bucket_stats = []
    for lo, hi in buckets:
        in_range = [r for r in results if lo <= r["output_tokens"] < hi]
        if not in_range:
            continue
        acc = sum(1 for r in in_range if r["correct"]) / len(in_range)
        hi_str = f"{int(hi)}" if hi < float('inf') else "max"
        bucket_stats.append({
            "range": f"[{int(lo)}, {hi_str})",
            "n": len(in_range),
            "accuracy": round(acc, 4),
            "avg_output_tokens": round(np.mean([r["output_tokens"] for r in in_range]), 1),
        })
        print(f"    Length [{int(lo):>5}, {hi_str:>5}): "
              f"n={len(in_range):2d}  acc={acc:.2%}")

    # Save
    out = {
        "model": model_path,
        "label": label,
        "config": {"max_tokens": max_tokens, "temperature": temperature, "tensor_parallel": tensor_parallel},
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_output_tokens": round(total_output_tokens / total, 1) if total > 0 else 0,
        "generation_time_s": round(gen_time, 1),
        "bucket_stats": bucket_stats,
        "results": results,
    }

    model_name = Path(model_path).name
    out_path = OUTPUT_DIR / f"aime2024_{label}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  → Saved to {out_path}")

    return out


def run_compare(results_dir: str = "results/aime_eval", plot: bool = False):
    """Compare all eval results and plot accuracy vs length."""
    results_dir = Path(results_dir)
    all_data = []
    for path in sorted(results_dir.glob("aime2024_*.json")):
        with open(path) as f:
            data = json.load(f)
        all_data.append(data)
        label = data.get("label", path.stem)
        print(f"  {label:<10s}  Acc: {data['accuracy']:.2%}  "
              f"({data['correct']}/{data['total']})  "
              f"Avg out: {data['avg_output_tokens']:.0f} tok")

    if not all_data:
        print("  No results found")
        return

    # Print bucket comparison table
    print(f"\n  {'─'*80}")
    header = f"  {'Bucket':>18s}"
    for d in all_data:
        header += f"  {d['label']:>16s}"
    print(header)
    print(f"  {'─'*80}")

    # Get all buckets
    all_buckets = set()
    for d in all_data:
        for b in d.get("bucket_stats", []):
            all_buckets.add(b["range"])
    all_buckets = sorted(all_buckets)

    for bucket_name in all_buckets:
        row = f"  {bucket_name:>18s}"
        for d in all_data:
            matches = [b for b in d.get("bucket_stats", []) if b["range"] == bucket_name]
            if matches:
                b = matches[0]
                row += f"  {b['accuracy']:.2%} (n={b['n']})".rjust(16)
            else:
                row += " " * 16
        print(row)

    # ── Per-problem correctness matrix ──
    if len(all_data) == 2:
        d0, d1 = all_data[0], all_data[1]
        r0 = {r["idx"]: r for r in d0["results"]}
        r1 = {r["idx"]: r for r in d1["results"]}
        print(f"\n  {'─'*70}")
        print(f"  Per-Problem Comparison: {d0['label']} vs {d1['label']}")
        print(f"  {'─'*70}")
        both_correct = both_wrong = only_0 = only_1 = 0
        len_mismatch_problems = []
        for idx in sorted(r0.keys()):
            c0 = r0[idx]["correct"]
            c1 = r1[idx]["correct"]
            t0 = r0[idx]["output_tokens"]
            t1 = r1[idx]["output_tokens"]
            if c0 and c1:
                both_correct += 1
            elif not c0 and not c1:
                both_wrong += 1
            elif c0 and not c1:
                only_0 += 1
                len_mismatch_problems.append(f"      #{idx}: {d0['label']}=✓({t0} tok)  {d1['label']}=✗({t1} tok)")
            else:
                only_1 += 1
                len_mismatch_problems.append(f"      #{idx}: {d0['label']}=✗({t0} tok)  {d1['label']}=✓({t1} tok)")
        print(f"    Both correct: {both_correct}")
        print(f"    Both wrong:   {both_wrong}")
        print(f"    Only {d0['label']}: {only_0}")
        print(f"    Only {d1['label']}: {only_1}")
        if len_mismatch_problems:
            print(f"\n    Disagreement cases:")
            for p in len_mismatch_problems:
                print(p)

        # ── Length gap analysis ──
        print(f"\n  {'─'*70}")
        print(f"  Token Length Gap Analysis")
        print(f"  {'─'*70}")
        gaps = []
        for idx in sorted(r0.keys()):
            t0 = r0[idx]["output_tokens"]
            t1 = r1[idx]["output_tokens"]
            gaps.append((idx, t0, t1, r0[idx]["correct"], r1[idx]["correct"]))
        avg_gap = np.mean([abs(t0-t1) for _, t0, t1, _, _ in gaps])
        print(f"    Avg |length_diff|: {avg_gap:.0f} tokens")
        print(f"    {d0['label']} avg out: {d0['avg_output_tokens']:.0f} tokens")
        print(f"    {d1['label']} avg out: {d1['avg_output_tokens']:.0f} tokens")

        # Per-problem output length comparison for disagreement cases
        print(f"\n    Length comparison for disagreement cases:")
        for idx, t0, t1, c0, c1 in gaps:
            if c0 != c1:
                winner = d0['label'] if c0 else d1['label']
                print(f"      #{idx}: {d0['label']}={t0} tok  {d1['label']}={t1} tok  → {winner} wins")

    # Plot
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, d in enumerate(all_data):
                buckets = d.get("bucket_stats", [])
                if not buckets:
                    continue
                midpoints = []
                accs = []
                for b in buckets:
                    lo_str, hi_str = b["range"].strip("[]()").split(", ")
                    lo = int(lo_str)
                    hi = int(hi_str) if hi_str != "max" else 16384
                    midpoints.append(np.sqrt(lo * hi))  # geometric mean
                    accs.append(b["accuracy"])
                ax.plot(midpoints, accs, 'o-', color=colors[i % len(colors)],
                        label=f"{d['label']} (overall={d['accuracy']:.2%})", linewidth=2, markersize=8)

            ax.set_xscale('log')
            ax.set_xlabel("Output Token Length (geometric mean of bucket)", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title("AIME 2024: Accuracy vs Generation Length", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = results_dir / "accuracy_vs_length.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\n  Plot saved to {plot_path}")
        except ImportError:
            print("  (matplotlib not installed, skip plot)")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AIME 2024: NVFP4 vs BF16 accuracy vs generation length")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--label", type=str, default="", help="Label for this run")
    parser.add_argument("--data", type=str, default="/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.jsonl")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--compare", type=str, help="Compare results directory")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare, args.plot)
    elif args.model:
        problems = load_aime(args.data)
        label = args.label or Path(args.model).name
        run_vllm_eval(
            model_path=args.model,
            problems=problems,
            max_tokens=args.max_tokens,
            tensor_parallel=args.tensor_parallel,
            label=label,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
