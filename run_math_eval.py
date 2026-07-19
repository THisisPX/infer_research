"""Kill experiment v2: NVFP4 vs BF16 on MATH-500.
MATH-500 has 5 difficulty levels → can test if quant degradation is severity-dependent.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 run_math_eval.py \
      --model /workspace/volume/pengxiong/models/Qwen3-8B-NVFP4 \
      --label NVFP4

    CUDA_VISIBLE_DEVICES=0 python3 run_math_eval.py \
      --model /workspace/volume/distributed-training-softdata/models/Qwen3-8B \
      --label BF16

    python3 run_math_eval.py --compare results/aime_eval
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

OUTPUT_DIR = Path("results/aime_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# MATH-500 loader
# ═════════════════════════════════════════════════════════════════════

def load_math500(path: str = "/workspace/volume/pengxiong/datasets/MATH-500/test.jsonl") -> List[Dict]:
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    # Group by level for stratified analysis
    levels = {}
    for p in problems:
        lv = p.get("level", 0)
        if lv not in levels:
            levels[lv] = 0
        levels[lv] += 1
    print(f"Loaded {len(problems)} MATH-500 problems")
    print(f"  Level distribution: {dict(sorted(levels.items()))}")
    return problems


# ═════════════════════════════════════════════════════════════════════
# Answer extraction for MATH (LaTeX / math expressions)
# ═════════════════════════════════════════════════════════════════════

def extract_math_answer(text: str) -> Optional[str]:
    """Extract final answer from model output for MATH problems.

    MATH answers are LaTeX expressions: e.g. \\left(3, \\frac{\\pi}{2}\\right)
    Strategy: extract \boxed{...} content, normalize whitespace.
    """
    # Try \boxed{...} with nested braces
    boxed = _extract_boxed(text)
    if boxed:
        return normalize_answer(boxed)

    # Try "answer is" / "final answer" pattern
    m = re.search(
        r'(?:answer|final answer|result)\s*(?:is|:|=)\s*\$?\s*(.+?)(?:\$|\.|\n|$)',
        text[-2000:], re.IGNORECASE
    )
    if m:
        return normalize_answer(m.group(1).strip())

    # Fallback: last \boxed{} or last $...$ in text
    math_exprs = re.findall(r'\$(.+?)\$', text)
    if math_exprs:
        return normalize_answer(math_exprs[-1])

    return None


def _extract_boxed(text: str) -> Optional[str]:
    """Extract content of \boxed{...} handling nested braces."""
    # Find last \boxed{...} with balanced braces
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    for m in reversed(matches):
        start = m.end() - 1  # position of {
        depth = 1
        i = start + 1
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            return text[start+1:i-1]
    return None


def normalize_answer(s: str) -> str:
    """Normalize math answer for comparison."""
    s = s.strip()
    # Remove surrounding $ signs
    s = re.sub(r'^\$\s*', '', s)
    s = re.sub(r'\s*\$$', '', s)
    # Collapse whitespace
    s = ' '.join(s.split())
    # Remove trailing punctuation
    s = s.rstrip('.,;')
    return s


def answers_match(pred: Optional[str], label: str) -> bool:
    """Check if predicted answer matches the label."""
    if pred is None:
        return False
    pred_norm = normalize_answer(pred)
    label_norm = normalize_answer(label)
    return pred_norm == label_norm


# ═════════════════════════════════════════════════════════════════════
# Prompt formatting
# ═════════════════════════════════════════════════════════════════════

MATH_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{}."
)


def format_math_prompt(problem: Dict) -> str:
    """Format MATH problem as Qwen3 chat template."""
    content = problem["problem"]
    return (
        f"<|im_start|>system\n{MATH_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ═════════════════════════════════════════════════════════════════════
# vLLM inference
# ═════════════════════════════════════════════════════════════════════

def run_vllm_eval(
    model_path: str,
    problems: List[Dict],
    max_tokens: int = 4096,   # MATH answers tend to be shorter than AIME
    temperature: float = 0.0,
    tensor_parallel: int = 1,
    label: str = "",
) -> Dict:
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"MATH-500 Eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Max tokens: {max_tokens}, TP: {tensor_parallel}")
    print(f"{'='*60}")

    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        max_model_len=12288,
    )
    print(f"  Model loaded in {time.time()-t0:.0f}s")

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0 if temperature == 0.0 else 0.95,
    )

    prompts = [format_math_prompt(p) for p in problems]
    tokenizer = llm.get_tokenizer()
    prompt_lengths = [len(tokenizer.encode(p)) for p in prompts]
    print(f"  Prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)}, mean={np.mean(prompt_lengths):.0f}")

    print(f"  Generating {len(prompts)} responses...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - t0
    print(f"  Done in {gen_time:.0f}s ({gen_time/len(prompts):.1f}s/problem)")

    results = []
    correct = 0
    total = 0
    total_out_tok = 0

    for i, (prob, output) in enumerate(zip(problems, outputs)):
        response = output.outputs[0].text
        out_tok = len(output.outputs[0].token_ids)
        total_out_tok += out_tok

        pred = extract_math_answer(response)
        label_val = prob["answer"]
        is_correct = answers_match(pred, label_val)
        is_trunc = out_tok >= max_tokens - 5

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": i,
            "level": prob.get("level", 0),
            "subject": prob.get("subject", ""),
            "label": label_val,
            "prediction": pred,
            "correct": is_correct,
            "is_truncated": is_trunc,
            "prompt_tokens": prompt_lengths[i],
            "output_tokens": out_tok,
            "total_tokens": prompt_lengths[i] + out_tok,
            "response": response,
        })

    accuracy = correct / total if total > 0 else 0
    n_trunc = sum(1 for r in results if r["is_truncated"])
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.2%}")
    if n_trunc:
        print(f"  Truncated: {n_trunc}/{total}")

    # ── By difficulty level ──
    print(f"\n  Accuracy by Level:")
    print(f"  {'Level':>8s}  {'N':>5s}  {'Correct':>8s}  {'Accuracy':>10s}  {'Avg Tok':>10s}")
    print(f"  {'─'*50}")
    level_stats = {}
    for lv in sorted(set(p.get("level", 0) for p in problems)):
        lr = [r for r in results if r["level"] == lv]
        if not lr:
            continue
        acc = sum(1 for r in lr if r["correct"]) / len(lr)
        avg_tok = np.mean([r["output_tokens"] for r in lr])
        print(f"  {lv:>8d}  {len(lr):>5d}  {sum(1 for r in lr if r['correct']):>8d}  {acc:>10.2%}  {avg_tok:>10.0f}")
        level_stats[f"level_{lv}"] = {
            "n": len(lr), "correct": sum(1 for r in lr if r["correct"]),
            "accuracy": round(acc, 4), "avg_output_tokens": round(avg_tok, 1),
        }

    # ── By output token length bucket ──
    buckets = [(0, 512), (512, 1024), (1024, 2048), (2048, 3072), (3072, 4096), (4096, float('inf'))]
    bucket_stats = []
    for lo, hi in buckets:
        in_range = [r for r in results if lo <= r["output_tokens"] < hi]
        if not in_range:
            continue
        acc = sum(1 for r in in_range if r["correct"]) / len(in_range)
        hi_str = f"{int(hi)}" if hi < float('inf') else "max"
        bucket_stats.append({
            "range": f"[{int(lo)}, {hi_str})", "n": len(in_range),
            "accuracy": round(acc, 4),
            "avg_output_tokens": round(np.mean([r["output_tokens"] for r in in_range]), 1),
        })
        print(f"    [{int(lo):>4}, {hi_str:>5}): n={len(in_range):>3d}  acc={acc:.2%}")

    out = {
        "model": model_path, "label": label,
        "config": {"max_tokens": max_tokens, "temperature": temperature, "tensor_parallel": tensor_parallel},
        "accuracy": round(accuracy, 4), "correct": correct, "total": total,
        "avg_output_tokens": round(total_out_tok / total, 1) if total > 0 else 0,
        "generation_time_s": round(gen_time, 1),
        "n_truncated": n_trunc,
        "level_stats": level_stats,
        "bucket_stats": bucket_stats,
        "results": results,
    }

    model_name = Path(model_path).name
    out_path = OUTPUT_DIR / f"math500_{label}_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  → Saved to {out_path}")
    return out


# ═════════════════════════════════════════════════════════════════════
# Compare
# ═════════════════════════════════════════════════════════════════════

def run_compare(results_dir: str = "results/aime_eval"):
    results_dir = Path(results_dir)
    all_data = []
    for path in sorted(results_dir.glob("math500_*.json")):
        with open(path) as f:
            data = json.load(f)
        all_data.append(data)
        label = data.get("label", path.stem)
        n_trunc = data.get("n_truncated", "?")
        print(f"  {label:<10s}  Acc: {data['accuracy']:.2%}  "
              f"({data['correct']}/{data['total']})  "
              f"Avg out: {data['avg_output_tokens']:.0f} tok  "
              f"Trunc: {n_trunc}")

    if not all_data:
        print("  No math500 results found")
        return

    # ── Level comparison ──
    print(f"\n  {'─'*80}")
    print(f"  Accuracy by Difficulty Level")
    print(f"  {'─'*80}")
    all_levels = set()
    for d in all_data:
        for k in d.get("level_stats", {}):
            all_levels.add(k)
    for lv in sorted(all_levels):
        row = f"  {lv:>10s}"
        for d in all_data:
            ls = d.get("level_stats", {}).get(lv, {})
            if ls:
                row += f"  {ls['accuracy']:.2%} (n={ls['n']})"
            else:
                row += "  —"
        print(row)

    # ── Per-problem ──
    if len(all_data) == 2:
        d0, d1 = all_data[0], all_data[1]
        r0 = {r["idx"]: r for r in d0["results"]}
        r1 = {r["idx"]: r for r in d1["results"]}
        both_correct = both_wrong = only_0 = only_1 = 0
        skip_length_gap = []
        for idx in sorted(r0.keys()):
            if idx not in r1:
                continue
            c0, c1 = r0[idx]["correct"], r1[idx]["correct"]
            t0, t1 = r0[idx]["output_tokens"], r1[idx]["output_tokens"]
            if c0 and c1:
                both_correct += 1
            elif not c0 and not c1:
                both_wrong += 1
            elif c0 and not c1:
                only_0 += 1
                skip_length_gap.append(
                    f"    #{idx} L{r0[idx]['level']}: {d0['label']}=✓({t0} tok)  {d1['label']}=✗({t1} tok)  gap={t1-t0:+d}"
                )
            else:
                only_1 += 1
                skip_length_gap.append(
                    f"    #{idx} L{r1[idx]['level']}: {d0['label']}=✗({t0} tok)  {d1['label']}=✓({t1} tok)  gap={t1-t0:+d}"
                )
        print(f"\n  {'─'*50}")
        print(f"  Per-Problem Matrix: {d0['label']} vs {d1['label']}")
        print(f"  {'─'*50}")
        print(f"    Both correct: {both_correct}")
        print(f"    Both wrong:   {both_wrong}")
        print(f"    Only {d0['label']}: {only_0}")
        print(f"    Only {d1['label']}: {only_1}")
        if skip_length_gap:
            print(f"\n    Disagreement cases:")
            for s in skip_length_gap:
                print(s)

        # ── Level-stratified disagreement ──
        print(f"\n  {'─'*50}")
        print(f"  Disagreement by Level")
        print(f"  {'─'*50}")
        for lv in range(1, 6):
            lv_cases = [(idx, r0[idx], r1[idx]) for idx in sorted(r0.keys())
                        if idx in r1 and r0[idx]["level"] == lv and r0[idx]["correct"] != r1[idx]["correct"]]
            if not lv_cases:
                continue
            only0_lv = sum(1 for _, a, b in lv_cases if a["correct"])
            only1_lv = sum(1 for _, a, b in lv_cases if b["correct"])
            print(f"    Level {lv}: {len(lv_cases)} disagreements  "
                  f"({d0['label']}={only0_lv}, {d1['label']}={only1_lv})")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MATH-500: NVFP4 vs BF16 stratified accuracy")
    parser.add_argument("--model", type=str)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--data", type=str, default="/workspace/volume/pengxiong/datasets/MATH-500/test.jsonl")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--compare", type=str, help="Compare results directory")
    args = parser.parse_args()

    if args.compare:
        run_compare(args.compare)
    elif args.model:
        problems = load_math500(args.data)
        label = args.label or Path(args.model).name
        run_vllm_eval(
            model_path=args.model, problems=problems,
            max_tokens=args.max_tokens, tensor_parallel=args.tensor_parallel,
            label=label,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
