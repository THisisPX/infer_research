"""Compare NVFP4 vs BF16 outputs on disagreement cases (where one got it right, the other didn't).

Usage:
    python3 compare_outputs.py results/aime_eval
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_result(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def find_disagreement_cases(d0: Dict, d1: Dict) -> List[Dict]:
    """Find cases where correctness differs between two runs."""
    r0 = {r["idx"]: r for r in d0["results"]}
    r1 = {r["idx"]: r for r in d1["results"]}

    cases = []
    for idx in sorted(r0.keys()):
        if idx not in r1:
            continue
        c0, c1 = r0[idx]["correct"], r1[idx]["correct"]
        if c0 != c1:
            cases.append({
                "idx": idx,
                f"{d0['label']}_correct": c0,
                f"{d1['label']}_correct": c1,
                f"{d0['label']}_tokens": r0[idx]["output_tokens"],
                f"{d1['label']}_tokens": r1[idx]["output_tokens"],
                f"{d0['label']}_pred": r0[idx]["prediction"],
                f"{d1['label']}_pred": r1[idx]["prediction"],
                "label": r0[idx]["label"],
                f"{d0['label']}_output": r0[idx]["response"],
                f"{d1['label']}_output": r1[idx]["response"],
            })
    return cases


def extract_reasoning_steps(text: str, max_steps: int = 20) -> List[str]:
    """Extract key reasoning steps from model output.

    Splits on sentence boundaries (period + newline or double newline)
    and returns the most informative lines.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    steps = []
    for p in paragraphs:
        # Further split long paragraphs
        if len(p) > 500:
            # Split on numbered items or sentences
            sub = re.split(r'(?<=\d\.)\s+|(?<=\?\s)|(?<=\.)\s+(?=[A-Z])', p)
            steps.extend([s.strip() for s in sub if len(s.strip()) > 20])
        else:
            steps.append(p)

    # Take a representative sample: first 3, middle 3, last 5
    n = len(steps)
    if n <= 11:
        return steps

    sampled = steps[:3]
    mid_start = max(3, n // 2 - 1)
    sampled += [f"... [{mid_start}/{n}] ..."]
    sampled += steps[mid_start:mid_start + 2]
    sampled += [f"... [{n - 5}/{n}] ..."]
    sampled += steps[-5:]
    return sampled


def count_markers(text: str) -> Dict:
    """Count reasoning-related markers in output."""
    return {
        "sentences": len(re.split(r'[.!?]\s+', text)),
        "paragraphs": len(re.split(r'\n\n+', text)),
        "boxed_answers": len(re.findall(r'\\boxed', text)),
        "therefore": len(re.findall(r'(?:therefore|thus|hence|so the answer)', text, re.I)),
        "checking_steps": len(re.findall(r'(?:wait|let me|check|verify|hmm|actually)', text, re.I)),
        "calculations": len(re.findall(r'(?:\\frac|\\sum|\\int|\\sqrt|\\pi|= \d)', text)),
        "numbered_steps": len(re.findall(r'^\d+\.', text, re.MULTILINE)),
    }


def print_case_comparison(case: Dict, label_0: str, label_1: str, problem_text: str, max_output_chars: int = 30000):
    """Print a side-by-side comparison of one disagreement case."""
    idx = case["idx"]
    answer = case["label"]
    pred_0 = case[f"{label_0}_pred"]
    pred_1 = case[f"{label_1}_pred"]
    tok_0 = case[f"{label_0}_tokens"]
    tok_1 = case[f"{label_1}_tokens"]
    correct_0 = case[f"{label_0}_correct"]
    correct_1 = case[f"{label_1}_correct"]
    out_0 = case[f"{label_0}_output"]
    out_1 = case[f"{label_1}_output"]

    print(f"\n{'='*100}")
    print(f"CASE #{idx}  |  Correct answer: {answer}  |  "
          f"{label_0}: {'✓' if correct_0 else '✗'}(→{pred_0})  "
          f"{label_1}: {'✓' if correct_1 else '✗'}(→{pred_1})")
    print(f"{'='*100}")

    # Problem summary (first 300 chars)
    print(f"\n{'─'*100}")
    print(f"PROBLEM: {problem_text[:300].strip()}...")
    print(f"{'─'*100}")

    # Token counts and markers
    markers_0 = count_markers(out_0)
    markers_1 = count_markers(out_1)

    print(f"\n{'─'*100}")
    print(f"STATS:")
    print(f"  {'':30s}  {label_0:<15s}  {label_1:<15s}")
    print(f"  {'Output tokens':30s}  {tok_0:<15d}  {tok_1:<15d}")
    print(f"  {'Prediction':30s}  {str(pred_0):<15s}  {str(pred_1):<15s}")
    print(f"  {'Sentences':30s}  {markers_0['sentences']:<15d}  {markers_1['sentences']:<15d}")
    print(f"  {'Paragraphs':30s}  {markers_0['paragraphs']:<15d}  {markers_1['paragraphs']:<15d}")
    print(f"  {'\\boxed{} occurrences':30s}  {markers_0['boxed_answers']:<15d}  {markers_1['boxed_answers']:<15d}")
    print(f"  {'Checking steps':30s}  {markers_0['checking_steps']:<15d}  {markers_1['checking_steps']:<15d}")
    print(f"  {'Therefore/Thus/Hence':30s}  {markers_0['therefore']:<15d}  {markers_1['therefore']:<15d}")
    print(f"  {'LaTeX calculations':30s}  {markers_0['calculations']:<15d}  {markers_1['calculations']:<15d}")
    print(f"{'─'*100}")

    # Side-by-side output (truncated to max_output_chars each)
    print(f"\n{'─'*100}")
    print(f"OUTPUT COMPARISON")
    print(f"{'─'*100}")

    # Show first 3000 chars side by side
    chunk_0 = out_0[:max_output_chars]
    chunk_1 = out_1[:max_output_chars]

    lines_0 = chunk_0.split('\n')
    lines_1 = chunk_1.split('\n')

    width_0 = 48
    width_1 = 48
    separator = f"  {'─'*width_0}  │  {'─'*width_1}"

    print(f"  {label_0:<{width_0}s}  │  {label_1:<{width_1}s}")
    print(separator)

    max_lines = max(len(lines_0), len(lines_1))
    for i in range(min(max_lines, 100)):  # limit to 100 lines
        l0 = lines_0[i] if i < len(lines_0) else ""
        l1 = lines_1[i] if i < len(lines_1) else ""

        # Truncate long lines
        l0 = l0[:width_0 - 1] + "…" if len(l0) > width_0 else l0
        l1 = l1[:width_1 - 1] + "…" if len(l1) > width_1 else l1

        # Highlight disagreement in answers
        print(f"  {l0:<{width_0}s}  │  {l1:<{width_1}s}")

    # If truncated
    total_0 = len(out_0)
    total_1 = len(out_1)
    if total_0 > max_output_chars or total_1 > max_output_chars:
        print(f"\n  [... truncated: {label_0}={total_0} chars, {label_1}={total_1} chars ...]")

    # ── Last 2000 chars (the concluding section is most informative) ──
    print(f"\n{'─'*100}")
    print(f"LAST 2000 CHARACTERS (conclusion / final reasoning)")
    print(f"{'─'*100}")
    tail_0 = out_0[-2000:] if len(out_0) > 2000 else out_0
    tail_1 = out_1[-2000:] if len(out_1) > 2000 else out_1

    tail_lines_0 = tail_0.split('\n')
    tail_lines_1 = tail_1.split('\n')
    max_tail = max(len(tail_lines_0), len(tail_lines_1))
    for i in range(min(max_tail, 50)):
        l0 = tail_lines_0[i] if i < len(tail_lines_0) else ""
        l1 = tail_lines_1[i] if i < len(tail_lines_1) else ""
        l0 = l0[:width_0 - 1] + "…" if len(l0) > width_0 else l0
        l1 = l1[:width_1 - 1] + "…" if len(l1) > width_1 else l1
        print(f"  {l0:<{width_0}s}  │  {l1:<{width_1}s}")


def main():
    parser = argparse.ArgumentParser(description="Compare NVFP4 vs BF16 disagreement outputs")
    parser.add_argument("results_dir", type=str, default="results/aime_eval")
    parser.add_argument("--label-0", type=str, default="BF16")
    parser.add_argument("--label-1", type=str, default="NVFP4")
    parser.add_argument("--output", type=str, default="", help="Save comparison to file")
    parser.add_argument("--max-chars", type=int, default=30000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Find result files
    files = list(results_dir.glob("aime2024_*.json"))
    if len(files) < 2:
        print(f"Need 2 result files in {results_dir}, found {len(files)}: {[f.name for f in files]}")
        sys.exit(1)

    # Match files by label
    d0_path = d1_path = None
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        if data.get("label") == args.label_0:
            d0_path = f
        elif data.get("label") == args.label_1:
            d1_path = f

    if d0_path is None and len(files) >= 2:
        # Fallback: use first two files
        d0_path = files[0]
        d1_path = files[-1]

    if d0_path is None or d1_path is None:
        print(f"Could not find both {args.label_0} and {args.label_1} results")
        sys.exit(1)

    print(f"Loading {args.label_0}: {d0_path.name}")
    d0 = load_result(str(d0_path))
    print(f"Loading {args.label_1}: {d1_path.name}")
    d1 = load_result(str(d1_path))

    # Find disagreement cases
    cases = find_disagreement_cases(d0, d1)
    print(f"\nFound {len(cases)} disagreement cases")

    # Load AIME problems for context
    import json as json_mod
    aime_path = "/workspace/volume/pengxiong/datasets/aime-2024/aime-2024.jsonl"
    problems = {}
    if Path(aime_path).exists():
        with open(aime_path) as f:
            for line in f:
                prob = json_mod.loads(line)
                # Extract just the problem text
                for msg in prob["prompt"]:
                    if msg["role"] == "user":
                        problems[len(problems)] = msg["content"]
                        break
                else:
                    problems[len(problems)] = prob["prompt"][0].get("content", "")

    # Print each case
    all_output = []
    for i, case in enumerate(cases):
        prob_text = problems.get(case["idx"], "[problem not found]")
        print_case_comparison(case, args.label_0, args.label_1, prob_text, args.max_chars)

    # ── Summary statistics ──
    print(f"\n\n{'='*100}")
    print(f"SUMMARY: Disagreement Pattern Analysis")
    print(f"{'='*100}")

    only_0 = sum(1 for c in cases if c[f"{args.label_0}_correct"] and not c[f"{args.label_1}_correct"])
    only_1 = sum(1 for c in cases if c[f"{args.label_1}_correct"] and not c[f"{args.label_0}_correct"])
    print(f"  Only {args.label_0} correct: {only_0}")
    print(f"  Only {args.label_1} correct: {only_1}")

    avg_t0 = sum(c[f"{args.label_0}_tokens"] for c in cases) / len(cases)
    avg_t1 = sum(c[f"{args.label_1}_tokens"] for c in cases) / len(cases)
    print(f"  Avg output tokens ({args.label_0}): {avg_t0:.0f}")
    print(f"  Avg output tokens ({args.label_1}): {avg_t1:.0f}")

    # Token gap in Only-0 cases
    if only_0 > 0:
        only0 = [c for c in cases if c[f"{args.label_0}_correct"] and not c[f"{args.label_1}_correct"]]
        avg_gap = sum(c[f"{args.label_1}_tokens"] - c[f"{args.label_0}_tokens"] for c in only0) / only_0
        print(f"\n  When {args.label_0} wins: {args.label_1} generates {avg_gap:+.0f} tokens more on average")
        for c in only0:
            print(f"    #{c['idx']}: {args.label_1} +{c[f'{args.label_1}_tokens'] - c[f'{args.label_0}_tokens']} tokens "
                  f"({c[f'{args.label_0}_tokens']} vs {c[f'{args.label_1}_tokens']})")


if __name__ == "__main__":
    main()
