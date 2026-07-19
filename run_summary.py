"""Aggregate all AIME + MATH-500 results into a single summary table.

Usage:
    python3 run_summary.py results/aime_eval
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


OUTPUT_DIR = Path("results/summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def get_label_safe(d: Dict) -> str:
    """Extract clean label from result dict."""
    label = d.get("label", "")
    model = d.get("model", "")
    # Shorten model name
    model_name = Path(model).name if model else ""
    return f"{label}" if label else model_name


def main():
    parser = argparse.ArgumentParser(description="Aggregate all benchmark results")
    parser.add_argument("results_dir", type=str, default="results/aime_eval")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # ── Collect all results ──
    aime_files = sorted(results_dir.glob("aime2024_*.json"))
    math_files = sorted(results_dir.glob("math500_*.json"))
    all_files = aime_files + math_files

    if not all_files:
        print(f"No result files found in {results_dir}")
        print(f"  aime files: {[f.name for f in aime_files]}")
        print(f"  math files: {[f.name for f in math_files]}")
        return

    results: List[Dict] = []
    for f in all_files:
        d = load_json(f)
        d["_file"] = f.name
        results.append(d)

    # ── Identify benchmark type from file name ──
    for r in results:
        fname = r["_file"]
        if "aime2024" in fname:
            r["_benchmark"] = "AIME 2024"
            r["_n_total"] = 30
        elif "math500" in fname:
            r["_benchmark"] = "MATH-500"
            r["_n_total"] = 500
        else:
            r["_benchmark"] = fname.split("_")[0]
            r["_n_total"] = r.get("total", 0)

    # ═════════════════════════════════════════════════════════════════
    # Table 1: Overall comparison (fair pairs only)
    # ═════════════════════════════════════════════════════════════════

    print(f"\n{'='*110}")
    print("TABLE 1: Overall NVFP4 vs BF16 Accuracy")
    print(f"{'='*110}")

    # Group by (benchmark, max_tokens) for fair comparison
    groups = {}
    for r in results:
        if "_file" not in r:
            continue
        key = (r["_benchmark"], r.get("config", {}).get("max_tokens", "?"))
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    for (bench, max_tok), grp in sorted(groups.items()):
        grp.sort(key=lambda r: r.get("label", ""))
        print(f"\n  {bench} (max_tokens={max_tok})")
        header = f"  {'Model':<20s}  {'Acc':>8s}  {'Trunc':>8s}  {'AvgOut ':>8s}  {'N':>5s}"
        print(f"  {'─'*65}")
        for r in grp:
            label = get_label_safe(r)
            acc = r.get("accuracy", 0)
            trunc = r.get("n_truncated", "?")
            avg = int(r.get("avg_output_tokens", 0))
            n = r.get("total", r.get("_n_total", "?"))
            print(f"  {label:<20s}  {acc:>7.2%}  {str(trunc):>8s}  {avg:>8d}  {str(n):>5s}")
        # Show delta if we have both BF16 and NVFP4
        bf16_entries = [r for r in grp if "BF16" in get_label_safe(r)]
        nvfp4_entries = [r for r in grp if "NVFP4" in get_label_safe(r)]
        if bf16_entries and nvfp4_entries:
            bf = bf16_entries[0]
            nv = nvfp4_entries[0]
            delta = nv.get("accuracy", 0) - bf.get("accuracy", 0)
            print(f"  {'Δ (NVFP4 - BF16)':<20s}  {delta:>+7.2%}")

    # ═════════════════════════════════════════════════════════════════
    # Table 2: MATH-500 by difficulty level (BF16_8K vs NVFP4_8K)
    # ═════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*110}")
    print("TABLE 2: MATH-500 Accuracy by Difficulty Level (max_tokens=8192)")
    print(f"{'='*110}")

    # Find the 8K runs
    math_8k = [r for r in results
               if r["_benchmark"] == "MATH-500"
               and r.get("config", {}).get("max_tokens") == 8192]

    if len(math_8k) >= 2:
        math_8k.sort(key=lambda r: get_label_safe(r))
        labels = [get_label_safe(r) for r in math_8k]

        print(f"\n  {'Level':>10s}", end="")
        for lbl in labels:
            print(f"  {lbl:<14s}", end="")
        if "BF16" in "".join(labels) and "NVFP4" in "".join(labels):
            print(f"  {'Δ':>8s}", end="")
        print()

        print(f"  {'─'*10}", end="")
        for _ in labels:
            print(f"  {'─'*14}", end="")
        print(f"  {'─'*8}" if "BF16" in "".join(labels) and "NVFP4" in "".join(labels) else "")

        for lv in range(1, 6):
            key = f"level_{lv}"
            print(f"  {'Level '+str(lv):>10s}", end="")
            vals = []
            for r in math_8k:
                ls = r.get("level_stats", {}).get(key, {})
                if ls:
                    acc = ls.get("accuracy", 0)
                    n = ls.get("n", 0)
                    vals.append(acc)
                    print(f"  {acc:>6.2%} (n={n:<3d})", end="")
                else:
                    print(f"  {'—':>14s}", end="")
                    vals.append(None)
            # Delta
            bf_idx = next((i for i, lbl in enumerate(labels) if "BF16" in lbl), None)
            nv_idx = next((i for i, lbl in enumerate(labels) if "NVFP4" in lbl), None)
            if bf_idx is not None and nv_idx is not None:
                bf_val = vals[bf_idx]
                nv_val = vals[nv_idx]
                if bf_val is not None and nv_val is not None:
                    delta_lv = nv_val - bf_val
                    print(f"  {delta_lv:>+7.2%}", end="")
            print()

        # Overall row
        print(f"  {'─'*10}", end="")
        for _ in labels:
            print(f"  {'─'*14}", end="")
        print(f"  {'─'*8}" if "BF16" in "".join(labels) and "NVFP4" in "".join(labels) else "")
        print(f"  {'Overall':>10s}", end="")
        for r in math_8k:
            acc = r.get("accuracy", 0)
            print(f"  {acc:>6.2%}  (n={r.get('total',500)})", end="")
        bf = next((r for r in math_8k if "BF16" in get_label_safe(r)), None)
        nv = next((r for r in math_8k if "NVFP4" in get_label_safe(r)), None)
        if bf and nv:
            print(f"  {nv.get('accuracy',0)-bf.get('accuracy',0):>+7.2%}", end="")
        print()

    # ═════════════════════════════════════════════════════════════════
    # Table 3: Detailed stats (token counts, disagreement matrix)
    # ═════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*110}")
    print("TABLE 3: Token Efficiency & Length Analysis")
    print(f"{'='*110}")

    for (bench, max_tok), grp in sorted(groups.items()):
        grp.sort(key=lambda r: get_label_safe(r))
        print(f"\n  {bench} (max_tokens={max_tok})")
        header = (f"  {'Model':<20s}  {'AvgOut':>8s}  {'MinOut':>8s}  "
                  f"{'MaxOut':>8s}  {'Trunc':>8s}  {'Trunc%':>8s}")
        print(f"  {'─'*80}")
        for r in grp:
            label = get_label_safe(r)
            avg = int(r.get("avg_output_tokens", 0))
            results_list = r.get("results", [])
            min_out = min((x["output_tokens"] for x in results_list), default=0)
            max_out = max((x["output_tokens"] for x in results_list), default=0)
            trunc = r.get("n_truncated", 0)
            n = r.get("total", r.get("_n_total", 1))
            trunc_pct = trunc / n if n > 0 else 0
            print(f"  {label:<20s}  {avg:>8d}  {min_out:>8d}  "
                  f"{max_out:>8d}  {trunc:>8d}  {trunc_pct:>7.1%}")

    # ═════════════════════════════════════════════════════════════════
    # Table 4: AIME length-bucketed accuracy
    # ═════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*110}")
    print("TABLE 4: AIME 2024 Accuracy by Output Length Bucket")
    print(f"{'='*110}")

    aime_results = [r for r in results if r["_benchmark"] == "AIME 2024"]
    if aime_results:
        # Collect all unique buckets
        all_buckets = set()
        for r in aime_results:
            for b in r.get("bucket_stats", []):
                all_buckets.add(b["range"])
        all_buckets = sorted(all_buckets,
            key=lambda x: int(x.strip("[]()").split(",")[0].strip()))

        labels = [get_label_safe(r) for r in aime_results]
        print(f"\n  {'Bucket':>18s}", end="")
        for lbl in labels:
            print(f"  {lbl:<18s}", end="")
        print()
        print(f"  {'─'*18}", end="")
        for _ in labels:
            print(f"  {'─'*18}", end="")
        print()

        for bn in all_buckets:
            print(f"  {bn:>18s}", end="")
            for r in aime_results:
                matches = [b for b in r.get("bucket_stats", []) if b["range"] == bn]
                if matches:
                    b = matches[0]
                    print(f"  {b['accuracy']:.2%} (n={b['n']})".ljust(18), end="")
                else:
                    print(f"  {'—':>18s}", end="")
            print()

    # ═════════════════════════════════════════════════════════════════
    # Save summary JSON
    # ═════════════════════════════════════════════════════════════════

    out_path = OUTPUT_DIR / "summary.json"
    # Build a simplified structure
    summary = {
        "overall": {
            r["_file"]: {
                "benchmark": r["_benchmark"],
                "label": get_label_safe(r),
                "accuracy": r.get("accuracy", 0),
                "correct": r.get("correct", 0),
                "total": r.get("total", 0),
                "avg_output_tokens": r.get("avg_output_tokens", 0),
                "n_truncated": r.get("n_truncated", 0),
                "max_tokens": r.get("config", {}).get("max_tokens", 0),
                "level_stats": r.get("level_stats", {}),
            }
            for r in results if "_file" in r
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\n  Summary JSON → {out_path}")


if __name__ == "__main__":
    main()
