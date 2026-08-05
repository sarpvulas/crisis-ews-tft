#!/usr/bin/env python
"""Final thesis-grade comparison: paired Wilcoxon + bootstrap CI.

Reads per-seed result JSONs for multiple models on the same task/split, runs
the per-seed paired Wilcoxon signed-rank test for every (focal, baseline)
pair, and prints a publication-style table with means, stds, and p-values.

Usage:
  python scripts/compare_models.py \\
      --results-dir /scratch/.../results \\
      --tag stage3-multiseed \\
      --task B \\
      --focal tdt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.stats import paired_wilcoxon


METRICS = ("pr_auc", "roc_auc", "calibration_error")


def _load_per_seed(results_dir: str, tag: str, task: str) -> Dict[str, Dict[int, dict]]:
    """Group result JSONs by (model, seed). Returns {model: {seed: test_metrics_dict}}."""
    out: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(path) as f:
                blob = json.load(f)
        except Exception:
            continue
        if blob.get("tag") != tag or blob.get("task") != task:
            continue
        seed = int(blob.get("seed", -1))
        model = blob.get("model", "?")
        out[model][seed] = blob.get("test_metrics", {})
    return dict(out)


def _summary(per_seed: Dict[int, dict], metric: str) -> Dict[str, float]:
    vals = [v.get(metric) for v in per_seed.values() if isinstance(v.get(metric), (int, float))]
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _paired_seeds(a: Dict[int, dict], b: Dict[int, dict], metric: str) -> tuple:
    """Return (scores_a, scores_b) arrays aligned by seed."""
    common = sorted(set(a.keys()) & set(b.keys()))
    sa, sb = [], []
    for s in common:
        va = a[s].get(metric)
        vb = b[s].get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if not math.isnan(va) and not math.isnan(vb):
                sa.append(va)
                sb.append(vb)
    return np.array(sa), np.array(sb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--tag", required=True, help="Exact tag to filter by.")
    parser.add_argument("--task", default="B")
    parser.add_argument("--focal", required=True, help="Model whose performance is on trial.")
    parser.add_argument("--metric", default="pr_auc")
    args = parser.parse_args()

    grouped = _load_per_seed(args.results_dir, args.tag, args.task)
    if args.focal not in grouped:
        print(f"ERROR: focal model '{args.focal}' not found. Have: {sorted(grouped)}")
        return

    print(f"\n=== Per-model summary on {args.metric} (test set) ===\n")
    headers = ["model", "n", "mean", "std", "min", "max"]
    rows = []
    for model, per_seed in sorted(grouped.items()):
        s = _summary(per_seed, args.metric)
        rows.append({
            "model": model, "n": s["n"], "mean": s["mean"],
            "std": s["std"], "min": s.get("min"), "max": s.get("max"),
        })
    # Sort by mean desc.
    rows.sort(key=lambda r: -(r["mean"] if not math.isnan(r["mean"]) else -1))
    w = max(len(r["model"]) for r in rows) + 2
    print(f"{'model':<{w}} {'n':>3}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
    print("-" * (w + 38))
    for r in rows:
        print(f"{r['model']:<{w}} {r['n']:>3}  "
              f"{r['mean']:>7.4f}  {r['std']:>7.4f}  "
              f"{r['min']:>7.4f}  {r['max']:>7.4f}")

    print(f"\n=== Paired Wilcoxon: focal={args.focal} vs each baseline (alternative='greater') ===\n")
    focal_per_seed = grouped[args.focal]
    print(f"{'baseline':<{w}} {'n_pairs':>7}  {'mean_diff':>9}  {'pvalue':>9}  {'verdict':>9}")
    print("-" * (w + 42))
    for model, per_seed in sorted(grouped.items()):
        if model == args.focal:
            continue
        sa, sb = _paired_seeds(focal_per_seed, per_seed, args.metric)
        if len(sa) < 2:
            print(f"{model:<{w}} {len(sa):>7}  (insufficient paired data)")
            continue
        diff = float(np.mean(sa - sb))
        try:
            res = paired_wilcoxon(sa, sb, alternative="greater")
            verdict = "**" if res.pvalue < 0.05 else "ns"
            print(f"{model:<{w}} {res.n:>7}  {diff:>+9.4f}  {res.pvalue:>9.4f}  {verdict:>9}")
        except Exception as e:
            print(f"{model:<{w}} {len(sa):>7}  paired-test error: {e}")
    print()


if __name__ == "__main__":
    main()
