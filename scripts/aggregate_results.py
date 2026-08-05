#!/usr/bin/env python
"""Aggregate per-run JSON results into a comparison table.

Walks a results dir, parses every JSON produced by `scripts/run_experiment.py`,
and prints (or writes) a sorted table by chosen metric. Filters by tag / task
/ split so you can extract one screen / one sweep at a time.

Usage:
  # Print Stage-1 screen results, sorted by test PR-AUC desc
  python scripts/aggregate_results.py /scratch/.../results --tag stage1-screen --task B --metric pr_auc

  # Write CSV for spreadsheet import
  python scripts/aggregate_results.py results --tag stage1-screen --csv stage1.csv

  # Group across seeds for a single architecture (mean ± std)
  python scripts/aggregate_results.py results --tag stage3-main --groupby model
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional


METRICS_OF_INTEREST = (
    "roc_auc", "pr_auc", "calibration_error", "brier_score",
)


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"warn: failed to load {path}: {e}", file=sys.stderr)
        return None


def _matches(
    blob: Dict[str, Any],
    tag: Optional[str],
    task: Optional[str],
    split: Optional[str],
    tag_prefix: Optional[str] = None,
) -> bool:
    if tag is not None and blob.get("tag") != tag:
        return False
    if tag_prefix is not None and not str(blob.get("tag", "")).startswith(tag_prefix):
        return False
    if task is not None and blob.get("task") != task:
        return False
    if split is not None and blob.get("split") != split:
        return False
    return True


def _row(blob: Dict[str, Any]) -> Dict[str, Any]:
    test = blob.get("test_metrics", {})
    val = blob.get("val_metrics", {})
    return {
        "model": blob.get("model", "?"),
        "task": blob.get("task", "?"),
        "split": blob.get("split", "?"),
        "seed": blob.get("seed", -1),
        "tag": blob.get("tag", ""),
        "n_params": blob.get("n_params", 0),
        "wall_time_s": blob.get("wall_time_s", float("nan")),
        "final_epoch": blob.get("final_epoch", 0),
        **{f"val_{m}": val.get(m, float("nan")) for m in METRICS_OF_INTEREST},
        **{f"test_{m}": test.get(m, float("nan")) for m in METRICS_OF_INTEREST},
    }


def _format_table(rows: List[Dict[str, Any]], sort_by: str, columns: List[str]) -> str:
    if not rows:
        return "(no rows)"
    rows = sorted(rows, key=lambda r: (-(r.get(sort_by) if r.get(sort_by) is not None and not math.isnan(r.get(sort_by, float("nan"))) else -1)))
    widths = {c: max(len(c), max(len(_fmt_cell(r.get(c))) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    body = "\n".join(" | ".join(_fmt_cell(r.get(c)).ljust(widths[c]) for c in columns) for r in rows)
    return f"{header}\n{sep}\n{body}"


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.4f}"
    return str(v)


def _group(rows: List[Dict[str, Any]], by: str) -> List[Dict[str, Any]]:
    """Group rows by `by` and return mean ± std for each numeric column."""
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(by, "?")].append(r)
    out = []
    for key, grp in groups.items():
        agg = {by: key, "n_runs": len(grp)}
        for c in [f"val_{m}" for m in METRICS_OF_INTEREST] + [f"test_{m}" for m in METRICS_OF_INTEREST] + ["wall_time_s", "n_params"]:
            vals = [r.get(c) for r in grp if isinstance(r.get(c), (int, float)) and not math.isnan(r.get(c, float("nan")))]
            if not vals:
                agg[c] = float("nan")
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0
            std = math.sqrt(var)
            agg[c] = mean
            agg[f"{c}_std"] = std
        out.append(agg)
    return out


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-run JSON results")
    parser.add_argument("results_dir", help="Directory containing per-run JSON files.")
    parser.add_argument("--tag", default=None, help="Filter by exact tag.")
    parser.add_argument("--tag-prefix", default=None,
                        help="Filter by tag prefix (e.g. 'sweep-trial' for all 30 sweep trials).")
    parser.add_argument("--task", default=None, help="Filter by task A or B.")
    parser.add_argument("--split", default=None, help="Filter by split name.")
    parser.add_argument("--metric", default="pr_auc",
                        help="Metric to sort by (e.g. pr_auc, roc_auc). Test value is used.")
    parser.add_argument("--groupby", default=None,
                        help="Group rows by this column and report mean ± std.")
    parser.add_argument("--csv", default=None, help="Write CSV here instead of printing.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    blobs = [b for b in (_load(p) for p in paths) if b is not None]
    blobs = [b for b in blobs if _matches(b, args.tag, args.task, args.split, args.tag_prefix)]
    rows = [_row(b) for b in blobs]

    if args.groupby:
        rows = _group(rows, args.groupby)
        sort_key = f"test_{args.metric}"
        columns = [args.groupby, "n_runs",
                   f"test_{args.metric}", f"test_{args.metric}_std",
                   f"val_{args.metric}", "wall_time_s", "n_params"]
    else:
        sort_key = f"test_{args.metric}"
        columns = ["model", "task", "seed", f"test_{args.metric}", f"val_{args.metric}",
                   "test_roc_auc", "test_calibration_error", "test_brier_score",
                   "wall_time_s", "final_epoch", "n_params"]

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in sorted(rows, key=lambda r: -(r.get(sort_key) or -1)):
                writer.writerow({c: r.get(c) for c in columns})
        print(f"wrote {len(rows)} rows -> {args.csv}")
    else:
        print(_format_table(rows, sort_key, columns))


if __name__ == "__main__":
    main()
