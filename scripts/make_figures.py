#!/usr/bin/env python
"""Regenerate the pgfplots data files used by docs/dissertation.tex.

Writes:
  docs/figures/spx_timeline.dat   -- t, close, dd(%), ews, in_crisis (daily)
  docs/figures/drawdown_hist.dat  -- dd(%) bin centre, density

Also prints the drawdown-distribution stats and the per-fold date ranges that
back Figure (timeline), Figure (CV), Table (thresholds). Run from repo root:
    python scripts/make_figures.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10")
F6 = os.path.join(ROOT, "data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f/folds")
OUT = os.path.join(ROOT, "docs/figures")
os.makedirs(OUT, exist_ok=True)


def dec_year(d):
    return d.year + (d.dayofyear - 1) / 365.25


def main():
    df = pd.read_parquet(os.path.join(SM, "task_b_features.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    df["t"] = df["date"].apply(dec_year)

    # --- timeline ---
    out = df[["t", "close", "drawdown", "ews_label", "in_crisis"]].copy()
    out["dd"] = out["drawdown"] * 100.0
    out[["t", "close", "dd", "ews_label", "in_crisis"]].to_csv(
        os.path.join(OUT, "spx_timeline.dat"), sep=" ", index=False,
        header=["t", "close", "dd", "ews", "incrisis"], float_format="%.4f")

    # --- drawdown distribution ---
    absdd = np.abs(df["drawdown"].dropna().values) * 100.0
    mu, sd = absdd.mean(), absdd.std()
    bins = np.arange(0, 60, 1.0)
    hist, edges = np.histogram(absdd, bins=bins, density=True)
    ctr = (edges[:-1] + edges[1:]) / 2
    with open(os.path.join(OUT, "drawdown_hist.dat"), "w") as f:
        f.write("dd density\n")
        for c, h in zip(ctr, hist):
            f.write("%.2f %.5f\n" % (c, h))

    print("|dd|%% mean=%.3f std=%.3f" % (mu, sd))
    for thr in [5, 10, 12, 15, 20, 30]:
        print("  thr %2d%%  exceed=%4.1f%%  z=%+.2f"
              % (thr, (absdd >= thr).mean() * 100, (thr - mu) / sd))

    # --- fold date ranges ---
    print("\nfold date ranges (6f):")
    for fl in ["gfc", "eurozone", "china2015", "selloff2018", "covid", "bear2022"]:
        rng = {}
        for sp in ["train", "val", "test"]:
            p = os.path.join(F6, "%s_%s.parquet" % (fl, sp))
            if os.path.exists(p):
                d = pd.read_parquet(p)
                d["date"] = pd.to_datetime(d["date"])
                rng[sp] = (d["date"].min().date(), d["date"].max().date())
        print("  %-12s %s" % (fl, rng))


if __name__ == "__main__":
    main()
