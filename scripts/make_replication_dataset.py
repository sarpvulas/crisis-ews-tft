"""Build the mock (synthetic surrogate) datasets committed under data/mock/.

Bloomberg's licence does not allow redistributing the underlying series, so the
repository ships mock surrogates of the two datasets the README quickstart
needs: the canonical six-fold single-market directory and the nine-market panel.
The mock values are generated from the licensed pull by heavy seeded stochastic
perturbation; they do not correspond to any real market quote and must not be
used as market data. They exist solely so the pipeline, models, and evaluation
protocol can be executed end-to-end.

Perturbation scheme
-------------------
* Every continuous FEATURE column receives additive stationary AR(1) noise
  (persistence ``AR1_RHO``) with unconditional standard deviation
  ``NOISE_FRAC`` x that column's full-sample std. Autocorrelated noise is
  deliberate: unlike iid noise, it occupies the same frequency band as the
  underlying series, so the original values cannot be recovered by smoothing.
* The noise draw is keyed on (market, date, column): a given trading day
  receives the identical perturbation in task_b_features, every fold's
  train/val/test file, etc. The split structure therefore stays exactly
  consistent -- no artificial train/test discrepancy is introduced.
* Labels (``ews_label``, ``in_crisis``, ``regime_label``), the ``drawdown``
  regression target, ``date`` and ``market_id`` are copied byte-identical;
  the crisis chronology itself is public knowledge.
* Everything is seeded; re-running the script reproduces the same files.

Usage (from the repo root, with the original data present under data/shared/):
    python scripts/make_replication_dataset.py
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd

SEED = 20260805
NOISE_FRAC = 0.20  # unconditional noise std as a fraction of each column's std
AR1_RHO = 0.90     # noise persistence; high = not removable by smoothing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO, "data", "shared", "processed")
OUT_ROOT = os.path.join(REPO, "data", "mock")

# (relative directory, reference file used for column stds + the date universe)
DATASETS = [
    ("taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f", "task_b_features.parquet"),
    ("panel_dd10_rp252", "panel.parquet"),
]

PROTECTED = {"date", "market_id", "ews_label", "in_crisis", "drawdown", "regime_label"}


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """(market, date) key columns; single-market files get a constant market."""
    market = df["market_id"] if "market_id" in df.columns else "SPX"
    return pd.DataFrame({"_market": market, "_date": df["date"].to_numpy()})


def _ar1_noise(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    """Stationary AR(1) series with unconditional std ``sigma``."""
    eps = rng.normal(0.0, sigma * np.sqrt(1.0 - AR1_RHO ** 2), n)
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, sigma)
    for t in range(1, n):
        noise[t] = AR1_RHO * noise[t - 1] + eps[t]
    return noise


def _noise_table(universe: pd.DataFrame, columns: list[str],
                 stds: pd.Series) -> pd.DataFrame:
    """One noise value per (market, date) per feature column, deterministically.

    Sorting by (market, date) makes the AR(1) recursion run along each market's
    calendar; the chain restarts fresh at every market boundary.
    """
    table = universe.sort_values(["_market", "_date"]).reset_index(drop=True)
    run_lengths = table.groupby("_market", sort=True).size()
    for col in columns:
        rng = np.random.default_rng([SEED, zlib.crc32(col.encode())])
        sigma = NOISE_FRAC * stds[col]
        table[col] = np.concatenate([_ar1_noise(rng, sigma, n) for n in run_lengths])
    return table


def perturb_dataset(rel_dir: str, ref_name: str) -> None:
    src_dir = os.path.join(SRC_ROOT, rel_dir)
    out_dir = os.path.join(OUT_ROOT, rel_dir)
    ref = pd.read_parquet(os.path.join(src_dir, ref_name))

    feature_cols = [c for c in ref.columns
                    if c not in PROTECTED and pd.api.types.is_float_dtype(ref[c])]
    stds = ref[feature_cols].std(skipna=True)
    feature_cols = [c for c in feature_cols if np.isfinite(stds[c]) and stds[c] > 0]

    files = sorted(
        os.path.join(root, f)
        for root, _dirs, names in os.walk(src_dir)
        for f in names if f.endswith(".parquet")
    )
    # The (market, date) universe spans every file, so any row anywhere gets a key.
    universe = (pd.concat([_key_frame(pd.read_parquet(f, columns=None)[["date"] +
                (["market_id"] if "market_id" in ref.columns else [])])
                for f in files])
                .drop_duplicates().reset_index(drop=True))
    noise = _noise_table(universe, feature_cols, stds)

    for path in files:
        df = pd.read_parquet(path)
        keys = _key_frame(df)
        aligned = keys.merge(noise, on=["_market", "_date"], how="left", sort=False)
        if len(aligned) != len(df):
            raise RuntimeError(f"key alignment changed row count for {path}")
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col] + aligned[col].to_numpy()
        rel = os.path.relpath(path, src_dir)
        dest = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        df.to_parquet(dest, index=False)
        print(f"  wrote {os.path.join(rel_dir, rel)}  ({len(df)} rows)")


def verify_dataset(rel_dir: str) -> None:
    """Perturbed files must differ only in feature values, and only slightly."""
    src_dir = os.path.join(SRC_ROOT, rel_dir)
    out_dir = os.path.join(OUT_ROOT, rel_dir)
    for root, _dirs, names in os.walk(src_dir):
        for name in sorted(n for n in names if n.endswith(".parquet")):
            src = pd.read_parquet(os.path.join(root, name))
            rel = os.path.relpath(os.path.join(root, name), src_dir)
            out = pd.read_parquet(os.path.join(out_dir, rel))
            assert list(src.columns) == list(out.columns), rel
            assert src.shape == out.shape, rel
            assert (src.dtypes == out.dtypes).all(), rel
            for col in PROTECTED & set(src.columns):
                pd.testing.assert_series_equal(src[col], out[col], obj=f"{rel}:{col}")
            deltas = []
            for col in src.columns:
                if col in PROTECTED or not pd.api.types.is_float_dtype(src[col]):
                    continue
                d = (out[col] - src[col]).abs()
                s = src[col].std(skipna=True)
                if np.isfinite(s) and s > 0:
                    assert d.max() > 0, f"{rel}:{col} unchanged"
                    deltas.append(d.mean() / s)
            print(f"  ok {os.path.join(rel_dir, rel)}: "
                  f"mean |delta|/std = {np.mean(deltas):.4f}")


def main() -> int:
    for rel_dir, ref_name in DATASETS:
        if not os.path.isdir(os.path.join(SRC_ROOT, rel_dir)):
            print(f"ERROR: source dataset missing: {rel_dir}", file=sys.stderr)
            return 1
        print(f"perturbing {rel_dir} (noise = {NOISE_FRAC:.0%} of column std, "
              f"seed = {SEED})")
        perturb_dataset(rel_dir, ref_name)
    print("verifying")
    for rel_dir, _ in DATASETS:
        verify_dataset(rel_dir)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
