#!/usr/bin/env python
"""Download multi-market data and build panel EWS dataset."""
import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from data.pipelines.config import PipelineConfig
from data.pipelines.downloaders.yahoo import YahooDownloader
from data.pipelines.downloaders.fred import FREDDownloader
from data.pipelines.processing.panel_features import build_panel_dataset
from data.pipelines.processing.splits import walk_forward_crisis_splits
from data.pipelines.run_pipeline import _load_yahoo_csv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_panel_pipeline(cfg: PipelineConfig, output_dir: Path, skip_download: bool = False):
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    yahoo = YahooDownloader(cfg)

    # === 1. Download market indices ===
    if not skip_download:
        logger.info("Downloading market indices...")
        for name, ticker in cfg.MARKET_TICKERS.items():
            path = raw_dir / f"market_{name}.csv"
            if path.exists() and skip_download:
                continue
            df = yahoo.download_ticker(ticker, cfg.TASK_B_START, cfg.DATA_END)
            if df is not None:
                df.to_csv(path)
                logger.info(f"  Saved market_{name}.csv: {len(df)} rows")

        # Download auxiliary signals (VIX, spreads, etc.)
        logger.info("Downloading auxiliary Yahoo tickers...")
        for name, ticker in cfg.YAHOO_TICKERS.items():
            path = raw_dir / f"yahoo_{name}.csv"
            if path.exists():
                continue
            df = yahoo.download_ticker(ticker, cfg.TASK_B_START, cfg.DATA_END)
            if df is not None:
                df.to_csv(path)

        logger.info("Downloading FRED data...")
        fred = FREDDownloader(cfg)
        fred_market = fred.download_all_market()
        for name, series in fred_market.items():
            series.to_csv(raw_dir / f"fred_{name}.csv")
    else:
        logger.info("Skipping download, using cached data...")

    # === 2. Load market price data ===
    market_prices = {}
    for name in cfg.MARKET_TICKERS:
        path = raw_dir / f"market_{name}.csv"
        if not path.exists():
            # Fall back to yahoo_ prefix (sp500 was downloaded as yahoo_sp500)
            path = raw_dir / f"yahoo_{name}.csv"
        if not path.exists():
            logger.warning(f"No data for {name}, skipping")
            continue
        try:
            df = _load_yahoo_csv(path)
            # Normalize column names
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if "close" in cl:
                    col_map[c] = "Close"
                elif "volume" in cl:
                    col_map[c] = "Volume"
                elif "high" in cl:
                    col_map[c] = "High"
                elif "low" in cl:
                    col_map[c] = "Low"
            df = df.rename(columns=col_map)
            if "Close" not in df.columns:
                logger.warning(f"No Close column for {name}, skipping")
                continue
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])
            market_prices[name] = df
            logger.info(f"  Loaded {name}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
        except Exception as e:
            logger.error(f"  Failed to load {name}: {e}")

    logger.info(f"Loaded {len(market_prices)} markets")

    # === 3. Load global features ===
    global_features = {}

    # VIX
    for name, fname in [("vix", "yahoo_vix"), ("vix3m", "yahoo_vix3m")]:
        path = raw_dir / f"{fname}.csv"
        if path.exists():
            df = _load_yahoo_csv(path)
            close_col = [c for c in df.columns if "close" in c.lower()]
            if close_col:
                series = pd.to_numeric(df[close_col[0]], errors="coerce").dropna()
                global_features[name] = series
                logger.info(f"  Global feature '{name}': {len(series)} rows")

    # FRED features
    for name in ["hy_spread", "ig_spread", "ted_spread", "dxy"]:
        path = raw_dir / f"fred_{name}.csv"
        if path.exists():
            series = pd.read_csv(path, index_col=0, parse_dates=True).squeeze()
            series = pd.to_numeric(series, errors="coerce").dropna()
            global_features[name] = series

    # Yahoo: gold, move_index, skew
    for name in ["gold", "move_index", "skew"]:
        path = raw_dir / f"yahoo_{name}.csv"
        if path.exists():
            df = _load_yahoo_csv(path)
            close_col = [c for c in df.columns if "close" in c.lower()]
            if close_col:
                series = pd.to_numeric(df[close_col[0]], errors="coerce").dropna()
                global_features[name] = series

    logger.info(f"Loaded {len(global_features)} global features")

    # === 4. Build panel dataset ===
    logger.info("Building panel dataset...")
    panel = build_panel_dataset(
        market_prices=market_prices,
        global_features=global_features,
        crisis_threshold=cfg.CRISIS_THRESHOLD,
        ews_horizon=cfg.EWS_HORIZON_DAYS,
        peak_window=cfg.PEAK_WINDOW,
        common_start=cfg.COMMON_START_DATE,
    )

    # === 5. Save ===
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_dir / "panel_features.parquet", index=False)

    # Market-level stats
    logger.info("\nPer-market summary:")
    for market in sorted(panel["market_id"].unique()):
        mdf = panel[panel["market_id"] == market]
        n_ews = int(mdf["ews_label"].sum())
        n_crisis = int(mdf["in_crisis"].sum())
        dates = pd.to_datetime(mdf["date"])
        logger.info(f"  {market:<12} {len(mdf):>5} rows, {n_ews:>4} warn, {n_crisis:>4} crisis  ({dates.min().date()} to {dates.max().date()})")

    logger.info(f"\nPanel saved to {output_dir / 'panel_features.parquet'}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-market panel pipeline")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--output-dir", default="data/processed/panel")
    args = parser.parse_args()

    cfg = PipelineConfig()
    run_panel_pipeline(cfg, Path(args.output_dir), skip_download=args.skip_download)


if __name__ == "__main__":
    main()
