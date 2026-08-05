"""CLI entrypoint: download -> process -> split -> save to Parquet."""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data.pipelines.config import PipelineConfig
from data.pipelines.downloaders.fred import FREDDownloader
from data.pipelines.downloaders.yahoo import YahooDownloader
from data.pipelines.processing.task_b_features import (
    build_task_b_features,
)
from data.pipelines.processing.labels import (
    build_ews_labels,
    build_regime_labels,
)
from data.pipelines.processing.splits import temporal_split, walk_forward_crisis_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_yahoo_csv(path: Path) -> pd.DataFrame:
    """Load a Yahoo Finance CSV that may have multi-row header from yfinance."""
    df = pd.read_csv(path)
    # yfinance writes: row0 = "Price,Close,High,...", row1 = "Ticker,^GSPC,...", row2 = "Date,..."
    # Detect by checking if first column is "Price" and second row has ticker symbols
    if df.columns[0] == "Price" and len(df) > 1:
        # Row 0 has ticker names, row 1 has NaN/Date — skip both
        cols = df.columns[1:]  # Drop 'Price' → ['Close','High','Low','Open','Volume']
        data = df.iloc[2:].copy()  # Skip ticker row + empty row
        data.columns = [df.columns[0]] + list(cols)
        data = data.rename(columns={data.columns[0]: "Date"})
        data = data.set_index("Date")
        data.index = pd.to_datetime(data.index)
        # Convert all columns to numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.dropna(how="all")
        return data
    else:
        # Normal CSV
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df


def run_pipeline(cfg: PipelineConfig, output_dir: Path, skip_download: bool = False):
    """Download market data, engineer features, build labels, split, and save."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Download ===
    yahoo = YahooDownloader(cfg)
    fred = FREDDownloader(cfg)

    if not skip_download:
        logger.info("Downloading Yahoo Finance data...")
        yahoo_data = yahoo.download_all()
        # Save raw
        for name, df in yahoo_data.items():
            df.to_csv(raw_dir / f"yahoo_{name}.csv")
            logger.info(f"  Saved yahoo_{name}.csv: {len(df)} rows")

        logger.info("Downloading FRED market data...")
        fred_market = fred.download_all_market()
        for name, series in fred_market.items():
            series.to_csv(raw_dir / f"fred_{name}.csv")
            logger.info(f"  Saved fred_{name}.csv: {len(series)} rows")
    else:
        logger.info("Skipping download, loading cached data...")

    # === 2. Load raw data ===
    # S&P 500 prices
    sp500_path = raw_dir / "yahoo_sp500.csv"
    if not sp500_path.exists():
        logger.error("S&P 500 data not found. Run without --skip-download first.")
        return

    sp500 = _load_yahoo_csv(sp500_path)
    logger.info(f"S&P 500 loaded: {sp500.shape}")

    # Load VIX
    vix_series = None
    vix_path = raw_dir / "yahoo_vix.csv"
    if vix_path.exists():
        vix_df = _load_yahoo_csv(vix_path)
        vix_series = vix_df["Close"] if "Close" in vix_df.columns else None
        logger.info(f"VIX loaded: {len(vix_series) if vix_series is not None else 0} rows")

    # Load VIX3M
    vix3m_series = None
    vix3m_path = raw_dir / "yahoo_vix3m.csv"
    if vix3m_path.exists():
        vix3m_df = _load_yahoo_csv(vix3m_path)
        vix3m_series = vix3m_df["Close"] if "Close" in vix3m_df.columns else None

    # Load FRED spreads
    def _load_fred_series(name):
        path = raw_dir / f"fred_{name}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df.iloc[:, 0] if len(df.columns) > 0 else df.squeeze()
        # Also check fred_market_ prefix
        path2 = raw_dir / f"fred_market_{name}.csv"
        if path2.exists():
            df = pd.read_csv(path2, index_col=0, parse_dates=True)
            return df.iloc[:, 0] if len(df.columns) > 0 else df.squeeze()
        return None

    hy_spread = _load_fred_series("hy_spread")
    ig_spread = _load_fred_series("ig_spread")
    ted_spread = _load_fred_series("ted_spread")

    # Load Yahoo series
    def _load_yahoo_series(name):
        path = raw_dir / f"yahoo_{name}.csv"
        if path.exists():
            df = _load_yahoo_csv(path)
            return df["Close"] if "Close" in df.columns else None
        return None

    gold_series = _load_yahoo_series("gold")
    dxy_series = _load_yahoo_series("dxy")
    skew_series = _load_yahoo_series("skew")
    move_series = _load_yahoo_series("move_index")

    # === 3. Build features ===
    logger.info("Building Task B features...")
    features = build_task_b_features(
        prices=sp500,
        vix_data=vix_series,
        vix3m_data=vix3m_series,
        hy_spread=hy_spread,
        ig_spread=ig_spread,
        ted_spread=ted_spread,
        gold=gold_series,
        dxy=dxy_series,
        skew=skew_series,
        move_index=move_series,
        weekly=cfg.WEEKLY_RESAMPLE,
        common_start_date=cfg.COMMON_START_DATE or None,
    )
    logger.info(f"Features built: {features.shape}")

    # Drop rows with too many NaNs (warmup period)
    features = features.dropna(thresh=int(0.5 * features.shape[1]))
    logger.info(f"After dropping warmup NaN rows: {features.shape}")

    # === 4. Build labels ===
    close_col = "close" if "close" in features.columns else "Close"

    # EWS labels — forward-looking crisis onset prediction
    logger.info("Building EWS labels...")
    if close_col in features.columns:
        horizon = cfg.EWS_HORIZON_DAYS // (5 if cfg.WEEKLY_RESAMPLE else 1)
        ews_labels = build_ews_labels(
            features[close_col],
            threshold=cfg.CRISIS_THRESHOLD,
            horizon_days=horizon,
            peak_window=cfg.PEAK_WINDOW,
        )
        ews_labels = ews_labels.set_index("date")
        features = features.join(ews_labels[["ews_label", "in_crisis", "drawdown"]], how="left")

    # Regime labels — auxiliary target for regime module
    logger.info("Building regime labels...")
    if vix_series is not None and close_col in features.columns:
        vix_daily = vix_series.reindex(features.index, method="ffill")

        regime_labels = build_regime_labels(
            prices=features[close_col],
            vix=vix_daily.dropna(),
            drawdown_threshold=cfg.CRISIS_THRESHOLD,
            vix_stress=cfg.VIX_STRESS_THRESHOLD,
        )
        regime_labels = regime_labels.set_index("date")
        features = features.join(regime_labels[["regime_label"]], how="left")

    # Fill remaining NaNs
    features = features.fillna(0)

    logger.info(f"Final features shape: {features.shape}")
    logger.info(f"Columns: {list(features.columns)}")

    if "ews_label" in features.columns:
        logger.info(f"EWS label distribution: {features['ews_label'].value_counts().to_dict()}")
        logger.info(f"In-crisis days (excluded from training): {features['in_crisis'].sum()}")
    if "regime_label" in features.columns:
        logger.info(f"Regime distribution: {features['regime_label'].value_counts().to_dict()}")

    # === 5. Split and save ===
    features_with_date = features.reset_index()
    features_with_date = features_with_date.rename(columns={features_with_date.columns[0]: "date"})

    # Walk-forward crisis-anchored splits (primary evaluation)
    logger.info("Building walk-forward crisis splits...")
    folds = walk_forward_crisis_splits(features_with_date, date_col="date")

    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        name = fold["name"]
        fold["train"].to_parquet(fold_dir / f"{name}_train.parquet", index=False)
        fold["val"].to_parquet(fold_dir / f"{name}_val.parquet", index=False)
        fold["test"].to_parquet(fold_dir / f"{name}_test.parquet", index=False)
        logger.info(f"  Fold '{name}': {fold['meta']}")

    # Also keep a simple split for quick iteration (uses last fold's split)
    if folds:
        last = folds[-1]
        last["train"].to_parquet(output_dir / "task_b_train.parquet", index=False)
        last["val"].to_parquet(output_dir / "task_b_val.parquet", index=False)
        last["test"].to_parquet(output_dir / "task_b_test.parquet", index=False)
    else:
        # Fallback to simple temporal split
        train, val, test = temporal_split(
            features_with_date, date_col="date",
            train_end=cfg.TRAIN_END, val_end=cfg.VAL_END,
        )
        train.to_parquet(output_dir / "task_b_train.parquet", index=False)
        val.to_parquet(output_dir / "task_b_val.parquet", index=False)
        test.to_parquet(output_dir / "task_b_test.parquet", index=False)

    features_with_date.to_parquet(output_dir / "task_b_features.parquet", index=False)

    logger.info(f"Saved data to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Use cached raw data")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    cfg = PipelineConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("=== Market Crash Prediction Pipeline ===")
    logger.info("=" * 60)
    run_pipeline(cfg, output_dir, args.skip_download)


if __name__ == "__main__":
    main()
