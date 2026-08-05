# data/pipelines/processing/panel_features.py
"""Build panel dataset: per-market features + global features + EWS labels for 12 markets."""
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(prices / prices.shift(periods))


def compute_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def compute_rolling_drawdown(prices: pd.Series, window: int) -> pd.Series:
    peak = prices.rolling(window=window, min_periods=1).max()
    return (prices - peak) / peak


def compute_hurst(series: pd.Series, window: int = 63) -> pd.Series:
    def hurst_rs(x):
        if len(x) < 20:
            return np.nan
        mean = x.mean()
        deviate = np.cumsum(x - mean)
        r = deviate.max() - deviate.min()
        s = x.std(ddof=1)
        if s == 0:
            return np.nan
        return np.log(r / s) / np.log(len(x))
    return series.rolling(window).apply(hurst_rs, raw=True)


def build_market_features(prices_df: pd.DataFrame, market_name: str) -> pd.DataFrame:
    """Build per-market features from OHLCV data.

    Args:
        prices_df: DataFrame with Close, Volume columns (DatetimeIndex).
        market_name: Market identifier string.

    Returns:
        DataFrame with per-market features, indexed by date.
    """
    close = prices_df["Close"] if "Close" in prices_df.columns else prices_df["close"]
    volume = prices_df.get("Volume", prices_df.get("volume", pd.Series(dtype=float)))

    df = pd.DataFrame(index=close.index)
    df["market_id"] = market_name

    # Returns
    ret_1d = compute_log_returns(close, 1)
    df["log_ret_1d"] = ret_1d
    df["log_ret_5d"] = compute_log_returns(close, 5)
    df["log_ret_21d"] = compute_log_returns(close, 21)

    # Volatility
    df["rvol_5d"] = compute_realized_vol(ret_1d, 5)
    df["rvol_21d"] = compute_realized_vol(ret_1d, 21)
    df["rvol_63d"] = compute_realized_vol(ret_1d, 63)
    df["vol_of_vol_63d"] = df["rvol_21d"].rolling(63).std()
    df["return_skew_63d"] = ret_1d.rolling(63).skew()

    # Drawdown + micro-structure
    df["drawdown_63d"] = compute_rolling_drawdown(close, 63)
    df["hurst_63d"] = compute_hurst(ret_1d, 63)

    # Liquidity (if volume available)
    if len(volume) > 0 and volume.notna().sum() > 0:
        vol_mean = volume.rolling(63).mean()
        vol_std = volume.rolling(63).std()
        df["volume_zscore_63d"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        ratio = ret_1d.abs() / volume.replace(0, np.nan)
        df["amihud_21d"] = ratio.rolling(21).mean()

    df["close"] = close

    return df


def build_panel_dataset(
    market_prices: Dict[str, pd.DataFrame],
    global_features: Dict[str, pd.Series],
    crisis_threshold: float = 0.20,
    ews_horizon: int = 63,
    peak_window: int = 252,
    common_start: str = "2000-01-01",
) -> pd.DataFrame:
    """Build full panel dataset with per-market features, global features, and EWS labels.

    Args:
        market_prices: {market_name: OHLCV DataFrame} for each market.
        global_features: {feature_name: Series} for global indicators (VIX, spreads, etc.).
        crisis_threshold: Drawdown threshold for crisis definition.
        ews_horizon: Forward horizon for EWS labels (trading days).
        peak_window: Rolling window for peak calculation.
        common_start: Earliest date to include.

    Returns:
        Panel DataFrame with columns: date, market_id, [features], ews_label, in_crisis, regime_label.
    """
    from data.pipelines.processing.labels import (
        build_ews_labels, build_regime_labels,
    )

    all_market_dfs = []
    start_ts = pd.Timestamp(common_start)

    for market_name, prices_df in market_prices.items():
        logger.info(f"Processing {market_name}: {len(prices_df)} rows")

        # Build per-market features
        mf = build_market_features(prices_df, market_name)
        mf = mf[mf.index >= start_ts]

        if len(mf) < peak_window + ews_horizon:
            logger.warning(f"Skipping {market_name}: only {len(mf)} rows after trim")
            continue

        # Add global features (align by date)
        for feat_name, series in global_features.items():
            mf[feat_name] = series.reindex(mf.index, method="ffill")

        # Derived cross-features
        if "vix" in mf.columns and "vix3m" in mf.columns:
            mf["vix_term_structure"] = mf["vix"] - mf["vix3m"]
        if "vix" in mf.columns and "rvol_21d" in mf.columns:
            mf["implied_realized_spread"] = mf["vix"] / 100 - mf["rvol_21d"]
        if "hy_spread" in mf.columns and "ig_spread" in mf.columns:
            mf["credit_spread_diff"] = mf["hy_spread"] - mf["ig_spread"]

        # Build EWS labels for this market
        close = mf["close"]
        ews = build_ews_labels(close, crisis_threshold, ews_horizon, peak_window)
        ews = ews.set_index("date")
        mf = mf.join(ews[["ews_label", "in_crisis", "drawdown"]], how="left")

        # Regime labels (use VIX if available, otherwise vol-based)
        if "vix" in mf.columns:
            vix_series = mf["vix"].dropna()
            if len(vix_series) > 0:
                regime = build_regime_labels(close, vix_series)
                regime = regime.set_index("date")
                mf = mf.join(regime[["regime_label"]], how="left")

        if "regime_label" not in mf.columns:
            mf["regime_label"] = 0

        # Drop warmup NaNs
        mf = mf.dropna(thresh=int(0.5 * mf.shape[1]))
        mf = mf.fillna(0)

        # Reset index to get date column
        mf = mf.reset_index()
        mf = mf.rename(columns={mf.columns[0]: "date"})

        all_market_dfs.append(mf)
        n_crisis = int(mf["in_crisis"].sum()) if "in_crisis" in mf.columns else 0
        n_ews = int(mf["ews_label"].sum()) if "ews_label" in mf.columns else 0
        logger.info(f"  {market_name}: {len(mf)} rows, {n_ews} warning days, {n_crisis} crisis days")

    panel = pd.concat(all_market_dfs, ignore_index=True)
    logger.info(
        f"Panel dataset: {len(panel)} total rows, "
        f"{panel['market_id'].nunique()} markets, "
        f"{int(panel['ews_label'].sum())} total warning days"
    )

    return panel
