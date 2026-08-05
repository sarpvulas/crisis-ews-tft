# data/pipelines/processing/task_b_features.py
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def compute_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)

def compute_drawdown_from_peak(prices: pd.Series) -> pd.Series:
    peak = prices.expanding().max()
    return (prices - peak) / peak

def compute_rolling_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Drawdown from rolling window peak."""
    peak = prices.rolling(window=window, min_periods=1).max()
    return (prices - peak) / peak

def compute_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(prices / prices.shift(periods))

def compute_hurst_exponent(series: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Hurst exponent via R/S analysis."""
    def hurst_rs(x):
        if len(x) < 20:
            return np.nan
        n = len(x)
        mean = x.mean()
        deviate = np.cumsum(x - mean)
        r = deviate.max() - deviate.min()
        s = x.std(ddof=1)
        if s == 0:
            return np.nan
        return np.log(r / s) / np.log(n)
    return series.rolling(window).apply(hurst_rs, raw=True)

def compute_amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    ratio = returns.abs() / volume.replace(0, np.nan)
    return ratio.rolling(window).mean()

def compute_volume_zscore(volume: pd.Series, window: int = 63) -> pd.Series:
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)

def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly: last close, max high, min low, sum volume, mean others."""
    agg_rules = {}
    for col in df.columns:
        if col == "close":
            agg_rules[col] = "last"
        elif col == "high":
            agg_rules[col] = "max"
        elif col == "low":
            agg_rules[col] = "min"
        elif col == "volume":
            agg_rules[col] = "sum"
        else:
            agg_rules[col] = "mean"
    return df.resample("W-FRI").agg(agg_rules).dropna(how="all")

def build_task_b_features(
    prices: pd.DataFrame,
    vix_data: Optional[pd.Series] = None,
    vix3m_data: Optional[pd.Series] = None,
    hy_spread: Optional[pd.Series] = None,
    ig_spread: Optional[pd.Series] = None,
    ted_spread: Optional[pd.Series] = None,
    gold: Optional[pd.Series] = None,
    dxy: Optional[pd.Series] = None,
    skew: Optional[pd.Series] = None,
    move_index: Optional[pd.Series] = None,
    weekly: bool = True,
    common_start_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build Task B feature matrix from market data."""
    df = pd.DataFrame(index=prices.index)

    close = prices["Close"] if "Close" in prices.columns else prices["close"]
    volume = prices.get("Volume", prices.get("volume", pd.Series(dtype=float)))

    # Price features
    df["log_ret_1d"] = compute_log_returns(close, 1)
    df["log_ret_5d"] = compute_log_returns(close, 5)
    df["log_ret_21d"] = compute_log_returns(close, 21)
    df["rvol_5d"] = compute_realized_vol(df["log_ret_1d"], 5)
    df["rvol_21d"] = compute_realized_vol(df["log_ret_1d"], 21)
    df["rvol_63d"] = compute_realized_vol(df["log_ret_1d"], 63)
    # Short-horizon drawdown only (63d peak != 252d crisis label peak)
    df["drawdown_63d"] = compute_rolling_drawdown(close, 63)
    # NOTE: drawdown_252d/504d removed — mechanically encodes the crisis label definition
    # Replace with vol-of-vol and return dispersion as genuine leading indicators
    df["vol_of_vol_63d"] = df["rvol_21d"].rolling(63).std()
    df["return_skew_63d"] = df["log_ret_1d"].rolling(63).skew()
    df["hurst_63d"] = compute_hurst_exponent(df["log_ret_1d"], 63)
    df["close"] = close

    # Liquidity features
    if len(volume) > 0:
        df["amihud_21d"] = compute_amihud_illiquidity(df["log_ret_1d"], volume, 21)
        df["volume_zscore_63d"] = compute_volume_zscore(volume, 63)
        df["volume"] = volume

    # External features (align by date)
    for name, series in [
        ("vix", vix_data), ("vix3m", vix3m_data),
        ("hy_spread", hy_spread), ("ig_spread", ig_spread),
        ("ted_spread", ted_spread),
        ("gold", gold), ("dxy", dxy),
        ("skew", skew), ("move_index", move_index),
    ]:
        if series is not None:
            df[name] = series.reindex(df.index, method="ffill")

    # Derived features
    if "vix" in df.columns and "vix3m" in df.columns:
        df["vix_term_structure"] = df["vix"] - df["vix3m"]
    if "vix" in df.columns and "rvol_21d" in df.columns:
        df["implied_realized_spread"] = df["vix"] / 100 - df["rvol_21d"]
    if "gold" in df.columns:
        df["gold_spx_ratio"] = df["gold"] / close
    if "dxy" in df.columns:
        df["dxy_ret_5d"] = compute_log_returns(df["dxy"], 5)
    if "hy_spread" in df.columns and "ig_spread" in df.columns:
        df["credit_spread_diff"] = df["hy_spread"] - df["ig_spread"]
    if "vix" in df.columns and "skew" in df.columns:
        df["vix_skew_ratio"] = df["vix"] / df["skew"]

    if weekly:
        df = resample_to_weekly(df)

    # Apply common start date trim
    if common_start_date:
        start = pd.Timestamp(common_start_date)
        before = len(df)
        df = df[df.index >= start]
        logger.info(f"Trimmed to common start {common_start_date}: {before} -> {len(df)} rows")

    logger.info(f"Task B features: {df.shape}")
    return df
