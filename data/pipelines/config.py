# data/pipelines/config.py
from dataclasses import dataclass, field
from typing import Dict, List
import os

@dataclass
class PipelineConfig:
    # API Keys
    FRED_API_KEY: str = field(default_factory=lambda: os.environ.get("FRED_API_KEY", ""))

    # Date boundaries
    TASK_B_START: str = "2000-01-01"
    DATA_END: str = "2024-12-31"

    # Common start date — hard-trims ALL sources to start from this date
    COMMON_START_DATE: str = "2000-01-01"

    # Temporal splits (60/20/20)
    TRAIN_END: str = "2014-12-31"
    VAL_START: str = "2015-01-01"
    VAL_END: str = "2019-12-31"
    TEST_START: str = "2020-01-01"

    # FRED series mapping: {feature_name: series_id}
    FRED_SERIES_MARKET: Dict[str, str] = field(default_factory=lambda: {
        "hy_spread": "BAMLH0A0HYM2",
        "ig_spread": "BAMLC0A0CM",
        "ted_spread": "TEDRATE",
        "vix": "VIXCLS",
        "dxy": "DTWEXBGS",
        "margin_debt": "BOGZ1FL663067003Q",
    })

    # Yahoo Finance tickers — auxiliary signals
    YAHOO_TICKERS: Dict[str, str] = field(default_factory=lambda: {
        "sp500": "^GSPC",
        "vix": "^VIX",
        "vix3m": "^VIX3M",
        "vvix": "^VVIX",
        "skew": "^SKEW",
        "move_index": "^MOVE",
        "gold": "GC=F",
        "dxy": "DX-Y.NYB",
        "tnx_2y": "^IRX",
    })

    # Multi-market equity indices for panel EWS
    MARKET_TICKERS: Dict[str, str] = field(default_factory=lambda: {
        "sp500": "^GSPC",
        "tsx": "^GSPTSE",
        "ftse": "^FTSE",
        "dax": "^GDAXI",
        "cac": "^FCHI",
        "nikkei": "^N225",
        "hangseng": "^HSI",
        "asx": "^AXJO",
        "kospi": "^KS11",
        "bovespa": "^BVSP",
        "nifty": "^NSEI",
        "shanghai": "000001.SS",
    })

    # EWS parameters (following Dichtl et al. 2023)
    CRISIS_THRESHOLD: float = 0.20      # 20% drawdown = bear market
    EWS_HORIZON_DAYS: int = 63          # ~3 months forward warning horizon
    PEAK_WINDOW: int = 252              # 1-year rolling peak
    WEEKLY_RESAMPLE: bool = False

    # Regime label parameters
    VIX_STRESS_THRESHOLD: float = 25.0
