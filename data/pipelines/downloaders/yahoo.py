# data/pipelines/downloaders/yahoo.py
import logging
from typing import Dict, Optional
import pandas as pd
import yfinance as yf
from data.pipelines.config import PipelineConfig

logger = logging.getLogger(__name__)

class YahooDownloader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def download_ticker(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                logger.warning(f"No data for {ticker}")
                return None
            logger.info(f"Downloaded {ticker}: {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return None

    def download_all(self) -> Dict[str, pd.DataFrame]:
        results = {}
        for name, ticker in self.config.YAHOO_TICKERS.items():
            df = self.download_ticker(ticker, self.config.TASK_B_START, self.config.DATA_END)
            if df is not None:
                results[name] = df
        return results
