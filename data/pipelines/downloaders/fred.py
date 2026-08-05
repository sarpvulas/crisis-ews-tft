# data/pipelines/downloaders/fred.py
import logging
from typing import Dict, Optional
import pandas as pd
from data.pipelines.config import PipelineConfig

logger = logging.getLogger(__name__)

class FREDDownloader:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = None
        if config.FRED_API_KEY:
            try:
                from fredapi import Fred
                self.client = Fred(api_key=config.FRED_API_KEY)
            except ImportError:
                logger.warning("fredapi not installed, FRED downloads disabled")

    def download_series(
        self, series_id: str, start: str, end: str
    ) -> Optional[pd.Series]:
        if self.client is None:
            logger.warning(f"No FRED client, skipping {series_id}")
            return None
        try:
            data = self.client.get_series(series_id, start, end)
            logger.info(f"Downloaded {series_id}: {len(data)} observations")
            return data
        except Exception as e:
            logger.error(f"Failed to download {series_id}: {e}")
            return None

    def download_all_market(self) -> Dict[str, pd.Series]:
        results = {}
        for name, series_id in self.config.FRED_SERIES_MARKET.items():
            s = self.download_series(series_id, self.config.TASK_B_START, self.config.DATA_END)
            if s is not None:
                results[name] = s
        return results
