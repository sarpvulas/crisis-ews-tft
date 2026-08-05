# data/pipelines/downloaders/bis_imf.py
import logging
from typing import Optional
import pandas as pd
import requests
from data.pipelines.config import PipelineConfig

logger = logging.getLogger(__name__)

BIS_CREDIT_GAP_URL = "https://www.bis.org/statistics/c_gaps.xlsx"

class BISDownloader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def download_credit_gap(self, save_path: str) -> Optional[pd.DataFrame]:
        """Download BIS credit-to-GDP gap data."""
        try:
            resp = requests.get(BIS_CREDIT_GAP_URL, timeout=60)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            df = pd.read_excel(save_path, sheet_name=0, skiprows=3)
            logger.info(f"Downloaded BIS credit gap: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to download BIS credit gap: {e}")
            return None

    def parse_credit_gap(self, filepath: str) -> pd.DataFrame:
        """Parse a local BIS credit gap CSV/Excel file."""
        df = pd.read_csv(filepath)
        return df

class IMFDownloader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def download_weo(self, save_path: str) -> Optional[pd.DataFrame]:
        """Download IMF WEO data. Falls back to local file if API unavailable."""
        try:
            # IMF WEO is typically distributed as tab-separated files
            url = "https://www.imf.org/external/pubs/ft/weo/data/WEOApr2024all.ashx"
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(save_path, sep="\t", encoding="latin1")
            logger.info(f"Downloaded IMF WEO: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to download IMF WEO: {e}")
            return None
