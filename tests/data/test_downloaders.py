# tests/data/test_downloaders.py
import pandas as pd
from unittest.mock import patch, MagicMock
from data.pipelines.downloaders.fred import FREDDownloader
from data.pipelines.config import PipelineConfig

def test_fred_downloader_returns_dataframe():
    cfg = PipelineConfig()
    downloader = FREDDownloader(cfg)

    # Mock the fredapi.Fred client
    mock_fred = MagicMock()
    mock_series = pd.Series(
        [1.0, 1.5, 2.0],
        index=pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"]),
        name="VIXCLS",
    )
    mock_fred.get_series.return_value = mock_series

    with patch.object(downloader, "client", mock_fred):
        result = downloader.download_series("VIXCLS", "2000-01-01", "2000-12-31")

    assert isinstance(result, pd.Series)
    assert len(result) == 3
    mock_fred.get_series.assert_called_once()

def test_fred_downloader_handles_missing_key():
    cfg = PipelineConfig()
    cfg.FRED_API_KEY = ""
    downloader = FREDDownloader(cfg)
    assert downloader.client is None


from data.pipelines.downloaders.yahoo import YahooDownloader

def test_yahoo_downloader_returns_dataframe():
    cfg = PipelineConfig()
    downloader = YahooDownloader(cfg)

    with patch("data.pipelines.downloaders.yahoo.yf.download") as mock_dl:
        mock_df = pd.DataFrame(
            {"Close": [100, 101, 99], "Volume": [1e6, 1.1e6, 0.9e6]},
            index=pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"]),
        )
        mock_dl.return_value = mock_df
        result = downloader.download_ticker("^GSPC", "2000-01-01", "2000-12-31")

    assert isinstance(result, pd.DataFrame)
    assert "Close" in result.columns


from unittest.mock import mock_open
from data.pipelines.downloaders.bis_imf import BISDownloader

def test_bis_downloader_parses_csv():
    cfg = PipelineConfig()
    downloader = BISDownloader(cfg)

    # BIS credit gap CSV has specific structure
    mock_csv = "Country,Date,Credit_GDP_Gap\nUS,2000-Q1,2.5\nUS,2000-Q2,3.1\n"
    with patch("builtins.open", mock_open(read_data=mock_csv)):
        with patch("pandas.read_csv") as mock_read:
            mock_read.return_value = pd.DataFrame({
                "Country": ["US", "US"],
                "Date": ["2000-Q1", "2000-Q2"],
                "Credit_GDP_Gap": [2.5, 3.1],
            })
            result = downloader.parse_credit_gap("dummy.csv")

    assert isinstance(result, pd.DataFrame)
    assert "Credit_GDP_Gap" in result.columns
