# tests/data/test_config.py
from data.pipelines.config import PipelineConfig

def test_config_has_required_fields():
    cfg = PipelineConfig()
    assert cfg.TRAIN_END == "2014-12-31"
    assert cfg.VAL_END == "2019-12-31"
    assert len(cfg.FRED_SERIES_MARKET) > 0
    assert cfg.TASK_B_START == "2000-01-01"

def test_config_ews_params():
    cfg = PipelineConfig()
    assert cfg.CRISIS_THRESHOLD == 0.20
    assert cfg.EWS_HORIZON_DAYS == 63
    assert cfg.PEAK_WINDOW == 252

def test_config_no_leakage_in_dates():
    cfg = PipelineConfig()
    assert cfg.TRAIN_END < cfg.VAL_END
    assert cfg.VAL_END < cfg.TEST_START
