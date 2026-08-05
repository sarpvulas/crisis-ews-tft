"""
Build a MULTI-MARKET PANEL Task-B crisis-EWS dataset from the shared Bloomberg
data — the multi-entity reformulation the thesis proposes. Generalises the
single-market SPX builder (build_taskb_from_shared.py) over ~9 broad equity
indices that have full 2000-2026 coverage in the shared `wide.parquet`:

    SPX (S&P 500), NDX (Nasdaq-100), RTY (Russell 2000), SX5E (EuroStoxx 50),
    SXXP (Stoxx 600), UKX (FTSE 100), MCX (FTSE 250), MXEF (MSCI EM),
    MXWO (MSCI World).

Per market: the same per-entity features + rolling-peak-drawdown onset labels.
Global features (VIX, credit spreads, gold, DXY, MOVE, TED) are broadcast to
every market's calendar. Market identity becomes a STATIC CATEGORICAL feature
(activating TFT's entity-embedding branch). Output is one long panel parquet
with a `market_id` column, consumed by training.dataset.PanelCrisisDataset.

Usage:
    python scripts/build_panel_from_shared.py --threshold 0.10 --peak-window 252
Output -> data/shared/processed/panel_<...>/panel.parquet
"""
import os, sys, logging, argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.pipelines.processing.task_b_features import build_task_b_features
from data.pipelines.processing.labels import build_regime_labels
from data.pipelines.processing.crisis_labels import build_crisis_labels
from data.pipelines.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_panel")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WIDE = os.path.join(_REPO, "data", "shared", "processed", "wide.parquet")
OUT_BASE = os.path.join(_REPO, "data", "shared", "processed")

# onset-labeling knobs (match the single-market best config)
EXIT_THRESHOLD    = -0.10
REBOUND_THRESHOLD = 0.20
RECOVERY_DAYS     = 30
POST_CRISIS_BUF   = 30

# Broad equity indices with full coverage (Bloomberg tickers -> readable name)
MARKETS = {
    "SPX":  "sp500",      "NDX":  "nasdaq100",  "RTY":  "russell2000",
    "SX5E": "eurostoxx50", "SXXP": "stoxx600",   "UKX":  "ftse100",
    "MCX":  "ftse250",    "MXEF": "msci_em",    "MXWO": "msci_world",
}

# Global (cross-market) feature source tickers
G = {"vix": "VIX Index | PX_LAST", "hy": "LF98OAS Index | PX_LAST",
     "ig": "LUACOAS Index | PX_LAST", "gold": "XAU Curncy | PX_LAST",
     "dxy": "DXY Curncy | PX_LAST", "move": "MOVE Index | PX_LAST",
     "libor": "US0003M Index | PX_LAST", "tbill": "USGG3M Index | PX_LAST"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="onset drawdown threshold (fraction). 0.10 = market-stress EWS.")
    ap.add_argument("--peak-window", type=int, default=252,
                    help="trailing rolling-max peak window (days) for drawdown.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = PipelineConfig()
    name = f"panel_dd{int(round(args.threshold*100))}_rp{args.peak_window}"
    OUT = args.out or os.path.join(OUT_BASE, name)
    os.makedirs(OUT, exist_ok=True)

    w = pd.read_parquet(WIDE)
    w.index = pd.to_datetime(w.index)
    log.info("Loaded wide matrix %s (%s -> %s)", w.shape, w.index.min().date(), w.index.max().date())

    def gget(key):
        c = G[key]
        if c not in w.columns:
            raise KeyError(f"missing global series {c!r}")
        return w[c]

    panels = []
    for tick, mname in MARKETS.items():
        close_col = f"{tick} Index | PX_LAST"
        if close_col not in w.columns:
            log.warning("  SKIP %s — no PX_LAST", tick); continue
        close = w[close_col].dropna()
        cal = close.index
        vol_col = f"{tick} Index | PX_VOLUME"
        volume = w[vol_col].reindex(cal) if vol_col in w.columns else pd.Series(float("nan"), index=cal)
        prices = pd.DataFrame({"Close": close, "Volume": volume})

        ted = (gget("libor").reindex(cal).ffill() - gget("tbill").reindex(cal).ffill())
        feats = build_task_b_features(
            prices=prices,
            vix_data=gget("vix").reindex(cal).ffill(), vix3m_data=None,
            hy_spread=gget("hy").reindex(cal).ffill(), ig_spread=gget("ig").reindex(cal).ffill(),
            ted_spread=ted, gold=gget("gold").reindex(cal).ffill(),
            dxy=gget("dxy").reindex(cal).ffill(), skew=None,
            move_index=gget("move").reindex(cal).ffill(),
            weekly=cfg.WEEKLY_RESAMPLE, common_start_date=cfg.COMMON_START_DATE or None,
        )
        feats = feats.dropna(thresh=int(0.5 * feats.shape[1]))

        cl = build_crisis_labels(
            feats["close"], horizon=cfg.EWS_HORIZON_DAYS,
            entry_threshold=-args.threshold, exit_threshold=EXIT_THRESHOLD,
            recovery_days=RECOVERY_DAYS, post_crisis_buffer=POST_CRISIS_BUF,
            recovery_mode="trough", rebound_threshold=REBOUND_THRESHOLD,
            peak_window=args.peak_window,
        ).set_index("date")
        feats["ews_label"] = cl["y"].reindex(feats.index).astype("Float64").fillna(0).astype(int)
        feats["in_crisis"] = (cl["crisis_active"] | cl["in_buffer"]).reindex(feats.index).fillna(False)
        feats["drawdown"]  = cl["drawdown"].reindex(feats.index)
        reg = build_regime_labels(feats["close"], gget("vix").reindex(feats.index).ffill().dropna(),
                                  drawdown_threshold=cfg.CRISIS_THRESHOLD,
                                  vix_stress=cfg.VIX_STRESS_THRESHOLD).set_index("date")
        feats = feats.join(reg[["regime_label"]], how="left").fillna({"regime_label": 0})

        mp = feats.reset_index().rename(columns={feats.reset_index().columns[0]: "date"})
        mp["market_id"] = mname
        n_onset = int(cl["crisis_start"].sum())
        log.info("  %-12s %-12s rows=%d  onsets=%d  warn_days=%d",
                 tick, mname, len(mp), n_onset, int(mp["ews_label"].sum()))
        panels.append(mp)

    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(os.path.join(OUT, "panel.parquet"), index=False)
    log.info("\nPANEL DONE -> %s/panel.parquet", OUT)
    log.info("markets=%d rows=%d total_warn_days=%d  date %s..%s",
             panel["market_id"].nunique(), len(panel), int(panel["ews_label"].sum()),
             panel["date"].min().date(), panel["date"].max().date())
    log.info("feature cols (%d): %s", panel.shape[1], list(panel.columns))


if __name__ == "__main__":
    main()
