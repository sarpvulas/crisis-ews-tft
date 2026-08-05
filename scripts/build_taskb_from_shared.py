"""
Build a Task-B (broad-equity crisis EWS) dataset from the lecturer's shared
Bloomberg data, reusing the project's existing feature/label/split functions so
the result is directly comparable to the original thesis Task B.

Stage 1 (faithful replication): SPX close/volume + the standard global feature
set, reconstructed from the shared series:

    close      = SPX Index | PX_LAST          vix     = VIX Index | PX_LAST
    volume     = SPX Index | PX_VOLUME        gold    = XAU Curncy | PX_LAST
    hy_spread  = LF98OAS Index | PX_LAST      dxy     = DXY Curncy | PX_LAST
    ig_spread  = LUACOAS Index | PX_LAST      move    = MOVE Index | PX_LAST
    ted_spread = US0003M - USGG3M             (vix3m, skew: not in shared data)

Output -> data/shared/processed/taskb_spx_bbg/{task_b_features,task_b_train,
task_b_val,task_b_test}.parquet  +  folds/  (walk-forward crisis splits)
"""
import os, sys, logging, argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.pipelines.processing.task_b_features import build_task_b_features
from data.pipelines.processing.labels import build_ews_labels, build_regime_labels
from data.pipelines.processing.crisis_labels import build_crisis_labels
from data.pipelines.processing.splits import walk_forward_crisis_splits, temporal_split
from data.pipelines.config import PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("build_taskb_shared")

# Paths are resolved relative to the repo root so the build runs identically on
# Windows and inside the Linux training container (where C:\ paths don't exist).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WIDE = os.path.join(_REPO, "data", "shared", "processed", "wide.parquet")
OUT_BASE = os.path.join(_REPO, "data", "shared", "processed")

# onset-labeling (hysteresis) knobs — see report/crisis-rule.html
EXIT_THRESHOLD   = -0.10   # peak-mode: recovery level (DD from all-time peak) that can end a crisis
REBOUND_THRESHOLD = 0.20   # trough-mode: rebound off the crisis low that can end a crisis (+20%)
RECOVERY_DAYS    = 30      # days the recovery condition must hold before "out"
POST_CRISIS_BUF  = 30      # extra excluded days after recovery

KEYS = {
    "close":  "SPX Index | PX_LAST",
    "volume": "SPX Index | PX_VOLUME",
    "vix":    "VIX Index | PX_LAST",
    "hy":     "LF98OAS Index | PX_LAST",
    "ig":     "LUACOAS Index | PX_LAST",
    "gold":   "XAU Curncy | PX_LAST",
    "dxy":    "DXY Curncy | PX_LAST",
    "move":   "MOVE Index | PX_LAST",
    "libor3m":"US0003M Index | PX_LAST",
    "tbill3m":"USGG3M Index | PX_LAST",
}

def get(w, key, required=True):
    if key not in w.columns:
        msg = f"missing series: {key!r}"
        if required: raise KeyError(msg)
        log.warning("  (optional) %s", msg); return None
    return w[key]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeler", choices=["ews", "onset"], default="ews",
                    help="ews = original 252d-rolling-peak EWS; onset = hysteresis onset-only with recovery exit + buffer")
    ap.add_argument("--recovery", choices=["peak", "trough"], default="peak",
                    help="onset exit rule: peak = recover to within 10%% of all-time high; trough = +20%% rebound off the crisis low")
    ap.add_argument("--peak-window", type=int, default=None,
                    help="onset labeler drawdown reference: omit = all-time cummax peak (legacy); "
                         "int (e.g. 252) = trailing rolling-max peak (52-week-high drawdown), "
                         "avoids the post-crash multi-year drawdown artifact")
    ap.add_argument("--threshold", type=float, default=None,
                    help="onset drawdown threshold (fraction, e.g. 0.10). Omit = cfg.CRISIS_THRESHOLD "
                         "(0.20 = bear-market/crisis). Lower (0.10-0.12) yields more events for learning "
                         "but reframes the target as a drawdown/market-stress EWS, not a 20%% crisis.")
    ap.add_argument("--out", default=None, help="output dir (default derived from labeler/recovery/peak-window/threshold)")
    ap.add_argument("--embargo", type=int, default=63,
                    help="purge the last N rows of train/val per fold to remove the forward-label-horizon "
                         "leak across split boundaries (default 63 = decoder_steps). 0 disables.")
    args = ap.parse_args()

    cfg = PipelineConfig()
    onset_threshold = args.threshold if args.threshold is not None else cfg.CRISIS_THRESHOLD
    if args.labeler == "ews":
        default_out = "taskb_spx_bbg"
    elif args.recovery == "trough":
        default_out = "taskb_spx_bbg_onset_trough"
    else:
        default_out = "taskb_spx_bbg_onset"
    # rolling-peak / non-default-threshold onset variants get a distinct dir so
    # the legacy shared dataset (cummax, 20%) is never overwritten.
    if args.labeler == "onset" and args.peak_window is not None:
        default_out += f"_rp{args.peak_window}"
    if args.labeler == "onset" and args.threshold is not None:
        default_out += f"_dd{int(round(onset_threshold * 100))}"
    if args.embargo > 0:
        default_out += f"_emb{args.embargo}"
    OUT = args.out or os.path.join(OUT_BASE, default_out)
    os.makedirs(OUT, exist_ok=True)
    log.info("Labeler: %s  ->  %s", args.labeler, OUT)
    w = pd.read_parquet(WIDE)
    w.index = pd.to_datetime(w.index)
    log.info("Loaded shared wide matrix: %s  (%s -> %s)", w.shape, w.index.min().date(), w.index.max().date())

    # real trading calendar = days SPX actually traded
    close = get(w, KEYS["close"]).dropna()
    cal = close.index
    volume = get(w, KEYS["volume"], required=False)
    prices = pd.DataFrame({"Close": close, "Volume": (volume.reindex(cal) if volume is not None else float("nan"))})

    vix  = get(w, KEYS["vix"]).reindex(cal).ffill()
    hy   = get(w, KEYS["hy"]).reindex(cal).ffill()
    ig   = get(w, KEYS["ig"]).reindex(cal).ffill()
    gold = get(w, KEYS["gold"]).reindex(cal).ffill()
    dxy  = get(w, KEYS["dxy"]).reindex(cal).ffill()
    move = get(w, KEYS["move"]).reindex(cal).ffill()
    libor = get(w, KEYS["libor3m"], required=False)
    tbill = get(w, KEYS["tbill3m"], required=False)
    ted = None
    if libor is not None and tbill is not None:
        ted = (libor.reindex(cal).ffill() - tbill.reindex(cal).ffill())
        log.info("  TED spread = US0003M - USGG3M  (mean=%.3f)", ted.dropna().mean())

    feats = build_task_b_features(
        prices=prices, vix_data=vix, vix3m_data=None,
        hy_spread=hy, ig_spread=ig, ted_spread=ted,
        gold=gold, dxy=dxy, skew=None, move_index=move,
        weekly=cfg.WEEKLY_RESAMPLE, common_start_date=cfg.COMMON_START_DATE or None,
    )
    feats = feats.dropna(thresh=int(0.5 * feats.shape[1]))
    log.info("Features after warmup drop: %s", feats.shape)

    # labels
    if args.labeler == "ews":
        # original: 252d rolling-peak EWS, in_crisis = drawdown>=20%
        ews = build_ews_labels(feats["close"], threshold=cfg.CRISIS_THRESHOLD,
                               horizon_days=cfg.EWS_HORIZON_DAYS, peak_window=cfg.PEAK_WINDOW).set_index("date")
        feats = feats.join(ews[["ews_label", "in_crisis", "drawdown"]], how="left")
    else:
        # onset-only with hysteresis recovery exit + post-crisis buffer.
        # y is NA during active crisis + buffer; we keep rows contiguous (no drop)
        # and route the exclusion through `in_crisis` so the dataset freezes those
        # windows out of train/val instead of corrupting the sliding-window calendar.
        cl = build_crisis_labels(
            feats["close"], horizon=cfg.EWS_HORIZON_DAYS,
            entry_threshold=-onset_threshold, exit_threshold=EXIT_THRESHOLD,
            recovery_days=RECOVERY_DAYS, post_crisis_buffer=POST_CRISIS_BUF,
            recovery_mode=args.recovery, rebound_threshold=REBOUND_THRESHOLD,
            peak_window=args.peak_window,
        ).set_index("date")
        feats["ews_label"] = cl["y"].reindex(feats.index).astype("Float64").fillna(0).astype(int)
        feats["in_crisis"] = (cl["crisis_active"] | cl["in_buffer"]).reindex(feats.index).fillna(False)
        feats["drawdown"]  = cl["drawdown"].reindex(feats.index)
        frozen = int(cl["crisis_active"].sum()) + int(cl["in_buffer"].sum())
        log.info("Onset labeler [recovery=%s]: %d episodes, %d active + %d buffer = %d frozen (%.1f%%), %d NA(excluded)",
                 args.recovery, int(cl["crisis_start"].sum()), int(cl["crisis_active"].sum()),
                 int(cl["in_buffer"].sum()), frozen, frozen/len(feats)*100, int(cl["y"].isna().sum()))
    reg = build_regime_labels(feats["close"], vix.reindex(feats.index).ffill().dropna(),
                              drawdown_threshold=cfg.CRISIS_THRESHOLD, vix_stress=cfg.VIX_STRESS_THRESHOLD).set_index("date")
    feats = feats.join(reg[["regime_label"]], how="left").fillna(0)

    log.info("EWS label dist : %s", feats["ews_label"].value_counts().to_dict())
    log.info("In-crisis days : %d", int(feats["in_crisis"].sum()))
    log.info("Regime dist    : %s", feats["regime_label"].value_counts().to_dict())
    log.info("Feature cols (%d): %s", feats.shape[1], list(feats.columns))

    fwd = feats.reset_index().rename(columns={feats.reset_index().columns[0]: "date"})
    folds = walk_forward_crisis_splits(fwd, date_col="date", embargo_days=args.embargo)
    fold_dir = os.path.join(OUT, "folds"); os.makedirs(fold_dir, exist_ok=True)
    for f in folds:
        f["train"].to_parquet(os.path.join(fold_dir, f"{f['name']}_train.parquet"), index=False)
        f["val"].to_parquet(os.path.join(fold_dir, f"{f['name']}_val.parquet"), index=False)
        f["test"].to_parquet(os.path.join(fold_dir, f"{f['name']}_test.parquet"), index=False)

    if folds:
        last = folds[-1]
        last["train"].to_parquet(os.path.join(OUT, "task_b_train.parquet"), index=False)
        last["val"].to_parquet(os.path.join(OUT, "task_b_val.parquet"), index=False)
        last["test"].to_parquet(os.path.join(OUT, "task_b_test.parquet"), index=False)
        log.info("Primary split = fold '%s': %s", last["name"], last["meta"])
    else:
        tr, va, te = temporal_split(fwd, "date", cfg.TRAIN_END, cfg.VAL_END)
        tr.to_parquet(os.path.join(OUT, "task_b_train.parquet"), index=False)
        va.to_parquet(os.path.join(OUT, "task_b_val.parquet"), index=False)
        te.to_parquet(os.path.join(OUT, "task_b_test.parquet"), index=False)

    fwd.to_parquet(os.path.join(OUT, "task_b_features.parquet"), index=False)
    log.info("DONE -> %s", OUT)
    log.info("Folds produced: %s", [f["name"] for f in folds])

if __name__ == "__main__":
    main()
