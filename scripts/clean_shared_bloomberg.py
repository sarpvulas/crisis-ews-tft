"""
Clean the lecturer's shared Bloomberg xlsx exports into tidy parquet.

Each xlsx is a Bloomberg multi-security paste with a multi-row header block:

    r?:  Start Date | 2000-01-01
    r?:  End Date   | ...
    rT:  (group)    | JP Morgan        | ...          # optional (bank files)
    rT': (ticker)   | SPX Index        | (blank->ffill across its fields)
    rL:  (label)    | Last Price       | Volume       # optional human labels
    rD:  Dates      | PX_LAST          | PX_VOLUME    # field-code header row
    rD+: 2000-01-03 | 1455.22          | 997052300    # daily data

Strategy: locate the row whose first cell == "Dates" (rD). Field codes come
from rD; the ticker row is the nearest row above rD whose cells look like
Bloomberg identifiers (forward-filled across merged-cell blanks); an optional
group row above that carries entity names. Emits:

    data/shared/processed/long.parquet     (date, source, group, ticker, field, value)
    data/shared/processed/wide.parquet     (date index x  "ticker | field"  columns)
    data/shared/processed/manifest.csv      (per-series coverage / missingness)
"""
import os, re, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SRC_DIR = r"C:\Users\Sarp\git\ra-tft\data\shared\expanded"
OUT_DIR = r"C:\Users\Sarp\git\ra-tft\data\shared\processed"
os.makedirs(OUT_DIR, exist_ok=True)

TICKER_RE = re.compile(r"(Index|Equity|Curncy|Comdty|CDS|CDSI|Comdty)", re.I)

def _is_ticker_row(vals):
    cells = [str(v) for v in vals if isinstance(v, str) and v.strip()]
    if not cells:
        return False
    hits = sum(bool(TICKER_RE.search(c)) for c in cells)
    return hits >= max(1, len(cells) // 2)

def parse_file(path):
    raw = pd.read_excel(path, engine="openpyxl", header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip().str.lower()
    dmatch = col0[col0 == "dates"].index
    if len(dmatch) == 0:
        raise ValueError(f"no 'Dates' header row found in {os.path.basename(path)}")
    D = int(dmatch[0])

    fields = raw.iloc[D].copy()
    # ticker row: nearest row above D that looks like identifiers
    T = None
    for r in range(D - 1, max(-1, D - 5), -1):
        if _is_ticker_row(raw.iloc[r].tolist()[1:]):
            T = r
            break
    if T is None:
        T = D - 1
    tickers = raw.iloc[T].copy()
    tickers.iloc[1:] = tickers.iloc[1:].replace("", np.nan).ffill()  # merged cells

    # optional group/entity row directly above ticker row
    group = None
    if T - 1 >= 0:
        cand = raw.iloc[T - 1]
        if cand.iloc[1:].replace("", np.nan).notna().any() and not _is_ticker_row(cand.tolist()[1:]):
            group = cand.replace("", np.nan).ffill()

    data = raw.iloc[D + 1:].copy()
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    data = data[dates.notna()]
    dates = dates[dates.notna()]

    records = []
    for c in range(1, raw.shape[1]):
        tk = tickers.iloc[c]
        fd = fields.iloc[c]
        if (not isinstance(tk, str) or not tk.strip()) and (not isinstance(fd, str) or not fd.strip()):
            continue
        vals = pd.to_numeric(data.iloc[:, c], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        gp = group.iloc[c] if group is not None else np.nan
        sub = pd.DataFrame({
            "date": dates.values,
            "source": os.path.basename(path),
            "group": gp,
            "ticker": str(tk).strip() if isinstance(tk, str) else np.nan,
            "field": str(fd).strip() if isinstance(fd, str) else np.nan,
            "value": vals.values,
        })
        records.append(sub[sub["value"].notna()])
    if not records:
        return pd.DataFrame(columns=["date","source","group","ticker","field","value"])
    return pd.concat(records, ignore_index=True)

def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.xlsx")))
    files = [f for f in files if not os.path.basename(f).startswith("~$")]
    all_long = []
    print(f"Parsing {len(files)} files from {SRC_DIR}\n")
    for f in files:
        try:
            lf = parse_file(f)
            n_series = lf.groupby(["ticker","field"]).ngroups if len(lf) else 0
            print(f"  {os.path.basename(f):56s} series={n_series:4d}  rows={len(lf):8d}")
            all_long.append(lf)
        except Exception as e:
            print(f"  !! {os.path.basename(f):56s} FAILED: {e}")
    long = pd.concat(all_long, ignore_index=True)

    # series key + dedup duplicates (same ticker|field across files): keep most-populated
    long["key"] = long["ticker"].fillna("?") + " | " + long["field"].fillna("?")
    cov = long.groupby(["key","source"]).size().rename("n").reset_index()
    best = cov.sort_values("n", ascending=False).drop_duplicates("key")[["key","source"]]
    best_set = set(map(tuple, best.values))
    long["_keep"] = [ (k,s) in best_set for k,s in zip(long["key"], long["source"]) ]
    long = long[long["_keep"]].drop(columns="_keep")

    # wide matrix on a daily business-day calendar
    wide = long.pivot_table(index="date", columns="key", values="value", aggfunc="first")
    full_idx = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(full_idx)
    wide.index.name = "date"

    # manifest
    man = (long.groupby("key")
              .agg(source=("source","first"), ticker=("ticker","first"),
                   field=("field","first"), group=("group","first"),
                   n_valid=("value","size"),
                   first=("date","min"), last=("date","max"))
              .reset_index())
    man["pct_missing"] = (1 - man["n_valid"] / len(wide)).round(3)
    man = man.sort_values(["source","key"]).reset_index(drop=True)

    long.drop(columns="key").to_parquet(os.path.join(OUT_DIR, "long.parquet"), index=False)
    wide.to_parquet(os.path.join(OUT_DIR, "wide.parquet"))
    man.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)

    print("\n" + "="*72)
    print(f"TOTAL unique series : {wide.shape[1]}")
    print(f"Daily calendar      : {wide.index.min().date()} -> {wide.index.max().date()}  ({len(wide)} business days)")
    print(f"Overall density     : {long['value'].notna().sum() / (wide.shape[0]*wide.shape[1]):.1%} cells populated")
    print(f"Written             : long.parquet, wide.parquet, manifest.csv  -> {OUT_DIR}")
    print("="*72)
    print("\nSeries per source:")
    print(man.groupby("source")["key"].count().to_string())
    print("\nSparsest 12 series (highest % missing):")
    print(man.sort_values("pct_missing", ascending=False)
             .head(12)[["key","source","first","last","pct_missing"]].to_string(index=False))

if __name__ == "__main__":
    main()
