"""Build a 50/50 chronological split of the Task-B dataset, in either direction.

Supervisor-requested experiment: cut the sample in half and train on the RIGHT
(recent) half while testing on the LEFT (old) half — i.e. deliberately backwards
in time. The motivation is statistical, not operational: under the walk-forward
protocol the GFC fold can only be trained on 2000-2004, which is why it is the
weakest fold. Reversing the arrow of time lets the model learn from the
data-rich modern era (China-2015, 2018, COVID, 2022) and be tested on the two
largest episodes in the sample (dot-com, GFC) plus the eurozone crisis.

Because a reversed result is uninterpretable on its own, this script also builds
the FORWARD split over the identical halves (`--direction forward`) as the
control: same boundary, same sizes, same embargo, only the arrow of time
differs. The pair answers "does the model generalise backwards as well as it
generalises forwards?"

Layout in both directions is train -> val -> test ordered by DISTANCE FROM THE
TEST SET, so validation always sits between train and test exactly as it does in
the walk-forward folds:

    forward:   [ train ][ val ][ TEST ]        (test = recent half)
    reverse:   [ TEST ][ val ][ train ]        (test = old half)

Embargo: labels look 63 trading days forward, so the last `--embargo` rows of
whichever block precedes the next one in CALENDAR time are purged. In the
reverse direction that means purging the tail of TEST (its labels are determined
by prices the model trained on) and the tail of VAL.

Output is written as task_b_{train,val,test}.parquet, the layout CrisisDataset
expects from --data-dir.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

DEFAULT_SRC = os.path.join(
    _REPO, "data", "shared", "processed",
    "taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f", "task_b_features.parquet",
)


def describe(name: str, df: pd.DataFrame, window: int) -> None:
    if df.empty:
        print(f"  {name:<6} EMPTY")
        return
    onsets = int((df["ews_label"].diff() == 1).sum()) + int(df["ews_label"].iloc[0] == 1)
    n_win = max(0, len(df) - window + 1)
    print(f"  {name:<6} rows={len(df):>5}  {df['date'].min().date()} -> {df['date'].max().date()}"
          f"  windows~{n_win:>5}  pos={int(df['ews_label'].sum()):>4}"
          f"  rate={df['ews_label'].mean():.3f}  onsets={onsets}"
          f"  in_crisis={int(df['in_crisis'].sum()):>4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="Source task_b_features.parquet (labelled full history).")
    ap.add_argument("--direction", choices=["reverse", "forward"], required=True,
                    help="reverse = train on recent half, test on old half (the "
                         "supervisor's request). forward = the control.")
    ap.add_argument("--val-rows", type=int, default=756,
                    help="Validation block length in trading rows (~3y). Must exceed "
                         "encoder+decoder=315 or the split yields zero windows.")
    ap.add_argument("--embargo", type=int, default=63,
                    help="Rows purged from the tail of each block that precedes "
                         "another in calendar time (= decoder_steps).")
    ap.add_argument("--out", default=None, help="Output directory.")
    args = ap.parse_args()

    window = 252 + 63

    df = pd.read_parquet(args.src)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    mid = n // 2
    print(f"source: {args.src}")
    print(f"  {n} rows, {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  50/50 boundary at row {mid} = {df['date'].iloc[mid].date()}")

    old, new = df.iloc[:mid].copy(), df.iloc[mid:].copy()

    if args.direction == "reverse":
        # [ TEST = old ][ val ][ train ]  — val adjacent to test, train newest.
        test = old
        val = new.iloc[: args.val_rows].copy()
        train = new.iloc[args.val_rows:].copy()
        # test and val each precede another block in calendar time.
        test = test.iloc[: len(test) - args.embargo].copy()
        val = val.iloc[: len(val) - args.embargo].copy()
    else:
        # [ train ][ val ][ TEST = new ]  — the standard arrangement.
        test = new
        val = old.iloc[len(old) - args.val_rows:].copy()
        train = old.iloc[: len(old) - args.val_rows].copy()
        train = train.iloc[: len(train) - args.embargo].copy()
        val = val.iloc[: len(val) - args.embargo].copy()

    out = args.out or os.path.join(
        _REPO, "data", "shared", "processed", f"taskb_half_{args.direction}")
    os.makedirs(out, exist_ok=True)

    print(f"\ndirection = {args.direction}  (embargo {args.embargo} rows, val {args.val_rows} rows)")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        describe(name, part, window)
        part.reset_index(drop=True).to_parquet(
            os.path.join(out, f"task_b_{name}.parquet"), index=False)

    # Guard rails: silent zero-window splits are the classic failure here.
    problems = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        if len(part) < window:
            problems.append(f"{name} has {len(part)} rows < window {window} -> ZERO windows")
        if name != "test" and part["ews_label"].sum() == 0:
            problems.append(f"{name} has no positive labels")
    if val["ews_label"].sum() == 0:
        problems.append("val has no positives -> PR-AUC selection and Platt both degrade")
    for p in problems:
        print(f"  !! WARNING: {p}")

    # Confirm the embargo really opened a calendar gap.
    if args.direction == "reverse":
        gap = (val["date"].min() - test["date"].max()).days
        print(f"\n  embargo gap test->val : {gap} calendar days")
        gap2 = (train["date"].min() - val["date"].max()).days
        print(f"  embargo gap val->train: {gap2} calendar days")
    else:
        gap = (val["date"].min() - train["date"].max()).days
        print(f"\n  embargo gap train->val: {gap} calendar days")
        gap2 = (test["date"].min() - val["date"].max()).days
        print(f"  embargo gap val->test : {gap2} calendar days")

    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    main()
