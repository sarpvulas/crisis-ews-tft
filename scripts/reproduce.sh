#!/usr/bin/env bash
#
# Reproduce the single-market headline result with Docker: the AF recipe
# (auxiliary forward-drawdown head + focal loss) vs. the vanilla TFT, across the
# six crisis-anchored walk-forward folds, at seed 0.
#
# Usage (from the repo root, after `docker compose build train`):
#     bash scripts/reproduce.sh
#     EPOCHS=10 bash scripts/reproduce.sh        # faster smoke run
#
# Note: this is CPU-only and runs ~12 trainings, so it takes a while. The
# dissertation numbers average ~30 seeds on GPU (see scripts/job_conf_6fold.sh);
# seed 0 here reproduces the *direction* (AF >= vanilla, esp. on severe crises),
# not the exact multi-seed means.
#
set -euo pipefail
DS=data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
FOLDS="gfc eurozone china2015 selloff2018 covid bear2022"
EPOCHS="${EPOCHS:-40}"
mkdir -p results checkpoints

if [ ! -d "$DS/folds" ]; then
  # The original Bloomberg-derived data is not distributed; fall back to the
  # committed mock surrogate dataset (see README "Data & licensing").
  DS=data/mock/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
  echo "NOTE: using the mock surrogate dataset at $DS"
  echo "      (expect deviations from the dissertation numbers; directions should hold)"
fi
if [ ! -d "$DS/folds" ]; then
  echo "ERROR: dataset not found at $DS/folds — is the data present?" >&2
  exit 1
fi

for FOLD in $FOLDS; do
  DIR="$DS/_repro/$FOLD"
  mkdir -p "$DIR"
  for SP in train val test; do cp -f "$DS/folds/${FOLD}_${SP}.parquet" "$DIR/task_b_${SP}.parquet"; done

  echo ">>> fold=$FOLD : vanilla TFT (M0)"
  docker compose run --rm train --model ra-tft --task B --seed 0 --epochs "$EPOCHS" \
    --use-regime-module 0 --use-regime-attention 0 --gamma 0 --posthoc-calibrate platt \
    --data-dir "$DIR" --tag "repro-M0-$FOLD" \
    --results-dir results --checkpoint-dir checkpoints --device auto

  echo ">>> fold=$FOLD : AF recipe"
  docker compose run --rm train --model ra-tft --task B --seed 0 --epochs "$EPOCHS" \
    --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4 --posthoc-calibrate platt \
    --data-dir "$DIR" --tag "repro-AF-$FOLD" \
    --results-dir results --checkpoint-dir checkpoints --device auto
done

echo
echo "=== summary: test PR-AUC (seed 0) — vanilla TFT vs AF ==="
docker compose run --rm -T --entrypoint python train - <<'PY'
import json, glob, statistics as st
FOLDS = "gfc eurozone china2015 selloff2018 covid bear2022".split()
def pr(tag, fold):
    fs = glob.glob("results/ra-tft__taskB__primary__seed0__%s-%s.json" % (tag, fold))
    if not fs:
        return None
    return (json.load(open(fs[0])).get("test_metrics") or {}).get("pr_auc")
print("%-12s %8s %8s" % ("fold", "vanilla", "AF"))
m0, af = [], []
for fl in FOLDS:
    a, b = pr("repro-M0", fl), pr("repro-AF", fl)
    if a is not None: m0.append(a)
    if b is not None: af.append(b)
    print("%-12s %8s %8s" % (fl, "%.3f" % a if a is not None else "--", "%.3f" % b if b is not None else "--"))
if m0 and af:
    print("%-12s %8.3f %8.3f" % ("macro", st.mean(m0), st.mean(af)))
PY

rm -rf "$DS/_repro"
echo "Done. Per-run metrics JSONs are in results/."
