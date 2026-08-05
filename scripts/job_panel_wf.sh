#!/bin/bash -l
#
# PANEL WALK-FORWARD — robustness check for the panel result. Instead of a
# single 2019-2026 test split, evaluate the panel on FIVE crisis-anchored
# walk-forward folds (expanding train, calm val, crisis test), pooled across the
# 9 markets, with a per-market 63-day embargo. Compares the tuned deep recipe
# (AFv2, capacity 128 + per-market norm) against vanilla+norm (M0pm) and
# XGBoost. This turns the single-split panel claim into an episode-level one.
# Tags pwf-AFv2-<fold> / pwf-M0pm-<fold> / pwf-xgb-<fold>.
#
#SBATCH --job-name=panelwf
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== panelwf seed=$SEED on $(hostname) at $(date -Iseconds) ==="

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
PANEL=~/ra-tft/data/shared/processed/panel_dd10_rp252/panel.parquet
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

# fold: name train_end val_end test_end (expanding train, 2-year val so >=315-day
# windows form, pooled across all 9 markets). val too short -> 0 val windows.
FOLDS=(
  "gfc 2005-06-30 2007-06-30 2010-06-30"
  "eurozone 2008-12-31 2010-12-31 2013-06-30"
  "china2015 2012-06-30 2014-06-30 2016-12-31"
  "covid 2016-06-30 2018-06-30 2021-06-30"
  "bear2022 2019-06-30 2021-06-30 2023-12-31"
)

panelrun () {  # <tag> <fold> <tr> <vl> <te> <extra>
  echo; echo "--- seed=$SEED $1 ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" $6 \
    --panel-path "$PANEL" --panel-train-end "$3" --panel-val-end "$4" --panel-test-end "$5" \
    --panel-per-market-norm 1 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt --tag "$1" \
    --checkpoint-dir "$SCRATCH/checkpoints/$1-${SLURM_ARRAY_JOB_ID}-${SEED}" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 failed"
}

AFV2="--epochs 40 --state-size 128 --hidden-size 128 --lr 5e-4 --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 2.5"
M0PM="--epochs 40 --use-regime-module 0 --use-regime-attention 0 --gamma 0"

for row in "${FOLDS[@]}"; do
  read -r FN TR VL TE <<< "$row"
  panelrun "pwf-AFv2-$FN" "$FN" "$TR" "$VL" "$TE" "$AFV2"
  panelrun "pwf-M0pm-$FN" "$FN" "$TR" "$VL" "$TE" "$M0PM"
  if [ "$SEED" = "0" ]; then
    echo; echo "--- seed=$SEED pwf-xgb-$FN ---"
    python -u scripts/run_experiment.py --model xgboost --task B --seed 0 \
      --panel-path "$PANEL" --panel-train-end "$TR" --panel-val-end "$VL" --panel-test-end "$TE" \
      --panel-per-market-norm 1 --encoder-steps 252 --decoder-steps 63 --device cuda \
      --posthoc-calibrate platt --tag "pwf-xgb-$FN" \
      --checkpoint-dir "$SCRATCH/checkpoints/pwf-xgb-${SLURM_ARRAY_JOB_ID}-$FN" \
      --results-dir "$RES" || echo "!! seed=$SEED xgb $FN failed"
  fi
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
