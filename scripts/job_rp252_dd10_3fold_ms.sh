#!/bin/bash -l
#
# Multi-seed, 3-fold comparison on the cleaned shared Task B:
#   * rolling-252-day peak drawdown (no all-time-high phantom onsets)
#   * 10% drawdown onset threshold (market-stress EWS; more learnable events)
#   * checkpoint selection on val PR-AUC + full deterministic seeding (synced code)
# Job ARRAY: one seed per array task. Each task runs all 3 walk-forward folds
# (gfc, covid, bear2022) x {xgboost, lstm, ra-tft, tdt}.
#
#SBATCH --job-name=rp252dd10-3f
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100|a40'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "=== rp252dd10-3fold seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # $1=model  $2=fold-dir  $3=fold-name  $4=extra-flags
  echo; echo "--- seed=$SEED model=$1 fold=$3 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --data-dir "$2" --tag "rp252dd10-$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/rp252dd10-${SLURM_ARRAY_JOB_ID}-${SEED}-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $3 failed"
}

for FOLD in gfc covid bear2022; do
  # CrisisDataset loads task_b_{split}.parquet from --data-dir; build a per-fold
  # view by symlinking the fold's files under that canonical name (no data copy).
  FDIR="$SCRATCH/folds_view/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do
    ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"
  done

  # xgboost is deterministic across seeds -> run it once (seed 0 only).
  if [ "$SEED" = "0" ]; then run xgboost "$FDIR" "$FOLD" ""; fi
  run lstm   "$FDIR" "$FOLD" "--epochs 40"
  run ra-tft "$FDIR" "$FOLD" "--epochs 40"
  run tdt    "$FDIR" "$FOLD" "--epochs 40"
done

echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
