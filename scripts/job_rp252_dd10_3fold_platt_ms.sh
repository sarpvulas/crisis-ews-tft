#!/bin/bash -l
#
# Same as job_rp252_dd10_3fold_ms.sh but with Platt post-hoc calibration:
# fits a logistic (Platt) scaler on val probs, applies to test, and writes
# calibrated_test_metrics alongside the raw test_metrics. Ranking metrics
# (PR-AUC/ROC-AUC) are unchanged; this improves Brier / calibration error.
# NOTE: the gfc fold's val set has no positives -> Platt is skipped there.
#
#SBATCH --job-name=rp252dd10-platt
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
echo "=== rp252dd10-platt seed=$SEED on $(hostname) at $(date -Iseconds) ==="
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
  echo; echo "--- seed=$SEED model=$1 fold=$3 (platt) ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$2" --tag "rp252dd10platt-$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/rp252dd10platt-${SLURM_ARRAY_JOB_ID}-${SEED}-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $3 failed"
}

for FOLD in gfc covid bear2022; do
  FDIR="$SCRATCH/folds_view_platt/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do
    ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"
  done

  if [ "$SEED" = "0" ]; then run xgboost "$FDIR" "$FOLD" ""; fi
  run lstm   "$FDIR" "$FOLD" "--epochs 40"
  run ra-tft "$FDIR" "$FOLD" "--epochs 40"
  run tdt    "$FDIR" "$FOLD" "--epochs 40"
done

echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
