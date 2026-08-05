#!/bin/bash -l
#
# Robustness/determinism run: 30 seeds x 3 folds on the cleaned 10% rolling-peak
# dataset. Determinism-hardened at the JOB level:
#   * pinned to a SINGLE GPU type (a100) -> no a100-vs-a40 float differences
#   * CUBLAS_WORKSPACE_CONFIG exported BEFORE python -> deterministic cuBLAS GEMMs
#   * 30 seeds -> tight, reproducible mean (shrinks selection-amplified variance)
# (cuDNN-LSTM bit-exactness is a separate follow-up; not forced here.)
#
#SBATCH --job-name=rp252dd10-det30
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # before any CUDA context -> deterministic cuBLAS
SEED=$SLURM_ARRAY_TASK_ID
echo "=== rp252dd10-det30 seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # $1=model $2=fold-dir $3=fold $4=extra
  echo; echo "--- seed=$SEED model=$1 fold=$3 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$2" --tag "rp252dd10det-$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/det30-${SLURM_ARRAY_JOB_ID}-${SEED}-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $3 failed"
}

for FOLD in gfc covid bear2022; do
  FDIR="$SCRATCH/fv_det30/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done
  if [ "$SEED" = "0" ]; then run xgboost "$FDIR" "$FOLD" ""; fi
  run lstm   "$FDIR" "$FOLD" "--epochs 40"
  run ra-tft "$FDIR" "$FOLD" "--epochs 40"
  run tdt    "$FDIR" "$FOLD" "--epochs 40"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
