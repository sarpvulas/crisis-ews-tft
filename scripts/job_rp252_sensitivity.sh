#!/bin/bash -l
#
# Drawdown-threshold sensitivity sweep: thresholds 12/15/20% x 3 folds x 10 seeds
# on the rolling-252 peak labeling. (The 10% point comes from the det30 job.)
# Same job-level determinism hardening: a100-pinned + CUBLAS_WORKSPACE_CONFIG.
#
#SBATCH --job-name=rp252-sens
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== rp252-sens seed=$SEED on $(hostname) at $(date -Iseconds) ==="

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # $1=model $2=fold-dir $3=fold $4=dd-tag $5=extra
  echo; echo "--- seed=$SEED model=$1 fold=$3 thr=$4 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $5 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$2" --tag "rp252$4-$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/sens-${SLURM_ARRAY_JOB_ID}-${SEED}-$4-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $4 $3 failed"
}

for DD in dd12 dd15 dd20; do
  DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_$DD
  for FOLD in gfc covid bear2022; do
    FDIR="$SCRATCH/fv_sens/${SLURM_ARRAY_JOB_ID}-${SEED}/$DD/$FOLD"
    mkdir -p "$FDIR"
    for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done
    if [ "$SEED" = "0" ]; then run xgboost "$FDIR" "$FOLD" "$DD" ""; fi
    run lstm   "$FDIR" "$FOLD" "$DD" "--epochs 40"
    run ra-tft "$FDIR" "$FOLD" "$DD" "--epochs 40"
    run tdt    "$FDIR" "$FOLD" "$DD" "--epochs 40"
  done
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
