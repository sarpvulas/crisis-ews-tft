#!/bin/bash -l
#
# Confirmatory 30-seed run on the EMBARGOED clean splits — powers the M4-vs-M0
# significance test that job 35123500 left underpowered (preemption -> n5-10).
# Only the 3 decision cells, so each task is short (~9 model-folds) and survives
# a100 preemption better:
#   emb2-M0   = vanilla TFT (regime all off)
#   emb2-M4   = full RA-TFT
#   emb2-Lfp3 = full RA-TFT + focal loss + pos_weight 3.4 (best loss config)
# 30 seeds x 3 folds, a100-pinned, deterministic cuBLAS, platt, 40 epochs.
#
#SBATCH --job-name=emb-confirm
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== emb-confirm seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <fold-dir> <fold> <tag-suffix> <extra-flags>
  echo; echo "--- seed=$SEED fold=$2 tag=$3 ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$1" --tag "emb2-$3-$2" \
    --checkpoint-dir "$SCRATCH/checkpoints/emb2-${SLURM_ARRAY_JOB_ID}-${SEED}-$3-$2" \
    --results-dir "$RES" || echo "!! seed=$SEED $3 $2 failed"
}

for FOLD in gfc covid bear2022; do
  FDIR="$SCRATCH/fv_emb2/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  run "$FDIR" "$FOLD" M0   "--epochs 40 --use-regime-module 0 --use-regime-attention 0 --gamma 0"
  run "$FDIR" "$FOLD" M4   "--epochs 40"
  run "$FDIR" "$FOLD" Lfp3 "--epochs 40 --loss focal --focal-gamma 2 --pos-weight 3.4"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
