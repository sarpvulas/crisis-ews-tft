#!/bin/bash -l
#
# Tier-1 ablations on the EMBARGOED, leak-fixed dd10 dataset (taskb_..._dd10_emb63):
#   (A) Regime factorial  — decompose RA-TFT into {module, attention, aux-loss}
#   (B) Loss/objective     — focal vs weighted-BCE, and pos_weight 10 vs 3.4
#   (C) Baselines          — vanilla tft, lstm, xgboost on the SAME clean splits
# 10 seeds x 3 folds, a100-pinned, deterministic cuBLAS, platt calibration, 40 epochs.
# Each cell gets a unique --tag so all results coexist (model 'ra-tft' repeats).
#
#SBATCH --job-name=emb-abl
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
echo "=== emb-abl seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

# run <model> <fold-dir> <fold> <tag-suffix> <extra-flags>
run () {
  echo; echo "--- seed=$SEED model=$1 fold=$3 tag=$4 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $5 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$2" --tag "emb-$4-$3" \
    --checkpoint-dir "$SCRATCH/checkpoints/emb-${SLURM_ARRAY_JOB_ID}-${SEED}-$4-$3" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $4 $3 failed"
}

for FOLD in gfc covid bear2022; do
  FDIR="$SCRATCH/fv_emb/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  # (C) baselines on clean splits
  if [ "$SEED" = "0" ]; then run xgboost "$FDIR" "$FOLD" base "" ; fi
  run lstm "$FDIR" "$FOLD" base "--epochs 40"
  run tft  "$FDIR" "$FOLD" base "--epochs 40"

  # (A) regime factorial (all use the ra-tft class; toggles select the cell)
  run ra-tft "$FDIR" "$FOLD" M0 "--epochs 40 --use-regime-module 0 --use-regime-attention 0 --gamma 0"
  run ra-tft "$FDIR" "$FOLD" M1 "--epochs 40 --use-regime-module 1 --use-regime-attention 0 --gamma 0"
  run ra-tft "$FDIR" "$FOLD" M2 "--epochs 40 --use-regime-module 1 --use-regime-attention 0 --gamma 0.3"
  run ra-tft "$FDIR" "$FOLD" M3 "--epochs 40 --use-regime-module 1 --use-regime-attention 1 --gamma 0"
  run ra-tft "$FDIR" "$FOLD" M4 "--epochs 40 --use-regime-module 1 --use-regime-attention 1 --gamma 0.3"

  # (B) loss/objective study on the full RA-TFT (M4) config
  run ra-tft "$FDIR" "$FOLD" Lfocal    "--epochs 40 --loss focal --focal-gamma 2 --pos-weight 10"
  run ra-tft "$FDIR" "$FOLD" Lpw3      "--epochs 40 --pos-weight 3.4"
  run ra-tft "$FDIR" "$FOLD" Lfocalpw3 "--epochs 40 --loss focal --focal-gamma 2 --pos-weight 3.4"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
