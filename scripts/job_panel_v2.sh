#!/bin/bash -l
#
# PANEL IMPROVEMENT experiment — push the deep model past XGBoost (0.373 PR /
# 0.585 ROC) on the 9-market panel. Builds on AUX+focal with panel-specific
# levers chosen by the user:
#   panel-M0pm  = vanilla TFT + per-market normalization      (does norm help the baseline?)
#   panel-AFpm  = AUX+focal   + per-market normalization      (isolate the norm effect on AF)
#   panel-AFv2  = AUX+focal + per-market-norm + capacity(128) + 60 epochs + lr 5e-4
#                 + pos_weight 2.5 (retuned for panel prevalence ~0.30)  [kitchen-sink]
# Baselines to beat already exist: panel-M0/M4/AF (global norm) + XGBoost 0.373.
# 12 seeds, a100, deterministic cuBLAS, platt.
#
#SBATCH --job-name=panelv2
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== panelv2 seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
PANEL=~/ra-tft/data/shared/processed/panel_dd10_rp252/panel.parquet
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <tag> <extra-flags>
  echo; echo "--- seed=$SEED tag=$1 ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" $2 \
    --panel-path "$PANEL" --panel-train-end 2015-12-31 --panel-val-end 2018-12-31 \
    --panel-per-market-norm 1 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --tag "panel-$1" \
    --checkpoint-dir "$SCRATCH/checkpoints/panelv2-${SLURM_ARRAY_JOB_ID}-${SEED}-$1" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 failed"
}

run M0pm "--epochs 40 --use-regime-module 0 --use-regime-attention 0 --gamma 0"
run AFpm "--epochs 40 --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4"
run AFv2 "--epochs 60 --state-size 128 --hidden-size 128 --lr 5e-4 --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 2.5"
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
