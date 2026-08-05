#!/bin/bash -l
#
# MULTI-MARKET PANEL experiment — the thesis's multi-entity reformulation.
# 9 broad equity indices (panel_dd10_rp252/panel.parquet), single temporal split
# (train<=2015, val<=2018, test 2019-2026) with 63-day embargo + market-identity
# static-categorical embedding. ~110 crisis onset episodes across the panel ->
# the statistical power single-market (3-6 episodes) lacked.
# Models: baselines {xgb[seed0], lstm, tft} + regime ablation/winner {M0,M4,AF}.
# 15 seeds, a100-pinned, deterministic cuBLAS, platt, 40 epochs.
#
#SBATCH --job-name=panel
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== panel seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
PANEL=~/ra-tft/data/shared/processed/panel_dd10_rp252/panel.parquet
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <model> <tag> <extra-flags>
  echo; echo "--- seed=$SEED model=$1 tag=$2 ---"
  python -u scripts/run_experiment.py \
    --model "$1" --task B --seed "$SEED" $3 \
    --panel-path "$PANEL" --panel-train-end 2015-12-31 --panel-val-end 2018-12-31 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --tag "panel-$2" \
    --checkpoint-dir "$SCRATCH/checkpoints/panel-${SLURM_ARRAY_JOB_ID}-${SEED}-$2" \
    --results-dir "$RES" || echo "!! seed=$SEED $1 $2 failed"
}

if [ "$SEED" = "0" ]; then run xgboost base "" ; fi
run lstm   base   "--epochs 40"
run tft    base   "--epochs 40"
run ra-tft M0     "--epochs 40 --use-regime-module 0 --use-regime-attention 0 --gamma 0"
run ra-tft M4     "--epochs 40"
run ra-tft AF     "--epochs 40 --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4"
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
