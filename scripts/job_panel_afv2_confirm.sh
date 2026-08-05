#!/bin/bash -l
#
# PANEL AFv2 CONFIRMATORY — lock the headline "deep model beats XGBoost on the
# 9-market panel". The improvement run (job 35199368) gave AFv2 0.399 PR vs
# XGBoost 0.373 but only n=8 landed (a100 preemption) -> AFv2-vs-AF p=0.094
# (marginal). This run adds 20 fresh seeds (12-31), AFv2 cell ONLY, so almost
# every task lands one usable AFv2 result. Merges by seed with the existing
# panel-AFv2 files (seeds 0-11). Same config as job_panel_v2.sh AFv2 cell.
#
#SBATCH --job-name=afv2conf
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=12-31
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== afv2conf seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
PANEL=~/ra-tft/data/shared/processed/panel_dd10_rp252/panel.parquet
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

echo; echo "--- seed=$SEED tag=AFv2 ---"
python -u scripts/run_experiment.py \
  --model ra-tft --task B --seed "$SEED" \
  --epochs 60 --state-size 128 --hidden-size 128 --lr 5e-4 \
  --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 2.5 \
  --panel-path "$PANEL" --panel-train-end 2015-12-31 --panel-val-end 2018-12-31 \
  --panel-per-market-norm 1 \
  --encoder-steps 252 --decoder-steps 63 --device cuda \
  --posthoc-calibrate platt \
  --tag "panel-AFv2" \
  --checkpoint-dir "$SCRATCH/checkpoints/afv2conf-${SLURM_ARRAY_JOB_ID}-${SEED}" \
  --results-dir "$RES" || echo "!! seed=$SEED AFv2 failed"
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
