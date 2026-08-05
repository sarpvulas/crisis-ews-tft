#!/bin/bash -l
#
# Does the PANEL-WINNING config (AFv2) also win SINGLE-MARKET? Runs the exact
# AFv2 recipe (AUX+focal + state/hidden 128 + 60 epochs + lr 5e-4 + pos_weight
# 2.5) on the 6-fold embargoed single-market dd10 dataset, to compare against
# the lean AF winner (conf-AF) and vanilla (conf-M0). Hypothesis: the big model
# overfits the small single-market folds -> capacity must match data scale.
# Tag conf-AFv2sm-<fold>. 15 seeds x 6 folds, a100, platt.
#
#SBATCH --job-name=afv2sm
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-14
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== afv2sm seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_afv2sm/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done
  echo; echo "--- seed=$SEED fold=$FOLD AFv2sm ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" --epochs 60 \
    --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 2.5 \
    --state-size 128 --hidden-size 128 --lr 5e-4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$FDIR" --tag "conf-AFv2sm-$FOLD" \
    --checkpoint-dir "$SCRATCH/checkpoints/afv2sm-${SLURM_ARRAY_JOB_ID}-${SEED}-$FOLD" \
    --results-dir "$RES" || echo "!! seed=$SEED AFv2sm $FOLD failed"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
