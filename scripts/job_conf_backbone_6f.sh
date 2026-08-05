#!/bin/bash -l
#
# BACKBONE RECONCILIATION — run LSTM + XGBoost on the SAME 6-fold embargoed
# single-market dataset used by the conf-M0/M4/AF headline, so the backbone
# table (TFT vs LSTM vs XGBoost) shares one protocol with the AF table instead
# of citing an older 3-fold exploratory run. Vanilla TFT (conf-M0) already
# exists on these 6 folds; here we add LSTM (30 seeds) + XGBoost (deterministic,
# seed0). Tags conf-lstm-<fold> / conf-xgb-<fold> -> merge with conf-* aggregation.
#
#SBATCH --job-name=cbackbone
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== cbackbone seed=$SEED on $(hostname) at $(date -Iseconds) ==="

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_cbb/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  echo; echo "--- seed=$SEED fold=$FOLD lstm ---"
  python -u scripts/run_experiment.py \
    --model lstm --task B --seed "$SEED" --epochs 40 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$FDIR" --tag "conf-lstm-$FOLD" \
    --checkpoint-dir "$SCRATCH/checkpoints/cbb-lstm-${SLURM_ARRAY_JOB_ID}-${SEED}-$FOLD" \
    --results-dir "$RES" || echo "!! seed=$SEED lstm $FOLD failed"

  if [ "$SEED" = "0" ]; then
    echo; echo "--- seed=$SEED fold=$FOLD xgboost (deterministic) ---"
    python -u scripts/run_experiment.py \
      --model xgboost --task B --seed "$SEED" \
      --encoder-steps 252 --decoder-steps 63 --device cuda \
      --posthoc-calibrate platt \
      --data-dir "$FDIR" --tag "conf-xgb-$FOLD" \
      --checkpoint-dir "$SCRATCH/checkpoints/cbb-xgb-${SLURM_ARRAY_JOB_ID}-$FOLD" \
      --results-dir "$RES" || echo "!! seed=$SEED xgb $FOLD failed"
  fi
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
