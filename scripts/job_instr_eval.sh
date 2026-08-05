#!/bin/bash -l
#
# INSTRUMENTED EVAL — re-run vanilla TFT (M0) and AUX+focal (AF) on the 6-fold
# embargoed single-market dataset with --dump-preds, so per-window test
# predictions+labels are saved (.npz). These feed (i) reliability/calibration
# diagrams and (ii) the Sarlin (2013) cost-sensitive usefulness metric, the two
# journal-level additions. 10 seeds x 6 folds x {M0, AF}; platt calibration.
# Tags instr-M0 / instr-AF -> files ra-tft__taskB__primary__seed{S}__instr-*-{fold}.json
# plus matching __preds.npz.
#
#SBATCH --job-name=instreval
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
echo "=== instreval seed=$SEED on $(hostname) at $(date -Iseconds) ==="

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

COMMON="--model ra-tft --task B --epochs 40 --encoder-steps 252 --decoder-steps 63 \
  --use-regime-module 0 --use-regime-attention 0 --device cuda \
  --posthoc-calibrate platt --dump-preds --results-dir $RES"

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_instr/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  echo; echo "--- seed=$SEED fold=$FOLD M0 ---"
  python -u scripts/run_experiment.py $COMMON --seed "$SEED" --gamma 0 \
    --data-dir "$FDIR" --tag "instr-M0-$FOLD" \
    --checkpoint-dir "$SCRATCH/checkpoints/instr-M0-${SLURM_ARRAY_JOB_ID}-${SEED}-$FOLD" \
    || echo "!! seed=$SEED M0 $FOLD failed"

  echo; echo "--- seed=$SEED fold=$FOLD AF ---"
  python -u scripts/run_experiment.py $COMMON --seed "$SEED" \
    --use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4 \
    --data-dir "$FDIR" --tag "instr-AF-$FOLD" \
    --checkpoint-dir "$SCRATCH/checkpoints/instr-AF-${SLURM_ARRAY_JOB_ID}-${SEED}-$FOLD" \
    || echo "!! seed=$SEED AF $FOLD failed"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
