#!/bin/bash -l
#
# CONFIRMATORY run on the EXPANDED 6-fold embargoed dataset (dd10_emb63_6f):
# folds = gfc, eurozone, china2015, selloff2018, covid, bear2022.
# Three key models only, 30 seeds x 6 folds -> doubles the independent crisis
# episodes (the real power lever) for the significance test:
#   conf-M0 = vanilla TFT
#   conf-M4 = full RA-TFT
#   conf-AF = AUX + focal/pw3.4  (the winning recipe)
# a100-pinned, deterministic cuBLAS, platt, 40 epochs.
#
#SBATCH --job-name=conf6f
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== conf6f seed=$SEED on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
DATA=~/ra-tft/data/shared/processed/taskb_spx_bbg_onset_trough_rp252_dd10_emb63_6f
RES=$SCRATCH/results
mkdir -p "$RES" "$SCRATCH/logs"
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb
cd ~/ra-tft

run () {  # <fold-dir> <fold> <tag-suffix> <extra-flags>
  echo; echo "--- seed=$SEED fold=$2 tag=$3 ---"
  python -u scripts/run_experiment.py \
    --model ra-tft --task B --seed "$SEED" --epochs 40 $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$1" --tag "conf-$3-$2" \
    --checkpoint-dir "$SCRATCH/checkpoints/conf6f-${SLURM_ARRAY_JOB_ID}-${SEED}-$3-$2" \
    --results-dir "$RES" || echo "!! seed=$SEED $3 $2 failed"
}

for FOLD in gfc eurozone china2015 selloff2018 covid bear2022; do
  FDIR="$SCRATCH/fv_conf/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  run "$FDIR" "$FOLD" M0 "--use-regime-module 0 --use-regime-attention 0 --gamma 0"
  run "$FDIR" "$FOLD" M4 ""
  run "$FDIR" "$FOLD" AF "--use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
