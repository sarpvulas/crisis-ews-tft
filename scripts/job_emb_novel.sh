#!/bin/bash -l
#
# OVERNIGHT novel TFT-on-top variants on the embargoed clean splits.
# Baselines for pairing already exist at 30 seeds: emb2-M0 (vanilla), emb2-M4
# (full RA-TFT) — so this job only runs the NEW cells (tag prefix nov-):
#   Multi-task aux head : nov-aux, nov-auxfp3 (aux + focal/pw3.4)
#   Regime-cond. VSN    : nov-rvsn
#   MC-dropout (calib)  : nov-mc
#   num_regime_states   : nov-K2, nov-K4, nov-K5
#   focal_gamma sweep   : nov-FG1, nov-FG3  (on focal/pw3.4 base)
# 30 seeds x 3 folds, a100-pinned, deterministic cuBLAS, platt, 40 epochs.
#
#SBATCH --job-name=emb-novel
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-29
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --constraint='a100'
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err

set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
SEED=$SLURM_ARRAY_TASK_ID
echo "=== emb-novel seed=$SEED on $(hostname) at $(date -Iseconds) ==="
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
    --model ra-tft --task B --seed "$SEED" --epochs 40 $4 \
    --encoder-steps 252 --decoder-steps 63 --device cuda \
    --posthoc-calibrate platt \
    --data-dir "$1" --tag "nov-$3-$2" \
    --checkpoint-dir "$SCRATCH/checkpoints/nov-${SLURM_ARRAY_JOB_ID}-${SEED}-$3-$2" \
    --results-dir "$RES" || echo "!! seed=$SEED $3 $2 failed"
}

for FOLD in gfc covid bear2022; do
  FDIR="$SCRATCH/fv_nov/${SLURM_ARRAY_JOB_ID}-${SEED}/$FOLD"
  mkdir -p "$FDIR"
  for SP in train val test; do ln -sf "$DATA/folds/${FOLD}_${SP}.parquet" "$FDIR/task_b_${SP}.parquet"; done

  # multi-task aux head
  run "$FDIR" "$FOLD" aux    "--use-aux 1 --lambda-aux 0.5"
  run "$FDIR" "$FOLD" auxfp3 "--use-aux 1 --lambda-aux 0.5 --loss focal --focal-gamma 2 --pos-weight 3.4"
  # regime-conditioned VSN
  run "$FDIR" "$FOLD" rvsn   "--use-regime-vsn 1"
  # MC-dropout calibration (inference-time, 20 passes)
  run "$FDIR" "$FOLD" mc     "--mc-dropout 20"
  # num_regime_states sweep
  run "$FDIR" "$FOLD" K2     "--num-regime-states 2"
  run "$FDIR" "$FOLD" K4     "--num-regime-states 4"
  run "$FDIR" "$FOLD" K5     "--num-regime-states 5"
  # focal_gamma sweep on focal/pw3.4 base
  run "$FDIR" "$FOLD" FG1    "--loss focal --focal-gamma 1 --pos-weight 3.4"
  run "$FDIR" "$FOLD" FG3    "--loss focal --focal-gamma 3 --pos-weight 3.4"
done
echo; echo "=== seed=$SEED done at $(date -Iseconds) ==="
