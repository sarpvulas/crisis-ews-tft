#!/bin/bash -l
#
# Single-job training script for KCL CREATE HPC.
#
# Usage:
#   sbatch scripts/job_train.sh --model ra-tft --task B --seed 0 --epochs 50
#
# Args after the sbatch directives are forwarded to scripts/run_experiment.py.
# The script is preemption-safe: it writes checkpoints every epoch to
# /scratch/<user>/ra-tft/checkpoints/<job-name>/ and resumes from the latest
# on SLURM requeue.

#SBATCH --job-name=ra-tft
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%j.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%j.err

set -euo pipefail

echo "=== Job $SLURM_JOB_ID started on $(hostname) at $(date -Iseconds) ==="
echo "Args: $@"

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

# Where to write checkpoints / wandb cache. Logs are in /scratch via #SBATCH above.
SCRATCH=/scratch/users/$USER/ra-tft
CKPT_DIR=$SCRATCH/checkpoints/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p "$CKPT_DIR" "$SCRATCH/wandb" "$SCRATCH/results"

# Wandb in offline mode — sync from Mac later with `wandb sync /scratch/.../wandb/<run>`.
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

# Graceful preemption: trap USR1 -> exit clean -> Slurm requeues.
trap 'echo "[$(date -Iseconds)] received USR1 (preemption), exiting cleanly"; exit 0' USR1

cd ~/ra-tft

python -u scripts/run_experiment.py \
    --checkpoint-dir "$CKPT_DIR" \
    --results-dir "$SCRATCH/results" \
    --resume-latest \
    "$@"

echo "=== Job $SLURM_JOB_ID finished at $(date -Iseconds) ==="
