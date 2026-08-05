#!/bin/bash -l
#
# Stage-2 HP sweep array job.
#
# Usage:
#   sbatch --array=0-29 scripts/job_stage2_sweep.sh sweep_configs/tdt_task_b.yaml
#
# Each task picks trial id = $SLURM_ARRAY_TASK_ID and runs that one trial.
# Result JSONs land in /scratch/.../results/ tagged with the trial id, so
# aggregate_results.py can group them post-hoc.

#SBATCH --job-name=stage2-sweep
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/%u/ra-tft/logs/%x-%A_%a.out
#SBATCH --error=/scratch/users/%u/ra-tft/logs/%x-%A_%a.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail
SWEEP_CONFIG=${1:?usage: $0 sweep_configs/<name>.yaml}
TRIAL=${SLURM_ARRAY_TASK_ID:-0}

echo "=== Stage-2 sweep on $(hostname) trial=$TRIAL config=$SWEEP_CONFIG ==="
nvidia-smi | head -10 || true

module load python/3.11.6-gcc-13.2.0
source ~/envs/ra-tft/bin/activate

SCRATCH=/scratch/users/$USER/ra-tft
CKPT=$SCRATCH/checkpoints/stage2-sweep-$SLURM_ARRAY_JOB_ID/trial$TRIAL
RESULTS=$SCRATCH/results
mkdir -p "$CKPT" "$RESULTS"
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

trap 'echo "preempted, exiting"; exit 0' USR1

cd ~/ra-tft

# hp_sweep.py resolves the trial config from the YAML and execvp's into
# scripts/run_experiment.py with the corresponding CLI flags.
python -u scripts/hp_sweep.py \
    --sweep-config "$SWEEP_CONFIG" \
    --trial-id "$TRIAL" \
    --results-dir "$RESULTS" \
    -- \
    --data-dir "$SCRATCH/data/processed" \
    --device cuda \
    --checkpoint-dir "$CKPT"
