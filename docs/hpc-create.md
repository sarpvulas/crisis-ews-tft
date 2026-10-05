# Training RA-TFT on KCL CREATE HPC

Setup completed 2026-05-01. This document covers project-specific use of the cluster. For general access and connection instructions, see the King's e-Research CREATE documentation.

## Quick connect

```bash
ssh create
```

Account `kcl`, QoS `normal`, partitions available: `cpu`, `gpu`, `interruptible_cpu`, `interruptible_gpu`.

## Where things should live on the cluster

| Path | Use for | Backed up? |
|---|---|---|
| `/users/<k-number>` (`$HOME`) | Code, configs, small artifacts (~quota-limited) | ✅ yes |
| `/scratch/users/<k-number>` | Datasets, checkpoints, logs, wandb cache | ❌ no |
| `/rds/...` (group dir, if granted) | Final model artifacts, results worth keeping | ✅ yes |

**Rule of thumb:** code + recipes in `$HOME`, everything bulky in `/scratch`, only finished outputs you'd cite in a thesis copy to `/rds`.

## Sync local code to HPC

From this Mac (project root):

```bash
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'wandb' \
  --exclude 'checkpoints' \
  --exclude 'data' \
  --exclude '*.pt' \
  --exclude '*.png' \
  --exclude 'dashboard_experiments.db' \
  ./ create:~/ra-tft/
```

Excludes are deliberate: `data/` and `checkpoints/` belong on `/scratch`, not in the rsync'd code tree. Pull them onto the cluster separately (download from FRED on the cluster, or `scp` what you need).

## Recommended layout on cluster

```
/users/<k-number>/ra-tft/             ← code (git working tree)
/scratch/users/<k-number>/ra-tft/
    data/                            ← raw + processed datasets (rebuild from FRED if lost)
    checkpoints/<run-name>/          ← per-run model snapshots
    logs/                            ← Slurm stdout/stderr
    wandb/                           ← wandb local cache
```

## First-time setup on the cluster

```bash
# After sshing in
mkdir -p /scratch/users/$USER/ra-tft/{data,checkpoints,logs,wandb}
ln -s /scratch/users/$USER/ra-tft/data       ~/ra-tft/data
ln -s /scratch/users/$USER/ra-tft/checkpoints ~/ra-tft/checkpoints
ln -s /scratch/users/$USER/ra-tft/wandb       ~/ra-tft/wandb

# Python env (pick one strategy)
module avail 2>&1 | grep -iE 'python|conda|cuda'  # see what's offered
# Option A: module-loaded python + venv
module load python/3.11
python -m venv ~/envs/ra-tft && source ~/envs/ra-tft/bin/activate
pip install -r ~/ra-tft/requirements/cluster.txt

# Option B: miniconda in $HOME
# (only if module python doesn't have what you need)
```

A separate `requirements/cluster.txt` is worth maintaining — local dev on M3 Max uses MPS torch wheels, the cluster needs CUDA wheels. Don't reuse the M3 lockfile.

## Submitting training jobs

Use `interruptible_gpu` for long runs (infinite walltime, but preemptible — checkpoint!). Use `gpu` for short interactive smoke tests (2-day cap, no preemption).

Template `~/ra-tft/scripts/job_train.sh`:

```bash
#!/bin/bash -l
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
module load python/3.11
source ~/envs/ra-tft/bin/activate

CKPT_DIR=/scratch/users/$USER/ra-tft/checkpoints/$SLURM_JOB_NAME
mkdir -p "$CKPT_DIR"

trap 'echo "preempted, exiting cleanly"; exit 0' USR1

cd ~/ra-tft
python -m training.train \
    --ckpt-dir "$CKPT_DIR" \
    --resume-latest \
    "$@"
```

Submit: `sbatch ~/ra-tft/scripts/job_train.sh`

## Hardware caveat — local M3 Max → CUDA cluster

The local repo uses MPS (`torch.backends.mps`). On the cluster you'll be on CUDA (A100 or A40). Things to verify before kicking off long runs:

- `device = "cuda" if torch.cuda.is_available() else "cpu"` (no MPS fallback path)
- Custom Triton/MPS kernels (if any in the regime module) need CUDA equivalents
- `bfloat16` works on A100 but not on older A40 — use `--constraint=a100` if you rely on it
- Random seed parity: MPS and CUDA RNGs differ — don't expect bit-exact reproduction of local numbers

## Useful while training

```bash
squeue -u $USER                              # your jobs
scontrol show job <jobid>                    # detailed status, time remaining
sacct -u $USER --starttime=now-1day -X       # historical (today's) jobs
seff <jobid>                                 # post-run efficiency report (CPU/mem usage)
sstat -j <jobid>.batch --format=AveCPU,MaxRSS,AveDiskRead  # live resource stats

# Tail latest run log
tail -f /scratch/users/$USER/ra-tft/logs/$(ls -t /scratch/users/$USER/ra-tft/logs | head -1)
```

## Pulling results back

```bash
# From Mac, after training
rsync -avz create:/scratch/users/<k-number>/ra-tft/checkpoints/<run-name>/ \
    ./checkpoints/<run-name>/
```

Final model worth keeping → also copy to `/rds` on the cluster (if granted access) so it survives `/scratch` housekeeping.
