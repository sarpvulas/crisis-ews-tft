#!/usr/bin/env python
"""Hyperparameter sweep driver — offline-wandb friendly.

Why a custom driver instead of `wandb sweep`
--------------------------------------------
Wandb sweeps need a live wandb server to dispatch trials, which doesn't play
well with cluster login nodes that have flaky outbound HTTP. This driver does
the same job using just a YAML and Slurm's `--array` argument:

  sweep_config.yaml -> { model: 'tdt', task: 'B', method: 'grid|random',
                          n_trials: 30,
                          params: { lr: [...], d_model: [...], ... } }

  sbatch --array=0-29 scripts/job_array.sh \
      --sweep-config sweep_configs/tdt_task_b.yaml

Each array task computes its config by `sample_config(i, sweep_yaml)`, calls
`scripts/run_experiment.py` with those overrides, and lands a JSON in
`/scratch/.../results/`. Aggregation is a separate post-hoc script.

Methods
-------
  grid   : enumerate the cartesian product. n_trials is ignored; array size
           must equal product cardinality.
  random : draw `n_trials` independent samples. Reproducible via seed = trial_id.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from typing import Any, Dict, List

import yaml


def _materialise_grid(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Enumerate the cartesian product of parameter values."""
    keys = sorted(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _sample_random(params: Dict[str, Any], n_trials: int, seed: int) -> List[Dict[str, Any]]:
    """Independently sample n_trials configurations.

    Each parameter spec can be:
      [v1, v2, ...]          - choose uniformly
      {"min": a, "max": b}   - uniform float
      {"log_min": a, "log_max": b} - log-uniform float (good for lr)
      {"values": [...]}      - explicit categorical (same as a list)
    """
    rng = random.Random(seed)
    out = []
    for trial in range(n_trials):
        cfg = {}
        for key, spec in params.items():
            if isinstance(spec, list):
                cfg[key] = rng.choice(spec)
            elif isinstance(spec, dict) and "min" in spec:
                cfg[key] = rng.uniform(spec["min"], spec["max"])
            elif isinstance(spec, dict) and "log_min" in spec:
                import math
                lo, hi = math.log(spec["log_min"]), math.log(spec["log_max"])
                cfg[key] = math.exp(rng.uniform(lo, hi))
            elif isinstance(spec, dict) and "values" in spec:
                cfg[key] = rng.choice(spec["values"])
            else:
                cfg[key] = spec
        out.append(cfg)
    return out


def sample_config(trial_id: int, sweep_yaml_path: str) -> Dict[str, Any]:
    """Return the config dict for a given trial (zero-indexed)."""
    with open(sweep_yaml_path) as f:
        spec = yaml.safe_load(f)
    method = spec.get("method", "random")
    params = spec.get("params", {})
    if method == "grid":
        configs = _materialise_grid(params)
        if trial_id >= len(configs):
            raise IndexError(
                f"trial_id={trial_id} >= grid size {len(configs)}; "
                f"reduce --array range or use method=random"
            )
        return configs[trial_id]
    elif method == "random":
        n = int(spec.get("n_trials", 30))
        seed = int(spec.get("seed", 42))
        configs = _sample_random(params, n_trials=max(trial_id + 1, n), seed=seed)
        return configs[trial_id]
    else:
        raise ValueError(f"unknown method '{method}', expected 'grid' or 'random'")


def _cfg_to_cli_args(cfg: Dict[str, Any], model: str, task: str, extra: List[str]) -> List[str]:
    """Translate config keys to run_experiment.py CLI flags."""
    args = ["--model", model, "--task", task]
    # Mapping from sweep-config keys to CLI flag names.
    KEY_TO_FLAG = {
        "lr": "--lr",
        "batch_size": "--batch-size",
        "state_size": "--state-size",
        "hidden_size": "--hidden-size",
        "d_model": "--state-size",        # for TDT/iTransformer
        "attention_heads": "--attention-heads",
        "lstm_layers": "--lstm-layers",
        "e_layers": "--lstm-layers",      # for TDT/iTransformer
        "dropout": "--dropout",
        "gamma": "--gamma",
        "pos_weight": "--pos-weight",
        "encoder_steps": "--encoder-steps",
        "decoder_steps": "--decoder-steps",
    }
    for k, v in cfg.items():
        flag = KEY_TO_FLAG.get(k)
        if flag is None:
            # Pass-through as --<k> in case run_experiment grows new flags later.
            flag = "--" + k.replace("_", "-")
        args.extend([flag, str(v)])
    args.extend(extra)
    return args


def main():
    parser = argparse.ArgumentParser(description="Slurm-array friendly HP sweep driver")
    parser.add_argument("--sweep-config", required=True, help="Path to YAML sweep spec.")
    parser.add_argument("--trial-id", type=int, required=True,
                        help="Zero-indexed trial number (typically $SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved CLI args and config, don't run.")
    parser.add_argument("--results-dir", default=None,
                        help="Override results dir for this trial.")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="Extra args passed verbatim to run_experiment.py "
                             "(e.g. --data-dir, --epochs, --seed).")
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        spec = yaml.safe_load(f)
    model = spec["model"]
    task = spec.get("task", "B")
    base_extra = spec.get("base_args", [])

    cfg = sample_config(args.trial_id, args.sweep_config)
    # argparse's REMAINDER often includes the literal '--' separator; drop it.
    rest = list(args.rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    cfg_extra = list(base_extra) + rest
    if args.results_dir:
        cfg_extra += ["--results-dir", args.results_dir]
    cfg_extra += ["--tag", f"sweep-trial{args.trial_id}"]

    cmd = [sys.executable, "scripts/run_experiment.py"] + _cfg_to_cli_args(cfg, model, task, cfg_extra)

    if args.dry_run:
        print(json.dumps({"trial_id": args.trial_id, "cfg": cfg, "cmd": cmd}, indent=2, default=str))
        return

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
