#!/usr/bin/env python
"""Hyperparameter sweep: 8 RA-TFT configs in parallel on one GPU.

40 epochs per config (enough to rank configs). Top configs get full 100-epoch runs later.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from models.configs import TFTConfig
from models.ra_tft import RegimeAwareTFT
from models.crisis_loss import CrisisAwareLoss
from training.dataset import PanelCrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_roc_auc, compute_pr_auc, compute_calibration_error,
    compute_brier_score,
)

PANEL_PATH = "data/processed/panel/panel_features.parquet"
ENC, DEC = 252, 63
TEST_START = "2015-01-01"
SWEEP_EPOCHS = 40
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 8 configurations ===
CONFIGS = {
    "A_baseline":       dict(state_size=64,  lr=3e-4, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "B_bigger":         dict(state_size=128, lr=3e-4, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "C_bigger_fastlr":  dict(state_size=128, lr=1e-3, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "D_bigger_lowdrop": dict(state_size=128, lr=3e-4, dropout=0.1, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "E_5regimes":       dict(state_size=128, lr=3e-4, dropout=0.2, num_regime_states=5, attention_heads=4, use_regime_attention=True),
    "F_full_capacity":  dict(state_size=256, lr=1e-4, dropout=0.3, num_regime_states=3, attention_heads=8, use_regime_attention=True),
    "G_regime_no_attn": dict(state_size=128, lr=3e-4, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=False),
    "H_bigger_8heads":  dict(state_size=128, lr=3e-4, dropout=0.2, num_regime_states=3, attention_heads=8, use_regime_attention=True),
    "I_128_verylowdrop":dict(state_size=128, lr=3e-4, dropout=0.05,num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "J_128_highdrop":   dict(state_size=128, lr=3e-4, dropout=0.4, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "K_128_slowlr":     dict(state_size=128, lr=1e-4, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "L_128_4reg":       dict(state_size=128, lr=3e-4, dropout=0.2, num_regime_states=4, attention_heads=4, use_regime_attention=True),
    "M_256_fastlr":     dict(state_size=256, lr=3e-4, dropout=0.2, num_regime_states=3, attention_heads=8, use_regime_attention=True),
    "N_64_fastlr":      dict(state_size=64,  lr=1e-3, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
    "O_128_fastlr":     dict(state_size=128, lr=5e-4, dropout=0.2, num_regime_states=3, attention_heads=4, use_regime_attention=True),
}


def load_datasets():
    ds_train, ds_val = PanelCrisisDataset.train_val_split(
        PANEL_PATH, train_end=TEST_START, val_ratio=0.15, seed=42,
        encoder_steps=ENC, decoder_steps=DEC,
    )
    ds_test = PanelCrisisDataset(
        PANEL_PATH, split="test", encoder_steps=ENC, decoder_steps=DEC,
        train_end=TEST_START, val_end=TEST_START,
        feature_scaler=ds_train.feature_scaler,
        exclude_crisis_windows=False,
    )
    return ds_train, ds_val, ds_test


def evaluate_pytorch(model, dataset, device):
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False,
                        collate_fn=dataset.collate_fn, num_workers=0)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, targets in loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch_dev)
            probs = torch.sigmoid(outputs["predicted"].squeeze(-1))
            all_preds.append(probs.cpu().numpy().flatten())
            all_labels.append(targets["ews_label"].numpy().flatten())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_config(config_name, config, ds_train, ds_val, ds_test,
                 dims, n_markets, pos_weight, result_dict):
    """Train one RA-TFT configuration."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    n_hist = dims["num_historical_numeric"]
    n_future = dims["num_future_numeric"]

    cfg = TFTConfig(
        num_historical_numeric=n_hist, num_future_numeric=n_future,
        num_static_categorical=1, static_categorical_cardinalities=[n_markets],
        state_size=config["state_size"],
        hidden_size=config["state_size"],
        attention_heads=config["attention_heads"],
        lstm_layers=2,
        dropout=config["dropout"],
        encoder_steps=ENC, decoder_steps=DEC,
        task_type="ews", num_outputs=1,
        num_regime_states=config["num_regime_states"],
        use_regime_module=True,
        use_regime_attention=config["use_regime_attention"],
    )
    model = RegimeAwareTFT(cfg)

    # Log parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{config_name}] Params: {n_params:,}, state_size={config['state_size']}")

    ckpt_dir = f"checkpoints/sweep/{config_name}"
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": config["lr"], "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": ckpt_dir,
                "early_stopping_patience": 15},
        train_dataset=ds_train, val_dataset=ds_val,
    )

    t0 = time.time()
    history = trainer.train(epochs=SWEEP_EPOCHS)
    train_time = time.time() - t0
    best_val = min(history["val_loss"])
    n_ep = len(history["train_loss"])

    # Restore best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate on test
    y_s, y_t = evaluate_pytorch(model, ds_test, DEVICE)
    metrics = {"config": config}
    if len(np.unique(y_t)) >= 2:
        metrics["roc_auc"] = compute_roc_auc(y_t, y_s)
        metrics["pr_auc"] = compute_pr_auc(y_t, y_s)
        metrics["calibration_error"] = compute_calibration_error(y_t, y_s)
        metrics["brier_score"] = compute_brier_score(y_t, y_s)
    metrics["best_val_loss"] = best_val
    metrics["epochs"] = n_ep
    metrics["train_time_min"] = train_time / 60
    metrics["n_params"] = n_params

    result_dict[config_name] = metrics
    print(f"  [{config_name}] Done in {train_time/60:.1f}min: ROC={metrics.get('roc_auc',0):.4f} "
          f"PR={metrics.get('pr_auc',0):.4f} ECE={metrics.get('calibration_error',0):.4f}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Sweep: {len(CONFIGS)} configs x {SWEEP_EPOCHS} epochs (parallel on 1 GPU)")

    # Check GPU memory
    if DEVICE == "cuda":
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {total_memory:.1f} GB")

    print("\nLoading datasets...")
    ds_train, ds_val, ds_test = load_datasets()
    dims = ds_train.get_feature_dims()
    n_markets = ds_train.num_markets

    total_ews = sum(md["ews"].sum() for md in ds_train.market_data.values())
    total_samples = sum(len(md["ews"]) for md in ds_train.market_data.values())
    pos_rate = total_ews / max(total_samples, 1)
    pos_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 10.0)
    print(f"  Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")
    print(f"  Pos rate: {pos_rate:.3f}, pos_weight: {pos_weight:.1f}")

    # === Launch all configs in parallel ===
    print(f"\n{'='*80}")
    print("LAUNCHING ALL {len(CONFIGS)} CONFIGS IN PARALLEL")
    print(f"{'='*80}\n")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    processes = []
    for config_name, config in CONFIGS.items():
        p = mp.Process(
            target=train_config,
            args=(config_name, config, ds_train, ds_val, ds_test,
                  dims, n_markets, pos_weight, result_dict),
        )
        processes.append((config_name, p))

    t_start = time.time()
    for name, p in processes:
        p.start()
        print(f"  Started: {name}")

    # Wait for all
    for name, p in processes:
        p.join()

    total_time = (time.time() - t_start) / 60
    results = dict(result_dict)

    # === Print GPU utilization ===
    if DEVICE == "cuda":
        print(f"\n{'='*80}")
        print("GPU UTILIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"  Total wall time: {total_time:.1f} min")
        total_model_time = sum(r.get("train_time_min", 0) for r in results.values())
        print(f"  Sum of model train times: {total_model_time:.1f} min")
        print(f"  Parallelization speedup: {total_model_time/total_time:.1f}x")
        used_mem = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Peak GPU memory (main process): {used_mem:.1f} / {total_memory:.1f} GB")

    # === Results table ===
    print(f"\n{'='*100}")
    print(f"HYPERPARAMETER SWEEP RESULTS ({SWEEP_EPOCHS} epochs)")
    print(f"{'='*100}")
    print(f"{'Config':<22} {'ROC-AUC':>10} {'PR-AUC':>10} {'ECE':>10} {'Brier':>10} "
          f"{'Params':>10} {'Time':>8} {'Best Val':>10}")
    print("-" * 100)

    # Sort by ROC-AUC descending
    sorted_configs = sorted(results.items(), key=lambda x: x[1].get("roc_auc", 0), reverse=True)

    for name, m in sorted_configs:
        roc = f"{m.get('roc_auc', 0):.4f}"
        pr = f"{m.get('pr_auc', 0):.4f}"
        ece = f"{m.get('calibration_error', 0):.4f}"
        brier = f"{m.get('brier_score', 0):.4f}"
        params = f"{m.get('n_params', 0):,}"
        t = f"{m.get('train_time_min', 0):.1f}m"
        bv = f"{m.get('best_val_loss', 0):.4f}"
        marker = " <-- BEST" if name == sorted_configs[0][0] else ""
        print(f"{name:<22} {roc:>10} {pr:>10} {ece:>10} {brier:>10} {params:>10} {t:>8} {bv:>10}{marker}")

    print("=" * 100)
    print(f"\nTotal wall time: {total_time:.1f} min")

    # Save
    os.makedirs("output", exist_ok=True)
    with open("output/hp_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to output/hp_sweep.json")

    # Recommendation
    best_name, best_m = sorted_configs[0]
    print(f"\n>>> RECOMMENDATION: Run '{best_name}' for full 100 epochs with 3 seeds")


if __name__ == "__main__":
    main()
