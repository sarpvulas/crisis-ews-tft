#!/usr/bin/env python
"""Vanilla TFT validation: same configs as RA-TFT but without regime module.
5 seeds x 100 epochs (patience=10) to compare against RA-TFT results.
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
from models.tft import TemporalFusionTransformer
from models.crisis_loss import CrisisAwareLoss
from training.dataset import PanelCrisisDataset
from training.trainer import Trainer
from training.evaluation import (
    compute_roc_auc, compute_pr_auc, compute_calibration_error,
    compute_brier_score, PostHocCalibrator,
)

PANEL_PATH = "data/processed/panel/panel_features.parquet"
ENC, DEC = 252, 63
TEST_START = "2015-01-01"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ALL_CONFIGS = {
    "V_C_bigger_fastlr":  dict(state_size=128, lr=1e-3, dropout=0.2, attention_heads=4),
    "V_X_combo_best":     dict(state_size=128, lr=1e-3, dropout=0.05, attention_heads=4),
    "V_N_64_fastlr":      dict(state_size=64,  lr=1e-3, dropout=0.2, attention_heads=4),
    "V_I_128_verylowdrop":dict(state_size=128, lr=3e-4, dropout=0.05, attention_heads=4),
    "V_F_full_capacity":  dict(state_size=256, lr=1e-4, dropout=0.3, attention_heads=8),
    "V_M_256_fastlr":     dict(state_size=256, lr=3e-4, dropout=0.2, attention_heads=8),
}
SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 100
PATIENCE = 10


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


def train_one(config_name, config, seed, epochs, ds_train, ds_val, ds_test,
              dims, n_markets, pos_weight, result_dict):
    """Train one Vanilla TFT config with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        use_regime_module=False,
        use_regime_attention=False,
    )
    model = TemporalFusionTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_dir = f"checkpoints/validation_vanilla/{config_name}_seed{seed}"
    loss_fn = CrisisAwareLoss(pos_weight=pos_weight)
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        config={"lr": config["lr"], "batch_size": 64, "grad_clip": 1.0, "task": "B",
                "device": DEVICE, "checkpoint_dir": ckpt_dir,
                "early_stopping_patience": PATIENCE},
        train_dataset=ds_train, val_dataset=ds_val,
    )

    t0 = time.time()
    history = trainer.train(epochs=epochs)
    train_time = time.time() - t0
    best_val = min(history["val_loss"])
    n_ep = len(history["train_loss"])

    # Restore best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # Evaluate
    y_val_s, y_val_t = evaluate_pytorch(model, ds_val, DEVICE)
    y_s, y_t = evaluate_pytorch(model, ds_test, DEVICE)

    metrics = {"seed": seed, "config": config_name, "n_params": n_params,
               "epochs": n_ep, "best_val_loss": best_val, "train_time_min": train_time / 60}

    if len(np.unique(y_t)) >= 2:
        metrics["roc_auc"] = compute_roc_auc(y_t, y_s)
        metrics["pr_auc"] = compute_pr_auc(y_t, y_s)
        metrics["calibration_error"] = compute_calibration_error(y_t, y_s)
        metrics["brier_score"] = compute_brier_score(y_t, y_s)

        cal = PostHocCalibrator(method="platt")
        cal.fit(y_val_s, y_val_t)
        y_cal = cal.transform(y_s)
        metrics["cal_ece"] = compute_calibration_error(y_t, y_cal)
        metrics["cal_brier"] = compute_brier_score(y_t, y_cal)

    key = f"{config_name}_seed{seed}"
    result_dict[key] = metrics
    print(f"  [{config_name} seed={seed}] Done in {train_time/60:.1f}min: "
          f"ROC={metrics.get('roc_auc',0):.4f} PR={metrics.get('pr_auc',0):.4f} "
          f"ECE={metrics.get('calibration_error',0):.4f}")


def main():
    print(f"Device: {DEVICE}")
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

    all_results = {}

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    print(f"\n{'='*80}")
    print(f"VANILLA TFT VALIDATION: {len(ALL_CONFIGS)} configs x {len(SEEDS)} seeds x {EPOCHS} epochs (patience={PATIENCE})")
    print(f"{'='*80}")

    for config_name, config in ALL_CONFIGS.items():
        print(f"\n--- {config_name}: {len(SEEDS)} seeds x {EPOCHS} epochs ---")
        result_dict_seeds = manager.dict()

        seed_processes = []
        for seed in SEEDS:
            p = mp.Process(
                target=train_one,
                args=(config_name, config, seed, EPOCHS, ds_train, ds_val, ds_test,
                      dims, n_markets, pos_weight, result_dict_seeds),
            )
            seed_processes.append(p)
            p.start()
            print(f"    Started seed={seed}")

        for p in seed_processes:
            p.join()

        seed_results = dict(result_dict_seeds)
        all_results[config_name] = seed_results

        rocs = [m.get("roc_auc", 0) for m in seed_results.values()]
        prs = [m.get("pr_auc", 0) for m in seed_results.values()]
        eces = [m.get("calibration_error", 0) for m in seed_results.values()]
        cal_eces = [m.get("cal_ece", 0) for m in seed_results.values()]
        print(f"\n  {config_name} Summary:")
        print(f"    ROC-AUC:  {np.mean(rocs):.4f} +/- {np.std(rocs):.4f}  (seeds: {[f'{r:.4f}' for r in rocs]})")
        print(f"    PR-AUC:   {np.mean(prs):.4f} +/- {np.std(prs):.4f}")
        print(f"    ECE:      {np.mean(eces):.4f} +/- {np.std(eces):.4f}")
        print(f"    Cal ECE:  {np.mean(cal_eces):.4f} +/- {np.std(cal_eces):.4f}")

    # Final summary
    print(f"\n\n{'='*100}")
    print("VANILLA TFT FINAL RESULTS")
    print(f"{'='*100}")
    print(f"{'Config':<24} {'Mean ROC':>10} {'Std ROC':>10} {'Mean PR':>10} {'Mean ECE':>10} {'Mean CalECE':>12}")
    print("-" * 100)

    for config_name in ALL_CONFIGS:
        if config_name not in all_results:
            continue
        seed_results = all_results[config_name]
        rocs = [m.get("roc_auc", 0) for m in seed_results.values()]
        prs = [m.get("pr_auc", 0) for m in seed_results.values()]
        eces = [m.get("calibration_error", 0) for m in seed_results.values()]
        cal_eces = [m.get("cal_ece", 0) for m in seed_results.values()]
        print(f"{config_name:<24} {np.mean(rocs):>10.4f} {np.std(rocs):>10.4f} "
              f"{np.mean(prs):>10.4f} {np.mean(eces):>10.4f} {np.mean(cal_eces):>12.4f}")

    print(f"\nXGBoost reference:        0.6213    0.0000     0.1966     0.1475")
    print("=" * 100)

    os.makedirs("output", exist_ok=True)
    with open("output/validation_vanilla_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to output/validation_vanilla_results.json")


if __name__ == "__main__":
    main()
