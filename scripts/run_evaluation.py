#!/usr/bin/env python
"""Evaluate all trained models on daily test data. Unbuffered output."""
import sys, os, json, time, warnings
sys.stdout.reconfigure(line_buffering=True)  # Force line-buffered output
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from training.dataset import CrisisDataset
from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.baselines.lstm_baseline import LSTMBaseline

ENCODER, DECODER = 252, 63
DATA_DIR = 'data/processed'
SEEDS = [0, 1, 2, 3, 4]

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

# Load data — fit scaler on train, reuse for test
print('Loading data...', flush=True)
ds_train_for_scaler = CrisisDataset(split='train', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER)
ds_test = CrisisDataset(split='test', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER,
                         feature_scaler=ds_train_for_scaler.feature_scaler)
print(f'Test windows: {len(ds_test)}, Features: {ds_test._num_historical}', flush=True)

# Extract crash labels efficiently (per-window max drawdown)
crash_labels = []
for i in range(len(ds_test)):
    _, targets = ds_test[i]
    max_dd = targets['forward_drawdown'].abs().max().item()
    crash_labels.append(1 if max_dd >= 0.10 else 0)
y_true = np.array(crash_labels)
print(f'Crash labels: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)', flush=True)


def get_crash_scores(model, ds, device, batch_size=128):
    model.to(device)
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=ds.collate_fn)
    scores = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            pred = out['predicted'].cpu()
            if pred.dim() == 3 and pred.shape[-1] >= 7:
                tail = pred[:, :, 0]  # 5th percentile
            else:
                tail = pred.squeeze(-1)
            score = torch.abs(tail).max(dim=1).values.numpy()
            scores.extend(score.tolist())
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def evaluate(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return {}
    metrics = {}
    metrics['PR-AUC'] = float(average_precision_score(y_true, scores))
    metrics['ROC-AUC'] = float(roc_auc_score(y_true, scores))
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    metrics['F1'] = float(best_f1)
    metrics['Precision'] = float(precision_score(y_true, (scores >= best_t).astype(int), zero_division=0))
    metrics['Recall'] = float(recall_score(y_true, (scores >= best_t).astype(int), zero_division=0))
    # ECE
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - scores[mask].mean())
    metrics['ECE'] = float(ece)
    return metrics


all_results = defaultdict(list)

# Neural models only — baselines don't save checkpoints
for model_name in ['lstm', 'tft', 'ra-tft']:
    for seed in SEEDS:
        t0 = time.time()
        ckpt_path = f'checkpoints/{model_name}_seed{seed}/best_model.pt'
        if not os.path.exists(ckpt_path):
            print(f'{model_name} seed{seed}: MISSING', flush=True)
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        cfg = TFTConfig(
            num_historical_numeric=ds_test._num_historical,
            num_static_numeric=ds_test._num_static,
            num_future_numeric=ds_test._num_future,
            encoder_steps=ENCODER, decoder_steps=DECODER,
            task_type='regression', num_outputs=7,
            quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
            num_regime_states=3,
            use_regime_module=(model_name == 'ra-tft'),
            use_regime_attention=(model_name == 'ra-tft'),
        )

        if model_name == 'lstm':
            model = LSTMBaseline(
                input_dim=ds_test._num_historical, hidden_dim=64, num_layers=2,
                task='B', encoder_steps=ENCODER, decoder_steps=DECODER, num_quantiles=7,
            )
        elif model_name == 'ra-tft':
            model = RegimeAwareTFT(cfg)
        else:
            model = TemporalFusionTransformer(cfg)

        model.load_state_dict(ckpt['model_state_dict'])
        scores = get_crash_scores(model, ds_test, device)
        m = evaluate(y_true[:len(scores)], scores)
        all_results[model_name].append(m)
        elapsed = time.time() - t0
        print(f'{model_name} seed{seed}: PR-AUC={m.get("PR-AUC",0):.3f} ROC={m.get("ROC-AUC",0):.3f} F1={m.get("F1",0):.3f} ECE={m.get("ECE",0):.3f} ({elapsed:.0f}s)', flush=True)

# Summary
print(flush=True)
print('=' * 80, flush=True)
print(f'{"Model":<12} {"PR-AUC":>12} {"ROC-AUC":>12} {"F1":>10} {"ECE":>10}', flush=True)
print('-' * 80, flush=True)

summary = {}
for model_name in ['lstm', 'tft', 'ra-tft']:
    results = all_results[model_name]
    if not results:
        continue
    avg, std = {}, {}
    for key in results[0]:
        vals = [r[key] for r in results if key in r]
        avg[key] = float(np.mean(vals))
        std[key] = float(np.std(vals))
    print(f'{model_name:<12} {avg["PR-AUC"]:.3f}±{std["PR-AUC"]:.3f}  {avg["ROC-AUC"]:.3f}±{std["ROC-AUC"]:.3f}  {avg["F1"]:.3f}±{std["F1"]:.3f}  {avg["ECE"]:.3f}±{std["ECE"]:.3f}', flush=True)
    summary[model_name] = {'mean': avg, 'std': std, 'per_seed': results}

print('=' * 80, flush=True)

os.makedirs('output', exist_ok=True)
with open('output/comparison_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Results saved to output/comparison_results.json', flush=True)
