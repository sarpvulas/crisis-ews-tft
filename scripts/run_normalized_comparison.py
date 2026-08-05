#!/usr/bin/env python
"""Train + evaluate all models with feature normalization. Unbuffered output."""
import sys, os, json, time, warnings
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from training.dataset import CrisisDataset
from training.trainer import Trainer
from models.configs import TFTConfig
from models.tft import TemporalFusionTransformer
from models.ra_tft import RegimeAwareTFT
from models.baselines.lstm_baseline import LSTMBaseline
from models.crisis_loss import CrisisAwareLoss

ENCODER, DECODER = 252, 63
DATA_DIR = 'data/processed'
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

# Load datasets with normalization
print('Loading data with feature normalization...', flush=True)
ds_train = CrisisDataset(split='train', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER)
scaler = ds_train.feature_scaler
ds_val = CrisisDataset(split='val', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
ds_test = CrisisDataset(split='test', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
print(f'Train: {len(ds_train)} windows, Val: {len(ds_val)}, Test: {len(ds_test)}', flush=True)
print(f'Features: {ds_train._num_historical} (normalized)', flush=True)

# Extract crash labels from test set
crash_labels = []
for i in range(len(ds_test)):
    _, targets = ds_test[i]
    max_dd = targets['forward_drawdown'].abs().max().item()
    crash_labels.append(1 if max_dd >= 0.10 else 0)
y_true = np.array(crash_labels)
print(f'Test crash: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)', flush=True)


def get_crash_scores(model, ds, device):
    model.to(device); model.eval()
    loader = DataLoader(ds, batch_size=128, collate_fn=ds.collate_fn)
    scores = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            pred = out['predicted'].cpu()
            tail = pred[:, :, 0] if pred.dim() == 3 and pred.shape[-1] >= 7 else pred.squeeze(-1)
            scores.extend(torch.abs(tail).max(dim=1).values.numpy().tolist())
    scores = np.array(scores)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores


def evaluate(y_true, scores):
    if len(np.unique(y_true)) < 2: return {}
    m = {}
    m['PR-AUC'] = float(average_precision_score(y_true, scores))
    m['ROC-AUC'] = float(roc_auc_score(y_true, scores))
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    m['F1'] = float(best_f1)
    m['Precision'] = float(precision_score(y_true, (scores >= best_t).astype(int), zero_division=0))
    m['Recall'] = float(recall_score(y_true, (scores >= best_t).astype(int), zero_division=0))
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        if mask.sum() > 0:
            ece += mask.sum() / len(y_true) * abs(y_true[mask].mean() - scores[mask].mean())
    m['ECE'] = float(ece)
    return m


all_results = defaultdict(list)
N_HIST = ds_train._num_historical

for model_name in ['lstm', 'tft', 'ra-tft']:
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)

        # Build model
        cfg = TFTConfig(
            num_historical_numeric=N_HIST, num_static_numeric=ds_train._num_static,
            num_future_numeric=ds_train._num_future, encoder_steps=ENCODER, decoder_steps=DECODER,
            task_type='regression', num_outputs=7, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
            num_regime_states=3,
            use_regime_module=(model_name == 'ra-tft'),
            use_regime_attention=(model_name == 'ra-tft'),
        )

        if model_name == 'lstm':
            model = LSTMBaseline(input_dim=N_HIST, hidden_dim=64, num_layers=2, task='B',
                                  encoder_steps=ENCODER, decoder_steps=DECODER, num_quantiles=7)
        elif model_name == 'ra-tft':
            model = RegimeAwareTFT(cfg)
        else:
            model = TemporalFusionTransformer(cfg)

        ckpt_dir = f'checkpoints/{model_name}_seed{seed}'

        # Train
        loss_fn = CrisisAwareLoss()
        trainer = Trainer(
            model=model,
            config={'lr': LR, 'batch_size': BATCH_SIZE, 'grad_clip': 1.0,
                    'task': 'B', 'device': device, 'checkpoint_dir': ckpt_dir},
            train_dataset=ds_train, val_dataset=ds_val, loss_fn=loss_fn,
        )
        trainer.train(epochs=EPOCHS)

        # Evaluate
        scores = get_crash_scores(model, ds_test, device)
        met = evaluate(y_true[:len(scores)], scores)
        all_results[model_name].append(met)
        elapsed = time.time() - t0
        print(f'{model_name} seed{seed}: PR-AUC={met.get("PR-AUC",0):.3f} ROC={met.get("ROC-AUC",0):.3f} '
              f'F1={met.get("F1",0):.3f} ECE={met.get("ECE",0):.3f} ({elapsed:.0f}s)', flush=True)

# Summary
print(flush=True)
print('=' * 80, flush=True)
print(f'{"Model":<12} {"PR-AUC":>12} {"ROC-AUC":>12} {"F1":>10} {"ECE":>10}', flush=True)
print('-' * 80, flush=True)

summary = {}
for model_name in ['lstm', 'tft', 'ra-tft']:
    results = all_results[model_name]
    if not results: continue
    avg, std = {}, {}
    for key in results[0]:
        vals = [r[key] for r in results]
        avg[key] = float(np.mean(vals)); std[key] = float(np.std(vals))
    print(f'{model_name:<12} {avg["PR-AUC"]:.3f}±{std["PR-AUC"]:.3f}  {avg["ROC-AUC"]:.3f}±{std["ROC-AUC"]:.3f}  '
          f'{avg["F1"]:.3f}±{std["F1"]:.3f}  {avg["ECE"]:.3f}±{std["ECE"]:.3f}', flush=True)
    summary[model_name] = {'mean': avg, 'std': std, 'per_seed': results}

print('=' * 80, flush=True)

os.makedirs('output', exist_ok=True)
with open('output/comparison_results_normalized.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('Results saved to output/comparison_results_normalized.json', flush=True)
