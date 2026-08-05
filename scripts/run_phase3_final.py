#!/usr/bin/env python
"""Phase 3: Final comparison — best loss config on LSTM, TFT, RA-TFT (5 seeds each)."""
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
from models.crisis_loss import CrisisAwareLoss, quantile_loss, regime_nll_loss

ENCODER, DECODER = 252, 63
DATA_DIR = 'data/processed'
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3

# Best config from grid search
GAMMA = 0.8
CRISIS_WEIGHT = 1.5
QUANTILE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}', flush=True)
print(f'Best config: gamma={GAMMA}, crisis_weight={CRISIS_WEIGHT}', flush=True)

# Load data
print('Loading data...', flush=True)
ds_train = CrisisDataset(split='train', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER)
scaler = ds_train.feature_scaler
ds_val = CrisisDataset(split='val', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
ds_test = CrisisDataset(split='test', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
N_HIST = ds_train._num_historical
print(f'Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}, Features: {N_HIST}', flush=True)

# Test crash labels
crash_labels = []
for i in range(len(ds_test)):
    _, targets = ds_test[i]
    crash_labels.append(1 if targets['forward_drawdown'].abs().max().item() >= 0.10 else 0)
y_true = np.array(crash_labels)
print(f'Test crashes: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)', flush=True)


def make_loss_fn(gamma):
    """Create loss fn with crisis weighting and custom gamma."""
    loss_fn = CrisisAwareLoss(gamma=gamma, quantiles=QUANTILES)

    def weighted_forward(outputs, targets):
        total = torch.tensor(0.0, device=outputs["predicted"].device)
        preds = outputs["predicted"]
        tgt = targets["forward_drawdown"]

        q_losses = []
        for i, (q, w) in enumerate(zip(QUANTILES, QUANTILE_WEIGHTS)):
            errors = tgt - preds[:, :, i]
            q_losses.append(w * torch.max((q - 1) * errors, q * errors).mean())
        l_crash = torch.stack(q_losses).sum()

        if CRISIS_WEIGHT > 1.0:
            max_dd = tgt.abs().max(dim=1).values
            is_crash = (max_dd >= 0.10).float()
            sample_weights = 1.0 + (CRISIS_WEIGHT - 1.0) * is_crash
            l_crash = l_crash * sample_weights.mean()

        total = total + l_crash

        if "regime_probs" in outputs and "regime_label" in targets:
            l_regime = regime_nll_loss(outputs["regime_probs"], targets["regime_label"])
            total = total + gamma * l_regime

        return total

    loss_fn.forward = weighted_forward
    return loss_fn


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


results = {}
for model_name in ['lstm', 'tft', 'ra-tft']:
    print(f'\n{"="*60}', flush=True)
    print(f'MODEL: {model_name}', flush=True)
    print(f'{"="*60}', flush=True)

    is_regime = model_name == 'ra-tft'
    seed_results = []

    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)

        cfg = TFTConfig(
            num_historical_numeric=N_HIST, num_static_numeric=ds_train._num_static,
            num_future_numeric=ds_train._num_future, encoder_steps=ENCODER, decoder_steps=DECODER,
            task_type='regression', num_outputs=7, quantiles=QUANTILES,
            state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
            num_regime_states=3, use_regime_module=is_regime, use_regime_attention=is_regime,
        )

        if model_name == 'lstm':
            model = LSTMBaseline(input_dim=N_HIST, hidden_dim=64, num_layers=2, task='B',
                                  encoder_steps=ENCODER, decoder_steps=DECODER, num_quantiles=7)
        elif model_name == 'ra-tft':
            model = RegimeAwareTFT(cfg)
        else:
            model = TemporalFusionTransformer(cfg)

        # Each model gets its optimal loss config
        if is_regime:
            loss_fn = make_loss_fn(GAMMA)  # gamma=0.8, cw=1.5
        else:
            loss_fn = CrisisAwareLoss(gamma=0.0)  # plain quantile loss, no crisis weighting

        ckpt_dir = f'checkpoints/final_{model_name}_seed{seed}'
        trainer = Trainer(
            model=model,
            config={'lr': LR, 'batch_size': BATCH_SIZE, 'grad_clip': 1.0,
                    'task': 'B', 'device': device, 'checkpoint_dir': ckpt_dir,
                    'early_stopping_patience': 5},
            train_dataset=ds_train, val_dataset=ds_val, loss_fn=loss_fn,
        )
        trainer.train(epochs=EPOCHS)

        # Load best checkpoint
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        scores = get_crash_scores(model, ds_test, device)
        met = evaluate(y_true[:len(scores)], scores)
        seed_results.append(met)
        elapsed = time.time() - t0
        print(f'  seed{seed}: PR-AUC={met.get("PR-AUC",0):.3f} ROC={met.get("ROC-AUC",0):.3f} '
              f'F1={met.get("F1",0):.3f} ECE={met.get("ECE",0):.3f} ({elapsed:.0f}s)', flush=True)

    # Compute mean/std/sem
    avg, std, sem = {}, {}, {}
    for key in seed_results[0]:
        vals = [r[key] for r in seed_results]
        avg[key] = float(np.mean(vals))
        std[key] = float(np.std(vals))
        sem[key] = float(np.std(vals) / np.sqrt(len(vals)))
    results[model_name] = {'avg': avg, 'std': std, 'sem': sem, 'per_seed': seed_results}
    print(f'  >> {model_name}: PR-AUC={avg["PR-AUC"]:.3f}±{sem["PR-AUC"]:.3f}(SEM) '
          f'ROC={avg["ROC-AUC"]:.3f}±{sem["ROC-AUC"]:.3f}(SEM)', flush=True)

# Final table
print(f'\n{"="*80}', flush=True)
print(f'FINAL RESULTS (gamma={GAMMA}, crisis_wt={CRISIS_WEIGHT})', flush=True)
print(f'{"Model":<12} {"PR-AUC":>14} {"ROC-AUC":>14} {"F1":>12} {"ECE":>12}', flush=True)
print('-' * 80, flush=True)
for model_name in ['lstm', 'tft', 'ra-tft']:
    a = results[model_name]['avg']
    s = results[model_name]['sem']
    print(f'{model_name:<12} {a["PR-AUC"]:.3f}±{s["PR-AUC"]:.3f}  {a["ROC-AUC"]:.3f}±{s["ROC-AUC"]:.3f}  '
          f'{a["F1"]:.3f}±{s["F1"]:.3f}  {a["ECE"]:.3f}±{s["ECE"]:.3f}', flush=True)
print('=' * 80, flush=True)

os.makedirs('output', exist_ok=True)
with open('output/final_comparison.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('Results saved to output/final_comparison.json', flush=True)
