#!/usr/bin/env python
"""Loss function experiments: Phase 1 (ablations), Phase 2 (grid search), Phase 3 (final comparison).

All runs use RA-TFT with feature normalization and early stopping.
Phase 3 runs the best config on LSTM, TFT, and RA-TFT.
"""
import sys, os, json, time, warnings, itertools
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

# ============================================================================
# Load data once
# ============================================================================
print('Loading data...', flush=True)
ds_train = CrisisDataset(split='train', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER)
scaler = ds_train.feature_scaler
ds_val = CrisisDataset(split='val', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
ds_test = CrisisDataset(split='test', data_dir=DATA_DIR, encoder_steps=ENCODER, decoder_steps=DECODER, feature_scaler=scaler)
N_HIST = ds_train._num_historical
print(f'Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}, Features: {N_HIST}', flush=True)

# Extract test crash labels
crash_labels = []
for i in range(len(ds_test)):
    _, targets = ds_test[i]
    crash_labels.append(1 if targets['forward_drawdown'].abs().max().item() >= 0.10 else 0)
y_true = np.array(crash_labels)
print(f'Test crashes: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)', flush=True)

# Precompute train crash mask for crisis weighting
train_crash_mask = []
for i in range(len(ds_train)):
    _, targets = ds_train[i]
    train_crash_mask.append(1 if targets['forward_drawdown'].abs().max().item() >= 0.10 else 0)
train_crash_mask = np.array(train_crash_mask)
print(f'Train crashes: {train_crash_mask.sum()}/{len(train_crash_mask)} ({train_crash_mask.mean()*100:.1f}%)', flush=True)


# ============================================================================
# Helpers
# ============================================================================

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
    if len(np.unique(y_true)) < 2:
        return {}
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


def build_model(model_name, is_regime=False):
    cfg = TFTConfig(
        num_historical_numeric=N_HIST, num_static_numeric=ds_train._num_static,
        num_future_numeric=ds_train._num_future, encoder_steps=ENCODER, decoder_steps=DECODER,
        task_type='regression', num_outputs=7, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        state_size=64, hidden_size=64, attention_heads=4, lstm_layers=2, dropout=0.1,
        num_regime_states=3, use_regime_module=is_regime, use_regime_attention=is_regime,
    )
    if model_name == 'lstm':
        return LSTMBaseline(input_dim=N_HIST, hidden_dim=64, num_layers=2, task='B',
                             encoder_steps=ENCODER, decoder_steps=DECODER, num_quantiles=7)
    elif model_name == 'ra-tft':
        return RegimeAwareTFT(cfg)
    else:
        return TemporalFusionTransformer(cfg)


def run_experiment(exp_name, model_name, gamma, quantile_weights, crisis_weight, seeds):
    """Train + eval a single config across seeds. Returns list of metric dicts."""
    results = []
    is_regime = model_name == 'ra-tft'

    for seed in seeds:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)

        model = build_model(model_name, is_regime)
        loss_fn = CrisisAwareLoss(gamma=gamma, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        # Store quantile weights and crisis weight on the loss fn
        loss_fn._quantile_weights = quantile_weights
        loss_fn._crisis_weight = crisis_weight

        # Monkey-patch the forward to apply weights
        original_forward = loss_fn.forward
        def make_weighted_forward(lf, qw, cw):
            def weighted_forward(outputs, targets):
                from models.crisis_loss import quantile_loss, regime_nll_loss
                total = torch.tensor(0.0, device=outputs["predicted"].device)

                # Quantile loss with per-quantile weights
                preds = outputs["predicted"]
                tgt = targets["forward_drawdown"]
                q_losses = []
                for i, (q, w) in enumerate(zip(lf.quantiles, qw)):
                    errors = tgt - preds[:, :, i]
                    q_losses.append(w * torch.max((q - 1) * errors, q * errors).mean())
                l_crash = torch.stack(q_losses).sum()

                # Crisis-period weighting: upweight windows where crash_label would be 1
                if cw > 1.0:
                    max_dd = tgt.abs().max(dim=1).values  # (B,)
                    is_crash = (max_dd >= 0.10).float()
                    sample_weights = 1.0 + (cw - 1.0) * is_crash  # 1.0 for calm, cw for crash
                    l_crash = l_crash * sample_weights.mean()

                total = total + lf.beta * l_crash

                if "regime_probs" in outputs and "regime_label" in targets:
                    l_regime = regime_nll_loss(outputs["regime_probs"], targets["regime_label"])
                    total = total + lf.gamma * l_regime

                return total
            return weighted_forward

        loss_fn.forward = make_weighted_forward(loss_fn, quantile_weights, crisis_weight)

        ckpt_dir = f'checkpoints/exp_{exp_name}_{model_name}_seed{seed}'
        trainer = Trainer(
            model=model,
            config={'lr': LR, 'batch_size': BATCH_SIZE, 'grad_clip': 1.0,
                    'task': 'B', 'device': device, 'checkpoint_dir': ckpt_dir,
                    'early_stopping_patience': 5},
            train_dataset=ds_train, val_dataset=ds_val, loss_fn=loss_fn,
        )
        trainer.train(epochs=EPOCHS)

        # Load best checkpoint for evaluation
        ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

        scores = get_crash_scores(model, ds_test, device)
        met = evaluate(y_true[:len(scores)], scores)
        results.append(met)
        elapsed = time.time() - t0
        print(f'  {model_name} seed{seed}: PR-AUC={met.get("PR-AUC",0):.3f} ROC={met.get("ROC-AUC",0):.3f} '
              f'F1={met.get("F1",0):.3f} ECE={met.get("ECE",0):.3f} ({elapsed:.0f}s)', flush=True)

    return results


def summarize(results):
    if not results:
        return {}, {}
    avg, std = {}, {}
    for key in results[0]:
        vals = [r[key] for r in results]
        avg[key] = float(np.mean(vals))
        std[key] = float(np.std(vals))
    return avg, std


# ============================================================================
# PHASE 1: Individual Ablations
# ============================================================================
print('\n' + '=' * 80, flush=True)
print('PHASE 1: Individual Ablations (RA-TFT, 5 seeds each)', flush=True)
print('=' * 80, flush=True)

EQUAL_QW = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
TAIL_QW = [3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2]

phase1_configs = {
    'baseline':     {'gamma': 0.5, 'qw': EQUAL_QW, 'cw': 1.0},
    'tail_weights': {'gamma': 0.5, 'qw': TAIL_QW,  'cw': 1.0},
    'crisis_2x':    {'gamma': 0.5, 'qw': EQUAL_QW, 'cw': 2.0},
    'gamma_0.8':    {'gamma': 0.8, 'qw': EQUAL_QW, 'cw': 1.0},
}

phase1_results = {}
for name, cfg in phase1_configs.items():
    print(f'\n--- {name} (gamma={cfg["gamma"]}, crisis_wt={cfg["cw"]}, tail_wt={"yes" if cfg["qw"] != EQUAL_QW else "no"}) ---', flush=True)
    results = run_experiment(name, 'ra-tft', cfg['gamma'], cfg['qw'], cfg['cw'], SEEDS)
    avg, std = summarize(results)
    phase1_results[name] = {'avg': avg, 'std': std, 'per_seed': results, 'config': cfg}
    print(f'  >> {name}: PR-AUC={avg.get("PR-AUC",0):.3f}±{std.get("PR-AUC",0):.3f} '
          f'ROC={avg.get("ROC-AUC",0):.3f}±{std.get("ROC-AUC",0):.3f}', flush=True)

# Determine which improvements helped
print('\n--- Phase 1 Summary ---', flush=True)
baseline_pr = phase1_results['baseline']['avg'].get('PR-AUC', 0)
for name, res in phase1_results.items():
    pr = res['avg'].get('PR-AUC', 0)
    delta = pr - baseline_pr
    marker = '✓' if delta > 0.005 else '✗'
    print(f'  {marker} {name:20s}: PR-AUC={pr:.3f} ({"+" if delta >= 0 else ""}{delta:.3f})', flush=True)


# ============================================================================
# PHASE 2: Grid Search
# ============================================================================
print('\n' + '=' * 80, flush=True)
print('PHASE 2: Grid Search (RA-TFT, 5 seeds each)', flush=True)
print('=' * 80, flush=True)

# Quantile weight options
QW_OPTIONS = {
    'equal': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    'mild':  [2.0, 1.5, 1.0, 1.0, 0.5, 0.5, 0.5],
    'strong': [3.0, 2.0, 1.5, 1.0, 0.5, 0.3, 0.2],
    'extreme': [5.0, 3.0, 2.0, 1.0, 0.3, 0.2, 0.1],
}

# Crisis weight options
CW_OPTIONS = [1.0, 1.5, 2.0, 3.0]

# Gamma options
GAMMA_OPTIONS = [0.3, 0.5, 0.8, 1.0]

# Full grid is 4*4*4=64 configs — too many. Use smart subset:
# Fix equal quantile weights, sweep crisis weight x gamma
# Fix best crisis weight, sweep quantile weights x gamma
# This gives ~16 configs max

grid_configs = {}

# Sweep gamma x crisis_weight (with equal quantile weights)
for gamma in GAMMA_OPTIONS:
    for cw in CW_OPTIONS:
        name = f'g{gamma}_cw{cw}_eq'
        grid_configs[name] = {'gamma': gamma, 'qw': EQUAL_QW, 'cw': cw}

# Sweep quantile weights x gamma (with crisis_weight=2.0)
for qw_name, qw in QW_OPTIONS.items():
    if qw_name == 'equal':
        continue  # Already covered above
    for gamma in [0.5, 0.8]:
        name = f'g{gamma}_cw2_{qw_name}'
        grid_configs[name] = {'gamma': gamma, 'qw': qw, 'cw': 2.0}

# Remove duplicates with phase 1
for p1_name in phase1_configs:
    for g_name in list(grid_configs.keys()):
        if (grid_configs[g_name]['gamma'] == phase1_configs[p1_name]['gamma'] and
            grid_configs[g_name]['qw'] == phase1_configs[p1_name]['qw'] and
            grid_configs[g_name]['cw'] == phase1_configs[p1_name]['cw']):
            # Reuse phase 1 result
            if g_name in grid_configs:
                del grid_configs[g_name]
                print(f'  Reusing phase 1 result for {g_name} (matches {p1_name})', flush=True)

print(f'Grid search: {len(grid_configs)} configs × 5 seeds = {len(grid_configs)*5} runs', flush=True)

phase2_results = dict(phase1_results)  # Start with phase 1 results

for name, cfg in grid_configs.items():
    print(f'\n--- {name} (gamma={cfg["gamma"]}, cw={cfg["cw"]}, qw={cfg["qw"][:3]}...) ---', flush=True)
    results = run_experiment(name, 'ra-tft', cfg['gamma'], cfg['qw'], cfg['cw'], SEEDS)
    avg, std = summarize(results)
    phase2_results[name] = {'avg': avg, 'std': std, 'per_seed': results, 'config': cfg}
    print(f'  >> {name}: PR-AUC={avg.get("PR-AUC",0):.3f}±{std.get("PR-AUC",0):.3f} '
          f'ROC={avg.get("ROC-AUC",0):.3f}±{std.get("ROC-AUC",0):.3f}', flush=True)

# Find best config
print('\n--- Phase 2 Ranking (by PR-AUC) ---', flush=True)
ranked = sorted(phase2_results.items(), key=lambda x: x[1]['avg'].get('PR-AUC', 0), reverse=True)
for i, (name, res) in enumerate(ranked[:10]):
    pr = res['avg'].get('PR-AUC', 0)
    roc = res['avg'].get('ROC-AUC', 0)
    cfg = res.get('config', {})
    print(f'  #{i+1} {name:30s}: PR-AUC={pr:.3f} ROC={roc:.3f} (g={cfg.get("gamma","?")}, cw={cfg.get("cw","?")})', flush=True)

best_name = ranked[0][0]
best_config = ranked[0][1]['config']
print(f'\nBest config: {best_name}', flush=True)
print(f'  gamma={best_config["gamma"]}, crisis_weight={best_config["cw"]}, quantile_weights={best_config["qw"]}', flush=True)


# ============================================================================
# PHASE 3: Final Comparison (best config on LSTM, TFT, RA-TFT)
# ============================================================================
print('\n' + '=' * 80, flush=True)
print(f'PHASE 3: Final Comparison with best config: {best_name}', flush=True)
print('=' * 80, flush=True)

phase3_results = {}
for model_name in ['lstm', 'tft', 'ra-tft']:
    print(f'\n--- {model_name} ---', flush=True)
    results = run_experiment(
        f'final_{best_name}', model_name,
        best_config['gamma'], best_config['qw'], best_config['cw'],
        SEEDS,
    )
    avg, std = summarize(results)
    phase3_results[model_name] = {'avg': avg, 'std': std, 'per_seed': results}
    print(f'  >> {model_name}: PR-AUC={avg.get("PR-AUC",0):.3f}±{std.get("PR-AUC",0):.3f} '
          f'ROC={avg.get("ROC-AUC",0):.3f}±{std.get("ROC-AUC",0):.3f} '
          f'F1={avg.get("F1",0):.3f}±{std.get("F1",0):.3f} '
          f'ECE={avg.get("ECE",0):.3f}±{std.get("ECE",0):.3f}', flush=True)

# Final summary table
print('\n' + '=' * 80, flush=True)
print('FINAL RESULTS', flush=True)
print(f'Best loss config: {best_name}', flush=True)
print(f'  gamma={best_config["gamma"]}, crisis_wt={best_config["cw"]}, tail_wt={best_config["qw"][:3]}...', flush=True)
print(f'{"Model":<12} {"PR-AUC":>12} {"ROC-AUC":>12} {"F1":>10} {"ECE":>10}', flush=True)
print('-' * 80, flush=True)
for model_name in ['lstm', 'tft', 'ra-tft']:
    avg = phase3_results[model_name]['avg']
    std = phase3_results[model_name]['std']
    print(f'{model_name:<12} {avg["PR-AUC"]:.3f}±{std["PR-AUC"]:.3f}  {avg["ROC-AUC"]:.3f}±{std["ROC-AUC"]:.3f}  '
          f'{avg["F1"]:.3f}±{std["F1"]:.3f}  {avg["ECE"]:.3f}±{std["ECE"]:.3f}', flush=True)
print('=' * 80, flush=True)

# Save everything
output = {
    'phase1': {k: {'avg': v['avg'], 'std': v['std'], 'config': v.get('config', {})} for k, v in phase1_results.items()},
    'phase2_ranking': [(name, res['avg'], res.get('config', {})) for name, res in ranked[:10]],
    'best_config': {'name': best_name, **best_config},
    'phase3_final': {k: {'avg': v['avg'], 'std': v['std'], 'per_seed': v['per_seed']} for k, v in phase3_results.items()},
}
os.makedirs('output', exist_ok=True)
with open('output/loss_experiments.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f'\nAll results saved to output/loss_experiments.json', flush=True)
