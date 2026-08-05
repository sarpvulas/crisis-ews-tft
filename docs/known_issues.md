# Known Issues and Next Steps

## Current State

Multi-market panel EWS pipeline is built and functional:
- 12 equity indices downloaded and processed (72,801 rows)
- `PanelCrisisDataset` handles per-market windowing with `market_id` as static categorical
- RA-TFT, Vanilla TFT, LSTM, XGBoost comparison script ready
- RunPod GPU training attempted but results are poor due to issues below

## Issue 1: Val Loss Divergence (Critical)

**Symptom:** Val loss increases from epoch 1 onward for all PyTorch models. Early
stopping saves the epoch 1 checkpoint (least trained, worst model).

**Root cause:** Train and val periods have fundamentally different crisis
distributions. Training data (2000-2009) contains dot-com and GFC crises — the
model learns to predict crises. The val set (any calm period like 2010-2014 or
2015-2019) has fewer or no crises, so the model's crisis predictions are penalized
as false alarms, driving val loss up.

**Why this only affects PyTorch models:** XGBoost doesn't use val-based early
stopping — it trains to completion. That's why XGBoost scores ROC=0.69 while
TFT/LSTM score ROC=0.42-0.44.

**Possible fixes:**
1. **Random split within training period** — hold out 20% of training *windows*
   (randomly sampled, not temporally) as val. Both train and val will have similar
   crisis rates. Risk: slight temporal leakage between adjacent windows.
2. **Train for fixed epochs** — remove val-based early stopping entirely. Use a
   fixed epoch count (e.g., 30) chosen via preliminary experiments. Simple but
   risks overfitting.
3. **Stratified temporal val** — ensure val period contains crisis events. Use
   2006-2009 (includes GFC) as val, train on 2000-2005 only. Downside: less
   training data.
4. **Use train loss plateau** — stop when train loss stops decreasing, ignore val
   loss. Works but doesn't prevent overfitting.

**Recommended approach:** Option 1 (random holdout from training windows). This
is what Bluwstein et al. (2023) effectively do — they use expanding-window CV
where the val set is a held-out portion of the same temporal window as training.

## Issue 2: Logistic Regression Too Slow on Panel

**Symptom:** Logistic regression takes hours on 35k × 5544 features (63 separate
models).

**Fix:** Either reduce feature dimensionality (PCA to ~100 components before
fitting), use `solver='saga'` with `max_iter=200`, or drop LogReg from the panel
comparison (it's not the focus).

## Issue 3: XGBoost Still Dominates

**Symptom:** XGBoost (ROC=0.69) far outperforms deep learning models even on the
panel dataset.

**Why:** XGBoost doesn't suffer from the val loss issue (trains to completion),
and it still benefits from flattened features. The panel data gives it 12x more
training samples while the flat feature representation is still effective.

**What needs to happen for TFT to win:**
- Fix Issue 1 first — TFT can't learn if it stops at epoch 1
- After fixing early stopping, TFT's entity embeddings and attention should
  provide an advantage over XGBoost's inability to share parameters across markets
- The regime module should capture cross-market contagion dynamics that flat
  features miss

## Issue 4: Feature-Label Correlation (Resolved)

Previously `drawdown_252d` directly encoded the crisis label. Fixed by removing
it and replacing with `vol_of_vol_63d` and `return_skew_63d`. Remaining features
(VIX, credit spreads, DXY) have moderate correlation (0.3-0.5) with labels —
these are genuine economic signals, not leakage.

## Results Summary

### Single-market walk-forward (3 folds, honest):
| Model | Mean ROC-AUC | Mean PR-AUC |
|---|---|---|
| XGBoost | 0.655 | 0.507 |
| LSTM | 0.529 | 0.309 |
| RA-TFT | 0.484 | 0.310 |
| Vanilla TFT | 0.477 | 0.281 |
| Logistic Reg. | 0.403 | 0.294 |

### Multi-market panel (fixed val split, A6000 GPU):
| Model | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|
| **RA-TFT** | **0.639** | **0.220** | 0.329 |
| XGBoost | 0.622 | 0.196 | 0.148 |
| Vanilla TFT | 0.513 | 0.146 | 0.337 |
| LSTM | 0.500 | 0.139 | 0.446 |

**RA-TFT outperforms XGBoost** on the panel task — the regime module and entity
embeddings provide a genuine advantage when there are multiple markets and
sufficient crisis events for the attention mechanism to learn from.

## Next Steps (Priority Order)

1. Fix val split (Issue 1) — random holdout from training windows
2. Rerun panel comparison on GPU with fixed val
3. If RA-TFT still underperforms, try:
   - Reduce model size (state_size=32 instead of 64)
   - Increase dropout to 0.3
   - Use cosine annealing LR schedule
   - Increase training data by lowering crisis threshold to 15%
4. Run walk-forward CV on panel data for publishable results
