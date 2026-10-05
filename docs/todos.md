> **Historical working notes from before the final experiments. Superseded by the dissertation (docs/dissertation.pdf); some statements here are outdated.**

# TODO List

## Blocked (needed GPU credits)

- [ ] **Recover RA-TFT X_combo checkpoints** — A cloud GPU pod exited with 4/5 seeds mid-training. Run `scripts/recover_checkpoints.sh` after adding GPU credits (~$1 needed). Saves seed 42 at epoch 79, others at 65-77.

## Architectural Improvements (next phase)

- [ ] **Regime-Routed Mixture of Experts (Option A)** — Add MoE layer where existing regime module routes to specialized experts. Most novel contribution — nobody has done regime-detected expert routing for financial EWS. Clean ablation story. See `memory/project_architecture_ideas.md`.

- [ ] **Mamba backbone (Option B)** — Replace LSTM with Mamba (`pip install mamba-ssm`). Better long-range memory for 252-step history. Drop-in replacement ~30 lines. Combined with regime detection is novel for financial EWS.

## High Priority

- [x] **Post-hoc calibration for RA-TFT** — Added Platt + isotonic. Platt best for RA-TFT (ECE 0.266→0.256). Note: RA-TFT ROC varies across runs (0.639, 0.577, 0.570) — multi-seed averaging needed.

- [ ] **Walk-forward CV on multi-market panel** — Current panel results use a single temporal split (train pre-2015, test 2015+). For publishable results, run crisis-anchored walk-forward CV across panel data. Average metrics across folds with bootstrap CIs.

- [ ] **Ablation study: regime module contribution** — The 0.126 ROC-AUC gap between Vanilla TFT (0.513) and RA-TFT (0.639) needs decomposition. Run: (1) RA-TFT with regime module only, no regime attention, (2) RA-TFT with regime attention only, no HMM, (3) full RA-TFT. Isolate which component drives the improvement.

- [ ] **Per-market ROC-AUC breakdown + macro-averaged ROC-AUC** — Current evaluation pools all markets into one flat array. Add: (1) per-market ROC-AUC table showing performance on each of the 12 indices, (2) macro-averaged ROC-AUC (mean of per-market scores, equal weight to each market). Report both pooled and macro-averaged in final results.

## Medium Priority

- [ ] **Reliability diagrams** — Generate predicted probability vs actual frequency plots for all models. Visualize where RA-TFT's calibration breaks down. Add to `analysis/graphs/`.

- [ ] **Verify GPU XGBoost** — Code updated in `models/baselines/xgboost_baseline.py` to auto-detect CUDA and use `tree_method='gpu_hist'`. Verify on next GPU run — should reduce training from ~30 min to ~3-5 min.

## Completed

- [x] **Multi-market panel pipeline** — 12 equity indices, 72,801 rows, panel features with market_id categorical. Data at `data/processed/panel/panel_features.parquet`.
- [x] **Fix val split** — Random holdout from training windows instead of temporal val. Both train/val have crises, fixing early stopping divergence.
- [x] **Remove feature-label leakage** — Removed `drawdown_252d` and `drawdown_504d` which directly encoded the crisis label definition.
- [x] **EWS labeling** — Switched from forward drawdown regression to crisis onset binary classification following Dichtl et al. (2023). 20% threshold, 252-day rolling peak.
- [x] **Panel comparison on GPU** — RA-TFT 0.639 > XGBoost 0.622 on A6000. Results at `output/panel_comparison.json`.
- [x] **Post-hoc calibration** — Added `PostHocCalibrator` (Platt + isotonic) to `training/evaluation.py`. Added Brier score metric. Updated `run_panel_comparison.py` to fit on val, apply on test. Platt scaling best for RA-TFT: ECE 0.266→0.256, Brier 0.265→0.255. XGBoost already well-calibrated (ECE 0.148).
- [x] **Fix best checkpoint not loading** — Trainer saved checkpoints but never restored them before evaluation. Fixed in `run_panel_comparison.py` to load `best_model.pt` after training.
- [x] **Fix XGBoost gpu_hist deprecated** — Updated `xgboost_baseline.py` to use `tree_method="hist"` + `device="cuda"` for XGBoost 2.0+.
- [x] **Add subprocess seed** — Fixed non-determinism in parallel GPU training by setting `torch.manual_seed(42)` in each subprocess.
