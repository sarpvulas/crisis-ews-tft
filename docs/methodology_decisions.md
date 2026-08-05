# Methodology Decisions — Crisis Labeling

---

## Decision 4 — Fix target leakage + checkpoint restore (2026-06-23)

**Bug found (target leakage).** `training/dataset.py::CrisisDataset` auto-detects
"extra" numeric feature columns but its `exclude` set was **missing the label
columns** `ews_label`, `in_crisis`, `drawdown` — so `ews_label` and `drawdown`
were fed into the models **as input features** (label leakage). The panel
variant `PanelCrisisDataset` already excluded them correctly; the single-market
path did not.

**Why it surfaced now.** The leak was benign on the old data (leaked
encoder-window `ews_label` correlated +0.64 with the target in train, +0.02 in
test). Under the new onset-trough labels it became *harmful*: correlation +0.05
in train but **−0.29 in test** — a feature that points the wrong way at test.
RA-TFT's variable-selection network *amplifies* whichever feature tracks the
label in training, so it latched onto the leak and scored **below random**
(test ROC 0.319). The LSTM, which mixes features uniformly, barely noticed
(ROC 0.864). This is why RA-TFT specifically collapsed.

**Fix.** Added `ews_label`, `in_crisis`, `drawdown` to the `exclude` set
(`training/dataset.py:166-170`), matching `PanelCrisisDataset`.

**Second bug (checkpoint).** `Trainer` saved `best_model.pt` on val improvement
but never restored it; evaluation ran on the *last* (often worst-val) epoch.
Fixed: `Trainer.train()` now reloads the best-val checkpoint before returning
(`training/trainer.py`).

**Implication.** All earlier shared-data deep-model numbers (Decisions 2-3) were
contaminated by the leak and are superseded by the clean re-run (CREATE job
35070317). XGBoost uses a separate path (to be confirmed). Note: the calm-val
(2016-18) / crisis-test (2019-24) split (`splits.py`) remains a separate,
known contributor to val→test ranking instability; not changed here.

**Found via** a 3-agent deep investigation (architecture / training-loss /
dataset-compatibility), which converged on the leak as the primary cause.

**Result — leaked vs clean (trough labeling, test, single seed, job 35070317).**

| model | LEAKED PR/ROC/ECE | CLEAN PR/ROC/ECE |
|---|---|---|
| ra-tft | 0.157 / 0.319 / 0.224 | **0.480 / 0.561 / 0.151** |
| lstm | 0.550 / 0.864 / 0.233 | 0.435 / 0.829 / 0.154 |
| xgboost | 0.581 / 0.714 / 0.160 | 0.244 / 0.577 / 0.225 |
| tdt | 0.313 / 0.582 / 0.500 | 0.216 / 0.378 / 0.589 |

Two findings: (1) **diagnosis confirmed** — RA-TFT recovers from below-random
(0.319) to 0.561 ROC and from worst to **best PR-AUC (0.480)** once the leak it
amplified is removed. (2) **XGBoost was the biggest leak beneficiary** — PR
collapses 0.581→0.244 (near-deterministic, so not seed noise); a large part of
its apparent dominance on the shared data was leakage.

**Clean leaderboard (single seed):** RA-TFT best PR (0.480) + calibration
(0.151); LSTM best ROC (0.829) + calibration (0.154); XGBoost mid-pack; TDT
weakest. This *inverts* the "GBDTs dominate, deep models can't calibrate"
headline on leak-free data — confirmed multi-seed below.

**Multi-seed confirmation (9 seeds, clean trough, test; CREATE array 35071280).**

| model | PR-AUC | ROC-AUC | ECE |
|---|---|---|---|
| lstm | **0.441 ± 0.013** | **0.832 ± 0.013** | 0.183 ± 0.014 |
| ra-tft | 0.328 ± 0.125 | 0.637 ± 0.125 | **0.158 ± 0.036** |
| tdt | 0.246 ± 0.113 | 0.465 ± 0.157 | 0.362 ± 0.141 |
| xgboost | 0.244 ± 0.000 | 0.577 ± 0.000 | 0.225 ± 0.000 |

Paired vs XGBoost (PR-AUC, Wilcoxon): **LSTM +0.197, 9/9 seeds, p=0.004
(significant)**; RA-TFT +0.084, 6/9, p=0.20 (positive, not significant — high
variance from the val/test split); TDT +0.002, p=0.65 (tied).

**Conclusion.** On leak-free shared data, **LSTM significantly beats XGBoost on
PR-AUC and ROC-AUC and is better calibrated** — the original "boosted trees
dominate, deep models can't calibrate" headline does *not* hold once the label
leak and post-crisis bias are removed. RA-TFT beats XGBoost on average and is the
best-calibrated model but is not yet significant (variance ← calm-val/crisis-test
split, the next fix). TDT (novel) needs its class-head/balanced-sampler config
(these runs used defaults). XGBoost's prior dominance was substantially leakage.
Caveat: deep models ≤40 epochs; single temporal split (walk-forward CV pending).



Report-ready log of labeling-methodology changes, with motivation, parameters,
measured effects, and citations. Newest first. All numbers are on the shared
Bloomberg dataset (S&P 500, 2000-01-03 → 2026-05-28, 6,889 daily rows).

---

## Decision 3 — Trough-based recovery exit (2026-06-23)

**Change.** When closing a crisis episode under the onset rule, measure recovery
from the crisis **trough** instead of the all-time peak. A crisis ends once the
price has rebounded **≥ +20% above the lowest price seen since the crisis
started**, held for 30 consecutive days (then a 30-day buffer).

**Why.** With recovery measured from the running all-time peak (Decision 2), a
crisis stayed "active" until the index returned to within 10% of its previous
high — a near-full recovery that historically took *years* (dot-com active until
2006, GFC until 2012). This froze **45.6% of all days** out of training and
starved the deep models (RA-TFT's ranking collapsed). Dating a crash's end by a
rebound off the bottom matches both intuition and standard bear-market dating.

**Effect (measured).**

| | Peak-based (Decision 2) | Trough-based (this) |
|---|---|---|
| Crisis episodes | 4 | **6** (also isolates 2005 dip, 2011 Euro crisis) |
| Active days | 3,018 | **1,785** |
| Frozen (active + 30d buffer) | **45.6%** | **28.5%** |
| Warning (positive) days | 252 | **378** |
| Training-split positives (bear2022) | 126 | **252** |

Net: reclaims ~18 percentage points of training data (the 2003-2007 and
2010-2013 grind-backs) while still excluding the 4 genuine crashes.

**Parameters.** `recovery_mode="trough"`, `rebound_threshold=0.20`,
`recovery_days=30`, `post_crisis_buffer=30`, `entry_threshold=-0.20`.

**Code.** `data/pipelines/processing/crisis_labels.py` —
`identify_crisis_episodes(..., recovery_mode="trough", rebound_threshold=0.20)`.
Built via `scripts/build_taskb_from_shared.py --labeler onset --recovery trough`
→ `data/shared/processed/taskb_spx_bbg_onset_trough/`.

**Model results (single seed, primary bear2022 split, test set).** Three-way
comparison EWS vs onset-peak vs onset-trough:

| model | EWS PR/ROC/ECE | onset-peak | onset-trough |
|---|---|---|---|
| xgboost | 0.749 / 0.829 / 0.185 | 0.525 / 0.838 / 0.222 | 0.581 / 0.714 / **0.160** |
| lstm | 0.569 / 0.837 / 0.287 | 0.441 / 0.846 / 0.255 | **0.550 / 0.864 / 0.233** |
| ra-tft | 0.372 / 0.546 / 0.485 | 0.161 / 0.335 / 0.224 | 0.157 / 0.319 / 0.224 |
| tdt | 0.272 / 0.384 / 0.387 | 0.236 / 0.493 / 0.382 | 0.313 / **0.582** / 0.500 |

Test positive rate is 22.6% for both onset variants (peak vs trough differs only
in 2003-2007 / 2010-2013, which fall in the *training* window — so trough mode
enriches training without changing the test target, exactly as intended).

**Findings.**
- Unfreezing ~18 pts of training data **helped 3 of 4 models**: LSTM (ROC
  0.846→**0.864**, the best of any run; ECE 0.287→0.233), TDT (ROC
  0.493→**0.582**), XGBoost (PR 0.525→0.581, ECE 0.222→**0.160**).
- **RA-TFT did not recover** (ROC 0.319) — its failure is val-loss divergence /
  instability (see `docs/known_issues.md` Issue 1), not data volume. More data is
  not the fix; the early-stopping / regime fix is.
- onset-trough is the best labeling to date: methodologically clean *and*
  improves deep-model ranking + calibration vs both EWS and onset-peak.

**Caveats.** Single seed; deep models ≤40 epochs (LSTM/RA-TFT early-stopped at
6). `rebound_threshold` (20%) and `recovery_days` (30) tunable; sensitivity not
yet swept. Next: multi-seed + more epochs to make deep-model numbers trustworthy.
Run: CREATE job 35069869.

---

## Decision 2 — Onset-only labeling with hysteresis recovery (2026-06-23)

**Change.** Replace the persistent EWS label with an **onset-only** label:
count only the *first* day a crisis begins, freeze the model during the crisis,
and require an explicit recovery before predicting again.

* **Entry / onset:** first day drawdown from the running peak crosses ≤ −20%.
* **Freeze:** during the active crisis the label is NA — excluded from training,
  cannot spawn a new onset.
* **Exit (hysteresis):** crisis ends only after drawdown recovers above −10%
  (peak-based) for 30 consecutive days. The −20%/−10% gap is a dead-zone that
  prevents flicker (a small dip re-triggering a "new" crisis mid-episode).
* **Buffer:** a further 30 days post-recovery are also NA.

**Why.** Two problems with the persistent rule:
1. *Duration leakage / overfitting* — a long crisis contributes many correlated
   positive rows; the model can learn how long crashes last rather than whether
   one is coming. With only ~30 episodes this is a real overfitting risk. The one
   informative event is the first tip over the edge.
2. *Post-crisis bias* (Bussière & Fratzscher 2006) — labeling the recovery
   period as "calm" teaches the wrong baseline, because markets rebound
   abnormally after a crash. Excluding crisis + recovery is the simpler variant
   of their fix (Fuertes & Kalotychou 2007; Savona & Vezzoli 2015 drop
   post-crisis rows; their own preferred fix is a multinomial 4-state label).

**Effect (measured, peak-based recovery).** Episodes 32 → 4 (genuine: dot-com,
GFC, COVID, 2022); positive days 878 → 252; frozen days 649 → 3,138 (45.6%).
Model A/B (single seed): **ROC-AUC held or improved** for the strong models
(XGB 0.829→0.838, LSTM 0.837→0.846 — ROC is base-rate invariant, the fair
cross-rule metric), and **deep-model calibration improved markedly**
(RA-TFT ECE 0.485→0.224). PR-AUC fell, but mostly mechanically: the positive
base rate dropped 33.9% → 22.6% and PR-AUC is not comparable across base rates.

**Code.** `data/pipelines/processing/crisis_labels.py` (`build_crisis_labels`);
exclusion routed through `in_crisis` so the sliding-window calendar stays
contiguous (no row dropping). `scripts/build_taskb_from_shared.py --labeler onset`.

**Implementation note.** Rows are *not* dropped (that would corrupt the 252-day
sliding windows). Instead `in_crisis = crisis_active OR in_buffer`, and the
dataset already excludes windows whose final encoder step is `in_crisis` from
train/val.

---

## Decision 1 — Baseline EWS label (inherited from original thesis)

**Definition.** Crisis = drawdown from a 252-day rolling peak ≥ 20%. Onset =
first day entering a crisis. `ews_label[t] = 1` if any onset in [t+1, t+63]
(≈3-month warning horizon), else 0. In-crisis days flagged and excluded from
training. Follows Dichtl, Drobetz & Otto (2023).

**Parameters.** `threshold=0.20`, `horizon_days=63`, `peak_window=252`.
**Code.** `data/pipelines/processing/labels.py` (`build_ews_labels`); params in
`data/pipelines/config.py`.

**Limitation that motivated Decision 2.** The label persists for the whole
warning window of every onset and treats recovery as calm → duration leakage and
post-crisis bias.

---

## References

- Bussière, M. & Fratzscher, M. (2006). *Towards a new early warning system of
  financial crises.* J. International Money and Finance 25(6). (ECB WP 145.) —
  post-crisis bias; multinomial fix.
- Fuertes, A.-M. & Kalotychou, E. (2007); Savona, R. & Vezzoli, M. (2015) —
  drop post-crisis observations from the estimation sample.
- Dichtl, H., Drobetz, W. & Otto, T. (2023) — drawdown-based EWS labeling.
