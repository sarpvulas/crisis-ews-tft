# Problem Statement: Multi-Market Crisis Early Warning System

## Motivation

Financial crises propagate across markets through contagion, shared exposures, and
investor behavior. A single-market EWS trained on S&P 500 alone faces two
fundamental limitations that prevent deep learning architectures from reaching
their potential:

**1. Insufficient crisis events for temporal models.**
The S&P 500 experienced only 4 episodes of >= 20% drawdown since 2000 (dot-com,
GFC, COVID, 2022 bear market). With ~37 crisis onset days across 6,288 trading
days, there is not enough signal for attention-based architectures to learn
meaningful temporal patterns. Simpler models (XGBoost, logistic regression) that
flatten the time dimension into a large feature vector outperform deep sequence
models in this low-data regime — a finding consistent with Bluwstein et al. (2023,
Journal of Financial Econometrics) who report that gradient-boosted trees match or
beat neural networks on single-country crisis prediction.

**2. No cross-entity learning.**
The Temporal Fusion Transformer was designed for multi-entity forecasting (Lim et
al., 2021 — original paper: 369 retail stores). Its static variable selection
network learns entity-specific behavior while sharing temporal patterns across
entities. With a single market, this mechanism is unused. XGBoost, which cannot
share parameters across entities, suffers no disadvantage.

## Approach: Panel EWS Across Major Equity Markets

We reformulate the task as a **multi-market panel EWS**: predict crisis onset
probability simultaneously across 12 major equity indices. This setup:

- **Multiplies crisis events.** Different markets crash at different times and
  with different dynamics. The 2015 China crash, 2011 Eurozone debt crisis, and
  2018 EM selloff each provide additional crisis episodes that the S&P 500 alone
  does not capture. Across 12 markets and 24 years, we observe 30-50 distinct
  crisis episodes instead of 4.

- **Enables cross-market learning.** TFT's static covariate encoder learns market
  identity embeddings — "developed vs. emerging," "export-driven vs. domestic" —
  and conditions attention and gating on these embeddings. The model learns that
  stress in US credit markets precedes crises in export-dependent economies, or
  that VIX spikes lead emerging market drawdowns by 2-3 weeks.

- **Activates the regime module.** RA-TFT's regime detection module can now learn
  global regime states (calm/stress/crisis) from the cross-sectional pattern of
  multiple markets simultaneously, rather than from a single noisy signal.
  Cross-market contagion dynamics — where stress in one market cascades to
  others — provide the sequential structure that LSTM and attention are designed
  to capture.

- **Creates a natural advantage for sequential models over tree-based models.**
  XGBoost flattens the 252-day history into ~7,500 features per market and
  cannot share parameters across markets. With 12 markets it must either train
  one model per market (few samples each) or pool all markets (losing entity
  specificity). TFT handles both naturally through its architecture.

## Markets

| Region | Market | Ticker | Available From |
|--------|--------|--------|---------------|
| North America | S&P 500 | ^GSPC | 2000 |
| North America | S&P/TSX (Canada) | ^GSPTSE | 2000 |
| Europe | FTSE 100 (UK) | ^FTSE | 2000 |
| Europe | DAX (Germany) | ^GDAXI | 2000 |
| Europe | CAC 40 (France) | ^FCHI | 2000 |
| Asia-Pacific | Nikkei 225 (Japan) | ^N225 | 2000 |
| Asia-Pacific | Hang Seng (HK) | ^HSI | 2000 |
| Asia-Pacific | ASX 200 (Australia) | ^AXJO | 2000 |
| Asia-Pacific | KOSPI (South Korea) | ^KS11 | 2000 |
| Emerging | Bovespa (Brazil) | ^BVSP | 2000 |
| Emerging | Nifty 50 (India) | ^NSEI | 2000 |
| Emerging | Shanghai Composite | 000001.SS | 2000 |

## Features

**Per-market features (computed individually for each market):**
- Log returns (1d, 5d, 21d)
- Realized volatility (5d, 21d, 63d)
- Short-horizon drawdown (63d rolling peak)
- Hurst exponent (63d)
- Vol-of-vol (63d)
- Return skewness (63d)
- Volume z-score (63d), Amihud illiquidity (21d)

**Global features (shared across all markets, from FRED/Yahoo):**
- VIX, VIX term structure (VIX - VIX3M)
- US credit spreads (HY, IG, HY-IG differential)
- TED spread
- DXY (dollar index), gold
- MOVE index (bond volatility)

**Static categorical features:**
- Market identity (12 categories) — enables TFT's static embedding

## Labels

Per-market EWS label following Dichtl et al. (2023):
- Crisis = drawdown from 252-day rolling peak >= 20%
- Onset = first day entering a crisis episode
- y_t = 1 if crisis onset in [t+1, t+63], else 0
- Observations already in crisis excluded from training

## Evaluation

Walk-forward cross-validation anchored on crisis episodes, averaged across
markets. Primary metrics: ROC-AUC, PR-AUC (more important given class
imbalance). Bootstrap confidence intervals on PR-AUC.

## Expected Outcome

With 12 markets and 30-50 crisis episodes, the RA-TFT's regime detection and
cross-entity learning mechanisms have sufficient data to outperform flat models.
We expect the ranking to shift from:

**Single-market:** XGBoost > Logistic > LSTM > TFT > RA-TFT
**Multi-market panel:** RA-TFT > TFT > LSTM > XGBoost > Logistic

This is the natural domain for attention-based architectures: enough entities to
learn shared patterns, enough events to train the attention mechanism, and enough
temporal structure to justify sequential modeling over feature flattening.
