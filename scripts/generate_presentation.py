#!/usr/bin/env python
"""Generate all presentation figures, tables, and architecture diagrams."""
import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT = 'output/presentation'
os.makedirs(OUT, exist_ok=True)

# Consistent style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'lstm': '#636EFA',
    'tft': '#EF553B',
    'ra-tft': '#00CC96',
    'ra-tft-tv': '#AB63FA',
    'calm': '#00CC96',
    'stress': '#FFA15A',
    'crisis': '#EF553B',
}


# ============================================================================
# FIGURE 1: Architecture Diagram
# ============================================================================
def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.8, 9.5)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Only two colors: blue (TFT base) and green (novel)
    C_BASE = '#D6E4F0'    # light blue fill
    C_BASE_E = '#4A7FB5'  # blue edge
    C_NOVEL = '#D4EDDA'   # light green fill
    C_NOVEL_E = '#28A745' # green edge
    C_OUT = '#F8D7DA'     # light red fill
    C_OUT_E = '#DC3545'   # red edge
    C_ARROW = '#555555'

    def box(x, y, w, h, text, fc=C_BASE, ec=C_BASE_E, fontsize=11, bold=False):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                               facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2))

    # Legend at bottom
    ax.text(4, -0.5, '■  Blue = TFT baseline components', fontsize=12, color=C_BASE_E, fontweight='bold')
    ax.text(10, -0.5, '■  Green = Novel regime-aware additions', fontsize=12, color=C_NOVEL_E, fontweight='bold')

    # ---- Bottom: Inputs ----
    box(1, 0.2, 3.5, 0.8, 'Historical Features\n252 days × 16 features', fontsize=11)
    box(5.5, 0.2, 3, 0.8, 'Future Features\n63 days × 2', fontsize=11)

    # ---- Embeddings + VSN ----
    box(0.5, 1.5, 8.5, 0.7, 'Input Embedding  →  Variable Selection Networks (VSN)', fontsize=12)
    arr(2.75, 1.0, 4.75, 1.5)
    arr(7, 1.0, 4.75, 1.5)

    # ---- LSTM ----
    box(0.5, 2.7, 4, 0.8, 'Encoder LSTM\n252 steps', fontsize=12)
    box(5, 2.7, 4, 0.8, 'Decoder LSTM\n63 steps', fontsize=12)
    arr(4.75, 2.2, 2.5, 2.7)
    arr(4.75, 2.2, 7, 2.7)
    arr(4.5, 3.1, 5, 3.1)

    # ---- Gated skip ----
    box(0.5, 4.0, 8.5, 0.6, 'Gated Layer Norm + Skip Connection', fontsize=11)
    arr(4.75, 3.5, 4.75, 4.0)

    # ---- NOVEL: Regime Module (right side) ----
    box(10.5, 2.2, 5, 2.0,
        'Neural HMM\nRegime Detection\n\n3 states: Calm / Stress / Crisis\nTime-varying transitions',
        fc=C_NOVEL, ec=C_NOVEL_E, fontsize=12, bold=True)
    arr(9, 3.1, 10.5, 3.2)  # LSTM → regime

    # ---- Static Enrichment ----
    box(0.5, 5.1, 8.5, 0.6, 'Static Enrichment GRN', fontsize=11)
    arr(4.75, 4.6, 4.75, 5.1)

    # ---- NOVEL: Regime-Conditioned Attention ----
    box(0.5, 6.2, 8.5, 0.8, 'Interpretable Multi-Head Attention\nwith Regime-Conditioned Bias',
        fc=C_NOVEL, ec=C_NOVEL_E, fontsize=12, bold=True)
    arr(4.75, 5.7, 4.75, 6.2)

    # Regime → Attention arrow
    box(10.5, 5.5, 5, 0.8, 'Regime Probabilities\nP(calm), P(stress), P(crisis)',
        fc=C_NOVEL, ec=C_NOVEL_E, fontsize=11)
    arr(13, 4.2, 13, 5.5)
    arr(10.5, 5.9, 9, 6.6)

    # ---- Output ----
    box(0.5, 7.5, 4, 0.7, 'Position-wise GRN\n+ Gated Skip', fontsize=11)
    box(5, 7.5, 4, 0.7, 'Output → 7 Quantiles × 63 days',
        fc=C_OUT, ec=C_OUT_E, fontsize=12, bold=True)
    arr(4.75, 7.0, 2.5, 7.5)
    arr(4.5, 7.85, 5, 7.85)

    # Final output label
    box(5, 8.5, 4, 0.5, 'Drawdown Distribution',
        fc=C_OUT, ec=C_OUT_E, fontsize=12, bold=True)
    arr(7, 8.2, 7, 8.5)

    fig.savefig(f'{OUT}/architecture_diagram.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'Saved: {OUT}/architecture_diagram.png')
    plt.close()


# ============================================================================
# FIGURE 2: Model Comparison Bar Chart
# ============================================================================
def draw_comparison_bars():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = ['LSTM', 'Vanilla TFT', 'RA-TFT\n(Static HMM)', 'RA-TFT\n(TV-HMM)']
    pr_aucs = [0.306, 0.284, 0.319, 0.352]
    pr_sems = [0.012, 0.012, 0.015, 0.031]
    roc_aucs = [0.501, 0.411, 0.521, 0.508]
    roc_sems = [0.018, 0.041, 0.023, 0.031]
    colors = [COLORS['lstm'], COLORS['tft'], COLORS['ra-tft'], COLORS['ra-tft-tv']]

    # PR-AUC
    ax = axes[0]
    bars = ax.bar(models, pr_aucs, yerr=pr_sems, capsize=5, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0.317, color='gray', linestyle='--', alpha=0.7, label='Random baseline (crash rate)')
    ax.set_ylabel('PR-AUC')
    ax.set_title('Precision-Recall AUC', fontweight='bold')
    ax.set_ylim(0.2, 0.42)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, pr_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ROC-AUC
    ax = axes[1]
    bars = ax.bar(models, roc_aucs, yerr=roc_sems, capsize=5, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random baseline (0.5)')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('ROC-AUC', fontweight='bold')
    ax.set_ylim(0.3, 0.6)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, roc_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Model Comparison — Market Crash Prediction (S&P 500 Daily, 2013-2024)',
                  fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/model_comparison_bars.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/model_comparison_bars.png')
    plt.close()


# ============================================================================
# FIGURE 3: Grid Search Heatmap
# ============================================================================
def draw_grid_search_heatmap():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    gammas = [0.3, 0.5, 0.8, 1.0]
    cws = [1.0, 1.5, 2.0, 3.0]

    # PR-AUC results from grid search (equal quantile weights)
    data = np.array([
        [0.348, 0.359, 0.359, 0.322],  # gamma=0.3
        [0.340, 0.329, 0.340, 0.351],  # gamma=0.5 (baseline at cw=1.0)
        [0.344, 0.371, 0.373, 0.319],  # gamma=0.8
        [0.358, 0.388, 0.320, 0.359],  # gamma=1.0
    ])

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.31, vmax=0.40)
    ax.set_xticks(range(len(cws)))
    ax.set_xticklabels([f'{cw}x' for cw in cws])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f'γ={g}' for g in gammas])
    ax.set_xlabel('Crisis Weight', fontsize=12)
    ax.set_ylabel('Regime Loss Weight (γ)', fontsize=12)
    ax.set_title('Grid Search: PR-AUC by Loss Configuration\n(RA-TFT, 5 seeds each)', fontweight='bold')

    # Annotate cells
    for i in range(len(gammas)):
        for j in range(len(cws)):
            val = data[i, j]
            color = 'white' if val < 0.33 or val > 0.37 else 'black'
            weight = 'bold' if val >= 0.37 else 'normal'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, label='PR-AUC', shrink=0.8)
    fig.tight_layout()
    fig.savefig(f'{OUT}/grid_search_heatmap.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/grid_search_heatmap.png')
    plt.close()


# ============================================================================
# FIGURE 4: Data Overview
# ============================================================================
def draw_data_overview():
    import pandas as pd

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1])

    # Load data
    frames = []
    for split, color, label in [('train', '#636EFA', 'Train 1990-2005'),
                                  ('val', '#FFA15A', 'Val 2006-2012'),
                                  ('test', '#00CC96', 'Test 2013-2024')]:
        df = pd.read_parquet(f'data/processed/task_b_{split}.parquet')
        df['split'] = label
        df['color'] = color
        frames.append(df)
    full = pd.concat(frames)
    full['date'] = pd.to_datetime(full['date'])

    # Price with crash periods highlighted
    ax = axes[0]
    ax.plot(full['date'], full['close'], color='#333', linewidth=0.5, alpha=0.8)
    crash_mask = full['crash_label'] == 1
    ax.fill_between(full['date'], full['close'].min(), full['close'].max(),
                     where=crash_mask, color='red', alpha=0.15, label='Crash periods (≥10% drawdown)')

    # Mark splits
    for split_date, label in [('2006-01-01', 'Val start'), ('2013-01-01', 'Test start')]:
        ax.axvline(pd.Timestamp(split_date), color='black', linestyle='--', alpha=0.5)
        ax.text(pd.Timestamp(split_date), full['close'].max() * 0.95, f'  {label}',
                fontsize=9, alpha=0.7)

    ax.set_ylabel('S&P 500 Close')
    ax.set_title('S&P 500 with Crash Periods and Temporal Splits', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # Regime distribution per split
    ax = axes[1]
    splits = ['Train 1990-2005', 'Val 2006-2012', 'Test 2013-2024']
    regime_counts = []
    for split_name in splits:
        split_df = full[full['split'] == split_name]
        counts = split_df['regime_label'].value_counts().sort_index()
        total = len(split_df)
        regime_counts.append([counts.get(i, 0) / total * 100 for i in range(3)])

    regime_counts = np.array(regime_counts)
    x = np.arange(len(splits))
    w = 0.25
    bars_calm = ax.bar(x - w, regime_counts[:, 0], w, color=COLORS['calm'], label='Calm', edgecolor='white')
    bars_stress = ax.bar(x, regime_counts[:, 1], w, color=COLORS['stress'], label='Stress', edgecolor='white')
    bars_crisis = ax.bar(x + w, regime_counts[:, 2], w, color=COLORS['crisis'], label='Crisis', edgecolor='white')
    # Add percentage labels on each bar
    for bars in [bars_calm, bars_stress, bars_crisis]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel('% of days')
    ax.set_title('Regime Distribution per Split', fontweight='bold')
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUT}/data_overview.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/data_overview.png')
    plt.close()


# ============================================================================
# FIGURE 5: Ablation — Phase 1 Results
# ============================================================================
def draw_ablation():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    configs = ['Baseline\n(γ=0.5)', 'Tail\nWeights', 'Crisis\n2x', 'γ=0.8',
               'γ=0.8\ncw=1.5', 'γ=1.0\ncw=1.5']
    pr_aucs = [0.340, 0.340, 0.334, 0.344, 0.371, 0.388]
    pr_stds = [0.051, 0.065, 0.049, 0.040, 0.085, 0.108]
    sems = [s / np.sqrt(5) for s in pr_stds]

    colors = ['#9E9E9E', '#9E9E9E', '#9E9E9E', '#4CAF50', '#2E7D32', '#1B5E20']
    bars = ax.bar(configs, pr_aucs, yerr=sems, capsize=5, color=colors, edgecolor='white', linewidth=1.5)

    ax.axhline(y=0.317, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.axhline(y=0.340, color='blue', linestyle=':', alpha=0.4, label='Baseline RA-TFT')
    ax.set_ylabel('PR-AUC (mean ± SEM)')
    ax.set_title('Loss Configuration Ablation — RA-TFT\n(Stronger regime supervision → better performance)', fontweight='bold')
    ax.set_ylim(0.28, 0.45)
    ax.legend(fontsize=9)

    for bar, val in zip(bars, pr_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.tight_layout()
    fig.savefig(f'{OUT}/ablation_results.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/ablation_results.png')
    plt.close()


# ============================================================================
# FIGURE 6: HMM Comparison (Static vs Time-Varying)
# ============================================================================
def draw_hmm_comparison():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    models = ['Static HMM', 'Time-Varying HMM']
    pr_aucs = [0.319, 0.352]
    sems = [0.015, 0.031]
    colors = [COLORS['ra-tft'], COLORS['ra-tft-tv']]

    # Per-seed scatter
    static_seeds = [0.367, 0.341, 0.318, 0.296, 0.272]
    tv_seeds = [0.371, 0.286, 0.476, 0.333, 0.294]

    bars = ax.bar(models, pr_aucs, yerr=sems, capsize=6, color=colors,
                   edgecolor='white', linewidth=2, width=0.5, alpha=0.7)

    # Overlay individual seeds
    for i, (seeds, x) in enumerate([(static_seeds, 0), (tv_seeds, 1)]):
        jitter = np.random.uniform(-0.1, 0.1, len(seeds))
        ax.scatter([x + j for j in jitter], seeds, color='black', s=40, zorder=5, alpha=0.6)

    ax.axhline(y=0.317, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
    ax.set_ylabel('PR-AUC')
    ax.set_title('Static vs Time-Varying HMM Transitions\n(dots = individual seeds)', fontweight='bold')
    ax.set_ylim(0.2, 0.55)
    ax.legend(fontsize=9)

    for bar, val, sem in zip(bars, pr_aucs, sems):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sem + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(f'{OUT}/hmm_comparison.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/hmm_comparison.png')
    plt.close()


# ============================================================================
# FIGURE 7: Results Summary Table (as image)
# ============================================================================
def draw_results_table():
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    ax.axis('off')

    columns = ['Model', 'PR-AUC ↑', 'ROC-AUC ↑', 'F1 ↑', 'ECE ↓']
    data = [
        ['LSTM (baseline)',   '0.306 ± 0.012', '0.501 ± 0.018', '0.489 ± 0.007', '0.120 ± 0.016'],
        ['Vanilla TFT',      '0.284 ± 0.012', '0.411 ± 0.041', '0.492 ± 0.013', '0.334 ± 0.017'],
        ['RA-TFT (static HMM)',  '0.319 ± 0.015', '0.521 ± 0.023', '0.493 ± 0.009', '0.194 ± 0.021'],
        ['RA-TFT (TV-HMM)',  '0.352 ± 0.031', '0.508 ± 0.031', '0.482 ± 0.013', '0.212 ± 0.031'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.2)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor('#4A7FB5')
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=14)

    # Highlight best model rows in green
    for j in range(len(columns)):
        table[3, j].set_facecolor('#E8F5E9')
        table[4, j].set_facecolor('#D4EDDA')
        table[4, j].set_text_props(fontweight='bold')

    # Alternate row colors for non-highlighted
    for j in range(len(columns)):
        table[1, j].set_facecolor('white')
        table[2, j].set_facecolor('#F5F5F5')

    ax.set_title('Model Comparison — S&P 500 Daily Crash Prediction\nmean ± SEM across 5 seeds  |  Test period: 2013–2024',
                  fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/results_table.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/results_table.png')
    plt.close()


# ============================================================================
# FIGURE 8: Next Steps Roadmap
# ============================================================================
def draw_roadmap():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Next Steps', fontsize=16, fontweight='bold', pad=15)

    steps = [
        (1, 3.5, 'Bloomberg Data\n6 new features\n(credit, options)', '#E3F2FD', '#1976D2'),
        (4, 3.5, 'Walk-Forward CV\n6 folds × 20 seeds\nNo data leakage', '#FFF3E0', '#E65100'),
        (7, 3.5, 'Multi-Frequency\nEnsemble\nDaily+Weekly+Monthly', '#E8F5E9', '#2E7D32'),
        (10, 3.5, 'Final Thesis\nFigures & Tables\nStatistical tests', '#F3E5F5', '#7B1FA2'),
    ]

    for x, y, text, fc, ec in steps:
        rect = FancyBboxPatch((x, y - 0.8), 2.5, 1.6, boxstyle="round,pad=0.15",
                               facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 1.25, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i+1][0], steps[i][1]),
                    xytext=(steps[i][0] + 2.5, steps[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=2))

    # Timeline
    ax.text(2.25, 1.5, 'Next week', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(5.25, 1.5, 'Week 2', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(8.25, 1.5, 'Week 3', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(11.25, 1.5, 'Week 4', ha='center', fontsize=9, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(f'{OUT}/roadmap.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}/roadmap.png')
    plt.close()


# ============================================================================
# Generate all
# ============================================================================
if __name__ == '__main__':
    print('Generating presentation figures...\n')
    draw_architecture()
    draw_comparison_bars()
    draw_grid_search_heatmap()
    draw_data_overview()
    draw_ablation()
    draw_hmm_comparison()
    draw_results_table()
    draw_roadmap()
    print(f'\nAll figures saved to {OUT}/')
    print(f'Files: {sorted(os.listdir(OUT))}')
