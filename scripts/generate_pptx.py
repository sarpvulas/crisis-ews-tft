#!/usr/bin/env python
"""Generate polished PowerPoint presentation for thesis progress report."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

OUT = 'output/presentation'
IMG = OUT

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Modern dark color palette
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)     # deep navy
BG_CARD = RGBColor(0x1A, 0x24, 0x3B)     # card background
ACCENT1 = RGBColor(0x4F, 0x8C, 0xFF)     # blue accent
ACCENT2 = RGBColor(0x00, 0xD4, 0xAA)     # green/teal accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x99, 0xA3, 0xBB)
LIGHT = RGBColor(0xCC, 0xD5, 0xE8)
ORANGE = RGBColor(0xFF, 0x9F, 0x43)

SW = prs.slide_width
SH = prs.slide_height


def dark_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK


def add_accent_line(slide, top=Inches(1.05)):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top, Inches(2), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT1
    line.line.fill.background()


def add_slide_number(slide, num, total=15):
    tb = slide.shapes.add_textbox(Inches(12.3), Inches(7.0), Inches(0.8), Inches(0.4))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = f'{num}/{total}'
    p.font.size = Pt(10)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT


# ============================================================================
# Slide 1: Title
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

# Accent line
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(2.3), Inches(3), Inches(0.05))
line.fill.solid(); line.fill.fore_color.rgb = ACCENT1; line.line.fill.background()

tb = s.shapes.add_textbox(Inches(1.5), Inches(2.6), Inches(10), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = 'Regime-Aware Temporal Fusion Transformer'
p.font.size = Pt(38); p.font.color.rgb = WHITE; p.font.bold = True
p2 = tf.add_paragraph()
p2.text = 'for Market Crash Prediction'
p2.font.size = Pt(38); p2.font.color.rgb = ACCENT1; p2.font.bold = True

tb2 = s.shapes.add_textbox(Inches(1.5), Inches(4.5), Inches(10), Inches(1))
tf2 = tb2.text_frame; tf2.word_wrap = True
p3 = tf2.paragraphs[0]
p3.text = 'Thesis Progress Report  •  March 2026'
p3.font.size = Pt(18); p3.font.color.rgb = GRAY

add_slide_number(s, 1)


# ============================================================================
# Slide 2: Problem Statement
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Problem Statement'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

# Three cards
cards = [
    ('GOAL', 'Predict S&P 500 crashes\n(≥10% drawdown in 63 days)\n\nEarly warning for risk\nmanagement & portfolio\nprotection', ACCENT1),
    ('CHALLENGE', 'Markets are near-efficient\n\nCrashes are rare (13.4%)\n\nDifferent regimes need\ndifferent strategies\n\nDaily frequency = noisy', ORANGE),
    ('APPROACH', 'Extend TFT with learned\nregime detection module\n\n3 states: calm / stress / crisis\n\nTime-varying transitions\nadapt to market conditions', ACCENT2),
]
for i, (title, body, color) in enumerate(cards):
    left = Inches(0.8 + i * 4.1)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(3.7), Inches(5.3))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = color; card.line.width = Pt(1.5)

    tb = s.shapes.add_textbox(left + Inches(0.3), Inches(1.8), Inches(3.1), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(14); p.font.color.rgb = color; p.font.bold = True

    tb2 = s.shapes.add_textbox(left + Inches(0.3), Inches(2.5), Inches(3.1), Inches(4))
    tf = tb2.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = body; p.font.size = Pt(14); p.font.color.rgb = LIGHT; p.line_spacing = Pt(20)

add_slide_number(s, 2)


# ============================================================================
# Slide 3: Architecture
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.6))
p = tb.text_frame.paragraphs[0]
p.text = 'RA-TFT Architecture'; p.font.size = Pt(28); p.font.color.rgb = WHITE; p.font.bold = True

tb2 = s.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11), Inches(0.4))
p2 = tb2.text_frame.paragraphs[0]
p2.text = 'Novel components highlighted in green: Neural HMM + Regime-Conditioned Attention'
p2.font.size = Pt(13); p2.font.color.rgb = GRAY

# Image — scale to fit (max height 6.0in)
img_path = f'{IMG}/architecture_diagram.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.8), Inches(1.3), width=Inches(11.7), height=Inches(5.9))

add_slide_number(s, 3)


# ============================================================================
# Slide 4: Novel Components
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Novel Components'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

components = [
    ('1', 'Neural Hidden Markov Model',
     '• 3 learned states: Calm / Stress / Crisis\n• Time-varying transition matrix\n• Differentiable forward algorithm\n• End-to-end trainable with main task', ACCENT1),
    ('2', 'Regime-Conditioned Attention',
     '• Attention weights modulated by regime\n• Crisis → looks further back in history\n• Calm → focuses on recent patterns\n• Additive bias on attention logits', ACCENT2),
    ('3', 'Multi-Task Loss (γ=0.8)',
     '• Primary: 7-quantile drawdown prediction\n• Auxiliary: regime classification (NLL)\n• Crisis-period sample weighting (1.5x)\n• Stronger regime supervision = key finding', ORANGE),
]
for i, (num, title, body, color) in enumerate(components):
    left = Inches(0.8 + i * 4.1)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(3.7), Inches(5.3))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = color; card.line.width = Pt(1.5)

    # Number circle
    circle = s.shapes.add_shape(MSO_SHAPE.OVAL, left + Inches(0.2), Inches(1.7), Inches(0.5), Inches(0.5))
    circle.fill.solid(); circle.fill.fore_color.rgb = color; circle.line.fill.background()
    circle.text_frame.paragraphs[0].text = num
    circle.text_frame.paragraphs[0].font.size = Pt(18)
    circle.text_frame.paragraphs[0].font.color.rgb = WHITE
    circle.text_frame.paragraphs[0].font.bold = True
    circle.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    tb = s.shapes.add_textbox(left + Inches(0.9), Inches(1.75), Inches(2.6), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(15); p.font.color.rgb = WHITE; p.font.bold = True

    tb2 = s.shapes.add_textbox(left + Inches(0.3), Inches(2.5), Inches(3.1), Inches(4))
    tf = tb2.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = body; p.font.size = Pt(13); p.font.color.rgb = LIGHT; p.line_spacing = Pt(19)

add_slide_number(s, 4)


# ============================================================================
# Slide 5: Data Overview
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.6))
p = tb.text_frame.paragraphs[0]
p.text = 'Data Overview — S&P 500 Daily (1990–2024)'; p.font.size = Pt(28); p.font.color.rgb = WHITE; p.font.bold = True

# Stats bar
stats = '8,795 trading days   •   1,181 crash days (13.4%)   •   16 normalized features   •   No temporal leakage'
tb2 = s.shapes.add_textbox(Inches(0.8), Inches(0.9), Inches(11), Inches(0.4))
p2 = tb2.text_frame.paragraphs[0]
p2.text = stats; p2.font.size = Pt(12); p2.font.color.rgb = ACCENT2

img_path = f'{IMG}/data_overview.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.5), Inches(1.4), width=Inches(12.3), height=Inches(5.8))

add_slide_number(s, 5)


# ============================================================================
# Slide 6: Features
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Feature Engineering — 16 Features'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

categories = [
    ('Price (8)', 'Log returns (1d, 5d, 21d)\nRealized vol (5d, 21d, 63d)\n52-week drawdown\nHurst exponent'),
    ('Volatility (2)', 'VIX (implied vol)\nImplied-realized spread'),
    ('Cross-Asset (4)', 'Gold price\nGold/SPX ratio\nDollar index (DXY)\nDXY 5d return'),
    ('Liquidity (2)', 'Amihud illiquidity\nVolume z-score'),
]
for i, (cat, feats) in enumerate(categories):
    left = Inches(0.8 + i * 3.1)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(2.8), Inches(3.5))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = ACCENT1; card.line.width = Pt(1)

    tb = s.shapes.add_textbox(left + Inches(0.2), Inches(1.7), Inches(2.4), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = cat; p.font.size = Pt(14); p.font.color.rgb = ACCENT1; p.font.bold = True

    tb2 = s.shapes.add_textbox(left + Inches(0.2), Inches(2.2), Inches(2.4), Inches(2.5))
    tf = tb2.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = feats; p.font.size = Pt(13); p.font.color.rgb = LIGHT; p.line_spacing = Pt(20)

# Note at bottom
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.3), Inches(11), Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = 'All features z-score normalized (fitted on training set only) • Coming next week: 6 Bloomberg features (credit OAS, put-call, SKEW, MOVE)'
p.font.size = Pt(12); p.font.color.rgb = GRAY

add_slide_number(s, 6)


# ============================================================================
# Slide 7: Data Sources
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Data Sources'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

sources = [
    ('Yahoo Finance', 'S&P 500 (^GSPC)', 'Daily OHLCV prices\nGold futures (GC=F)\nDollar index (DX-Y.NYB)', ACCENT1),
    ('FRED API', 'Federal Reserve Economic Data', 'VIX (VIXCLS)\nHigh-yield spread (BAMLH0A0HYM2)\nTED spread, Treasury yields\nNBER recession dates', ACCENT2),
    ('Derived / Computed', 'Engineered from raw data', 'Log returns, realized volatility\nHurst exponent, Amihud illiquidity\nImplied-realized vol spread\nGold/SPX ratio, drawdown', ORANGE),
    ('Bloomberg (planned)', 'Next week', 'Put-call ratio\nCBOE SKEW index\nIG/HY credit OAS\nMOVE index (bond vol)\nTED spread (cleaner)', RGBColor(0xE0, 0x60, 0x60)),
]
for i, (source, sub, details, color) in enumerate(sources):
    left = Inches(0.5 + i * 3.2)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(2.9), Inches(5.0))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = color; card.line.width = Pt(1.5)

    tb = s.shapes.add_textbox(left + Inches(0.2), Inches(1.7), Inches(2.5), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    p.text = source; p.font.size = Pt(16); p.font.color.rgb = WHITE; p.font.bold = True

    tb2 = s.shapes.add_textbox(left + Inches(0.2), Inches(2.2), Inches(2.5), Inches(0.4))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = sub; p2.font.size = Pt(11); p2.font.color.rgb = color; p2.font.italic = True

    tb3 = s.shapes.add_textbox(left + Inches(0.2), Inches(2.7), Inches(2.5), Inches(3.5))
    tf = tb3.text_frame; tf.word_wrap = True
    p3 = tf.paragraphs[0]
    p3.text = details; p3.font.size = Pt(13); p3.font.color.rgb = LIGHT; p3.line_spacing = Pt(20)

# Bottom note
tb = s.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11), Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = 'All data freely available (except Bloomberg) • No survivorship bias • Daily frequency from 1990'
p.font.size = Pt(12); p.font.color.rgb = GRAY

add_slide_number(s, 7)


# ============================================================================
# Slide 8: Results Table
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.6))
p = tb.text_frame.paragraphs[0]
p.text = 'Results — Model Comparison'; p.font.size = Pt(28); p.font.color.rgb = WHITE; p.font.bold = True

tb2 = s.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11), Inches(0.4))
p2 = tb2.text_frame.paragraphs[0]
p2.text = 'Test set: S&P 500 daily, 2013–2024  •  Mean ± SEM across 5 seeds'; p2.font.size = Pt(13); p2.font.color.rgb = GRAY

img_path = f'{IMG}/results_table.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.5), Inches(1.3), width=Inches(12.3), height=Inches(4.0))

# Metric explanations
tb3 = s.shapes.add_textbox(Inches(0.8), Inches(5.5), Inches(5.8), Inches(1.5))
tf3 = tb3.text_frame; tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = 'PR-AUC (Precision-Recall AUC)'; p.font.size = Pt(14); p.font.color.rgb = ACCENT1; p.font.bold = True
p2 = tf3.add_paragraph()
p2.text = 'When the model says "crash", how often is it right,\nand how many actual crashes does it catch?\nBaseline (random) = 0.317 (crash rate)'
p2.font.size = Pt(12); p2.font.color.rgb = LIGHT; p2.space_before = Pt(4)

tb4 = s.shapes.add_textbox(Inches(7), Inches(5.5), Inches(5.8), Inches(1.5))
tf4 = tb4.text_frame; tf4.word_wrap = True
p = tf4.paragraphs[0]
p.text = 'ROC-AUC'; p.font.size = Pt(14); p.font.color.rgb = ACCENT2; p.font.bold = True
p2 = tf4.add_paragraph()
p2.text = 'Given a random crash day and a random calm day,\nhow often does the model rank the crash day higher?\nBaseline (random) = 0.500'
p2.font.size = Pt(12); p2.font.color.rgb = LIGHT; p2.space_before = Pt(4)

add_slide_number(s, 8)


# ============================================================================
# Slide 8: Results Bar Chart (separate slide)
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.6))
p = tb.text_frame.paragraphs[0]
p.text = 'Results — Visual Comparison'; p.font.size = Pt(28); p.font.color.rgb = WHITE; p.font.bold = True

img_path2 = f'{IMG}/model_comparison_bars.png'
if os.path.exists(img_path2):
    s.shapes.add_picture(img_path2, Inches(0.5), Inches(1.2), width=Inches(12.3), height=Inches(5.8))

add_slide_number(s, 9)


# ============================================================================
# Slide 8: Grid Search
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Loss Function Optimization'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

# Bullets on left
bullets = ('Grid search: 18 configs × 5 seeds\n\n'
           'Key findings:\n\n'
           '✓  Stronger regime supervision\n'
           '    (γ = 0.8–1.0) consistently helps\n\n'
           '✓  Mild crisis upweighting (1.5x)\n'
           '    improves performance\n\n'
           '✗  Crisis weight 3.0x hurts\n\n'
           '✗  Tail-weighted quantile loss:\n'
           '    no significant improvement\n\n'
           'Selected: γ=0.8, crisis weight=1.5x')

tb2 = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.2), Inches(5.5))
tf = tb2.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = bullets; p.font.size = Pt(14); p.font.color.rgb = LIGHT; p.line_spacing = Pt(18)

img_path = f'{IMG}/grid_search_heatmap.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(6.5), Inches(1.3), width=Inches(6.3))

add_slide_number(s, 10)


# ============================================================================
# Slide 9: Ablation
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.3), Inches(11), Inches(0.6))
p = tb.text_frame.paragraphs[0]
p.text = 'Ablation — Loss Configuration'; p.font.size = Pt(28); p.font.color.rgb = WHITE; p.font.bold = True

tb2 = s.shapes.add_textbox(Inches(0.8), Inches(0.85), Inches(11), Inches(0.4))
p2 = tb2.text_frame.paragraphs[0]
p2.text = 'Progressive improvement: stronger regime supervision is the key factor'
p2.font.size = Pt(13); p2.font.color.rgb = GRAY

img_path = f'{IMG}/ablation_results.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(0.8), Inches(1.4), width=Inches(11.7), height=Inches(5.5))

add_slide_number(s, 11)


# ============================================================================
# Slide 10: HMM Comparison
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Time-Varying HMM Transitions'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

bullets = ('Static HMM (baseline):\n'
           '  Fixed 3×3 transition matrix\n'
           '  Same probabilities in bull & bear markets\n\n'
           'Time-Varying HMM (ours):\n'
           '  Transition matrix conditioned on\n'
           '  current hidden state\n\n'
           '  → "When VIX spikes, increase\n'
           '     P(calm → stress)"\n\n'
           'Result:\n'
           '  PR-AUC: 0.319 → 0.352  (+10.3%)\n'
           '  Faster convergence (epoch 6-8)\n'
           '  Higher ceiling (seed hit 0.476)')

tb2 = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.5), Inches(5.5))
tf = tb2.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = bullets; p.font.size = Pt(14); p.font.color.rgb = LIGHT; p.line_spacing = Pt(18)

img_path = f'{IMG}/hmm_comparison.png'
if os.path.exists(img_path):
    s.shapes.add_picture(img_path, Inches(6.8), Inches(1.5), width=Inches(6))

add_slide_number(s, 12)


# ============================================================================
# Slide 11: Literature Context
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Literature Context'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

refs = [
    ('Lim et al. 2021', 'Int. J. Forecasting', 'Original TFT — our baseline architecture.\nMulti-horizon forecasting with interpretable attention.', ACCENT1),
    ('Gu et al. 2020', 'Review of Financial Studies', 'Deep learning for asset pricing. ROC-AUC 0.55–0.60\non daily data — contextualizes our results.', ACCENT2),
    ('Chatzis et al. 2018', 'J. Banking & Finance', 'LSTM for crisis prediction. Shows regime-aware\nmodels outperform vanilla approaches.', ORANGE),
    ('De Prado 2018', 'Advances in Financial ML', 'Walk-forward CV methodology. Justifies our\nevaluation approach (temporal splits, no leakage).', RGBColor(0xE0, 0x60, 0x60)),
]
for i, (author, journal, desc, color) in enumerate(refs):
    top = Inches(1.5 + i * 1.4)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top, Inches(11.7), Inches(1.2))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = color; card.line.width = Pt(1)

    tb = s.shapes.add_textbox(Inches(1.1), top + Inches(0.1), Inches(3.5), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = author; p.font.size = Pt(15); p.font.color.rgb = WHITE; p.font.bold = True

    tb2 = s.shapes.add_textbox(Inches(4.8), top + Inches(0.1), Inches(3), Inches(0.4))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = journal; p2.font.size = Pt(12); p2.font.color.rgb = color; p2.font.italic = True

    tb3 = s.shapes.add_textbox(Inches(1.1), top + Inches(0.5), Inches(11), Inches(0.7))
    tf = tb3.text_frame; tf.word_wrap = True
    p3 = tf.paragraphs[0]
    p3.text = desc; p3.font.size = Pt(12); p3.font.color.rgb = LIGHT

add_slide_number(s, 13)


# ============================================================================
# Slide 12: Next Steps
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

tb = s.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = 'Next Steps'; p.font.size = Pt(30); p.font.color.rgb = WHITE; p.font.bold = True
add_accent_line(s)

steps = [
    ('WEEK 1', 'Bloomberg Data', '6 new features:\ncredit OAS, put-call ratio,\nSKEW, MOVE, TED spread', ACCENT1),
    ('WEEK 2', 'Walk-Forward CV', '6 folds × 20 seeds\nExpanding window\nNo data leakage', ACCENT2),
    ('WEEK 3', 'Multi-Frequency', 'Daily + Weekly + Monthly\nensemble model\nEach captures different dynamics', ORANGE),
    ('WEEK 4', 'Final Thesis', 'Statistical significance\n(Wilcoxon, 20 seeds)\nPubl.-ready figures & tables', RGBColor(0xE0, 0x60, 0x60)),
]
for i, (week, title, desc, color) in enumerate(steps):
    left = Inches(0.5 + i * 3.2)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, Inches(1.5), Inches(2.9), Inches(5.0))
    card.fill.solid(); card.fill.fore_color.rgb = BG_CARD; card.line.color.rgb = color; card.line.width = Pt(1.5)

    tb = s.shapes.add_textbox(left + Inches(0.2), Inches(1.7), Inches(2.5), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = week; p.font.size = Pt(11); p.font.color.rgb = color; p.font.bold = True

    tb2 = s.shapes.add_textbox(left + Inches(0.2), Inches(2.1), Inches(2.5), Inches(0.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = title; p2.font.size = Pt(18); p2.font.color.rgb = WHITE; p2.font.bold = True

    tb3 = s.shapes.add_textbox(left + Inches(0.2), Inches(2.8), Inches(2.5), Inches(3.2))
    tf = tb3.text_frame; tf.word_wrap = True
    p3 = tf.paragraphs[0]
    p3.text = desc; p3.font.size = Pt(13); p3.font.color.rgb = LIGHT; p3.line_spacing = Pt(20)

    # Arrow between cards (except last) — place in the 0.3in gap
    if i < len(steps) - 1:
        arrow_left = left + Inches(2.95)
        arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arrow_left, Inches(3.8), Inches(0.25), Inches(0.25))
        arr.fill.solid(); arr.fill.fore_color.rgb = ACCENT1; arr.line.fill.background()

add_slide_number(s, 14)


# ============================================================================
# Slide 13: Summary
# ============================================================================
s = prs.slides.add_slide(prs.slide_layouts[6])
dark_bg(s)

line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(1.8), Inches(3), Inches(0.05))
line.fill.solid(); line.fill.fore_color.rgb = ACCENT2; line.line.fill.background()

tb = s.shapes.add_textbox(Inches(1.5), Inches(2.1), Inches(10), Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = 'Summary'; p.font.size = Pt(36); p.font.color.rgb = WHITE; p.font.bold = True

points = [
    ('✓', 'RA-TFT with time-varying HMM achieves PR-AUC 0.352', ACCENT2),
    ('✓', 'Beats vanilla TFT (0.284) and LSTM (0.306)', ACCENT2),
    ('✓', 'Key insight: stronger regime supervision = better crash prediction', ACCENT1),
    ('→', 'Next: Bloomberg features + walk-forward CV + multi-frequency ensemble', ORANGE),
]
for i, (icon, text, color) in enumerate(points):
    top = Inches(3.2 + i * 0.7)
    tb = s.shapes.add_textbox(Inches(1.5), top, Inches(0.5), Inches(0.5))
    p = tb.text_frame.paragraphs[0]
    p.text = icon; p.font.size = Pt(20); p.font.color.rgb = color; p.font.bold = True

    tb2 = s.shapes.add_textbox(Inches(2.2), top, Inches(9), Inches(0.5))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = text; p2.font.size = Pt(18); p2.font.color.rgb = LIGHT

add_slide_number(s, 15)


# ============================================================================
# Save
# ============================================================================
output_path = f'{OUT}/thesis_presentation.pptx'
prs.save(output_path)
print(f'Saved: {output_path}')
print(f'Slides: {len(prs.slides)}')
