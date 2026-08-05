"""Generate a brief PDF summary of the data actually used in the RA-TFT project.

Writes the PDF to the user's Desktop.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fpdf import FPDF

REPO = Path(__file__).resolve().parents[1]
OUT = Path.home() / "Desktop" / "ra_tft_data_summary.pdf"


def _ascii(s: str) -> str:
    return (
        s.replace("—", "--")
         .replace("–", "-")
         .replace("→", "->")
         .replace("‘", "'").replace("’", "'")
         .replace("“", '"').replace("”", '"')
         .replace("×", "x")
    )


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "RA-TFT Data Used -- Brief Summary", align="R",
                 new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(20, 20, 20)
        self.cell(0, 10, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), self.l_margin + 60, self.get_y())
        self.ln(6)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 9, _ascii(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.4)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.2, _ascii(text))
        self.ln(2)

    def kv_table(self, rows, col_widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(235, 240, 250)
        self.set_text_color(20, 20, 20)
        headers = list(rows[0].keys())
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for r in rows[1:]:
            for h, w in zip(headers, col_widths):
                self.cell(w, 6, str(r[h]), border=1)
            self.ln()
        self.ln(3)


def load_summaries():
    task_a = pd.read_parquet(REPO / "data/processed/task_a_features.parquet")
    task_b = pd.read_parquet(REPO / "data/processed/task_b_features.parquet")
    panel = pd.read_parquet(REPO / "data/processed/panel/panel_features.parquet")
    lv = pd.read_csv(REPO / "data/labels/laeven_valencia.csv")
    return task_a, task_b, panel, lv


def main():
    task_a, task_b, panel, lv = load_summaries()
    pdf = Report()
    pdf.add_page()

    # Title
    pdf.h1("RA-TFT — Data Used")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Regime-Aware Temporal Fusion Transformer for crisis prediction",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # 1. Sources
    pdf.h2("1. Data Sources (all free)")
    pdf.body(
        "- FRED API: macro indicators, Treasury yields, credit spreads, volatility (VIX, TED), "
        "housing/property, financial conditions.\n"
        "- Yahoo Finance: 12 international equity indexes (S&P 500, FTSE 100, DAX, CAC, "
        "Nikkei, Hang Seng, KOSPI, Nifty, Shanghai, TSX, ASX, Bovespa), VIX/VIX3M, "
        "MOVE, SKEW, DXY, gold, commodities.\n"
        "- BIS Statistics: quarterly global property prices (8 countries).\n"
        "- Laeven & Valencia (2018): systemic banking-crisis dates used as ground-truth labels."
    )

    # 2. Crisis labels
    pdf.h2("2. Crisis Labels (Laeven & Valencia)")
    pdf.body(
        f"42 systemic banking-crisis episodes across "
        f"{lv['country'].nunique()} countries, spanning {lv['start'].min()} to {lv['end'].max()}. "
        "Used to derive binary crisis labels and regime labels (normal / stress / crisis)."
    )

    # 3. Task A
    pdf.h2("3. Task A — Country-Month Macro Panel")
    pdf.body(
        f"Shape: {task_a.shape[0]} rows x {task_a.shape[1]} cols  |  "
        f"Date range: {task_a['date'].min().date()} -> {task_a['date'].max().date()}  |  "
        f"Countries: {task_a['country'].nunique()}\n"
        "Features (8): 10Y-2Y spread, US real rate, M2 growth, unemployment, CPI, "
        "VIX, commodity index, country code.  Target: crisis_label / regime_label."
    )

    # 4. Task B
    pdf.h2("4. Task B — Daily Single-Market Features (S&P 500)")
    pdf.body(
        f"Shape: {task_b.shape[0]} rows x {task_b.shape[1]} cols  |  "
        f"Date range: {task_b['date'].min().date()} -> {task_b['date'].max().date()}\n"
        "32 engineered daily features: log returns (1d/5d/21d), realized vol "
        "(5d/21d/63d), drawdown, vol-of-vol, return skew, Hurst exponent, Amihud "
        "illiquidity, volume z-score, VIX, VIX3M, HY/IG/TED spreads, gold, DXY, SKEW, "
        "MOVE, VIX term structure, implied-realized spread, gold/SPX ratio, "
        "DXY 5d return, credit-spread diff, VIX/SKEW ratio. "
        "Targets: ews_label, in_crisis, drawdown, regime_label."
    )

    # 5. Panel
    pdf.h2("5. International Panel — 12 Equity Markets (daily)")
    counts = panel.groupby("market_id")["date"].agg(["min", "max", "count"]).reset_index()
    pdf.body(
        f"Shape: {panel.shape[0]:,} rows x {panel.shape[1]} cols  |  "
        f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}  |  "
        f"Markets: {panel['market_id'].nunique()}"
    )
    rows = [{"Market": "Market", "Start": "Start", "End": "End", "Rows": "Rows"}]
    for _, r in counts.iterrows():
        rows.append({
            "Market": r["market_id"].upper(),
            "Start": str(r["min"].date()),
            "End": str(r["max"].date()),
            "Rows": f"{int(r['count']):,}",
        })
    pdf.kv_table(rows, col_widths=[40, 35, 35, 25])

    # 6. Splits
    pdf.h2("6. Train / Val / Test Splits")
    pdf.body(
        "Time-based splits stored under data/processed/:\n"
        "- task_a_{train,val,test}.parquet, task_b_{train,val,test}.parquet\n"
        "- Crisis-era folds under data/processed/folds/: gfc (2008), covid (2020), "
        "bear2022 — each with train/val/test parquet for out-of-sample regime testing."
    )

    # 7. What was assessed but NOT used
    pdf.h2("7. Notes on Scope")
    pdf.body(
        "A separate data-availability audit (output/data_availability_report.pdf) "
        "assessed 52 candidate items: 45 (87%) were free-tier accessible, 20 are "
        "currently in use, 25 more are available to add (Priority-1 candidates: HY/IG "
        "credit spreads, Case-Shiller HPI, NFCI, jobless claims). 21 items are "
        "Bloomberg-only (e.g. iTraxx/CDX CDS, V2X/VFTSE implied vol) and were excluded. "
        "4 series were delisted/removed (LIBOR, Shanghai composite via Yahoo, DX-Y, V2X)."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
