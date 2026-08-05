"""Generate PDF report of data availability for the RA-TFT thesis."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF


class DataReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5, "RA-TFT Data Availability Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 4.5, text)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None, highlight_col=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 7.5)
        self.set_fill_color(245, 245, 250)
        self.set_text_color(40, 40, 40)
        self.set_draw_color(220, 220, 220)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()

        # Rows
        self.set_font("Courier", "", 7)
        for row_idx, row in enumerate(rows):
            bg = row_idx % 2 == 0
            if bg:
                self.set_fill_color(252, 252, 255)
            else:
                self.set_fill_color(255, 255, 255)

            for i, cell in enumerate(row):
                self.set_text_color(40, 40, 40)
                align = "L" if i == 0 or i == 1 else "C"

                # Color the availability column
                if highlight_col is not None and i == highlight_col:
                    if "YES" in str(cell):
                        self.set_text_color(16, 125, 76)  # green
                    elif "NO" in str(cell) or "ERROR" in str(cell):
                        self.set_text_color(200, 40, 40)  # red
                    elif "PARTIAL" in str(cell):
                        self.set_text_color(180, 120, 0)  # amber

                self.cell(col_widths[i], 5, str(cell), border=1, fill=bg, align=align)
            self.ln()
        self.ln(3)

    def status_badge(self, text, color):
        r, g, b = color
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        w = self.get_string_width(text) + 6
        self.cell(w, 5, text, fill=True, align="C")
        self.set_text_color(40, 40, 40)


def main():
    pdf = DataReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title Page ──
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 15, "Data Availability Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Regime-Aware TFT for Financial Crisis Prediction", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(59, 130, 246)
    pdf.set_line_width(1)
    pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, "Generated: March 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Sources: FRED API, Yahoo Finance, BIS Statistics", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # Summary box
    pdf.set_fill_color(240, 245, 255)
    pdf.set_draw_color(59, 130, 246)
    pdf.rect(25, pdf.get_y(), pdf.w - 50, 35, style="DF")
    pdf.set_xy(30, pdf.get_y() + 5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 6, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(30)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 5, "Total items assessed: 52  |  Available free (post-2000): 45 (87%)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(30)
    pdf.cell(0, 5, "Currently using: 20  |  Can add: 25 new features", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(30)
    pdf.cell(0, 5, "Bloomberg-only (unavailable): 21 items  |  Delisted/removed: 4 items", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # ── Page 2: FRED Available ──
    pdf.add_page()
    pdf.section_title("1. FRED Series - Available Post-2000 (33 series)")

    pdf.subsection("Volatility & Credit Spreads")
    w = [30, 55, 22, 22, 12, 22]
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["VIXCLS", "VIX (daily)", "1990-01", "2026-03", "9144", "YES"],
            ["BAMLH0A0HYM2", "ICE BofA HY OAS spread", "1997-01", "2026-03", "7626", "YES"],
            ["BAMLC0A0CM", "ICE BofA IG OAS spread", "1997-01", "2026-03", "7625", "YES"],
            ["BAMLH0A0HYM2EY", "HY effective yield", "1997-01", "2026-03", "7626", "YES"],
            ["TEDRATE", "TED spread (LIBOR-TBill)", "1986-01", "2022-01", "8853", "YES*"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("Treasury Yields")
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["DGS1", "US 1Y Treasury yield", "1962-01", "2026-03", "16033", "YES"],
            ["DGS2", "US 2Y Treasury yield", "1976-06", "2026-03", "12441", "YES"],
            ["DGS5", "US 5Y Treasury yield", "1962-01", "2026-03", "16033", "YES"],
            ["DGS10", "US 10Y Treasury yield", "1962-01", "2026-03", "16033", "YES"],
            ["DGS30", "US 30Y Treasury yield", "1977-02", "2026-03", "12263", "YES"],
            ["DTB3", "US 3M T-Bill rate", "1954-01", "2026-03", "18039", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("Yield Curve & Funding")
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["T10Y2Y", "10Y-2Y spread", "1976-06", "2026-03", "12442", "YES"],
            ["T10Y3M", "10Y-3M spread", "1982-01", "2026-03", "11051", "YES"],
            ["T10YFF", "10Y minus Fed Funds", "1962-01", "2026-03", "16031", "YES"],
            ["FEDFUNDS", "Fed Funds rate (monthly)", "1954-07", "2026-02", "860", "YES"],
            ["DFF", "Fed Funds rate (daily)", "1954-07", "2026-03", "26189", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.add_page()
    pdf.subsection("Real Estate / Property")
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["CSUSHPISA", "Case-Shiller National HPI", "1987-01", "2025-12", "468", "YES"],
            ["SPCS20RSA", "Case-Shiller 20-City HPI", "2000-01", "2025-12", "312", "YES"],
            ["MORTGAGE30US", "30Y Fixed Mortgage Rate", "1971-04", "2026-03", "2868", "YES"],
            ["HOUST", "Housing Starts", "1959-01", "2026-01", "805", "YES"],
            ["PERMIT", "Building Permits", "1960-01", "2026-01", "793", "YES"],
            ["USSTHPI", "All-Transactions HPI", "1975-01", "2025-10", "204", "YES"],
            ["RHORUSQ156N", "Homeownership Rate", "1965-01", "2025-10", "244", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("BIS Global Property Prices (Quarterly)")
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["QUSN628BIS", "US Property Prices", "1970-01", "2025-07", "223", "YES"],
            ["QGBN628BIS", "UK Property Prices", "1968-04", "2025-07", "230", "YES"],
            ["QJPN628BIS", "Japan Property Prices", "1955-01", "2025-07", "283", "YES"],
            ["QDEN628BIS", "Germany Property Prices", "1970-01", "2025-07", "223", "YES"],
            ["QAUN628BIS", "Australia Property Prices", "1970-01", "2025-07", "223", "YES"],
            ["QESN628BIS", "Spain Property Prices", "1971-01", "2025-07", "219", "YES"],
            ["QIEN628BIS", "Ireland Property Prices", "1970-01", "2025-07", "223", "YES"],
            ["QCNN628BIS", "China Property Prices", "2005-04", "2025-07", "82", "NO(2005)"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("Macro Indicators")
    pdf.add_table(
        ["Series ID", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["UNRATE", "Unemployment Rate", "1948-01", "2026-02", "937", "YES"],
            ["CPIAUCSL", "CPI All Urban Consumers", "1947-01", "2026-02", "949", "YES"],
            ["PPIACO", "PPI All Commodities", "1913-01", "2026-01", "1357", "YES"],
            ["INDPRO", "Industrial Production", "1919-01", "2026-02", "1286", "YES"],
            ["UMCSENT", "Consumer Sentiment", "1952-11", "2026-01", "669", "YES"],
            ["ICSA", "Initial Jobless Claims", "1967-01", "2026-03", "3088", "YES"],
            ["NFCI", "Chicago Fed Fin. Conditions", "1971-01", "2026-03", "2879", "YES"],
            ["GDP", "GDP (quarterly)", "1947-01", "2025-10", "316", "YES"],
            ["M2SL", "M2 Money Supply", "1959-01", "2026-01", "805", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    # ── Page: Yahoo Finance ──
    pdf.add_page()
    pdf.section_title("2. Yahoo Finance - Available Post-2000 (14 tickers)")

    pdf.subsection("Equity Indexes")
    pdf.add_table(
        ["Ticker", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["^GSPC", "S&P 500", "1990-01", "2026-03", "9117", "YES"],
            ["^BKX", "KBW Bank Index", "1993-02", "2026-03", "8321", "YES"],
            ["^FTSE", "FTSE 100", "1990-01", "2026-03", "9145", "YES"],
            ["^GDAXI", "DAX", "1990-01", "2026-03", "9159", "YES"],
            ["^GSPTSE", "Canada S&P/TSX", "1990-01", "2026-03", "9101", "YES"],
            ["^N225", "Nikkei 225", "1990-01", "2026-03", "8883", "YES"],
            ["^HSI", "Hang Seng", "1990-01", "2026-03", "8934", "YES"],
            ["XLF", "Financial Select SPDR", "1998-12", "2026-03", "6848", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("Volatility")
    pdf.add_table(
        ["Ticker", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["^VIX", "CBOE VIX", "1990-01", "2026-03", "9117", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("Commodities")
    pdf.add_table(
        ["Ticker", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["GC=F", "Gold Futures", "2000-08", "2026-03", "6409", "YES"],
            ["SI=F", "Silver Futures", "2000-08", "2026-03", "6411", "YES"],
            ["CL=F", "Crude Oil Futures", "2000-08", "2026-03", "6418", "YES"],
            ["HG=F", "Copper Futures", "2000-08", "2026-03", "6414", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    pdf.subsection("FX")
    pdf.add_table(
        ["Ticker", "Description", "Start", "End", "Rows", "2000+?"],
        [
            ["USDJPY=X", "USD/JPY", "1996-10", "2026-03", "7616", "YES"],
        ],
        col_widths=w, highlight_col=5,
    )

    # ── Page: Partial / Post-2000 start ──
    pdf.add_page()
    pdf.section_title("3. Partial Availability (Starts 2001-2007)")

    pdf.body_text("These series start after 2000 but before 2008. They miss the dot-com crash (2001) "
                  "but cover the GFC (2008) and all subsequent crises. Usable with noted limitation.")

    pdf.add_table(
        ["Source", "ID", "Description", "Start", "Rows", "Note"],
        [
            ["FRED", "SOFR", "Secured Overnight Rate", "2018", "1985", "Too short - skip"],
            ["FRED", "STLFSI2", "St Louis Fin. Stress", "1994", "1463", "Ends 2022 - skip"],
            ["FRED", "QCNN628BIS", "China Property Prices", "2005", "82", "Usable (short)"],
            ["Yahoo", "^STOXX50E", "Euro Stoxx 50", "2007", "4752", "Misses pre-GFC"],
            ["Yahoo", "^STOXX", "STOXX Europe 600", "2004", "5504", "Usable"],
            ["Yahoo", "^VIX3M", "VIX 3-Month", "2006", "4947", "Misses dot-com"],
            ["Yahoo", "^VVIX", "VVIX", "2007", "4821", "Misses pre-GFC"],
            ["Yahoo", "HYG", "HY Bond ETF", "2007", "4763", "Use FRED spread"],
            ["Yahoo", "LQD", "IG Bond ETF", "2002", "5945", "Close enough"],
            ["Yahoo", "TLT", "20Y Treasury ETF", "2002", "5945", "Close enough"],
            ["Yahoo", "EURUSD=X", "EUR/USD", "2003", "5782", "Usable"],
            ["Yahoo", "GBPUSD=X", "GBP/USD", "2003", "5794", "Usable"],
            ["Yahoo", "USDCHF=X", "USD/CHF", "2003", "5848", "Usable"],
            ["Yahoo", "USDNOK=X", "USD/NOK", "2001", "6241", "Usable"],
            ["Yahoo", "KBE", "S&P Bank ETF", "2005", "5113", "Use ^BKX instead"],
            ["Yahoo", "KRE", "Regional Bank ETF", "2006", "4963", "Misses dot-com"],
        ],
        col_widths=[15, 28, 42, 15, 12, 48], highlight_col=5,
    )

    # ── Page: Bloomberg Only ──
    pdf.add_page()
    pdf.section_title("4. Bloomberg Only - NOT Available Free")

    pdf.body_text("These items require a Bloomberg Terminal subscription. No reliable free alternatives exist. "
                  "If Bloomberg access becomes available, these would significantly strengthen the analysis, "
                  "especially CDS data for credit risk pricing.")

    pdf.add_table(
        ["Category", "Item", "Description", "Free Alt?"],
        [
            ["Volatility", "V2X Index", "Euro Stoxx 50 implied vol", "NO"],
            ["Volatility", "VFTSE Index", "FTSE 100 implied vol", "NO"],
            ["Volatility", "EURUSDV1M", "FX implied vol (EUR/USD)", "NO"],
            ["Volatility", "OVDV / IVOL", "Bank equity implied vol", "NO"],
            ["Equity", "SX7E Index", "Euro Stoxx Banks", "Partial"],
            ["Equity", "FTNMX301010", "FTSE UK Banks", "NO"],
            ["CDS", "iTraxx EUR 5Y", "European IG CDS index", "NO"],
            ["CDS", "CDX IG 5Y", "US IG CDS index", "NO"],
            ["CDS", "Sovereign CDS", "Government default risk", "Partial*"],
            ["CDS", "Corporate CDS", "Company default risk", "NO"],
            ["Bank Risk", "Individual CDS", "Bank-specific default risk", "NO"],
            ["Bank Risk", "BETA/VAR/CRPR", "Systematic/tail/credit risk", "NO"],
            ["Bonds", "LF98TRUU", "Bloomberg US Corp IG Index", "NO"],
            ["Funding", "EURIBOR-OIS", "Eurozone banking stress", "NO"],
            ["Credit", "LIBOR spread", "LIBOR-based funding stress", "NO"],
            ["Yield Curve", "UK/DE curves", "Non-US yield curves (full)", "Partial"],
            ["Systemic", "NECOFs", "Network contagion factors", "NO"],
        ],
        col_widths=[22, 30, 55, 22], highlight_col=3,
    )
    pdf.body_text("* Partial: WorldGovernmentBonds.com has some sovereign CDS data but limited history.")

    # ── Page: Removed/Broken ──
    pdf.section_title("5. Removed or Broken Series")
    pdf.add_table(
        ["Source", "ID", "Issue"],
        [
            ["FRED", "USD3MTD156N", "3M LIBOR - series deleted from FRED (LIBOR discontinued 2023)"],
            ["FRED", "USDONTD156N", "Overnight LIBOR - series deleted from FRED"],
            ["Yahoo", "^SSEC", "Shanghai Composite - delisted, no timezone data"],
            ["Yahoo", "DX-Y.NYB", "US Dollar Index - no price data returned"],
            ["Yahoo", "^V2X", "Euro VIX - delisted on Yahoo Finance"],
        ],
        col_widths=[15, 30, 115],
    )

    # ── Page: Recommendation ──
    pdf.add_page()
    pdf.section_title("6. Recommendation: Features to Add")

    pdf.body_text("Priority 1 - High-impact features available immediately (FRED + Yahoo):")

    pdf.add_table(
        ["#", "Source", "ID", "Feature", "Why Critical"],
        [
            ["1", "FRED", "BAMLH0A0HYM2", "HY credit spread", "#1 crisis predictor in literature"],
            ["2", "FRED", "BAMLC0A0CM", "IG credit spread", "Credit tightening signal"],
            ["3", "FRED", "DGS2", "2Y Treasury yield", "Fed policy expectations"],
            ["4", "FRED", "DGS10", "10Y Treasury yield", "Growth/inflation signal"],
            ["5", "FRED", "CSUSHPISA", "Case-Shiller HPI", "Real estate bubble detection"],
            ["6", "FRED", "MORTGAGE30US", "30Y Mortgage rate", "Housing affordability stress"],
            ["7", "FRED", "NFCI", "Financial Conditions", "Composite stress measure"],
            ["8", "FRED", "ICSA", "Jobless Claims", "Weekly labor market stress"],
            ["9", "FRED", "UMCSENT", "Consumer Sentiment", "Sentiment collapse signal"],
            ["10", "Yahoo", "^BKX", "US Bank Index", "Banking sector health"],
            ["11", "Yahoo", "CL=F", "Crude Oil", "Global growth proxy"],
            ["12", "Yahoo", "HG=F", "Copper", "Economic activity signal"],
        ],
        col_widths=[8, 14, 30, 38, 60],
    )

    pdf.body_text("Priority 2 - International contagion & safe havens:")
    pdf.add_table(
        ["#", "Source", "ID", "Feature", "Why Useful"],
        [
            ["13", "Yahoo", "^FTSE", "FTSE 100", "UK/European crisis contagion"],
            ["14", "Yahoo", "^GDAXI", "DAX", "German/EU economic health"],
            ["15", "Yahoo", "^N225", "Nikkei 225", "Asian crisis contagion"],
            ["16", "Yahoo", "^HSI", "Hang Seng", "China/Asia exposure"],
            ["17", "Yahoo", "SI=F", "Silver", "Safe haven / industrial"],
            ["18", "FRED", "BIS series", "Global property prices", "Real estate across 7 countries"],
        ],
        col_widths=[8, 14, 25, 40, 63],
    )

    pdf.ln(5)
    pdf.body_text("Adding Priority 1 features would increase Task B from 20 to 32 features. "
                  "The HY credit spread alone is documented as the single most predictive variable "
                  "for financial crises in Gilchrist & Zakrajsek (2012) and Adrian, Boyarchenko & Giannone (2019).")

    # Save
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "data_availability_report.pdf")
    pdf.output(path)
    print(f"PDF saved to: {path}")


if __name__ == "__main__":
    main()
