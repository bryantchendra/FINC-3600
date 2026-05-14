"""
NSWDF Portfolio Dashboard — main Dash application.

Modules 1, 2, and 3 are fully implemented.
Modules 4-6 are stubs awaiting the next build slice.

Run with:
    python app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, State, dcc, html, dash_table, callback_context

from typing import Optional

from modules import trust_calcs as tc
from modules import metrics as mt
from modules import optimiser as op
from modules import stress as st
from modules import drought as dr

# ---------------------------------------------------------------------------
# Paths and historical data  (computed ONCE at startup — never reactive)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
RETURNS_PATH = DATA_DIR / "index_returns.csv"

_returns_df = pd.read_csv(RETURNS_PATH, index_col=0)
assert list(_returns_df.columns) == tc.ASSET_CLASSES, (
    "CSV column order does not match ASSET_CLASSES in trust_calcs.py"
)

# Arithmetic mean x 12  -> feeds mean-variance engine (cma-store)
HIST_ARITH_ANNUAL_RETURNS: pd.Series = _returns_df.mean() * 12
# Monthly std x sqrt(12)
HIST_ANNUAL_VOL: pd.Series = _returns_df.std() * np.sqrt(12)
# Sample correlation matrix (always PSD)
HIST_CORR: pd.DataFrame = _returns_df.corr()

# Geometric mean  -> display only (historical reference column + CFO Table 1)
def _geom_annual(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    return (1 + df).prod() ** (12 / n) - 1

HIST_GEOM_ANNUAL_RETURNS: pd.Series = _geom_annual(_returns_df)

# Descriptive statistics (monthly values, from raw data)
HIST_DESC: pd.DataFrame = _returns_df.describe(percentiles=[0.25, 0.5, 0.75]).T
HIST_DESC["skewness"] = _returns_df.skew()
HIST_DESC["kurtosis"] = _returns_df.kurt()
HIST_DESC = HIST_DESC[["mean", "std", "min", "25%", "50%", "75%", "max",
                        "skewness", "kurtosis"]]

# Pre-compute historical trust-level monthly returns for the backtest
HIST_TRUST_MONTHLY_GROSS = tc.historical_trust_returns_monthly(_returns_df)
HIST_TRUST_MONTHLY_NET   = tc.historical_trust_returns_monthly_net(_returns_df)
HIST_TRUST_CUMULATIVE_NET = tc.historical_cumulative_wealth(HIST_TRUST_MONTHLY_NET)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.{decimals}f}%"

def _fmt_signed_pct(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    sign = "+" if x >= 0 else "\u2212"
    return f"{sign}{abs(x) * 100:.{decimals}f}%"

def _fmt_aud(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:,.0f}"

ASSET_RATIONALES: dict[str, str] = {
    "Cash": "Defensive liquidity anchor; return expectation follows the RBA cash-rate path.",
    "Australian Short Duration Bond": "Low-duration income sleeve; limits mark-to-market loss if rates stay volatile.",
    "Australian Fixed Income": "Diversifies equity risk, but carries duration sensitivity during inflation shocks.",
    "Global Fixed Income (Hedged)": "Defensive global duration exposure; hedging reduces currency noise.",
    "Global Credit (Hedged)": "Adds spread income; vulnerable if recession widens credit spreads.",
    "Australian Listed Equity": "Domestic growth exposure; sensitive to earnings, rates, and commodity-cycle risk.",
    "Global Listed Equity (Unhedged)": "Global growth plus AUD diversification; benefits when AUD weakens.",
    "Global Listed Equity (Hedged)": "Global equity growth without major currency translation effects.",
    "Australian Listed Property": "Income and real-asset exposure; rate-sensitive and cyclical.",
    "Global Infrastructure (Unhedged)": "Long-duration real assets with inflation-linked cashflow characteristics.",
    "Global Private Equity": "Highest growth assumption; illiquidity and valuation lag require risk discipline.",
}

TRUST_ROLES: dict[str, str] = {
    "STI": "Liquidity reserve and capital-stability sleeve for near-term drought response.",
    "MTG": "Balanced growth sleeve that helps meet CPI+ while preserving medium-term liquidity.",
    "LTG": "Long-horizon return engine; accepts larger drawdowns to support real capital growth.",
}

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True, title="NSWDF Dashboard")
server = app.server

# ---------------------------------------------------------------------------
# Style tokens
# ---------------------------------------------------------------------------

COLORS = {
    "bg":          "#FAFAF7",
    "panel":       "#FFFFFF",
    "border":      "#E6E2D9",
    "ink":         "#1F1B16",
    "muted":       "#6B6557",
    "accent":      "#2E5E50",   # darker forest teal (used for portfolio bar)
    "hist_col":    "#E8F0EE",   # teal tint for non-editable historical columns
    "warn_bg":     "#FFF4D6",
    "warn_border": "#D4A93A",
    "warn_ink":    "#5C4400",
    "pass":        "#2E6B3F",
    "fail":        "#A23737",
    # Trust series colours — consistent across all charts
    "STI":         "#A08040",   # rich warm gold
    "MTG":         "#2E6B5E",   # deep teal
    "LTG":         "#7B3D5F",   # plum
    # Heatmap diverging
    "heat_neg":  "#A23737",
    "heat_zero": "#F5F1E6",
    "heat_pos":  "#2E5C7A",
}

# Consistent per-asset colours used across all three interactive EDA charts
ASSET_COLORS: dict[str, str] = {
    "Cash":                             "#4E9BAE",
    "Australian Short Duration Bond":   "#7BC5C5",
    "Australian Fixed Income":          "#3A6B5E",
    "Global Fixed Income (Hedged)":     "#5B8A72",
    "Global Credit (Hedged)":           "#A0C878",
    "Australian Listed Equity":         "#A08040",
    "Global Listed Equity (Unhedged)":  "#E07B54",
    "Global Listed Equity (Hedged)":    "#D4517A",
    "Australian Listed Property":       "#7B3D5F",
    "Global Infrastructure (Unhedged)": "#9B59B6",
    "Global Private Equity":            "#34495E",
}

FONT_STACK = (
    "'Source Serif 4','Source Serif Pro','Iowan Old Style',"
    "'Palatino Linotype',Palatino,Georgia,serif"
)
MONO_STACK = "'JetBrains Mono','IBM Plex Mono',Menlo,Consolas,monospace"

GLOBAL_CSS = f"""
body {{
    margin: 0;
    background: {COLORS['bg']};
    color: {COLORS['ink']};
    font-family: {FONT_STACK};
    font-size: 15px;
    line-height: 1.5;
}}
.app-container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 32px 28px 96px 28px;
}}
.app-header h1 {{
    font-family: {FONT_STACK};
    font-weight: 600;
    font-size: 32px;
    letter-spacing: -0.01em;
    margin: 0 0 6px 0;
    color: {COLORS['ink']};
}}
.app-header .subtitle {{
    color: {COLORS['muted']};
    font-size: 14px;
    margin-bottom: 24px;
}}
.panel {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 22px 24px;
    margin-bottom: 20px;
}}
.panel h2 {{
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 4px 0;
    color: {COLORS['accent']};
}}
.panel h3 {{
    font-size: 15px;
    font-weight: 600;
    margin: 18px 0 10px 0;
    color: {COLORS['ink']};
}}
.panel .section-note {{
    font-size: 13px;
    color: {COLORS['muted']};
    margin-bottom: 18px;
}}
.hist-note {{
    font-size: 12px;
    color: {COLORS['muted']};
    font-style: italic;
    margin-top: 8px;
}}
.cpi-input {{
    font-family: {MONO_STACK};
    font-size: 15px;
    padding: 6px 10px;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    width: 100px;
}}
.kv-grid {{
    display: grid;
    grid-template-columns: max-content auto;
    gap: 6px 18px;
    align-items: baseline;
}}
.kv-grid .k {{ color: {COLORS['muted']}; font-size: 13px; }}
.kv-grid .v {{ font-family: {MONO_STACK}; }}
.dash-table-container .dash-spreadsheet-container {{
    font-family: {MONO_STACK} !important;
    font-size: 13px !important;
}}
.dash-table-container th {{
    background: {COLORS['bg']} !important;
    color: {COLORS['ink']} !important;
    font-family: {FONT_STACK} !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}}
.dcc-tab {{ font-family: {FONT_STACK}; }}

/* --- Executive / rubric alignment --- */
.decision-band {{
    background: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-left: 3px solid {COLORS['accent']};
    border-radius: 4px;
    padding: 14px 16px;
    font-size: 13.5px;
    line-height: 1.55;
}}
.source-note {{
    background: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 12px 14px;
    color: {COLORS['muted']};
    font-size: 12.5px;
}}

/* --- Interactive chart controls --- */
.chart-controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px 24px;
    align-items: flex-start;
    margin-bottom: 14px;
    padding: 12px 16px;
    background: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}
.ctrl-group {{
    display: flex;
    flex-direction: column;
    gap: 6px;
}}
.ctrl-label {{
    font-size: 11px;
    font-weight: 600;
    color: {COLORS['muted']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.ctrl-btn {{
    font-size: 12px;
    padding: 2px 10px;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    background: {COLORS['panel']};
    cursor: pointer;
    color: {COLORS['accent']};
    font-family: {FONT_STACK};
}}
.ctrl-btn:hover {{ background: {COLORS['hist_col']}; }}
.era-legend {{
    display: flex;
    gap: 14px;
    align-items: center;
    font-size: 12px;
    color: {COLORS['muted']};
    margin-top: 6px;
}}
.era-swatch {{
    display: inline-block;
    width: 14px;
    height: 10px;
    border-radius: 2px;
    margin-right: 4px;
    vertical-align: middle;
    opacity: 0.7;
}}

/* --- Module 2 --- */
.trust-row {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}}
.trust-card {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-top: 3px solid var(--trust-accent);
    border-radius: 4px;
    padding: 18px 20px;
}}
.trust-card .trust-name {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--trust-accent);
    margin: 0;
}}
.trust-card .trust-tag {{
    font-size: 12px;
    color: {COLORS['muted']};
    margin: 2px 0 14px 0;
}}
.trust-card .net-return {{
    font-family: {MONO_STACK};
    font-size: 28px;
    font-weight: 600;
    color: {COLORS['ink']};
    line-height: 1;
}}
.trust-card .net-return-label {{
    font-size: 11px;
    color: {COLORS['muted']};
    margin-top: 4px;
    margin-bottom: 14px;
}}
.trust-card .stats-grid {{
    display: grid;
    grid-template-columns: max-content auto;
    column-gap: 14px;
    row-gap: 5px;
    font-size: 12.5px;
    border-top: 1px solid {COLORS['border']};
    padding-top: 12px;
}}
.trust-card .stats-grid .k {{ color: {COLORS['muted']}; }}
.trust-card .stats-grid .v {{ font-family: {MONO_STACK}; text-align: right; }}
.pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.pill-pass {{
    background: rgba(46, 107, 63, 0.12);
    color: {COLORS['pass']};
}}
.pill-fail {{
    background: rgba(162, 55, 55, 0.10);
    color: {COLORS['fail']};
}}
.target-line {{
    font-size: 11.5px;
    color: {COLORS['muted']};
    margin-top: 8px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.chart-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}}
.cfo-tables {{
    display: flex;
    flex-direction: column;
    gap: 18px;
}}
.cfo-table-title {{
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 8px 0;
    color: {COLORS['ink']};
}}
.backtest-stats-table {{ margin-top: 10px; }}
@media (max-width: 980px) {{
    .trust-row, .chart-row {{ grid-template-columns: 1fr; }}
}}

/* --- Module 3 --- */
.alloc-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    margin-bottom: 18px;
}}
.alloc-block {{
    background: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 16px 18px;
}}
.alloc-block .block-title {{
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {COLORS['accent']};
    margin: 0 0 4px 0;
}}
.alloc-block .block-note {{
    font-size: 11.5px;
    color: {COLORS['muted']};
    margin: 0 0 14px 0;
}}
.alloc-grid {{
    display: grid;
    grid-template-columns: 56px 1fr 80px;
    align-items: center;
    gap: 10px 14px;
    font-size: 13px;
}}
.alloc-grid .lbl {{ font-weight: 600; color: var(--row-color); }}
.alloc-grid .val {{ text-align: right; font-family: {MONO_STACK}; font-size: 13px; }}
.alloc-num-input {{
    width: 100%;
    padding: 5px 8px;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    font-family: {MONO_STACK};
    font-size: 13px;
    text-align: right;
}}
.alloc-sum {{
    margin-top: 10px;
    font-size: 12px;
    color: {COLORS['muted']};
    display: flex;
    justify-content: space-between;
}}
.alloc-sum-bad {{ color: {COLORS['fail']}; }}
.live-metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-top: 3px solid {COLORS['accent']};
    border-radius: 4px;
    padding: 16px 20px;
}}
.metric-block .metric-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
}}
.metric-block .metric-value {{
    font-family: {MONO_STACK};
    font-size: 22px;
    font-weight: 600;
    color: {COLORS['ink']};
    margin-top: 2px;
}}
.constraint-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    margin-top: 12px;
    font-size: 12.5px;
    border-top: 1px solid {COLORS['border']};
    padding-top: 12px;
}}
.constraint-item {{ display: flex; align-items: center; gap: 6px; }}
.opt-controls {{
    display: grid;
    grid-template-columns: 1.4fr 1fr auto;
    gap: 12px;
    align-items: end;
}}
.opt-controls label {{
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
    margin-bottom: 4px;
}}
.opt-button {{
    background: {COLORS['accent']};
    color: white;
    border: none;
    padding: 9px 18px;
    border-radius: 3px;
    cursor: pointer;
    font-family: {FONT_STACK};
    font-size: 14px;
    font-weight: 600;
}}
.opt-button:hover {{ background: #2C5446; }}
.opt-button-secondary {{
    background: {COLORS['panel']};
    color: {COLORS['accent']};
    border: 1px solid {COLORS['accent']};
}}
.opt-button-secondary:hover {{ background: rgba(58, 107, 94, 0.06); }}
.opt-result-card {{
    margin-top: 16px;
    background: {COLORS['bg']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 16px 18px;
}}
.opt-infeasible {{
    background: rgba(162, 55, 55, 0.06);
    border: 1px solid {COLORS['fail']};
    color: {COLORS['fail']};
    padding: 14px 16px;
    border-radius: 4px;
    margin-top: 16px;
    font-size: 13.5px;
}}
.tx-grid {{
    display: grid;
    grid-template-columns: max-content auto;
    column-gap: 18px;
    row-gap: 4px;
    font-size: 13px;
    margin-top: 10px;
}}
.tx-grid .k {{ color: {COLORS['muted']}; }}
.tx-grid .v {{ font-family: {MONO_STACK}; text-align: right; }}

/* --- Module 4 --- */
.scenario-meta {{
    background: {COLORS['bg']};
    border-left: 3px solid {COLORS['accent']};
    padding: 12px 16px;
    margin-top: 12px;
    margin-bottom: 16px;
    font-size: 13px;
    line-height: 1.55;
}}
.scenario-meta .meta-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
    font-weight: 600;
    margin-bottom: 4px;
}}
.factor-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 4px;
}}
.factor-table th, .factor-table td {{
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid {COLORS['border']};
}}
.factor-table th {{
    background: {COLORS['bg']};
    font-family: {FONT_STACK};
    font-weight: 600;
    font-size: 12px;
}}
.factor-table .trust-cell {{ font-weight: 600; color: var(--trust-accent); }}
.factor-table .num {{ font-family: {MONO_STACK}; text-align: right; }}
.factor-tag {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(58, 107, 94, 0.10);
    color: {COLORS['accent']};
}}
.factor-tag-Currency {{ background: rgba(123,61,95,0.12); color: {COLORS['LTG']}; }}
.factor-tag-Duration {{ background: rgba(194,160,96,0.18); color: #6B5320; }}
.factor-tag-Equity-beta {{ background: rgba(58,107,94,0.12); color: {COLORS['accent']}; }}
.factor-tag-Credit-spread {{ background: rgba(162,55,55,0.10); color: {COLORS['fail']}; }}

/* --- Module 5 --- */
.drought-controls {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 8px;
}}
.drought-control {{ display: flex; flex-direction: column; }}
.drought-control label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
    margin-bottom: 4px;
}}
.summary-card {{
    background: {COLORS['panel']};
    border: 1px solid {COLORS['border']};
    border-left: 3px solid {COLORS['accent']};
    border-radius: 4px;
    padding: 18px 20px;
    margin-top: 14px;
}}
.summary-card .summary-title {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
    margin: 0 0 10px 0;
}}
.summary-card .summary-headline {{
    font-family: {MONO_STACK};
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 6px;
}}
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 12px;
}}
.summary-item .lbl {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {COLORS['muted']};
}}
.summary-item .val {{
    font-family: {MONO_STACK};
    font-size: 16px;
    font-weight: 600;
    color: {COLORS['ink']};
    margin-top: 2px;
}}
.summary-verdict {{
    margin-top: 14px;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 13.5px;
}}
.summary-verdict-pass {{
    background: rgba(46, 107, 63, 0.08);
    border: 1px solid {COLORS['pass']};
    color: {COLORS['pass']};
}}
.summary-verdict-fail {{
    background: rgba(162, 55, 55, 0.06);
    border: 1px solid {COLORS['fail']};
    color: {COLORS['fail']};
}}
"""

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{GLOBAL_CSS}</style>
</head>
<body>
    <div class="app-container">
        {{%app_entry%}}
    </div>
    <footer></footer>
    {{%config%}}
    {{%scripts%}}
    {{%renderer%}}
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Initial CMA state helpers
# ---------------------------------------------------------------------------

def _initial_cma_rv_data() -> list[dict]:
    """
    Four columns per row:
      hist_return     geometric annual return (display only, non-editable)
      hist_vol        historical annual vol    (display only, non-editable)
      expected_return arithmetic annual return (editable, feeds cma-store)
      volatility      annual vol               (editable, feeds cma-store)
    """
    rows = []
    for ac in tc.ASSET_CLASSES:
        h_ret = round(float(HIST_GEOM_ANNUAL_RETURNS[ac]) * 100, 3)
        f_ret = round(float(HIST_ARITH_ANNUAL_RETURNS[ac]) * 100, 3)
        rows.append({
            "asset_class":     ac,
            "hist_return":     h_ret,
            "hist_vol":        round(float(HIST_ANNUAL_VOL[ac]) * 100, 3),
            "delta":           round(f_ret - h_ret, 3),
            "expected_return": f_ret,
            "volatility":      round(float(HIST_ANNUAL_VOL[ac]) * 100, 3),
        })
    return rows


def _initial_cma_store() -> dict:
    """
    cma-store schema (consumed by Modules 2-6 — do not change):
      returns      list[float]  11 arithmetic annual returns, DECIMAL
      vols         list[float]  11 annual volatilities, DECIMAL
      corr         list[list]   11x11 PSD correlation matrix (fixed historical)
      cpi          float        CPI assumption, DECIMAL
      psd_adjusted bool         always False
    """
    return {
        "returns":      HIST_ARITH_ANNUAL_RETURNS.tolist(),
        "vols":         HIST_ANNUAL_VOL.tolist(),
        "corr":         HIST_CORR.values.tolist(),
        "cpi":          0.025,
        "psd_adjusted": False,
    }

# ---------------------------------------------------------------------------
# Static EDA figures (built once at startup — never reactive)
# ---------------------------------------------------------------------------

CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F7F7F5",
    font=dict(family=FONT_STACK, size=12, color=COLORS["ink"]),
    margin=dict(l=56, r=24, t=40, b=48),
)

SHORT_LABELS = list(tc.ASSET_CLASS_SHORT.values())
_dates = pd.to_datetime(_returns_df.index, format="%b %Y")

# Date range helpers for period selectors
_DATE_MIN_Y = int(_dates.min().year)
_DATE_MIN_M = int(_dates.min().month)
_DATE_MAX_Y = int(_dates.max().year)
_DATE_MAX_M = int(_dates.max().month)

# Realized calendar-year geometric returns (used by annualised chart mode)
_returns_df_dt = _returns_df.copy()
_returns_df_dt.index = _dates
_annual_returns_df = (
    (_returns_df_dt + 1)
    .groupby(_returns_df_dt.index.year)
    .prod() - 1
)  # index = int year, columns = ASSET_CLASSES
_ANNUAL_YEARS = list(_annual_returns_df.index.astype(int))
_YEAR_ONLY_OPTIONS = [{"label": str(y), "value": y} for y in _ANNUAL_YEARS]

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_OPTIONS = [{"label": m, "value": i + 1} for i, m in enumerate(_MONTH_NAMES)]
_YEAR_OPTIONS  = [{"label": str(y), "value": y}
                  for y in range(_DATE_MIN_Y, _DATE_MAX_Y + 1)]

# ---------------------------------------------------------------------------
# Macro indicators  (Jan 2006 – Feb 2026, aligned to returns DatetimeIndex)
# ---------------------------------------------------------------------------
MACRO_PATH = DATA_DIR / "macro_indicators.csv"
_macro_raw = pd.read_csv(MACRO_PATH)
_macro_raw.index = pd.to_datetime(_macro_raw["Date"], format="%b %Y")
_macro_raw = _macro_raw.drop(columns=["Date"])
_macro_df: pd.DataFrame = _macro_raw.reindex(_dates).ffill()

# Derived series for correlation analysis
_macro_df["AUD/USD Δ MoM %"]    = _macro_df["AUD/USD"].pct_change() * 100   # monthly FX return
_macro_df["Fed Funds Δ MoM pp"] = _macro_df["Fed Funds Rate (%)"].diff()    # rate change in pp
_macro_df["RBA Rate Δ MoM pp"]  = _macro_df["RBA Rate (%)"].diff()          # RBA rate change in pp

# column name → short display label for the correlation heatmap x-axis
_MACRO_CORR_VARS: dict[str, str] = {
    "AUD/USD Δ MoM %":    "AUD/USD\nΔ MoM",
    "AUS CPI (YoY %)":    "AUS CPI\nYoY %",
    "US CPI (YoY %)":     "US CPI\nYoY %",
    "RBA Rate Δ MoM pp":  "RBA Rate\nΔ MoM",
    "Fed Funds Δ MoM pp": "Fed Funds\nΔ MoM",
}

# Fixed asset groupings for domicile regime analysis
_AUS_ASSETS: list[str] = [
    "Cash",
    "Australian Short Duration Bond",
    "Australian Fixed Income",
    "Australian Listed Equity",
    "Australian Listed Property",
]
_GLOBAL_ASSETS: list[str] = [
    "Global Fixed Income (Hedged)",
    "Global Credit (Hedged)",
    "Global Listed Equity (Unhedged)",
    "Global Listed Equity (Hedged)",
    "Global Infrastructure (Unhedged)",
    "Global Private Equity",
]

# Historical era shading definitions
_ERA_SHADES = [
    dict(x0="2007-11-01", x1="2009-06-30",
         fillcolor="#FFE8B2", opacity=0.35, label="GFC"),
    dict(x0="2020-02-01", x1="2020-04-30",
         fillcolor="#B2D6F5", opacity=0.35, label="COVID-19"),
]

# Rolling-vol variant: GFC shifted +1 month, COVID unchanged.
_ERA_SHADES_ROLLING = [
    dict(x0="2007-12-01", x1="2009-07-31",
         fillcolor="#FFE8B2", opacity=0.35, label="GFC"),
    dict(x0="2020-02-01", x1="2020-04-30",
         fillcolor="#B2D6F5", opacity=0.35, label="COVID-19"),
]


def _add_era_shading(fig: go.Figure,
                     start_m: int = 1,  start_y: int = 2000,
                     end_m:   int = 12, end_y:   int = 2100,
                     annual_mode: bool = False,
                     era_shades: list | None = None) -> None:
    """
    Overlay era shaded bands only where they intersect the selected period.
    Pass era_shades to override the default _ERA_SHADES (e.g. rolling-vol
    variant with the GFC window shifted +12 months).
    In annual_mode the x-axis carries integer years so vrect boundaries are
    shifted by ±0.45 to straddle the year markers.
    """
    period_start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
    period_end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=28)

    for era in (era_shades if era_shades is not None else _ERA_SHADES):
        era_start = pd.Timestamp(era["x0"])
        era_end   = pd.Timestamp(era["x1"])
        # Skip eras that fall entirely outside the selected window
        if era_end < period_start or era_start > period_end:
            continue
        if annual_mode:
            x0 = era_start.year - 0.45
            x1 = era_end.year   + 0.45
        else:
            x0 = era["x0"]
            x1 = era["x1"]
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=era["fillcolor"], opacity=era["opacity"],
            layer="below", line_width=0,
            annotation_text=era["label"],
            annotation_position="top left",
            annotation=dict(font_size=10, font_color=COLORS["muted"],
                            showarrow=False),
        )


def _filter_dates(start_m: int, start_y: int,
                  end_m: int, end_y: int) -> pd.Series:
    """Boolean mask over _dates for the requested month-year range."""
    start = pd.Timestamp(year=int(start_y), month=int(start_m), day=1)
    end   = pd.Timestamp(year=int(end_y),   month=int(end_m),   day=28)
    return (_dates >= start) & (_dates <= end)


# ---------------------------------------------------------------------------
# Dynamic figure builders (called from callbacks)
# ---------------------------------------------------------------------------

def _build_returns_time_fig(selected_assets: list, return_mode: str,
                             start_m: int, start_y: int,
                             end_m: int, end_y: int) -> go.Figure:
    fig = go.Figure()

    if return_mode == "annualised":
        # Realized calendar-year geometric returns filtered to selected period
        sy, ey = int(start_y), int(end_y)
        ann_slice = _annual_returns_df[
            (_annual_returns_df.index >= sy) & (_annual_returns_df.index <= ey)
        ]
        _add_era_shading(fig, start_m, start_y, end_m, end_y, annual_mode=True)
        for ac in tc.ASSET_CLASSES:
            if ac not in selected_assets:
                continue
            fig.add_trace(go.Scatter(
                x=ann_slice.index.tolist(),
                y=ann_slice[ac].tolist(),
                mode="lines+markers",
                name=tc.ASSET_CLASS_SHORT[ac],
                line=dict(width=1.5, color=ASSET_COLORS[ac]),
                marker=dict(size=5),
                hovertemplate=(f"<b>{tc.ASSET_CLASS_SHORT[ac]}</b><br>"
                               "%{x}: %{y:.2%}<extra></extra>"),
            ))
        fig.update_layout(
            **CHART_LAYOUT,
            title=f"Realized Annual Returns ({sy}–{ey}, Calendar Year Geometric)",
            xaxis_title="Year",
            xaxis=dict(tickmode="linear", dtick=1, tickangle=-45,
                       range=[sy - 0.6, ey + 0.6]),
            yaxis_title="Annual Return",
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
            height=460,
        )
    else:
        # Monthly returns — filtered to period, with era shading
        mask = _filter_dates(start_m, start_y, end_m, end_y)
        dates_slice = _dates[mask]
        df_slice = _returns_df_dt.loc[mask]
        _add_era_shading(fig, start_m, start_y, end_m, end_y, annual_mode=False)
        for ac in tc.ASSET_CLASSES:
            if ac not in selected_assets:
                continue
            fig.add_trace(go.Scatter(
                x=dates_slice, y=df_slice[ac], mode="lines",
                name=tc.ASSET_CLASS_SHORT[ac],
                line=dict(width=1.2, color=ASSET_COLORS[ac]),
                hovertemplate=(f"<b>{tc.ASSET_CLASS_SHORT[ac]}</b><br>"
                               "%{x|%b %Y}: %{y:.2%}<extra></extra>"),
            ))
        fig.update_layout(
            **CHART_LAYOUT,
            title="Monthly Returns by Asset Class",
            xaxis_title="Month",
            yaxis_title="Monthly Return",
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", y=-0.30, font=dict(size=11)),
            height=460,
        )
    return fig


def _build_cumulative_fig(selected_assets: list,
                           start_m: int, start_y: int,
                           end_m: int, end_y: int) -> go.Figure:
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice    = _returns_df.loc[mask]
    dates_slice = _dates[mask]
    if df_slice.empty:
        return go.Figure()
    cum = (1 + df_slice).cumprod()
    fig = go.Figure()
    for ac in tc.ASSET_CLASSES:
        if ac not in selected_assets:
            continue
        fig.add_trace(go.Scatter(
            x=dates_slice, y=cum[ac], mode="lines",
            name=tc.ASSET_CLASS_SHORT[ac],
            line=dict(width=1.5, color=ASSET_COLORS[ac]),
            hovertemplate=(f"<b>{tc.ASSET_CLASS_SHORT[ac]}</b><br>"
                           "%{x|%b %Y}: $%{y:.3f}<extra></extra>"),
        ))
    _add_era_shading(fig, start_m, start_y, end_m, end_y)
    fig.update_layout(
        **CHART_LAYOUT, title="Cumulative Returns (Growth of $1)",
        xaxis_title="Month", yaxis_title="Growth of $1",
        legend=dict(orientation="h", y=-0.30, font=dict(size=11)), height=460,
    )
    return fig


def _build_rolling_vol_fig(selected_assets: list,
                            start_m: int, start_y: int,
                            end_m: int, end_y: int) -> go.Figure:
    rv_full = _returns_df.rolling(12).std() * np.sqrt(12)
    mask        = _filter_dates(start_m, start_y, end_m, end_y)
    rv_slice    = rv_full.loc[mask]
    dates_slice = _dates[mask]
    if rv_slice.empty:
        return go.Figure()
    fig = go.Figure()
    for ac in tc.ASSET_CLASSES:
        if ac not in selected_assets:
            continue
        fig.add_trace(go.Scatter(
            x=dates_slice, y=rv_slice[ac], mode="lines",
            name=tc.ASSET_CLASS_SHORT[ac],
            line=dict(width=1.2, color=ASSET_COLORS[ac]),
            hovertemplate=(f"<b>{tc.ASSET_CLASS_SHORT[ac]}</b><br>"
                           "%{x|%b %Y}: %{y:.2%}<extra></extra>"),
        ))
    _add_era_shading(fig, start_m, start_y, end_m, end_y,
                     era_shades=_ERA_SHADES_ROLLING)
    fig.update_layout(
        **CHART_LAYOUT, title="12-Month Rolling Annualised Volatility",
        xaxis_title="Month", yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=-0.30, font=dict(size=11)), height=460,
    )
    return fig


def _build_desc_stats_data(start_m: int, start_y: int,
                            end_m: int, end_y: int) -> list[dict]:
    """Return row dicts for the descriptive-stats DataTable over the chosen period."""
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df_dt.loc[mask]
    if df_slice.empty:
        return []
    desc = df_slice.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["skewness"] = df_slice.skew()
    desc["kurtosis"] = df_slice.kurt()
    desc = desc[["mean", "std", "min", "25%", "50%", "75%", "max",
                 "skewness", "kurtosis"]]
    df = desc.reset_index().rename(columns={"index": "Asset Class"})
    # % columns stay as decimals — the DataTable format spec ".1%" handles display
    # skewness and kurtosis are pure numbers — round to 2 dp
    for col in ["skewness", "kurtosis"]:
        df[col] = df[col].round(2)
    return df.to_dict("records")


def _build_histograms_fig(start_m: int, start_y: int,
                           end_m: int, end_y: int) -> go.Figure:
    """Monthly return distribution histograms for the chosen period."""
    mask     = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df_dt.loc[mask]
    if df_slice.empty:
        return go.Figure()

    n     = len(tc.ASSET_CLASSES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols   # 4 rows for 11 assets

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=SHORT_LABELS,
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
    )

    for idx, ac in enumerate(tc.ASSET_CLASSES):
        row = idx // ncols + 1
        col = idx %  ncols + 1
        fig.add_trace(
            go.Histogram(
                x=df_slice[ac],
                nbinsx=30,
                marker_color=ASSET_COLORS[ac],
                opacity=0.8,
                showlegend=False,
                hovertemplate="Return: %{x:.1%}<br>Count: %{y}<extra></extra>",
            ),
            row=row, col=col,
        )

    fig.update_xaxes(tickformat=".1%", tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#F7F7F5",
        font=dict(family=FONT_STACK, size=11, color=COLORS["ink"]),
        height=nrows * 300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def _build_corr_heatmap_eda_fig(start_m: int, start_y: int,
                                  end_m: int, end_y: int) -> go.Figure:
    """EDA correlation heatmap recomputed for the chosen period."""
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df_dt.loc[mask]
    if df_slice.empty:
        return go.Figure()
    z = df_slice.corr().values
    corr_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
    period_label = (f"{_MONTH_NAMES[int(start_m)-1]} {start_y} – "
                    f"{_MONTH_NAMES[int(end_m)-1]} {end_y}")
    fig = go.Figure(go.Heatmap(
        z=z, x=SHORT_LABELS, y=SHORT_LABELS,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        colorbar=dict(title="Correlation", thickness=14),
    ))
    fig.update_layout(
        **corr_layout,
        title=f"Correlation Matrix — {period_label}",
        height=540,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=110, r=20, t=50, b=110),
    )
    return fig


# ---------------------------------------------------------------------------
# UI helper components for the interactive chart panels
# ---------------------------------------------------------------------------

def _asset_checklist(checklist_id: str, all_btn_id: str, none_btn_id: str) -> html.Div:
    options = [{"label": f" {tc.ASSET_CLASS_SHORT[ac]}", "value": ac}
               for ac in tc.ASSET_CLASSES]
    return html.Div([
        html.Div([
            html.Button("All",  id=all_btn_id,  n_clicks=0, className="ctrl-btn",
                        style={"marginRight": "6px"}),
            html.Button("None", id=none_btn_id, n_clicks=0, className="ctrl-btn"),
        ], style={"marginBottom": "5px"}),
        dcc.Checklist(
            id=checklist_id,
            options=options,
            value=tc.ASSET_CLASSES[:],
            inline=True,
            labelStyle={"fontSize": "12px", "marginRight": "14px",
                        "cursor": "pointer"},
            inputStyle={"marginRight": "4px",
                        "accentColor": COLORS["accent"]},
        ),
    ])


def _date_range_row(prefix: str,
                    def_sm: int, def_sy: int,
                    def_em: int, def_ey: int) -> html.Div:
    dd_style = {"display": "inline-block", "verticalAlign": "middle"}
    lbl_style = {"fontSize": "13px", "color": COLORS["muted"],
                 "marginRight": "5px", "verticalAlign": "middle"}
    return html.Div([
        html.Span("From:", style=lbl_style),
        dcc.Dropdown(id=f"{prefix}-start-m", options=_MONTH_OPTIONS,
                     value=def_sm, clearable=False,
                     style={**dd_style, "width": "78px"}),
        html.Span("", style={"display": "inline-block", "width": "5px"}),
        dcc.Dropdown(id=f"{prefix}-start-y", options=_YEAR_OPTIONS,
                     value=def_sy, clearable=False,
                     style={**dd_style, "width": "88px"}),
        html.Span("  To:", style={**lbl_style, "marginLeft": "14px"}),
        dcc.Dropdown(id=f"{prefix}-end-m", options=_MONTH_OPTIONS,
                     value=def_em, clearable=False,
                     style={**dd_style, "width": "78px"}),
        html.Span("", style={"display": "inline-block", "width": "5px"}),
        dcc.Dropdown(id=f"{prefix}-end-y", options=_YEAR_OPTIONS,
                     value=def_ey, clearable=False,
                     style={**dd_style, "width": "88px"}),
    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
              "gap": "4px"})


def _build_scatter_fig(selected_assets: list,
                       start_m: int, start_y: int,
                       end_m: int, end_y: int) -> go.Figure:
    """Risk-Return scatter for a chosen sub-period."""
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df_dt.loc[mask]
    if df_slice.empty:
        return go.Figure()
    n = len(df_slice)
    ann_ret  = (((1 + df_slice).prod() ** (12 / n)) - 1) * 100   # geometric
    ann_vol  = df_slice.std() * np.sqrt(12) * 100
    cash_ret = float(ann_ret.iloc[0])
    sharpes  = (ann_ret - cash_ret) / ann_vol

    fig = go.Figure()
    for ac in tc.ASSET_CLASSES:
        if ac not in selected_assets:
            continue
        fig.add_trace(go.Scatter(
            x=[float(ann_vol[ac])], y=[float(ann_ret[ac])],
            mode="markers+text", name=tc.ASSET_CLASS_SHORT[ac],
            text=[tc.ASSET_CLASS_SHORT[ac]], textposition="top center",
            textfont=dict(size=10, color=ASSET_COLORS[ac]),
            marker=dict(size=10, color=ASSET_COLORS[ac]), showlegend=False,
            hovertemplate=(
                f"<b>{ac}</b><br>"
                f"Period: {_MONTH_NAMES[int(start_m)-1]} {start_y}–"
                f"{_MONTH_NAMES[int(end_m)-1]} {end_y}<br>"
                "Return: %{y:.2f}%<br>"
                "Vol: %{x:.2f}%<br>"
                f"Sharpe: {float(sharpes[ac]):.2f}"
                "<extra></extra>"
            ),
        ))
    period_label = (f"{_MONTH_NAMES[int(start_m)-1]} {start_y} – "
                    f"{_MONTH_NAMES[int(end_m)-1]} {end_y}")
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"Risk-Return by Asset Class ({period_label}, Geometric Return)",
        xaxis_title="Annualised Volatility (%)",
        yaxis_title="Annualised Return (%)",
        height=540,
    )
    return fig


def _fig_corr_heatmap_eda() -> go.Figure:
    """Static EDA heatmap for Module 1 — uses CHART_LAYOUT margin override."""
    z = HIST_CORR.values
    corr_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
    fig = go.Figure(go.Heatmap(
        z=z, x=SHORT_LABELS, y=SHORT_LABELS,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2), texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        colorbar=dict(title="Correlation", thickness=14),
    ))
    fig.update_layout(**corr_layout,
        title="Correlation Matrix — Historical Asset Class Returns",
        height=540,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        margin=dict(l=110, r=20, t=50, b=110))
    return fig


def _fig_histograms() -> go.Figure:
    n = len(tc.ASSET_CLASSES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=SHORT_LABELS,
                        horizontal_spacing=0.08, vertical_spacing=0.10)
    for idx, ac in enumerate(tc.ASSET_CLASSES):
        r, c = divmod(idx, ncols)
        fig.add_trace(go.Histogram(x=_returns_df[ac], nbinsx=30,
            marker_color=COLORS["accent"], opacity=0.8, showlegend=False,
            hovertemplate="Return: %{x:.4f}<br>Count: %{y}<extra></extra>"),
            row=r + 1, col=c + 1)
    fig.update_xaxes(tickformat=".3f", tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#F7F7F5",
        font=dict(family=FONT_STACK, size=11, color=COLORS["ink"]),
        title="Monthly Return Distributions by Asset Class",
        height=nrows * 210, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# All EDA figures are now fully reactive — no static pre-builds needed

# ---------------------------------------------------------------------------
# Descriptive statistics table
# ---------------------------------------------------------------------------

def _desc_stats_table() -> dash_table.DataTable:
    df = HIST_DESC.copy().reset_index()
    df = df.rename(columns={"index": "Asset Class"})
    # % columns stay as decimals; skew/kurt rounded to 2 dp
    for col in ["skewness", "kurtosis"]:
        df[col] = df[col].round(2)

    # (col_id, display_name, d3_format_spec)
    col_specs = [
        ("mean",     "Mean",      ".1%"),
        ("std",      "Std Dev",   ".1%"),
        ("min",      "Min",       ".1%"),
        ("25%",      "25th %",    ".1%"),
        ("50%",      "Median",    ".1%"),
        ("75%",      "75th %",    ".1%"),
        ("max",      "Max",       ".1%"),
        ("skewness", "Skewness",  ".2f"),
        ("kurtosis", "Kurtosis",  ".2f"),
    ]

    col_defs = [{"name": "Asset Class", "id": "Asset Class", "editable": False}]
    for col_id, col_name, fmt in col_specs:
        col_defs.append({"name": col_name, "id": col_id, "type": "numeric",
                         "format": {"specifier": fmt}, "editable": False})

    highlight = [{"if": {"column_id": col_id},
                  "backgroundColor": COLORS["hist_col"]}
                 for col_id in ("mean", "std")]

    return dash_table.DataTable(
        id="desc-stats-table", columns=col_defs,
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"padding": "7px 10px", "fontFamily": MONO_STACK,
                    "fontSize": "12px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "Asset Class"},
             "fontFamily": FONT_STACK, "textAlign": "left", "minWidth": "220px"}],
        style_header={"borderBottom": f"2px solid {COLORS['border']}",
                      "fontFamily": FONT_STACK, "fontWeight": "600", "fontSize": "12px"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
        style_data_conditional=highlight, editable=False,
    )

# ---------------------------------------------------------------------------
# CMA input table
# ---------------------------------------------------------------------------

_HIST_GREY   = "#EBEBEB"   # light grey for historical reference columns
_HIST_GREY_H = "#DCDCDC"   # slightly darker for header

# Delta column colour bands (values are % p.a.)
# light: |Δ| < 0.5%  |  mild: 0.5% ≤ |Δ| < 1.5%  |  dark: |Δ| ≥ 1.5%
# Positive → warm gold family (STI palette)  |  Negative → plum family (LTG palette)
_DELTA_STYLES: list[dict] = [
    # ── Positive delta (forecast > hist) — teal green shades ────────────────
    {"if": {"filter_query": "{delta} >= 0 && {delta} < 0.5",    "column_id": "delta"},
     "backgroundColor": "#C8E8E0", "color": "#1A3F38"},   # light   0–0.5%
    {"if": {"filter_query": "{delta} >= 0.5 && {delta} < 1.5",  "column_id": "delta"},
     "backgroundColor": "#5A9E8F", "color": "#0D2820"},   # mild  0.5–1.5%
    {"if": {"filter_query": "{delta} >= 1.5",                    "column_id": "delta"},
     "backgroundColor": "#2E6B5E", "color": "#FFFFFF"},   # dark   ≥ 1.5%
    # ── Negative delta (forecast < hist) — plum shades ───────────────────────
    {"if": {"filter_query": "{delta} < 0 && {delta} > -0.5",    "column_id": "delta"},
     "backgroundColor": "#E8D4DF", "color": "#4A1A30"},   # light   0–0.5%
    {"if": {"filter_query": "{delta} <= -0.5 && {delta} > -1.5","column_id": "delta"},
     "backgroundColor": "#B07090", "color": "#2A0018"},   # mild  0.5–1.5%
    {"if": {"filter_query": "{delta} <= -1.5",                   "column_id": "delta"},
     "backgroundColor": "#7B3D5F", "color": "#FFFFFF"},   # dark   ≥ 1.5%
]


def _cma_rv_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        id="cma-rv-table",
        columns=[
            {"name": "Asset Class",                         "id": "asset_class",
             "editable": False},
            {"name": "Hist. Return % p.a. (Geometric)",     "id": "hist_return",
             "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
            {"name": "Hist. Vol % p.a.",                    "id": "hist_vol",
             "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
            {"name": "Forecast Return % p.a.",               "id": "expected_return",
             "type": "numeric", "format": {"specifier": ".1f"}, "editable": True},
            {"name": "Forecast Vol % p.a.",                  "id": "volatility",
             "type": "numeric", "format": {"specifier": ".1f"}, "editable": True},
            {"name": "Δ Return (% p.a.)",                    "id": "delta",
             "type": "numeric", "format": {"specifier": "+.1f"}, "editable": False},
        ],
        data=_initial_cma_rv_data(),
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px 10px", "fontFamily": MONO_STACK, "fontSize": "13px"},
        style_cell_conditional=[
            {"if": {"column_id": "asset_class"},
             "fontFamily": FONT_STACK, "textAlign": "left", "minWidth": "240px"},
            {"if": {"column_id": "hist_return"},
             "textAlign": "right", "minWidth": "200px",
             "backgroundColor": _HIST_GREY},
            {"if": {"column_id": "hist_vol"},
             "textAlign": "right", "minWidth": "140px",
             "backgroundColor": _HIST_GREY},
            {"if": {"column_id": "delta"},
             "textAlign": "right", "minWidth": "185px",
             "fontWeight": "600"},
            {"if": {"column_id": "expected_return"},
             "textAlign": "right", "minWidth": "170px"},
            {"if": {"column_id": "volatility"},
             "textAlign": "right", "minWidth": "140px"},
        ],
        style_header_conditional=[
            {"if": {"column_id": "hist_return"}, "backgroundColor": _HIST_GREY_H},
            {"if": {"column_id": "hist_vol"},    "backgroundColor": _HIST_GREY_H},
            {"if": {"column_id": "delta"},       "backgroundColor": "#F5F1E6"},
        ],
        style_header={"borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
        style_data_conditional=_DELTA_STYLES,
        editable=False,
        tooltip_header={
            "hist_return":     "Geometric annualised return for the selected analysis period (read-only).",
            "hist_vol":        "Historical annualised volatility for the selected period (read-only).",
            "delta":           "Δ = Forecast Return − Hist. Return (% p.a.). Green = above history; Red = below. Shade intensity = magnitude (light 0–500 bps, mild 500–1000 bps, dark > 1000 bps).",
            "expected_return": "Your 10-year forward-looking return forecast (arithmetic). Feeds all downstream modules.",
            "volatility":      "Your 10-year forward-looking volatility forecast.",
        },
    )


def _asset_rationale_table() -> dash_table.DataTable:
    return dash_table.DataTable(
        columns=[
            {"name": "Asset Class", "id": "asset_class"},
            {"name": "Decision-useful forecast rationale", "id": "rationale"},
        ],
        data=[
            {"asset_class": ac, "rationale": ASSET_RATIONALES[ac]}
            for ac in tc.ASSET_CLASSES
        ],
        style_table={"overflowX": "auto"},
        style_cell={
            "padding": "8px 10px",
            "fontFamily": FONT_STACK,
            "fontSize": "12.5px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_cell_conditional=[
            {"if": {"column_id": "asset_class"}, "fontWeight": "600", "minWidth": "260px"},
            {"if": {"column_id": "rationale"}, "minWidth": "460px"},
        ],
        style_header={
            "backgroundColor": COLORS["bg"],
            "fontFamily": FONT_STACK,
            "fontWeight": "600",
            "fontSize": "12px",
            "borderBottom": f"2px solid {COLORS['border']}",
        },
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
    )

# ---------------------------------------------------------------------------
# Module 1 layout
# ---------------------------------------------------------------------------

def module_1_layout() -> html.Div:
    date_range = f"{_returns_df.index[0]} to {_returns_df.index[-1]}"
    return html.Div([
        # CMA Input panel
        html.Div([
            html.H2("Module 1 — Capital Market Assumptions"),
            html.Div(
                "Grey columns show historical figures for the selected period (read-only). "
                "White Forecast columns are editable — enter your 10-year forward views. "
                "The Δ column shows Forecast Return minus Historical Return; "
                "green = above history, red = below. "
                "Only the Forecast columns feed downstream modules.",
                className="section-note",
            ),
            # ── Global controls: Analysis Period + CPI side by side ──────────
            html.Div([
                html.Div([
                    html.Div(
                        "Analysis Period  —  historical reference columns and all "
                        "EDA charts update to this window",
                        className="ctrl-label",
                    ),
                    _date_range_row("m1",
                                    _DATE_MIN_M, _DATE_MIN_Y,
                                    _DATE_MAX_M, _DATE_MAX_Y),
                ], className="ctrl-group"),
                html.Div(style={"width": "1px", "background": COLORS["border"],
                                "alignSelf": "stretch", "margin": "0 4px"}),
                html.Div([
                    html.Div("CPI Assumption", className="ctrl-label"),
                    html.Div([
                        dcc.Input(id="cpi-input", type="number", value=2.5,
                                  step=0.1, min=0, max=20, className="cpi-input",
                                  style={"width": "80px"}),
                        html.Span(" % p.a.", style={"marginLeft": "6px",
                                                     "color": COLORS["muted"],
                                                     "fontSize": "13px"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginTop": "4px"}),
                    html.Div("CPI + 2.5% p.a. fund target",
                             style={"fontSize": "11px", "color": COLORS["muted"],
                                    "marginTop": "4px"}),
                ], className="ctrl-group"),
            ], className="chart-controls", style={"marginBottom": "16px"}),
            html.Div([
                html.Div([
                    _cma_rv_table(),
                    html.Div(
                        "Grey = historical reference (read-only). "
                        "White = forecast inputs (editable — Forecast Return &amp; Forecast Vol). "
                        "Historical Return uses geometric compounding; "
                        "Forecast Return uses arithmetic convention for mean-variance calculations.",
                        className="hist-note",
                    ),
                ]),
            ]),
        ], className="panel"),

        html.Div([
            html.H2("Forecast rationale and source guardrails"),
            html.Div(
                "Use this panel to convert statistical outputs into the qualitative "
                "justification the CFO brief rubric asks for. Keep final written "
                "statements concise and cite the sources used for each judgement.",
                className="section-note",
            ),
            _asset_rationale_table(),
            html.Div(
                "Core source trail: Refinitiv monthly proxy returns for asset-class "
                "risk, return, and correlation; Abercrombie FM IM for fixed trust "
                "weights, fees, spreads, and liquidity features; Board Policy for "
                "CPI+2.5%, 10% within 12 months, 25% within 3 years, and moderate-high "
                "risk appetite; RBA/ABS/FRED/BoM or CSIRO material only where it was "
                "publicly available before 2 April 2026.",
                className="source-note",
                style={"marginTop": "12px"},
            ),
        ], className="panel"),

        # ── Macro Context: Timeline ───────────────────────────────────────────
        html.Div([
            html.H3("Macro Indicators Timeline",
                    style={"margin": "0 0 4px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            html.Div(
                "Choose any macro indicator for the primary (left) axis and an optional "
                "overlay (right axis) to compare two series side by side. "
                "GFC and COVID eras are shaded.",
                className="section-note",
            ),
            html.Div([
                html.Div([
                    html.Div("Primary (left axis)", className="ctrl-label"),
                    dcc.Dropdown(
                        id="m1-macro-primary",
                        options=[
                            {"label": "AUD/USD",             "value": "AUD/USD"},
                            {"label": "AUS CPI (YoY %)",     "value": "AUS CPI (YoY %)"},
                            {"label": "US CPI (YoY %)",      "value": "US CPI (YoY %)"},
                            {"label": "RBA Rate (%)",        "value": "RBA Rate (%)"},
                            {"label": "Fed Funds Rate (%)",  "value": "Fed Funds Rate (%)"},
                        ],
                        value="AUD/USD",
                        clearable=False,
                        style={"width": "230px", "fontSize": "13px"},
                    ),
                ], className="ctrl-group", style={"marginRight": "20px"}),
                html.Div([
                    html.Div("Overlay (right axis)", className="ctrl-label"),
                    dcc.Dropdown(
                        id="m1-macro-overlay",
                        options=[
                            {"label": "AUD/USD",             "value": "AUD/USD"},
                            {"label": "AUS CPI (YoY %)",     "value": "AUS CPI (YoY %)"},
                            {"label": "US CPI (YoY %)",      "value": "US CPI (YoY %)"},
                            {"label": "RBA Rate (%)",        "value": "RBA Rate (%)"},
                            {"label": "Fed Funds Rate (%)",  "value": "Fed Funds Rate (%)"},
                            {"label": "None",                 "value": "none"},
                        ],
                        value="RBA Rate (%)",
                        clearable=False,
                        style={"width": "230px", "fontSize": "13px"},
                    ),
                ], className="ctrl-group"),
            ], className="chart-controls"),
            dcc.Graph(id="m1-macro-timeline", config={"displayModeBar": True}),
        ], className="panel"),

        # ── Macro Context: Regime Risk–Return by Domicile ──────────────────────
        html.Div(
            html.H3("Annualised Returns by Macro Regime",
                    style={"margin": "0 0 4px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            className="panel",
            style={"marginBottom": "8px"},
        ),
        html.Div([
            # ── AUS Domicile ──────────────────────────────────────────────────
            html.Div([
                html.H4("AUS Domicile Assets",
                        style={"margin": "0 0 2px 0", "color": COLORS["ink"],
                               "fontSize": "14px", "fontWeight": "600"}),
                html.Div(
                    "Cash · AU Short Duration Bond · AU Fixed Income · "
                    "AU Listed Equity · AU Listed Property",
                    className="section-note",
                    style={"marginBottom": "8px"},
                ),
                html.Div([
                    html.Div("Regime", className="ctrl-label"),
                    dcc.Dropdown(
                        id="m1-aus-regime-dd",
                        options=[
                            {"label": "AUS CPI Regime",  "value": "aus_cpi"},
                            {"label": "RBA Rate Regime", "value": "rba_rate"},
                            {"label": "AUD/USD Regime",  "value": "audusd"},
                        ],
                        value="aus_cpi",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ], className="ctrl-group"),
                dcc.Graph(id="m1-aus-regime-chart",
                          config={"displayModeBar": False}),
            ], className="panel",
               style={"flex": "1", "minWidth": "0", "margin": "0 10px 0 0"}),

            # ── Global / US Domicile ───────────────────────────────────────────
            html.Div([
                html.H4("Global / US Assets",
                        style={"margin": "0 0 2px 0", "color": COLORS["ink"],
                               "fontSize": "14px", "fontWeight": "600"}),
                html.Div(
                    "Global Fixed Income (H) · Global Credit (H) · "
                    "Global Equity Unhedged · Global Equity Hedged · "
                    "Global Infrastructure · Global Private Equity",
                    className="section-note",
                    style={"marginBottom": "8px"},
                ),
                html.Div([
                    html.Div("Regime", className="ctrl-label"),
                    dcc.Dropdown(
                        id="m1-us-regime-dd",
                        options=[
                            {"label": "US CPI Regime",       "value": "us_cpi"},
                            {"label": "Fed Funds Regime",     "value": "fed_funds"},
                            {"label": "AUD/USD Regime",       "value": "audusd"},
                        ],
                        value="us_cpi",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ], className="ctrl-group"),
                dcc.Graph(id="m1-us-regime-chart",
                          config={"displayModeBar": False}),
            ], className="panel",
               style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "stretch",
                  "marginBottom": "20px"}),

        # ── Macro Context: Asset–Macro Correlation Heatmap ───────────────────
        html.Div([
            html.H3("Asset Class – Macro Factor Correlations",
                    style={"margin": "0 0 4px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            html.Div(
                "Pearson correlation between each asset class's monthly return and four "
                "macro variables over the selected period. "
                "AUD/USD Δ MoM and Fed Funds Δ MoM use month-over-month changes; "
                "CPI series use the YoY level. "
                "Negative correlation with AUD/USD Δ MoM means the asset benefits "
                "when the AUD weakens (e.g. unhedged global equity).",
                className="section-note",
            ),
            dcc.Graph(id="m1-macro-corr", config={"displayModeBar": False}),
        ], className="panel"),

        # ── EDA Section ───────────────────────────────────────────────────────
        html.Div([
            html.H2("Exploratory Analysis on Historical Data"),
            html.Div(id="eda-period-note", className="section-note"),
            html.H3("Descriptive Statistics on Monthly Returns",
                    style={"margin": "14px 0 8px 0"}),
            _desc_stats_table(),
            html.Div(
                "% columns shown to 1 decimal place. Skewness and Kurtosis are "
                "pure numbers (2 dp). Mean and Std Dev columns are highlighted.",
                className="hist-note",
            ),
        ], className="panel"),

        # ── Monthly Return Distributions (standalone, outside EDA box) ────────
        html.Div([
            html.H3("Monthly Return Distributions",
                    style={"margin": "0 0 4px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            dcc.Graph(id="m1-histograms", config={"displayModeBar": True}),
        ], className="panel"),

        # ── Monthly Returns Over Time ─────────────────────────────────────────
        html.Div([
            html.H3("Monthly Returns Over Time",
                    style={"margin": "0 0 10px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            html.Div([
                html.Div([
                    html.Div("Return Mode", className="ctrl-label"),
                    dcc.RadioItems(
                        id="ret-mode-radio",
                        options=[
                            {"label": " Monthly", "value": "monthly"},
                            {"label": " Annualised (Calendar Year)", "value": "annualised"},
                        ],
                        value="monthly", inline=True,
                        labelStyle={"fontSize": "13px", "marginRight": "18px",
                                    "cursor": "pointer"},
                        inputStyle={"marginRight": "4px",
                                    "accentColor": COLORS["accent"]},
                    ),
                ], className="ctrl-group"),
                html.Div([
                    html.Div("Asset Classes", className="ctrl-label"),
                    _asset_checklist("ret-asset-check", "ret-all-btn", "ret-none-btn"),
                ], className="ctrl-group", style={"flex": "1"}),
            ], className="chart-controls"),
            html.Div([
                html.Span([html.Span(style={"backgroundColor": "#FFE8B2",
                                            "border": "1px solid #ccc"},
                                     className="era-swatch"),
                           "GFC (Sep 2008 – Mar 2009)"]),
                html.Span([html.Span(style={"backgroundColor": "#B2D6F5",
                                            "border": "1px solid #ccc"},
                                     className="era-swatch"),
                           "COVID-19 (Feb – Apr 2020)"]),
                html.Span("Shading shown in Monthly mode only.",
                          style={"fontSize": "11px", "color": COLORS["muted"],
                                 "marginLeft": "8px"}),
            ], className="era-legend"),
            dcc.Graph(id="returns-time-chart", config={"displayModeBar": True}),
        ], className="panel"),

        # ── Cumulative Returns ────────────────────────────────────────────────
        html.Div([
            html.H3("Cumulative Returns (Growth of $1)",
                    style={"margin": "0 0 10px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            html.Div([
                html.Div([
                    html.Div("Asset Classes", className="ctrl-label"),
                    _asset_checklist("cum-asset-check", "cum-all-btn", "cum-none-btn"),
                ], className="ctrl-group", style={"flex": "1"}),
            ], className="chart-controls"),
            dcc.Graph(id="cumulative-chart", config={"displayModeBar": True}),
        ], className="panel"),

        # ── 12-Month Rolling Annualised Volatility ────────────────────────────
        html.Div([
            html.H3("12-Month Rolling Annualised Volatility",
                    style={"margin": "0 0 10px 0", "color": COLORS["accent"],
                           "fontSize": "16px", "fontWeight": "600"}),
            html.Div([
                html.Div([
                    html.Div("Asset Classes", className="ctrl-label"),
                    _asset_checklist("vol-asset-check", "vol-all-btn", "vol-none-btn"),
                ], className="ctrl-group", style={"flex": "1"}),
            ], className="chart-controls"),
            dcc.Graph(id="rolling-vol-chart", config={"displayModeBar": True}),
        ], className="panel"),

        # ── Risk-Return Scatter + Correlation Matrix (side by side) ─────────
        html.Div([
            # Left: Risk-Return Scatter
            html.Div([
                html.H3("Risk-Return Scatter",
                        style={"margin": "0 0 10px 0", "color": COLORS["accent"],
                               "fontSize": "16px", "fontWeight": "600"}),
                html.Div([
                    html.Div([
                        html.Div("Asset Classes", className="ctrl-label"),
                        _asset_checklist("scatter-asset-check",
                                         "scatter-all-btn", "scatter-none-btn"),
                    ], className="ctrl-group", style={"flex": "1"}),
                ], className="chart-controls"),
                dcc.Graph(id="scatter-chart", config={"displayModeBar": True}),
            ], className="panel", style={"flex": "1.15", "minWidth": "0",
                                         "margin": "0 10px 0 0"}),

            # Right: Correlation Matrix
            html.Div([
                html.H3("Correlation Matrix",
                        style={"margin": "0 0 4px 0", "color": COLORS["accent"],
                               "fontSize": "16px", "fontWeight": "600"}),
                html.Div("Pairwise correlations for the selected period. "
                         "Read-only — does not affect downstream calculations.",
                         className="section-note"),
                dcc.Graph(id="m1-corr-eda", config={"displayModeBar": True}),
            ], className="panel", style={"flex": "0.85", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "stretch",
                  "marginBottom": "20px"}),
    ])


# ---------------------------------------------------------------------------
# Module 2 — Trust Characteristics
# ---------------------------------------------------------------------------

TRUST_TAGLINES = {
    "STI": "Capital preservation \u00b7 Daily liquidity \u00b7 Cash + 0.5%",
    "MTG": "Moderate growth \u00b7 Monthly liquidity \u00b7 CPI + 2.0%",
    "LTG": "Long-term growth \u00b7 Quarterly liquidity \u00b7 CPI + 3.0%",
}


def _trust_card(trust_name: str, c: dict) -> html.Div:
    pill_cls = "pill pill-pass" if c["meets_target"] else "pill pill-fail"
    pill_text = "MEETS TARGET" if c["meets_target"] else "BELOW TARGET"
    target_basis = {"STI": "Cash + 0.50%", "MTG": "CPI + 2.00%", "LTG": "CPI + 3.00%"}[trust_name]
    target_label = f"Target: {_fmt_pct(c['target_return'])} ({target_basis})"
    return html.Div([
        html.P(trust_name, className="trust-name"),
        html.Div(TRUST_TAGLINES[trust_name], className="trust-tag"),
        html.Div(_fmt_pct(c["net_return"]), className="net-return"),
        html.Div("Net expected return p.a.", className="net-return-label"),
        html.Div([html.Span(target_label), html.Span(pill_text, className=pill_cls)],
                 className="target-line"),
        html.Div([
            html.Span("Gross return", className="k"),
            html.Span(_fmt_pct(c["gross_return"]), className="v"),
            html.Span("Asset cost", className="k"),
            html.Span(f"\u2212{_fmt_pct(c['weighted_asset_cost'], 3)}", className="v"),
            html.Span("Trust ongoing", className="k"),
            html.Span(f"\u2212{_fmt_pct(c['ongoing_cost'], 3)}", className="v"),
            html.Span("Volatility", className="k"),
            html.Span(_fmt_pct(c["volatility"]), className="v"),
            html.Span("Sharpe ratio", className="k"),
            html.Span(f"{c['sharpe']:.3f}" if not np.isnan(c['sharpe']) else "—", className="v"),
            html.Span("CPI+ spread", className="k"),
            html.Span(_fmt_signed_pct(c["cpi_plus_spread"]), className="v"),
        ], className="stats-grid"),
    ], className="trust-card", style={"--trust-accent": COLORS[trust_name]})


def correlation_heatmap_figure(corr_matrix: np.ndarray) -> go.Figure:
    short_labels = [tc.ASSET_CLASS_SHORT[a] for a in tc.ASSET_CLASSES]
    text = [[f"{v:.2f}" for v in row] for row in corr_matrix]
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix, x=short_labels, y=short_labels,
        zmin=-1, zmax=1, zmid=0,
        colorscale=[[0.0, COLORS["heat_neg"]], [0.5, COLORS["heat_zero"]], [1.0, COLORS["heat_pos"]]],
        text=text, texttemplate="%{text}",
        textfont={"family": MONO_STACK, "size": 10},
        colorbar=dict(title="\u03c1", tickfont=dict(family=FONT_STACK, size=11), len=0.7, thickness=12),
        hovertemplate="%{y} \u00d7 %{x}<br>\u03c1 = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=440,
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=11),
        xaxis=dict(side="bottom", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
    )
    return fig


def trust_comparison_figure(chars: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Net Return", "Volatility", "Sharpe Ratio"),
        horizontal_spacing=0.10)
    trusts = tc.TRUST_NAMES
    colors = [COLORS[t] for t in trusts]
    net_returns = [chars[t]["net_return"] for t in trusts]
    vols        = [chars[t]["volatility"] for t in trusts]
    sharpes     = [chars[t]["sharpe"] for t in trusts]

    for col, (vals, fmt_fn) in enumerate(
        [(net_returns, lambda v: _fmt_pct(v)),
         (vols,        lambda v: _fmt_pct(v)),
         (sharpes,     lambda v: f"{v:.2f}")], start=1):
        fig.add_trace(go.Bar(x=trusts, y=vals, marker_color=colors, showlegend=False,
            text=[fmt_fn(v) for v in vals], textposition="outside",
            textfont=dict(family=MONO_STACK, size=11),
            hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=col)

    fig.update_yaxes(tickformat=".1%", row=1, col=1, gridcolor=COLORS["border"])
    fig.update_yaxes(tickformat=".1%", row=1, col=2, gridcolor=COLORS["border"])
    fig.update_yaxes(row=1, col=3, gridcolor=COLORS["border"])
    fig.update_xaxes(showgrid=False)
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12))
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(family=FONT_STACK, size=12, color=COLORS["ink"])
    return fig


def historical_backtest_figure() -> go.Figure:
    cum = HIST_TRUST_CUMULATIVE_NET
    fig = go.Figure()
    for t in tc.TRUST_NAMES:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[t].values, mode="lines", name=t,
            line=dict(color=COLORS[t], width=2),
            hovertemplate=f"<b>{t}</b><br>%{{x}}<br>Wealth: %{{y:.3f}}\u00d7<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=20, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(gridcolor=COLORS["border"], tickformat=".2f",
            title=dict(text="Cumulative wealth (\u00d7 starting capital)",
                       font=dict(size=11, color=COLORS["muted"])),
            tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified")
    return fig


def _backtest_stats_rows() -> list[dict]:
    rows = []
    for t in tc.TRUST_NAMES:
        s = HIST_TRUST_MONTHLY_NET[t].values
        rows.append({"trust": t,
            "ann_return":  _fmt_pct(mt.annualised_geometric(s)),
            "ann_vol":     _fmt_pct(mt.annualised_vol(s)),
            "max_dd":      _fmt_pct(mt.max_drawdown(s)),
            "var_95":      _fmt_pct(mt.var_historic(s, 0.95)),
            "cvar_95":     _fmt_pct(mt.cvar_historic(s, 0.95)),
            "best_month":  _fmt_pct(float(np.max(s))),
            "worst_month": _fmt_pct(float(np.min(s))),
        })
    return rows


def _backtest_stats_table():
    return dash_table.DataTable(
        id="backtest-stats-table",
        columns=[{"name": "Trust", "id": "trust"},
                 {"name": "Ann. Return (geom.)", "id": "ann_return"},
                 {"name": "Ann. Volatility",     "id": "ann_vol"},
                 {"name": "Max Drawdown",         "id": "max_dd"},
                 {"name": "Monthly VaR 95%",      "id": "var_95"},
                 {"name": "Monthly CVaR 95%",     "id": "cvar_95"},
                 {"name": "Best Month",           "id": "best_month"},
                 {"name": "Worst Month",          "id": "worst_month"}],
        data=_backtest_stats_rows(),
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px 10px", "fontFamily": MONO_STACK,
                    "fontSize": "12.5px", "textAlign": "right"},
        style_cell_conditional=[{"if": {"column_id": "trust"},
            "fontFamily": FONT_STACK, "textAlign": "left", "fontWeight": "600"}],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
            "fontWeight": "600", "fontSize": "12px",
            "borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
    )


def _cfo_table_style():
    return dict(
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px 12px", "fontFamily": MONO_STACK,
                    "fontSize": "13px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "asset_class"},
             "fontFamily": FONT_STACK, "textAlign": "left"},
            {"if": {"column_id": "metric"},
             "fontFamily": FONT_STACK, "textAlign": "left", "fontWeight": "600"},
            {"if": {"column_id": "field"},
             "fontFamily": FONT_STACK, "textAlign": "left", "fontWeight": "600"},
        ],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
            "fontWeight": "600", "fontSize": "12px",
            "borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
        export_format="csv",
    )


def module_2_layout():
    return html.Div([
        html.Div([
            html.H2("Module 2 — Trust Characteristics"),
            html.Div("Forward-looking trust metrics derived from the Module 1 CMAs "
                     "and the fixed trust weight vectors. Edits in Module 1 propagate here.",
                     className="section-note"),
            html.Div(id="m2-trust-cards"),
        ], className="panel"),

        html.Div([
            html.H2("Trust role in the NSWDF portfolio"),
            html.Div(
                "This turns the unit-trust metrics into decision language for the CFO: "
                "what each trust contributes to liquidity, risk, and real-return capacity.",
                className="section-note"),
            html.Div(id="m2-trust-role-cards"),
        ], className="panel"),

        html.Div([
            html.H2("Correlation matrix and trust comparison"),
            html.Div("The heatmap reflects the fixed historical correlation matrix. "
                     "The comparison panel shows net return, volatility, and Sharpe across the three trusts.",
                     className="section-note"),
            html.Div([
                dcc.Graph(id="m2-corr-heatmap", config={"displayModeBar": False}),
                dcc.Graph(id="m2-comparison-chart", config={"displayModeBar": False}),
            ], className="chart-row"),
        ], className="panel"),

        html.Div([
            html.H2("Historical backtest"),
            html.Div(
                f"Cumulative wealth from {_returns_df.index[0]} to {_returns_df.index[-1]}, "
                "monthly rebalancing to fixed trust weights, net of costs. Static — does not "
                "react to CMA edits.",
                className="section-note"),
            dcc.Graph(id="m2-backtest-chart", figure=historical_backtest_figure(),
                      config={"displayModeBar": False}),
            html.Div([_backtest_stats_table()], className="backtest-stats-table"),
        ], className="panel"),

        html.Div([
            html.H2("CFO Brief — Export-ready tables"),
            html.Div("These three tables match the CFO brief template exactly. "
                     "Each has a CSV export button. Tables 1 and 2 react to Module 1 edits. "
                     "Table 3 reflects the current Module 3 allocation.",
                     className="section-note"),
            html.Div([
                html.Div([
                    html.H3("Table 1 — Asset Class Review", className="cfo-table-title"),
                    dash_table.DataTable(id="m2-cfo-table-1",
                        columns=[{"name": "Asset Class", "id": "asset_class"},
                                 {"name": "Historical Return", "id": "historical_return"},
                                 {"name": "Forecast Return", "id": "forecast_return"},
                                 {"name": "Difference in Return", "id": "difference"},
                                 {"name": "Historical Risk", "id": "historical_risk"}],
                        data=[], **_cfo_table_style()),
                ]),
                html.Div([
                    html.H3("Table 2 — Unit Trust Performance", className="cfo-table-title"),
                    dash_table.DataTable(id="m2-cfo-table-2",
                        columns=[{"name": "", "id": "metric"},
                                 {"name": "STI", "id": "STI"},
                                 {"name": "MTG", "id": "MTG"},
                                 {"name": "LTG", "id": "LTG"}],
                        data=[], **_cfo_table_style()),
                ]),
                html.Div([
                    html.H3("Table 3 — NSWDF Portfolio", className="cfo-table-title"),
                    dash_table.DataTable(id="m2-cfo-table-3",
                        columns=[{"name": "", "id": "field"}, {"name": "", "id": "value"}],
                        data=[], **_cfo_table_style()),
                ]),
            ], className="cfo-tables"),
        ], className="panel"),
    ])


# ---------------------------------------------------------------------------
# Module 3 — Portfolio Optimisation
# ---------------------------------------------------------------------------

PORTFOLIO_AUD = 3_000_000_000

OBJECTIVE_LABELS = {
    "max_sharpe":  "Maximise Sharpe ratio",
    "min_vol":     "Minimise volatility (s.t. CPI + 2.5%)",
    "max_return":  "Maximise return (s.t. volatility cap)",
}


def _alloc_block(block_id, title, note, input_kind, default):
    rows = []
    for trust in tc.TRUST_NAMES:
        if input_kind == "number":
            ctrl = dcc.Input(id=f"{block_id}-{trust}", type="number",
                min=0, max=100, step=0.1,
                value=round(default[trust] * 100, 1), className="alloc-num-input")
        else:
            ctrl = dcc.Slider(id=f"{block_id}-{trust}", min=0, max=100, step=1,
                value=round(default[trust] * 100), marks=None,
                tooltip={"placement": "bottom", "always_visible": False})
        rows.extend([
            html.Span(trust, className="lbl", style={"--row-color": COLORS[trust]}),
            ctrl,
            html.Span(id=f"{block_id}-{trust}-display", className="val"),
        ])
    return html.Div([
        html.P(title, className="block-title"),
        html.Div(note, className="block-note"),
        html.Div(rows, className="alloc-grid"),
        html.Div(id=f"{block_id}-sum", className="alloc-sum"),
    ], className="alloc-block")


def _live_metrics_block():
    return html.Div([
        html.Div([html.Div("Net expected return", className="metric-label"),
                  html.Div(id="m3-live-return", className="metric-value")],
                 className="metric-block"),
        html.Div([html.Div("Volatility", className="metric-label"),
                  html.Div(id="m3-live-vol", className="metric-value")],
                 className="metric-block"),
        html.Div([html.Div("Sharpe ratio", className="metric-label"),
                  html.Div(id="m3-live-sharpe", className="metric-value")],
                 className="metric-block"),
        html.Div(id="m3-constraints", className="constraint-row",
                 style={"gridColumn": "1 / -1"}),
    ], className="live-metrics")


def _scatter_figure(grid_eval, target, current_w, proposed_w, optimal_w):
    fig = go.Figure()
    sharpe = grid_eval["sharpe"].values
    fig.add_trace(go.Scatter(
        x=grid_eval["volatility"].values, y=grid_eval["net_return"].values,
        mode="markers",
        marker=dict(size=4, color=sharpe,
            colorscale=[[0.0, "#D8C99B"], [0.5, "#9AB3A6"], [1.0, COLORS["accent"]]],
            showscale=True,
            colorbar=dict(title=dict(text="Sharpe", font=dict(size=11)),
                          tickfont=dict(size=10), len=0.7, thickness=12),
            opacity=0.55, line=dict(width=0)),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.3f}<extra></extra>",
        name="Feasible portfolios"))
    fig.add_hline(y=target, line=dict(color=COLORS["fail"], width=1, dash="dot"),
        annotation_text=f"CPI + 2.5% target ({target*100:.2f}%)",
        annotation_position="top right",
        annotation_font=dict(size=11, color=COLORS["fail"]))
    if current_w is not None:
        fig.add_trace(go.Scatter(x=[current_w["vol"]], y=[current_w["ret"]], mode="markers",
            marker=dict(size=14, color="white", line=dict(color=COLORS["ink"], width=2)),
            name="Current holdings",
            hovertemplate=(f"Current: STI {current_w['weights']['STI']*100:.1f}% / "
                           f"MTG {current_w['weights']['MTG']*100:.1f}% / "
                           f"LTG {current_w['weights']['LTG']*100:.1f}%<br>"
                           "Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>")))
    if proposed_w is not None:
        fig.add_trace(go.Scatter(x=[proposed_w["vol"]], y=[proposed_w["ret"]], mode="markers",
            marker=dict(size=14, color=COLORS["accent"], line=dict(color="white", width=2)),
            name="Proposed allocation",
            hovertemplate=(f"Proposed: STI {proposed_w['weights']['STI']*100:.1f}% / "
                           f"MTG {proposed_w['weights']['MTG']*100:.1f}% / "
                           f"LTG {proposed_w['weights']['LTG']*100:.1f}%<br>"
                           "Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>")))
    if optimal_w is not None:
        fig.add_trace(go.Scatter(x=[optimal_w["vol"]], y=[optimal_w["ret"]], mode="markers",
            marker=dict(size=18, color="#D4A93A", symbol="star",
                        line=dict(color=COLORS["ink"], width=1)),
            name="Optimiser",
            hovertemplate=(f"Optimal: STI {optimal_w['weights']['STI']*100:.1f}% / "
                           f"MTG {optimal_w['weights']['MTG']*100:.1f}% / "
                           f"LTG {optimal_w['weights']['LTG']*100:.1f}%<br>"
                           "Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>")))
    fig.update_layout(height=440, margin=dict(l=50, r=20, t=20, b=50),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(tickformat=".1%", gridcolor=COLORS["border"],
            title=dict(text="Annualised volatility", font=dict(size=11, color=COLORS["muted"])),
            tickfont=dict(size=11)),
        yaxis=dict(tickformat=".1%", gridcolor=COLORS["border"],
            title=dict(text="Net expected return", font=dict(size=11, color=COLORS["muted"])),
            tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _tornado_figure(sens, baseline_vol, objective):
    pv = sens.pivot(index="asset_class", columns="bump_bps", values="volatility")
    pv = pv.reindex(tc.ASSET_CLASSES)
    sens_range = pv.max(axis=1) - pv.min(axis=1)
    pv = pv.loc[sens_range.sort_values(ascending=True).index]
    fig = go.Figure()
    bump_colors = {-100: "#A23737", -50: "#D4A93A", 50: "#9AB3A6", 100: COLORS["accent"]}
    for bump in [-100, -50, 50, 100]:
        if bump not in pv.columns:
            continue
        delta = (pv[bump] - baseline_vol) * 100
        fig.add_trace(go.Bar(y=pv.index, x=delta.values, orientation="h",
            name=f"{'+' if bump > 0 else ''}{bump} bps", marker_color=bump_colors[bump],
            text=[f"{d:+.2f} pp" for d in delta.values], textposition="outside",
            textfont=dict(family=MONO_STACK, size=10),
            hovertemplate=f"<b>%{{y}}</b><br>Bump: {'+' if bump > 0 else ''}{bump} bps<br>Change: %{{x:+.2f}} pp<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=COLORS["ink"], width=1))
    fig.update_layout(height=520, barmode="group",
        margin=dict(l=180, r=20, t=20, b=50),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=11),
        xaxis=dict(title=dict(text="Change in optimal portfolio volatility (pp, vs baseline)",
                               font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickfont=dict(size=11),
                   zeroline=False, ticksuffix=" pp"),
        yaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def module_3_layout():
    return html.Div([
        html.Div([
            html.H2("Module 3 — Portfolio Optimisation"),
            html.Div("Set Current Holdings (the existing position) and a Proposed Allocation "
                     "(what you're considering). Sliders auto-rebalance to sum to 100%. "
                     "The Proposed Allocation feeds Module 2 Table 3.",
                     className="section-note"),
            html.Div([
                _alloc_block("current", "Current Holdings",
                    "Existing trust position. Baseline for transaction cost calculation.",
                    "number", {"STI": 0.33, "MTG": 0.33, "LTG": 0.34}),
                _alloc_block("proposed", "Proposed Allocation",
                    "What you're considering. Sliders auto-rebalance.",
                    "slider", {"STI": 0.33, "MTG": 0.33, "LTG": 0.34}),
            ], className="alloc-row"),
            _live_metrics_block(),
        ], className="panel"),

        html.Div([
            html.H2("Optimiser"),
            html.Div("Grid search at 1% resolution, refined with SLSQP. Constraints: "
                     "weights sum to 1, all \u2265 0, w_STI \u2265 10%, "
                     "w_STI + w_MTG \u2265 25%, net return \u2265 CPI + 2.5%.",
                     className="section-note"),
            html.Div([
                html.Div([
                    html.Label("Objective"),
                    dcc.Dropdown(id="m3-objective",
                        options=[{"label": v, "value": k} for k, v in OBJECTIVE_LABELS.items()],
                        value="max_sharpe", clearable=False,
                        style={"fontFamily": FONT_STACK, "fontSize": "14px"}),
                ]),
                html.Div(id="m3-volcap-wrapper", children=[
                    html.Label("Volatility cap (%)"),
                    dcc.Input(id="m3-volcap", type="number", min=0.5, max=30,
                              step=0.1, value=8.0, className="alloc-num-input",
                              style={"width": "120px"}),
                ]),
                html.Button("Run optimiser", id="m3-run-button",
                            className="opt-button", n_clicks=0),
            ], className="opt-controls"),
            html.Div(id="m3-opt-result"),
        ], className="panel"),

        html.Div([
            html.H2("Feasible portfolio scatter"),
            html.Div("Every liquidity-feasible portfolio at 1% resolution, coloured by "
                     "Sharpe ratio. White = current holdings, teal = proposed, gold star = optimiser. "
                     "Dotted red line = CPI + 2.5% target.",
                     className="section-note"),
            dcc.Graph(id="m3-scatter", config={"displayModeBar": False}),
        ], className="panel"),

        html.Div([
            html.H2("Board Policy compliance"),
            html.Div(
                "Assignment-facing check against the NSWDF Investment Directive: return target, "
                "liquidity, permitted instruments, diversification, and risk appetite.",
                className="section-note"),
            html.Div(id="m3-board-compliance"),
        ], className="panel"),

        html.Div([
            html.H2("Sensitivity sweep"),
            html.Div("Each row perturbs one asset class expected return by \u00b1100 / \u00b150 bps, "
                     "re-runs the optimiser, and shows the change in optimal portfolio volatility "
                     "vs the unperturbed baseline. Rows ordered by sensitivity range.",
                     className="section-note"),
            dcc.Loading(dcc.Graph(id="m3-tornado", config={"displayModeBar": False}),
                        type="circle", color=COLORS["accent"]),
        ], className="panel"),

        dcc.Store(id="m3-opt-store"),
    ])


# ---------------------------------------------------------------------------
# Placeholder for Modules 4-6
# ---------------------------------------------------------------------------

def placeholder_layout(module_name: str):
    return html.Div([
        html.Div([html.H2(module_name),
                  html.Div("Coming up in the next build slice.", className="section-note")],
                 className="panel")])


# ---------------------------------------------------------------------------
# Module 4 — Market Stress Testing
# ---------------------------------------------------------------------------

SCENARIO_ORDER = [
    "GFC",
    "COVID Crash",
    "COVID Inflation Shock (2022)",
    "AUD Depreciation Shock",
    "Interest Rate Shock (+200bps)",
]


def _factor_class(label: str) -> str:
    return "factor-tag-" + label.replace(" ", "-")


def _shock_table_initial_rows(baseline_returns: np.ndarray,
                              shocked_returns: np.ndarray) -> list[dict]:
    rows = []
    for i, ac in enumerate(tc.ASSET_CLASSES):
        rows.append({
            "asset_class": ac,
            "baseline": round(float(baseline_returns[i]) * 100, 3),
            "shocked":  round(float(shocked_returns[i]) * 100, 3),
            "delta":    round(float(shocked_returns[i] - baseline_returns[i]) * 100, 3),
        })
    return rows


def shock_compare_figure(baseline_returns: np.ndarray,
                         shocked_returns: np.ndarray,
                         portfolio_weights: dict) -> go.Figure:
    base_nets  = {t: tc.trust_net_return(t, baseline_returns) for t in tc.TRUST_NAMES}
    shock_nets = st.trust_returns_under_shock(shocked_returns)
    base_port  = sum(portfolio_weights.get(t, 0) * base_nets[t] for t in tc.TRUST_NAMES)
    shock_port = st.portfolio_return_under_shock(portfolio_weights, shocked_returns)

    labels      = tc.TRUST_NAMES + ["Portfolio"]
    base_vals   = [base_nets[t] for t in tc.TRUST_NAMES] + [base_port]
    shock_vals  = [shock_nets[t] for t in tc.TRUST_NAMES] + [shock_port]
    base_colors = [COLORS[t] for t in tc.TRUST_NAMES] + [COLORS["accent"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=base_vals, name="Normal (CMA)",
        marker_color="#D4C49A",
        marker_line=dict(color=COLORS["border"], width=1),
        text=[_fmt_pct(v) for v in base_vals], textposition="outside",
        textfont=dict(family=MONO_STACK, size=11),
        hovertemplate="<b>%{x}</b><br>Normal: %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=shock_vals, name="Stressed",
        marker_color=base_colors,
        text=[_fmt_pct(v) for v in shock_vals], textposition="outside",
        textfont=dict(family=MONO_STACK, size=11),
        hovertemplate="<b>%{x}</b><br>Stressed: %{y:.2%}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=COLORS["ink"], width=1))
    fig.update_layout(height=380, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        barmode="group",
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(tickformat=".1%", gridcolor=COLORS["border"], tickfont=dict(size=11),
            title=dict(text="Annual return", font=dict(size=11, color=COLORS["muted"]))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _factor_breakdown_rows(shocked_returns: np.ndarray, df_for_drawdown,
                            scenario_name: str, window_label: Optional[str]) -> list[dict]:
    rows = []
    nets = st.trust_returns_under_shock(shocked_returns)
    for t in tc.TRUST_NAMES:
        dom, _ = st.dominant_factor(t, shocked_returns)
        dd_str = "—"
        if window_label is not None:
            try:
                start, end = window_label.split(" \u2013 ")
                dd = st.trust_drawdown_from_window(t, df_for_drawdown, start, end)
                dd_str = _fmt_pct(dd)
            except Exception:
                dd_str = "—"
        rows.append({"trust": t, "net_return": _fmt_pct(nets[t]),
                     "dominant_factor": dom, "window_drawdown": dd_str})
    return rows


def module_4_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Module 4 — Market Stress Testing"),
            html.Div(
                "Apply named historical or analytical shocks to the asset class returns "
                "and see how each trust and the overall portfolio responds. Shocked returns "
                "for the four historical scenarios are derived from the Refinitiv CSV windows; "
                "the rate shock is analytical (\u00b1 duration). Buy/sell spreads are NOT "
                "applied here \u2014 this is a return-characterisation exercise.",
                className="section-note"),
            html.Div([
                html.Div([
                    html.Label("Scenario",
                               style={"fontSize": "11px", "textTransform": "uppercase",
                                      "letterSpacing": "0.05em", "color": COLORS["muted"],
                                      "marginBottom": "4px", "display": "block"}),
                    dcc.Dropdown(id="m4-scenario",
                        options=[{"label": s, "value": s} for s in SCENARIO_ORDER],
                        value="GFC", clearable=False,
                        style={"fontFamily": FONT_STACK, "fontSize": "14px"}),
                ], style={"flex": "1 1 320px", "marginRight": "16px"}),
                html.Div([
                    html.Button("Reset to scenario defaults", id="m4-reset-button",
                                className="opt-button opt-button-secondary", n_clicks=0),
                ], style={"alignSelf": "flex-end"}),
            ], style={"display": "flex", "alignItems": "flex-end", "marginBottom": "10px"}),
            html.Div(id="m4-scenario-meta", className="scenario-meta"),
        ], className="panel"),

        html.Div([
            html.H2("Stressed return: trust and portfolio"),
            html.Div("Normal (CMA) vs stressed net return for each trust and the portfolio. "
                     "Portfolio uses the Proposed Allocation from Module 3.",
                     className="section-note"),
            dcc.Graph(id="m4-compare-chart", config={"displayModeBar": False}),
        ], className="panel"),

        html.Div([
            html.H2("Trust factor exposure"),
            html.Div("Net return under stress, dominant factor exposure, and \u2014 for "
                     "historical scenarios \u2014 the trust's peak-to-trough drawdown "
                     "across the actual window.",
                     className="section-note"),
            html.Div(id="m4-verdict", className="decision-band",
                     style={"marginBottom": "14px"}),
            html.Div(id="m4-factor-table"),
        ], className="panel"),

        html.Div([
            html.H2("Custom shock overrides"),
            html.Div("Edit the Shocked column to override any asset class's stressed return "
                     "(in %). The chart and factor table update live. Use 'Reset to scenario "
                     "defaults' to revert.",
                     className="section-note"),
            dash_table.DataTable(
                id="m4-shock-table",
                columns=[
                    {"name": "Asset Class", "id": "asset_class", "editable": False},
                    {"name": "Baseline (% p.a.)", "id": "baseline",
                     "type": "numeric", "format": {"specifier": ".3f"}, "editable": False},
                    {"name": "Shocked (% p.a.)", "id": "shocked",
                     "type": "numeric", "format": {"specifier": ".3f"}, "editable": True},
                    {"name": "Delta (% p.a.)", "id": "delta",
                     "type": "numeric", "format": {"specifier": ".3f"}, "editable": False},
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px 10px", "fontFamily": MONO_STACK,
                            "fontSize": "12.5px", "textAlign": "right"},
                style_cell_conditional=[
                    {"if": {"column_id": "asset_class"},
                     "fontFamily": FONT_STACK, "textAlign": "left", "minWidth": "260px"}],
                style_data_conditional=[
                    {"if": {"column_id": "shocked"}, "backgroundColor": COLORS["bg"]},
                    {"if": {"column_id": "delta", "filter_query": "{delta} > 0.001"},
                     "color": COLORS["pass"]},
                    {"if": {"column_id": "delta", "filter_query": "{delta} < -0.001"},
                     "color": COLORS["fail"]},
                ],
                style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
                    "fontWeight": "600", "fontSize": "12px",
                    "borderBottom": f"2px solid {COLORS['border']}"},
                style_data={"borderBottom": f"1px solid {COLORS['border']}"},
                editable=True,
            ),
        ], className="panel"),

        dcc.Store(id="m4-shocked-store"),
    ])


# ---------------------------------------------------------------------------
# Module 5 — Drought Scenario
# ---------------------------------------------------------------------------

SEVERITY_OPTIONS = ["Mild", "Moderate", "Severe"]


def _severity_slider_bounds(severity: str) -> tuple:
    lo, hi = dr.SEVERITY_BANDS[severity]
    mid = (lo + hi) / 2
    return int(lo / 1e6), int(hi / 1e6), int(mid / 1e6)


def _projection_value_figure(result: dr.ProjectionResult, onset_year: int) -> go.Figure:
    years  = [0] + [y.year for y in result.years]
    values = [result.initial_value] + [y.ending_value for y in result.years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=7, color=COLORS["accent"]),
        name="Portfolio value",
        hovertemplate="Year %{x}<br>Value: $%{y:,.0f}<extra></extra>"))
    drought_years = [y.year for y in result.years if y.drawdown > 0]
    for dy in drought_years:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.4)
    if drought_years:
        fig.add_annotation(x=drought_years[0], y=max(values),
            text=f"Drought onset (Y{drought_years[0]})", showarrow=False, yshift=10,
            font=dict(size=11, color=COLORS["fail"]))
    fig.add_hline(y=result.initial_value, line=dict(color=COLORS["muted"], width=1, dash="dash"),
        annotation_text="Starting value", annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=360, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value (AUD)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$.2s", tickfont=dict(size=11)))
    return fig


def _trust_composition_figure(result: dr.ProjectionResult) -> go.Figure:
    years = [0] + [y.year for y in result.years]
    if result.years:
        first = result.years[0]
        sti = [first.starting_weights["STI"] * first.starting_value]
        mtg = [first.starting_weights["MTG"] * first.starting_value]
        ltg = [first.starting_weights["LTG"] * first.starting_value]
    else:
        sti, mtg, ltg = [0], [0], [0]
    for y in result.years:
        sti.append(y.ending_holdings["STI"])
        mtg.append(y.ending_holdings["MTG"])
        ltg.append(y.ending_holdings["LTG"])
    fig = go.Figure()
    for trust, vals in [("STI", sti), ("MTG", mtg), ("LTG", ltg)]:
        fig.add_trace(go.Scatter(x=years, y=vals, mode="lines", stackgroup="one",
            name=trust, line=dict(width=0.5, color=COLORS[trust]),
            fillcolor=COLORS[trust],
            hovertemplate=f"<b>{trust}</b><br>Year %{{x}}<br>$%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=70, r=20, t=20, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Trust holdings (AUD)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$.2s", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _projection_summary_table(result: dr.ProjectionResult) -> dash_table.DataTable:
    rows = []
    for y in result.years:
        rows.append({
            "year":    y.year,
            "start":   _fmt_aud(y.starting_value),
            "growth":  _fmt_pct((y.pre_drawdown_value / y.starting_value - 1)
                                if y.starting_value > 0 else 0),
            "drawdown": _fmt_aud(y.drawdown) if y.drawdown > 0 else "—",
            "spread":   _fmt_aud(sum(y.spread_costs.values()))
                        if any(v > 0 for v in y.spread_costs.values()) else "—",
            "end":     _fmt_aud(y.ending_value),
            "sti_pct": _fmt_pct(y.ending_weights["STI"]),
            "mtg_pct": _fmt_pct(y.ending_weights["MTG"]),
            "ltg_pct": _fmt_pct(y.ending_weights["LTG"]),
            "liq_12m": _fmt_pct(y.liquidity_within_12m),
            "liq_3y":  _fmt_pct(y.liquidity_within_3y),
        })
    return dash_table.DataTable(
        id="m5-projection-table",
        columns=[
            {"name": "Year",        "id": "year"},
            {"name": "Starting",    "id": "start"},
            {"name": "Growth",      "id": "growth"},
            {"name": "Drawdown",    "id": "drawdown"},
            {"name": "Spread cost", "id": "spread"},
            {"name": "Ending",      "id": "end"},
            {"name": "STI",         "id": "sti_pct"},
            {"name": "MTG",         "id": "mtg_pct"},
            {"name": "LTG",         "id": "ltg_pct"},
            {"name": "Liq 12m",     "id": "liq_12m"},
            {"name": "Liq 3y",      "id": "liq_3y"},
        ],
        data=rows,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "7px 10px", "fontFamily": MONO_STACK,
                    "fontSize": "12px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "year"},
             "fontFamily": FONT_STACK, "textAlign": "center", "fontWeight": "600"}],
        style_data_conditional=[
            {"if": {"column_id": "drawdown", "filter_query": "{drawdown} != \"—\""},
             "color": COLORS["fail"]}],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
            "fontWeight": "600", "fontSize": "12px",
            "borderBottom": f"2px solid {COLORS['border']}"},
    )


def _onset_split_from_inputs(sti_pct, mtg_pct, ltg_pct) -> dict[str, float]:
    raw = {
        "STI": max(0.0, float(sti_pct or 0.0)),
        "MTG": max(0.0, float(mtg_pct or 0.0)),
        "LTG": max(0.0, float(ltg_pct or 0.0)),
    }
    total = sum(raw.values())
    if total <= 0:
        return {"STI": 1.0, "MTG": 0.0, "LTG": 0.0}
    return {t: v / total for t, v in raw.items()}


def _onset_split_controls() -> html.Div:
    return html.Div([
        html.Div([
            html.Label("Year-onset drawdown split"),
            html.Div(
                "How the onset-year relief amount is redeemed by trust. Values are normalised "
                "to 100%; any unfunded assigned amount spills over STI -> MTG -> LTG.",
                style={"fontSize": "11.5px", "color": COLORS["muted"],
                       "lineHeight": "1.35", "marginBottom": "8px"},
            ),
            html.Div([
                html.Div([html.Label("STI (%)"),
                          dcc.Input(id="m5-onset-split-STI", type="number",
                                    min=0, max=100, step=1, value=100,
                                    className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Label("MTG (%)"),
                          dcc.Input(id="m5-onset-split-MTG", type="number",
                                    min=0, max=100, step=1, value=0,
                                    className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Label("LTG (%)"),
                          dcc.Input(id="m5-onset-split-LTG", type="number",
                                    min=0, max=100, step=1, value=0,
                                    className="alloc-num-input")],
                         className="drought-control"),
            ], className="drought-controls",
               style={"gridTemplateColumns": "repeat(3, minmax(140px, 1fr))"}),
            html.Div(id="m5-onset-split-summary",
                     style={"fontSize": "12px", "color": COLORS["muted"],
                            "marginTop": "6px"}),
        ], className="alloc-block"),
    ])


def _summary_card(summary: dict, total_drawdown: float, total_spread: float) -> html.Div:
    if not summary:
        return html.Div()
    sustain_text = (
        "Residual portfolio CAN sustain remaining drawdowns while remaining solvent."
        if summary["can_sustain_residual"] else
        "Residual portfolio CANNOT cover remaining drawdowns without exhausting the fund."
    )
    sustain_cls = ("summary-verdict summary-verdict-pass"
                   if summary["can_sustain_residual"]
                   else "summary-verdict summary-verdict-fail")
    liq_breach = not summary["meets_12m"] or not summary["meets_3y"]
    return html.Div([
        html.P(f"After Year {summary['year']} drawdown", className="summary-title"),
        html.Div(_fmt_aud(summary["remaining_value"]), className="summary-headline"),
        html.Div("Remaining portfolio value",
                 style={"fontSize": "12px", "color": COLORS["muted"]}),
        html.Div([
            html.Div([html.Div("Drawdown this year", className="lbl"),
                      html.Div(_fmt_aud(summary["drawdown_this_year"]), className="val")],
                     className="summary-item"),
            html.Div([html.Div("Spread cost this year", className="lbl"),
                      html.Div(_fmt_aud(summary["spread_cost_this_year"]), className="val")],
                     className="summary-item"),
            html.Div([html.Div("Residual drawdowns to come", className="lbl"),
                      html.Div(_fmt_aud(summary["residual_drawdown_to_come"]), className="val")],
                     className="summary-item"),
            html.Div([html.Div("New trust mix", className="lbl"),
                      html.Div(f"STI {summary['ending_weights']['STI']*100:.1f}% / "
                               f"MTG {summary['ending_weights']['MTG']*100:.1f}% / "
                               f"LTG {summary['ending_weights']['LTG']*100:.1f}%",
                               className="val", style={"fontSize": "13px"})],
                     className="summary-item"),
            html.Div([html.Div("12m liquidity", className="lbl"),
                      html.Div([
                          html.Span(_fmt_pct(summary["liquidity_12m"]), className="val",
                              style={"fontSize": "16px", "fontWeight": "600", "marginRight": "8px"}),
                          html.Span("PASS" if summary["meets_12m"] else "FAIL",
                              className="pill " + ("pill-pass" if summary["meets_12m"] else "pill-fail")),
                      ])], className="summary-item"),
            html.Div([html.Div("3y liquidity", className="lbl"),
                      html.Div([
                          html.Span(_fmt_pct(summary["liquidity_3y"]), className="val",
                              style={"fontSize": "16px", "fontWeight": "600", "marginRight": "8px"}),
                          html.Span("PASS" if summary["meets_3y"] else "FAIL",
                              className="pill " + ("pill-pass" if summary["meets_3y"] else "pill-fail")),
                      ])], className="summary-item"),
        ], className="summary-grid"),
        html.Div(sustain_text, className=sustain_cls),
        (html.Div("Liquidity constraints breached at this point in the projection.",
                  className="summary-verdict summary-verdict-fail",
                  style={"marginTop": "8px"}) if liq_breach else None),
    ], className="summary-card")


def module_5_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Module 5 — Drought Scenario (Deterministic)"),
            html.Div("A drought is a CASHFLOW event, not a market shock. The fund value may "
                     "compound fine, but redemptions must respect trust liquidity (STI daily, "
                     "MTG monthly, LTG quarterly) and incur sell spreads. Liquidations follow "
                     "the order STI \u2192 MTG \u2192 LTG.",
                     className="section-note"),
            html.Div([
                html.Div([html.Label("Drought severity"),
                          dcc.Dropdown(id="m5-severity",
                              options=[{"label": s, "value": s} for s in SEVERITY_OPTIONS],
                              value="Severe", clearable=False,
                              style={"fontFamily": FONT_STACK, "fontSize": "14px"})],
                         className="drought-control"),
                html.Div([html.Label("Total relief amount"),
                          dcc.Slider(id="m5-relief", min=50, max=2000, step=10, value=1500,
                              marks={50: "$50M", 500: "$500M", 1000: "$1B", 2000: "$2B"},
                              tooltip={"placement": "bottom", "always_visible": True,
                                       "template": "${value}M"})],
                         className="drought-control", style={"gridColumn": "span 2"}),
                html.Div([html.Label("Onset year"),
                          dcc.Input(id="m5-onset", type="number", min=1, max=8, step=1,
                                    value=4, className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Label("Year-onset fraction (%)"),
                          dcc.Input(id="m5-fraction", type="number", min=10, max=100,
                                    step=5, value=50, className="alloc-num-input")],
                         className="drought-control"),
            ], className="drought-controls"),
            _onset_split_controls(),
            html.Div(id="m5-config-summary",
                     style={"fontSize": "12.5px", "color": COLORS["muted"], "marginTop": "10px"}),
        ], className="panel"),

        html.Div([html.H2("Portfolio value trajectory"),
                  html.Div("AUD value at end of each year given the Proposed Allocation "
                           "from Module 3, CMA returns, and the drought schedule above.",
                           className="section-note"),
                  dcc.Graph(id="m5-value-chart", config={"displayModeBar": False})],
                 className="panel"),

        html.Div([html.H2("Trust composition over time"),
                  html.Div("AUD held in each trust at end of each year. STI band shrinks "
                           "first during drought years.",
                           className="section-note"),
                  dcc.Graph(id="m5-composition-chart", config={"displayModeBar": False})],
                 className="panel"),

        html.Div([html.H2("Year-onset outcome"),
                  html.Div("Key state immediately after the year-onset drawdown.",
                           className="section-note"),
                  html.Div(id="m5-exec-verdict", className="decision-band",
                           style={"marginBottom": "14px"}),
                  html.Div(id="m5-summary-card")],
                 className="panel"),

        html.Div([html.H2("Year-by-year summary"),
                  html.Div("Per-year breakdown of value, growth, drawdown, spread cost, "
                           "ending trust mix, and liquidity coverage.",
                           className="section-note"),
                  html.Div(id="m5-projection-table-container"),
                  html.Div(id="m5-totals", style={"marginTop": "12px", "fontSize": "13px"})],
                 className="panel"),

        # Monte Carlo section
        html.Div([
            html.H2("Module 5b — Drought Scenario (Monte Carlo)"),
            html.Div([
                html.Strong("Indicative ranges only. "),
                "10,000 simulated paths under per-year drought probabilities "
                "(None 52.5%, Mild 30%, Moderate 12.5%, Severe 5%) and Normal-distributed "
                "trust returns from the CMA inputs. Uncertainty in the CMA point estimates "
                "is NOT captured \u2014 the return distribution is centred on whatever "
                "the team has entered.",
            ], className="section-note",
               style={"backgroundColor": COLORS["warn_bg"],
                      "border": f"1px solid {COLORS['warn_border']}",
                      "color": COLORS["warn_ink"],
                      "padding": "10px 14px", "borderRadius": "4px", "marginBottom": "16px"}),
            html.Div([
                html.Div([html.Label("Number of paths"),
                          dcc.Dropdown(id="m5-mc-paths",
                              options=[{"label": "1,000 (faster)", "value": 1000},
                                       {"label": "10,000 (default)", "value": 10000},
                                       {"label": "50,000 (slower)", "value": 50000}],
                              value=10000, clearable=False,
                              style={"fontFamily": FONT_STACK, "fontSize": "14px"})],
                         className="drought-control"),
                html.Div([html.Label("Random seed"),
                          dcc.Input(id="m5-mc-seed", type="number", min=0, step=1,
                                    value=42, className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Button("Re-run simulation", id="m5-mc-run",
                                      className="opt-button", n_clicks=0,
                                      style={"alignSelf": "flex-end"})],
                         className="drought-control"),
            ], className="drought-controls"),
            dcc.Loading(id="m5-mc-loading", type="circle", color=COLORS["accent"],
                children=[html.Div(id="m5-mc-summary", style={"marginTop": "16px"}),
                          dcc.Graph(id="m5-mc-fan-chart", config={"displayModeBar": False}),
                          dcc.Graph(id="m5-mc-exhaustion-chart", config={"displayModeBar": False})]),
        ], className="panel"),
    ])


# ---------------------------------------------------------------------------
# Module 6 — Combined Stress
# ---------------------------------------------------------------------------

def _combined_value_figure(baseline: dr.ProjectionResult, stressed: dr.ProjectionResult,
                            shock_year: int, drought_years: list) -> go.Figure:
    years = [0] + [y.year for y in baseline.years]
    base_vals   = [baseline.initial_value] + [y.ending_value for y in baseline.years]
    stress_vals = [stressed.initial_value] + [y.ending_value for y in stressed.years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=base_vals, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2), marker=dict(size=5),
        name="Drought only",
        hovertemplate="Year %{x}<br>Value: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=stress_vals, mode="lines+markers",
        line=dict(color=COLORS["fail"], width=2), marker=dict(size=5),
        name="Combined (crash + drought)",
        hovertemplate="Year %{x}<br>Value: $%{y:,.0f}<extra></extra>"))

    for dy in drought_years:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.35)
    fig.add_vline(x=shock_year, line=dict(color=COLORS["ink"], width=1.5, dash="dash"),
                  annotation_text=f"Market shock (Y{shock_year})",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=COLORS["ink"]))
    fig.add_hline(y=baseline.initial_value,
                  line=dict(color=COLORS["muted"], width=1, dash="dash"),
                  annotation_text="Starting value", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=400, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value (AUD)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$.2s", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def module_6_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Module 6 — Combined Stress (Market Crash + Drought)"),
            html.Div("Stack a one-year market shock on top of the drought projection. "
                     "Pick the Module 4 scenario and the year the shock lands "
                     "(independent of drought onset). The engine applies stressed trust "
                     "returns for that single year, then reverts to CMA returns. Drought "
                     "parameters are inherited from Module 5; allocation from Module 3.",
                     className="section-note"),
            html.Div([
                html.Strong("Note on annualisation. "),
                "Short-window scenarios (e.g. COVID Crash, 2 months) are annualised by "
                "compounding the window return to a full year, which amplifies the implied "
                "annual loss. Treat the comparison across scenarios as ordinal (which is "
                "worse) rather than as absolute loss forecasts.",
            ], style={"backgroundColor": COLORS["warn_bg"],
                      "border": f"1px solid {COLORS['warn_border']}",
                      "color": COLORS["warn_ink"], "padding": "10px 14px",
                      "borderRadius": "4px", "marginTop": "10px", "marginBottom": "16px",
                      "fontSize": "13px", "lineHeight": "1.5"}),
            html.Div([
                html.Div([html.Label("Market shock scenario"),
                          dcc.Dropdown(id="m6-scenario",
                              options=[{"label": s, "value": s} for s in SCENARIO_ORDER],
                              value="GFC", clearable=False,
                              style={"fontFamily": FONT_STACK, "fontSize": "14px"})],
                         className="drought-control"),
                html.Div([html.Label("Shock year"),
                          dcc.Input(id="m6-shock-year", type="number", min=1, max=10,
                                    step=1, value=4, className="alloc-num-input")],
                         className="drought-control"),
            ], className="drought-controls"),
            html.Div(id="m6-config-summary",
                     style={"fontSize": "12.5px", "color": COLORS["muted"], "marginTop": "10px"}),
        ], className="panel"),

        html.Div([html.H2("Combined trajectory"),
                  html.Div("Drought-only path (teal) vs combined market-shock-plus-drought "
                           "path (red). Dotted lines = drought years; dashed line = shock year.",
                           className="section-note"),
                  dcc.Graph(id="m6-value-chart", config={"displayModeBar": False})],
                 className="panel"),

        html.Div([html.H2("Joint impact summary"),
                  html.Div("Side-by-side outcomes for drought-only and combined-stress scenarios.",
                           className="section-note"),
                  html.Div(id="m6-summary-grid")],
                 className="panel"),
    ])


# ---------------------------------------------------------------------------
# Executive summary / assignment alignment
# ---------------------------------------------------------------------------

def _normalised_weights(alloc: dict | None) -> dict[str, float]:
    if alloc and sum(float(v) for v in alloc.values()) > 0:
        total = sum(float(alloc.get(t, 0.0)) for t in tc.TRUST_NAMES)
        return {t: float(alloc.get(t, 0.0)) / total for t in tc.TRUST_NAMES}
    return {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}


def _portfolio_metrics_from_store(cma_store: dict, alloc: dict) -> dict:
    returns, vols, corr, cpi = _store_to_arrays(cma_store)
    cov = tc.cma_to_covariance(vols, corr)
    weights = _normalised_weights(alloc)
    p_return = tc.portfolio_net_return(weights, returns)
    p_vol = tc.portfolio_volatility(weights, cov)
    cash = float(returns[0])
    sharpe = (p_return - cash) / p_vol if p_vol > 0 else float("nan")
    liq = mt.liquidity_coverage(weights)
    target = cpi + op.TARGET_SPREAD
    return {
        "weights": weights,
        "return": p_return,
        "vol": p_vol,
        "sharpe": sharpe,
        "liq": liq,
        "target": target,
        "cpi": cpi,
    }


def _board_compliance_table(metrics: dict) -> dash_table.DataTable:
    w = metrics["weights"]
    liq = metrics["liq"]
    target_ok = metrics["return"] >= metrics["target"] - 1e-9
    rows = [
        {"requirement": "Return target: CPI + 2.5% p.a.",
         "result": f"{_fmt_pct(metrics['return'])} vs {_fmt_pct(metrics['target'])}",
         "status": "Pass" if target_ok else "Fail"},
        {"requirement": "At least 10% of Fund available within 12 months",
         "result": _fmt_pct(liq["within_12m"]),
         "status": "Pass" if liq["meets_12m"] else "Fail"},
        {"requirement": "At least 25% of Fund available within 3 years",
         "result": _fmt_pct(liq["within_3y"]),
         "status": "Pass" if liq["meets_3y"] else "Fail"},
        {"requirement": "Invest only through STI, MTG and LTG",
         "result": "All portfolio weights are in permitted trusts",
         "status": "Pass"},
        {"requirement": "Maintain diversification across trusts",
         "result": f"STI {w['STI']*100:.1f}% / MTG {w['MTG']*100:.1f}% / LTG {w['LTG']*100:.1f}%",
         "status": "Pass" if min(w.values()) > 0 else "Review"},
        {"requirement": "Moderate-high risk appetite",
         "result": f"Portfolio volatility {_fmt_pct(metrics['vol'])}; growth exposure through MTG/LTG",
         "status": "Explain"},
    ]
    return dash_table.DataTable(
        columns=[
            {"name": "Board Policy / Directive requirement", "id": "requirement"},
            {"name": "Current result", "id": "result"},
            {"name": "Status", "id": "status"},
        ],
        data=rows,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "8px 10px", "fontFamily": FONT_STACK,
                    "fontSize": "12.5px", "textAlign": "left", "whiteSpace": "normal"},
        style_cell_conditional=[
            {"if": {"column_id": "result"}, "fontFamily": MONO_STACK},
            {"if": {"column_id": "status"}, "fontWeight": "600", "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{status} = Pass", "column_id": "status"},
             "color": COLORS["pass"]},
            {"if": {"filter_query": "{status} = Fail", "column_id": "status"},
             "color": COLORS["fail"]},
            {"if": {"filter_query": "{status} = Explain", "column_id": "status"},
             "color": COLORS["warn_ink"]},
        ],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
                      "fontWeight": "600", "fontSize": "12px",
                      "borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
    )


# ---------------------------------------------------------------------------
# Top-level app layout
# ---------------------------------------------------------------------------

app.layout = html.Div([
    dcc.Store(id="cma-store", data=_initial_cma_store()),
    dcc.Store(id="portfolio-allocation-store",
              data={"STI": 0.33, "MTG": 0.33, "LTG": 0.34}),
    html.Div([
        html.H1("NSWDF Portfolio Dashboard"),
        html.Div("AUD 3 billion drought reserve \u2014 STI / MTG / LTG allocation analysis",
                 className="subtitle"),
    ], className="app-header"),
    dcc.Tabs(id="main-tabs", value="m1", children=[
        dcc.Tab(label="1. CMA Inputs",          value="m1", children=module_1_layout()),
        dcc.Tab(label="2. Trust Characteristics", value="m2", children=module_2_layout()),
        dcc.Tab(label="3. Optimisation",        value="m3", children=module_3_layout()),
        dcc.Tab(label="4. Market Stress",       value="m4",
                children=module_4_layout()),
        dcc.Tab(label="5. Drought",             value="m5",
                children=module_5_layout()),
        dcc.Tab(label="6. Combined Stress",     value="m6",
                children=module_6_layout()),
    ]),
])

# ---------------------------------------------------------------------------
# Macro context figure builders  (Module 1 — three new panels)
# ---------------------------------------------------------------------------

_MACRO_VAR_COLORS = {
    "AUD/USD":            "#D4A93A",
    "RBA Rate (%)":       "#2E5C7A",
    "Fed Funds Rate (%)": "#7B3D5F",
    "AUS CPI (YoY %)":    COLORS["accent"],
    "US CPI (YoY %)":     COLORS["fail"],
}

_MACRO_VAR_FMT = {
    "AUD/USD":            ".4f",
    "RBA Rate (%)":       ".2f",
    "Fed Funds Rate (%)": ".2f",
    "AUS CPI (YoY %)":    ".2f",
    "US CPI (YoY %)":     ".2f",
}


def _build_macro_timeline_fig(primary_var: str, overlay_var: str,
                               sm: int, sy: int, em: int, ey: int) -> go.Figure:
    """
    User-chosen primary variable on left y-axis + optional overlay on right y-axis.
    Era shading is applied where it overlaps the selected window.
    """
    mask = _filter_dates(sm, sy, em, ey)
    df   = _macro_df.loc[mask]
    dts  = _dates[mask]

    fig = go.Figure()
    if df.empty:
        return fig

    # Primary series (left axis)
    p_color = _MACRO_VAR_COLORS.get(primary_var, "#888888")
    p_fmt   = _MACRO_VAR_FMT.get(primary_var, ".2f")
    if primary_var in df.columns:
        fig.add_trace(go.Scatter(
            x=dts, y=df[primary_var],
            name=primary_var,
            yaxis="y1",
            line=dict(color=p_color, width=2.5),
            hovertemplate=f"%{{x|%b %Y}}<br>{primary_var}: %{{y:{p_fmt}}}<extra></extra>",
        ))

    # Overlay series (right axis, optional; skip if same as primary)
    use_overlay = (
        overlay_var and overlay_var != "none"
        and overlay_var in df.columns
        and overlay_var != primary_var
    )
    if use_overlay:
        ov_color = _MACRO_VAR_COLORS.get(overlay_var, "#888888")
        ov_fmt   = _MACRO_VAR_FMT.get(overlay_var, ".2f")
        fig.add_trace(go.Scatter(
            x=dts, y=df[overlay_var],
            name=overlay_var,
            yaxis="y2",
            line=dict(color=ov_color, width=1.8, dash="dot"),
            hovertemplate=f"%{{x|%b %Y}}<br>{overlay_var}: %{{y:{ov_fmt}}}<extra></extra>",
        ))
        yaxis2_cfg = dict(
            title=dict(text=overlay_var, font=dict(size=11, color=ov_color)),
            overlaying="y", side="right",
            showgrid=False,
            tickfont=dict(size=11),
            zeroline=False,
        )
    else:
        yaxis2_cfg = dict(showticklabels=False, showgrid=False, overlaying="y")

    _add_era_shading(fig, sm, sy, em, ey)

    fig.update_layout(
        **CHART_LAYOUT,
        height=360,
        yaxis=dict(
            title=dict(text=primary_var, font=dict(size=11, color=p_color)),
            tickfont=dict(size=11),
            tickformat=p_fmt,
            zeroline=False,
        ),
        yaxis2=yaxis2_cfg,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)",
                    borderwidth=0, font=dict(size=12)),
        hovermode="x unified",
    )
    return fig


def _build_macro_corr_fig(sm: int, sy: int, em: int, ey: int) -> go.Figure:
    """
    Heatmap: 11 asset classes (rows) × 4 macro factors (columns).
    Pearson correlations computed for the selected period.
    """
    mask      = _filter_dates(sm, sy, em, ey)
    ret_slice = _returns_df_dt.loc[mask]
    mac_slice = _macro_df.loc[mask]

    fig = go.Figure()
    if ret_slice.shape[0] < 6:
        return fig

    corr_cols   = list(_MACRO_CORR_VARS.keys())
    disp_labels = list(_MACRO_CORR_VARS.values())

    # Drop the first row that pct_change / diff introduces as NaN
    mac_clean = mac_slice[corr_cols].dropna()
    ret_clean = ret_slice.loc[mac_clean.index]

    n_assets = len(tc.ASSET_CLASSES)
    n_macro  = len(corr_cols)
    z        = np.zeros((n_assets, n_macro))
    for j, mv in enumerate(corr_cols):
        for i, ac in enumerate(tc.ASSET_CLASSES):
            z[i, j] = ret_clean[ac].corr(mac_clean[mv])

    short_names = [tc.ASSET_CLASS_SHORT.get(ac, ac) for ac in tc.ASSET_CLASSES]

    fig.add_trace(go.Heatmap(
        z=z,
        x=disp_labels,
        y=short_names,
        zmin=-1, zmax=1,
        colorscale=[
            [0,   COLORS["heat_neg"]],
            [0.5, COLORS["heat_zero"]],
            [1,   COLORS["heat_pos"]],
        ],
        text=np.round(z, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        showscale=True,
        colorbar=dict(title="r", thickness=12, len=0.8,
                      tickfont=dict(size=10)),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))

    corr_layout = {k: v for k, v in CHART_LAYOUT.items() if k != "margin"}
    fig.update_layout(
        **corr_layout,
        height=420,
        margin=dict(l=130, r=80, t=30, b=70),
        xaxis=dict(side="bottom", tickfont=dict(size=11)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


def _build_domicile_regime_fig(assets: list, regime_type: str,
                               sm: int, sy: int, em: int, ey: int) -> go.Figure:
    """
    Grouped bar: annualised geometric return per asset, split into three regimes.

    regime_type:
      'aus_cpi'   – AUS CPI YoY: <2% Disinflationary / 2–3% Mild / >3% Inflationary
      'rba_rate'  – RBA Rate: Low / Mid / High tertiles from selected period
      'us_cpi'    – US CPI YoY: <1.7% Disinflationary / 1.7–2.3% Mild / >2.3% Inflationary
      'fed_funds' – Fed Funds Rate: Low / Mid / High tertiles from selected period
      'audusd'    – AUD/USD: Weak / Mid / Strong tertiles from selected period
    """
    fig = go.Figure()
    mask      = _filter_dates(sm, sy, em, ey)
    ret_slice = _returns_df_dt.loc[mask, assets]
    mac_slice = _macro_df.loc[mask]

    if ret_slice.shape[0] < 9:
        return fig

    if regime_type == "aus_cpi":
        series = mac_slice["AUS CPI (YoY %)"]
        regime_defs = [
            ("Disinflationary", series < 2.0,
             "#4A7FA8", "Disinflationary  (AUS CPI < 2%)"),
            ("Mild",            (series >= 2.0) & (series < 3.0),
             "#5E9E76", "Mild  (AUS CPI 2–3%)"),
            ("Inflationary",    series >= 3.0,
             "#A23737", "Inflationary  (AUS CPI > 3%)"),
        ]
    elif regime_type == "rba_rate":
        series = mac_slice["RBA Rate (%)"]
        q33 = float(series.quantile(1 / 3))
        q67 = float(series.quantile(2 / 3))
        regime_defs = [
            ("Low",  series < q33,
             "#4A7FA8", f"RBA Low  (< {q33:.2f}%)"),
            ("Mid",  (series >= q33) & (series < q67),
             "#5E9E76", f"RBA Mid  ({q33:.2f}–{q67:.2f}%)"),
            ("High", series >= q67,
             "#A23737", f"RBA High  (≥ {q67:.2f}%)"),
        ]
    elif regime_type == "us_cpi":
        series = mac_slice["US CPI (YoY %)"]
        regime_defs = [
            ("Disinflationary", series < 1.7,
             "#4A7FA8", "Disinflationary  (US CPI < 1.7%)"),
            ("Mild",            (series >= 1.7) & (series < 2.3),
             "#5E9E76", "Mild  (US CPI 1.7–2.3%)"),
            ("Inflationary",    series >= 2.3,
             "#A23737", "Inflationary  (US CPI > 2.3%)"),
        ]
    elif regime_type == "fed_funds":
        series = mac_slice["Fed Funds Rate (%)"]
        q33 = float(series.quantile(1 / 3))
        q67 = float(series.quantile(2 / 3))
        regime_defs = [
            ("Low",  series < q33,
             "#4A7FA8", f"Fed Funds Low  (< {q33:.2f}%)"),
            ("Mid",  (series >= q33) & (series < q67),
             "#5E9E76", f"Fed Funds Mid  ({q33:.2f}–{q67:.2f}%)"),
            ("High", series >= q67,
             "#A23737", f"Fed Funds High  (≥ {q67:.2f}%)"),
        ]
    else:  # audusd
        series = mac_slice["AUD/USD"]
        q33 = float(series.quantile(1 / 3))
        q67 = float(series.quantile(2 / 3))
        regime_defs = [
            ("Weak",   series < q33,
             "#A23737", f"Weak AUD  (< {q33:.3f})"),
            ("Mid",    (series >= q33) & (series < q67),
             "#A08040", f"Mid AUD  ({q33:.3f}–{q67:.3f})"),
            ("Strong", series >= q67,
             "#2E6B5E", f"Strong AUD  (≥ {q67:.3f})"),
        ]

    short_labels = [tc.ASSET_CLASS_SHORT.get(ac, ac) for ac in assets]

    for _key, r_mask, color, legend_label in regime_defs:
        df_r     = ret_slice.loc[r_mask]
        n_months = int(r_mask.sum())
        if n_months < 2:
            ann_rets = [0.0] * len(assets)
        else:
            ann_rets = (((1 + df_r).prod() ** (12 / n_months)) - 1).mul(100).tolist()

        fig.add_trace(go.Bar(
            name=legend_label,
            x=short_labels,
            y=ann_rets,
            marker_color=color,
            opacity=0.85,
            hovertemplate=(
                "%{x}<br>Ann. Return: %{y:.1f}%"
                f"<br>{legend_label}  ({n_months} months)<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_width=1, line_color=COLORS["border"])
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="group",
        height=360,
        yaxis=dict(
            title=dict(text="Ann. Return % p.a.", font=dict(size=11)),
            ticksuffix="%",
            tickfont=dict(size=11),
            zeroline=True, zerolinecolor=COLORS["border"],
        ),
        xaxis=dict(tickfont=dict(size=10)),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    fig.update_layout(margin=dict(l=55, r=15, t=20, b=60))
    return fig


# ---------------------------------------------------------------------------
# Callbacks — Module 1
# ---------------------------------------------------------------------------

def _compute_hist_rv_for_period(start_m: int, start_y: int,
                                 end_m: int, end_y: int) -> dict:
    """
    Return {asset_class: (geom_return_pct, ann_vol_pct)} for the chosen period.
    Used to refresh the shaded historical reference columns in the CMA table.
    """
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df_dt.loc[mask]
    if df_slice.empty:
        return {ac: (0.0, 0.0) for ac in tc.ASSET_CLASSES}
    n = len(df_slice)
    geom_ret = ((1 + df_slice).prod() ** (12 / n)) - 1
    ann_vol  = df_slice.std() * np.sqrt(12)
    return {ac: (round(float(geom_ret[ac]) * 100, 3),
                 round(float(ann_vol[ac])  * 100, 3))
            for ac in tc.ASSET_CLASSES}


def _rv_table_to_arrays(rv_data):
    """Read ONLY the forecast columns — historical columns are deliberately ignored."""
    by_ac = {row["asset_class"]: row for row in rv_data}
    returns, vols = [], []
    for ac in tc.ASSET_CLASSES:
        row = by_ac.get(ac, {})
        r = row.get("expected_return")
        v = row.get("volatility")
        try:
            r = float(r) / 100 if r not in (None, "") else 0.0
        except (TypeError, ValueError):
            r = 0.0
        try:
            v = float(v) / 100 if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            v = 0.0
        returns.append(r)
        vols.append(v)
    return np.array(returns), np.array(vols)


@app.callback(
    Output("cma-rv-table", "data"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
    State("cma-rv-table", "data"),
)
def update_cma_hist_columns(sm, sy, em, ey, current_data):
    """
    Refresh the grey Hist. Return / Hist. Vol columns and recompute the Δ
    column when the global period changes. Forecast columns are untouched.
    """
    sm = sm or _DATE_MIN_M;  sy = sy or _DATE_MIN_Y
    em = em or _DATE_MAX_M;  ey = ey or _DATE_MAX_Y
    hist = _compute_hist_rv_for_period(sm, sy, em, ey)
    updated = []
    for row in (current_data or _initial_cma_rv_data()):
        new_row = dict(row)
        ac = new_row["asset_class"]
        h_ret, h_vol = hist[ac]
        f_ret = new_row.get("expected_return", h_ret)
        new_row["hist_return"] = h_ret
        new_row["hist_vol"]    = h_vol
        new_row["delta"]       = round(float(f_ret) - h_ret, 3)
        updated.append(new_row)
    return updated


@app.callback(
    Output("cma-store", "data"),
    Input("cma-rv-table", "data"),
    Input("cpi-input", "value"),
    prevent_initial_call=False,
)
def update_cma_store(rv_data, cpi_pct):
    """
    Update cma-store when forecast columns or CPI change.
    Correlation matrix is always fixed to HIST_CORR.
    psd_adjusted is always False (historical corr is always PSD).
    """
    returns, vols = _rv_table_to_arrays(rv_data)
    try:
        cpi_decimal = float(cpi_pct) / 100 if cpi_pct is not None else 0.025
    except (TypeError, ValueError):
        cpi_decimal = 0.025
    return {
        "returns":      returns.tolist(),
        "vols":         vols.tolist(),
        "corr":         HIST_CORR.values.tolist(),
        "cpi":          cpi_decimal,
        "psd_adjusted": False,
    }

# ---------------------------------------------------------------------------
# Callbacks — Module 1 interactive EDA charts
# ---------------------------------------------------------------------------

# ── Monthly Returns Over Time ──────────────────────────────────────────────

@app.callback(
    Output("returns-time-chart", "figure"),
    Input("ret-asset-check", "value"),
    Input("ret-mode-radio",  "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_returns_time(selected_assets, return_mode, sm, sy, em, ey):
    return _build_returns_time_fig(
        selected_assets or [], return_mode or "monthly",
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("ret-asset-check", "value"),
    Input("ret-all-btn",  "n_clicks"),
    Input("ret-none-btn", "n_clicks"),
    State("ret-asset-check", "value"),
    prevent_initial_call=True,
)
def toggle_ret_assets(_all, _none, current):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    return tc.ASSET_CLASSES[:] if btn == "ret-all-btn" else []


# ── Cumulative Returns ─────────────────────────────────────────────────────

@app.callback(
    Output("cumulative-chart", "figure"),
    Input("cum-asset-check", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_cumulative(selected_assets, sm, sy, em, ey):
    return _build_cumulative_fig(
        selected_assets or [],
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("cum-asset-check", "value"),
    Input("cum-all-btn",  "n_clicks"),
    Input("cum-none-btn", "n_clicks"),
    State("cum-asset-check", "value"),
    prevent_initial_call=True,
)
def toggle_cum_assets(_all, _none, current):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    return tc.ASSET_CLASSES[:] if btn == "cum-all-btn" else []


# ── 12-Month Rolling Volatility ────────────────────────────────────────────

@app.callback(
    Output("rolling-vol-chart", "figure"),
    Input("vol-asset-check", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_rolling_vol(selected_assets, sm, sy, em, ey):
    return _build_rolling_vol_fig(
        selected_assets or [],
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("vol-asset-check", "value"),
    Input("vol-all-btn",  "n_clicks"),
    Input("vol-none-btn", "n_clicks"),
    State("vol-asset-check", "value"),
    prevent_initial_call=True,
)
def toggle_vol_assets(_all, _none, current):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    return tc.ASSET_CLASSES[:] if btn == "vol-all-btn" else []


# ── Risk-Return Scatter ────────────────────────────────────────────────────

@app.callback(
    Output("scatter-chart", "figure"),
    Input("scatter-asset-check", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_scatter_eda(selected_assets, sm, sy, em, ey):
    return _build_scatter_fig(
        selected_assets or [],
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("scatter-asset-check", "value"),
    Input("scatter-all-btn",  "n_clicks"),
    Input("scatter-none-btn", "n_clicks"),
    State("scatter-asset-check", "value"),
    prevent_initial_call=True,
)
def toggle_scatter_assets(_all, _none, current):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    return tc.ASSET_CLASSES[:] if btn == "scatter-all-btn" else []


# ── Descriptive Statistics ─────────────────────────────────────────────────

@app.callback(
    Output("desc-stats-table", "data"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_desc_stats(sm, sy, em, ey):
    return _build_desc_stats_data(
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("eda-period-note", "children"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_eda_note(sm, sy, em, ey):
    sm = sm or _DATE_MIN_M;  sy = sy or _DATE_MIN_Y
    em = em or _DATE_MAX_M;  ey = ey or _DATE_MAX_Y
    period = (f"{_MONTH_NAMES[int(sm)-1]} {sy} "
              f"to {_MONTH_NAMES[int(em)-1]} {ey}")
    return (f"All charts and tables reflect the Analysis Period set above. "
            f"Source: Refinitiv monthly return series ({period}). "
            "Read-only — does not affect any calculations.")


# ── Monthly Return Distributions ───────────────────────────────────────────

@app.callback(
    Output("m1-histograms", "figure"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_histograms_eda(sm, sy, em, ey):
    return _build_histograms_fig(
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


# ── Correlation Matrix ─────────────────────────────────────────────────────

@app.callback(
    Output("m1-corr-eda", "figure"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_corr_eda(sm, sy, em, ey):
    return _build_corr_heatmap_eda_fig(
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


# ── Macro Context callbacks ────────────────────────────────────────────────

@app.callback(
    Output("m1-macro-timeline", "figure"),
    Input("m1-macro-primary", "value"),
    Input("m1-macro-overlay", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_macro_timeline(primary, overlay, sm, sy, em, ey):
    return _build_macro_timeline_fig(
        primary or "AUD/USD",
        overlay or "none",
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("m1-macro-corr", "figure"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_macro_corr(sm, sy, em, ey):
    return _build_macro_corr_fig(
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("m1-aus-regime-chart", "figure"),
    Input("m1-aus-regime-dd", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_aus_regime_chart(regime_type, sm, sy, em, ey):
    return _build_domicile_regime_fig(
        _AUS_ASSETS,
        regime_type or "aus_cpi",
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


@app.callback(
    Output("m1-us-regime-chart", "figure"),
    Input("m1-us-regime-dd", "value"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_us_regime_chart(regime_type, sm, sy, em, ey):
    return _build_domicile_regime_fig(
        _GLOBAL_ASSETS,
        regime_type or "us_cpi",
        sm or _DATE_MIN_M, sy or _DATE_MIN_Y,
        em or _DATE_MAX_M, ey or _DATE_MAX_Y,
    )


# ---------------------------------------------------------------------------
# Callbacks — Module 2
# ---------------------------------------------------------------------------

def _store_to_arrays(store):
    returns = np.asarray(store["returns"], dtype=float)
    vols    = np.asarray(store["vols"],    dtype=float)
    corr    = np.asarray(store["corr"],    dtype=float)
    cpi     = float(store["cpi"])
    return returns, vols, corr, cpi


@app.callback(
    Output("m2-trust-cards",     "children"),
    Output("m2-trust-role-cards", "children"),
    Output("m2-corr-heatmap",    "figure"),
    Output("m2-comparison-chart","figure"),
    Output("m2-cfo-table-1",     "data"),
    Output("m2-cfo-table-2",     "data"),
    Input("cma-store", "data"),
)
def update_module_2(store):
    if not store:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, [], []
    returns, vols, corr, cpi = _store_to_arrays(store)
    cov        = tc.cma_to_covariance(vols, corr)
    cash_return = float(returns[0])
    chars      = tc.trust_characteristics(returns, cov, cash_return, cpi)

    cards      = html.Div([_trust_card(t, chars[t]) for t in tc.TRUST_NAMES],
                          className="trust-row")
    role_cards = html.Div([
        html.Div([
            html.P(t, className="trust-name"),
            html.Div(TRUST_ROLES[t], className="trust-tag"),
            html.Div([
                html.Span("Return ", className="k"),
                html.Span(_fmt_pct(chars[t]["net_return"]), className="v"),
                html.Span("Risk ", className="k"),
                html.Span(_fmt_pct(chars[t]["volatility"]), className="v"),
                html.Span("Portfolio use ", className="k"),
                html.Span("Liquidity" if t == "STI" else ("Balanced growth" if t == "MTG" else "Return engine"),
                          className="v"),
            ], className="kv-grid"),
        ], className="trust-card", style={"--trust-accent": COLORS[t]})
        for t in tc.TRUST_NAMES
    ], className="trust-row")
    heatmap    = correlation_heatmap_figure(corr)
    comparison = trust_comparison_figure(chars)

    table1_rows = []
    for i, ac in enumerate(tc.ASSET_CLASSES):
        forecast   = float(returns[i])
        historical = float(HIST_GEOM_ANNUAL_RETURNS[ac])
        hist_risk  = float(HIST_ANNUAL_VOL[ac])
        table1_rows.append({
            "asset_class":     ac,
            "historical_return": _fmt_pct(historical),
            "forecast_return":   _fmt_pct(forecast),
            "difference":        _fmt_signed_pct(forecast - historical),
            "historical_risk":   _fmt_pct(hist_risk),
        })

    table2_rows = [
        {"metric": "Expected Return",
         **{t: _fmt_pct(chars[t]["net_return"]) for t in tc.TRUST_NAMES}},
        {"metric": "Risk",
         **{t: _fmt_pct(chars[t]["volatility"]) for t in tc.TRUST_NAMES}},
    ]
    return cards, role_cards, heatmap, comparison, table1_rows, table2_rows


@app.callback(
    Output("m2-cfo-table-3", "data"),
    Input("cma-store", "data"),
    Input("portfolio-allocation-store", "data"),
)
def update_module_2_table_3(cma_store, alloc):
    if not cma_store or not alloc:
        return []
    returns, vols, corr, cpi = _store_to_arrays(cma_store)
    cov = tc.cma_to_covariance(vols, corr)
    w   = {t: float(alloc.get(t, 0.0)) for t in tc.TRUST_NAMES}
    total = sum(w.values())
    if total <= 0:
        w = {t: 1 / 3 for t in tc.TRUST_NAMES}
    else:
        w = {t: v / total for t, v in w.items()}
    p_return = tc.portfolio_net_return(w, returns)
    p_vol    = tc.portfolio_volatility(w, cov)
    mix_str  = (f"STI: {w['STI']*100:.1f}%; "
                f"MTG: {w['MTG']*100:.1f}%; "
                f"LTG: {w['LTG']*100:.1f}%")
    return [{"field": "Recommended mix of Unit Trusts", "value": mix_str},
            {"field": "Forecast Return", "value": _fmt_pct(p_return)},
            {"field": "Forecast Risk",   "value": _fmt_pct(p_vol)}]

# ---------------------------------------------------------------------------
# Callbacks — Module 3
# ---------------------------------------------------------------------------

def _rebalance_other_two(fixed_val, other_a, other_b):
    budget = max(0.0, 100.0 - fixed_val)
    s = other_a + other_b
    if s <= 1e-9:
        return budget / 2, budget / 2
    return (other_a / s) * budget, (other_b / s) * budget


@app.callback(
    Output("proposed-STI", "value"),
    Output("proposed-MTG", "value"),
    Output("proposed-LTG", "value"),
    Input("proposed-STI", "value"),
    Input("proposed-MTG", "value"),
    Input("proposed-LTG", "value"),
    prevent_initial_call=True,
)
def rebalance_proposed(sti, mtg, ltg):
    trigger = callback_context.triggered_id
    if trigger is None:
        return dash.no_update, dash.no_update, dash.no_update
    sti = sti or 0; mtg = mtg or 0; ltg = ltg or 0
    if trigger == "proposed-STI":
        new_mtg, new_ltg = _rebalance_other_two(sti, mtg, ltg)
        if abs(new_mtg - mtg) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return dash.no_update, round(new_mtg), round(new_ltg)
    if trigger == "proposed-MTG":
        new_sti, new_ltg = _rebalance_other_two(mtg, sti, ltg)
        if abs(new_sti - sti) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return round(new_sti), dash.no_update, round(new_ltg)
    if trigger == "proposed-LTG":
        new_sti, new_mtg = _rebalance_other_two(ltg, sti, mtg)
        if abs(new_sti - sti) < 0.5 and abs(new_mtg - mtg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return round(new_sti), round(new_mtg), dash.no_update
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("portfolio-allocation-store", "data"),
    Input("proposed-STI", "value"),
    Input("proposed-MTG", "value"),
    Input("proposed-LTG", "value"),
)
def push_proposed_to_store(sti, mtg, ltg):
    sti = (sti or 0) / 100; mtg = (mtg or 0) / 100; ltg = (ltg or 0) / 100
    total = sti + mtg + ltg
    if total <= 0:
        return {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    return {"STI": sti/total, "MTG": mtg/total, "LTG": ltg/total}


def _format_sum_label(total_pct, prefix="Total"):
    if abs(total_pct - 100) < 0.5:
        return f"{prefix}: 100.0% \u2713", "alloc-sum"
    return f"{prefix}: {total_pct:.1f}% (will be normalised)", "alloc-sum alloc-sum-bad"


@app.callback(
    Output("current-STI-display", "children"),
    Output("current-MTG-display", "children"),
    Output("current-LTG-display", "children"),
    Output("current-sum", "children"),
    Output("current-sum", "className"),
    Output("proposed-STI-display", "children"),
    Output("proposed-MTG-display", "children"),
    Output("proposed-LTG-display", "children"),
    Output("proposed-sum", "children"),
    Output("proposed-sum", "className"),
    Output("m3-live-return", "children"),
    Output("m3-live-vol",    "children"),
    Output("m3-live-sharpe", "children"),
    Output("m3-constraints", "children"),
    Output("m3-volcap-wrapper", "style"),
    Input("current-STI", "value"),
    Input("current-MTG", "value"),
    Input("current-LTG", "value"),
    Input("proposed-STI", "value"),
    Input("proposed-MTG", "value"),
    Input("proposed-LTG", "value"),
    Input("cma-store", "data"),
    Input("m3-objective", "value"),
)
def update_live(c_sti, c_mtg, c_ltg, p_sti, p_mtg, p_ltg, store, objective):
    if not store:
        return [dash.no_update] * 15
    c_sti = c_sti or 0; c_mtg = c_mtg or 0; c_ltg = c_ltg or 0
    p_sti = p_sti or 0; p_mtg = p_mtg or 0; p_ltg = p_ltg or 0
    c_total = c_sti + c_mtg + c_ltg
    p_total = p_sti + p_mtg + p_ltg
    c_sum_text, c_sum_cls = _format_sum_label(c_total, "Current total")
    p_sum_text, p_sum_cls = _format_sum_label(p_total, "Proposed total")
    if p_total <= 0:
        w_proposed = {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    else:
        w_proposed = {"STI": p_sti/p_total, "MTG": p_mtg/p_total, "LTG": p_ltg/p_total}
    returns, vols, corr, cpi = _store_to_arrays(store)
    cov  = tc.cma_to_covariance(vols, corr)
    cash = float(returns[0])
    p_return = tc.portfolio_net_return(w_proposed, returns)
    p_vol    = tc.portfolio_volatility(w_proposed, cov)
    p_sharpe = (p_return - cash) / p_vol if p_vol > 0 else float("nan")
    liq    = mt.liquidity_coverage(w_proposed)
    target = cpi + op.TARGET_SPREAD
    meets_target = p_return >= target - 1e-9

    def _pill(label, ok):
        return html.Div([html.Span(label),
                         html.Span("PASS" if ok else "FAIL",
                                   className="pill " + ("pill-pass" if ok else "pill-fail"))],
                        className="constraint-item")

    constraints = [
        _pill(f"12m liquidity (\u2265 10%): {liq['within_12m']*100:.1f}%", liq["meets_12m"]),
        _pill(f"3y liquidity (\u2265 25%): {liq['within_3y']*100:.1f}%",   liq["meets_3y"]),
        _pill(f"Return target (\u2265 {target*100:.2f}%): {p_return*100:.2f}%", meets_target),
    ]
    volcap_style = {"display": "block"} if objective == "max_return" else {"display": "none"}
    return (
        f"{c_sti:.1f}%", f"{c_mtg:.1f}%", f"{c_ltg:.1f}%",
        c_sum_text, c_sum_cls,
        f"{p_sti:.0f}%", f"{p_mtg:.0f}%", f"{p_ltg:.0f}%",
        p_sum_text, p_sum_cls,
        _fmt_pct(p_return), _fmt_pct(p_vol),
        f"{p_sharpe:.3f}" if not np.isnan(p_sharpe) else "—",
        constraints, volcap_style,
    )


@app.callback(
    Output("m3-scatter", "figure"),
    Input("cma-store", "data"),
    Input("portfolio-allocation-store", "data"),
    Input("current-STI", "value"),
    Input("current-MTG", "value"),
    Input("current-LTG", "value"),
    Input("m3-opt-store", "data"),
)
def update_scatter(store, alloc, c_sti, c_mtg, c_ltg, opt_data):
    if not store:
        return go.Figure()
    returns, vols, corr, cpi = _store_to_arrays(store)
    cov  = tc.cma_to_covariance(vols, corr)
    cash = float(returns[0])
    grid      = op.generate_grid()
    grid_eval = op.evaluate_grid(grid, returns, cov, cash)
    target    = cpi + op.TARGET_SPREAD

    def metrics_for(w):
        return {"weights": w,
                "ret": tc.portfolio_net_return(w, returns),
                "vol": tc.portfolio_volatility(w, cov)}

    c_total = (c_sti or 0) + (c_mtg or 0) + (c_ltg or 0)
    current_marker = (metrics_for({"STI": (c_sti or 0)/c_total,
                                   "MTG": (c_mtg or 0)/c_total,
                                   "LTG": (c_ltg or 0)/c_total})
                      if c_total > 0 else None)
    proposed_marker = metrics_for(alloc) if alloc and sum(alloc.values()) > 0 else None
    optimal_marker  = (metrics_for(opt_data["weights"])
                       if opt_data and opt_data.get("feasible") else None)
    return _scatter_figure(grid_eval, target, current_marker, proposed_marker, optimal_marker)


@app.callback(
    Output("m3-board-compliance", "children"),
    Input("cma-store", "data"),
    Input("portfolio-allocation-store", "data"),
)
def update_board_compliance(cma_store, alloc):
    if not cma_store:
        return html.Div()
    metrics = _portfolio_metrics_from_store(cma_store, alloc or {})
    return html.Div([
        _board_compliance_table(metrics),
        html.Div(
            "Rubric note: the 'moderate-high risk appetite' item should be explained in words, "
            "not treated as a pure numeric pass/fail. Link it to the Fund's long horizon, "
            "episodic withdrawals, and resilience under stress.",
            className="hist-note",
        ),
    ])


def _opt_result_card(opt, current_w, proposed_w, target):
    if not opt.feasible:
        return html.Div([html.Strong("Infeasible. "), opt.message],
                        className="opt-infeasible")
    tx = mt.transaction_cost_aud(current_w, opt.weights, PORTFOLIO_AUD,
                                  tc.TRUST_BUY_SPREADS, tc.TRUST_SELL_SPREADS)
    meets_target = opt.net_return >= target - 1e-9
    pill_cls = "pill " + ("pill-pass" if meets_target else "pill-fail")
    return html.Div([
        html.Div([
            html.Div([
                html.Div(OBJECTIVE_LABELS[opt.objective],
                         style={"fontSize": "12px", "color": COLORS["muted"],
                                "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                html.Div(f"STI: {opt.weights['STI']*100:.1f}%   "
                         f"MTG: {opt.weights['MTG']*100:.1f}%   "
                         f"LTG: {opt.weights['LTG']*100:.1f}%",
                         style={"fontFamily": MONO_STACK, "fontSize": "16px",
                                "fontWeight": "600", "marginTop": "4px"}),
                html.Div([
                    html.Span(f"Return {_fmt_pct(opt.net_return)}", style={"marginRight": "16px"}),
                    html.Span(f"Vol {_fmt_pct(opt.volatility)}",    style={"marginRight": "16px"}),
                    html.Span(f"Sharpe {opt.sharpe:.3f}",           style={"marginRight": "16px"}),
                    html.Span("MEETS TARGET" if meets_target else "BELOW TARGET", className=pill_cls),
                ], style={"marginTop": "8px", "fontSize": "13px", "fontFamily": MONO_STACK}),
            ], style={"flex": "1 1 auto"}),
            html.Button("Apply to proposed sliders", id="m3-apply-button",
                        className="opt-button opt-button-secondary",
                        n_clicks=0, style={"marginLeft": "16px"}),
        ], style={"display": "flex", "alignItems": "flex-start"}),
        html.Div([
            html.Div("Transaction costs (current \u2192 optimal, $3B base)",
                     style={"fontSize": "12px", "color": COLORS["muted"],
                            "textTransform": "uppercase", "letterSpacing": "0.05em",
                            "marginTop": "14px", "marginBottom": "6px"}),
            html.Div([
                html.Span("Sell spread cost", className="k"),
                html.Span(_fmt_aud(tx["sell_cost_aud"]), className="v"),
                html.Span("Buy spread cost", className="k"),
                html.Span(_fmt_aud(tx["buy_cost_aud"]), className="v"),
                html.Span("Round-trip total", className="k"),
                html.Span(_fmt_aud(tx["total_cost_aud"]), className="v",
                          style={"fontWeight": "600"}),
                html.Span("As fraction of fund", className="k"),
                html.Span(f"{tx['total_cost_aud'] / PORTFOLIO_AUD * 100:.3f}%", className="v"),
            ], className="tx-grid"),
        ]),
    ], className="opt-result-card")


@app.callback(
    Output("m3-opt-store",   "data"),
    Output("m3-opt-result",  "children"),
    Input("m3-run-button",   "n_clicks"),
    State("m3-objective",    "value"),
    State("m3-volcap",       "value"),
    State("cma-store",       "data"),
    State("current-STI",     "value"),
    State("current-MTG",     "value"),
    State("current-LTG",     "value"),
    State("portfolio-allocation-store", "data"),
    prevent_initial_call=True,
)
def run_optimiser(n_clicks, objective, volcap_pct, store, c_sti, c_mtg, c_ltg, alloc):
    if not n_clicks or not store:
        return dash.no_update, dash.no_update
    returns, vols, corr, cpi = _store_to_arrays(store)
    cov  = tc.cma_to_covariance(vols, corr)
    cash = float(returns[0])
    target   = cpi + op.TARGET_SPREAD
    vol_cap  = (volcap_pct / 100) if volcap_pct is not None else None
    try:
        result = op.optimise(objective, returns, cov, cash, cpi,
                             vol_cap=vol_cap if objective == "max_return" else None)
    except Exception as e:
        return None, html.Div([html.Strong("Optimisation error. "), str(e)],
                               className="opt-infeasible")
    c_total = (c_sti or 0) + (c_mtg or 0) + (c_ltg or 0)
    if c_total > 0:
        current_w = {"STI": (c_sti or 0)/c_total,
                     "MTG": (c_mtg or 0)/c_total,
                     "LTG": (c_ltg or 0)/c_total}
    else:
        current_w = {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    card = _opt_result_card(result, current_w, alloc or {}, target)
    return result.to_dict(), card


@app.callback(
    Output("proposed-STI", "value", allow_duplicate=True),
    Output("proposed-MTG", "value", allow_duplicate=True),
    Output("proposed-LTG", "value", allow_duplicate=True),
    Input("m3-apply-button", "n_clicks"),
    State("m3-opt-store",    "data"),
    prevent_initial_call=True,
)
def apply_optimised(n_clicks, opt_data):
    if not n_clicks or not opt_data or not opt_data.get("feasible"):
        return dash.no_update, dash.no_update, dash.no_update
    w = opt_data["weights"]
    return round(w["STI"] * 100), round(w["MTG"] * 100), round(w["LTG"] * 100)


@app.callback(
    Output("m3-tornado", "figure"),
    Input("cma-store",    "data"),
    Input("m3-objective", "value"),
    Input("m3-volcap",    "value"),
)
def update_tornado(store, objective, volcap_pct):
    if not store:
        return go.Figure()
    returns, vols, corr, cpi = _store_to_arrays(store)
    cov     = tc.cma_to_covariance(vols, corr)
    cash    = float(returns[0])
    vol_cap = (volcap_pct / 100) if volcap_pct is not None else None
    sens    = op.sensitivity_sweep(objective, returns, cov, cash, cpi,
                                   vol_cap=vol_cap if objective == "max_return" else None)
    baseline = sens.attrs.get("baseline")
    if baseline is None or not baseline.feasible:
        fig = go.Figure()
        fig.add_annotation(
            text="Baseline optimisation is infeasible under current CMAs. "
                 "Sensitivity sweep cannot be run.",
            showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(family=FONT_STACK, size=14, color=COLORS["fail"]))
        fig.update_layout(height=300, plot_bgcolor=COLORS["panel"],
                          paper_bgcolor=COLORS["panel"])
        return fig
    return _tornado_figure(sens, baseline.volatility, objective)


# ---------------------------------------------------------------------------
# Callbacks — Module 4
# ---------------------------------------------------------------------------

_HIST_BASELINE = np.asarray(HIST_ARITH_ANNUAL_RETURNS.values, dtype=float)
_PRECOMPUTED_SCENARIOS = st.build_all_scenarios(_returns_df, _HIST_BASELINE)
SCENARIO_WINDOWS_LIVE = set(st.SCENARIO_WINDOWS.keys())


def _scenario_defaults(scenario_name: str,
                        cma_baseline: np.ndarray) -> tuple:
    if scenario_name in SCENARIO_WINDOWS_LIVE:
        sc = _PRECOMPUTED_SCENARIOS[scenario_name]
        return sc.asset_returns, sc.description, sc.window_label
    if scenario_name == "AUD Depreciation Shock":
        sc = st.build_aud_shock_scenario(_returns_df, cma_baseline)
        return sc.asset_returns, sc.description, sc.window_label
    if scenario_name == "Interest Rate Shock (+200bps)":
        sc = st.build_rate_shock_scenario(cma_baseline)
        return sc.asset_returns, sc.description, None
    raise ValueError(f"Unknown scenario: {scenario_name}")


@app.callback(
    Output("m4-shock-table",   "data"),
    Output("m4-shocked-store", "data"),
    Output("m4-scenario-meta", "children"),
    Input("m4-scenario",       "value"),
    Input("m4-reset-button",   "n_clicks"),
    State("cma-store",         "data"),
)
def update_m4_scenario(scenario_name, n_clicks, cma_store):
    if not cma_store or not scenario_name:
        return dash.no_update, dash.no_update, dash.no_update
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    shocked, desc, window_label = _scenario_defaults(scenario_name, cma_baseline)
    rows = _shock_table_initial_rows(cma_baseline, shocked)
    meta_children = [
        html.Div("Scenario", className="meta-label"),
        html.Div(scenario_name, style={"fontSize": "16px", "fontWeight": "600",
                                        "marginBottom": "8px"}),
        html.Div(desc),
    ]
    return rows, shocked.tolist(), meta_children


@app.callback(
    Output("m4-shocked-store", "data", allow_duplicate=True),
    Input("m4-shock-table",    "data"),
    State("m4-shocked-store",  "data"),
    prevent_initial_call=True,
)
def update_m4_overrides(table_data, prev_shocked):
    if not table_data:
        return dash.no_update
    new_shocked = []
    for i, row in enumerate(table_data):
        try:
            v = float(row.get("shocked")) / 100
        except (TypeError, ValueError):
            v = float(prev_shocked[i]) if prev_shocked else 0.0
        new_shocked.append(v)
    return new_shocked


@app.callback(
    Output("m4-shock-table", "data", allow_duplicate=True),
    Input("m4-shock-table",  "data_timestamp"),
    State("m4-shock-table",  "data"),
    State("cma-store",       "data"),
    prevent_initial_call=True,
)
def update_m4_delta_column(_, table_data, cma_store):
    if not table_data or not cma_store:
        return dash.no_update
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    new_rows = []
    changed = False
    for i, row in enumerate(table_data):
        try:
            shocked_pct = float(row.get("shocked"))
        except (TypeError, ValueError):
            shocked_pct = float(row.get("baseline") or 0)
        baseline_pct = float(cma_baseline[i] * 100)
        delta_pct = round(shocked_pct - baseline_pct, 3)
        if abs(delta_pct - float(row.get("delta") or 0)) > 1e-9:
            changed = True
        new_row = dict(row)
        new_row["baseline"] = round(baseline_pct, 3)
        new_row["delta"] = delta_pct
        new_rows.append(new_row)
    if not changed and all(
        abs(float(table_data[i]["baseline"]) - float(cma_baseline[i] * 100)) < 1e-9
        for i in range(len(tc.ASSET_CLASSES))
    ):
        return dash.no_update
    return new_rows


@app.callback(
    Output("m4-compare-chart", "figure"),
    Output("m4-factor-table",  "children"),
    Output("m4-verdict",       "children"),
    Input("m4-shocked-store",  "data"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",         "data"),
    State("m4-scenario",       "value"),
)
def update_m4_outputs(shocked, alloc, cma_store, scenario_name):
    if not shocked or not cma_store:
        return go.Figure(), html.Div(), ""
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    shocked_arr  = np.asarray(shocked, dtype=float)
    if alloc and sum(alloc.values()) > 0:
        w = {t: alloc.get(t, 0) / sum(alloc.values()) for t in tc.TRUST_NAMES}
    else:
        w = {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    fig = shock_compare_figure(cma_baseline, shocked_arr, w)
    window_label = None
    if scenario_name and scenario_name in SCENARIO_WINDOWS_LIVE:
        s, e = st.SCENARIO_WINDOWS[scenario_name]
        window_label = f"{s} \u2013 {e}"
    elif scenario_name == "AUD Depreciation Shock":
        sc = _PRECOMPUTED_SCENARIOS.get(scenario_name)
        if sc and sc.window_label:
            window_label = sc.window_label
    rows = _factor_breakdown_rows(shocked_arr, _returns_df, scenario_name, window_label)
    trust_nets = st.trust_returns_under_shock(shocked_arr)
    portfolio_stress = sum(w[t] * trust_nets[t] for t in tc.TRUST_NAMES)
    worst_trust = min(trust_nets, key=trust_nets.get)
    verdict = (
        f"{scenario_name} stress implies a portfolio stressed return of "
        f"{_fmt_pct(portfolio_stress)}. {worst_trust} is the most exposed trust "
        f"({_fmt_pct(trust_nets[worst_trust])}), so the CFO narrative should explain "
        "whether the recommended allocation is using STI for stability while accepting "
        "MTG/LTG drawdown risk for long-term CPI+ return capacity."
    )
    factor_header = html.Tr([
        html.Th("Trust"),
        html.Th("Net Return Under Stress", style={"textAlign": "right"}),
        html.Th("Dominant Factor"),
        html.Th("Window Drawdown", style={"textAlign": "right"}),
    ])
    body_rows = []
    for r in rows:
        body_rows.append(html.Tr([
            html.Td(r["trust"], className="trust-cell",
                    style={"--trust-accent": COLORS[r["trust"]]}),
            html.Td(r["net_return"], className="num"),
            html.Td(html.Span(r["dominant_factor"],
                              className=f"factor-tag {_factor_class(r['dominant_factor'])}")),
            html.Td(r["window_drawdown"], className="num"),
        ]))
    factor_table = html.Table(
        [html.Thead(factor_header), html.Tbody(body_rows)],
        className="factor-table",
    )
    note = html.Div(html.Em(
        "\u2018Dominant factor\u2019 is the asset-class group with the largest absolute "
        "contribution (weight \u00d7 shocked return) to the trust\u2019s gross return "
        "under the shock."),
        style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "12px"})
    return fig, html.Div([factor_table, note]), verdict


# ---------------------------------------------------------------------------
# Callbacks — Module 5
# ---------------------------------------------------------------------------

@app.callback(
    Output("m5-relief", "min"),
    Output("m5-relief", "max"),
    Output("m5-relief", "value"),
    Output("m5-relief", "marks"),
    Input("m5-severity", "value"),
    State("m5-relief",   "value"),
)
def update_relief_bounds(severity, current_value):
    lo, hi, default = _severity_slider_bounds(severity)
    if current_value is None or current_value < lo or current_value > hi:
        new_val = default
    else:
        new_val = current_value
    marks = {lo: f"${lo}M", (lo + hi) // 2: f"${(lo + hi) // 2}M", hi: f"${hi}M"}
    return lo, hi, new_val, marks


@app.callback(
    Output("m5-value-chart",                "figure"),
    Output("m5-composition-chart",          "figure"),
    Output("m5-exec-verdict",               "children"),
    Output("m5-summary-card",               "children"),
    Output("m5-projection-table-container", "children"),
    Output("m5-totals",                     "children"),
    Output("m5-config-summary",             "children"),
    Output("m5-onset-split-summary",        "children"),
    Input("m5-severity",  "value"),
    Input("m5-relief",    "value"),
    Input("m5-onset",     "value"),
    Input("m5-fraction",  "value"),
    Input("m5-onset-split-STI", "value"),
    Input("m5-onset-split-MTG", "value"),
    Input("m5-onset-split-LTG", "value"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",    "data"),
)
def update_module_5(severity, relief_m, onset, fraction_pct,
                    split_sti, split_mtg, split_ltg, alloc, cma_store):
    if not cma_store or not alloc or relief_m is None or onset is None:
        return go.Figure(), go.Figure(), "", html.Div(), html.Div(), html.Div(), "", ""
    relief_aud = float(relief_m) * 1e6
    onset      = int(onset)
    fraction   = max(0.0, min(1.0, float(fraction_pct or 50) / 100))
    onset_split = _onset_split_from_inputs(split_sti, split_mtg, split_ltg)
    schedule   = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                     year_4_fraction=fraction, residual_split=(0.5, 0.5))
    total_w = sum(alloc.values())
    weights = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
               if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns = np.asarray(cma_store["returns"], dtype=float)
    result  = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                         drawdown_splits={onset: onset_split})
    value_fig   = _projection_value_figure(result, onset)
    comp_fig    = _trust_composition_figure(result)
    summary     = dr.post_drawdown_summary(result, onset)
    summary_card = _summary_card(summary, result.total_drawdown, result.total_spread_cost)
    proj_table  = _projection_summary_table(result)
    yrs_breach  = sum(1 for y in result.years if not (y.meets_12m and y.meets_3y))
    onset_mix = "—"
    if summary:
        onset_mix = (f"STI {summary['ending_weights']['STI']*100:.1f}% / "
                     f"MTG {summary['ending_weights']['MTG']*100:.1f}% / "
                     f"LTG {summary['ending_weights']['LTG']*100:.1f}%")
    drought_verdict = (
        f"At the Year {onset} severe-drought drawdown, the portfolio retains "
        f"{_fmt_aud(summary['remaining_value']) if summary else '—'} with a post-drawdown mix of "
        f"{onset_mix}. It {'can' if summary and summary['can_sustain_residual'] else 'cannot'} "
        "cover the scheduled residual drawdowns over the next two years without exhausting the Fund. "
        f"The model flags {yrs_breach} year(s) where policy liquidity thresholds are breached, "
        "which should be treated as a key trade-off in the executive deck."
    )
    totals = html.Div([html.Div([
        html.Span("Final value: ",         style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_aud(result.final_value),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total drawdown: ",      style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_aud(result.total_drawdown),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total spread cost: ",   style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_aud(result.total_spread_cost),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Years with liquidity breach: ",
                  style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(f"{yrs_breach}",
                  style={"fontFamily": MONO_STACK, "fontWeight": "600",
                         "color": COLORS["fail"] if yrs_breach > 0 else COLORS["pass"]}),
    ])])
    config_text = (
        f"Severity: {severity}. Total relief: {_fmt_aud(relief_aud)}. "
        f"Onset year {onset}, with {fraction*100:.0f}% of relief in Year {onset} "
        f"and the remaining {(1-fraction)*100:.0f}% split equally across Years "
        f"{onset+1} and {onset+2}. "
    )
    if result.fund_exhausted:
        config_text += f"FUND EXHAUSTED in Year {result.exhaustion_year}."
    onset_drawdown = schedule.get(onset, 0.0)
    split_summary = (
        f"Year {onset} drawdown {_fmt_aud(onset_drawdown)} is targeted as: "
        f"STI {_fmt_aud(onset_drawdown * onset_split['STI'])} "
        f"({onset_split['STI']*100:.1f}%), "
        f"MTG {_fmt_aud(onset_drawdown * onset_split['MTG'])} "
        f"({onset_split['MTG']*100:.1f}%), "
        f"LTG {_fmt_aud(onset_drawdown * onset_split['LTG'])} "
        f"({onset_split['LTG']*100:.1f}%)."
    )
    return (value_fig, comp_fig, drought_verdict, summary_card, proj_table,
            totals, config_text, split_summary)


# ---------------------------------------------------------------------------
# Callbacks — Module 5b (Monte Carlo)
# ---------------------------------------------------------------------------

def _mc_fan_figure(mc: dr.MonteCarloResult, initial_value: float) -> go.Figure:
    bands = mc.percentile_bands()
    years = list(range(mc.horizon + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years + years[::-1],
        y=list(bands["p95"]) + list(bands["p5"][::-1]),
        fill="toself", fillcolor="rgba(58, 107, 94, 0.10)",
        line=dict(width=0), name="5-95 pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years + years[::-1],
        y=list(bands["p75"]) + list(bands["p25"][::-1]),
        fill="toself", fillcolor="rgba(58, 107, 94, 0.22)",
        line=dict(width=0), name="25-75 pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=bands["p50"], mode="lines", name="Median (P50)",
        line=dict(color=COLORS["accent"], width=2.5),
        hovertemplate="Year %{x}<br>P50: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=bands["p5"], mode="lines", name="P5",
        line=dict(color=COLORS["fail"], width=1, dash="dot"),
        hovertemplate="Year %{x}<br>P5: $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=bands["p95"], mode="lines", name="P95",
        line=dict(color=COLORS["pass"], width=1, dash="dot"),
        hovertemplate="Year %{x}<br>P95: $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=initial_value, line=dict(color=COLORS["muted"], width=1, dash="dash"),
        annotation_text="Starting value", annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=400, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value (AUD)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$.2s", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _mc_exhaustion_figure(mc: dr.MonteCarloResult) -> go.Figure:
    cum_exhausted = np.zeros(mc.horizon + 1)
    for y in range(1, mc.horizon + 1):
        cum_exhausted[y] = ((mc.exhaustion_year >= 1) & (mc.exhaustion_year <= y)).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(mc.horizon + 1)), y=cum_exhausted,
        marker_color=COLORS["fail"], opacity=0.65,
        text=[f"{v*100:.1f}%" if v > 0 else "" for v in cum_exhausted],
        textposition="outside", textfont=dict(family=MONO_STACK, size=11),
        hovertemplate="Year %{x}<br>Cumulative P(exhaustion): %{y:.2%}<extra></extra>",
        name="Cumulative exhaustion"))
    fig.update_layout(height=260, margin=dict(l=70, r=20, t=20, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="P(fund exhausted by year)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat=".1%", tickfont=dict(size=11),
                   range=[0, max(0.05, float(cum_exhausted.max()) * 1.25)]),
        showlegend=False)
    return fig


def _mc_summary_strip(mc: dr.MonteCarloResult, initial_value: float) -> html.Div:
    bands = mc.percentile_bands()
    p_exhaust  = mc.exhaustion_probability()
    above_start = (mc.values[:, -1] >= initial_value).mean()
    return html.Div([
        html.Div([html.Div("Paths", className="lbl"),
                  html.Div(f"{mc.n_paths:,}", className="val")], className="summary-item"),
        html.Div([html.Div("P(fund exhausted by Y10)", className="lbl"),
                  html.Div(f"{p_exhaust*100:.2f}%", className="val",
                           style={"color": COLORS["fail"] if p_exhaust > 0.05 else COLORS["pass"]})],
                 className="summary-item"),
        html.Div([html.Div("P(final \u2265 starting)", className="lbl"),
                  html.Div(f"{above_start*100:.1f}%", className="val")], className="summary-item"),
        html.Div([html.Div("Final value range (P5 \u2013 P95)", className="lbl"),
                  html.Div(f"{_fmt_aud(bands['p5'][-1])} \u2013 {_fmt_aud(bands['p95'][-1])}",
                           className="val", style={"fontSize": "13px"})], className="summary-item"),
        html.Div([html.Div("Median final value", className="lbl"),
                  html.Div(_fmt_aud(bands["p50"][-1]), className="val")], className="summary-item"),
        html.Div([html.Div("Mean total drawdown over 10y", className="lbl"),
                  html.Div(_fmt_aud(float(mc.total_drawdowns.mean())), className="val")],
                 className="summary-item"),
    ], className="summary-grid")


@app.callback(
    Output("m5-mc-summary",           "children"),
    Output("m5-mc-fan-chart",         "figure"),
    Output("m5-mc-exhaustion-chart",  "figure"),
    Input("m5-mc-run",                "n_clicks"),
    Input("portfolio-allocation-store","data"),
    Input("cma-store",                "data"),
    State("m5-mc-paths",              "value"),
    State("m5-mc-seed",               "value"),
)
def run_monte_carlo(n_clicks, alloc, cma_store, n_paths, seed):
    if not cma_store or not alloc:
        return html.Div(), go.Figure(), go.Figure()
    total_w = sum(alloc.values())
    weights = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
               if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns = np.asarray(cma_store["returns"], dtype=float)
    vols    = np.asarray(cma_store["vols"],    dtype=float)
    corr    = np.asarray(cma_store["corr"],    dtype=float)
    cov     = tc.cma_to_covariance(vols, corr)
    initial_value = 3_000_000_000
    try:
        mc = dr.monte_carlo(initial_value, weights, returns, cov,
                             n_paths=int(n_paths or 10000), horizon=10,
                             seed=int(seed) if seed is not None else None)
    except Exception as e:
        return (html.Div(f"MC error: {e}", className="opt-infeasible"),
                go.Figure(), go.Figure())
    return _mc_summary_strip(mc, initial_value), _mc_fan_figure(mc, initial_value), \
           _mc_exhaustion_figure(mc)


# ---------------------------------------------------------------------------
# Callbacks — Module 6
# ---------------------------------------------------------------------------

def _module_6_summary(baseline: dr.ProjectionResult, stressed: dr.ProjectionResult,
                       shock_year: int, scenario_name: str) -> html.Div:
    delta_final = stressed.final_value - baseline.final_value
    pct_impact  = (delta_final / baseline.final_value if baseline.final_value > 0 else 0.0)
    base_breach   = sum(1 for y in baseline.years if not (y.meets_12m and y.meets_3y))
    stress_breach = sum(1 for y in stressed.years if not (y.meets_12m and y.meets_3y))
    by = baseline.years[shock_year - 1] if 1 <= shock_year <= len(baseline.years) else None
    sy = stressed.years[shock_year - 1] if 1 <= shock_year <= len(stressed.years) else None
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Drought only", className="lbl",
                         style={"color": COLORS["accent"], "fontWeight": "600"}),
                html.Div(_fmt_aud(baseline.final_value), className="val",
                         style={"fontSize": "20px"}),
                html.Div("Final value (Y10)",
                         style={"fontSize": "11px", "color": COLORS["muted"], "marginBottom": "10px"}),
                html.Div([
                    html.Div(["Total spread cost: ", html.Strong(_fmt_aud(baseline.total_spread_cost))]),
                    html.Div(["Years with liquidity breach: ", html.Strong(str(base_breach))]),
                    html.Div(["Fund exhausted: ", html.Strong("Yes" if baseline.fund_exhausted else "No")]),
                ], style={"fontSize": "13px", "lineHeight": "1.6"}),
            ], className="summary-card", style={"borderLeft": f"3px solid {COLORS['accent']}"}),
            html.Div([
                html.Div(f"Combined ({scenario_name}, Y{shock_year} shock)", className="lbl",
                         style={"color": COLORS["fail"], "fontWeight": "600"}),
                html.Div(_fmt_aud(stressed.final_value), className="val",
                         style={"fontSize": "20px"}),
                html.Div("Final value (Y10)",
                         style={"fontSize": "11px", "color": COLORS["muted"], "marginBottom": "10px"}),
                html.Div([
                    html.Div(["Total spread cost: ", html.Strong(_fmt_aud(stressed.total_spread_cost))]),
                    html.Div(["Years with liquidity breach: ", html.Strong(str(stress_breach))]),
                    html.Div(["Fund exhausted: ", html.Strong("Yes" if stressed.fund_exhausted else "No")]),
                ], style={"fontSize": "13px", "lineHeight": "1.6"}),
            ], className="summary-card", style={"borderLeft": f"3px solid {COLORS['fail']}"}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}),
        html.Div([
            html.Div("Joint impact (combined vs drought-only)", className="lbl",
                     style={"fontSize": "11px", "textTransform": "uppercase",
                            "letterSpacing": "0.05em", "color": COLORS["muted"], "marginBottom": "8px"}),
            html.Div([
                html.Div([html.Div("Final-value delta", className="lbl"),
                          html.Div([
                              html.Span(_fmt_aud(delta_final), style={"fontFamily": MONO_STACK,
                                  "fontWeight": "600",
                                  "color": COLORS["pass"] if delta_final >= 0 else COLORS["fail"]}),
                              html.Span(f" ({pct_impact*100:+.2f}%)",
                                        style={"fontFamily": MONO_STACK, "color": COLORS["muted"],
                                               "marginLeft": "4px"}),
                          ])], className="summary-item"),
                html.Div([html.Div(f"Y{shock_year} ending value (drought only)", className="lbl"),
                          html.Div(_fmt_aud(by.ending_value) if by else "—",
                                   className="val", style={"fontSize": "14px"})],
                         className="summary-item"),
                html.Div([html.Div(f"Y{shock_year} ending value (combined)", className="lbl"),
                          html.Div(_fmt_aud(sy.ending_value) if sy else "—",
                                   className="val", style={"fontSize": "14px"})],
                         className="summary-item"),
            ], className="summary-grid"),
        ], style={"marginTop": "16px"}),
    ])


@app.callback(
    Output("m6-value-chart",   "figure"),
    Output("m6-summary-grid",  "children"),
    Output("m6-config-summary","children"),
    Input("m6-scenario",       "value"),
    Input("m6-shock-year",     "value"),
    Input("m5-severity",       "value"),
    Input("m5-relief",         "value"),
    Input("m5-onset",          "value"),
    Input("m5-fraction",       "value"),
    Input("m5-onset-split-STI", "value"),
    Input("m5-onset-split-MTG", "value"),
    Input("m5-onset-split-LTG", "value"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",         "data"),
)
def update_module_6(scenario_name, shock_year, severity, relief_m, onset,
                    fraction_pct, split_sti, split_mtg, split_ltg, alloc, cma_store):
    if not cma_store or not alloc or relief_m is None or onset is None:
        return go.Figure(), html.Div(), ""
    relief_aud = float(relief_m) * 1e6
    onset      = int(onset)
    shock_year = int(shock_year or onset)
    fraction   = max(0.0, min(1.0, float(fraction_pct or 50) / 100))
    onset_split = _onset_split_from_inputs(split_sti, split_mtg, split_ltg)
    schedule   = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                     year_4_fraction=fraction, residual_split=(0.5, 0.5))
    total_w = sum(alloc.values())
    weights = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
               if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns = np.asarray(cma_store["returns"], dtype=float)
    shocked_assets, _, _ = _scenario_defaults(scenario_name, returns)
    shocked_trust_nets   = st.trust_returns_under_shock(shocked_assets)
    baseline = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                          drawdown_splits={onset: onset_split})
    stressed = dr.project(3_000_000_000, weights, returns, schedule,
                          horizon=10,
                          trust_return_overrides={shock_year: shocked_trust_nets},
                          drawdown_splits={onset: onset_split})
    drought_years = list(schedule.keys())
    fig     = _combined_value_figure(baseline, stressed, shock_year, drought_years)
    summary = _module_6_summary(baseline, stressed, shock_year, scenario_name)
    config  = (f"Market shock: {scenario_name} in Year {shock_year}. "
               f"Drought: {severity} severity, total relief {_fmt_aud(relief_aud)}, "
               f"onset Year {onset}. ")
    if shock_year == onset:
        config += "Shock and drought onset coincide."
    elif shock_year < onset:
        config += f"Shock precedes drought onset by {onset - shock_year} year(s)."
    else:
        config += f"Shock follows drought onset by {shock_year - onset} year(s)."
    return fig, summary, config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
