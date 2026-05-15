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
from dash import Dash, Input, Output, State, ALL, dcc, html, dash_table, callback_context
from dash.exceptions import PreventUpdate

from typing import Optional

from modules import trust_calcs as tc
from modules import metrics as mt
from modules import optimiser as op
from modules import stress as st
from modules import drought as dr

# ---------------------------------------------------------------------------
# User state persistence  (saved to user_state.json next to app.py)
# ---------------------------------------------------------------------------

_STATE_FILE = Path(__file__).resolve().parent / "user_state.json"


def _load_state() -> dict:
    try:
        return json.loads(_STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(updates: dict) -> None:
    try:
        state = _load_state()
        state.update(updates)
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


_SAVED = _load_state()

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


def _fmt_m(x) -> str:
    """Format an AUD value in millions, e.g. 1_500_000_000 → '$1,500.0M'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x / 1_000_000:,.1f}M"

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
      volatility      annual vol copied from hist_vol (read-only, feeds cma-store)
    """
    saved_returns = _SAVED.get("cma_store", {}).get("returns")  # list of decimals
    rows = []
    for i, ac in enumerate(tc.ASSET_CLASSES):
        h_ret = round(float(HIST_GEOM_ANNUAL_RETURNS[ac]) * 100, 3)
        if saved_returns and i < len(saved_returns):
            f_ret = round(float(saved_returns[i]) * 100, 3)
        else:
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
    saved = _SAVED.get("cma_store", {})
    return {
        "returns":      saved.get("returns", HIST_ARITH_ANNUAL_RETURNS.tolist()),
        "vols":         HIST_ANNUAL_VOL.tolist(),
        "corr":         HIST_CORR.values.tolist(),
        "cpi":          saved.get("cpi", 0.025),
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
             "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
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
             "textAlign": "right", "minWidth": "140px",
             "backgroundColor": _HIST_GREY},
        ],
        style_header_conditional=[
            {"if": {"column_id": "hist_return"}, "backgroundColor": _HIST_GREY_H},
            {"if": {"column_id": "hist_vol"},    "backgroundColor": _HIST_GREY_H},
            {"if": {"column_id": "volatility"},   "backgroundColor": _HIST_GREY_H},
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
            "volatility":      "Locked to Hist. Vol % p.a. for the selected analysis period.",
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
                                    _SAVED.get("period", {}).get("sm", _DATE_MIN_M),
                                    _SAVED.get("period", {}).get("sy", _DATE_MIN_Y),
                                    _SAVED.get("period", {}).get("em", _DATE_MAX_M),
                                    _SAVED.get("period", {}).get("ey", _DATE_MAX_Y)),
                ], className="ctrl-group"),
                html.Div(style={"width": "1px", "background": COLORS["border"],
                                "alignSelf": "stretch", "margin": "0 4px"}),
                html.Div([
                    html.Div("CPI Assumption", className="ctrl-label"),
                    html.Div([
                        dcc.Input(id="cpi-input", type="number",
                                  value=round(_SAVED.get("cma_store", {}).get("cpi", 0.025) * 100, 2),
                                  step=0.1, min=0, max=20, className="cpi-input",
                                  style={"width": "80px"}),
                        html.Span(" % p.a.", style={"marginLeft": "6px",
                                                     "color": COLORS["muted"],
                                                     "fontSize": "13px"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "marginTop": "4px"}),
                    html.Div([
                        html.Span("CPI + 2.5% p.a. fund target",
                                  style={"fontSize": "11px", "color": COLORS["muted"]}),
                        html.Span(id="m1-hist-cpi-ref",
                                  style={"fontSize": "11px", "color": COLORS["muted"],
                                         "marginLeft": "10px"}),
                    ], style={"marginTop": "4px"}),
                ], className="ctrl-group"),
            ], className="chart-controls", style={"marginBottom": "16px"}),
            html.Div([
                html.Div([
                    _cma_rv_table(),
                    html.Div(
                        "Grey = historical reference (read-only). "
                        "White = editable Forecast Return input. Forecast Vol is greyed out "
                        "and locked to Hist. Vol for the selected period. "
                        "Historical Return uses geometric compounding; "
                        "Forecast Return uses arithmetic convention for mean-variance calculations.",
                        className="hist-note",
                    ),
                ]),
            ]),
            html.Div(id="m1-cma-flags", style={"marginTop": "12px"}),
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


def _asset_geom_returns_for_period(start_m: int, start_y: int,
                                   end_m: int, end_y: int) -> np.ndarray:
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df.loc[mask]
    if df_slice.empty:
        return np.zeros(len(tc.ASSET_CLASSES))
    n = len(df_slice)
    geom = (1 + df_slice).prod() ** (12 / n) - 1
    return geom.reindex(tc.ASSET_CLASSES).to_numpy(dtype=float)


def _to_event_window_return(annual_returns: np.ndarray, n_months: Optional[int]) -> np.ndarray:
    n = n_months or 12
    return (1 + annual_returns) ** (n / 12) - 1


def _asset_returns_for_basis(start_m: int, start_y: int,
                             end_m: int, end_y: int,
                             return_basis: str,
                             n_months: Optional[int]) -> np.ndarray:
    annual_returns = _asset_geom_returns_for_period(start_m, start_y, end_m, end_y)
    if return_basis == "event_window":
        return _to_event_window_return(annual_returns, n_months)
    return annual_returns


def _forecast_returns_for_basis(forecast_returns: np.ndarray,
                                return_basis: str,
                                n_months: Optional[int]) -> np.ndarray:
    if return_basis == "event_window":
        return _to_event_window_return(forecast_returns, n_months)
    return forecast_returns


def _scenario_adjusted_returns(scenario_returns: np.ndarray,
                               forecast_returns: np.ndarray,
                               selected_hist_returns: np.ndarray,
                               scenario_name: str,
                               return_basis: str = "annualised",
                               n_months: Optional[int] = None) -> np.ndarray:
    """
    Historical stress scenarios are applied as a fixed scenario delta:
        forecast return + (scenario stress return - selected-period hist return).

    For short event-window shocks, forecast and selected-period historical
    returns are first converted to the same event length so the result is a
    cumulative percentage change, not an annualised rate.

    Analytical shocks are already built directly off the current forecast returns,
    so they are left as-is.
    """
    if scenario_name in SCENARIO_WINDOWS_LIVE or scenario_name == "AUD Depreciation Shock":
        comparable_forecast = _forecast_returns_for_basis(forecast_returns, return_basis, n_months)
        return comparable_forecast + (scenario_returns - selected_hist_returns)
    return scenario_returns


def _trust_geom_returns_for_period(start_m: int, start_y: int,
                                   end_m: int, end_y: int) -> dict[str, float]:
    mask = _filter_dates(start_m, start_y, end_m, end_y)
    df_slice = _returns_df.loc[mask]
    if df_slice.empty:
        return {t: 0.0 for t in tc.TRUST_NAMES}
    trust_monthly = tc.historical_trust_returns_monthly_net(df_slice)
    n = len(trust_monthly)
    return {
        t: float((1 + trust_monthly[t]).prod() ** (12 / n) - 1)
        for t in tc.TRUST_NAMES
    }


def _trust_returns_for_basis(start_m: int, start_y: int,
                             end_m: int, end_y: int,
                             return_basis: str,
                             n_months: Optional[int]) -> dict[str, float]:
    annual_returns = _trust_geom_returns_for_period(start_m, start_y, end_m, end_y)
    if return_basis == "event_window":
        return {
            t: float((1 + r) ** ((n_months or 12) / 12) - 1)
            for t, r in annual_returns.items()
        }
    return annual_returns


def _trust_nets_for_basis(asset_returns: np.ndarray,
                          return_basis: str,
                          n_months: Optional[int]) -> dict[str, float]:
    if return_basis == "event_window":
        return st.trust_returns_under_event_shock(asset_returns, n_months or 12)
    return st.trust_returns_under_shock(asset_returns)


def _scenario_trust_net_path(
    scenario_name: str,
    cma_returns: np.ndarray,
) -> dict[int, dict[str, float]]:
    """
    Multi-year trust net-return path for a named crisis scenario.

    Returns {year_offset: {trust_name: annual_net_return}} where year_offset
    starts at 1. All returns are annualised (see stress.build_crisis_path).
    Used by Modules 5 and 6 to build multi-year trust_return_overrides.
    """
    asset_path = st.build_crisis_path(scenario_name, _returns_df, cma_returns)
    return {
        yr: st.trust_returns_under_shock(arr)
        for yr, arr in asset_path.items()
    }


def shock_compare_figure(baseline_returns: np.ndarray,
                         shocked_returns: np.ndarray,
                         portfolio_weights: dict,
                         return_basis: str = "annualised",
                         n_months: Optional[int] = None) -> go.Figure:
    base_nets  = _trust_nets_for_basis(baseline_returns, return_basis, n_months)
    shock_nets = _trust_nets_for_basis(shocked_returns, return_basis, n_months)
    base_port  = sum(portfolio_weights.get(t, 0) * base_nets[t] for t in tc.TRUST_NAMES)
    shock_port = sum(portfolio_weights.get(t, 0) * shock_nets[t] for t in tc.TRUST_NAMES)
    y_title = "Event-window return" if return_basis == "event_window" else "Annual return"

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
            title=dict(text=y_title, font=dict(size=11, color=COLORS["muted"]))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _factor_breakdown_rows(shocked_returns: np.ndarray, df_for_drawdown,
                            scenario_name: str, window_label: Optional[str],
                            selected_hist: dict[str, float],
                            return_basis: str = "annualised",
                            n_months: Optional[int] = None) -> tuple[list[dict], str]:
    rows = []
    nets = _trust_nets_for_basis(shocked_returns, return_basis, n_months)
    duration = "—"
    stress_window = window_label or "Analytical shock"
    if window_label is not None:
        try:
            start, end = window_label.split(" \u2013 ")
            duration = f"{len(st._window_returns(df_for_drawdown, start, end))} months"
        except Exception:
            duration = "—"
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
        hist_return = selected_hist.get(t, 0.0)
        rows.append({
            "trust": t,
            "stress_window": stress_window,
            "duration": duration,
            "stress_return": _fmt_pct(nets[t]),
            "selected_hist_return": _fmt_pct(hist_return),
            "delta_return": _fmt_signed_pct(nets[t] - hist_return),
            "dominant_factor": dom,
            "window_drawdown": dd_str,
        })
    return rows, duration



def _build_m4_crisis_path_figure(
    asset_path: dict[int, np.ndarray],
    cma_baseline: np.ndarray,
    portfolio_weights: dict[str, float],
    recovery_years: int = 3,
) -> go.Figure:
    """
    Indexed value chart showing portfolio/trust evolution through the full
    crisis (one entry per crisis year) then CMA recovery. Pre-crisis = 1.0.

    crisis year N applies asset_path[N] as the annual return for that year.
    Recovery years apply CMA trust net returns.
    """
    n_crisis = len(asset_path)
    cma_trust_nets = {t: tc.trust_net_return(t, cma_baseline) for t in tc.TRUST_NAMES}

    crisis_nets_per_year: list[dict[str, float]] = []
    for yr in sorted(asset_path.keys()):
        crisis_nets_per_year.append(st.trust_returns_under_shock(asset_path[yr]))

    total_points = n_crisis + recovery_years
    x = list(range(0, total_points + 1))

    trust_vals: dict[str, list[float]] = {t: [1.0] for t in tc.TRUST_NAMES}
    for t in tc.TRUST_NAMES:
        for nets in crisis_nets_per_year:
            trust_vals[t].append(trust_vals[t][-1] * (1.0 + nets[t]))
        for _ in range(recovery_years):
            trust_vals[t].append(trust_vals[t][-1] * (1.0 + cma_trust_nets[t]))

    total_w = sum(portfolio_weights.values()) or 1.0
    w = {t: portfolio_weights[t] / total_w for t in tc.TRUST_NAMES}
    port_vals = [
        sum(w[t] * trust_vals[t][i] for t in tc.TRUST_NAMES)
        for i in range(total_points + 1)
    ]

    fig = go.Figure()
    for t in tc.TRUST_NAMES:
        fig.add_trace(go.Scatter(
            x=x, y=trust_vals[t], mode="lines+markers", name=t,
            line=dict(color=COLORS[t], width=2, dash="dot"),
            marker=dict(size=6, color=COLORS[t]),
            hovertemplate=f"<b>{t}</b><br>Year %{{x}}<br>Index: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=x, y=port_vals, mode="lines+markers", name="Portfolio",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=8, color=COLORS["accent"], symbol="diamond"),
        hovertemplate="<b>Portfolio</b><br>Year %{x}<br>Index: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line=dict(color=COLORS["ink"], width=1, dash="dash"),
                  annotation_text="Pre-crisis level", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=COLORS["muted"]))
    if n_crisis > 0:
        fig.add_vrect(x0=0.5, x1=n_crisis + 0.5,
                      fillcolor=COLORS["fail"], opacity=0.06, line_width=0,
                      annotation_text="Crisis period",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=COLORS["fail"]))
    x_labels = (
        ["Pre-crisis"]
        + [f"Crisis Y{i}" for i in range(1, n_crisis + 1)]
        + [f"Recovery Y{i}" for i in range(1, recovery_years + 1)]
    )
    fig.update_layout(
        height=400, margin=dict(l=60, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(tickmode="array", tickvals=x, ticktext=x_labels,
                   showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(tickformat=".3f", gridcolor=COLORS["border"], tickfont=dict(size=11),
                   title=dict(text="Indexed value (1.0 = pre-crisis)",
                              font=dict(size=11, color=COLORS["muted"]))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


def _m4_liquidity_check_div(
    shocked_arr: np.ndarray,
    portfolio_weights: dict,
    cma_baseline: np.ndarray,
) -> html.Div:
    """
    Shows pre- vs post-shock trust weight drift and whether Board Policy
    liquidity floors (STI ≥ 10%, STI+MTG ≥ 25%) are still met after the shock.
    """
    shocked_nets = st.trust_returns_under_shock(shocked_arr)
    total_pre = sum(portfolio_weights.get(t, 0) for t in tc.TRUST_NAMES) or 1.0
    pre_w = {t: portfolio_weights.get(t, 0) / total_pre for t in tc.TRUST_NAMES}

    post_vals = {t: pre_w[t] * (1 + shocked_nets[t]) for t in tc.TRUST_NAMES}
    post_total = sum(post_vals.values()) or 1.0
    post_w = {t: post_vals[t] / post_total for t in tc.TRUST_NAMES}

    liq_12m_pre  = pre_w["STI"]
    liq_3y_pre   = pre_w["STI"] + pre_w["MTG"]
    liq_12m_post = post_w["STI"]
    liq_3y_post  = post_w["STI"] + post_w["MTG"]

    def _pill(ok: bool, val: float, threshold: float) -> html.Span:
        color = COLORS["pass"] if ok else COLORS["fail"]
        return html.Span([
            html.Span(f"{val*100:.1f}% ", style={"fontFamily": MONO_STACK}),
            html.Span(f"(min {threshold*100:.0f}%) ",
                      style={"color": COLORS["muted"], "fontSize": "11px"}),
            html.Span("PASS" if ok else "FAIL",
                      style={"backgroundColor": color, "color": "#fff",
                             "borderRadius": "3px", "padding": "1px 6px",
                             "fontSize": "11px", "fontWeight": "600",
                             "verticalAlign": "middle"}),
        ])

    body_rows = []
    for t in tc.TRUST_NAMES:
        delta = post_w[t] - pre_w[t]
        body_rows.append(html.Tr([
            html.Td(t, style={"fontWeight": "600", "paddingRight": "20px",
                              "borderBottom": f"1px solid {COLORS['border']}",
                              "paddingTop": "6px", "paddingBottom": "6px"}),
            html.Td(f"{pre_w[t]*100:.1f}%",
                    style={"fontFamily": MONO_STACK, "textAlign": "right",
                           "paddingRight": "20px",
                           "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(f"{post_w[t]*100:.1f}%",
                    style={"fontFamily": MONO_STACK, "textAlign": "right",
                           "paddingRight": "20px",
                           "borderBottom": f"1px solid {COLORS['border']}"}),
            html.Td(f"{delta*100:+.1f}pp",
                    style={"fontFamily": MONO_STACK, "textAlign": "right",
                           "color": COLORS["pass"] if delta >= 0 else COLORS["fail"],
                           "borderBottom": f"1px solid {COLORS['border']}"}),
        ]))

    _th_style = {"textAlign": "left", "paddingRight": "20px", "fontSize": "12px",
                 "color": COLORS["muted"], "textTransform": "uppercase",
                 "letterSpacing": "0.04em", "borderBottom": f"2px solid {COLORS['border']}",
                 "paddingBottom": "6px"}
    weight_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Trust",            style=_th_style),
            html.Th("Pre-shock weight", style={**_th_style, "textAlign": "right"}),
            html.Th("Post-shock weight",style={**_th_style, "textAlign": "right"}),
            html.Th("Drift",            style={**_th_style, "textAlign": "right"}),
        ])),
        html.Tbody(body_rows),
    ], style={"borderCollapse": "collapse", "width": "100%", "fontSize": "13.5px"})

    _card = {"padding": "12px 16px", "background": COLORS["bg"],
             "borderRadius": "6px", "flex": "1 1 300px"}
    _lbl = {"fontSize": "12px", "color": COLORS["muted"], "textTransform": "uppercase",
            "letterSpacing": "0.04em", "marginBottom": "6px"}
    check_row = html.Div([
        html.Div([
            html.Div("12-month liquidity — STI ≥ 10%", style=_lbl),
            html.Div([
                html.Span("Pre: ", style={"color": COLORS["muted"], "fontSize": "12px"}),
                _pill(liq_12m_pre >= 0.10, liq_12m_pre, 0.10),
                html.Span("  →  Post-shock: ",
                          style={"color": COLORS["muted"], "fontSize": "12px",
                                 "margin": "0 6px"}),
                _pill(liq_12m_post >= 0.10, liq_12m_post, 0.10),
            ]),
        ], style=_card),
        html.Div([
            html.Div("3-year liquidity — STI + MTG ≥ 25%", style=_lbl),
            html.Div([
                html.Span("Pre: ", style={"color": COLORS["muted"], "fontSize": "12px"}),
                _pill(liq_3y_pre >= 0.25, liq_3y_pre, 0.25),
                html.Span("  →  Post-shock: ",
                          style={"color": COLORS["muted"], "fontSize": "12px",
                                 "margin": "0 6px"}),
                _pill(liq_3y_post >= 0.25, liq_3y_post, 0.25),
            ]),
        ], style=_card),
    ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"})

    note = html.Div(
        "Post-shock weights reflect trust-level value drift: a trust that falls harder "
        "shrinks in relative weight, potentially breaching the Board Policy liquidity floor. "
        "No rebalancing or redemption is assumed — this is a mark-to-market effect only.",
        style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "12px",
               "fontStyle": "italic"},
    )
    return html.Div([weight_table, check_row, note])


def module_4_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Module 4 — Market Stress Testing"),
            html.Div(
                "Apply named historical or analytical shocks to the asset class returns "
                "and see how each trust and the overall portfolio responds. Shocked returns "
                "for the historical scenarios are derived from the Refinitiv CSV windows; "
                "short crash windows are treated as cumulative event shocks rather than "
                "annualised rates. The rate shock is analytical (\u00b1 duration). Buy/sell spreads are NOT "
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
            html.H2("Trust stress-period returns"),
            html.Div("Trust returns over the selected stress window, compared with "
                     "the selected-period historical return from Module 1 on the "
                     "same basis. Short shocks such as COVID Crash are shown as "
                     "cumulative event-window changes rather than annualised rates.",
                     className="section-note"),
            html.Div(id="m4-verdict", className="decision-band",
                     style={"marginBottom": "14px"}),
            html.Div(id="m4-factor-table"),
        ], className="panel"),

        html.Div([
            html.H2("Crisis multi-year return path"),
            html.Div(
                "Each historical scenario is split into consecutive 12-month annual chunks "
                "and annualised. The chart shows indexed trust and portfolio value through "
                "every crisis year, then CMA-rate recovery. This multi-year path is what "
                "Modules 5 and 6 apply as trust return overrides when you select the "
                "scenario there — so the stress test reflects the full historical horizon "
                "of each crisis, not just a single shocked year.",
                className="section-note"),
            html.Div(id="m4-path-description",
                     style={"fontSize": "12px", "color": COLORS["muted"],
                            "marginBottom": "10px", "fontStyle": "italic"}),
            dcc.Graph(id="m4-crisis-path-chart", config={"displayModeBar": False}),
        ], className="panel"),

        html.Div([
            html.H2("Scenario asset class returns"),
            html.Div("Asset class returns for the selected stress scenario versus the CMA baseline. "
                     "Delta = Stress Return − Baseline.",
                     className="section-note"),
            dash_table.DataTable(
                id="m4-shock-table",
                columns=[
                    {"name": "Asset Class", "id": "asset_class", "editable": False},
                    {"name": "Baseline (%)", "id": "baseline",
                     "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
                    {"name": "Stress Return (%)", "id": "shocked",
                     "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
                    {"name": "Delta (%)", "id": "delta",
                     "type": "numeric", "format": {"specifier": ".1f"}, "editable": False},
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px 10px", "fontFamily": MONO_STACK,
                            "fontSize": "12.5px", "textAlign": "right",
                            "backgroundColor": COLORS["bg"]},
                style_cell_conditional=[
                    {"if": {"column_id": "asset_class"},
                     "fontFamily": FONT_STACK, "textAlign": "left", "minWidth": "260px"}],
                style_data_conditional=_DELTA_STYLES,
                style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
                    "fontWeight": "600", "fontSize": "12px",
                    "borderBottom": f"2px solid {COLORS['border']}"},
                style_data={"borderBottom": f"1px solid {COLORS['border']}",
                            "backgroundColor": COLORS["bg"]},
                editable=False,
            ),
        ], className="panel"),

        html.Div([
            html.H2("Liquidity check under stress"),
            html.Div(
                "How trust weights drift after a one-year shock (mark-to-market, "
                "no rebalancing assumed) and whether the portfolio still meets the "
                "Board Policy minimum liquidity floors: STI ≥ 10% within 12 months "
                "and STI + MTG ≥ 25% within 3 years.",
                className="section-note"),
            html.Div(id="m4-liquidity-check"),
        ], className="panel"),

        dcc.Store(id="m4-shocked-store"),
        dcc.Store(id="m4-path-store"),
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
    M = 1_000_000
    years  = [0] + [y.year for y in result.years]
    values = [result.initial_value / M] + [y.ending_value / M for y in result.years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=values, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5), marker=dict(size=7, color=COLORS["accent"]),
        name="Portfolio value",
        hovertemplate="Year %{x}<br>Value: $%{y:,.1f}M<extra></extra>"))
    drought_years = [y.year for y in result.years if y.drawdown > 0]
    for dy in drought_years:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.4)
    if drought_years:
        fig.add_annotation(x=drought_years[0], y=max(values),
            text=f"Drought onset (Y{drought_years[0]})", showarrow=False, yshift=10,
            font=dict(size=11, color=COLORS["fail"]))
    fig.add_hline(y=result.initial_value / M, line=dict(color=COLORS["muted"], width=1, dash="dash"),
        annotation_text="Starting value", annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=360, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)))
    return fig


def _trust_composition_figure(result: dr.ProjectionResult) -> go.Figure:
    M = 1_000_000
    years = [0] + [y.year for y in result.years]
    if result.years:
        first = result.years[0]
        sti = [first.starting_weights["STI"] * first.starting_value / M]
        mtg = [first.starting_weights["MTG"] * first.starting_value / M]
        ltg = [first.starting_weights["LTG"] * first.starting_value / M]
    else:
        sti, mtg, ltg = [0], [0], [0]
    for y in result.years:
        sti.append(y.ending_holdings["STI"] / M)
        mtg.append(y.ending_holdings["MTG"] / M)
        ltg.append(y.ending_holdings["LTG"] / M)
    fig = go.Figure()
    for trust, vals in [("STI", sti), ("MTG", mtg), ("LTG", ltg)]:
        fig.add_trace(go.Scatter(x=years, y=vals, mode="lines", stackgroup="one",
            name=trust, line=dict(width=0.5, color=COLORS[trust]),
            fillcolor=COLORS[trust],
            hovertemplate=f"<b>{trust}</b><br>Year %{{x}}<br>$%{{y:,.1f}}M<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=70, r=20, t=20, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Trust holdings ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _projection_summary_table(result: dr.ProjectionResult,
                               table_id: str = "m5-projection-table") -> dash_table.DataTable:
    rows = []
    for y in result.years:
        rows.append({
            "year":         y.year,
            "start":        _fmt_m(y.starting_value),
            "growth":       _fmt_pct((y.pre_drawdown_value / y.starting_value - 1)
                                     if y.starting_value > 0 else 0),
            "drawdown":     _fmt_m(y.drawdown) if y.drawdown > 0 else "—",
            "spread":       _fmt_m(sum(y.spread_costs.values()))
                            if any(v > 0 for v in y.spread_costs.values()) else "—",
            "rebal_cost":   _fmt_m(y.rebalance_cost) if y.rebalance_cost > 0 else "—",
            "end":          _fmt_m(y.ending_value),
            "sti_pct":      _fmt_pct(y.ending_weights["STI"]),
            "mtg_pct":      _fmt_pct(y.ending_weights["MTG"]),
            "ltg_pct":      _fmt_pct(y.ending_weights["LTG"]),
            "liq_12m":      _fmt_pct(y.liquidity_within_12m),
            "liq_3y":       _fmt_pct(y.liquidity_within_3y),
        })
    return dash_table.DataTable(
        id=table_id,
        columns=[
            {"name": "Year",          "id": "year"},
            {"name": "Starting",      "id": "start"},
            {"name": "Growth",        "id": "growth"},
            {"name": "Drawdown",      "id": "drawdown"},
            {"name": "Spread cost",   "id": "spread"},
            {"name": "Rebal. cost",   "id": "rebal_cost"},
            {"name": "Ending",        "id": "end"},
            {"name": "STI",           "id": "sti_pct"},
            {"name": "MTG",           "id": "mtg_pct"},
            {"name": "LTG",           "id": "ltg_pct"},
            {"name": "Liq 12m",       "id": "liq_12m"},
            {"name": "Liq 3y",        "id": "liq_3y"},
        ],
        data=rows,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "7px 10px", "fontFamily": MONO_STACK,
                    "fontSize": "12px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "year"},
             "fontFamily": FONT_STACK, "textAlign": "center", "fontWeight": "600"}],
        style_data_conditional=[
            {"if": {"column_id": "drawdown",   "filter_query": '{drawdown} != "—"'},
             "color": COLORS["fail"]},
            {"if": {"column_id": "rebal_cost", "filter_query": '{rebal_cost} != "—"'},
             "color": COLORS["accent"], "fontWeight": "600"},
        ],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
            "fontWeight": "600", "fontSize": "12px",
            "borderBottom": f"2px solid {COLORS['border']}"},
    )


def _master_fund_return_table(
    result: dr.ProjectionResult,
    asset_returns: np.ndarray,
    trust_return_overrides: dict,
    cpi: float,
    drought_schedule: dict = None,
    rebalance_year: int = None,
    new_alloc: dict = None,
    stress_scenario: str = None,
    stress_year: int = None,
) -> dash_table.DataTable:
    """
    Year-by-year master fund gross/net return (excl. drawdown), per-trust
    contribution to net return, CPI+2.5% target flag, and event notes.
    Weights used = ending weights after drift for each year.
    """
    rows = []
    target_spread = cpi + 0.025
    geom_gross_factor = 1.0
    geom_net_factor   = 1.0
    contrib_sum = {t: 0.0 for t in tc.TRUST_NAMES}
    n_years = len(result.years)

    for y in result.years:
        yr = y.year
        total_w = sum(y.ending_weights.values()) or 1.0
        w = {t: y.ending_weights[t] / total_w for t in tc.TRUST_NAMES}

        if yr in trust_return_overrides:
            net_r = trust_return_overrides[yr]
            gross_r = {
                t: net_r[t] + tc.trust_weighted_asset_cost(t) + tc.TRUST_ONGOING_COSTS[t]
                for t in tc.TRUST_NAMES
            }
        else:
            net_r   = {t: tc.trust_net_return(t, asset_returns)   for t in tc.TRUST_NAMES}
            gross_r = {t: tc.trust_gross_return(t, asset_returns) for t in tc.TRUST_NAMES}

        port_gross = sum(w[t] * gross_r[t] for t in tc.TRUST_NAMES)
        port_net   = sum(w[t] * net_r[t]   for t in tc.TRUST_NAMES)

        geom_gross_factor *= (1 + port_gross)
        geom_net_factor   *= (1 + port_net)
        for t in tc.TRUST_NAMES:
            contrib_sum[t] += w[t] * net_r[t]

        # Build event note for this year
        notes = []
        if drought_schedule and drought_schedule.get(yr, 0) > 0:
            notes.append(f"Drought drawdown {_fmt_m(drought_schedule[yr])}")
        if rebalance_year is not None and yr == rebalance_year and new_alloc:
            alloc_str = (f"STI {new_alloc['STI']*100:.0f}% / "
                         f"MTG {new_alloc['MTG']*100:.0f}% / "
                         f"LTG {new_alloc['LTG']*100:.0f}%")
            notes.append(f"Rebalance → {alloc_str}")
        if yr in trust_return_overrides and stress_scenario:
            yr_offset = yr - (stress_year or yr) + 1
            notes.append(f"{stress_scenario} (Y{yr_offset})")

        rows.append({
            "year":        yr,
            "gross":       f"{port_gross * 100:.1f}%",
            "net":         f"{port_net * 100:.1f}%",
            "sti_contrib": f"{w['STI'] * net_r['STI'] * 100:.2f}%",
            "mtg_contrib": f"{w['MTG'] * net_r['MTG'] * 100:.2f}%",
            "ltg_contrib": f"{w['LTG'] * net_r['LTG'] * 100:.2f}%",
            "meets":       "",
            "notes":       " | ".join(notes) if notes else "—",
        })

    # 10-year geometric average row
    geom_gross = geom_gross_factor ** (1 / n_years) - 1 if n_years else 0.0
    geom_net   = geom_net_factor   ** (1 / n_years) - 1 if n_years else 0.0
    meets_target = geom_net >= target_spread - 1e-9
    rows.append({
        "year":        "10Y Avg",
        "gross":       f"{geom_gross * 100:.1f}%",
        "net":         f"{geom_net * 100:.1f}%",
        "sti_contrib": f"{contrib_sum['STI'] / n_years * 100:.2f}%" if n_years else "—",
        "mtg_contrib": f"{contrib_sum['MTG'] / n_years * 100:.2f}%" if n_years else "—",
        "ltg_contrib": f"{contrib_sum['LTG'] / n_years * 100:.2f}%" if n_years else "—",
        "meets":       "Pass" if meets_target else "Fail",
        "notes":       "",
    })

    return dash_table.DataTable(
        columns=[
            {"name": "Year",             "id": "year"},
            {"name": "Gross Return",     "id": "gross"},
            {"name": "Net Return",       "id": "net"},
            {"name": "STI Contrib",      "id": "sti_contrib"},
            {"name": "MTG Contrib",      "id": "mtg_contrib"},
            {"name": "LTG Contrib",      "id": "ltg_contrib"},
            {"name": f"Target Met (CPI+2.5% = {(cpi+0.025)*100:.1f}%)", "id": "meets"},
            {"name": "Events",           "id": "notes"},
        ],
        data=rows,
        style_table={"overflowX": "auto"},
        style_cell={"padding": "7px 10px", "fontFamily": MONO_STACK,
                    "fontSize": "12px", "textAlign": "right"},
        style_cell_conditional=[
            {"if": {"column_id": "year"},
             "fontFamily": FONT_STACK, "textAlign": "center", "fontWeight": "600"},
            {"if": {"column_id": "meets"},
             "fontFamily": FONT_STACK, "textAlign": "center", "fontWeight": "600"},
            {"if": {"column_id": "notes"},
             "fontFamily": FONT_STACK, "textAlign": "left", "fontSize": "11.5px",
             "color": COLORS["muted"], "whiteSpace": "normal", "minWidth": "200px"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": '{meets} = "Pass"', "column_id": "meets"},
             "color": COLORS["pass"]},
            {"if": {"filter_query": '{meets} = "Fail"', "column_id": "meets"},
             "color": COLORS["fail"]},
            # Highlight the summary row
            {"if": {"filter_query": '{year} = "10Y Avg"'},
             "backgroundColor": COLORS["bg"], "fontWeight": "600"},
        ],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
                      "fontWeight": "600", "fontSize": "12px",
                      "borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
    )


def _branching_value_figure(
    bau_result: dr.ProjectionResult,
    stress_result: Optional[dr.ProjectionResult],
    onset_year: int,
    rebalance_year: int,
    stress_year: Optional[int],
    stress_label: Optional[str],
) -> go.Figure:
    """
    Three-line chart after the rebalance decision:
      - Branch (a) BAU: teal, full horizon.
      - Branch (b) Stress: orange, full horizon — diverges from (a) at stress_year.
    Vertical markers for drought years, rebalance year, and stress year.
    """
    M = 1_000_000  # scale factor
    years    = [0] + [y.year for y in bau_result.years]
    bau_vals = [(bau_result.initial_value / M)] + [y.ending_value / M for y in bau_result.years]

    fig = go.Figure()

    # ── Branch (a): BAU post-rebalance ────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=years, y=bau_vals, mode="lines+markers",
        name="Branch (a) — BAU",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=6, color=COLORS["accent"]),
        hovertemplate="Year %{x}<br>BAU: $%{y:,.1f}M<extra></extra>",
    ))

    # ── Branch (b): Stress ────────────────────────────────────────────────────
    if stress_result is not None:
        stress_vals = [(stress_result.initial_value / M)] + [y.ending_value / M for y in stress_result.years]
        fig.add_trace(go.Scatter(
            x=years, y=stress_vals, mode="lines+markers",
            name=f"Branch (b) — {stress_label or 'Stress'} Y{stress_year}",
            line=dict(color="#C07A2A", width=2.5, dash="dash"),
            marker=dict(size=6, color="#C07A2A"),
            hovertemplate="Year %{x}<br>Stress: $%{y:,.1f}M<extra></extra>",
        ))

    # ── Vertical markers ──────────────────────────────────────────────────────
    for dy in [y.year for y in bau_result.years if y.drawdown > 0]:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.4)
    if any(y.drawdown > 0 for y in bau_result.years):
        first_drought = next(y.year for y in bau_result.years if y.drawdown > 0)
        fig.add_annotation(x=first_drought, y=max(bau_vals) * 1.01,
            text=f"Drought Y{first_drought}–", showarrow=False,
            font=dict(size=10, color=COLORS["fail"]))

    fig.add_vline(x=rebalance_year,
        line=dict(color=COLORS["accent"], width=1.5, dash="dashdot"), opacity=0.7)
    fig.add_annotation(x=rebalance_year, y=max(bau_vals) * 1.03,
        text=f"Rebalance Y{rebalance_year}", showarrow=False,
        font=dict(size=10, color=COLORS["accent"]))

    if stress_year is not None:
        fig.add_vline(x=stress_year,
            line=dict(color="#C07A2A", width=1.5, dash="dot"), opacity=0.7)
        fig.add_annotation(x=stress_year, y=max(bau_vals) * 1.05,
            text=f"Stress Y{stress_year}", showarrow=False,
            font=dict(size=10, color="#C07A2A"))

    fig.add_hline(y=bau_result.initial_value / M,
        line=dict(color=COLORS["muted"], width=1, dash="dash"),
        annotation_text="Starting value", annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["muted"]))

    fig.update_layout(
        height=400, margin=dict(l=70, r=20, t=50, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    return fig


def _rebalancing_controls(onset: int) -> html.Div:
    """
    Post-drought rebalancing panel — new strategic allocation + stress-test controls.
    onset is used to compute the default rebalance year (onset + 3).
    Rebalancing occurs end-of-year (after growth, before that year's drawdown if any),
    so it is valid to rebalance in the final drought year (onset + 2) or later.
    """
    _s5 = _SAVED.get("m5", {})
    default_reb_year = _s5.get("rebalance_year", min(onset + 3, 9))
    return html.Div([
        html.H2("Post-drought rebalancing strategy"),
        html.Div(
            "After the drought drawdowns clear, set a new strategic allocation. "
            "You are not required to return to the Module 3 target — "
            "if the fund no longer needs to hold as much short-term liquidity, "
            "you may overweight LTG for higher long-run growth.",
            className="section-note",
        ),

        # ── Rebalance timing & new weights ───────────────────────────────────
        html.Div([
            html.Div([
                html.Label("Rebalance year"),
                dcc.Input(id="m5-rebalance-year", type="number",
                          min=onset, max=10, step=1, value=default_reb_year,
                          className="alloc-num-input"),
                html.Div("Occurs at year-end: after growth, before that year's drawdown.",
                         style={"fontSize": "11px", "color": COLORS["muted"],
                                "marginTop": "3px", "fontStyle": "italic"}),
            ], className="drought-control"),
            html.Div([html.Label("New STI (%)"),
                      dcc.Input(id="m5-reb-STI", type="number",
                                min=0, max=100, step=1, value=_s5.get("reb_STI", 20),
                                className="alloc-num-input")],
                     className="drought-control"),
            html.Div([html.Label("New MTG (%)"),
                      dcc.Input(id="m5-reb-MTG", type="number",
                                min=0, max=100, step=1, value=_s5.get("reb_MTG", 30),
                                className="alloc-num-input")],
                     className="drought-control"),
            html.Div([html.Label("New LTG (%)"),
                      dcc.Input(id="m5-reb-LTG", type="number",
                                min=0, max=100, step=1, value=_s5.get("reb_LTG", 50),
                                className="alloc-num-input")],
                     className="drought-control"),
        ], className="drought-controls",
           style={"gridTemplateColumns": "repeat(4, minmax(140px, 1fr))"}),

        # Live constraint checker + drift weights populated by callback
        html.Div(id="m5-rebalance-constraint",
                 style={"marginTop": "10px", "fontSize": "12.5px"}),
        html.Div(id="m5-drift-weights",
                 style={"marginTop": "8px", "fontSize": "12px",
                        "color": COLORS["muted"]}),

        html.H3("Board policy compliance — rebalanced allocation",
                style={"marginTop": "20px", "marginBottom": "4px",
                       "fontSize": "13px", "fontWeight": "600",
                       "textTransform": "uppercase", "letterSpacing": "0.05em",
                       "color": COLORS["muted"]}),
        html.Div(id="m5-reb-compliance"),

        html.Hr(style={"margin": "18px 0", "borderColor": COLORS["border"]}),

        # ── Stress-test the new allocation ───────────────────────────────────
        html.Div([
            html.Strong("Stress-test the rebalanced portfolio (Branch b)"),
            html.Div(
                "Apply a market shock at Year 8 or 9 to the rebalanced portfolio. "
                "This tests whether an LTG-heavy allocation can absorb a late-horizon "
                "downturn. Branch (a) continues under BAU; Branch (b) applies the shock.",
                style={"fontSize": "11.5px", "color": COLORS["muted"],
                       "lineHeight": "1.35", "marginTop": "4px", "marginBottom": "10px"},
            ),
        ]),
        html.Div([
            html.Div([
                html.Label("Stress scenario"),
                dcc.Dropdown(
                    id="m5-stress-scenario",
                    options=[{"label": s, "value": s} for s in SCENARIO_ORDER],
                    value=_s5.get("stress_scenario", "GFC"), clearable=False,
                    style={"fontFamily": FONT_STACK, "fontSize": "14px"},
                ),
            ], className="drought-control", style={"gridColumn": "span 2"}),
            html.Div([
                html.Label("Stress year"),
                dcc.Input(id="m5-stress-year", type="number",
                          min=1, max=10, step=1, value=_s5.get("stress_year", 9),
                          className="alloc-num-input"),
            ], className="drought-control"),
        ], className="drought-controls",
           style={"gridTemplateColumns": "2fr 1fr"}),
    ], className="panel")


def _m6_rebalancing_controls(onset: int) -> html.Div:
    """
    Post-event rebalancing panel for Module 6.
    After the combined crash+drought clears, set a new strategic allocation for recovery.
    No second stress-test branch — Module 6 assumes no repeat of the combined event.
    Rebalancing occurs end-of-year (after growth, before that year's drawdown if any),
    so it is valid to rebalance in the final drought year (onset + 2) or later.
    """
    _s6 = _SAVED.get("m6", {})
    default_reb_year = _s6.get("rebalance_year", min(onset + 3, 9))
    return html.Div([
        html.H2("Post-event rebalancing strategy"),
        html.Div(
            "After surviving the combined crash and drought, set a new strategic allocation "
            "for the recovery phase. The engine assumes no repeat of the combined event — "
            "the forward path uses CMA (BAU) returns from the rebalance year onward.",
            className="section-note",
        ),
        html.Div([
            html.Div([
                html.Label("Rebalance year"),
                dcc.Input(id="m6-rebalance-year", type="number",
                          min=onset, max=10, step=1, value=default_reb_year,
                          className="alloc-num-input"),
                html.Div("Occurs at year-end: after growth, before that year's drawdown.",
                         style={"fontSize": "11px", "color": COLORS["muted"],
                                "marginTop": "3px", "fontStyle": "italic"}),
            ], className="drought-control"),
            html.Div([html.Label("New STI (%)"),
                      dcc.Input(id="m6-reb-STI", type="number",
                                min=0, max=100, step=1, value=_s6.get("reb_STI", 15),
                                className="alloc-num-input")],
                     className="drought-control"),
            html.Div([html.Label("New MTG (%)"),
                      dcc.Input(id="m6-reb-MTG", type="number",
                                min=0, max=100, step=1, value=_s6.get("reb_MTG", 35),
                                className="alloc-num-input")],
                     className="drought-control"),
            html.Div([html.Label("New LTG (%)"),
                      dcc.Input(id="m6-reb-LTG", type="number",
                                min=0, max=100, step=1, value=_s6.get("reb_LTG", 50),
                                className="alloc-num-input")],
                     className="drought-control"),
        ], className="drought-controls",
           style={"gridTemplateColumns": "repeat(4, minmax(140px, 1fr))"}),

        html.Div(id="m6-rebalance-constraint",
                 style={"marginTop": "10px", "fontSize": "12.5px"}),
        html.Div(id="m6-drift-weights",
                 style={"marginTop": "8px", "fontSize": "12px", "color": COLORS["muted"]}),

        html.H3("Board policy compliance — rebalanced allocation",
                style={"marginTop": "20px", "marginBottom": "4px",
                       "fontSize": "13px", "fontWeight": "600",
                       "textTransform": "uppercase", "letterSpacing": "0.05em",
                       "color": COLORS["muted"]}),
        html.Div(id="m6-reb-compliance"),
    ], className="panel")


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
                "Auto-populated from each trust's actual compounded balance just before the "
                "drawdown. Override manually if needed — values are normalised to 100%; "
                "any shortfall spills over STI → MTG → LTG.",
                style={"fontSize": "11.5px", "color": COLORS["muted"],
                       "lineHeight": "1.35", "marginBottom": "8px"},
            ),
            html.Div(id="m5-predrawdown-balances",
                     style={"fontSize": "12px", "marginBottom": "10px",
                            "padding": "8px 10px", "borderRadius": "6px",
                            "background": "rgba(255,255,255,0.04)",
                            "border": "1px solid rgba(255,255,255,0.08)"}),
            html.Div([
                html.Div([html.Label("STI (%)"),
                          dcc.Input(id="m5-onset-split-STI", type="number",
                                    min=0, max=100, step=0.1,
                                    value=_SAVED.get("m5", {}).get("onset_split_STI", 33),
                                    className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Label("MTG (%)"),
                          dcc.Input(id="m5-onset-split-MTG", type="number",
                                    min=0, max=100, step=0.1,
                                    value=_SAVED.get("m5", {}).get("onset_split_MTG", 33),
                                    className="alloc-num-input")],
                         className="drought-control"),
                html.Div([html.Label("LTG (%)"),
                          dcc.Input(id="m5-onset-split-LTG", type="number",
                                    min=0, max=100, step=0.1,
                                    value=_SAVED.get("m5", {}).get("onset_split_LTG", 34),
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
        html.Div(_fmt_m(summary["remaining_value"]), className="summary-headline"),
        html.Div("Remaining portfolio value",
                 style={"fontSize": "12px", "color": COLORS["muted"]}),
        html.Div([
            html.Div([html.Div("Drawdown this year", className="lbl"),
                      html.Div(_fmt_m(summary["drawdown_this_year"]), className="val")],
                     className="summary-item"),
            html.Div([html.Div("Spread cost this year", className="lbl"),
                      html.Div(_fmt_m(summary["spread_cost_this_year"]), className="val")],
                     className="summary-item"),
            html.Div([html.Div("Residual drawdowns to come", className="lbl"),
                      html.Div(_fmt_m(summary["residual_drawdown_to_come"]), className="val")],
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
                              value=_SAVED.get("m5", {}).get("severity", "Severe"), clearable=False,
                              style={"fontFamily": FONT_STACK, "fontSize": "14px"})],
                         className="drought-control"),
                html.Div([html.Label("Total relief amount"),
                          dcc.Slider(id="m5-relief", min=50, max=2000, step=10,
                              value=_SAVED.get("m5", {}).get("relief", 1500),
                              marks={50: "$50M", 500: "$500M", 1000: "$1B", 2000: "$2B"},
                              tooltip={"placement": "bottom", "always_visible": True,
                                       "template": "${value}M"})],
                         className="drought-control", style={"gridColumn": "span 2"}),
                html.Div([html.Label("Onset year"),
                          html.Div("Year 4", className="alloc-num-input",
                                   style={"paddingTop": "6px", "fontWeight": "600"}),
                          dcc.Input(id="m5-onset", type="number", value=4,
                                    style={"display": "none"})],
                         className="drought-control"),
                html.Div([html.Label("Year-onset fraction (%)"),
                          html.Div("50%", className="alloc-num-input",
                                   style={"paddingTop": "6px", "fontWeight": "600"}),
                          dcc.Input(id="m5-fraction", type="number", value=50,
                                    style={"display": "none"})],
                         className="drought-control"),
            ], className="drought-controls"),
            _onset_split_controls(),
            html.Div(id="m5-config-summary",
                     style={"fontSize": "12.5px", "color": COLORS["muted"], "marginTop": "10px"}),
        ], className="panel"),

        html.Div([html.H2("Portfolio value trajectory (BAU)"),
                  html.Div("AUD value at end of each year given the Initial Allocation, "
                           "CMA returns, and the drought schedule above. No rebalancing applied.",
                           className="section-note"),
                  dcc.Graph(id="m5-value-chart", config={"displayModeBar": False})],
                 className="panel"),

        html.Div([html.H2("Year-onset outcome"),
                  html.Div("Key state immediately after the year-onset drawdown.",
                           className="section-note"),
                  html.Div(id="m5-exec-verdict", className="decision-band",
                           style={"marginBottom": "14px"}),
                  html.Div(id="m5-summary-card")],
                 className="panel"),

        # ── Rebalancing strategy ─────────────────────────────────────────────
        _rebalancing_controls(onset=4),   # default onset=4; callback refreshes live

        html.Div([
            html.H2("Branch comparison: post-rebalance BAU vs stress-test"),
            html.Div(
                "Both branches share the same path through the drought and the rebalance. "
                "Branch (a) continues under BAU returns. "
                "Branch (b) applies the chosen market shock at the specified year.",
                className="section-note",
            ),
            dcc.Graph(id="m5-branch-chart", config={"displayModeBar": False}),
            html.Div(id="m5-branch-summary",
                     style={"marginTop": "14px", "fontSize": "13px"}),
        ], className="panel"),

        html.Div([
            html.H2("Trust composition over time"),
            html.Div(
                "AUD held in each trust at end of each year, including the effect of the "
                "post-drought rebalance. STI band shrinks during drought years, then shifts "
                "to the new strategic allocation at the rebalance year.",
                className="section-note",
            ),
            html.Div([
                html.Span("View: ", style={"fontSize": "13px", "color": COLORS["muted"],
                                           "marginRight": "8px"}),
                dcc.RadioItems(
                    id="m5-comp-toggle",
                    options=[
                        {"label": "Branch (a) — BAU",    "value": "bau"},
                        {"label": "Branch (b) — Stress", "value": "stress"},
                    ],
                    value="bau",
                    inline=True,
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "20px", "fontSize": "13px",
                                "cursor": "pointer"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
            dcc.Graph(id="m5-composition-chart", config={"displayModeBar": False}),
        ], className="panel"),

        html.Div([
            html.H2("Year-by-year summary"),
            html.Div(
                "Toggle above also controls this table. All monetary values in millions (AUD). "
                "Growth = portfolio return for the year before drawdown; in the rebalance year "
                "this is net of the rebalance spread cost.",
                className="section-note",
            ),
            html.Div(id="m5-projection-table-container"),
            html.Div(id="m5-totals", style={"marginTop": "12px", "fontSize": "13px"}),
            html.H3("Master fund return summary",
                    style={"marginTop": "24px", "marginBottom": "4px",
                           "fontSize": "13px", "fontWeight": "600",
                           "textTransform": "uppercase", "letterSpacing": "0.05em",
                           "color": COLORS["muted"]}),
            html.Div(
                "Annual gross and net return (excl. drawdown impact), per-trust contribution "
                "to net return, and CPI+2.5% target flag. Weights are the starting-year mix.",
                className="section-note",
                style={"marginBottom": "8px"},
            ),
            html.Div(id="m5-return-summary"),
        ], className="panel"),

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
    M = 1_000_000
    years       = [0] + [y.year for y in baseline.years]
    base_vals   = [baseline.initial_value / M] + [y.ending_value / M for y in baseline.years]
    stress_vals = [stressed.initial_value / M] + [y.ending_value / M for y in stressed.years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=base_vals, mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2), marker=dict(size=5),
        name="Drought only",
        hovertemplate="Year %{x}<br>Value: $%{y:,.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=stress_vals, mode="lines+markers",
        line=dict(color=COLORS["fail"], width=2), marker=dict(size=5),
        name="Combined (crash + drought)",
        hovertemplate="Year %{x}<br>Value: $%{y:,.1f}M<extra></extra>"))

    for dy in drought_years:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.35)
    fig.add_vline(x=shock_year, line=dict(color=COLORS["ink"], width=1.5, dash="dash"),
                  annotation_text=f"Market shock (Y{shock_year})",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=COLORS["ink"]))
    fig.add_hline(y=baseline.initial_value / M,
                  line=dict(color=COLORS["muted"], width=1, dash="dash"),
                  annotation_text="Starting value", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=400, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def _m6_drawdown_profile(
    stressed: dr.ProjectionResult,
    schedule: dict,
) -> html.Div:
    """
    Drawdown profile panel for Module 6 — mirrors the Module 5 predrawdown-balances display
    but uses the STRESSED projection (crash returns already applied) so the holdings shown
    reflect the actual portfolio state before each drought redemption under the combined event.
    Shows actual redemption amounts per trust from the projection rather than a sequential estimate.
    """
    colors    = {"STI": "#5bc8f5", "MTG": "#7fba00", "LTG": "#f5a623"}
    tag_style = {"partial":     {"color": "#f5a623", "fontSize": "11px"},
                 "fully drawn": {"color": "#e05c5c", "fontSize": "11px"},
                 "untouched":   {"color": "#888",    "fontSize": "11px"}}

    def _tag(redemption_gross: float, holding_pre: float) -> str:
        s = tc.TRUST_SELL_SPREADS.get("STI", 0)  # approximate; per-trust done below
        if redemption_gross <= 0:
            return "untouched"
        if redemption_gross < holding_pre - 1.0:
            return "partial"
        return "fully drawn"

    rows = []
    for yr, drawdown in sorted(schedule.items()):
        if drawdown <= 0 or yr > len(stressed.years):
            continue
        y       = stressed.years[yr - 1]
        pre_val = y.pre_drawdown_value
        pre_h   = {t: y.pre_drawdown_weights[t] * pre_val for t in tc.TRUST_NAMES}
        redemp  = y.redemption_amounts   # {trust: gross_redemption}

        spans = []
        for trust in tc.TRUST_NAMES:
            gross = redemp.get(trust, 0.0)
            tag   = "untouched" if gross <= 0 \
                    else ("fully drawn" if gross >= pre_h[trust] - 1.0 else "partial")
            net_contrib = gross * (1 - tc.TRUST_SELL_SPREADS[trust])
            contrib_str = f"  (contributes {_fmt_m(net_contrib)} net)" if gross > 0 else ""
            spans.append(html.Span([
                html.Span(f"{trust}  {_fmt_m(pre_h[trust])}  "
                          f"({pre_h[trust]/pre_val*100:.1f}%)",
                          style={"color": colors[trust]}),
                html.Span(f"  [{tag}]{contrib_str}", style=tag_style[tag]),
            ], style={"marginRight": "16px", "display": "inline-block"}))

        note = ""
        if yr == min(schedule.keys()):
            note = " — onset-year split applied (see Module 5 drawdown split inputs)"
        rows.append(html.Div([
            html.Div(f"Year {yr}  —  drawdown {_fmt_m(drawdown)}  "
                     f"(total fund before drawdown {_fmt_m(pre_val)}){note}:",
                     style={"fontWeight": "600", "marginBottom": "3px",
                            "color": COLORS.get("text", "#e0e0e0")}),
            html.Div(spans),
        ], style={"marginBottom": "12px",
                  "padding": "8px 10px", "borderRadius": "6px",
                  "background": "rgba(255,255,255,0.03)",
                  "border": "1px solid rgba(255,255,255,0.07)"}))

    if not rows:
        return html.Div()
    return html.Div([
        html.Div("Pre-drawdown trust balances shown are from the combined (crash+drought) "
                 "stressed projection — holdings already reflect shocked returns before each "
                 "drought redemption.",
                 style={"fontSize": "11.5px", "color": COLORS["muted"],
                        "marginBottom": "10px", "lineHeight": "1.4"}),
        html.Div(rows),
    ])


def _m6_forward_figure(
    baseline: dr.ProjectionResult,
    stressed: dr.ProjectionResult,
    rebalanced: dr.ProjectionResult,
    shock_year: int,
    drought_years: list,
    rebalance_year: int,
) -> go.Figure:
    """
    Three-line recovery chart for Module 6:
      - Drought-only BAU (teal, reference)
      - Combined crash+drought, no rebalance (red)
      - Combined crash+drought → rebalance → BAU recovery (orange)
    """
    M     = 1_000_000
    years = [0] + [y.year for y in baseline.years]
    base_vals  = [baseline.initial_value / M]  + [y.ending_value / M for y in baseline.years]
    stress_vals= [stressed.initial_value / M]  + [y.ending_value / M for y in stressed.years]
    reb_vals   = [rebalanced.initial_value / M] + [y.ending_value / M for y in rebalanced.years]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=base_vals, mode="lines+markers",
        name="Drought only (reference)",
        line=dict(color=COLORS["accent"], width=2),
        marker=dict(size=5, color=COLORS["accent"]),
        hovertemplate="Year %{x}<br>Drought only: $%{y:,.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=stress_vals, mode="lines+markers",
        name="Combined — no rebalance",
        line=dict(color=COLORS["fail"], width=2, dash="dash"),
        marker=dict(size=5, color=COLORS["fail"]),
        hovertemplate="Year %{x}<br>Combined (no rebalance): $%{y:,.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=reb_vals, mode="lines+markers",
        name="Combined → rebalance → BAU recovery",
        line=dict(color="#C07A2A", width=2.5),
        marker=dict(size=6, color="#C07A2A"),
        hovertemplate="Year %{x}<br>Recovery path: $%{y:,.1f}M<extra></extra>"))

    for dy in drought_years:
        fig.add_vline(x=dy, line=dict(color=COLORS["fail"], width=1, dash="dot"), opacity=0.35)
    fig.add_vline(x=shock_year, line=dict(color=COLORS["ink"], width=1.5, dash="dash"),
                  annotation_text=f"Shock+drought Y{shock_year}",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=COLORS["ink"]))
    fig.add_vline(x=rebalance_year, line=dict(color="#C07A2A", width=1.5, dash="dashdot"),
                  annotation_text=f"Rebalance Y{rebalance_year}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#C07A2A"))
    fig.add_hline(y=baseline.initial_value / M,
                  line=dict(color=COLORS["muted"], width=1, dash="dash"),
                  annotation_text="Starting value", annotation_position="bottom right",
                  annotation_font=dict(size=10, color=COLORS["muted"]))

    fig.update_layout(height=420, margin=dict(l=70, r=20, t=50, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def module_6_layout() -> html.Div:
    return html.Div([
        # ── Controls ─────────────────────────────────────────────────────────────
        html.Div([
            html.H2("Module 6 — Combined Stress (Market Crash + Drought)"),
            html.Div(
                "Stress and drought hit simultaneously. After the event clears, set a new "
                "strategic allocation and see the recovery under BAU. Drought parameters are "
                "inherited from Module 5; allocation from Module 3.",
                className="section-note"),
            html.Div([
                html.Strong("Note on annualisation. "),
                "Short-window scenarios (e.g. COVID Crash, 2 months) are applied as "
                "cumulative event-window shocks rather than annualised rates. Longer "
                "historical windows remain annualised for CMA-style comparison.",
            ], style={"backgroundColor": COLORS["warn_bg"],
                      "border": f"1px solid {COLORS['warn_border']}",
                      "color": COLORS["warn_ink"], "padding": "10px 14px",
                      "borderRadius": "4px", "marginTop": "10px", "marginBottom": "16px",
                      "fontSize": "13px", "lineHeight": "1.5"}),
            html.Div([
                html.Div([html.Label("Market shock scenario"),
                          dcc.Dropdown(id="m6-scenario",
                              options=[{"label": s, "value": s} for s in SCENARIO_ORDER],
                              value=_SAVED.get("m6", {}).get("scenario", "GFC"), clearable=False,
                              style={"fontFamily": FONT_STACK, "fontSize": "14px"})],
                         className="drought-control"),
                html.Div([html.Label("Shock year"),
                          dcc.Input(id="m6-shock-year", type="number", min=1, max=10,
                                    step=1, value=_SAVED.get("m6", {}).get("shock_year", 4),
                                    className="alloc-num-input")],
                         className="drought-control"),
            ], className="drought-controls"),
            html.Div(id="m6-config-summary",
                     style={"fontSize": "12.5px", "color": COLORS["muted"], "marginTop": "10px"}),
        ], className="panel"),

        # ── Combined trajectory (existing: drought-only vs crash+drought) ────────
        html.Div([html.H2("Combined trajectory"),
                  html.Div("Drought-only path (teal) vs combined crash+drought path (red). "
                           "Dotted lines = drought years; dashed line = shock year.",
                           className="section-note"),
                  dcc.Graph(id="m6-value-chart", config={"displayModeBar": False})],
                 className="panel"),

        # ── Joint impact summary ─────────────────────────────────────────────────
        html.Div([html.H2("Joint impact summary"),
                  html.Div("Side-by-side outcomes for drought-only and combined-stress scenarios.",
                           className="section-note"),
                  html.Div(id="m6-summary-grid")],
                 className="panel"),

        # ── Drawdown profile ─────────────────────────────────────────────────────
        html.Div([
            html.H2("Drought drawdown profile — under combined stress"),
            html.Div(
                "Trust balances and actual redemptions at each drought year, computed from "
                "the combined crash+drought projection. Balances already reflect the shocked "
                "trust returns before each drawdown occurs.",
                className="section-note"),
            html.Div(id="m6-drawdown-profile"),
        ], className="panel"),

        # ── Post-event rebalancing controls ──────────────────────────────────────
        _m6_rebalancing_controls(onset=4),

        # ── Recovery trajectory: three-line chart ────────────────────────────────
        html.Div([html.H2("Recovery trajectory"),
                  html.Div(
                      "Three paths from the same starting point: drought-only BAU (reference, teal), "
                      "combined crash+drought with no rebalance (red), and the recovery path after "
                      "rebalancing (orange). The rebalanced path assumes BAU returns from rebalance "
                      "year onward — no repeat of the combined event.",
                      className="section-note"),
                  dcc.Graph(id="m6-forward-chart", config={"displayModeBar": False})],
                 className="panel"),

        # ── Year-by-year summary table ───────────────────────────────────────────
        html.Div([html.H2("Year-by-year projection — recovery path"),
                  html.Div("Based on the combined crash+drought → rebalance → BAU recovery.",
                           className="section-note"),
                  html.Div(id="m6-projection-table-container"),
                  html.Div(id="m6-totals",
                           style={"marginTop": "10px", "fontSize": "12.5px",
                                  "color": COLORS["muted"]})],
                 className="panel"),

        # ── Master fund return summary ────────────────────────────────────────────
        html.Div([html.H2("Master fund return summary — recovery path"),
                  html.Div(id="m6-return-summary")],
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


_AU_ASSETS = {
    "Cash", "Australian Short Duration Bond", "Australian Fixed Income",
    "Australian Listed Equity", "Australian Listed Property",
}
_BOND_ASSETS = {
    "Australian Short Duration Bond", "Australian Fixed Income",
    "Global Fixed Income (Hedged)", "Global Credit (Hedged)",
}
_CASH_ASSETS = {"Cash"}
# All remaining assets (not Cash, not Bond) are treated as Equity


def _portfolio_asset_class_exposure(trust_weights: dict) -> dict:
    """Compute effective portfolio-level exposure by domicile and asset type."""
    au = global_ = cash = bond = equity = 0.0
    for t, tw in trust_weights.items():
        for ac, aw in tc.TRUST_RAW_WEIGHTS[t].items():
            exposure = tw * aw
            if ac in _AU_ASSETS:
                au += exposure
            else:
                global_ += exposure
            if ac in _CASH_ASSETS:
                cash += exposure
            elif ac in _BOND_ASSETS:
                bond += exposure
            else:
                equity += exposure
    return {"au": au, "global": global_, "cash": cash, "bond": bond, "equity": equity}


def _board_compliance_table(metrics: dict) -> dash_table.DataTable:
    w = metrics["weights"]
    liq = metrics["liq"]
    target_ok = metrics["return"] >= metrics["target"] - 1e-9
    exp = _portfolio_asset_class_exposure(w)
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
        {"requirement": "Domicile exposure (AU vs Global)",
         "result": f"AU {exp['au']*100:.1f}% / Global {exp['global']*100:.1f}%",
         "status": "Info"},
        {"requirement": "Asset type exposure (Cash / Bond / Equity)",
         "result": f"Cash {exp['cash']*100:.1f}% / Bond {exp['bond']*100:.1f}% / Equity {exp['equity']*100:.1f}%",
         "status": "Info"},
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
            {"if": {"filter_query": "{status} = Info", "column_id": "status"},
             "color": COLORS["muted"]},
        ],
        style_header={"backgroundColor": COLORS["bg"], "fontFamily": FONT_STACK,
                      "fontWeight": "600", "fontSize": "12px",
                      "borderBottom": f"2px solid {COLORS['border']}"},
        style_data={"borderBottom": f"1px solid {COLORS['border']}"},
    )


# ---------------------------------------------------------------------------
# Top-level app layout
# ---------------------------------------------------------------------------

def module_7_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H2("Module 7 — Executive Scenario Summary"),
            html.Div(
                "Consolidated view of both stress scenarios. Reads live from all module "
                "inputs — update any parameter in Modules 1–6 and this summary refreshes.",
                className="section-note"),
        ], className="panel"),
        html.Div(id="m7-content"),
    ])


app.layout = html.Div([
    dcc.Store(id="cma-store", data=_initial_cma_store()),
    dcc.Store(id="portfolio-allocation-store",
              data=_SAVED.get("portfolio", {"STI": 0.33, "MTG": 0.33, "LTG": 0.34})),
    dcc.Store(id="m1-ignored-flags",
              data=_SAVED.get("ignored_flags", {})),
    html.Div([
        html.H1("NSWDF Portfolio Dashboard"),
        html.Div("AUD 3 billion drought reserve \u2014 STI / MTG / LTG allocation analysis",
                 className="subtitle"),
    ], className="app-header"),
    dcc.Tabs(id="main-tabs", value="m1", children=[
        dcc.Tab(label="1. CMA Inputs",          value="m1", children=module_1_layout()),
        dcc.Tab(label="2. Trust Characteristics", value="m2", children=module_2_layout()),
        dcc.Tab(label="3. Initial Allocation",   value="m3", children=module_3_layout()),
        dcc.Tab(label="4. Market Stress",       value="m4",
                children=module_4_layout()),
        dcc.Tab(label="5. Drought First",        value="m5",
                children=module_5_layout()),
        dcc.Tab(label="6. Combined Stress",     value="m6",
                children=module_6_layout()),
        dcc.Tab(label="7. Executive Summary",   value="m7",
                children=module_7_layout()),
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
    """Read forecast returns and locked historical volatility from the CMA table."""
    by_ac = {row["asset_class"]: row for row in rv_data}
    returns, vols = [], []
    for ac in tc.ASSET_CLASSES:
        row = by_ac.get(ac, {})
        r = row.get("expected_return")
        v = row.get("hist_vol", row.get("volatility"))
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
    column when the global period changes. Forecast Vol is locked to Hist. Vol.
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
        new_row["volatility"]  = h_vol
        new_row["delta"]       = round(float(f_ret) - h_ret, 3)
        updated.append(new_row)
    return updated


@app.callback(
    Output("m1-hist-cpi-ref", "children"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
)
def update_hist_cpi_ref(sm, sy, em, ey):
    """Show average AUS CPI YoY % for the selected analysis period."""
    sm = sm or _DATE_MIN_M;  sy = sy or _DATE_MIN_Y
    em = em or _DATE_MAX_M;  ey = ey or _DATE_MAX_Y
    try:
        start = pd.Timestamp(year=int(sy), month=int(sm), day=1)
        end   = pd.Timestamp(year=int(ey), month=int(em), day=1)
        mask  = (_macro_df.index >= start) & (_macro_df.index <= end)
        aus = _macro_df.loc[mask, "AUS CPI (YoY %)"].dropna().mean()
        us  = _macro_df.loc[mask, "US CPI (YoY %)"].dropna().mean()
        if pd.isna(aus) and pd.isna(us):
            return ""
        parts = []
        if not pd.isna(aus):
            parts.append(f"Hist. AUS CPI (avg): {aus:.2f}%")
        if not pd.isna(us):
            parts.append(f"US CPI (avg): {us:.2f}%")
        return "| " + "  |  ".join(parts)
    except Exception:
        return ""


@app.callback(
    Output("cma-rv-table", "data", allow_duplicate=True),
    Input("cma-rv-table", "data"),
    prevent_initial_call=True,
)
def update_delta_on_forecast_edit(data):
    """
    Recompute the Δ Return column whenever a user edits the Forecast Return
    column (column 3).  hist_return is already stored in each row, so no
    period inputs are needed.  Raises PreventUpdate if nothing changed to
    avoid an infinite loop with update_cma_hist_columns.
    """
    if not data:
        raise PreventUpdate
    updated = []
    changed = False
    for row in data:
        new_row = dict(row)
        try:
            f_ret = float(new_row.get("expected_return", 0))
        except (TypeError, ValueError):
            f_ret = 0.0
        try:
            h_ret = float(new_row.get("hist_return", 0))
        except (TypeError, ValueError):
            h_ret = 0.0
        expected_delta = round(f_ret - h_ret, 3)
        if new_row.get("delta") != expected_delta:
            new_row["delta"] = expected_delta
            changed = True
        updated.append(new_row)
    if not changed:
        raise PreventUpdate
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
# CMA consistency validation
# ---------------------------------------------------------------------------

# Risk-tier groupings: 0=Cash, 1=Bonds, 2=Listed Equity/Infra, 3=Private Equity
_CMA_RISK_TIERS: dict[int, list[str]] = {
    0: ["Cash"],
    1: [
        "Australian Short Duration Bond",
        "Australian Fixed Income",
        "Global Fixed Income (Hedged)",
        "Global Credit (Hedged)",
    ],
    2: [
        "Australian Listed Equity",
        "Global Listed Equity (Unhedged)",
        "Global Listed Equity (Hedged)",
        "Australian Listed Property",
        "Global Infrastructure (Unhedged)",
    ],
    3: ["Global Private Equity"],
}
_CMA_TIER_LABELS = {
    0: "Cash",
    1: "Bonds",
    2: "Listed Equity / Real Assets",
    3: "Private Equity",
}
# (unhedged, hedged) pairs of the same underlying constituent
_CMA_HEDGE_PAIRS = [
    ("Global Listed Equity (Unhedged)", "Global Listed Equity (Hedged)"),
]


def _compute_cma_flags(cma_store: dict) -> list[str]:
    """Return a list of flag strings based on CMA return/vol consistency rules."""
    returns_arr = cma_store.get("returns", [])
    vols_arr    = cma_store.get("vols", [])
    ret_by_ac   = {ac: returns_arr[i] * 100 for i, ac in enumerate(tc.ASSET_CLASSES)}
    vol_by_ac   = {ac: vols_arr[i]    * 100 for i, ac in enumerate(tc.ASSET_CLASSES)}

    flags: list[str] = []

    # Dimension 1: hedged vs unhedged same constituent
    for uh, h in _CMA_HEDGE_PAIRS:
        vol_u, vol_h = vol_by_ac.get(uh, 0), vol_by_ac.get(h, 0)
        ret_u, ret_h = ret_by_ac.get(uh, 0), ret_by_ac.get(h, 0)
        if vol_u > vol_h and ret_u <= ret_h:
            flags.append(
                f"Hedged/Unhedged — {uh} carries more vol ({vol_u:.1f}% vs {vol_h:.1f}%) "
                f"but a lower or equal forecast return ({ret_u:.1f}% vs {ret_h:.1f}%). "
                f"Currency risk should earn a premium."
            )
        elif vol_h > vol_u and ret_h <= ret_u:
            flags.append(
                f"Hedged/Unhedged — {h} carries more vol ({vol_h:.1f}% vs {vol_u:.1f}%) "
                f"but a lower or equal forecast return ({ret_h:.1f}% vs {ret_u:.1f}%). "
                f"Check hedging cost and basis-risk assumptions."
            )

    # Dimension 2: cross-tier hierarchy (Cash < Bonds < Equity/Infra < PE)
    tier_avg: dict[int, float] = {}
    for tier, assets in _CMA_RISK_TIERS.items():
        vals = [ret_by_ac[ac] for ac in assets if ac in ret_by_ac]
        tier_avg[tier] = sum(vals) / len(vals) if vals else 0.0
    for lo, hi in [(0, 1), (1, 2), (2, 3)]:
        if tier_avg[hi] <= tier_avg[lo]:
            flags.append(
                f"Risk hierarchy — {_CMA_TIER_LABELS[hi]} avg return "
                f"({tier_avg[hi]:.1f}%) ≤ {_CMA_TIER_LABELS[lo]} ({tier_avg[lo]:.1f}%). "
                f"Higher-risk tiers should earn a return premium."
            )

    # Dimension 3: within-tier vol-return consistency
    for tier, assets in _CMA_RISK_TIERS.items():
        pairs = [(ac, ret_by_ac[ac], vol_by_ac[ac]) for ac in assets if ac in ret_by_ac]
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                ac_a, ret_a, vol_a = pairs[i]
                ac_b, ret_b, vol_b = pairs[j]
                if abs(vol_a - vol_b) < 1.0:
                    continue
                if vol_a > vol_b and ret_a < ret_b:
                    flags.append(
                        f"Within-tier ({_CMA_TIER_LABELS[tier]}) — {ac_a} has higher vol "
                        f"({vol_a:.1f}%) than {ac_b} ({vol_b:.1f}%) "
                        f"but a lower forecast return ({ret_a:.1f}% vs {ret_b:.1f}%)."
                    )
                elif vol_b > vol_a and ret_b < ret_a:
                    flags.append(
                        f"Within-tier ({_CMA_TIER_LABELS[tier]}) — {ac_b} has higher vol "
                        f"({vol_b:.1f}%) than {ac_a} ({vol_a:.1f}%) "
                        f"but a lower forecast return ({ret_b:.1f}% vs {ret_a:.1f}%)."
                    )
    return flags


def _flag_row(flag_text: str, ignored: bool, note: str = "") -> html.Div:
    """Render a single flag row with a dismiss checkbox and optional note input."""
    text_style = (
        {"fontSize": "12.5px", "color": "#AAA",
         "textDecoration": "line-through", "lineHeight": "1.45", "flexShrink": "1"}
        if ignored else
        {"fontSize": "12.5px", "color": "#6B2A10", "lineHeight": "1.45", "flexShrink": "1"}
    )
    right_side: list = [html.Span(flag_text, style=text_style)]
    if ignored:
        right_side.append(
            dcc.Input(
                id={"type": "m1-flag-note", "index": flag_text},
                type="text",
                value=note,
                placeholder="Add a note…",
                debounce=True,
                style={
                    "marginLeft": "10px", "fontSize": "12px",
                    "color": "#555", "border": "1px solid #CCC",
                    "borderRadius": "3px", "padding": "2px 6px",
                    "width": "220px", "flexShrink": "0",
                    "backgroundColor": "#FAFAFA",
                },
            )
        )
    return html.Div([
        dcc.Checklist(
            id={"type": "m1-flag-cb", "index": flag_text},
            options=[{"label": "", "value": "ignored"}],
            value=["ignored"] if ignored else [],
            style={"marginTop": "2px", "flexShrink": "0"},
            inputStyle={"cursor": "pointer", "width": "14px", "height": "14px"},
        ),
        html.Div(right_side, style={
            "display": "flex", "alignItems": "flex-start",
            "flexWrap": "wrap", "gap": "0", "flex": "1",
        }),
    ], style={
        "display": "flex", "alignItems": "flex-start",
        "gap": "8px", "marginBottom": "8px",
    })


@app.callback(
    Output("m1-cma-flags", "children"),
    Input("cma-store", "data"),
    Input("m1-ignored-flags", "data"),
)
def update_cma_flags(cma_store, ignored_flags):
    if not cma_store:
        return []

    # ignored_flags is {flag_text: note_text}
    ignored_map: dict = ignored_flags or {}
    all_flags   = _compute_cma_flags(cma_store)
    active    = [f for f in all_flags if f not in ignored_map]
    dismissed = [f for f in all_flags if f in ignored_map]

    if not all_flags:
        return html.Div([
            html.Span("✓ ", style={"color": "#2E6B5E", "fontWeight": "700"}),
            html.Span(
                "All CMA return/vol assumptions are internally consistent "
                "(hedged/unhedged, cross-tier hierarchy, within-tier risk ordering).",
                style={"color": "#2E6B5E", "fontSize": "13px"},
            ),
        ], style={
            "padding": "8px 14px",
            "backgroundColor": "#EEF6F2",
            "borderRadius": "4px",
            "border": "1px solid #B5D8CB",
        })

    n_active = len(active)
    header_text = (
        f"⚠ {n_active} active CMA flag{'s' if n_active != 1 else ''}"
        + (f" · {len(dismissed)} dismissed" if dismissed else "")
    ) if n_active else (
        f"✓ All flags dismissed ({len(dismissed)} acknowledged)"
    )
    header_color = "#2E6B5E" if n_active == 0 else "#8B3A20"
    outer_style  = {
        "padding": "10px 14px",
        "borderRadius": "4px",
        "border": f"1px solid {'#B5D8CB' if n_active == 0 else '#E8C4B0'}",
        "backgroundColor": "#EEF6F2" if n_active == 0 else "#FDF2EE",
    }

    rows = (
        [_flag_row(f, ignored=False) for f in active]
        + [_flag_row(f, ignored=True, note=ignored_map.get(f, "")) for f in dismissed]
    )

    return html.Div([
        html.Div(header_text, style={
            "fontWeight": "700", "marginBottom": "10px",
            "color": header_color, "fontSize": "13px",
        }),
        html.Div(rows),
        html.Div(
            "Tick a flag to acknowledge / dismiss it. "
            "Flags are advisory and do not block downstream calculations.",
            style={"fontSize": "11.5px", "color": "#888",
                   "marginTop": "8px", "fontStyle": "italic"},
        ),
    ], style=outer_style)


@app.callback(
    Output("m1-ignored-flags", "data"),
    Input({"type": "m1-flag-cb",   "index": ALL}, "value"),
    Input({"type": "m1-flag-note", "index": ALL}, "value"),
    State({"type": "m1-flag-cb",   "index": ALL}, "id"),
    State({"type": "m1-flag-note", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def sync_flag_ignores(cb_values, note_values, cb_ids, note_ids):
    """Persist dismissed flags and their notes as {flag_text: note} dict."""
    result: dict[str, str] = {}
    # Build dismissed set from checkboxes
    for val, id_dict in zip(cb_values, cb_ids):
        if val:   # ["ignored"] = checked
            result[id_dict["index"]] = ""
    # Overlay notes (only note inputs for dismissed flags exist in DOM)
    for note, id_dict in zip(note_values, note_ids):
        flag_text = id_dict["index"]
        if flag_text in result:
            result[flag_text] = note or ""
    return result


# ---------------------------------------------------------------------------
# Persist user state to user_state.json on every meaningful change
# ---------------------------------------------------------------------------

@app.callback(
    Output("cma-store", "id"),   # dummy output — no real update needed
    Input("cma-store",                  "data"),
    Input("portfolio-allocation-store", "data"),
    Input("m1-ignored-flags",           "data"),
    Input("m1-start-m", "value"), Input("m1-start-y", "value"),
    Input("m1-end-m",   "value"), Input("m1-end-y",   "value"),
    # Module 5
    Input("m5-severity",          "value"),
    Input("m5-relief",            "value"),
    Input("m5-onset-split-STI",   "value"),
    Input("m5-onset-split-MTG",   "value"),
    Input("m5-onset-split-LTG",   "value"),
    Input("m5-rebalance-year",    "value"),
    Input("m5-reb-STI",           "value"),
    Input("m5-reb-MTG",           "value"),
    Input("m5-reb-LTG",           "value"),
    Input("m5-stress-scenario",   "value"),
    Input("m5-stress-year",       "value"),
    # Module 6
    Input("m6-scenario",          "value"),
    Input("m6-shock-year",        "value"),
    Input("m6-rebalance-year",    "value"),
    Input("m6-reb-STI",           "value"),
    Input("m6-reb-MTG",           "value"),
    Input("m6-reb-LTG",           "value"),
    prevent_initial_call=True,
)
def persist_user_state(
    cma, portfolio, ignored, sm, sy, em, ey,
    m5_severity, m5_relief,
    m5_split_sti, m5_split_mtg, m5_split_ltg,
    m5_reb_year, m5_reb_sti, m5_reb_mtg, m5_reb_ltg,
    m5_stress_scenario, m5_stress_year,
    m6_scenario, m6_shock_year,
    m6_reb_year, m6_reb_sti, m6_reb_mtg, m6_reb_ltg,
):
    _save_state({
        "cma_store":     cma,
        "portfolio":     portfolio,
        "ignored_flags": ignored,
        "period": {
            "sm": sm or _DATE_MIN_M,
            "sy": sy or _DATE_MIN_Y,
            "em": em or _DATE_MAX_M,
            "ey": ey or _DATE_MAX_Y,
        },
        "m5": {
            "severity":         m5_severity,
            "relief":           m5_relief,
            "onset_split_STI":  m5_split_sti,
            "onset_split_MTG":  m5_split_mtg,
            "onset_split_LTG":  m5_split_ltg,
            "rebalance_year":   m5_reb_year,
            "reb_STI":          m5_reb_sti,
            "reb_MTG":          m5_reb_mtg,
            "reb_LTG":          m5_reb_ltg,
            "stress_scenario":  m5_stress_scenario,
            "stress_year":      m5_stress_year,
        },
        "m6": {
            "scenario":       m6_scenario,
            "shock_year":     m6_shock_year,
            "rebalance_year": m6_reb_year,
            "reb_STI":        m6_reb_sti,
            "reb_MTG":        m6_reb_mtg,
            "reb_LTG":        m6_reb_ltg,
        },
    })
    raise PreventUpdate


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
    Output("m5-reb-STI", "value"),
    Output("m5-reb-MTG", "value"),
    Output("m5-reb-LTG", "value"),
    Input("m5-reb-STI",  "value"),
    Input("m5-reb-MTG",  "value"),
    Input("m5-reb-LTG",  "value"),
    prevent_initial_call=True,
)
def rebalance_m5_reb(sti, mtg, ltg):
    trigger = callback_context.triggered_id
    if trigger is None:
        return dash.no_update, dash.no_update, dash.no_update
    sti = sti or 0; mtg = mtg or 0; ltg = ltg or 0
    if trigger == "m5-reb-STI":
        new_mtg, new_ltg = _rebalance_other_two(sti, mtg, ltg)
        if abs(new_mtg - mtg) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return dash.no_update, round(new_mtg), round(new_ltg)
    if trigger == "m5-reb-MTG":
        new_sti, new_ltg = _rebalance_other_two(mtg, sti, ltg)
        if abs(new_sti - sti) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return round(new_sti), dash.no_update, round(new_ltg)
    if trigger == "m5-reb-LTG":
        new_sti, new_mtg = _rebalance_other_two(ltg, sti, mtg)
        if abs(new_sti - sti) < 0.5 and abs(new_mtg - mtg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return round(new_sti), round(new_mtg), dash.no_update
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("m6-reb-STI", "value"),
    Output("m6-reb-MTG", "value"),
    Output("m6-reb-LTG", "value"),
    Input("m6-reb-STI",  "value"),
    Input("m6-reb-MTG",  "value"),
    Input("m6-reb-LTG",  "value"),
    prevent_initial_call=True,
)
def rebalance_m6_reb(sti, mtg, ltg):
    trigger = callback_context.triggered_id
    if trigger is None:
        return dash.no_update, dash.no_update, dash.no_update
    sti = sti or 0; mtg = mtg or 0; ltg = ltg or 0
    if trigger == "m6-reb-STI":
        new_mtg, new_ltg = _rebalance_other_two(sti, mtg, ltg)
        if abs(new_mtg - mtg) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return dash.no_update, round(new_mtg), round(new_ltg)
    if trigger == "m6-reb-MTG":
        new_sti, new_ltg = _rebalance_other_two(mtg, sti, ltg)
        if abs(new_sti - sti) < 0.5 and abs(new_ltg - ltg) < 0.5:
            return dash.no_update, dash.no_update, dash.no_update
        return round(new_sti), dash.no_update, round(new_ltg)
    if trigger == "m6-reb-LTG":
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
        return sc.asset_returns, sc.description, sc.window_label, sc.return_basis, sc.n_months
    if scenario_name == "AUD Depreciation Shock":
        sc = st.build_aud_shock_scenario(_returns_df, cma_baseline)
        return sc.asset_returns, sc.description, sc.window_label, sc.return_basis, sc.n_months
    if scenario_name == "Interest Rate Shock (+200bps)":
        sc = st.build_rate_shock_scenario(cma_baseline)
        return sc.asset_returns, sc.description, None, sc.return_basis, sc.n_months
    raise ValueError(f"Unknown scenario: {scenario_name}")


@app.callback(
    Output("m4-shock-table",   "data"),
    Output("m4-shocked-store", "data"),
    Output("m4-scenario-meta", "children"),
    Output("m4-path-store",    "data"),
    Input("m4-scenario",       "value"),
    Input("m4-reset-button",   "n_clicks"),
    Input("cma-store",         "data"),
    Input("m1-start-m",        "value"),
    Input("m1-start-y",        "value"),
    Input("m1-end-m",          "value"),
    Input("m1-end-y",          "value"),
)
def update_m4_scenario(scenario_name, n_clicks, cma_store, sm, sy, em, ey):
    if not cma_store or not scenario_name:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    scenario_returns, desc, window_label, return_basis, n_months = _scenario_defaults(scenario_name, cma_baseline)
    sm = sm or _DATE_MIN_M; sy = sy or _DATE_MIN_Y
    em = em or _DATE_MAX_M; ey = ey or _DATE_MAX_Y
    selected_hist_returns = _asset_returns_for_basis(sm, sy, em, ey, return_basis, n_months)
    baseline_display = _forecast_returns_for_basis(cma_baseline, return_basis, n_months)
    shocked = _scenario_adjusted_returns(
        np.asarray(scenario_returns, dtype=float),
        cma_baseline,
        selected_hist_returns,
        scenario_name,
        return_basis,
        n_months,
    )
    if scenario_name in SCENARIO_WINDOWS_LIVE or scenario_name == "AUD Depreciation Shock":
        if return_basis == "event_window":
            desc = (
                f"{desc} Module 4 applies the shock as a same-horizon cumulative delta: "
                "forecast return converted to the event window + "
                "(event-window stress return - selected-period historical return converted "
                "to the event window). This avoids annualising short crash periods."
            )
        else:
            desc = (
                f"{desc} Module 4 applies the scenario as a delta to current forecasts: "
                "Forecast Return + (Scenario Stress Return - selected-period historical return)."
            )
    rows = _shock_table_initial_rows(baseline_display, shocked)
    meta_children = [
        html.Div("Scenario", className="meta-label"),
        html.Div(scenario_name, style={"fontSize": "16px", "fontWeight": "600",
                                        "marginBottom": "8px"}),
        html.Div(desc),
    ]
    # Build multi-year crisis path for m4-path-store
    try:
        asset_path = st.build_crisis_path(scenario_name, _returns_df, cma_baseline)
        path_store = {str(yr): arr.tolist() for yr, arr in asset_path.items()}
    except Exception:
        path_store = {"1": shocked.tolist()}
    return rows, shocked.tolist(), meta_children, path_store



@app.callback(
    Output("m4-compare-chart", "figure"),
    Output("m4-factor-table",  "children"),
    Output("m4-verdict",       "children"),
    Input("m4-shocked-store",  "data"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",         "data"),
    Input("m1-start-m",        "value"),
    Input("m1-start-y",        "value"),
    Input("m1-end-m",          "value"),
    Input("m1-end-y",          "value"),
    State("m4-scenario",       "value"),
)
def update_m4_outputs(shocked, alloc, cma_store, sm, sy, em, ey, scenario_name):
    if not shocked or not cma_store:
        return go.Figure(), html.Div(), ""
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    shocked_arr  = np.asarray(shocked, dtype=float)
    _, _, _, return_basis, n_months = _scenario_defaults(scenario_name, cma_baseline)
    baseline_display = _forecast_returns_for_basis(cma_baseline, return_basis, n_months)
    if alloc and sum(alloc.values()) > 0:
        w = {t: alloc.get(t, 0) / sum(alloc.values()) for t in tc.TRUST_NAMES}
    else:
        w = {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    fig = shock_compare_figure(baseline_display, shocked_arr, w, return_basis, n_months)
    window_label = None
    if scenario_name and scenario_name in SCENARIO_WINDOWS_LIVE:
        s, e = st.SCENARIO_WINDOWS[scenario_name]
        window_label = f"{s} \u2013 {e}"
    elif scenario_name == "AUD Depreciation Shock":
        sc = _PRECOMPUTED_SCENARIOS.get(scenario_name)
        if sc and sc.window_label:
            window_label = sc.window_label
    sm = sm or _DATE_MIN_M; sy = sy or _DATE_MIN_Y
    em = em or _DATE_MAX_M; ey = ey or _DATE_MAX_Y
    selected_hist = _trust_returns_for_basis(sm, sy, em, ey, return_basis, n_months)
    rows, duration = _factor_breakdown_rows(
        shocked_arr, _returns_df, scenario_name, window_label, selected_hist, return_basis, n_months
    )
    trust_nets = _trust_nets_for_basis(shocked_arr, return_basis, n_months)
    portfolio_stress = sum(w[t] * trust_nets[t] for t in tc.TRUST_NAMES)
    worst_trust = min(trust_nets, key=trust_nets.get)
    period_label = f"{_MONTH_NAMES[int(sm)-1]} {int(sy)} to {_MONTH_NAMES[int(em)-1]} {int(ey)}"
    basis_label = "cumulative event-window" if return_basis == "event_window" else "annualised"
    verdict = (
        f"{scenario_name} stress implies a portfolio stressed return of "
        f"{_fmt_pct(portfolio_stress)} on a {basis_label} basis over a stress window lasting "
        f"{duration}. {worst_trust} is the most exposed trust "
        f"({_fmt_pct(trust_nets[worst_trust])}). Delta return is measured against "
        f"the Module 1 selected-period trust historical return ({period_label})."
    )
    stress_col = "Stress Event Return" if return_basis == "event_window" else "Stress Ann. Return"
    hist_col = "Selected Hist. Event Return" if return_basis == "event_window" else "Selected Hist. Return"
    factor_header = html.Tr([
        html.Th("Trust"),
        html.Th("Stress Window"),
        html.Th("Lasted", style={"textAlign": "right"}),
        html.Th(stress_col, style={"textAlign": "right"}),
        html.Th(hist_col, style={"textAlign": "right"}),
        html.Th("Delta Return", style={"textAlign": "right"}),
        html.Th("Dominant Factor"),
        html.Th("Window Drawdown", style={"textAlign": "right"}),
    ])
    body_rows = []
    for r in rows:
        body_rows.append(html.Tr([
            html.Td(r["trust"], className="trust-cell",
                    style={"--trust-accent": COLORS[r["trust"]]}),
            html.Td(r["stress_window"]),
            html.Td(r["duration"], className="num"),
            html.Td(r["stress_return"], className="num"),
            html.Td(r["selected_hist_return"], className="num"),
            html.Td(r["delta_return"], className="num"),
            html.Td(html.Span(r["dominant_factor"],
                              className=f"factor-tag {_factor_class(r['dominant_factor'])}")),
            html.Td(r["window_drawdown"], className="num"),
        ]))
    factor_table = html.Table(
        [html.Thead(factor_header), html.Tbody(body_rows)],
        className="factor-table",
    )
    if return_basis == "event_window":
        note_text = (
            "Stress Event Return is the cumulative net trust return over the event window. "
            "Selected Hist. Event Return converts the Module 1 historical return to the "
            f"same {n_months or 0}-month horizon. Delta = Stress Event Return minus "
            "Selected Hist. Event Return."
        )
    else:
        note_text = (
            "Stress Ann. Return is the annualised net trust return over the stress window. "
            "Selected Hist. Return is the geometric annual net trust return for the "
            "Module 1 analysis period. Delta = Stress Ann. Return minus Selected Hist. Return."
        )
    note = html.Div(html.Em(note_text),
                    style={"fontSize": "12px", "color": COLORS["muted"], "marginTop": "12px"})
    return fig, html.Div([factor_table, note]), verdict


@app.callback(
    Output("m4-liquidity-check", "children"),
    Input("m4-shocked-store",    "data"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",           "data"),
)
def update_m4_recovery(shocked, alloc, cma_store):
    if not shocked or not cma_store:
        return html.Div()
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    shocked_arr  = np.asarray(shocked, dtype=float)
    total_w = sum(alloc.values()) if alloc else 0
    w = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
         if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    return _m4_liquidity_check_div(shocked_arr, w, cma_baseline)


@app.callback(
    Output("m4-crisis-path-chart", "figure"),
    Output("m4-path-description",  "children"),
    Input("m4-path-store",          "data"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",              "data"),
)
def update_m4_crisis_path(path_store, alloc, cma_store):
    if not path_store or not cma_store:
        return go.Figure(), ""
    cma_baseline = np.asarray(cma_store["returns"], dtype=float)
    total_w = sum(alloc.values()) if alloc else 0
    w = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
         if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})

    # Rebuild int-keyed asset path from stored dict
    asset_path: dict[int, np.ndarray] = {
        int(k): np.asarray(v, dtype=float) for k, v in path_store.items()
    }
    n_years = len(asset_path)
    fig = _build_m4_crisis_path_figure(asset_path, cma_baseline, w, recovery_years=3)

    # Summary description
    year_strs = []
    for yr in sorted(asset_path.keys()):
        nets = st.trust_returns_under_shock(asset_path[yr])
        port = sum(w.get(t, 1/3) * nets[t] for t in tc.TRUST_NAMES)
        year_strs.append(f"Crisis Y{yr}: portfolio net {_fmt_pct(port)}")
    desc = (
        f"This scenario spans {n_years} crisis year(s). "
        + " | ".join(year_strs)
        + ". Recovery years revert to CMA expected returns. "
        "Modules 5 and 6 apply every crisis year as a separate annual return override."
    )
    return fig, desc


# ---------------------------------------------------------------------------
# Callbacks — Module 5
# ---------------------------------------------------------------------------

@app.callback(
    Output("m5-onset-split-STI",       "value"),
    Output("m5-onset-split-MTG",       "value"),
    Output("m5-onset-split-LTG",       "value"),
    Output("m5-predrawdown-balances",  "children"),
    Input("m5-severity",               "value"),
    Input("m5-relief",                 "value"),
    Input("m5-onset",                  "value"),
    Input("m5-fraction",               "value"),
    Input("portfolio-allocation-store","data"),
    Input("cma-store",                 "data"),
)
def auto_populate_onset_split(severity, relief_m, onset, fraction_pct, alloc, cma_store):
    """Auto-populate split inputs using the STI → MTG → LTG sequential redemption
    rule applied to each trust's actual compounded balance before the onset-year
    drawdown. Also shows residual-year balances and contributions. Users can override."""
    if not cma_store or not alloc or onset is None or relief_m is None:
        return 100, 0, 0, ""
    onset      = int(onset)
    fraction   = max(0.0, min(1.0, float(fraction_pct or 50) / 100))
    total_w    = sum(alloc.values())
    weights    = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
                  if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns    = np.asarray(cma_store["returns"], dtype=float)
    relief_aud = float(relief_m) * 1e6
    schedule   = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                     year_4_fraction=fraction, residual_split=(0.5, 0.5))

    # ── Step 1: zero-drought BAU run → pre-drawdown holdings at onset year ──────
    bau = dr.project(3_000_000_000, weights, returns, {}, horizon=onset)
    if onset > len(bau.years):
        return 100, 0, 0, ""

    y_onset   = bau.years[onset - 1]
    pre_val   = y_onset.pre_drawdown_value
    pre_hold  = {t: y_onset.pre_drawdown_weights[t] * pre_val for t in tc.TRUST_NAMES}

    # ── Step 2: sequential rule → onset-year split ──────────────────────────────
    def _sequential_split(holdings, drawdown):
        """Return (trust_relief dict, split dict) under STI → MTG → LTG rule."""
        relief = {t: 0.0 for t in tc.TRUST_NAMES}
        remaining = drawdown
        for trust in tc.TRUST_NAMES:
            if remaining <= 0:
                break
            s    = tc.TRUST_SELL_SPREADS[trust]
            net  = holdings[trust] * (1 - s)
            take = min(net, remaining)
            relief[trust] = take
            remaining -= take
        total = sum(relief.values())
        sp = {t: relief[t] / total for t in tc.TRUST_NAMES} if total > 0 \
             else {"STI": 1.0, "MTG": 0.0, "LTG": 0.0}
        return relief, sp

    onset_drawdown = schedule.get(onset, 0.0)
    onset_relief, onset_split = _sequential_split(pre_hold, onset_drawdown)

    sti_pct = round(onset_split["STI"] * 100, 1)
    mtg_pct = round(onset_split["MTG"] * 100, 1)
    ltg_pct = round(onset_split["LTG"] * 100, 1)

    # ── Step 3: full drought run (with onset split) → residual-year states ───────
    # horizon = onset+2 so we capture both residual drawdown years
    drought_proj = dr.project(3_000_000_000, weights, returns, schedule,
                              horizon=min(onset + 2, 10),
                              drawdown_splits={onset: onset_split})

    # ── Step 4: build display rows for each drought year ────────────────────────
    colors    = {"STI": "#5bc8f5", "MTG": "#7fba00", "LTG": "#f5a623"}
    tag_style = {"partial":     {"color": "#f5a623", "fontSize": "11px"},
                 "fully drawn": {"color": "#e05c5c", "fontSize": "11px"},
                 "untouched":   {"color": "#888",    "fontSize": "11px"}}

    def _tag(relief_amt, holding, trust):
        net = holding * (1 - tc.TRUST_SELL_SPREADS[trust])
        if relief_amt <= 0:
            return "untouched"
        if relief_amt < net - 1.0:
            return "partial"
        return "fully drawn"

    def _year_row(year_num, holdings, drawdown, relief_dict):
        total = sum(holdings.values())
        spans = []
        for trust in tc.TRUST_NAMES:
            tag = _tag(relief_dict[trust], holdings[trust], trust)
            spans.append(html.Span([
                html.Span(f"{trust}  {_fmt_m(holdings[trust])}  "
                          f"({holdings[trust]/total*100:.1f}%)",
                          style={"color": colors[trust]}),
                html.Span(f"  [{tag}]", style=tag_style[tag]),
            ], style={"marginRight": "16px", "display": "inline-block"}))
        return html.Div([
            html.Div(f"Year {year_num}  —  drawdown {_fmt_m(drawdown)}  "
                     f"(total fund {_fmt_m(total)}):",
                     style={"fontWeight": "600", "marginBottom": "3px",
                            "color": COLORS.get("text", "#e0e0e0")}),
            html.Div(spans),
        ], style={"marginBottom": "10px"})

    rows = [_year_row(onset, pre_hold, onset_drawdown, onset_relief)]

    # Residual years: pre-drawdown holdings come from the ending holdings of the
    # prior year in the full drought projection, grown by one year's trust returns.
    for yr_offset in (1, 2):
        yr = onset + yr_offset
        dd = schedule.get(yr, 0.0)
        if dd <= 0:
            continue
        # The YearState for this year holds pre_drawdown_weights + pre_drawdown_value
        if yr > len(drought_proj.years):
            continue
        y_state = drought_proj.years[yr - 1]
        res_val  = y_state.pre_drawdown_value
        res_hold = {t: y_state.pre_drawdown_weights[t] * res_val for t in tc.TRUST_NAMES}
        res_relief, _ = _sequential_split(res_hold, dd)
        rows.append(_year_row(yr, res_hold, dd, res_relief))

    balance_info = html.Div(rows)
    return sti_pct, mtg_pct, ltg_pct, balance_info

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
    Output("m5-rebalance-year", "value"),
    Output("m5-rebalance-year", "min"),
    Output("m5-stress-year",    "min"),
    Input("m5-onset",           "value"),
    State("m5-rebalance-year",  "value"),
    State("m5-stress-year",     "value"),
    prevent_initial_call=True,
)
def sync_m5_year_bounds(onset, cur_reb, cur_stress):
    """Keep rebalance year and stress year valid when onset changes.
    Rebalancing occurs end-of-year (after growth, before drawdown that same year),
    so it is valid from the first drought year (onset) onward — no lower bound beyond onset."""
    if onset is None:
        return dash.no_update, dash.no_update, dash.no_update
    onset       = int(onset)
    min_reb     = onset               # rebalance valid from the first drought year onward
    cur_reb     = int(cur_reb or min_reb)
    new_reb     = cur_reb if cur_reb >= min_reb else min_reb
    new_reb     = min(new_reb, 10)
    min_stress  = new_reb + 1        # stress must come after the rebalance
    return new_reb, min_reb, min_stress


@app.callback(
    Output("m6-rebalance-year", "value"),
    Output("m6-rebalance-year", "min"),
    Input("m5-onset",           "value"),
    State("m6-rebalance-year",  "value"),
    prevent_initial_call=True,
)
def sync_m6_year_bounds(onset, cur_reb):
    """Module 6 inherits the drought onset from M5 — keep its rebalance year valid.
    Rebalancing occurs end-of-year so it is valid from the first drought year (onset) onward."""
    if onset is None:
        return dash.no_update, dash.no_update
    onset   = int(onset)
    min_reb = onset
    cur_reb = int(cur_reb or min_reb)
    new_reb = cur_reb if cur_reb >= min_reb else min_reb
    return min(new_reb, 10), min_reb


@app.callback(
    Output("m5-value-chart",                "figure"),
    Output("m5-composition-chart",          "figure"),
    Output("m5-exec-verdict",               "children"),
    Output("m5-summary-card",               "children"),
    Output("m5-projection-table-container", "children"),
    Output("m5-totals",                     "children"),
    Output("m5-config-summary",             "children"),
    Output("m5-onset-split-summary",        "children"),
    Output("m5-branch-chart",               "figure"),
    Output("m5-rebalance-constraint",       "children"),
    Output("m5-drift-weights",              "children"),
    Output("m5-branch-summary",             "children"),
    Output("m5-return-summary",             "children"),
    Output("m5-reb-compliance",             "children"),
    Input("m5-severity",       "value"),
    Input("m5-relief",         "value"),
    Input("m5-onset",          "value"),
    Input("m5-fraction",       "value"),
    Input("m5-onset-split-STI","value"),
    Input("m5-onset-split-MTG","value"),
    Input("m5-onset-split-LTG","value"),
    Input("portfolio-allocation-store", "data"),
    Input("cma-store",         "data"),
    Input("m5-rebalance-year", "value"),
    Input("m5-reb-STI",        "value"),
    Input("m5-reb-MTG",        "value"),
    Input("m5-reb-LTG",        "value"),
    Input("m5-stress-scenario","value"),
    Input("m5-stress-year",    "value"),
    Input("m5-comp-toggle",    "value"),
)
def update_module_5(severity, relief_m, onset, fraction_pct,
                    split_sti, split_mtg, split_ltg, alloc, cma_store,
                    rebalance_year, reb_sti, reb_mtg, reb_ltg,
                    stress_scenario, stress_year, comp_toggle):
    _empty = (go.Figure(), go.Figure(), "", html.Div(), html.Div(),
              html.Div(), "", "", go.Figure(), html.Div(), "", html.Div(), html.Div(), html.Div())
    if not cma_store or not alloc or relief_m is None or onset is None:
        return _empty
    relief_aud  = float(relief_m) * 1e6
    onset       = int(onset)
    fraction    = max(0.0, min(1.0, float(fraction_pct or 50) / 100))
    onset_split = _onset_split_from_inputs(split_sti, split_mtg, split_ltg)
    schedule    = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                      year_4_fraction=fraction, residual_split=(0.5, 0.5))
    total_w = sum(alloc.values())
    weights = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
               if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns = np.asarray(cma_store["returns"], dtype=float)

    # ── Base projection: BAU + drought, no rebalance ─────────────────────────
    result = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                        drawdown_splits={onset: onset_split})

    value_fig = _projection_value_figure(result, onset)
    comp_fig  = None  # populated after branch projections are computed
    summary      = dr.post_drawdown_summary(result, onset)
    summary_card = _summary_card(summary, result.total_drawdown, result.total_spread_cost)
    base_yrs_breach = sum(1 for y in result.years if not (y.meets_12m and y.meets_3y))
    onset_mix    = "—"
    if summary:
        onset_mix = (f"STI {summary['ending_weights']['STI']*100:.1f}% / "
                     f"MTG {summary['ending_weights']['MTG']*100:.1f}% / "
                     f"LTG {summary['ending_weights']['LTG']*100:.1f}%")
    drought_verdict = (
        f"At the Year {onset} severe-drought drawdown, the portfolio retains "
        f"{_fmt_m(summary['remaining_value']) if summary else '—'} with a post-drawdown mix of "
        f"{onset_mix}. It {'can' if summary and summary['can_sustain_residual'] else 'cannot'} "
        "cover the scheduled residual drawdowns over the next two years without exhausting the Fund. "
        f"The model flags {base_yrs_breach} year(s) where policy liquidity thresholds are breached, "
        "which should be treated as a key trade-off in the executive deck."
    )
    config_text = (
        f"Severity: {severity}. Total relief: {_fmt_m(relief_aud)}. "
        f"Onset year {onset}, with {fraction*100:.0f}% of relief in Year {onset} "
        f"and the remaining {(1-fraction)*100:.0f}% split equally across Years "
        f"{onset+1} and {onset+2}. "
    )
    if result.fund_exhausted:
        config_text += f"FUND EXHAUSTED in Year {result.exhaustion_year}."
    onset_drawdown = schedule.get(onset, 0.0)
    split_summary = (
        f"Year {onset} drawdown {_fmt_m(onset_drawdown)} is targeted as: "
        f"STI {_fmt_m(onset_drawdown * onset_split['STI'])} "
        f"({onset_split['STI']*100:.1f}%), "
        f"MTG {_fmt_m(onset_drawdown * onset_split['MTG'])} "
        f"({onset_split['MTG']*100:.1f}%), "
        f"LTG {_fmt_m(onset_drawdown * onset_split['LTG'])} "
        f"({onset_split['LTG']*100:.1f}%)."
    )

    # ── Rebalancing: constraint checker + drift weights ───────────────────────
    rebalance_year = int(rebalance_year or (onset + 2))
    raw_reb = {"STI": float(reb_sti or 20), "MTG": float(reb_mtg or 30), "LTG": float(reb_ltg or 50)}
    reb_total = sum(raw_reb.values())
    reb_w = {t: raw_reb[t] / reb_total for t in tc.TRUST_NAMES} if reb_total > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}

    meets_12m_new = reb_w["STI"] >= 0.10
    meets_3y_new  = (reb_w["STI"] + reb_w["MTG"]) >= 0.25

    def _pill_span(ok: bool, label: str) -> html.Span:
        cls = "pill pill-pass" if ok else "pill pill-fail"
        return html.Span([html.Span(label, style={"marginRight": "4px"}),
                          html.Span("PASS" if ok else "FAIL", className=cls)],
                         style={"marginRight": "12px"})

    constraint_div = html.Div([
        html.Span("New allocation liquidity check — ",
                  style={"color": COLORS["muted"], "marginRight": "6px"}),
        _pill_span(meets_12m_new, "12m (STI ≥ 10%)"),
        _pill_span(meets_3y_new,  "3y (STI+MTG ≥ 25%)"),
        html.Span(
            f"Normalised: STI {reb_w['STI']*100:.1f}% / MTG {reb_w['MTG']*100:.1f}% / LTG {reb_w['LTG']*100:.1f}%",
            style={"color": COLORS["muted"], "marginLeft": "8px", "fontSize": "12px"}),
    ])

    # Drift weights at rebalance year (from base projection)
    if rebalance_year <= len(result.years):
        drift_y = result.years[rebalance_year - 1]
        dw = drift_y.ending_weights
        drift_text = (
            f"Drifted mix at Y{rebalance_year} (no rebalance): "
            f"STI {dw['STI']*100:.1f}% / MTG {dw['MTG']*100:.1f}% / LTG {dw['LTG']*100:.1f}%  "
            f"→  Shift: STI {(reb_w['STI'] - dw['STI'])*100:+.1f} pp, "
            f"MTG {(reb_w['MTG'] - dw['MTG'])*100:+.1f} pp, "
            f"LTG {(reb_w['LTG'] - dw['LTG'])*100:+.1f} pp"
        )
    else:
        drift_text = ""

    # ── Branch projections ────────────────────────────────────────────────────
    reb_schedule = {rebalance_year: reb_w}

    # Branch (a): BAU + drought + rebalance, no stress
    bau_branch = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                            drawdown_splits={onset: onset_split},
                            rebalance_schedule=reb_schedule)

    # Branch (b): same + multi-year stress shock starting at stress_year
    stress_year    = int(stress_year or 9)
    stress_result  = None
    stress_overrides: dict = {}
    if stress_scenario:
        try:
            # Build the full year-by-year crisis path and map to simulation years
            trust_net_path = _scenario_trust_net_path(stress_scenario, returns)
            stress_overrides = {
                stress_year + yr_offset - 1: nets
                for yr_offset, nets in trust_net_path.items()
                if 1 <= stress_year + yr_offset - 1 <= 10
            }
            stress_result = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                                       drawdown_splits={onset: onset_split},
                                       rebalance_schedule=reb_schedule,
                                       trust_return_overrides=stress_overrides)
        except Exception:
            stress_result  = None
            stress_overrides = {}

    # Composition chart + year-by-year table: toggle selects BAU or stress branch
    comp_source = (stress_result if (comp_toggle == "stress" and stress_result is not None)
                   else bau_branch)
    comp_fig   = _trust_composition_figure(comp_source)
    proj_table = _projection_summary_table(comp_source)
    total_rebal_cost = sum(y.rebalance_cost for y in comp_source.years)
    yrs_breach  = sum(1 for y in comp_source.years if not (y.meets_12m and y.meets_3y))
    totals = html.Div([html.Div([
        html.Span("Final value: ",         style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(comp_source.final_value),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total drawdown: ",      style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(comp_source.total_drawdown),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total spread cost: ",   style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(comp_source.total_spread_cost),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Rebalance cost: ",      style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(total_rebal_cost),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600",
                         "color": COLORS["accent"], "marginRight": "24px"}),
        html.Span("Liquidity breaches: ",  style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(f"{yrs_breach}",
                  style={"fontFamily": MONO_STACK, "fontWeight": "600",
                         "color": COLORS["fail"] if yrs_breach > 0 else COLORS["pass"]}),
    ])])

    branch_fig = _branching_value_figure(
        bau_branch, stress_result, onset, rebalance_year,
        stress_year if stress_result else None,
        stress_scenario,
    )

    # Branch summary: end-of-horizon value comparison
    def _branch_card(label: str, res: dr.ProjectionResult, color: str) -> html.Div:
        reb_cost = sum(y.rebalance_cost for y in res.years)
        return html.Div([
            html.Div(label, style={"fontWeight": "600", "color": color,
                                   "marginBottom": "4px", "fontSize": "13px"}),
            html.Div(_fmt_m(res.final_value),
                     style={"fontFamily": MONO_STACK, "fontSize": "20px",
                            "fontWeight": "700", "color": COLORS["ink"]}),
            html.Div("Year-10 portfolio value",
                     style={"fontSize": "11px", "color": COLORS["muted"]}),
            html.Div([
                html.Span("Rebalance cost: ", style={"color": COLORS["muted"]}),
                html.Span(_fmt_m(reb_cost),
                          style={"fontFamily": MONO_STACK, "color": COLORS["accent"]}),
            ], style={"fontSize": "12px", "marginTop": "6px"}),
        ], style={"flex": "1", "padding": "14px 20px", "borderRadius": "6px",
                  "border": f"1px solid {COLORS['border']}",
                  "backgroundColor": COLORS["panel"], "marginRight": "12px"})

    branch_summary = html.Div([
        _branch_card("Branch (a) — BAU", bau_branch, COLORS["accent"]),
        _branch_card(f"Branch (b) — {stress_scenario} Y{stress_year}",
                     stress_result, "#C07A2A") if stress_result else html.Div(),
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "0"})

    # Board policy compliance for the rebalanced allocation
    cpi = float(cma_store.get("cpi", 0.025))
    returns_arr, vols_arr, corr_arr, _ = _store_to_arrays(cma_store)
    cov_arr = tc.cma_to_covariance(vols_arr, corr_arr)
    reb_metrics = {
        "weights": reb_w,
        "return":  tc.portfolio_net_return(reb_w, returns_arr),
        "vol":     tc.portfolio_volatility(reb_w, cov_arr),
        "liq":     mt.liquidity_coverage(reb_w),
        "target":  cpi + 0.025,
        "cpi":     cpi,
    }
    reb_compliance = _board_compliance_table(reb_metrics)

    # Master fund return summary (follows the toggled comp_source)
    comp_overrides = stress_overrides if (comp_toggle == "stress" and stress_result is not None) else {}
    return_summary = _master_fund_return_table(
        comp_source, returns, comp_overrides, cpi,
        drought_schedule=schedule,
        rebalance_year=rebalance_year,
        new_alloc=reb_w,
        stress_scenario=stress_scenario if (comp_toggle == "stress" and stress_result is not None) else None,
        stress_year=stress_year,
    )

    return (value_fig, comp_fig, drought_verdict, summary_card, proj_table,
            totals, config_text, split_summary,
            branch_fig, constraint_div, drift_text, branch_summary, return_summary,
            reb_compliance)


# ---------------------------------------------------------------------------
# Callbacks — Module 5b (Monte Carlo)
# ---------------------------------------------------------------------------

def _mc_fan_figure(mc: dr.MonteCarloResult, initial_value: float) -> go.Figure:
    M = 1_000_000
    bands = {k: v / M for k, v in mc.percentile_bands().items()}
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
        hovertemplate="Year %{x}<br>P50: $%{y:,.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=bands["p5"], mode="lines", name="P5",
        line=dict(color=COLORS["fail"], width=1, dash="dot"),
        hovertemplate="Year %{x}<br>P5: $%{y:,.1f}M<extra></extra>"))
    fig.add_trace(go.Scatter(x=years, y=bands["p95"], mode="lines", name="P95",
        line=dict(color=COLORS["pass"], width=1, dash="dot"),
        hovertemplate="Year %{x}<br>P95: $%{y:,.1f}M<extra></extra>"))
    fig.add_hline(y=initial_value / M, line=dict(color=COLORS["muted"], width=1, dash="dash"),
        annotation_text="Starting value", annotation_position="bottom right",
        annotation_font=dict(size=10, color=COLORS["muted"]))
    fig.update_layout(height=400, margin=dict(l=70, r=20, t=30, b=40),
        plot_bgcolor=COLORS["panel"], paper_bgcolor=COLORS["panel"],
        font=dict(family=FONT_STACK, color=COLORS["ink"], size=12),
        xaxis=dict(title=dict(text="Year", font=dict(size=11, color=COLORS["muted"])),
                   tick0=0, dtick=1, gridcolor=COLORS["border"], tickfont=dict(size=11)),
        yaxis=dict(title=dict(text="Portfolio value ($M)",
                              font=dict(size=11, color=COLORS["muted"])),
                   gridcolor=COLORS["border"], tickformat="$,.0f", tickfont=dict(size=11)),
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
                  html.Div(f"{_fmt_m(bands['p5'][-1])} \u2013 {_fmt_m(bands['p95'][-1])}",
                           className="val", style={"fontSize": "13px"})], className="summary-item"),
        html.Div([html.Div("Median final value", className="lbl"),
                  html.Div(_fmt_m(bands["p50"][-1]), className="val")], className="summary-item"),
        html.Div([html.Div("Mean total drawdown over 10y", className="lbl"),
                  html.Div(_fmt_m(float(mc.total_drawdowns.mean())), className="val")],
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
                html.Div(_fmt_m(baseline.final_value), className="val",
                         style={"fontSize": "20px"}),
                html.Div("Final value (Y10)",
                         style={"fontSize": "11px", "color": COLORS["muted"], "marginBottom": "10px"}),
                html.Div([
                    html.Div(["Total spread cost: ", html.Strong(_fmt_m(baseline.total_spread_cost))]),
                    html.Div(["Years with liquidity breach: ", html.Strong(str(base_breach))]),
                    html.Div(["Fund exhausted: ", html.Strong("Yes" if baseline.fund_exhausted else "No")]),
                ], style={"fontSize": "13px", "lineHeight": "1.6"}),
            ], className="summary-card", style={"borderLeft": f"3px solid {COLORS['accent']}"}),
            html.Div([
                html.Div(f"Combined ({scenario_name}, Y{shock_year} shock)", className="lbl",
                         style={"color": COLORS["fail"], "fontWeight": "600"}),
                html.Div(_fmt_m(stressed.final_value), className="val",
                         style={"fontSize": "20px"}),
                html.Div("Final value (Y10)",
                         style={"fontSize": "11px", "color": COLORS["muted"], "marginBottom": "10px"}),
                html.Div([
                    html.Div(["Total spread cost: ", html.Strong(_fmt_m(stressed.total_spread_cost))]),
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
                              html.Span(_fmt_m(delta_final), style={"fontFamily": MONO_STACK,
                                  "fontWeight": "600",
                                  "color": COLORS["pass"] if delta_final >= 0 else COLORS["fail"]}),
                              html.Span(f" ({pct_impact*100:+.2f}%)",
                                        style={"fontFamily": MONO_STACK, "color": COLORS["muted"],
                                               "marginLeft": "4px"}),
                          ])], className="summary-item"),
                html.Div([html.Div(f"Y{shock_year} ending value (drought only)", className="lbl"),
                          html.Div(_fmt_m(by.ending_value) if by else "—",
                                   className="val", style={"fontSize": "14px"})],
                         className="summary-item"),
                html.Div([html.Div(f"Y{shock_year} ending value (combined)", className="lbl"),
                          html.Div(_fmt_m(sy.ending_value) if sy else "—",
                                   className="val", style={"fontSize": "14px"})],
                         className="summary-item"),
            ], className="summary-grid"),
        ], style={"marginTop": "16px"}),
    ])


@app.callback(
    Output("m6-value-chart",               "figure"),
    Output("m6-summary-grid",              "children"),
    Output("m6-config-summary",            "children"),
    Output("m6-drawdown-profile",         "children"),
    Output("m6-forward-chart",             "figure"),
    Output("m6-projection-table-container","children"),
    Output("m6-totals",                    "children"),
    Output("m6-return-summary",            "children"),
    Output("m6-rebalance-constraint",      "children"),
    Output("m6-drift-weights",             "children"),
    Output("m6-reb-compliance",            "children"),
    Input("m6-scenario",                   "value"),
    Input("m6-shock-year",                 "value"),
    Input("m5-severity",                   "value"),
    Input("m5-relief",                     "value"),
    Input("m5-onset",                      "value"),
    Input("m5-fraction",                   "value"),
    Input("m5-onset-split-STI",            "value"),
    Input("m5-onset-split-MTG",            "value"),
    Input("m5-onset-split-LTG",            "value"),
    Input("portfolio-allocation-store",    "data"),
    Input("cma-store",                     "data"),
    Input("m6-rebalance-year",             "value"),
    Input("m6-reb-STI",                    "value"),
    Input("m6-reb-MTG",                    "value"),
    Input("m6-reb-LTG",                    "value"),
)
def update_module_6(scenario_name, shock_year, severity, relief_m, onset,
                    fraction_pct, split_sti, split_mtg, split_ltg, alloc, cma_store,
                    rebalance_year, reb_sti, reb_mtg, reb_ltg):
    _empty = (go.Figure(), html.Div(), "", html.Div(), go.Figure(),
              html.Div(), html.Div(), html.Div(), html.Div(), "", html.Div())
    if not cma_store or not alloc or relief_m is None or onset is None:
        return _empty

    relief_aud   = float(relief_m) * 1e6
    onset        = int(onset)
    shock_year   = int(shock_year or onset)
    fraction     = max(0.0, min(1.0, float(fraction_pct or 50) / 100))
    onset_split  = _onset_split_from_inputs(split_sti, split_mtg, split_ltg)
    schedule     = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                       year_4_fraction=fraction, residual_split=(0.5, 0.5))
    total_w      = sum(alloc.values())
    weights      = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
                    if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    returns      = np.asarray(cma_store["returns"], dtype=float)
    drought_years = list(schedule.keys())

    # ── Shock overrides (multi-year crisis path from shock_year) ─────────────
    trust_net_path = _scenario_trust_net_path(scenario_name, returns)
    m6_overrides   = {
        shock_year + yr_offset - 1: nets
        for yr_offset, nets in trust_net_path.items()
        if 1 <= shock_year + yr_offset - 1 <= 10
    }

    # ── Three projections ─────────────────────────────────────────────────────
    # 1. Drought-only BAU (reference)
    baseline = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                          drawdown_splits={onset: onset_split})
    # 2. Combined crash + drought, no rebalance
    stressed = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                          trust_return_overrides=m6_overrides,
                          drawdown_splits={onset: onset_split})
    # 3. Combined crash + drought → rebalance → BAU recovery
    rebalance_year = int(rebalance_year or (onset + 2))
    raw_reb  = {"STI": float(reb_sti or 15), "MTG": float(reb_mtg or 35), "LTG": float(reb_ltg or 50)}
    reb_total = sum(raw_reb.values())
    reb_w    = ({t: raw_reb[t] / reb_total for t in tc.TRUST_NAMES}
                if reb_total > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})
    rebalanced = dr.project(3_000_000_000, weights, returns, schedule, horizon=10,
                            trust_return_overrides=m6_overrides,
                            drawdown_splits={onset: onset_split},
                            rebalance_schedule={rebalance_year: reb_w})

    # ── Existing panels: combined trajectory + joint impact summary ───────────
    combined_fig = _combined_value_figure(baseline, stressed, shock_year, drought_years)
    summary_grid = _module_6_summary(baseline, stressed, shock_year, scenario_name)

    n_crisis = len(trust_net_path)
    config = (f"Market shock: {scenario_name} starting Year {shock_year} "
              f"({n_crisis} crisis year(s) applied). "
              f"Drought: {severity} severity, total relief {_fmt_m(relief_aud)}, "
              f"onset Year {onset}. ")
    if shock_year == onset:
        config += "Shock and drought onset coincide (simultaneous event)."
    elif shock_year < onset:
        config += f"Shock precedes drought onset by {onset - shock_year} year(s)."
    else:
        config += f"Shock follows drought onset by {shock_year - onset} year(s)."

    # ── Recovery chart ────────────────────────────────────────────────────────
    forward_fig = _m6_forward_figure(
        baseline, stressed, rebalanced, shock_year, drought_years, rebalance_year)

    # ── Year-by-year table (recovery path = rebalanced) ───────────────────────
    proj_table       = _projection_summary_table(rebalanced, table_id="m6-projection-table")
    total_rebal_cost = sum(y.rebalance_cost for y in rebalanced.years)
    yrs_breach       = sum(1 for y in rebalanced.years if not (y.meets_12m and y.meets_3y))
    totals = html.Div([html.Div([
        html.Span("Final value: ",        style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(rebalanced.final_value),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total drawdown: ",     style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(rebalanced.total_drawdown),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Total spread cost: ",  style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(rebalanced.total_spread_cost),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600", "marginRight": "24px"}),
        html.Span("Rebalance cost: ",     style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(_fmt_m(total_rebal_cost),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600",
                         "color": COLORS["accent"], "marginRight": "24px"}),
        html.Span("Liquidity breaches: ", style={"color": COLORS["muted"], "marginRight": "6px"}),
        html.Span(str(yrs_breach),
                  style={"fontFamily": MONO_STACK, "fontWeight": "600",
                         "color": COLORS["fail"] if yrs_breach > 0 else COLORS["pass"]}),
    ])])

    # ── Master fund return summary (recovery path) ─────────────────────────────
    cpi = float(cma_store.get("cpi", 0.025))
    return_summary = _master_fund_return_table(
        rebalanced, returns, m6_overrides, cpi,
        drought_schedule=schedule,
        rebalance_year=rebalance_year,
        new_alloc=reb_w,
        stress_scenario=scenario_name,
        stress_year=shock_year,
    )

    # ── Rebalancing constraint checker + drift weights ─────────────────────────
    meets_12m_new = reb_w["STI"] >= 0.10
    meets_3y_new  = (reb_w["STI"] + reb_w["MTG"]) >= 0.25

    def _pill(ok: bool, label: str) -> html.Span:
        cls = "pill pill-pass" if ok else "pill pill-fail"
        return html.Span([html.Span(label, style={"marginRight": "4px"}),
                          html.Span("PASS" if ok else "FAIL", className=cls)],
                         style={"marginRight": "12px"})

    constraint_div = html.Div([
        html.Span("New allocation liquidity check — ",
                  style={"color": COLORS["muted"], "marginRight": "6px"}),
        _pill(meets_12m_new, "12m (STI ≥ 10%)"),
        _pill(meets_3y_new,  "3y (STI+MTG ≥ 25%)"),
        html.Span(
            f"Normalised: STI {reb_w['STI']*100:.1f}% / MTG {reb_w['MTG']*100:.1f}% / "
            f"LTG {reb_w['LTG']*100:.1f}%",
            style={"color": COLORS["muted"], "marginLeft": "8px", "fontSize": "12px"}),
    ])

    # Drift weights at rebalance year — from the stressed (no-rebalance) projection
    if rebalance_year <= len(stressed.years):
        dw = stressed.years[rebalance_year - 1].ending_weights
        drift_text = (
            f"Drifted mix at Y{rebalance_year} (combined path, no rebalance): "
            f"STI {dw['STI']*100:.1f}% / MTG {dw['MTG']*100:.1f}% / LTG {dw['LTG']*100:.1f}%  "
            f"→  Shift: STI {(reb_w['STI'] - dw['STI'])*100:+.1f} pp, "
            f"MTG {(reb_w['MTG'] - dw['MTG'])*100:+.1f} pp, "
            f"LTG {(reb_w['LTG'] - dw['LTG'])*100:+.1f} pp"
        )
    else:
        drift_text = ""

    # ── Board policy compliance for the rebalanced allocation ─────────────────
    returns_arr, vols_arr, corr_arr, _ = _store_to_arrays(cma_store)
    cov_arr = tc.cma_to_covariance(vols_arr, corr_arr)
    reb_metrics = {
        "weights": reb_w,
        "return":  tc.portfolio_net_return(reb_w, returns_arr),
        "vol":     tc.portfolio_volatility(reb_w, cov_arr),
        "liq":     mt.liquidity_coverage(reb_w),
        "target":  cpi + 0.025,
        "cpi":     cpi,
    }
    reb_compliance = _board_compliance_table(reb_metrics)

    # ── Drawdown profile: actual trust holdings + redemptions under stressed path ─
    drawdown_profile = _m6_drawdown_profile(stressed, schedule)

    return (combined_fig, summary_grid, config, drawdown_profile, forward_fig,
            proj_table, totals, return_summary,
            constraint_div, drift_text, reb_compliance)


# ---------------------------------------------------------------------------
# Callbacks — Module 7 (Executive Summary)
# ---------------------------------------------------------------------------

@app.callback(
    Output("m7-content", "children"),
    Input("cma-store",                 "data"),
    Input("portfolio-allocation-store","data"),
    # Module 5 inputs
    Input("m5-severity",               "value"),
    Input("m5-relief",                 "value"),
    Input("m5-onset",                  "value"),
    Input("m5-fraction",               "value"),
    Input("m5-onset-split-STI",        "value"),
    Input("m5-onset-split-MTG",        "value"),
    Input("m5-onset-split-LTG",        "value"),
    Input("m5-rebalance-year",         "value"),
    Input("m5-reb-STI",                "value"),
    Input("m5-reb-MTG",                "value"),
    Input("m5-reb-LTG",                "value"),
    Input("m5-stress-scenario",        "value"),
    Input("m5-stress-year",            "value"),
    # Module 6 inputs
    Input("m6-scenario",               "value"),
    Input("m6-shock-year",             "value"),
    Input("m6-rebalance-year",         "value"),
    Input("m6-reb-STI",                "value"),
    Input("m6-reb-MTG",                "value"),
    Input("m6-reb-LTG",                "value"),
)
def update_module_7(
    cma_store, alloc,
    m5_severity, m5_relief, m5_onset, m5_fraction,
    m5_split_sti, m5_split_mtg, m5_split_ltg,
    m5_reb_year, m5_reb_sti, m5_reb_mtg, m5_reb_ltg,
    m5_stress_scenario, m5_stress_year,
    m6_scenario, m6_shock_year,
    m6_reb_year, m6_reb_sti, m6_reb_mtg, m6_reb_ltg,
):
    if not cma_store or not alloc or m5_relief is None or m5_onset is None:
        return html.Div("Configure Modules 1–6 first, then return here.",
                        style={"color": COLORS["muted"], "padding": "20px"})

    # ── Parse all inputs ─────────────────────────────────────────────────────
    returns      = np.asarray(cma_store["returns"], dtype=float)
    cpi          = float(cma_store.get("cpi", 0.025))
    returns_arr, vols_arr, corr_arr, _ = _store_to_arrays(cma_store)
    cov_arr      = tc.cma_to_covariance(vols_arr, corr_arr)

    total_w      = sum(alloc.values())
    init_w       = ({t: alloc.get(t, 0) / total_w for t in tc.TRUST_NAMES}
                    if total_w > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3})

    onset        = int(m5_onset)
    fraction     = max(0.0, min(1.0, float(m5_fraction or 50) / 100))
    relief_aud   = float(m5_relief) * 1e6
    onset_split  = _onset_split_from_inputs(m5_split_sti, m5_split_mtg, m5_split_ltg)
    schedule     = dr.build_drought_schedule(onset_year=onset, total_relief=relief_aud,
                       year_4_fraction=fraction, residual_split=(0.5, 0.5))

    # M5 rebalancing
    m5_reb_year  = int(m5_reb_year or (onset + 2))
    raw5         = {"STI": float(m5_reb_sti or 20), "MTG": float(m5_reb_mtg or 30),
                    "LTG": float(m5_reb_ltg or 50)}
    r5t          = sum(raw5.values())
    m5_reb_w     = {t: raw5[t] / r5t for t in tc.TRUST_NAMES} if r5t > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}

    # M5 stress branch
    m5_stress_year = int(m5_stress_year or 9)
    m5_trust_path  = _scenario_trust_net_path(m5_stress_scenario, returns) if m5_stress_scenario else {}
    m5_overrides   = {m5_stress_year + off - 1: nets
                      for off, nets in m5_trust_path.items()
                      if 1 <= m5_stress_year + off - 1 <= 10}

    # M6 shock
    m6_shock_year  = int(m6_shock_year or onset)
    m6_reb_year    = int(m6_reb_year or (onset + 2))
    raw6           = {"STI": float(m6_reb_sti or 15), "MTG": float(m6_reb_mtg or 35),
                      "LTG": float(m6_reb_ltg or 50)}
    r6t            = sum(raw6.values())
    m6_reb_w       = {t: raw6[t] / r6t for t in tc.TRUST_NAMES} if r6t > 0 else {"STI": 1/3, "MTG": 1/3, "LTG": 1/3}
    m6_trust_path  = _scenario_trust_net_path(m6_scenario, returns)
    m6_overrides   = {m6_shock_year + off - 1: nets
                      for off, nets in m6_trust_path.items()
                      if 1 <= m6_shock_year + off - 1 <= 10}

    # ── Run all projections ──────────────────────────────────────────────────
    # M5 projections
    m5_base       = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               drawdown_splits={onset: onset_split})
    m5_bau_branch = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               drawdown_splits={onset: onset_split},
                               rebalance_schedule={m5_reb_year: m5_reb_w})
    m5_stress_res = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               drawdown_splits={onset: onset_split},
                               rebalance_schedule={m5_reb_year: m5_reb_w},
                               trust_return_overrides=m5_overrides) if m5_overrides else None

    # M6 projections
    m6_baseline   = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               drawdown_splits={onset: onset_split})
    m6_stressed   = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               trust_return_overrides=m6_overrides,
                               drawdown_splits={onset: onset_split})
    m6_rebalanced = dr.project(3e9, init_w, returns, schedule, horizon=10,
                               trust_return_overrides=m6_overrides,
                               drawdown_splits={onset: onset_split},
                               rebalance_schedule={m6_reb_year: m6_reb_w})

    # ── Derived metrics ──────────────────────────────────────────────────────
    def _geom_net(res):
        f = 1.0
        for y in res.years:
            w  = y.ending_weights
            nr = {t: tc.trust_net_return(t, returns) for t in tc.TRUST_NAMES}
            f *= (1 + sum(w[t] * nr[t] for t in tc.TRUST_NAMES))
        return f ** (1 / len(res.years)) - 1 if res.years else 0.0

    def _breaches(res):
        return sum(1 for y in res.years if not (y.meets_12m and y.meets_3y))

    def _reb_cost(res):
        return sum(y.rebalance_cost for y in res.years)

    target = cpi + 0.025

    # ── Shared style helpers ─────────────────────────────────────────────────
    def _pass_fail(ok: bool) -> html.Span:
        return html.Span("PASS" if ok else "FAIL",
                         className="pill pill-pass" if ok else "pill pill-fail")

    def _val(v, style=None):
        return html.Span(v, style={"fontFamily": MONO_STACK, "fontWeight": "600",
                                   **(style or {})})

    def _row(label, value):
        return html.Div([
            html.Span(label, style={"color": COLORS["muted"], "minWidth": "260px",
                                    "display": "inline-block", "fontSize": "13px"}),
            value if isinstance(value, html.Span) else html.Span(
                value, style={"fontFamily": MONO_STACK, "fontSize": "13px"}),
        ], style={"marginBottom": "5px"})

    def _section_head(title, color=None):
        return html.H3(title, style={
            "fontSize": "13px", "fontWeight": "700", "textTransform": "uppercase",
            "letterSpacing": "0.06em", "color": color or COLORS["muted"],
            "marginTop": "18px", "marginBottom": "8px",
            "borderBottom": f"1px solid {COLORS['border']}", "paddingBottom": "4px"})

    def _trust_alloc_str(w):
        return (f"STI {w['STI']*100:.1f}%  /  MTG {w['MTG']*100:.1f}%  /  LTG {w['LTG']*100:.1f}%")

    # ── SECTION 1: Starting position ────────────────────────────────────────
    cma_trust_nets = {t: tc.trust_net_return(t, returns) for t in tc.TRUST_NAMES}
    init_net = tc.portfolio_net_return(init_w, returns)
    init_vol = tc.portfolio_volatility(init_w, cov_arr)

    sec1 = html.Div([
        _section_head("Fund starting position"),
        _row("Initial fund value",        _val("$3,000M (AUD)")),
        _row("Initial allocation",        _val(_trust_alloc_str(init_w))),
        _row("Portfolio net return (CMA)",_val(f"{init_net*100:.2f}% p.a.")),
        _row("Portfolio volatility (CMA)",_val(f"{init_vol*100:.2f}% p.a.")),
        _row("CPI assumption",            _val(f"{cpi*100:.2f}%")),
        _row("Return target (CPI + 2.5%)",_val(f"{target*100:.2f}%")),
        html.Div([
            html.Span("CMA trust net returns — ",
                      style={"color": COLORS["muted"], "fontSize": "13px"}),
            *[html.Span(f"{t}: {cma_trust_nets[t]*100:.2f}%",
                        style={"marginRight": "16px", "fontFamily": MONO_STACK,
                               "fontSize": "13px", "color": COLORS.get(t, COLORS["ink"])})
              for t in tc.TRUST_NAMES],
        ], style={"marginTop": "4px"}),
    ], className="panel")

    # ── SECTION 2: Drought configuration ────────────────────────────────────
    dd_rows = []
    for yr, amt in sorted(schedule.items()):
        label = "Onset year" if yr == onset else f"Year {yr} (residual)"
        dd_rows.append(_row(f"  {label} drawdown", _val(_fmt_m(amt))))

    sec2 = html.Div([
        _section_head("Drought scenario configuration (shared by both scenarios)"),
        _row("Severity",               _val(m5_severity or "—")),
        _row("Total relief",           _val(_fmt_m(relief_aud))),
        _row("Drought onset year",     _val(f"Year {onset}")),
        _row(f"Onset fraction ({fraction*100:.0f}% in Year {onset})",
             _val(f"{_fmt_m(schedule.get(onset, 0))} in Year {onset}")),
        *dd_rows,
        _row("Onset drawdown split",   _val(
            f"STI {_fmt_m(onset_split['STI'] * schedule.get(onset, 0))} ({onset_split['STI']*100:.1f}%)  /  "
            f"MTG {_fmt_m(onset_split['MTG'] * schedule.get(onset, 0))} ({onset_split['MTG']*100:.1f}%)  /  "
            f"LTG {_fmt_m(onset_split['LTG'] * schedule.get(onset, 0))} ({onset_split['LTG']*100:.1f}%)"
            f"  (STI → MTG → LTG sequential, auto-computed)")),
    ], className="panel")

    # ── SECTION 3: Scenario 1 — Drought First (Module 5) ────────────────────
    m5_bau_net    = _geom_net(m5_bau_branch)
    m5_stress_net = _geom_net(m5_stress_res) if m5_stress_res else None
    m5_reb_metrics = {
        "weights": m5_reb_w, "return": tc.portfolio_net_return(m5_reb_w, returns_arr),
        "vol": tc.portfolio_volatility(m5_reb_w, cov_arr),
        "liq": mt.liquidity_coverage(m5_reb_w), "target": target, "cpi": cpi,
    }
    m5_liq = mt.liquidity_coverage(m5_reb_w)

    # Drawdown impact from m5_base: ending values at each drought year
    drought_impact_rows = []
    for yr in sorted(schedule.keys()):
        if yr <= len(m5_base.years):
            y = m5_base.years[yr - 1]
            drought_impact_rows.append(_row(
                f"  Year {yr} — drawdown {_fmt_m(y.drawdown)}",
                _val(f"Ending value {_fmt_m(y.ending_value)}  "
                     f"(STI {y.ending_weights['STI']*100:.1f}% / "
                     f"MTG {y.ending_weights['MTG']*100:.1f}% / "
                     f"LTG {y.ending_weights['LTG']*100:.1f}%)")))

    stress_branch_rows = []
    if m5_stress_res:
        stress_branch_rows = [
            _section_head("Branch (b) — Late stress test", color="#C07A2A"),
            _row("Scenario applied",     _val(m5_stress_scenario or "—")),
            _row("Applied at year",       _val(f"Year {m5_stress_year}")),
            _row("Crisis years applied",  _val(str(len(m5_trust_path)))),
            _row("Y10 value",             _val(_fmt_m(m5_stress_res.final_value))),
            _row("Delta vs BAU branch",   _val(
                _fmt_m(m5_stress_res.final_value - m5_bau_branch.final_value),
                {"color": COLORS["fail"] if m5_stress_res.final_value < m5_bau_branch.final_value else COLORS["pass"]})),
            _row("Liquidity breaches",    _val(str(_breaches(m5_stress_res)))),
        ]

    sec3 = html.Div([
        _section_head("Scenario 1 — Drought First, then rebalance (Module 5)", color=COLORS["accent"]),
        html.Div([
            html.Div([
                _section_head("Drought impact (no rebalance baseline)"),
                *drought_impact_rows,
                _row("Y10 value (no rebalance)", _val(_fmt_m(m5_base.final_value))),
                _row("Liquidity breaches",        _val(str(_breaches(m5_base)))),
            ]),
            html.Hr(style={"borderColor": COLORS["border"], "margin": "14px 0"}),
            _section_head("Post-drought rebalancing"),
            _row("Rebalance year",       _val(f"Year {m5_reb_year}  (year-end: after growth, before drawdown)")),
            _row("New allocation",        _val(_trust_alloc_str(m5_reb_w))),
            _row("Rebalance cost",        _val(_fmt_m(_reb_cost(m5_bau_branch)))),
            _row("New STI ≥ 10%",         _pass_fail(m5_liq["meets_12m"])),
            _row("New STI+MTG ≥ 25%",     _pass_fail(m5_liq["meets_3y"])),
            html.Hr(style={"borderColor": COLORS["border"], "margin": "14px 0"}),
            _section_head("Branch (a) — BAU forward", color=COLORS["accent"]),
            _row("Y10 value",             _val(_fmt_m(m5_bau_branch.final_value))),
            _row("10Y avg net return",    _val(f"{m5_bau_net*100:.2f}%")),
            _row("Meets CPI+2.5% target", _pass_fail(m5_bau_net >= target - 1e-9)),
            _row("Liquidity breaches",    _val(str(_breaches(m5_bau_branch)))),
            *stress_branch_rows,
        ]),
    ], className="panel")

    # ── SECTION 4: Scenario 2 — Combined Stress + Drought (Module 6) ─────────
    # How the stress was applied: show crisis-year trust returns vs CMA
    cma_trust_nets_dict = {t: tc.trust_net_return(t, returns) for t in tc.TRUST_NAMES}
    stress_overlay_rows = []
    for off, nets in sorted(m6_trust_path.items()):
        sim_yr = m6_shock_year + off - 1
        if 1 <= sim_yr <= 10:
            stress_overlay_rows.append(html.Div([
                html.Div(f"Crisis year {off}  →  simulation Year {sim_yr}:",
                         style={"fontWeight": "600", "fontSize": "12.5px",
                                "color": COLORS.get("text", "#e0e0e0"),
                                "marginBottom": "2px"}),
                html.Div([
                    *[html.Span([
                        html.Span(f"{t}  ", style={"color": COLORS.get(t, COLORS["ink"])}),
                        html.Span(f"{nets[t]*100:.1f}%",
                                  style={"fontFamily": MONO_STACK,
                                         "color": COLORS["fail"] if nets[t] < cma_trust_nets_dict[t] else COLORS["pass"]}),
                        html.Span(f"  (CMA: {cma_trust_nets_dict[t]*100:.1f}%)",
                                  style={"fontSize": "11px", "color": COLORS["muted"],
                                         "marginRight": "18px"}),
                    ]) for t in tc.TRUST_NAMES],
                ]),
            ], style={"marginBottom": "8px", "padding": "6px 10px",
                      "borderRadius": "4px",
                      "background": "rgba(224,92,92,0.07)",
                      "border": "1px solid rgba(224,92,92,0.2)"}))

    m6_liq        = mt.liquidity_coverage(m6_reb_w)
    m6_reb_net    = _geom_net(m6_rebalanced)
    m6_no_reb_net = _geom_net(m6_stressed)

    # M6 drawdown impact from stressed projection
    m6_drought_rows = []
    for yr in sorted(schedule.keys()):
        if yr <= len(m6_stressed.years):
            y = m6_stressed.years[yr - 1]
            note = "  ← shock active" if yr in m6_overrides else ""
            m6_drought_rows.append(_row(
                f"  Year {yr} — drawdown {_fmt_m(y.drawdown)}{note}",
                _val(f"Ending value {_fmt_m(y.ending_value)}")))

    sec4 = html.Div([
        _section_head("Scenario 2 — Combined Market Shock + Drought (Module 6)",
                      color=COLORS["fail"]),
        html.Div([
            _section_head("Market shock applied"),
            _row("Scenario",             _val(m6_scenario or "—")),
            _row("Shock applied at",      _val(f"Year {m6_shock_year}")),
            _row("Crisis duration",       _val(f"{len(m6_trust_path)} year(s)")),
            _row("Simultaneous with drought onset",
                 _pass_fail(m6_shock_year == onset)),

            _section_head("How the stress was applied as overlay on the drought projection"),
            html.Div(
                "Each crisis year replaces the CMA trust net returns with the stressed returns "
                "below. Drought drawdowns proceed on top of the shocked portfolio value in "
                "the same year(s).",
                style={"fontSize": "12px", "color": COLORS["muted"],
                       "marginBottom": "10px", "lineHeight": "1.5"}),
            *stress_overlay_rows,

            html.Hr(style={"borderColor": COLORS["border"], "margin": "14px 0"}),
            _section_head("Combined impact (no rebalance)"),
            *m6_drought_rows,
            _row("Y10 value (no rebalance)",    _val(_fmt_m(m6_stressed.final_value))),
            _row("Delta vs drought-only",        _val(
                _fmt_m(m6_stressed.final_value - m6_baseline.final_value),
                {"color": COLORS["fail"] if m6_stressed.final_value < m6_baseline.final_value else COLORS["pass"]})),
            _row("Liquidity breaches",           _val(str(_breaches(m6_stressed)))),

            html.Hr(style={"borderColor": COLORS["border"], "margin": "14px 0"}),
            _section_head("Post-event rebalancing → BAU recovery", color="#C07A2A"),
            _row("Rebalance year",        _val(f"Year {m6_reb_year}  (year-end: after growth, before drawdown)")),
            _row("New allocation",         _val(_trust_alloc_str(m6_reb_w))),
            _row("Rebalance cost",         _val(_fmt_m(_reb_cost(m6_rebalanced)))),
            _row("New STI ≥ 10%",          _pass_fail(m6_liq["meets_12m"])),
            _row("New STI+MTG ≥ 25%",      _pass_fail(m6_liq["meets_3y"])),
            _row("Y10 value (recovery)",   _val(_fmt_m(m6_rebalanced.final_value))),
            _row("10Y avg net return",     _val(f"{m6_reb_net*100:.2f}%")),
            _row("Meets CPI+2.5% target",  _pass_fail(m6_reb_net >= target - 1e-9)),
            _row("Liquidity breaches",     _val(str(_breaches(m6_rebalanced)))),
        ]),
    ], className="panel")

    # ── SECTION 5: Comparison table ──────────────────────────────────────────
    def _td(v, color=None):
        s = {"padding": "8px 12px", "fontFamily": MONO_STACK, "fontSize": "13px",
             "borderBottom": f"1px solid {COLORS['border']}",
             "textAlign": "right"}
        if color:
            s["color"] = color
        return html.Td(v, style=s)

    def _th(v):
        return html.Th(v, style={"padding": "8px 12px", "fontSize": "12px",
                                  "fontWeight": "700", "textAlign": "right",
                                  "borderBottom": f"2px solid {COLORS['border']}",
                                  "color": COLORS["muted"]})

    def _thl(v):
        return html.Th(v, style={"padding": "8px 12px", "fontSize": "12px",
                                  "fontWeight": "700", "textAlign": "left",
                                  "borderBottom": f"2px solid {COLORS['border']}",
                                  "color": COLORS["muted"]})

    def _tdl(v):
        return html.Td(v, style={"padding": "8px 12px", "fontSize": "13px",
                                  "borderBottom": f"1px solid {COLORS['border']}",
                                  "textAlign": "left", "color": COLORS["ink"]})

    comparison = html.Table([
        html.Thead(html.Tr([
            _thl("Metric"),
            _th("Drought only\n(no rebalance)"),
            _th("M5 — BAU branch"),
            _th(f"M5 — {m5_stress_scenario or 'Stress'} branch") if m5_stress_res else _th("M5 — Stress branch"),
            _th("M6 — Combined\n(no rebalance)"),
            _th("M6 — Recovery path"),
        ])),
        html.Tbody([
            html.Tr([_tdl("Y10 portfolio value"),
                     _td(_fmt_m(m5_base.final_value)),
                     _td(_fmt_m(m5_bau_branch.final_value), color=COLORS["accent"]),
                     _td(_fmt_m(m5_stress_res.final_value) if m5_stress_res else "—", color="#C07A2A"),
                     _td(_fmt_m(m6_stressed.final_value), color=COLORS["fail"]),
                     _td(_fmt_m(m6_rebalanced.final_value), color="#C07A2A")]),
            html.Tr([_tdl("10Y avg net return"),
                     _td(f"{_geom_net(m5_base)*100:.2f}%"),
                     _td(f"{m5_bau_net*100:.2f}%"),
                     _td(f"{m5_stress_net*100:.2f}%" if m5_stress_net is not None else "—"),
                     _td(f"{m6_no_reb_net*100:.2f}%"),
                     _td(f"{m6_reb_net*100:.2f}%")]),
            html.Tr([_tdl(f"Meets CPI+2.5% ({target*100:.1f}%)"),
                     _td("Pass" if _geom_net(m5_base) >= target - 1e-9 else "Fail",
                         color=COLORS["pass"] if _geom_net(m5_base) >= target - 1e-9 else COLORS["fail"]),
                     _td("Pass" if m5_bau_net >= target - 1e-9 else "Fail",
                         color=COLORS["pass"] if m5_bau_net >= target - 1e-9 else COLORS["fail"]),
                     _td(("Pass" if m5_stress_net and m5_stress_net >= target - 1e-9 else "Fail") if m5_stress_res else "—",
                         color=COLORS["pass"] if (m5_stress_net or 0) >= target - 1e-9 else COLORS["fail"]),
                     _td("Pass" if m6_no_reb_net >= target - 1e-9 else "Fail",
                         color=COLORS["pass"] if m6_no_reb_net >= target - 1e-9 else COLORS["fail"]),
                     _td("Pass" if m6_reb_net >= target - 1e-9 else "Fail",
                         color=COLORS["pass"] if m6_reb_net >= target - 1e-9 else COLORS["fail"])]),
            html.Tr([_tdl("Liquidity breaches (years)"),
                     _td(str(_breaches(m5_base))),
                     _td(str(_breaches(m5_bau_branch))),
                     _td(str(_breaches(m5_stress_res)) if m5_stress_res else "—"),
                     _td(str(_breaches(m6_stressed))),
                     _td(str(_breaches(m6_rebalanced)))]),
            html.Tr([_tdl("Total rebalance cost"),
                     _td("—"),
                     _td(_fmt_m(_reb_cost(m5_bau_branch))),
                     _td(_fmt_m(_reb_cost(m5_stress_res)) if m5_stress_res else "—"),
                     _td("—"),
                     _td(_fmt_m(_reb_cost(m6_rebalanced)))]),
            html.Tr([_tdl("Total spread cost (drawdown)"),
                     _td(_fmt_m(m5_base.total_spread_cost)),
                     _td(_fmt_m(m5_bau_branch.total_spread_cost)),
                     _td(_fmt_m(m5_stress_res.total_spread_cost) if m5_stress_res else "—"),
                     _td(_fmt_m(m6_stressed.total_spread_cost)),
                     _td(_fmt_m(m6_rebalanced.total_spread_cost))]),
            html.Tr([_tdl("Fund exhausted"),
                     _td("Yes" if m5_base.fund_exhausted else "No",
                         color=COLORS["fail"] if m5_base.fund_exhausted else COLORS["pass"]),
                     _td("Yes" if m5_bau_branch.fund_exhausted else "No",
                         color=COLORS["fail"] if m5_bau_branch.fund_exhausted else COLORS["pass"]),
                     _td(("Yes" if m5_stress_res.fund_exhausted else "No") if m5_stress_res else "—",
                         color=COLORS["fail"] if m5_stress_res and m5_stress_res.fund_exhausted else COLORS["pass"]),
                     _td("Yes" if m6_stressed.fund_exhausted else "No",
                         color=COLORS["fail"] if m6_stressed.fund_exhausted else COLORS["pass"]),
                     _td("Yes" if m6_rebalanced.fund_exhausted else "No",
                         color=COLORS["fail"] if m6_rebalanced.fund_exhausted else COLORS["pass"])]),
        ]),
    ], style={"width": "100%", "borderCollapse": "collapse",
              "fontFamily": FONT_STACK, "color": COLORS["ink"]})

    sec5 = html.Div([
        _section_head("Scenario comparison"),
        html.Div(comparison, style={"overflowX": "auto"}),
    ], className="panel")

    return html.Div([sec1, sec2, sec3, sec4, sec5])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
