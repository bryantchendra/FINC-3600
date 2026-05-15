# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the Short-Term Income (STI), Medium-Term Growth
(MTG), and Long-Term Growth (LTG) unit trusts.

_Last updated: 15 May 2026 (session 3)_

## Status

| Module | Tab Label | Status |
| --- | --- | --- |
| 1. CMA Inputs + EDA | 1. CMA Inputs | Built |
| 2. Trust Characteristics + CFO Brief Tables | 2. Trust Characteristics | Built |
| 3. Initial Allocation + Optimisation | 3. Initial Allocation | Built |
| 4. Market Stress Testing | 4. Market Stress | Built |
| 5. Drought Scenario (BAU + stress branches) | 5. Drought First | Built |
| 6. Combined Stress (Market Crash + Drought) | 6. Combined Stress | Built |
| 7. Executive Summary | 7. Executive Summary | Built |

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050. The app runs with `debug=True` so edits to
`app.py` auto-reload.

If you see stale errors after a change:

```bash
rm -rf __pycache__ && python app.py
```

## Directory Structure

```text
FINC-3600-main/
├── app.py                      # Single Dash app — all 7 modules (~6,200 lines)
├── requirements.txt
├── data/
│   ├── index_returns.csv       # Monthly returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv    # Monthly macro series Jan 2006–Feb 2026
├── modules/
│   ├── trust_calcs.py          # Trust weights, costs, return, vol, Sharpe
│   ├── metrics.py              # Drawdown, VaR, CVaR, liquidity, transaction cost
│   ├── optimiser.py            # Grid search + SLSQP refinement
│   ├── stress.py               # Historical and analytical market shocks + multi-year crisis paths
│   └── drought.py              # Drought cashflow projection + post-drought rebalancing engine
└── CPI Forecast/               # Source data + notebooks for CPI forecasting
```

## Module Guide

### 1. CMA Inputs

- Editable 10-year forward return assumptions for 11 asset classes.
- Forecast volatility locked to historical annual volatility for the selected
  analysis period (greyed out, not editable).
- Historical reference columns for geometric return, annual volatility, and a
  Δ Return column that updates live as forecasts are edited.
- Single analysis-period date range at the top drives all charts, historical
  columns, and macro panels.
- CPI assumption input used for the CPI + 2.5% NSWDF return target.
- **CMA Consistency Flags panel** — appears below the table and checks three dimensions:
  - _Hedged/Unhedged pairs_: if the unhedged variant carries more vol, it should earn a higher return
  - _Cross-tier hierarchy_: avg return should increase Cash → Bonds → Listed Equity/Real Assets → Private Equity
  - _Within-tier_: for pairs in the same risk tier where vol differs by ≥ 1 pp, the higher-vol asset should have a higher return
  - Each flag has a **tick-to-dismiss** checkbox; ticking crosses out the flag text and reveals an inline note box for the user's rationale. Dismissed flags persist in `m1-ignored-flags` store.
- Macro context panels: timeline (dual-axis), asset–macro correlations heatmap,
  and annualised returns by macro regime.
- Standard EDA: monthly returns, cumulative returns, rolling volatility, risk–return
  scatter, return distributions, and descriptive statistics.

### 2. Trust Characteristics

- Forward-looking STI, MTG, and LTG cards: net return, gross return, asset cost,
  ongoing cost, volatility, Sharpe ratio, CPI+ spread, and target pass/fail pill.
  All reactive to Module 1 CMA edits.
- Historical backtest of the fixed trust weights (Jan 2006–Feb 2026).
- CFO Brief export tables:
  - Table 1: asset-class review
  - Table 2: unit-trust performance
  - Table 3: recommended NSWDF portfolio (pulls from Module 3)

### 3. Initial Allocation

- Current holdings and proposed allocation controls (slider with auto-rebalance).
- Live return, volatility, Sharpe, and Board Policy liquidity/return constraint pills.
- SLSQP optimiser with grid-search seed: `max_sharpe` / `min_vol` / `max_return`
  objectives, optional vol cap. Result card includes round-trip transaction cost.
- Feasible portfolio scatter (1% grid coloured by Sharpe; current, proposed,
  optimised markers).
- Sensitivity sweep tornado: each asset class bumped ±50/100bps, re-optimised.
- **Board Policy compliance table** includes two informational rows:
  - Domicile exposure: AU vs Global (% of total portfolio)
  - Asset type exposure: Cash / Bond / Equity (% of total portfolio)

### 4. Market Stress

- Five scenarios: GFC, COVID Crash, COVID Inflation Shock (2022), AUD Depreciation
  Shock, Interest Rate Shock (+200bps).
- Each scenario produces a **multi-year crisis path** matching the historical duration.
- **Crisis multi-year return path panel**: indexed value chart showing each trust and
  portfolio through all crisis years then CMA recovery. This path is what Modules 5/6 apply.
- Normal-vs-stressed trust and portfolio return chart.
- Trust factor exposure table: dominant factor, historical window drawdown, delta return.
- **Scenario asset class returns table** (read-only).
- **Liquidity check under stress**: pre/post-shock trust weight drift table + Board Policy
  floor pills.

### 5. Drought First

Models the **BAU + drought** scenario with post-drought rebalancing, then branches
into two forward paths.

**Branch structure:**
```
Base projection → drought drawdown → rebalance to new strategic allocation
                                          ├─ Branch (a): BAU forward to Year 10
                                          └─ Branch (b): multi-year market stress
                                                          at a user-specified year,
                                                          then CMA from Year N+crisis
```

**Controls:**
- Drought: severity (Mild/Moderate/Severe), total relief ($M), onset year,
  year-onset fraction (% drawn at onset vs residual split over Years +1/+2)
- **Onset drawdown split auto-populate**: STI / MTG / LTG % are computed from actual
  compounded pre-drawdown trust balances using the STI → MTG → LTG sequential
  redemption rule. Three drought years shown with [fully drawn / partial / untouched] tags.
- **Rebalance year**: minimum = onset (can rebalance in any drought year or later).
  _Rebalancing occurs at year-end: after that year's growth, before that year's drawdown._
- New strategic allocation: STI / MTG / LTG % — auto-sums to 100.
- Board Policy compliance table for the rebalanced allocation.
- Stress scenario (same list as Module 4) + stress onset year.
- Dynamic year bound enforcement: rebalance year ≥ onset; stress year > rebalance year.

**Panels (order):**
1. Portfolio value trajectory (BAU — base case, no rebalance)
2. Year-onset outcome card + pre-drawdown balance panel (3 drought years)
3. Post-drought rebalancing panel (constraint + drift + compliance)
4. Branch comparison chart: BAU vs stress-test value over 10 years
5. Trust composition over time — BAU/Stress toggle
6. Year-by-year summary table (same toggle)
7. Master fund return summary table (same toggle)

All monetary values displayed in **$M**.

### 6. Combined Stress

Stacks a **multi-year market crash** onto the drought simultaneously. Models the
worst-case scenario of both events coinciding.

**Three projections:**
1. **Drought-only BAU** — drought drawdowns with no market shock (reference line)
2. **Stressed (crash + drought)** — combined shock overlaid; no rebalancing
3. **Rebalanced recovery** — same combined shock, then rebalanced to new allocation,
   then BAU recovery from the rebalance year (no second stress branch)

**Controls:**
- Market crash scenario + shock year (independently configurable)
- Drought parameters inherited from Module 5
- Post-event rebalancing: rebalance year (minimum = onset), new STI / MTG / LTG %.
  _Rebalancing occurs at year-end: after growth, before that year's drawdown._

**Panels:**
1. Recovery trajectory chart (3 lines: drought-only / stressed / rebalanced)
2. Drawdown profile: actual per-trust redemptions under the stressed path
3. Post-event rebalancing controls
4. Year-by-year summary table
5. Return summary and joint impact cards

All monetary values in **$M**.

### 7. Executive Summary

Side-by-side comparison of Scenario 1 (Module 5) and Scenario 2 (Module 6).

**Sections:**
1. Starting position — fund value, allocation, trust metrics
2. Drought configuration — severity, relief, onset, drawdown schedule
3. Scenario 1 detail — drought impact, post-drought rebalancing (with year-end note),
   Branch (a) BAU outcomes, Branch (b) stress-test outcomes
4. Scenario 2 detail — stress scenario overlay, combined impact, post-event
   rebalancing (with year-end note), recovery outcomes
5. Five-column comparison table — key metrics side by side across both scenarios

## Key Conventions

- All returns and costs are stored as decimals internally; the UI displays percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` — all decimals.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weights are fixed per the Investment Mandate and defined in `modules/trust_calcs.py`.
- Global equity within MTG and LTG uses a 50/50 unhedged/hedged split.
- Buy/sell spreads apply only for drought redemptions, portfolio rebalancing, and
  combined-stress projections — not for ordinary trust characteristic calculations.
- Modules 5, 6, and 7 display all AUD values in millions ($M). Modules 1–4 use full AUD.
- **Rebalancing timing**: within each projection year the engine applies growth first,
  then rebalances, then takes any drawdown. Rebalancing is therefore end-of-year on the
  grown portfolio — the minimum rebalance year is `onset` (no lower bound beyond the
  drought start year).

## Assignment Alignment

| Deliverable | Module(s) |
| --- | --- |
| Asset-class review | 1 |
| Unit-trust analysis | 2 |
| Initial NSWDF portfolio + Board Policy compliance | 3 |
| Market stress testing | 4 |
| Extended severe drought analysis (BAU + rebalance + stress) | 5 |
| Combined market crash + drought resilience | 6 |
| Executive summary + scenario comparison | 7 |

The dashboard produces analysis and export-ready tables. The final CFO brief and
executive slide deck still require concise written judgement, source citations,
and AI acknowledgement as required by the assignment instructions.
