# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the Short-Term Income (STI), Medium-Term Growth
(MTG), and Long-Term Growth (LTG) unit trusts.

_Last updated: 15 May 2026_

## Status

| Module | Tab Label | Status |
| --- | --- | --- |
| 1. CMA Inputs + EDA | 1. CMA Inputs | Built |
| 2. Trust Characteristics + CFO Brief Tables | 2. Trust Characteristics | Built |
| 3. Initial Allocation + Optimisation | 3. Initial Allocation | Built |
| 4. Market Stress Testing | 4. Market Stress | Built |
| 5. Drought Scenario (BAU + stress branches) | 5. Drought First | Built |
| 6. Combined Stress (Market Crash + Drought) | 6. Combined Stress | Built |

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
├── app.py                      # Single Dash app — all 6 modules (~5,500 lines)
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
- Single analysis-period date range at the top of Module 1 drives all charts,
  historical columns, and macro panels.
- CPI assumption input used for the CPI + 2.5% NSWDF return target.
- Macro context panels: timeline (dual-axis), asset–macro correlations heatmap,
  and annualised returns by macro regime (CPI, rate, AUD/USD regimes).
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

### 4. Market Stress

- Five scenarios: GFC, COVID Crash, COVID Inflation Shock (2022), AUD Depreciation
  Shock, Interest Rate Shock (+200bps).
- Each scenario now produces a **multi-year crisis path** matching the historical
  duration of the event — not a single shocked year:
  - GFC (21 months) → 2 crisis years applied
  - COVID Crash (2 months) → 1 year (cumulative, not annualised)
  - COVID Inflation 2022 (12 months) → 1 year
  - AUD Depreciation → 1 year (worst rolling 12-month window)
  - Rate Shock → 2 years (Year 1 = full shock; Year 2 = 50% reversion)
- **Crisis multi-year return path panel**: indexed value chart (1.0 = pre-crisis)
  showing each trust and the portfolio through all crisis years, then CMA recovery.
  Tells you exactly what Modules 5 and 6 will apply.
- Normal-vs-stressed trust and portfolio return chart.
- Trust factor exposure table: dominant factor tag (equity beta, duration, credit
  spread, currency), historical window drawdown, delta return.
- Editable shocked-return override table (Year 1 only).
- Post-shock recovery trajectory (configurable recovery horizon).
- Liquidity check: pre/post-shock trust weight drift + Board Policy floor pills.

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
- Onset drawdown split: STI / MTG / LTG % target (unfunded amount spills
  STI → MTG → LTG automatically)
- Post-drought rebalancing:
  - Rebalance year (default = onset + 3)
  - New strategic allocation: STI / MTG / LTG % — independent of Module 3;
    can overweight LTG once drought obligations are met
  - Live liquidity constraint checker + drifted weights display
  - Stress scenario (same list as Module 4) + stress onset year

**Panels (order):**
1. Portfolio value trajectory (BAU — base case, no rebalance)
2. Year-onset outcome card
3. Post-drought rebalancing panel (constraint + drift)
4. Branch comparison chart: BAU vs stress-test value over 10 years (with Year-10 cards)
5. Trust composition over time — BAU/Stress toggle
6. Year-by-year summary table (same toggle)

All monetary values displayed in **$M**.

**Year-by-year table columns:**
Year | Starting ($M) | Growth ($M) | Rebal. Cost ($M) | Drawdown ($M) | Spread ($M) | Ending ($M) | STI% | MTG% | LTG% | 12m liq | 3y liq

**Monte Carlo sub-panel (5b):**
10,000-path simulation using the same drought schedule. Outputs: P5/P25/P50/P75/P95
fan chart, cumulative exhaustion probability bar chart, and summary strip.

### 6. Combined Stress

Stacks a market shock onto the drought cashflow. Uses the same multi-year crisis
paths as Module 5.

- Market shock scenario and shock year are independently configurable.
- Drought parameters are inherited from Module 5 (severity, relief, onset, fraction,
  onset split).
- Both drought-only and combined (crash + drought) paths shown on one chart.
- Config text reports the number of crisis years applied to the stressed path.
- Joint impact summary: side-by-side outcome cards and a final-value delta section
  showing how much the market crash changes the Year 10 outcome.

All monetary values in **$M**.

## Key Conventions

- All returns and costs are stored as decimals internally; the UI displays percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` — all decimals.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weights are fixed per the Investment Mandate and defined in `modules/trust_calcs.py`.
- Global equity within MTG and LTG uses a 50/50 unhedged/hedged split, encoded in the trust weight vectors.
- Buy/sell spreads apply only for drought redemptions, portfolio rebalancing, and combined-stress projections — not for ordinary trust characteristic calculations.
- Modules 5 and 6 display all AUD values in millions ($M). Modules 1–4 use full AUD where shown.

## Assignment Alignment

| Deliverable | Module(s) |
| --- | --- |
| Asset-class review | 1 |
| Unit-trust analysis | 2 |
| Initial NSWDF portfolio + Board Policy compliance | 3 |
| Market stress testing | 4 |
| Extended severe drought analysis (BAU + rebalance + stress) | 5 |
| Combined market crash + drought resilience | 6 |

The dashboard produces analysis and export-ready tables. The final CFO brief and
executive slide deck still require concise written judgement, source citations,
and AI acknowledgement as required by the assignment instructions.
