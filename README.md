# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the Short-Term Income (STI), Medium-Term Growth
(MTG), and Long-Term Growth (LTG) unit trusts.

_Last updated: 14 May 2026_

## Status

| Module | Status |
| --- | --- |
| 1. CMA Inputs + EDA | Built |
| 2. Trust Characteristics + CFO Brief Tables | Built |
| 3. Optimisation + Board Policy Compliance | Built |
| 4. Market Stress Testing | Built |
| 5. Drought Scenario + Onset Redemption Split | Built |
| 6. Combined Stress | Built |

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050 in your browser. The app runs with
`debug=True`, so edits to `app.py` auto-reload.

If you see stale errors after a big change:

```bash
rm -rf __pycache__ && python app.py
```

## Directory Structure

```text
FINC-3600/
├── app.py                      # Single Dash app with all 6 modules
├── requirements.txt
├── data/
│   ├── index_returns.csv       # Monthly returns Jan 2006-Feb 2026, 11 asset classes
│   └── macro_indicators.csv    # Monthly macro series Jan 2006-Feb 2026
├── modules/
│   ├── trust_calcs.py          # Trust weights, costs, return, vol, Sharpe, PSD
│   ├── metrics.py              # Liquidity and transaction-cost helpers
│   ├── optimiser.py            # Grid search + scipy refinement
│   ├── stress.py               # Historical and analytical market shocks
│   └── drought.py              # Drought cashflow projection engine
└── CPI Forecast/               # Source data + notebooks for CPI forecasting
```

## Current App Flow

### 1. CMA Inputs

- Editable 10-year forward return assumptions for 11 asset classes.
- Forecast volatility is greyed out and locked to historical annual volatility
  for the selected analysis period.
- Historical reference columns for geometric return, annual volatility, and
  forecast-minus-history delta.
- Global analysis-period controls that update historical columns and EDA charts.
- CPI assumption control used for the CPI + 2.5% NSWDF target.
- Forecast rationale and source guardrails for turning model outputs into
  CFO-brief-ready explanations.
- Macro context panels, monthly/annual returns, cumulative returns, rolling
  volatility, risk-return scatter, return distributions, descriptive statistics,
  and correlation matrix.

### 2. Trust Characteristics

- Forward-looking STI, MTG, and LTG return, volatility, Sharpe, cost, and target
  pass/fail metrics.
- Trust role cards explaining how each trust contributes to the NSWDF portfolio:
  liquidity, balanced growth, or long-term return.
- Historical backtest of the fixed trust weights.
- CFO brief export tables:
  - Table 1: asset-class review
  - Table 2: unit-trust performance
  - Table 3: recommended NSWDF portfolio

### 3. Optimisation

- Current holdings and proposed allocation controls.
- Live return, volatility, Sharpe, and liquidity/return constraint checks.
- Optimiser using 1% grid search plus SLSQP refinement.
- Feasible portfolio scatter and sensitivity sweep.
- Board Policy compliance table covering:
  - CPI + 2.5% return target
  - 10% available within 12 months
  - 25% available within 3 years
  - use of STI / MTG / LTG only
  - diversification across trusts
  - moderate-high risk appetite explanation

### 4. Market Stress

- Historical and analytical stress scenarios, including GFC, COVID/post-COVID
  inflation, AUD shock, and interest-rate shock.
- GFC stress uses the historical window from Nov 2007 to Jul 2009.
- Historical scenario asset-class returns are applied as a stress delta:
  `Forecast Return + (Scenario Stress Return - selected-period historical return)`.
- Normal-vs-stressed trust and portfolio return chart.
- Trust stress-period return table showing annualised stress-window return,
  how long the stress window lasted, selected-period historical geometric return,
  dynamic delta return, dominant exposure, and historical drawdown.
- CFO-style stress verdict summarising which trust is most exposed and how the
  recommended allocation should be explained.
- Editable shocked-return override table.

### 5. Drought

- Deterministic drought cashflow scenario with severity, total relief, onset
  year, and year-onset fraction controls.
- Year-onset drawdown split controls with one input for each trust:
  - STI (%)
  - MTG (%)
  - LTG (%)
- The onset split controls how the onset-year relief amount is redeemed across
  the trusts. Values are normalised to 100%. If a trust cannot cover its
  assigned amount, the unfunded amount spills over using the standard
  STI -> MTG -> LTG redemption order.
- Portfolio value trajectory, trust composition over time, year-onset outcome,
  liquidity pass/fail indicators, and year-by-year summary.
- Monte Carlo drought simulation with exhaustion probability and final-value
  percentile bands.

### 6. Combined Stress

- Combines a one-year market shock with the drought cashflow scenario.
- Inherits drought severity, relief amount, onset year, year-onset fraction, and
  the STI / MTG / LTG onset redemption split from Module 5.
- Compares drought-only and combined market-shock-plus-drought paths.
- Shows final value, spread cost, liquidity-breach years, fund exhaustion status,
  and final-value delta.

## Key Conventions

- All returns and costs are stored as decimals internally; the UI displays
  percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}`.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weights are fixed and defined in `modules/trust_calcs.py`.
- Global equity 50/50 unhedged/hedged exposure is encoded directly in the MTG
  and LTG trust weight vectors.
- Buy/sell spreads are applied only for transaction, drought, and combined-stress
  cashflow events, not for ordinary trust characteristic calculations.

## Assignment Alignment

The app is designed to support the FINC3600 Project 2 workflow:

- Asset-class review: Module 1
- Unit-trust analysis: Module 2
- Initial and refined NSWDF portfolio: Modules 2-3
- Board Policy compliance: Module 3
- Market stress testing: Module 4
- Extended severe drought analysis: Module 5
- Combined market crash plus drought resilience: Module 6

The dashboard supports analysis and export-ready tables, but the final CFO brief
and executive slide deck still need concise written judgement, source citations,
and AI acknowledgement as required by the assignment instructions.
