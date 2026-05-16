# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the Short-Term Income (STI), Medium-Term Growth
(MTG), and Long-Term Growth (LTG) unit trusts.

_Last updated: 17 May 2026 (M8 diversification floor, M5 stress toggle, active-constraints panel; M4 stress sim shows no-rebalance vs rebalanced; rebalancing label corrected to post-drawdown)_

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
| 8. Robust Scenario Optimiser | 8. Robust Optimiser | Built |

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
├── app.py                      # Single Dash app — all 8 modules
├── requirements.txt
├── USER_GUIDE.txt              # Plain-language guide for end users
├── data/
│   ├── index_returns.csv       # Monthly returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv    # Monthly macro series Jan 2006–Feb 2026
├── modules/
│   ├── trust_calcs.py          # Trust weights, costs, return, vol, Sharpe
│   ├── metrics.py              # Drawdown, VaR, CVaR, liquidity, transaction cost
│   ├── optimiser.py            # Grid search + SLSQP refinement (LTG_MAX = 0.50, TRUST_MIN = 0.00)
│   ├── robust_optimiser.py     # Three-decision robust scenario optimiser
│   ├── stress.py               # Historical and analytical market shocks + multi-year crisis + recovery trajectories
│   └── drought.py              # Drought cashflow projection + post-drought rebalancing engine
└── CPI Forecast/               # Source data + notebooks for CPI forecasting
```

## Key Conventions

- All returns and costs are stored as decimals internally; the UI displays percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` — all decimals.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weights are fixed per the Investment Mandate and defined in `modules/trust_calcs.py`.
- Global equity within MTG and LTG uses a 50/50 unhedged/hedged split.
- Buy/sell spreads apply only for drought redemptions, portfolio rebalancing, and
  combined-stress projections — not for ordinary trust characteristic calculations.
- Modules 5, 6, 7, and 8 display all AUD values in millions ($M). Modules 1–4 use full AUD.
- **LTG cap**: LTG allocation is hard-capped at 50% across all modules. Enforced in the
  `generate_grid` function (`LTG_MAX = 0.50` in `modules/optimiser.py`), all SLSQP refiners,
  and all UI sliders and rebalancing inputs.
- **Per-trust cap toggle** (Module 8): RadioItems `m8-trust-cap-toggle` switches between
  "50% cap per trust (Board policy)" and "No per-trust cap". State stored in `trust-cap-store`
  (boolean). When off, `trust_max=1.0` is passed through the full optimizer chain.
- **Diversification floor** (Module 8): RadioItems `m8-trust-min-select` sets a minimum
  allocation per trust: 5%, 10%, or 15%. Applied in `generate_grid` and all SLSQP refiners
  via `trust_min` parameter. Default 5%.
- **Rebalancing timing**: within each projection year the engine applies growth first,
  then takes any drawdown, then rebalances. Rebalancing is therefore end-of-year on the
  post-drawdown portfolio — the minimum rebalance year is `onset`.
- **Liquidity constraint**: hard all-years gate in Modules 4, 5, 6, and 8. Every year —
  including drought drawdown years — must satisfy STI ≥ 10% (12m) and STI+MTG ≥ 25%
  (3y). No pre-rebalance exemption applies.
- **Master Fund Return Summary 10Y Avg row**: gross return, net return, and all three
  per-trust contributions are computed as geometric means — `(∏(1+annual_value))^(1/n)−1`
  over the 10-year horizon. Pass/fail against CPI+2.5% uses the geometric net return,
  consistent with the optimizer's return hurdle check.
- **Master scenario selector**: the scenario dropdown in Module 4 drives Modules 5, 6, 7,
  and 8 simultaneously. M5 and M6 show a read-only label; there is no separate per-module
  scenario picker.
- **Recovery return floor**: recovery-phase returns are floored at the CMA baseline.
  Formula: `max(CMA_baseline, CMA_baseline + delta)`.

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
- **CMA Consistency Flags panel** — checks hedged/unhedged pairs, cross-tier hierarchy,
  and within-tier vol/return relationships. Each flag has a tick-to-dismiss checkbox.
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

- **Proposed Allocation** sliders with auto-rebalance. LTG is hard-capped at 50%.
- Live return, volatility, Sharpe, and Board Policy liquidity/return constraint pills.
- SLSQP optimiser with grid-search seed: `max_sharpe` / `min_vol` / `max_return`
  objectives, optional vol cap. Result card includes round-trip transaction cost
  (proposed → optimised).
- Feasible portfolio scatter (1% grid coloured by Sharpe; proposed and optimised markers).
- Sensitivity sweep tornado: each asset class bumped ±50/100bps, re-optimised.
- **Board Policy compliance table** includes two informational rows:
  - Domicile exposure: AU vs Global (% of total portfolio)
  - Asset type exposure: Cash / Bond / Equity (% of total portfolio)

### 4. Market Stress

- **Portfolio simulation — stress only** panel appears immediately below the scenario selector.
- Chart shows **three lines**: BAU (CMA), Stressed (no rebalance), Stressed + rebalanced —
  clearly showing the benefit of strategic rebalancing. Recovery-start rebalance removed.
- **Master scenario selector** — propagates to Modules 5, 6, 7, and 8.
- Five scenarios: GFC, COVID Crash, COVID Inflation Shock (2022), AUD Depreciation
  Shock, Interest Rate Shock (+200bps).
- Each scenario produces a **multi-year crisis path** matching the historical duration,
  followed by a **recovery trajectory** for GFC and COVID Inflation Shock (2022).
- **Post-stress rebalancing option**: after the crisis + recovery phase, set a new
  strategic allocation shown as the third line in the chart.

### 5. Drought First

Models the **BAU + drought** scenario with post-drought rebalancing, then branches
into two forward paths. The market stress scenario is set in Module 4.

**Branch structure:**
```
Base projection → drought drawdown → rebalance (year-end, after drawdown)
                                          ├─ Branch (a): BAU forward to Year 10
                                          └─ Branch (b): multi-year market stress at user year
```

### 6. Combined Stress

Stacks a **multi-year market crash** onto the drought simultaneously. Three projections:
drought-only BAU / stressed (crash + drought) / rebalanced recovery.

### 7. Executive Summary

Side-by-side comparison of Scenario 1 (Module 5) and Scenario 2 (Module 6).

### 8. Robust Optimiser

Searches for a three-decision allocation policy that passes certified scenario paths:

1. Initial STI / MTG / LTG allocation
2. Module 5 post-drought rebalance (tested on BAU continuation and optionally late-horizon stress)
3. Module 6 post-combined-stress rebalance (tested on BAU recovery)

**Controls:**
- Grid precision: 10 pp, 5 pp, or 2.5 pp allocation increments
- Liquidity pass rule: every year (default), post-rebalance years, or final year only
- **Per-trust cap toggle**: enforce 50% Board Policy cap or remove it
- **Diversification floor**: 5% / 10% / 15% minimum per trust (applied to all three trusts)
- **M5 late-stress branch toggle**: include (optimise against stress) or exclude (BAU-only M5 optimisation)
- **M5 stress pass mode**: soft (survival + liquidity, default) or hard (must also meet return ≥ CPI+2.5%)
- **Active constraints panel**: live bullet summary of all hard constraints, path gates, and search settings — updates as controls change
- Apply button that writes the recommended allocations back to Modules 3, 4, 5, and 6

**Pass criterion:**

| Path | Gate |
|------|------|
| M4 stress-only | Non-exhaustion + liquidity only — return hurdle relaxed |
| M5 BAU | Non-exhaustion + liquidity only — return hurdle relaxed |
| M5 stress (if included) | Non-exhaustion + liquidity (soft) or + return ≥ CPI+2.5% (hard) |
| M6 combined stress | Non-exhaustion + liquidity + return ≥ CPI+2.5% |

Seed weights removed — candidates come exclusively from the constraint-filtered grid.

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
| Robust three-decision allocation optimiser | 8 |
