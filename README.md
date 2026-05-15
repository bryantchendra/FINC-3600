# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the Short-Term Income (STI), Medium-Term Growth
(MTG), and Long-Term Growth (LTG) unit trusts.

_Last updated: 16 May 2026 (M4 stress-only simulation; M6 hard pass gate; all-years liquidity constraint; geometric 10Y avg contributions)_

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
├── data/
│   ├── index_returns.csv       # Monthly returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv    # Monthly macro series Jan 2006–Feb 2026
├── modules/
│   ├── trust_calcs.py          # Trust weights, costs, return, vol, Sharpe
│   ├── metrics.py              # Drawdown, VaR, CVaR, liquidity, transaction cost
│   ├── optimiser.py            # Grid search + SLSQP refinement
│   ├── robust_optimiser.py     # Three-decision robust scenario optimiser
│   ├── stress.py               # Historical and analytical market shocks + multi-year crisis + recovery trajectories
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
- Each scenario produces a **multi-year crisis path** matching the historical duration,
  followed by a **recovery trajectory** for GFC and COVID Inflation Shock (2022).
- **Delta approach**: all stressed returns (crisis and recovery) are computed as
  `CMA_baseline + (historical_return − selected_period_return)`, applied consistently
  to both the chart and the scenario table.
- **Multi-year crisis path** duration per scenario:
  - GFC (21 months): Year 1 = full annualised rate; Year 2 = (1+ann)^(9/12)×(1+CMA)^(3/12)−1 (9 months crisis + 3 months CMA blend)
  - COVID Crash (2 months): 1 year (cumulative event window, not annualised)
  - COVID Inflation 2022 (12 months): 1 full year, no partial blending needed
  - AUD Depreciation: 1 year
  - Rate Shock: 2 years (Y1 shock, Y2 50% reversion)

  Each partial year is blended: `(1+ann_crisis)^frac × (1+cma)^(1-frac) − 1`. Recovery uses the same per-trust blending.
- **Recovery profiles** (per-trust trough-to-recovery dates):
  - GFC: STI Feb 2009, MTG Feb 2011, LTG Jul 2013 (trough Jul 2009)
  - COVID Inflation Shock 2022: STI Feb 2023, MTG Mar 2024, LTG Dec 2023 (trough Dec 2022)
- **Crisis multi-year return path panel**: indexed value chart with orange shading over
  crisis years and green shading over the recovery window. Per-trust "recovered" year
  annotations. This full path (crisis + recovery) is what Modules 5 and 6 apply.
- Normal-vs-stressed trust and portfolio return chart.
- Trust factor exposure table: dominant factor, historical window drawdown, crisis delta.
- **Scenario asset class returns table**:
  - Columns: Asset Class | CMA Baseline (%) | Crisis Return (%) | Crisis Delta (pp) | Recovery Return (%) | Recovery Delta (pp)
  - Sub-note shows crisis window dates and recovery window dates
- **Liquidity check under stress**: pre/post-shock trust weight drift table + Board Policy
  floor pills.
- **Portfolio simulation — stress only** (new): 10-year projection applying the selected
  scenario's full crisis + recovery path starting at a chosen year (Year 1–9), with no
  drought drawdowns. Shows BAU vs stressed value chart, trust composition over time,
  year-by-year summary table (with liquidity checks), and master fund return summary.
  Uses the same delta-adjusted `_full_scenario_trust_path` overrides as Modules 5 and 6.

### 5. Drought First

Models the **BAU + drought** scenario with post-drought rebalancing, then branches
into two forward paths.

**Branch structure:**
```
Base projection → drought drawdown → rebalance to new strategic allocation
                                          ├─ Branch (a): BAU forward to Year 10
                                          └─ Branch (b): multi-year market stress (crisis + recovery)
                                                          at a user-specified year,
                                                          then CMA from Year N+crisis+recovery
```

Within each projection year the engine applies (in order):
1. Growth — holdings compound at trust returns
2. Drawdown — drought redemption taken from grown holdings
3. Rebalance — trades on the post-drawdown portfolio (year-end)

**Controls:**
- Drought: severity (Mild/Moderate/Severe), total relief ($M), onset year,
  year-onset fraction (% drawn at onset vs residual split over Years +1/+2)
- **Onset drawdown split auto-populate**: STI / MTG / LTG % are computed from actual
  compounded pre-drawdown trust balances using the STI → MTG → LTG sequential
  redemption rule. Three drought years shown with [fully drawn / partial / untouched] tags.
- **Rebalance year**: minimum = onset (can rebalance in any drought year or later).
  _Rebalancing occurs at year-end: after growth, after that year's drawdown._
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
  _Rebalancing occurs at year-end: after growth, after that year's drawdown._

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

### 8. Robust Optimiser

Searches for a three-decision allocation policy that can pass both scenario
architectures:

1. Initial STI / MTG / LTG allocation
2. Module 5 post-drought rebalance allocation, tested on both BAU continuation
   and the late-horizon stress branch
3. Module 6 post-combined-stress rebalance allocation, tested on BAU recovery

The optimiser uses the current Module 1 CMA assumptions, Module 5 drought
inputs, Module 5 stress branch settings, and Module 6 combined-stress settings.
It filters allocations by the CPI + 2.5% return target and Board Policy
liquidity floors, then projects the candidate policy through all required
paths. A passing result is a conditional guarantee under the current scenario
settings and selected grid precision, not a mathematical guarantee against all
possible future shocks.

**Pass criterion (all three paths must pass):**

| Path | Gate |
|------|------|
| M5 BAU | Non-exhaustion + liquidity (every year) + return ≥ CPI+2.5% |
| M5 stress | Non-exhaustion + liquidity (every year) only — return hurdle relaxed; GFC-level shocks preclude meeting the 10Y average during the crisis window |
| M6 combined stress | Non-exhaustion + liquidity (every year) + return ≥ CPI+2.5% |

Liquidity is a **hard all-years constraint** — every year including drought years must maintain STI ≥ 10% and STI+MTG ≥ 25%. There is no pre-rebalance exemption.

The 10-year average is computed as geometric mean of `sum(starting_weights[t] × trust_returns[t])` for each year — the weighted net return on the allocation held at the start of each year, before drought redemptions. Consistent with the Master Fund Return Summary table.

**Controls:**
- Grid precision: 10 pp, 5 pp, or 2.5 pp allocation increments
- Liquidity pass rule: every year (default), post-rebalance years, or final year only
- Apply button that writes the recommended allocations back to Modules 3, 5,
  and 6

Master score = min(m5_bau, m5_stress, m6_recovery).avg_annual_return + 1e-9 × surplus — worst-case geometric return across all three certified paths.

**Infeasibility report:** when no policy passes, a stage-by-stage diagnostic
table shows how many candidates were tested and which constraint (return
hurdle, liquidity, fund exhaustion) caused failures at each search stage.

## Key Conventions

- All returns and costs are stored as decimals internally; the UI displays percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` — all decimals.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weights are fixed per the Investment Mandate and defined in `modules/trust_calcs.py`.
- Global equity within MTG and LTG uses a 50/50 unhedged/hedged split.
- Buy/sell spreads apply only for drought redemptions, portfolio rebalancing, and
  combined-stress projections — not for ordinary trust characteristic calculations.
- Modules 5, 6, 7, and 8 display all AUD values in millions ($M). Modules 1–4 use full AUD.
- **Rebalancing timing**: within each projection year the engine applies growth first,
  then takes any drawdown, then rebalances. Rebalancing is therefore end-of-year on the
  post-drawdown portfolio — the minimum rebalance year is `onset` (no lower bound beyond
  the drought start year).
- **Liquidity constraint**: hard all-years gate in Modules 5, 6, and 8. Every year —
  including drought drawdown years — must satisfy STI ≥ 10% (12m) and STI+MTG ≥ 25%
  (3y). No pre-rebalance exemption applies.
- **Master Fund Return Summary 10Y Avg row**: gross return, net return, and all three
  per-trust contributions are computed as geometric means — `(∏(1+annual_value))^(1/n)−1`
  over the 10-year horizon. Pass/fail against CPI+2.5% uses the geometric net return,
  consistent with the optimizer's return hurdle check.

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

The dashboard produces analysis and export-ready tables. The final CFO brief and
executive slide deck still require concise written judgement, source citations,
and AI acknowledgement as required by the assignment instructions.
