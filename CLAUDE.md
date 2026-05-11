# NSWDF Portfolio Dashboard — Project Context

## Project Brief
NSW Drought Fund (AUD ~3 billion master fund) portfolio allocation dashboard.
Role: Master fund perspective. Objective: meet fund liquidity, returns, risk appetite, and drought response requirements across three unit trusts — STI (Short-Term Income), MTG (Medium-Term Growth), LTG (Long-Term Growth).

## Live File
`Project 2/FINC-3600-main/app.py` — single Dash app, all modules in one file.
Run: `cd "Project 2/FINC-3600-main" && python app.py` → http://127.0.0.1:8050

## Directory Structure
```
Project 2/FINC-3600-main/
├── app.py                          # Main Dash app (all 6 modules)
├── requirements.txt                # dash, plotly, pandas, numpy, scipy
├── README.md                       # Project README
├── data/
│   ├── index_returns.csv           # Monthly total returns Jan 2006–Feb 2026, 11 asset classes
│   │                               # Date format: "Jan 2006" string, columns = asset class names
│   └── macro_indicators.csv        # Monthly macro series Jan 2006–Feb 2026
│                                   # Columns: Date, AUS CPI (YoY %), US CPI (YoY %),
│                                   #          Fed Funds Rate (%), AUD/USD, RBA Rate (%)
│                                   # Sources: ABS (ffill), FRED CPIAUCSL, FRED FEDFUNDS,
│                                   #          FRED DEXUSAL (monthly avg), RBA Table A2 (decisions ffill)
├── modules/
│   ├── trust_calcs.py              # Core engine: trust weight vectors, gross/net return, vol, Sharpe
│   ├── metrics.py                  # Portfolio metrics helpers
│   ├── optimiser.py                # Grid search + scipy optimisation
│   ├── stress.py                   # Market stress scenario logic
│   └── drought.py                  # Drought scenario logic
└── CPI Forecast/
    ├── AU CPI Forecast/            # ABS source xlsx + Jupyter notebook
    └── US CPI Forecast/            # FRED source + Jupyter notebook
```

## Key app.py Internals (Module 1 / EDA)

### Data & Constants
- `_returns_df_dt`: DatetimeIndex DataFrame of monthly returns loaded from `data/index_returns.csv`
- `_annual_returns_df`: geometric calendar-year compounding of monthly returns
- `_macro_df`: DatetimeIndex DataFrame loaded from `data/macro_indicators.csv`, reindexed to `_dates` with ffill. Contains: AUS CPI (YoY %), US CPI (YoY %), Fed Funds Rate (%), AUD/USD, RBA Rate (%), plus derived columns AUD/USD Δ MoM %, Fed Funds Δ MoM pp, RBA Rate Δ MoM pp.
- `ASSET_COLORS`: dict mapping 11 asset class names → consistent hex colors
- `_HIST_GREY = "#EBEBEB"`, `_HIST_GREY_H = "#DCDCDC"` — background for historical columns in CMA table
- `_DELTA_STYLES`: conditional style list for delta column (teal-green positive, plum negative; thresholds 0.5% / 1.5%)
- `_ERA_SHADES`: GFC (2007-11-01 to 2009-06-30) and COVID (2020-02-01 to 2020-04-30) standard shading
- `_ERA_SHADES_ROLLING`: GFC (2007-12-01 to 2009-07-31, 1-month lag) + COVID — used for rolling vol chart
- `_MACRO_CORR_VARS`: dict of macro column name → display label for asset–macro correlation heatmap
- `_AUS_ASSETS`: fixed list — Cash, Australian Short Duration Bond, Australian Fixed Income, Australian Listed Equity, Australian Listed Property
- `_GLOBAL_ASSETS`: fixed list — Global Fixed Income (Hedged), Global Credit (Hedged), Global Listed Equity (Unhedged), Global Listed Equity (Hedged), Global Infrastructure (Unhedged), Global Private Equity
- `_MACRO_VAR_COLORS`, `_MACRO_VAR_FMT`: dicts mapping macro variable name → hex colour / d3 format string for timeline figure

### Key Helper Functions
- `_filter_dates(start_m, start_y, end_m, end_y)` → boolean mask over `_dates`
- `_add_era_shading(fig, ..., era_shades=None)` → adds vrects only if era overlaps selected period; pass `_ERA_SHADES_ROLLING` for rolling vol chart
- `_build_returns_time_fig(...)` → monthly line or annualised bar+line, era shading, filtered by period
- `_build_cumulative_fig(...)`, `_build_rolling_vol_fig(...)`, `_build_scatter_fig(...)`
- `_build_desc_stats_data(...)`, `_build_histograms_fig(...)`, `_build_corr_heatmap_eda_fig(...)`
- `_compute_hist_rv_for_period(...)` → `{asset: (geom_ret_pct, ann_vol_pct)}` for CMA pre-population
- `_build_macro_timeline_fig(primary_var, overlay_var, sm, sy, em, ey)` → dual-axis line chart; any of 5 macro vars on left axis, optional overlay on right
- `_build_macro_corr_fig(sm, sy, em, ey)` → Pearson correlation heatmap: 11 assets × 5 macro factors
- `_build_domicile_regime_fig(assets, regime_type, sm, sy, em, ey)` → grouped bar of annualised returns split by macro regime. `regime_type` ∈ `aus_cpi | rba_rate | us_cpi | fed_funds | audusd`. CPI uses fixed thresholds (AUS: <2% / 2–3% / >3%; US: <1.7% / 1.7–2.3% / >2.3%); rate vars use period tertiles.
- `_asset_checklist(...)` → Checklist + All/None buttons component
- `_date_range_row(prefix, ...)` → From/To month+year dropdowns (IDs: `{prefix}-start-m/y`, `{prefix}-end-m/y`)

### Global Period Selector
Single `_date_range_row("m1", ...)` at the top of Module 1 drives ALL charts and tables in Module 1 via shared Input IDs `m1-start-m`, `m1-start-y`, `m1-end-m`, `m1-end-y`.

### CMA Table (`cma-rv-table`) Columns (left → right)
Asset Class | Hist Return % p.a. (grey, read-only, 1dp) | Hist Vol % p.a. (grey, read-only, 1dp) | Forecast Return % p.a. (white, editable, 1dp) | Forecast Vol % p.a. (white, editable, 1dp) | Δ Return (conditional colour, 1dp)
- All numeric columns display to 1 decimal place
- Both Forecast columns are white (unshaded) and editable; only historical columns are grey
- `_DELTA_STYLES` drives conditional cell colour on Δ Return; thresholds at ±0.5% and ±1.5%

### Module 1 Layout — Section Order (after CMA panel)
1. **Macro Indicators Timeline** (`m1-macro-timeline`) — dropdowns: `m1-macro-primary` (left axis), `m1-macro-overlay` (right axis, optional). All 5 macro vars available in both dropdowns. Era shading applied.
2. **Annualised Returns by Macro Regime** — two side-by-side panels:
   - AUS Domicile (`m1-aus-regime-chart`): dropdown `m1-aus-regime-dd` (aus_cpi / rba_rate / audusd)
   - Global / US (`m1-us-regime-chart`): dropdown `m1-us-regime-dd` (us_cpi / fed_funds / audusd)
3. **Asset Class – Macro Factor Correlations** (`m1-macro-corr`) — static heatmap, period-reactive
4. **Exploratory Analysis on Historical Data** — descriptive stats table (`_desc_stats_table`), % cols 1dp, skew/kurt 2dp
5. **Monthly Return Distributions** (`m1-histograms`) — standalone panel
6. **Monthly Returns Over Time** (`returns-time-chart`) — monthly line or annualised bar+line; era shading in monthly mode
7. **Cumulative Returns** (`cumulative-chart`) — GFC + COVID era shading
8. **12-Month Rolling Annualised Volatility** (`rolling-vol-chart`) — `_ERA_SHADES_ROLLING` (1-month lag on GFC)
9. **Risk-Return Scatter + Correlation Matrix** (side by side: `scatter-chart`, `m1-corr-eda`)

### Module 1 Callbacks
- `update_macro_timeline(primary, overlay, sm, sy, em, ey)` → `m1-macro-timeline`
- `update_macro_corr(sm, sy, em, ey)` → `m1-macro-corr`
- `update_aus_regime_chart(regime_type, sm, sy, em, ey)` → `m1-aus-regime-chart`
- `update_us_regime_chart(regime_type, sm, sy, em, ey)` → `m1-us-regime-chart`
- `update_cma_hist_columns(sm, sy, em, ey, current_data)` → refreshes grey hist columns in `cma-rv-table`; forecast cols untouched
- `update_cma_store(rv_data, cpi_pct)` → writes `cma-store` from forecast columns
- All EDA figure callbacks take `m1-start/end-m/y` as Inputs; `prevent_initial_call=True` on All/None toggles

## Trust Architecture
- **STI** (Short-Term Income): liquidity-focused, low-risk
- **MTG** (Medium-Term Growth): balanced growth
- **LTG** (Long-Term Growth): max growth, longest horizon
- Global equity block within MTG and LTG = 50/50 Unhedged/Hedged split
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` all in decimals

## 11 Asset Classes (order matters — matches CSV columns and `trust_calcs.ASSET_CLASSES`)
Cash, Australian Short Duration Bond, Australian Fixed Income, Global Fixed Income (Hedged), Global Credit (Hedged), Australian Listed Equity, Global Listed Equity (Unhedged), Global Listed Equity (Hedged), Australian Listed Property, Global Infrastructure (Unhedged), Global Private Equity

## Module 2 — Trust Characteristics

### Purpose
Forward-looking trust metrics derived from `cma-store`. Reacts live to any Module 1 CMA edits.

### Layout Panels
1. **Trust cards** (`m2-trust-cards`) — three `.trust-card` divs (one per trust) showing net return, target pass/fail pill, gross return, asset cost, ongoing cost, volatility, Sharpe, CPI+ spread.
2. **Correlation heatmap + comparison chart** (`m2-corr-heatmap`, `m2-comparison-chart`) — side-by-side; heatmap uses fixed historical `HIST_CORR`; bar chart compares net return / vol / Sharpe across STI, MTG, LTG.
3. **Historical backtest** (`m2-backtest-chart`) — static cumulative wealth chart (monthly rebalancing, net of costs, Jan 2006–Feb 2026). `_backtest_stats_table()` beneath it shows Ann. Return, Vol, Max DD, VaR 95%, CVaR 95%, Best/Worst Month.
4. **CFO Brief tables** — three `DataTable`s with CSV export:
   - Table 1 (`m2-cfo-table-1`): Asset Class, Historical Return, Forecast Return, Difference, Historical Risk.
   - Table 2 (`m2-cfo-table-2`): Net Return and Risk for each trust (STI / MTG / LTG columns).
   - Table 3 (`m2-cfo-table-3`): Recommended trust mix, Forecast Return, Forecast Risk — fed by `portfolio-allocation-store` (Module 3).

### Key Helpers
- `_trust_card(trust_name, c)` → html.Div built from `tc.trust_characteristics()` output dict
- `correlation_heatmap_figure(corr_matrix)` → diverging RdBu heatmap
- `trust_comparison_figure(chars)` → 3-panel subplot (net return / vol / Sharpe)
- `historical_backtest_figure()` → static line chart from `HIST_TRUST_CUMULATIVE_NET`
- `_backtest_stats_rows()` → uses `mt.annualised_geometric`, `mt.max_drawdown`, `mt.var_historic`, `mt.cvar_historic`

### Callbacks
- `update_module_2(store)` → Input `cma-store`; Outputs trust cards, heatmap, comparison chart, CFO Tables 1 & 2.
- `update_module_2_table_3(cma_store, alloc)` → Inputs `cma-store` + `portfolio-allocation-store`; Output CFO Table 3.

---

## Module 3 — Portfolio Optimisation

### Purpose
Interactive trust-level allocation tool. Current Holdings vs Proposed Allocation, live metrics, and a SLSQP-based optimiser with sensitivity sweep.

### Layout Panels
1. **Allocation inputs** — two `_alloc_block()` panels:
   - *Current Holdings* (`current-{trust}`): `dcc.Input` number fields (manual entry).
   - *Proposed Allocation* (`proposed-{trust}`): `dcc.Slider` fields with auto-rebalance (moving one slider redistributes the other two proportionally).
   - `_live_metrics_block()`: live Net Return, Volatility, Sharpe + constraint pills (12m liquidity ≥ 10%, 3y liquidity ≥ 25%, Return ≥ CPI + 2.5%).
2. **Optimiser panel** — Objective dropdown (`max_sharpe` / `min_vol` / `max_return`), optional vol cap input, "Run optimiser" button. Result card shows optimal weights, metrics, and round-trip transaction cost vs $3B base.
3. **Feasible portfolio scatter** (`m3-scatter`) — 1% grid of all liquidity-feasible portfolios, coloured by Sharpe. White dot = current, teal dot = proposed, gold star = optimised.
4. **Sensitivity sweep tornado** (`m3-tornado`) — each asset class bumped ±50/100 bps, re-optimised; bar chart of change in optimal vol.

### Key Constants / Helpers
- `PORTFOLIO_AUD = 3_000_000_000`
- `OBJECTIVE_LABELS`: labels for the dropdown
- `_alloc_block(block_id, title, note, input_kind, default)` → reusable allocation grid (number or slider)
- `_live_metrics_block()` → KPI strip + constraint row
- `_scatter_figure(grid_eval, target, current_w, proposed_w, optimal_w)` → feasible-space scatter
- `_tornado_figure(sens, baseline_vol, objective)` → horizontal grouped bar chart
- `_opt_result_card(opt, current_w, proposed_w, target)` → result display with transaction cost breakdown

### Stores
- `m3-opt-store`: serialised `OptimisationResult` dict (`feasible`, `weights`, `net_return`, `volatility`, `sharpe`, `objective`)
- `portfolio-allocation-store`: `{STI, MTG, LTG}` decimals (written by proposed sliders; read by Modules 2, 4, 5, 6)

### Callbacks
- `rebalance_proposed(sti, mtg, ltg)` — auto-rebalances proposed sliders to sum to 100%.
- `push_proposed_to_store(sti, mtg, ltg)` — normalises and writes `portfolio-allocation-store`.
- `update_live(...)` — 15 outputs; recomputes live KPIs and constraint pills on every slider/store change.
- `update_scatter(store, alloc, ...)` — regenerates feasible scatter (calls `op.generate_grid()` + `op.evaluate_grid()`).
- `run_optimiser(...)` — fires on button click; calls `op.optimise()`; writes `m3-opt-store` and result card.
- `apply_optimised(...)` — "Apply to proposed sliders" button; reads `m3-opt-store` → writes slider values.
- `update_tornado(store, objective, volcap_pct)` — calls `op.sensitivity_sweep()`; always live (no button).

---

## Module 4 — Market Stress Testing

### Purpose
Apply named historical or analytical shocks to asset-class returns and observe trust/portfolio impact. Shocked returns are fully editable in a DataTable.

### Scenarios (SCENARIO_ORDER)
GFC | COVID Crash | COVID Inflation Shock (2022) | AUD Depreciation Shock | Interest Rate Shock (+200bps)

The first three and AUD shock use `st.SCENARIO_WINDOWS` windows from `_returns_df`; the rate shock is analytical (duration-based).

### Layout Panels
1. **Scenario selector** — `dcc.Dropdown` (`m4-scenario`) + "Reset to scenario defaults" button. `m4-scenario-meta` div shows a description block.
2. **Stressed return chart** (`m4-compare-chart`) — grouped bar: Normal (CMA) vs Stressed net return for each trust + portfolio.
3. **Factor exposure table** (`m4-factor-table`) — HTML table showing Net Return Under Stress, Dominant Factor tag, and historical window drawdown.
4. **Custom shock overrides** (`m4-shock-table`) — editable `DataTable`: Asset Class | Baseline | Shocked (editable) | Delta. Live-recomputes chart and factor table.

### Key Helpers / Internals
- `_PRECOMPUTED_SCENARIOS = st.build_all_scenarios(_returns_df, _HIST_BASELINE)` — built at startup
- `_scenario_defaults(name, cma_baseline)` → `(shocked_returns, description, window_label)`
- `shock_compare_figure(baseline, shocked, portfolio_weights)` → grouped bar figure
- `_factor_breakdown_rows(shocked, df, scenario, window)` → list of dicts for the factor table
- `_factor_class(label)` → CSS class name for colour-coded factor tags
- `m4-shocked-store`: `list[float]` — 11 shocked annual returns in decimals

### Callbacks
- `update_m4_scenario(scenario, n_clicks, cma_store)` → resets shock table, updates shocked-store and meta block.
- `update_m4_overrides(table_data, prev_shocked)` → propagates manual cell edits to `m4-shocked-store`.
- `update_m4_delta_column(_, table_data, cma_store)` → keeps the Delta column in sync with edits.
- `update_m4_outputs(shocked, alloc, cma_store, scenario)` → produces compare chart + factor table; reads `portfolio-allocation-store`.

---

## Module 5 — Drought Scenario

### Purpose
Deterministic cashflow projection: a drought forces fund liquidations (STI→MTG→LTG order) across a configurable schedule. Also includes a 5b Monte Carlo sub-panel.

### Controls
- Severity: Mild / Moderate / Severe (bounds slider via `dr.SEVERITY_BANDS`)
- Total relief: `dcc.Slider` in $M, bounds updated by severity selection
- Onset year: integer 1–8
- Year-onset fraction: % of total relief drawn in onset year; residual split 50/50 over next two years

### Layout Panels
1. **Controls + config summary** (`m5-config-summary`) — text summary of current scenario.
2. **Portfolio value trajectory** (`m5-value-chart`) — line chart of AUD value per year; drought years marked with dotted vertical lines.
3. **Trust composition** (`m5-composition-chart`) — stacked area chart of STI/MTG/LTG holdings over time.
4. **Year-onset outcome** (`m5-summary-card`) — `_summary_card()` showing remaining value, drawdown, spread cost, new trust mix, 12m/3y liquidity pass/fail.
5. **Year-by-year table** (`m5-projection-table-container`) — `DataTable` with columns: Year | Starting | Growth | Drawdown | Spread cost | Ending | STI% | MTG% | LTG% | Liq 12m | Liq 3y.
6. **Module 5b — Monte Carlo** — 10,000-path simulation; controls: n_paths dropdown, seed input, "Re-run simulation" button. Outputs: summary strip (`m5-mc-summary`), fan chart (`m5-mc-fan-chart`), cumulative exhaustion bar chart (`m5-mc-exhaustion-chart`).

### Key Helpers
- `_projection_value_figure(result, onset_year)` → line + vlines
- `_trust_composition_figure(result)` → stacked area
- `_projection_summary_table(result)` → DataTable from `dr.ProjectionResult`
- `_summary_card(summary, total_drawdown, total_spread)` → detailed outcome card
- `_mc_fan_figure(mc, initial_value)` → percentile fan (P5/P25/P50/P75/P95)
- `_mc_exhaustion_figure(mc)` → cumulative P(exhaustion) bar chart
- `_mc_summary_strip(mc, initial_value)` → KPI grid

### Callbacks
- `update_relief_bounds(severity, current_value)` → updates slider bounds/marks from `dr.SEVERITY_BANDS`.
- `update_module_5(severity, relief_m, onset, fraction_pct, alloc, cma_store)` → full deterministic projection; calls `dr.build_drought_schedule()` then `dr.project()`.
- `run_monte_carlo(n_clicks, alloc, cma_store, n_paths, seed)` → calls `dr.monte_carlo()`.

---

## Module 6 — Combined Stress (Market Crash + Drought)

### Purpose
Stacks a one-year market shock (from Module 4 scenarios) onto the Module 5 drought projection. The shock hits in a single configurable year; all other years use CMA returns. Drought parameters are inherited from Module 5; allocation from Module 3.

### Layout Panels
1. **Controls** — `m6-scenario` dropdown (same `SCENARIO_ORDER` as Module 4), `m6-shock-year` input (1–10), config summary text.
2. **Combined trajectory** (`m6-value-chart`) — two lines: "Drought only" (teal) vs "Combined (crash + drought)" (red). Dotted lines = drought years; dashed line = shock year.
3. **Joint impact summary** (`m6-summary-grid`) — two side-by-side summary cards (drought-only vs combined), plus a delta section showing final-value impact and year-of-shock ending values.

### Key Helpers
- `_combined_value_figure(baseline, stressed, shock_year, drought_years)` → dual-line figure
- `_module_6_summary(baseline, stressed, shock_year, scenario_name)` → joint impact HTML block

### Callback
- `update_module_6(scenario_name, shock_year, severity, relief_m, onset, fraction_pct, alloc, cma_store)` → single callback combining Module 4 shock lookup (`_scenario_defaults`) + two calls to `dr.project()` (baseline and stressed with `trust_return_overrides`).

---

## Module Logic Connections (`modules/` → `app.py`)

Each file in `modules/` is a pure computation layer — no Dash, no I/O. `app.py` imports them and calls them inside callbacks.

### `trust_calcs.py` (imported as `tc`) — Core engine, used by ALL modules
Single source of truth for the asset universe and trust definitions. Almost every callback touches it.
- `ASSET_CLASSES`, `ASSET_CLASS_SHORT` — canonical name lists; CSV column order must match.
- `TRUST_RAW_WEIGHTS`, `TRUST_NAMES`, `TRUST_BUY_SPREADS`, `TRUST_SELL_SPREADS`, `TRUST_ONGOING_COSTS` — fixed trust parameters from the IM.
- `build_trust_weight_vector(trust)` → 11-element weight array; used inside gross/net/vol functions.
- `cma_to_covariance(vols, corr)` → 11×11 cov matrix from `cma-store`; fed into all trust vol and portfolio vol calls.
- `trust_gross_return(trust, asset_returns)`, `trust_net_return(trust, asset_returns)` → scalar; Module 2, 4, 5, 6.
- `trust_volatility(trust, cov)`, `trust_sharpe(trust, asset_returns, cov, cash)` → Module 2.
- `portfolio_net_return(weights, asset_returns)`, `portfolio_volatility(weights, cov)` → Module 3 live metrics.
- `trust_characteristics(returns, cov, cash, cpi)` → dict of dicts; drives Module 2 trust cards + CFO tables.
- `historical_trust_returns_monthly(df)`, `historical_trust_returns_monthly_net(df)`, `historical_cumulative_wealth(df)` → Module 2 static backtest (computed once at startup).

### `metrics.py` (imported as `mt`) — Stateless metric helpers, used by Modules 2 & 3
No trust-level knowledge; operates on return arrays.
- `max_drawdown`, `var_historic`, `cvar_historic`, `annualised_geometric`, `annualised_vol` → Module 2 backtest stats table.
- `transaction_cost_aud(current_w, new_w, portfolio_aud, buy_spreads, sell_spreads)` → Module 3 optimiser result card; uses `tc.TRUST_BUY_SPREADS` / `TRUST_SELL_SPREADS` passed in from `app.py`.
- `liquidity_coverage(weights)` → Module 3 live constraint pills; returns `{within_12m, within_3y, meets_12m, meets_3y}`.

### `optimiser.py` (imported as `op`) — Grid search + SLSQP, used by Module 3 only
Knows about the 3-trust space and liquidity constraints; calls `tc` functions internally.
- `TARGET_SPREAD = 0.025` (the CPI + 2.5% spread target constant).
- `liquidity_feasible(w_sti, w_mtg)` → gate for the grid (w_STI ≥ 10%, w_STI + w_MTG ≥ 25%).
- `generate_grid()` → all 1%-resolution feasible weight triples.
- `evaluate_grid(grid, returns, cov, cash)` → DataFrame with net_return, volatility, sharpe per grid point; drives Module 3 scatter.
- `optimise(objective, returns, cov, cash, cpi, vol_cap)` → `OptimisationResult`; grid seed → SLSQP refinement.
- `sensitivity_sweep(objective, returns, cov, cash, cpi, ...)` → DataFrame of re-optimised results with bumped returns; drives Module 3 tornado.
- `OptimisationResult` dataclass has `.to_dict()` / `.from_dict()` for `m3-opt-store` serialisation.

### `stress.py` (imported as `st`) — Scenario engine, used by Modules 4 & 6
Builds stressed asset-return arrays from historical CSV windows or analytical rules.
- `SCENARIO_WINDOWS` — dict of `{scenario_name: (start_str, end_str)}`; keys match `SCENARIO_ORDER` in `app.py`.
- `StressScenario` dataclass: `asset_returns` (11-element array), `description`, `window_label`.
- `build_historical_scenario(returns_df, name, baseline)` → computes annualised window return, blends with baseline for non-equity classes.
- `build_aud_shock_scenario(returns_df, baseline)` → currency-specific analytical shock.
- `build_rate_shock_scenario(baseline)` → duration × 200 bps shock, no CSV data needed.
- `build_all_scenarios(returns_df, baseline)` → called once at app startup → `_PRECOMPUTED_SCENARIOS` dict.
- `trust_returns_under_shock(asset_returns)` → `{trust: net_return}`; used by Module 4 chart and Module 6 projection overrides.
- `portfolio_return_under_shock(weights, asset_returns)` → scalar; Module 4 compare chart portfolio bar.
- `dominant_factor(trust, asset_returns)` → `(label, contribution)`; Module 4 factor table.
- `trust_drawdown_from_window(trust, df, start, end)` → historical peak-to-trough over the actual scenario window; Module 4 factor table.

### `drought.py` (imported as `dr`) — Cashflow projection engine, used by Modules 5 & 6
Handles liquidation ordering (STI→MTG→LTG), sell-spread costs, liquidity tracking.
- `SEVERITY_BANDS` — `{severity: (lo_aud, hi_aud)}` bounds for the Module 5 slider.
- `YearState` dataclass: per-year snapshot (`starting_value`, `pre_drawdown_value`, `drawdown`, `ending_value`, `ending_weights`, `ending_holdings`, `spread_costs`, `liquidity_within_12m/3y`, `meets_12m/3y`).
- `ProjectionResult` dataclass: list of `YearState`, `final_value`, `total_drawdown`, `total_spread_cost`, `fund_exhausted`, `exhaustion_year`.
- `build_drought_schedule(onset_year, total_relief, year_4_fraction, residual_split)` → `{year: aud_amount}` dict.
- `project(initial_value, weights, asset_returns, schedule, horizon, trust_return_overrides)` → `ProjectionResult`. The optional `trust_return_overrides={year: {trust: net_return}}` is how Module 6 injects the market shock into a single year.
- `post_drawdown_summary(result, onset_year)` → dict for Module 5 summary card.
- `MonteCarloResult` dataclass + `monte_carlo(...)` → vectorised 10,000-path simulation for Module 5b.

### Cross-module data flow summary
```
cma-store  ──────────────────────────────────────────────────┐
(returns, vols, corr, cpi)                                   │
   │                                                          ▼
   ├─→ tc.trust_characteristics() ──→ Module 2 cards/tables  op.optimise()
   ├─→ tc.portfolio_net_return()  ──→ Module 3 live metrics   op.sensitivity_sweep()
   ├─→ st._scenario_defaults()   ──→ Module 4 shock table          │
   └─→ dr.project()              ──→ Module 5 trajectory           │
                                                                    │
portfolio-allocation-store  ───────────────────────────────────────┘
(STI, MTG, LTG weights, set by Module 3 proposed sliders)
   │
   ├─→ Module 2 CFO Table 3
   ├─→ Module 4 compare chart (portfolio bar)
   ├─→ Module 5 drought projection
   └─→ Module 6 combined projection
```

---

## Session Tips for Claude
- **Do not read all of `app.py`** — it is ~4,100 lines. Use this CLAUDE.md to identify the relevant module section, then read only that line range with `offset` + `limit`.
- Approximate line ranges in `app.py`:
  - Macro data loading + constants (`_macro_df`, `_AUS_ASSETS`, `_GLOBAL_ASSETS`, era shades) ≈ 750–810
  - CMA table + `_cma_rv_table()` ≈ 1270–1350
  - Module 1 layout ≈ 1350–1675
  - Module 2 layout ≈ 1675–1920
  - Module 3 layout ≈ 1920–2140
  - Module 4 layout ≈ 2140–2320
  - Module 5 layout ≈ 2320–2600
  - Module 6 layout ≈ 2600–2700
  - Module 1 callbacks ≈ 2993–3320
  - Module 2 callbacks ≈ 3322–3400
  - Module 3 callbacks ≈ 3402–3700
  - Module 4 callbacks ≈ 3704–3855
  - Module 5 callbacks ≈ 3858–4060
  - Module 6 callbacks ≈ 4061–end
- The live file path is: `FINC-3600-main/app.py` (CLAUDE.md now lives in the same directory).
- After edits: user runs `python app.py` in their terminal (auto-reload is on via `debug=True`).
- If stale `.pyc` errors appear: `rm -rf __pycache__ && python app.py`
