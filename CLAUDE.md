# NSWDF Portfolio Dashboard — Project Context

## Project Brief
NSW Drought Fund (AUD ~3 billion master fund) portfolio allocation dashboard.
Role: Master fund perspective. Objective: meet fund liquidity, returns, risk appetite, and drought response requirements across three unit trusts — STI (Short-Term Income), MTG (Medium-Term Growth), LTG (Long-Term Growth).

## Live File
`Project 2/FINC-3600-main/app.py` — single Dash app, all modules in one file (~5,500 lines).
Run: `cd "Project 2/FINC-3600-main" && python app.py` → http://127.0.0.1:8050

## Directory Structure
```
Project 2/FINC-3600-main/
├── app.py                          # Main Dash app (all 6 modules)
├── requirements.txt                # dash, plotly, pandas, numpy, scipy
├── README.md
├── data/
│   ├── index_returns.csv           # Monthly total returns Jan 2006–Feb 2026, 11 asset classes
│   │                               # Date format: "Mon YYYY" string, columns = asset class names
│   └── macro_indicators.csv        # Monthly macro series Jan 2006–Feb 2026
│                                   # Columns: Date, AUS CPI (YoY %), US CPI (YoY %),
│                                   #          Fed Funds Rate (%), AUD/USD, RBA Rate (%)
├── modules/
│   ├── trust_calcs.py              # Core engine: trust weight vectors, gross/net return, vol, Sharpe
│   ├── metrics.py                  # Portfolio metrics helpers
│   ├── optimiser.py                # Grid search + scipy optimisation
│   ├── stress.py                   # Market stress scenario logic + multi-year crisis paths
│   └── drought.py                  # Drought projection engine + post-drought rebalancing
└── CPI Forecast/
    ├── AU CPI Forecast/            # ABS source xlsx + Jupyter notebook
    └── US CPI Forecast/            # FRED source + Jupyter notebook
```

## Quick Navigation in app.py
app.py is ~5,500 lines. Use `grep -n` to locate sections rather than reading the whole file.
Key section anchors (run `grep -n "# Paths and historical\|Module [0-9]\|Callbacks" app.py` to verify):
- `# Paths and historical data` — startup data loading
- `# Formatting helpers` — `_fmt_pct`, `_fmt_aud`, `_fmt_m`
- `# Module 1 layout` / `# Callbacks — Module 1`
- `# Module 2 — Trust Characteristics` / `# Callbacks — Module 2`
- `# Module 3 — Portfolio Optimisation` / `# Callbacks — Module 3`
- `# Module 4 — Market Stress Testing` / `# Callbacks — Module 4`
- `# Module 5 — Drought Scenario` / `# Callbacks — Module 5` / `# Callbacks — Module 5b (Monte Carlo)`
- `# Module 6 — Combined Stress` / `# Callbacks — Module 6`

---

## Key app.py Internals (Shared / Module 1 / EDA)

### Data loaded at startup
- `_returns_df`: string-indexed ("Mon YYYY") DataFrame of monthly returns from `index_returns.csv`
- `_returns_df_dt`: DatetimeIndex version for date filtering
- `_annual_returns_df`: geometric calendar-year compounding
- `_macro_df`: DatetimeIndex macro DataFrame; derived columns: AUD/USD Δ MoM %, Fed Funds Δ MoM pp, RBA Rate Δ MoM pp
- `HIST_ARITH_ANNUAL_RETURNS`, `HIST_ANNUAL_VOL`, `HIST_CORR` — startup-computed stats fed into CMA defaults and stress scenarios
- `HIST_TRUST_MONTHLY_GROSS`, `HIST_TRUST_MONTHLY_NET`, `HIST_TRUST_CUMULATIVE_NET` — Module 2 static backtest

### Formatting helpers
- `_fmt_pct(x, decimals)` → "3.14%"
- `_fmt_signed_pct(x)` → "+1.23%" / "−0.50%"
- `_fmt_aud(x)` → "$1,234,567"
- `_fmt_m(x)` → "$1,234.5M" — **used exclusively in Modules 5 and 6 for all monetary values**

### Key Module 1 helpers
- `_filter_dates(start_m, start_y, end_m, end_y)` → boolean mask over `_dates`
- `_add_era_shading(fig, ..., era_shades=None)` → GFC / COVID vrects
- `_compute_hist_rv_for_period(...)` → `{asset: (geom_ret_pct, ann_vol_pct)}` for CMA prepopulation
- `_build_macro_timeline_fig(...)`, `_build_macro_corr_fig(...)`, `_build_domicile_regime_fig(...)`

### CMA Table (`cma-rv-table`)
Columns: Asset Class | Hist Return % (grey, read-only) | Hist Vol % (grey) | Forecast Return % (editable) | Forecast Vol % (editable) | Δ Return (conditional colour)
- `cma-store` written on every edit: `{returns, vols, corr, cpi, psd_adjusted}` all decimals
- `update_delta_on_forecast_edit` uses `PreventUpdate` to break the circular loop with `update_cma_hist_columns`

---

## Module 2 — Trust Characteristics

Forward-looking trust metrics derived from `cma-store`. Reacts live to CMA edits.

### Panels
1. Trust cards (`m2-trust-cards`) — net return, gross, asset cost, ongoing cost, vol, Sharpe, CPI+ spread, target pass/fail
2. Correlation heatmap + comparison chart (`m2-corr-heatmap`, `m2-comparison-chart`)
3. Historical backtest (`m2-backtest-chart`) + stats table (Ann. Return, Vol, Max DD, VaR 95%, CVaR 95%, Best/Worst Month)
4. CFO Brief Tables 1–3 (CSV exportable):
   - Table 1: Asset class review (Hist Return, Forecast Return, Difference, Hist Risk)
   - Table 2: Unit-trust performance (STI / MTG / LTG)
   - Table 3: Recommended NSWDF portfolio — fed by `portfolio-allocation-store` (Module 3)

---

## Module 3 — Initial Allocation (tab: "3. Initial Allocation")

Interactive trust-level allocation tool. Sets `portfolio-allocation-store` consumed by Modules 2, 4, 5, 6.

### Panels
1. Current Holdings + Proposed Allocation sliders with auto-rebalance (moving one slider redistributes others proportionally)
2. Live metrics: Net Return, Vol, Sharpe + constraint pills (12m ≥ 10%, 3y ≥ 25%, Return ≥ CPI + 2.5%)
3. Optimiser — `max_sharpe` / `min_vol` / `max_return`, vol cap, "Run optimiser" button → result card + transaction cost vs $3B
4. Feasible portfolio scatter (`m3-scatter`) — 1% grid coloured by Sharpe; white = current, teal = proposed, gold = optimised
5. Sensitivity sweep tornado (`m3-tornado`) — each asset ±50/100bps, re-optimised

### Stores
- `m3-opt-store`: serialised `OptimisationResult` dict
- `portfolio-allocation-store`: `{STI, MTG, LTG}` decimals — read by Modules 2, 4, 5, 6

---

## Module 4 — Market Stress Testing

Apply historical or analytical shocks to asset-class returns. **Key feature: each scenario produces a full multi-year crisis path** reflecting the historical horizon of the event. This path is consumed by Modules 5 and 6 for realistic multi-year stress overlays.

### Scenarios (`SCENARIO_ORDER`)
`GFC` | `COVID Crash` | `COVID Inflation Shock (2022)` | `AUD Depreciation Shock` | `Interest Rate Shock (+200bps)`

### Multi-year crisis path (`stress.build_crisis_path`)
Returns `{year_offset: asset_returns_array}` — one entry per crisis year:
- **GFC** (21 months): 2 years — Y1 = first 12 months annualised, Y2 = 9-month cumulative tail
- **COVID Crash** (2 months): 1 year — 2-month cumulative (kept as-is, not annualised, to avoid extreme rates)
- **COVID Inflation 2022** (12 months): 1 year — annualised (= cumulative for a full 12-month window)
- **AUD Depreciation**: 1 year — worst rolling 12-month window for the two unhedged assets
- **Rate Shock (+200bps)**: 2 years — Y1 = `CMA − duration × 0.02`, Y2 = 50% reversion toward CMA

### Layout Panels
1. **Scenario selector** — `m4-scenario` dropdown + reset button + `m4-scenario-meta` description
2. **Stressed return chart** (`m4-compare-chart`) — grouped bar: CMA vs stressed net return per trust + portfolio
3. **Trust stress-period returns** (`m4-factor-table`, `m4-verdict`) — stress return, dominant factor, window drawdown
4. **Crisis multi-year return path** (`m4-crisis-path-chart`, `m4-path-description`) — **new panel**: indexed value chart (1.0 = pre-crisis) through all crisis years + CMA recovery; crisis period shaded red. Description line shows per-year portfolio return.
5. **Custom shock overrides** (`m4-shock-table`) — editable DataTable; overrides Year 1 asset returns only
6. **Post-shock recovery trajectory** (`m4-recovery-chart`) — shock year + configurable recovery years at CMA
7. **Liquidity check** (`m4-liquidity-check`) — pre/post-shock weight drift + STI ≥ 10% / STI+MTG ≥ 25% pills

### Stores
- `m4-shocked-store`: `list[float]` — 11 Year 1 asset returns in decimals (shock table / compare chart / recovery / liquidity)
- `m4-path-store`: `{str(year_offset): list[float]}` — full multi-year asset return path (read by `update_m4_crisis_path`)

### Key Helpers
- `_scenario_defaults(name, cma_baseline)` → `(shocked_returns, description, window_label, return_basis, n_months)` — Year 1 only; shock table and compare chart
- `_scenario_trust_net_path(name, cma_returns)` → `{year_offset: {trust: net_return}}` — full multi-year; **consumed by Modules 5 and 6**
- `shock_compare_figure(...)` → grouped bar; `_build_m4_crisis_path_figure(asset_path, cma_baseline, weights, recovery_years)` → indexed value chart

### Callbacks
- `update_m4_scenario(...)` → shock table + `m4-shocked-store` + meta + **`m4-path-store`** (4 outputs)
- `update_m4_overrides(...)` → table edits → `m4-shocked-store`
- `update_m4_delta_column(...)` → keeps Delta column in sync
- `update_m4_outputs(shocked, alloc, cma_store, ...)` → compare chart + factor table + verdict
- **`update_m4_crisis_path(path_store, alloc, cma_store)`** → `m4-crisis-path-chart` + `m4-path-description`
- `update_m4_recovery(shocked, alloc, cma_store, recovery_years)` → recovery chart + liquidity check

---

## Module 5 — Drought First (tab: "5. Drought First")

Two-branch view: BAU + drought with post-drought rebalancing, branching into (a) BAU forward and (b) a late-horizon stress test.

### Design
```
Base projection → drought drawdown → rebalance to new allocation
                                          ├─ Branch (a): BAU forward to year 10
                                          └─ Branch (b): multi-year stress at year N, then CMA
```

### Controls
- Severity / total relief / onset year / year-onset fraction
- Onset drawdown split: STI / MTG / LTG % (normalised; onset-year relief apportioned across trusts; unfunded spills STI→MTG→LTG)
- **Post-drought rebalancing panel** (`_rebalancing_controls(onset)`):
  - Rebalance year input (default = onset + 3)
  - New strategic allocation: STI / MTG / LTG % — **independent of Module 3**; can overweight LTG after drought obligations are met
  - Liquidity constraint checker + drifted weights display (`m5-rebalance-constraint`, `m5-drift-weights`)
  - Stress scenario dropdown (same `SCENARIO_ORDER`) + stress onset year input

### Layout Panels (order)
1. Controls + config summary (`m5-config-summary`, `m5-onset-split-summary`)
2. **Portfolio value trajectory (BAU)** (`m5-value-chart`) — base projection, no rebalance
3. Year-onset outcome card (`m5-summary-card`, `m5-exec-verdict`)
4. Post-drought rebalancing panel — constraint pill + drift weights
5. **Branch comparison** (`m5-branch-chart`, `m5-branch-summary`) — BAU (teal) vs stress (orange dashed); vlines for drought / rebalance / stress onset; year-10 value cards per branch
6. **Trust composition over time** (`m5-composition-chart`) — BAU/Stress toggle (`m5-comp-toggle`)
7. **Year-by-year summary table** (`m5-projection-table-container`, `m5-totals`) — same toggle applies

All monetary values in **$M** via `_fmt_m()`.

### Year-by-year table columns
Year | Starting ($M) | Growth ($M) | Rebal. Cost ($M) | Drawdown ($M) | Spread ($M) | Ending ($M) | STI% | MTG% | LTG% | 12m liq | 3y liq

### Three projections in `update_module_5`
1. `result` — base BAU + drought, no rebalance (drives upper panels + exec verdict)
2. `bau_branch` — BAU + drought + rebalance at `rebalance_year` with new allocation (Branch a)
3. `stress_result` — same as (2) + **multi-year stress** starting at `stress_year` (Branch b):
```python
trust_net_path = _scenario_trust_net_path(stress_scenario, returns)
overrides = {stress_year + yr_offset - 1: nets
             for yr_offset, nets in trust_net_path.items()
             if 1 <= stress_year + yr_offset - 1 <= 10}
```

`comp_source = stress_result if comp_toggle == "stress" else bau_branch` drives composition chart + table toggle.

### Module 5b — Monte Carlo
`run_monte_carlo(...)` → `m5-mc-summary`, `m5-mc-fan-chart`, `m5-mc-exhaustion-chart`. Fan chart in $M.

---

## Module 6 — Combined Stress (Market Crash + Drought)

Stacks a **multi-year crisis** onto the drought projection. Both drought-only and combined paths shown, with a delta section.

### Controls
- `m6-scenario` (same `SCENARIO_ORDER`) + `m6-shock-year`
- Drought params inherited from Module 5

### Multi-year override logic (same as Module 5)
```python
trust_net_path = _scenario_trust_net_path(scenario_name, returns)
m6_overrides = {shock_year + yr_offset - 1: nets
                for yr_offset, nets in trust_net_path.items()
                if 1 <= shock_year + yr_offset - 1 <= 10}
stressed = dr.project(..., trust_return_overrides=m6_overrides, ...)
```
Config text reports "N crisis year(s) applied".

### Panels
1. **Combined trajectory** (`m6-value-chart`) — drought-only (teal) vs crash+drought (red dashed); vlines for drought and shock years
2. **Joint impact summary** (`m6-summary-grid`) — side-by-side cards + final-value delta section

All monetary values in **$M**.

---

## Module Logic Connections (`modules/` → `app.py`)

### `trust_calcs.py` (as `tc`) — Core engine, ALL modules
- `ASSET_CLASSES`, `TRUST_NAMES`, `TRUST_RAW_WEIGHTS`, `TRUST_BUY_SPREADS`, `TRUST_SELL_SPREADS`, `TRUST_ONGOING_COSTS`
- `build_trust_weight_vector(trust)` → 11-element weight array
- `cma_to_covariance(vols, corr)` → 11×11 cov matrix
- `trust_gross_return(trust, asset_returns)`, `trust_net_return(trust, asset_returns)` → scalar
- `trust_volatility(trust, cov)`, `trust_sharpe(trust, asset_returns, cov, cash)`
- `portfolio_net_return(weights, asset_returns)`, `portfolio_volatility(weights, cov)`
- `trust_characteristics(returns, cov, cash, cpi)` → dict of dicts (Module 2 cards + CFO tables)
- `historical_trust_returns_monthly(df)`, `historical_trust_returns_monthly_net(df)`, `historical_cumulative_wealth(df)`

### `metrics.py` (as `mt`) — Stateless metric helpers
- `max_drawdown`, `var_historic`, `cvar_historic`, `annualised_geometric`, `annualised_vol` → Module 2 backtest stats
- `transaction_cost_aud(current_w, new_w, portfolio_aud, buy_spreads, sell_spreads)` → Module 3 result card
- `liquidity_coverage(weights)` → `{within_12m, within_3y, meets_12m, meets_3y}` → Module 3 constraint pills

### `optimiser.py` (as `op`) — Grid search + SLSQP, Module 3 only
- `liquidity_feasible(w_sti, w_mtg)`, `generate_grid()`, `evaluate_grid(grid, returns, cov, cash)`
- `optimise(objective, returns, cov, cash, cpi, vol_cap)` → `OptimisationResult`
- `sensitivity_sweep(...)` → tornado DataFrame
- `OptimisationResult.to_dict()` / `.from_dict()` for `m3-opt-store` serialisation

### `stress.py` (as `st`) — Scenario engine, Modules 4, 5, 6
- `SCENARIO_WINDOWS`: `{name: (start_str, end_str)}` — "Mon YYYY" format; keys match `SCENARIO_ORDER`
- `StressScenario` dataclass: `asset_returns`, `description`, `window_label`, `return_basis`, `n_months`, `is_analytical`, `window_cumulative`
- `build_historical_scenario(name, returns_df, start, end)` → single `StressScenario` (used by shock table display)
- `build_aud_shock_scenario(returns_df, baseline)`, `build_rate_shock_scenario(baseline)` → analytical scenarios
- `build_all_scenarios(returns_df, baseline)` → `_PRECOMPUTED_SCENARIOS` (startup, used by `_scenario_defaults`)
- **`build_crisis_path(name, returns_df, cma_baseline)`** → `{int: np.ndarray}` — multi-year asset return path:
  - Historical: window sliced into 12-month chunks; full chunks annualised, partial tail kept as cumulative
  - AUD Shock: `{1: worst_12m_returns}`
  - Rate Shock: `{1: shocked_CMA - duration*0.02, 2: CMA + 0.5*(shock - CMA)}`
- `trust_returns_under_shock(asset_returns)` → `{trust: net_return}` (full-year costs); used for annualised paths
- `trust_returns_under_event_shock(asset_returns, n_months)` → pro-rates costs to event window; Module 4 compare chart for event-window basis
- `portfolio_return_under_shock(weights, asset_returns)`, `dominant_factor(trust, shocked_returns)`, `trust_drawdown_from_window(...)`

### `drought.py` (as `dr`) — Cashflow projection engine, Modules 5, 6
- `SEVERITY_BANDS`: `{severity: (lo_aud, hi_aud)}`
- **`YearState`** dataclass: `year`, `starting_value`, `growth`, `pre_drawdown_value`, `drawdown`, `spread_costs`, `ending_value`, `ending_weights`, `ending_holdings`, `meets_12m`, `meets_3y`, **`rebalance_cost`** (AUD; 0 if no rebalance that year)
- **`ProjectionResult`** dataclass: `years`, `initial_value`, `final_value`, `total_drawdown`, `total_spread_cost`, `fund_exhausted`, `exhaustion_year`
- `build_drought_schedule(onset_year, total_relief, year_4_fraction, residual_split)` → `{year: aud_amount}`
- **`project(initial_value, weights, asset_returns, schedule, horizon, drawdown_splits, trust_return_overrides, rebalance_schedule)`**:
  - `drawdown_splits`: `{onset_year: {trust: fraction}}` — apportions onset-year relief; unfunded spills STI→MTG→LTG
  - `trust_return_overrides`: `{year: {trust: net_return}}` — multi-year capable; Modules 5/6 pass the full crisis path dict
  - **`rebalance_schedule`**: `{year: {trust: target_weight}}` — triggers rebalance after growth, before drawdown; cost = Σ |trade| × spread; holdings scaled by `(V − cost) / V`; records `rebalance_cost` on `YearState`
- `post_drawdown_summary(result, onset_year)` → dict for Module 5 onset card
- `MonteCarloResult` + `monte_carlo(...)` → vectorised N-path simulation (Module 5b)

---

## Cross-Module Data Flow
```
cma-store  (returns, vols, corr, cpi — all decimals)
   ├─→ tc.trust_characteristics()          → Module 2 cards / CFO tables
   ├─→ tc.portfolio_net_return()           → Module 3 live metrics / optimiser
   ├─→ _PRECOMPUTED_SCENARIOS              → Module 4 shock table defaults
   ├─→ st.build_crisis_path()              → Module 4 m4-path-store / crisis chart
   ├─→ _scenario_trust_net_path()          → Module 5 stress branch, Module 6 overrides
   └─→ dr.project()                        → Module 5 (3 calls), Module 6 (2 calls)

portfolio-allocation-store  ({STI, MTG, LTG} decimals — set by Module 3 proposed sliders)
   ├─→ Module 2 CFO Table 3
   ├─→ Module 4 compare chart + crisis path chart
   ├─→ Module 5 all dr.project() calls
   └─→ Module 6 both dr.project() calls
```

---

## Session Tips for Claude
- **Do not read all of `app.py`** — use `grep -n` with section anchors above, then `Read` with `offset` + `limit`.
- `grep -n "def update_module_5\|def update_m4\|def update_module_6" app.py` → callback entry points.
- After edits: `python app.py` (auto-reload via `debug=True`). Stale `.pyc` errors: `rm -rf __pycache__ && python app.py`.
- **Git lock files**: sandbox creates `.git/HEAD.lock` and `.git/index.lock` that it cannot delete. If `git push` fails with lock errors, user must run `rm -f .git/HEAD.lock .git/index.lock` from their own terminal first.
- All monetary values in Modules 5 and 6 use `_fmt_m()`. Modules 1–4 use `_fmt_pct()` and `_fmt_aud()`.
- `portfolio-allocation-store` values are decimals summing to ~1.0. Always normalise: `total_w = sum(alloc.values()); w = {t: alloc[t]/total_w ...}`.
- Rebalancing cost for Module 5: `total_rebal_cost = sum(y.rebalance_cost for y in result.years)`.
