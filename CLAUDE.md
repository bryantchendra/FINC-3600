# NSWDF Portfolio Dashboard — Project Context

## Project Brief
NSW Drought Fund (AUD ~3 billion master fund) portfolio allocation dashboard.
Role: Master fund perspective. Objective: meet fund liquidity, returns, risk appetite, and drought response requirements across three unit trusts — STI (Short-Term Income), MTG (Medium-Term Growth), LTG (Long-Term Growth).

## Live File
`Project 2/FINC-3600-main/app.py` — single Dash app, all modules in one file (~7,700+ lines).
Run: `cd "Project 2/FINC-3600-main" && python app.py` → http://127.0.0.1:8050

## Directory Structure
```
Project 2/FINC-3600-main/
├── app.py                          # Main Dash app (all 7 modules)
├── requirements.txt                # dash, plotly, pandas, numpy, scipy
├── README.md
├── data/
│   ├── index_returns.csv           # Monthly total returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv        # Monthly macro series Jan 2006–Feb 2026
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
app.py is ~6,200 lines. Use `grep -n` to locate sections rather than reading the whole file.
Key section anchors:
- `# Paths and historical data` — startup data loading
- `# Formatting helpers` — `_fmt_pct`, `_fmt_aud`, `_fmt_m`
- `# Module 1 layout` / `# Callbacks — Module 1` / `# CMA consistency validation`
- `# Module 2 — Trust Characteristics` / `# Callbacks — Module 2`
- `# Module 3 — Portfolio Optimisation` / `# Callbacks — Module 3`
- `# Module 4 — Market Stress Testing` / `# Callbacks — Module 4`
- `# Module 5 — Drought Scenario` / `# Callbacks — Module 5` / `# Callbacks — Module 5b (Monte Carlo)`
- `# Module 6 — Combined Stress` / `# Callbacks — Module 6`
- `# Module 7 — Executive Summary` / `# Callbacks — Module 7`
- `# Callbacks — Module 8 (Robust Scenario Optimiser)`

---

## Key app.py Internals (Shared / Module 1 / EDA)

### Data loaded at startup
- `_returns_df`: string-indexed ("Mon YYYY") DataFrame of monthly returns from `index_returns.csv`
- `_returns_df_dt`: DatetimeIndex version for date filtering
- `_annual_returns_df`: geometric calendar-year compounding
- `_macro_df`: DatetimeIndex macro DataFrame
- `HIST_ARITH_ANNUAL_RETURNS`, `HIST_ANNUAL_VOL`, `HIST_CORR` — startup-computed stats
- `HIST_TRUST_MONTHLY_GROSS`, `HIST_TRUST_MONTHLY_NET`, `HIST_TRUST_CUMULATIVE_NET` — Module 2 static backtest

### Formatting helpers
- `_fmt_pct(x, decimals)` → "3.14%"
- `_fmt_signed_pct(x)` → "+1.23%" / "−0.50%"
- `_fmt_aud(x)` → "$1,234,567"
- `_fmt_m(x)` → "$1,234.5M" — **used exclusively in Modules 5, 6, and 7 for all monetary values**

### CMA Table (`cma-rv-table`)
Columns: Asset Class | Hist Return % (grey, read-only) | Hist Vol % (grey) | Forecast Return % (editable) | Forecast Vol % (greyed, locked to hist) | Δ Return (conditional colour)
- `cma-store` written on every edit: `{returns, vols, corr, cpi, psd_adjusted}` all decimals
- `update_delta_on_forecast_edit` uses `PreventUpdate` to break the circular loop with `update_cma_hist_columns`

### CMA Consistency Flags (`m1-cma-flags`)
Implemented via `_compute_cma_flags(cma_store)` → `list[str]`.

Three validation dimensions (module-level constants `_CMA_RISK_TIERS`, `_CMA_TIER_LABELS`, `_CMA_HEDGE_PAIRS`):
1. **Hedged/Unhedged pairs** — `_CMA_HEDGE_PAIRS`: `[("Global Listed Equity (Unhedged)", "Global Listed Equity (Hedged)")]`. If the higher-vol variant has lower or equal return, flag it.
2. **Cross-tier hierarchy** — `_CMA_RISK_TIERS` {0: Cash, 1: Bonds, 2: Listed Equity/Real Assets, 3: Private Equity}. Average return must increase tier-by-tier.
3. **Within-tier** — for any two assets in the same tier where vol differs by ≥ 1 pp, higher vol should mean higher return.

Rendering (`_flag_row(flag_text, ignored, note)`):
- Each flag has a `dcc.Checklist(id={"type":"m1-flag-cb","index":flag_text})` checkbox. Checking = dismissed.
- Dismissed flags get strikethrough text + inline `dcc.Input(id={"type":"m1-flag-note","index":flag_text}, debounce=True)` for rationale notes.
- `update_cma_flags`: `Input("cma-store")` + `Input("m1-ignored-flags")` → `Output("m1-cma-flags")`. Adding `m1-ignored-flags` as Input (not just State) allows dismissed state to immediately re-render without a cma-store change.
- `sync_flag_ignores`: `Input({"type":"m1-flag-cb",ALL})` + `Input({"type":"m1-flag-note",ALL})` → `Output("m1-ignored-flags")`. Store format: `{flag_text: note_text}` dict.
- `m1-ignored-flags` store: `dcc.Store(id="m1-ignored-flags", data={})` — dict, not list.

**Loop safety**: Dash reconciles by component ID. When `update_cma_flags` re-renders checkboxes with the same value they already held, Dash detects no change and does not re-trigger `sync_flag_ignores`.

---

## Module 2 — Trust Characteristics

Forward-looking trust metrics derived from `cma-store`. Reacts live to CMA edits.

### Panels
1. Trust cards (`m2-trust-cards`) — net return, gross, asset cost, ongoing cost, vol, Sharpe, CPI+ spread, target pass/fail
2. Correlation heatmap + comparison chart
3. Historical backtest + stats table (Ann. Return, Vol, Max DD, VaR 95%, CVaR 95%, Best/Worst Month)
4. CFO Brief Tables 1–3 (CSV exportable)

---

## Module 3 — Initial Allocation

Interactive trust-level allocation tool. Sets `portfolio-allocation-store` consumed by Modules 2, 4, 5, 6, 7.

**Current Holdings removed** — there is no longer a separate "current allocation" input. Transaction cost in the optimiser result card is now computed proposed → optimal. `run_optimiser` uses `alloc` (proposed) as `current_w`.

**LTG cap**: `LTG_MAX = 0.50` defined in `modules/optimiser.py`. Enforced in:
- `generate_grid` (replaces `w_ltg > 1 + 1e-12` with `w_ltg > LTG_MAX + 1e-12`)
- All three SLSQP refiners: added `w_STI + w_MTG >= 1 - LTG_MAX` constraint
- UI: `_alloc_block("proposed", ..., max_ltg=50)`, M4/M5/M6 rebalance inputs `max=50`

**`_alloc_block(block_id, title, note, input_kind, default, max_ltg=100)`**: `max_ltg` param caps LTG slider/input; other trusts remain at 100.

### Panels
1. Proposed Allocation sliders with auto-rebalance (LTG ≤ 50%)
2. Live metrics: Net Return, Vol, Sharpe + constraint pills
3. Optimiser — `max_sharpe` / `min_vol` / `max_return`, vol cap → result card + transaction cost (proposed → optimal)
4. Feasible portfolio scatter (proposed + optimal markers; no current marker)
5. Sensitivity sweep tornado
6. Board Policy compliance table (`_board_compliance_table`) — includes domicile and asset type Info rows

### `update_live` callback
Reduced from 15 outputs to 10 (current-* outputs removed). Inputs: `proposed-STI/MTG/LTG`, `cma-store`, `m3-objective`.

### Stores
- `m3-opt-store`: serialised `OptimisationResult` dict
- `portfolio-allocation-store`: `{STI, MTG, LTG}` decimals

---

## Module 4 — Market Stress Testing

**Master scenario selector**: `m4-scenario` dropdown is the single source of scenario selection for the entire app. Modules 5, 6, 7, and 8 all read from `m4-scenario` as an `Input` (M5, M6, M8) or `Input` (M7). The `m5-stress-scenario` and `m6-scenario` dropdowns have been removed. M5 and M6 layouts show a read-only "Set in Module 4" label. `persist_user_state` saves `m4_scenario` at top level (not under `m5` or `m6` sub-keys). `m4-scenario` default loads from `_SAVED.get("m4_scenario", "GFC")`.

**Recovery return floor**: `stress.build_scenario_recovery` floors recovery rates at CMA:
`annual_rate = cma + max(0.0, hist - selected_period)` — i.e. `max(cma, cma + delta)`.
When the delta-adjusted recovery is below CMA, CMA is used instead.

### Scenarios (`SCENARIO_ORDER`)
`GFC` | `COVID Crash` | `COVID Inflation Shock (2022)` | `AUD Depreciation Shock` | `Interest Rate Shock (+200bps)`

### Delta approach (universal)
All stressed returns — crisis AND recovery — use:
`stressed_return = CMA_baseline + (historical_scenario_return − selected_period_historical_return)`
Recovery is additionally floored at CMA baseline (see above).
Applied consistently to: crisis path chart, shock table, M4 stress simulation, Modules 5/6 projections, Module 7 summary, and Module 8 optimiser gates.

### Multi-year crisis path (`stress.build_crisis_path`)
Returns `{year_offset: asset_returns_array}`:
- GFC (21 months): 2 years
- COVID Crash (2 months): 1 year (cumulative, not annualised)
- COVID Inflation 2022: 1 year
- AUD Depreciation: 1 year (worst rolling 12m window)
- Rate Shock: 2 years (Y1 = CMA − duration×0.02, Y2 = 50% reversion)

### Recovery trajectories (`stress.build_scenario_recovery`)
Per-trust recovery defined in `stress.RECOVERY_PROFILES` for GFC and COVID Inflation Shock (2022) only.
Other scenarios use generic CMA recovery (no `RECOVERY_PROFILES` entry → returns `None`).

**Recovery profiles (trough → per-trust recovery date):**
| Scenario | Trust | Trough | Recovery |
|----------|-------|--------|---------|
| GFC | STI | Jul 2009 | Feb 2009 (already recovered) |
| GFC | MTG | Jul 2009 | Feb 2011 |
| GFC | LTG | Jul 2009 | Feb 2013 (latest-trust recovery horizon = 3y7m) |
| COVID Inflation Shock (2022) | STI | Dec 2022 | Feb 2023 |
| COVID Inflation Shock (2022) | MTG | Dec 2022 | Mar 2024 |
| COVID Inflation Shock (2022) | LTG | Dec 2022 | Dec 2023 |

Recovery window is split into 12-month annual buckets plus the true partial final-month bucket. Same convention as `build_crisis_path`, but the UI now displays the month-level horizon instead of describing the final partial bucket as a full year.

Both crisis and recovery use **month-fraction blending** for partial years: `(1+ann)^frac × (1+cma)^(1-frac) − 1` where `frac = months_this_year / 12`. Full years use the annualised rate directly. This ensures the total compounded loss matches the historical window exactly.

`build_scenario_recovery` signature:
```python
build_scenario_recovery(
    scenario_name: str,
    cma_trust_nets: dict[str, float],
    returns_df: pd.DataFrame,
    selected_period_trust_nets: dict[str, float],
) -> dict[int, dict[str, float]] | None
```
Returns `{recovery_year_offset: {trust: net_return}}` relative to the end of the crisis, or `None` if no recovery profile exists.

`recovery_window_for_scenario(scenario_name) -> tuple[str, str] | None` — returns `(recovery_start_label, latest_recovery_date_label)` for display in the asset class table.
`recovery_chunk_months(scenario_name, returns_df) -> list[int] | None` — returns month counts for recovery buckets, e.g. `[12, 12, 12, 7]` for a 3y7m displayed recovery horizon.
`recovery_horizon_label(scenario_name, returns_df) -> str | None` — human-readable latest-trust recovery horizon.

### Stores
- `m4-shocked-store`: `list[float]` — 11 Year 1 asset returns in decimals
- `m4-path-store`: `{"years": {str(year_offset): list[float]}, "scenario_name": str}` — full multi-year asset return path (note: nested under `"years"` key, NOT flat)

### Key Helpers
- `_scenario_defaults(name, cma_baseline)` → Year 1 shock (shock table + compare chart)
- `_scenario_trust_net_path(name, cma_returns)` → `{year_offset: {trust: net_return}}` — crisis-only raw historical helper used internally by `_full_scenario_trust_path`
- `_full_scenario_trust_path(scenario_name, cma_returns, selected_trust_nets)` → `{year_offset: {trust: net_return}}` — **delta-adjusted crisis + recovery; consumed by Modules 4 stress sim, 5, 6, 7 and 8**
- `_m4_stress_value_figure(bau, stressed, stress_onset)` → two-line BAU vs stressed value chart with vertical onset marker
- `_delta_color_rules(column_id)` → list of Dash DataTable conditional styles (teal positive, plum negative); `_DELTA_STYLES = _delta_color_rules("delta") + _delta_color_rules("recovery_delta")`

### Scenario asset class returns table (`m4-shock-table`)
Columns: Asset Class | CMA Baseline (%) | Crisis Return (%) | Crisis Delta (pp) | Recovery Return (%) | Recovery Delta (pp)
- "Crisis Delta" replaces former "Delta (%)"
- "Recovery Return" and "Recovery Delta" show the delta-adjusted recovery window return vs CMA
- `m4-shock-table-note` div — shows crisis window and recovery window date labels (populated in `update_m4_scenario` callback)

### Portfolio simulation — stress only (`update_m4_stress_simulation`)
New callback added. Inputs: `m4-path-store`, `portfolio-allocation-store`, `cma-store`, `m4-stress-onset` (Year 1–9), `m4-recovery-rebalance`, M1 period States.

Runs two projections (no drought schedule, no drawdowns):
1. **BAU** — CMA returns throughout
2. **Stressed** — `_full_scenario_trust_path` overrides shifted to start at `stress_onset`; when enabled, recovery-start rebalance trades back to the Module 3 initial allocation at the end of the final crisis bucket so the reset mix earns the recovery phase

Outputs:
- `m4-sim-value-chart` — BAU vs stressed value trajectory chart
- `m4-sim-composition-chart` — stacked area trust composition (stressed path)
- `m4-sim-table-container` — `_projection_summary_table(stressed, table_id="m4-sim-projection-table")` with config note
- `m4-sim-totals` — final value, vs-BAU delta (green/red), spread cost, liquidity breach count
- `m4-sim-return-summary` — `_master_fund_return_table` for stressed path

Liquidity check is the same hard all-years gate (STI ≥ 10%, STI+MTG ≥ 25%) as Modules 5/6/8.

---

## Module 5 — Drought First

Two-branch view: BAU + drought with post-drought rebalancing, then branching into (a) BAU forward and (b) a late-horizon stress test.

### Design
```
Base projection → drought drawdown → rebalance to new allocation
                                          ├─ Branch (a): BAU forward to year 10
                                          └─ Branch (b): multi-year stress at year N, then CMA
```

### Rebalancing timing — CRITICAL
Within each projection year the engine applies (in order):
1. **Growth** — holdings compound at full-year trust returns
2. **Drawdown** — drought redemption taken from the grown (pre-rebalance) portfolio
3. **Rebalance** — trades on the post-drawdown portfolio at year-end

The minimum rebalance year is **`onset`**. Rebalancing in a drought year means drawdown is taken first, then the portfolio is rebalanced from the remaining balance.

`YearState.pre_rebalance_weights` — post-drawdown, pre-rebalance composition. Used by Module 5/6 drift panels to show the drifted position before rebalancing. This replaced the old use of `ending_weights` for drift display.

### Controls
- Severity / total relief / onset year / year-onset fraction
- **Onset drawdown split** (`m5-onset-split-STI/MTG/LTG`): auto-populated from actual compounded pre-drawdown balances using the STI → MTG → LTG sequential redemption rule. `m5-predrawdown-balances` div shows 3 drought years with [fully drawn / partial / untouched] tags.
- **Post-drought rebalancing panel** (`_rebalancing_controls(onset)`):
  - Rebalance year input — `min=onset`, default = `min(onset+3, 9)`. Sub-label: "Occurs at year-end: after growth, after that year's drawdown."
  - New strategic allocation — auto-sums to 100, LTG ≤ 50% (`rebalance_m5_reb` callback)
  - Liquidity constraint checker + drifted weights display
  - Board Policy compliance table
  - Stress scenario: read-only label "Set in Module 4" + stress onset year (`m5-stress-year`)

### Year-bound enforcement callbacks
- `sync_m5_year_bounds(onset)`: enforces `rebalance_year ≥ onset`; `stress_year > rebalance_year`. Fires on onset change.
- `sync_m6_year_bounds(onset)`: same minimum for M6 rebalance year.

### Layout Panels (order)
1. Controls + config summary
2. Portfolio value trajectory (BAU)
3. Year-onset outcome card + pre-drawdown balance panel
4. Post-drought rebalancing panel
5. Master fund return summary + BAU/Stress toggle near the top of the workflow
6. Branch comparison chart (BAU teal / crisis orange vrect / recovery green vrect / stress dashed)
7. Trust composition over time (same toggle)
8. Year-by-year summary table (same toggle)

### Three projections in `update_module_5`
1. `result` — base BAU + drought, no rebalance
2. `bau_branch` — BAU + drought + rebalance (Branch a)
3. `stress_result` — same as (2) + full crisis+recovery path starting at `stress_year` (Branch b)

`update_module_5` reads Module 1 analysis period as `State` (not `Input`) to compute `selected_trust_nets` for the delta approach. Passes `stress_n_crisis` and `stress_n_recovery` to `_branching_value_figure` and `_master_fund_return_table`.

### `_projection_summary_table(result, table_id)`
Always pass explicit `table_id` to avoid duplicate ID bugs when called from multiple modules.
- M4 stress sim: `table_id="m4-sim-projection-table"`
- M5: `table_id="m5-projection-table"` (default)
- M6: `table_id="m6-projection-table"`

Liquidity columns (Liq 12m, Liq 3y) highlight red for ANY year below the Board Policy floor — there is no pre-rebalance or drought-year exemption. The `yrs_breach` count in the totals bar reflects all years.

---

## Module 6 — Combined Stress (Market Crash + Drought)

Stacks a multi-year market crash onto the drought simultaneously.

### Design
```
stressed path: growth + crash overlay + drought drawdown
                       ↓ rebalance_year
rebalanced path: BAU recovery (no second stress)
```

### Three projections in `update_module_6`
1. `baseline` — drought-only BAU (reference; no crash)
2. `stressed` — combined crash + drought, no rebalancing
3. `rebalanced` — combined crash + drought → rebalance → BAU recovery

`update_module_6` reads Module 1 analysis period as `State` to compute `selected_trust_nets`. Uses `_full_scenario_trust_path` (crisis + recovery) for the stress override. Config text shows `"N crisis year(s) + M recovery bucket(s) (X years Y months) applied"` when recovery exists.

### Controls (`_m6_rebalancing_controls(onset)`)
- Rebalance year (`m6-rebalance-year`) — `min=onset`, default = `min(onset+3, 9)`. Same year-end note as M5.
- New STI / MTG / LTG % allocation — auto-sums to 100 (`rebalance_m6_reb` callback)
- Inherits drought params from Module 5 (same onset, severity, relief, fraction, onset split)

### Drawdown profile (`_m6_drawdown_profile(stressed, schedule)`)
Shows actual per-trust redemptions (`y.redemption_amounts`) from the stressed projection for each drought year. Uses [fully drawn / partial / untouched] tags per trust, drawn from `stressed.years[yr]`.

### Panels
1. Combined trajectory chart + joint impact cards
2. Drawdown profile panel (stressed path redemptions)
3. Post-event rebalancing controls
4. Master fund return summary near the top of the recovery view
5. Recovery trajectory chart (3 lines: drought-only / stressed / rebalanced)
6. Year-by-year summary table (`m6-projection-table`)

---

## Module 7 — Executive Summary

Side-by-side summary of both scenarios. Fires on Module 1 period changes and any M5 or M6 input change. Uses `_full_scenario_trust_path` for M5 and M6, so the executive summary matches the full delta-adjusted crisis + recovery paths shown in Modules 4, 5 and 6.

---

## Module 8 — Robust Scenario Optimiser

Searches for a three-decision allocation policy that passes four scenario paths simultaneously.

### Decisions optimised
1. **Initial STI / MTG / LTG allocation** — starting portfolio weights
2. **M5 post-drought rebalance** — allocation after drought + rebalance, tested on BAU continuation and late-horizon stress branch
3. **M6 post-combined-stress rebalance** — allocation after market crash + drought, tested on BAU recovery

The Module 4 stress-only path is also a certified gate, but it is not a fourth decision: its recovery-start rebalance trades back to the initial allocation, so it still depends only on `w0`.

### Pass criterion (all four paths are hard gates)

| Path | Gate |
|------|------|
| M4 stress-only | Non-exhaustion + liquidity (every year) only — return hurdle relaxed (`check_return=False`) |
| M5 BAU | Non-exhaustion + liquidity (every year) only — return hurdle relaxed (`check_return=False`); drought drawdowns suppress the 10Y geometric average in many valid cases |
| M5 stress | Non-exhaustion + liquidity (every year) only — return hurdle relaxed (`check_return=False`) |
| M6 combined stress | Non-exhaustion + liquidity (every year) + return ≥ CPI+2.5% (full gate) |

Liquidity is an **all-years hard constraint** — every simulation year must satisfy STI ≥ 10% and STI+MTG ≥ 25%. Default `liquidity_mode = "all_years"` in both `optimise_three_decision` and the `run_robust_optimiser` callback (fallback also `"all_years"`).

M6 combined stress is the sole **hard return gate**. M4, M5 BAU, and M5 stress are survival + liquidity gates only — all are shown in the path certificate table with Pass (green) / Fail (red) status and included in the worst-path return metric.

The 10Y average is computed as geometric mean of `sum(starting_weights[t] × trust_returns[t])` per year — the weighted net return on the allocation held at the start of each year, before drought redemptions. Consistent with the Master Fund Return Summary table.

### Search algorithm (`robust_optimiser.optimise_three_decision`)
1. Filter all grid allocations to those meeting CPI+2.5% return and Board liquidity floors → `candidates`
2. **Seed injection**: user-configured allocations (M3 initial, M5 reb, M6 reb) are appended to `candidates` as exact weights, bypassing grid rounding. Controlled via `seed_weights` parameter.
3. **Apply M4 stress-only gate** to each candidate initial allocation. No drought is applied. When Module 4 recovery-start rebalance is enabled, the path rebalances back to `w0` at the end of the final crisis bucket, so it remains affected only by `w0`.
4. **Pre-compute M6 best rebalance** for each M4-surviving initial allocation (O(n²) projections, 1 per reb6). M6 recovery only depends on `(w0, reb6)` — independent of M5 — so this pre-pass prunes initials with no feasible M6 before the expensive M5 search. Only passing M6 entries are cached.
5. **Search M5** only for initials where M4 and M6 are feasible (2 projections per reb5: BAU + stress).
6. Master score = `min(m4_stress, m5_bau, m5_stress, m6_recovery).avg_annual_return + 1e-9 × surplus` — worst-case geometric return across all four paths.

### Diagnostics (`RobustOptimisationResult.diagnostics`)
When infeasible, `diagnostics.stages` is a list of per-stage dicts:
- `stage` — search stage label
- `tested` / `passed` / `failed` — candidate counts
- `return_fail` / `liquidity_fail` / `exhaustion_fail` — failure breakdown
- `best_avg_return` / `best_final_value` / `min_post_breaches` — closest-miss metrics
- `return_hurdle` / `liquidity_mode` — settings used

Rendered in app by `_m8_diagnostic_report(result)` as a stage-by-stage table.

### Stores
- `m8-opt-store` — serialised `RobustOptimisationResult.to_dict()` (includes `diagnostics`)

### Key helpers in app.py
| Helper | Purpose |
|--------|---------|
| `_m8_alloc_table(result)` | Three-row table: Initial / M5 reb / M6 reb with weights, return, vol, surplus, liquidity |
| `_m8_path_table(result)` | Four-row path certificate (M4 stress-only, M5 BAU, M5 stress, M6 combined): Pass/Fail badge, 10Y avg return, Y10 value, liquidity breaches, costs |
| `_m8_diagnostic_report(result)` | Stage-by-stage infeasibility breakdown (only shown when infeasible) |
| `_m8_result_view(...)` | Top-level layout: summary cards (worst-path return across all 4 paths), alloc table, path certificate, optional diagnostics |
| `apply_robust_optimiser` | Writes best allocations to `proposed-STI/MTG/LTG`, `m5-reb-*`, `m6-reb-*` |

### Sections
1. **Starting position** — fund value, allocation, trust metrics
2. **Drought configuration** — severity, relief, onset, drawdown schedule
3. **Scenario 1 (M5)** — drought impact, rebalancing (Year N, year-end), Branch (a)/(b) outcomes
4. **Scenario 2 (M6)** — stress overlay detail (per-trust per-year stressed vs CMA return), combined impact, post-event rebalancing (Year N, year-end), recovery outcomes
5. **Comparison table** — 5-column summary across both scenarios

Rebalance year rows display: `"Year N  (year-end: after growth, after drawdown)"`.

---

## Module Logic Connections (`modules/` → `app.py`)

### `trust_calcs.py` (as `tc`)
- `ASSET_CLASSES` (11-element list), `TRUST_NAMES`, `TRUST_RAW_WEIGHTS`, `TRUST_BUY_SPREADS`, `TRUST_SELL_SPREADS`, `TRUST_ONGOING_COSTS`
- `trust_net_return(trust, asset_returns)`, `trust_gross_return`, `trust_volatility`, `trust_sharpe`
- `trust_characteristics(returns, cov, cash, cpi)` → Module 2 cards
- `historical_trust_returns_monthly`, `historical_cumulative_wealth`

### `stress.py` (as `st`)
- `build_crisis_path(name, returns_df, cma_baseline)` → `{int: np.ndarray}` multi-year crisis path
- `_scenario_trust_net_path(name, cma_returns)` → `{year_offset: {trust: net_return}}` — crisis-only raw historical helper used internally by `_full_scenario_trust_path`
- `build_scenario_recovery(scenario_name, cma_trust_nets, returns_df, selected_period_trust_nets)` → `{int: {trust: float}} | None` — delta-adjusted per-trust recovery path; `None` for scenarios without a recovery profile
- `recovery_window_for_scenario(scenario_name)` → `(start_label, end_label) | None` — date range for table column header
- `recovery_chunk_months(scenario_name, returns_df)` and `recovery_horizon_label(scenario_name, returns_df)` → month-level recovery bucket metadata for chart/config labels
- `RECOVERY_PROFILES` dict — per-trust `(trough_date, recovery_date)` for GFC and COVID Inflation Shock (2022)
- Helper functions: `_months_between`, `_advance_months`, `_trust_net_return_for_window`

### `robust_optimiser.py` (as `ro`)
- `optimise_three_decision(asset_returns, cov_matrix, cpi, drought_schedule, onset_split, m5_rebalance_year, m5_stress_overrides, m6_rebalance_year, m6_stress_overrides, m4_stress_overrides=None, m4_rebalance_year=None, ..., liquidity_mode="all_years", seed_weights=None)` → `RobustOptimisationResult`
- `RobustOptimisationResult` fields: `feasible`, `message`, `grid_step`, `candidates_tested`, `score`, `initial`, `module5_rebalance`, `module6_rebalance`, `m4_stress`, `m5_bau`, `m5_stress`, `m6_recovery`, `diagnostics`
- `PathEvaluation` fields: `passed`, `final_value`, `worst_year_value`, `avg_annual_return`, `liquidity_breaches`, `post_rebalance_breaches`, `rebalance_cost`, `spread_cost`, `message`
- Pass criterion — all four paths are hard certified gates:
  - **M4 stress-only**: not exhausted AND `all_years` liquidity only (`check_return=False`) — return hurdle relaxed
  - **M5 BAU**: not exhausted AND `all_years` liquidity only (`check_return=False`) — return hurdle relaxed; drought drawdowns suppress the 10Y geometric average in many valid cases
  - **M5 stress**: not exhausted AND `all_years` liquidity only (`check_return=False`) — return hurdle relaxed
  - **M6 combined stress**: not exhausted AND `all_years` liquidity AND avg_annual_return ≥ CPI+2.5% (full gate)
- `AllocationCandidate` fields: `weights`, `net_return`, `volatility`, `liquidity_12m`, `liquidity_3y`, `return_surplus`
- `_avg_annual_return`: geometric mean of `sum(starting_weights[t] × trust_returns[t])` per year — uses `YearState.starting_weights` and `YearState.trust_returns` (actual rates applied). Consistent with Master Fund Return Summary table.
- `seed_weights`: optional `list[dict[str, float]]` — exact allocations injected into candidate pool (bypass grid rounding). App passes M3 initial + M5 reb + M6 reb configured weights.

### `drought.py` (as `dr`)
- **`project(initial_value, weights, asset_returns, schedule, horizon, drawdown_splits, trust_return_overrides, rebalance_schedule)`**
  - `drawdown_splits`: `{onset_year: {trust: fraction}}` — target apportionment; unfunded spills STI→MTG→LTG
  - `trust_return_overrides`: `{year: {trust: net_return}}` — multi-year crisis overlay
  - `rebalance_schedule`: `{year: {trust: target_weight}}` — triggers end-of-year rebalance (after growth, after drawdown)
- **Intra-year sequence**: Growth → Drawdown → Rebalance (rebalance is post-drawdown, year-end)
- `YearState` fields used in app: `pre_drawdown_value`, `pre_drawdown_weights`, `redemption_amounts`, `rebalance_cost`, `ending_value`, `ending_weights`, `ending_holdings`, `pre_rebalance_weights` — post-drawdown, pre-rebalance trust weights (used by M5/M6 drift panels)
- `build_drought_schedule(onset_year, total_relief, year_4_fraction, residual_split)` → `{year: aud_amount}`
- `post_drawdown_summary(result, onset_year)` → dict for onset card
- `monte_carlo(...)` → Module 5b fan chart

---

## Cross-Module Data Flow
```
cma-store  (returns, vols, corr, cpi — all decimals)
   ├─→ tc.trust_characteristics()          → Module 2 cards / CFO tables
   ├─→ tc.portfolio_net_return()           → Module 3 live metrics / optimiser
   ├─→ _PRECOMPUTED_SCENARIOS              → Module 4 shock table defaults
   ├─→ st.build_crisis_path()              → Module 4 m4-path-store / crisis chart
   ├─→ _scenario_trust_net_path()          → internal crisis-only raw helper
   ├─→ _full_scenario_trust_path()         → Module 4 stress sim, Module 5 stress branch, Module 6 overrides, Module 7, Module 8 (delta crisis + recovery)
   ├─→ dr.project()                        → Module 4 stress sim (2 calls), Module 5 (3 calls), Module 6 (3 calls), Module 7 (6 calls)
   └─→ ro.optimise_three_decision()        → Module 8 grid search (O(n²) M6 + O(n²) M5 projections)

portfolio-allocation-store  ({STI, MTG, LTG} decimals — set by Module 3 or Module 8 apply)
   ├─→ Module 2 CFO Table 3
   ├─→ Module 4 compare chart + crisis path chart + stress simulation
   ├─→ Module 5 all dr.project() calls
   ├─→ Module 6 all dr.project() calls
   └─→ Module 7 dr.project() calls

m1-ignored-flags  ({flag_text: note_text} dict — persists CMA flag dismissals)
   └─→ update_cma_flags (Input — re-renders flags panel immediately on dismiss)
```

---

## Session Tips for Claude
- **Do not read all of `app.py`** — use `grep -n` with section anchors above, then `Read` with `offset` + `limit`.
- `grep -n "def update_module_5\|def update_m4\|def update_module_6\|def update_module_7" app.py` → callback entry points.
- After edits: `python -c "import app"` to check for syntax errors before running. Full run: `python app.py` (auto-reload via `debug=True`). Stale `.pyc` errors: `rm -rf __pycache__ && python app.py`.
- **Git lock files**: sandbox creates `.git/HEAD.lock` and `.git/index.lock` that it cannot delete. If `git push` fails with lock errors, user must run `rm -f .git/HEAD.lock .git/index.lock` from their own terminal first.
- All monetary values in Modules 5, 6, 7 use `_fmt_m()`. Modules 1–4 use `_fmt_pct()` and `_fmt_aud()`.
- `portfolio-allocation-store` values are decimals summing to ~1.0. Always normalise before use.
- `m1-ignored-flags` store format is a **dict** `{flag_text: note_text}`, NOT a list.
- Pattern-matching callbacks use `ALL` imported from `dash`: `from dash import ..., ALL, ...`
- **Rebalancing timing**: `min_reb = onset` in both `sync_m5_year_bounds` and `sync_m6_year_bounds`. The engine sequence is Growth → Drawdown → Rebalance within each year. Drift display uses `YearState.pre_rebalance_weights` (post-drawdown, pre-rebalance), NOT `ending_weights`. Default displayed value is still `min(onset+3, 9)` as a sensible starting point.
- `_projection_summary_table(result, table_id)` — always pass explicit `table_id` to avoid duplicate component ID errors across modules. Current IDs: `"m4-sim-projection-table"`, `"m5-projection-table"` (default), `"m6-projection-table"`.
- **Liquidity constraint**: hard all-years gate everywhere (M4 stress sim, M5, M6, M8). All years including drought drawdown years must pass STI ≥ 10% and STI+MTG ≥ 25%. No pre-rebalance exemption. Default `liquidity_mode="all_years"` in `optimise_three_decision` and both fallback paths in `run_robust_optimiser`.
- **Master Fund Return Summary 10Y Avg**: all columns (gross, net, per-trust contributions) use geometric mean — `(∏(1+annual_value))^(1/n)−1`. Note: per-trust geometric contributions do not sum exactly to geometric net (multi-period attribution is non-additive). Pass/fail uses geometric net, consistent with optimizer.
- **Path certificates**: all four paths are Pass/Fail certified gates. M6 combined stress is the only path with a hard return hurdle; M4, M5 BAU, and M5 stress are survival + liquidity only. `_m8_path_table` lists all four; `worst_avg_ret` in `_m8_result_view` takes min across all four.
- **`m4-path-store` format**: `{"years": {str(yr): list[float]}, "scenario_name": str}` — values are nested under `"years"`, NOT flat. Read as `store["years"]`.
- **Grid lines**: all charts in Modules 4, 5, and 6 use `showgrid=False, zeroline=False` on both axes. Modules 1–3 still retain `gridcolor=COLORS["border"]` on some charts — do not remove without explicit instruction.
- **`_full_scenario_trust_path` vs `_scenario_trust_net_path`**: use `_full_scenario_trust_path` for all M4/M5/M6/M7/M8 stress overrides (includes delta-adjusted crisis + recovery). `_scenario_trust_net_path` is only the raw crisis helper inside that full-path function.
- **M8 seed weights**: `run_robust_optimiser` callback always passes `seed_weights=[m3_initial, m5_reb, m6_reb]` (from allocation store + reb inputs) so user-configured allocations bypass grid rounding and are tested exactly.
- **M4 master scenario**: `update_module_5` uses `Input("m4-scenario","value")` (not `m5-stress-scenario`). `update_module_6` uses `Input("m4-scenario","value")` (not `m6-scenario`). `update_module_7` uses one `Input("m4-scenario","value")` mapped to both `m5_stress_scenario` and `m6_scenario` via `m5_stress_scenario = m4_scenario or "GFC"; m6_scenario = m4_scenario or "GFC"`. `run_robust_optimiser` reads `State("m4-scenario")` and sets `m5_stress_scenario = m4_scenario; m6_scenario = m4_scenario`.
- **`persist_user_state`**: `m4-scenario` is now an `Input`; `m5-stress-scenario` and `m6-scenario` inputs removed. Saved as `"m4_scenario": m4_scenario` at top level (not under `m5` or `m6`). M5 save dict no longer includes `stress_scenario`; M6 save dict no longer includes `scenario`.
- **LTG cap**: `LTG_MAX = 0.50` in `optimiser.py`. Change this one constant to adjust the policy cap globally. All grid search and SLSQP paths enforce it automatically.
