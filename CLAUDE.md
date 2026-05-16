# NSWDF Portfolio Dashboard — Project Context

## Project Brief
NSW Drought Fund (AUD ~3 billion master fund) portfolio allocation dashboard.
Role: Master fund perspective. Objective: meet fund liquidity, returns, risk appetite, and drought response requirements across three unit trusts — STI (Short-Term Income), MTG (Medium-Term Growth), LTG (Long-Term Growth).

## Live File
`Project 2/FINC-3600-main/app.py` — single Dash app, all modules in one file (~8,700 lines).
Run: `cd "Project 2/FINC-3600-main" && python app.py` → http://127.0.0.1:8050

## Directory Structure
```
Project 2/FINC-3600-main/
├── app.py                          # Main Dash app (all 8 modules)
├── requirements.txt                # dash, plotly, pandas, numpy, scipy
├── README.md
├── data/
│   ├── index_returns.csv           # Monthly total returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv        # Monthly macro series Jan 2006–Feb 2026
├── modules/
│   ├── trust_calcs.py              # Core engine: trust weight vectors, gross/net return, vol, Sharpe
│   ├── metrics.py                  # Portfolio metrics helpers
│   ├── optimiser.py                # Grid search + scipy optimisation (LTG_MAX=0.50, TRUST_MIN=0.00)
│   ├── robust_optimiser.py         # Three-decision robust scenario optimiser
│   ├── stress.py                   # Market stress scenario logic + multi-year crisis paths
│   └── drought.py                  # Drought projection engine + post-drought rebalancing
└── CPI Forecast/
    ├── AU CPI Forecast/            # ABS source xlsx + Jupyter notebook
    └── US CPI Forecast/            # FRED source + Jupyter notebook
```

## Quick Navigation in app.py
Use `grep -n` to locate sections rather than reading the whole file.
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

**Current Holdings removed** — transaction cost in the optimiser result card is now computed proposed → optimal. `run_optimiser` uses `alloc` (proposed) as `current_w`.

**LTG cap**: `LTG_MAX = 0.50` defined in `modules/optimiser.py`. Enforced in:
- `generate_grid` (replaces `w_ltg > 1 + 1e-12` with `w_ltg > LTG_MAX + 1e-12`)
- All three SLSQP refiners: added `w_STI + w_MTG >= 1 - LTG_MAX` constraint
- UI: `_alloc_block("proposed", ..., max_ltg=50)`, M4/M5/M6 rebalance inputs `max=50`

**`_alloc_block(block_id, title, note, input_kind, default, max_ltg=100)`**: `max_ltg` param caps LTG slider/input; other trusts remain at 100.

**Per-trust cap toggle** (`trust-cap-store`, `m8-trust-cap-toggle`):
- `dcc.Store(id="trust-cap-store", data=True)` — boolean, True = 50% cap enforced.
- `sync_trust_cap` callback: `Input("m8-trust-cap-toggle") → Output("trust-cap-store")`. Value `"cap"` → `True`, `"nocap"` → `False`.
- `update_trust_cap_limits` callback: 21 outputs — updates `max` AND conditionally clamps `value` for 9 allocation components (`proposed-STI/MTG/LTG`, `m5-reb-STI/MTG/LTG`, `m6-reb-STI/MTG/LTG`, `m4-reb-STI/MTG/LTG`). Uses `_clamp_or_noupdate(v, mx)` — returns `dash.no_update` when value is already within the new max to prevent rebalance cascade.
- `trust_max`: propagated as keyword arg through `generate_grid(step, trust_max)`, `optimise(…, trust_max)`, `sensitivity_sweep(…, trust_max)`, `_candidate_grid(…, trust_max)`, `optimise_three_decision(…, trust_max)`. Defaults to `op.TRUST_MAX` everywhere.
- Rebalance callbacks (`rebalance_proposed`, `rebalance_m5_reb`, `rebalance_m6_reb`): read `State("trust-cap-store")`, pass `cap = 50 if cap_on else 100` to `_rebalance_other_two`.
- `apply_robust_optimiser`: clamps written values to `min(round(…*100), 50)` only (optimizer already respects `trust_max` internally).

### Panels
1. Proposed Allocation sliders with auto-rebalance (LTG ≤ 50%)
2. Live metrics: Net Return, Vol, Sharpe + constraint pills
3. Optimiser — `max_sharpe` / `min_vol` / `max_return`, vol cap → result card + transaction cost (proposed → optimal)
4. Feasible portfolio scatter (proposed + optimal markers; no current marker)
5. Sensitivity sweep tornado
6. Board Policy compliance table (`_board_compliance_table`) — includes domicile and asset type Info rows

### `update_live` callback
Inputs: `proposed-STI/MTG/LTG`, `cma-store`, `m3-objective`.

### Stores
- `m3-opt-store`: serialised `OptimisationResult` dict
- `portfolio-allocation-store`: `{STI, MTG, LTG}` decimals

---

## Module 4 — Market Stress Testing

**Master scenario selector**: `m4-scenario` dropdown is the single source of scenario selection for the entire app. Modules 5, 6, 7, and 8 all read from `m4-scenario`. The `m5-stress-scenario` and `m6-scenario` dropdowns have been removed. `persist_user_state` saves `m4_scenario` at top level. `m4-scenario` default loads from `_SAVED.get("m4_scenario", "GFC")`.

**Recovery return floor**: `stress.build_scenario_recovery` floors recovery rates at CMA:
`annual_rate = cma + max(0.0, hist - selected_period)` — i.e. `max(cma, cma + delta)`.

### Portfolio simulation — stress only (`update_m4_stress_simulation`)
Inputs: `m4-path-store`, `portfolio-allocation-store`, `cma-store`, `m4-stress-onset` (Year 1–9), `m4-reb-year/STI/MTG/LTG`, M1 period States.

**Recovery-start rebalance removed.** Runs three projections:
1. **BAU** — CMA returns throughout
2. **Stressed (no rebalance)** — stress overrides, no rebalancing at all
3. **Stressed + rebalanced** — stress overrides + strategic rebalance at `m4-reb-year`

Chart (`_m4_stress_value_figure`) shows all three lines simultaneously so the user can see the direct benefit of rebalancing. The `stressed` line is always clean (no recovery-start rebalance). Composition chart and return summary reflect the rebalanced path when configured, otherwise the stressed path.

`_stress_recovery_rebalance_year` helper deleted. `m4-recovery-rebalance` Checklist removed.

### Scenarios (`SCENARIO_ORDER`)
`GFC` | `COVID Crash` | `COVID Inflation Shock (2022)` | `AUD Depreciation Shock` | `Interest Rate Shock (+200bps)`

### Delta approach (universal)
All stressed returns — crisis AND recovery — use:
`stressed_return = CMA_baseline + (historical_scenario_return − selected_period_historical_return)`
Recovery is additionally floored at CMA baseline.
Applied consistently to: crisis path chart, shock table, M4 stress simulation, Modules 5/6 projections, Module 7 summary, and Module 8 optimiser gates.

### Multi-year crisis path (`stress.build_crisis_path`)
Returns `{year_offset: asset_returns_array}`:
- GFC (21 months): 2 years
- COVID Crash (2 months): 1 year (cumulative, not annualised)
- COVID Inflation 2022: 1 year
- AUD Depreciation: 1 year (worst rolling 12m window)
- Rate Shock: 2 years (Y1 = CMA − duration×0.02, Y2 = 50% reversion)

### Stores
- `m4-shocked-store`: `list[float]` — 11 Year 1 asset returns in decimals
- `m4-path-store`: `{"years": {str(year_offset): list[float]}, "scenario_name": str}` — full multi-year asset return path (note: nested under `"years"` key, NOT flat)

### Key Helpers
- `_full_scenario_trust_path(scenario_name, cma_returns, selected_trust_nets)` → `{year_offset: {trust: net_return}}` — **delta-adjusted crisis + recovery; consumed by Modules 4 stress sim, 5, 6, 7 and 8**
- `_m4_stress_value_figure(bau, stressed, stress_onset, rebalanced=None, reb_year=None)` → three-line chart

**M4 rebalance year persistence**: `m4-reb-year` saved to `user_state.json` under `"m4": {"reb_year": …}`.

---

## Module 5 — Drought First

Two-branch view: BAU + drought with post-drought rebalancing, then branching into (a) BAU forward and (b) a late-horizon stress test.

### Design
```
Base projection → drought drawdown → rebalance (year-end, after drawdown)
                                          ├─ Branch (a): BAU forward to year 10
                                          └─ Branch (b): multi-year stress at year N, then CMA
```

### Rebalancing timing — CRITICAL
Within each projection year the engine applies (in order):
1. **Growth** — holdings compound at full-year trust returns
2. **Drawdown** — drought redemption taken from the grown (pre-rebalance) portfolio
3. **Rebalance** — trades on the post-drawdown portfolio at year-end

The minimum rebalance year is **`onset`**. Sub-label in UI: "Occurs at year-end: after growth, after that year's drawdown."

`YearState.pre_rebalance_weights` — post-drawdown, pre-rebalance composition. Used by Module 5/6 drift panels to show the drifted position before rebalancing.

### Controls
- Severity / total relief / onset year / year-onset fraction
- **Onset drawdown split** (`m5-onset-split-STI/MTG/LTG`): auto-populated from actual compounded pre-drawdown balances using the STI → MTG → LTG sequential redemption rule.
- **Post-drought rebalancing panel** (`_rebalancing_controls(onset)`):
  - Rebalance year input — `min=onset`, default = `min(onset+3, 9)`. Sub-label: "Occurs at year-end: after growth, after that year's drawdown."
  - New strategic allocation — auto-sums to 100, LTG ≤ 50% (`rebalance_m5_reb` callback)
  - Liquidity constraint checker + drifted weights display
  - Board Policy compliance table
  - Stress scenario: read-only label "Set in Module 4" + stress onset year (`m5-stress-year`)

### Year-bound enforcement callbacks
- `sync_m5_year_bounds(onset)`: enforces `rebalance_year ≥ onset`; `stress_year > rebalance_year`. Fires on onset change.
- `sync_m6_year_bounds(onset)`: same minimum for M6 rebalance year.

### Three projections in `update_module_5`
1. `result` — base BAU + drought, no rebalance
2. `bau_branch` — BAU + drought + rebalance (Branch a)
3. `stress_result` — same as (2) + full crisis+recovery path starting at `stress_year` (Branch b)

### `_master_fund_return_table` — Post-reb Avg row
Added a **Post-reb Avg (YN–10)** summary row:
- Shown only when `rebalance_year is not None` and at least one post-rebalance year exists.
- Accumulates separate `post_gross_factor / post_net_factor / post_contrib_factor` for years `yr > rebalance_year`.
- Events column shows: `"New alloc: STI X% / MTG Y% / LTG Z%"`.
- Styled with tinted green background + italic.
- Applies to M4, M5, and M6 return summaries.

### `_projection_summary_table(result, table_id)`
Always pass explicit `table_id` to avoid duplicate ID bugs. Current IDs: `"m4-sim-projection-table"`, `"m5-projection-table"` (default), `"m6-projection-table"`.

---

## Module 6 — Combined Stress (Market Crash + Drought)

Stacks a multi-year market crash onto the drought simultaneously.

### Design
```
stressed path: growth + crash overlay + drought drawdown
                       ↓ rebalance_year (year-end, after drawdown)
rebalanced path: BAU recovery (no second stress)
```

### Controls (`_m6_rebalancing_controls(onset)`)
- Rebalance year (`m6-rebalance-year`) — `min=onset`, default = `min(onset+3, 9)`. Sub-label: "Occurs at year-end: after growth, after that year's drawdown."
- Inherits drought params from Module 5 (same onset, severity, relief, fraction, onset split)

---

## Module 7 — Executive Summary

Side-by-side summary of both scenarios. Fires on Module 1 period changes and any M5 or M6 input change. Uses `_full_scenario_trust_path` for M5 and M6.

---

## Module 8 — Robust Scenario Optimiser

Searches for a three-decision allocation policy that passes certified scenario paths simultaneously.

### Decisions optimised
1. **Initial STI / MTG / LTG allocation** — starting portfolio weights
2. **M5 post-drought rebalance** — allocation after drought + rebalance
3. **M6 post-combined-stress rebalance** — allocation after market crash + drought

### Pass criterion

| Path | Gate |
|------|------|
| M4 stress-only | Non-exhaustion + liquidity only — return hurdle relaxed (`check_return=False`) |
| M5 BAU | Non-exhaustion + liquidity only — return hurdle relaxed (`check_return=False`) |
| M5 stress (if included) | Non-exhaustion + liquidity (soft) or + return ≥ CPI+2.5% (hard) |
| M6 combined stress | Non-exhaustion + liquidity + return ≥ CPI+2.5% (full gate) |

Liquidity is an **all-years hard constraint** — every simulation year must satisfy STI ≥ 10% and STI+MTG ≥ 25%.

### Module 8 controls
- **`m8-trust-cap-toggle`** (RadioItems): `"cap"` (default) = enforce 50% per-trust cap; `"nocap"` = no per-trust cap.
- **`m8-trust-min-select`** (RadioItems): diversification floor applied to all three trusts — `"0.05"` / `"0.10"` / `"0.15"`. Parsed as `float`. Default `"0.05"`. Passed as `trust_min` to `generate_grid` and all SLSQP refiners.
- **`m8-include-m5-stress`** (RadioItems): `"include"` (default) = M5 stress branch is a certified gate; `"exclude"` = M5 stress projection skipped entirely, only M5 BAU is evaluated. Passed as `include_m5_stress=(value != "exclude")` to `optimise_three_decision`.
- **`m8-m5-pass-mode`** (RadioItems): `"soft"` (default) = M5 stress survival + liquidity only; `"hard"` = also requires return ≥ CPI+2.5%. Only active when M5 stress is included.
- **`m8-constraints-summary`** (reactive panel): live bullet list of all active constraints, grouped into Hard allocation constraints / Scenario path gates / Search settings. Callback `update_m8_constraints_summary` fires on any control change.

### Search algorithm (`robust_optimiser.optimise_three_decision`)
**Seed weights removed** — all candidates come exclusively from `_candidate_grid`, which filters by `trust_min`, `trust_max`, and return hurdle. No user-configured seeds bypass the grid.

Signature (key params):
```python
optimise_three_decision(
    ...,
    liquidity_mode="all_years",
    trust_max=op.TRUST_MAX,
    trust_min=op.TRUST_MIN,
    include_m5_stress=True,
    m5_stress_check_return=False,
) -> RobustOptimisationResult
```

Search order:
1. Filter grid allocations to those meeting `trust_min`, `trust_max`, liquidity floors, and CPI+2.5% return → `candidates`
2. **Apply M4 stress-only gate** to each candidate initial allocation.
3. **Pre-compute M6 best rebalance** for each M4-surviving initial (O(n²) projections). Prunes initials with no feasible M6 before the expensive M5 search.
4. **Search M5**: for each surviving initial, evaluate each reb5 candidate. If `include_m5_stress=True`, runs both BAU and stress projections and gates on both. If `include_m5_stress=False`, runs BAU only.
5. Master score = worst-case geometric return across all certified paths + 1e-9 × surplus.

### `optimiser.py` — constraint parameters
- `TRUST_MAX = 0.50` — Board policy cap (per trust)
- `TRUST_MIN = 0.00` — Default diversification floor (no floor)
- `generate_grid(step, trust_max, trust_min)`: STI effective floor = `max(LIQUIDITY_12M_MIN, trust_min)`. MTG and LTG each floored at `trust_min`. w_LTG floor replaces the old `w_ltg < -1e-12` check.
- All three SLSQP refiners (`_refine_max_sharpe`, `_refine_min_vol`, `_refine_max_return`): lower bounds changed to `(max(0.0, trust_min), trust_max)` for STI and MTG; LTG min constraint `1 - trust_min - w[0] - w[1] >= 0` replaces old `1 - w[0] - w[1] >= 0`.
- `optimise(…, trust_min)` and `sensitivity_sweep(…, trust_min)` accept and propagate `trust_min`.

### Diagnostics (`RobustOptimisationResult.diagnostics`)
When infeasible, `diagnostics.stages` is a list of per-stage dicts. Rendered by `_m8_diagnostic_report(result)` as a stage-by-stage table.

### Key helpers in app.py
| Helper | Purpose |
|--------|---------|
| `_m8_alloc_table(result)` | Three-row table: Initial / M5 reb / M6 reb with weights, return, vol, surplus, liquidity |
| `_m8_path_table(result)` | Path certificate (M4, M5 BAU, M5 stress if included, M6): Pass/Fail, 10Y avg return, Y10 value, liquidity breaches |
| `_m8_diagnostic_report(result)` | Stage-by-stage infeasibility breakdown (only shown when infeasible) |
| `_m8_result_view(...)` | Top-level layout: summary cards, alloc table, path certificate, optional diagnostics |
| `update_m8_constraints_summary` | Live bullet panel: hard constraints / path gates / search settings — updates on any control change |
| `apply_robust_optimiser` | Writes best allocations to `proposed-STI/MTG/LTG`, `m5-reb-*`, `m6-reb-*` |

---

## Module Logic Connections (`modules/` → `app.py`)

### `trust_calcs.py` (as `tc`)
- `ASSET_CLASSES` (11-element list), `TRUST_NAMES`, `TRUST_RAW_WEIGHTS`, `TRUST_BUY_SPREADS`, `TRUST_SELL_SPREADS`, `TRUST_ONGOING_COSTS`
- `trust_net_return(trust, asset_returns)`, `trust_gross_return`, `trust_volatility`, `trust_sharpe`
- `trust_characteristics(returns, cov, cash, cpi)` → Module 2 cards
- `historical_trust_returns_monthly`, `historical_cumulative_wealth`

### `stress.py` (as `st`)
- `build_crisis_path(name, returns_df, cma_baseline)` → `{int: np.ndarray}` multi-year crisis path
- `build_scenario_recovery(scenario_name, cma_trust_nets, returns_df, selected_period_trust_nets)` → `{int: {trust: float}} | None`
- `RECOVERY_PROFILES` dict — per-trust `(trough_date, recovery_date)` for GFC and COVID Inflation Shock (2022)
- Helper functions: `_months_between`, `_advance_months`, `_trust_net_return_for_window`

### `robust_optimiser.py` (as `ro`)
- `optimise_three_decision(…, trust_max, trust_min, include_m5_stress, m5_stress_check_return)` → `RobustOptimisationResult`
- `RobustOptimisationResult` fields: `feasible`, `message`, `grid_step`, `candidates_tested`, `score`, `initial`, `module5_rebalance`, `module6_rebalance`, `m4_stress`, `m5_bau`, `m5_stress` (None when excluded), `m6_recovery`, `diagnostics`
- `PathEvaluation` fields: `passed`, `final_value`, `worst_year_value`, `avg_annual_return`, `liquidity_breaches`, `post_rebalance_breaches`, `rebalance_cost`, `spread_cost`, `message`
- `AllocationCandidate` fields: `weights`, `net_return`, `volatility`, `liquidity_12m`, `liquidity_3y`, `return_surplus`

### `drought.py` (as `dr`)
- **`project(initial_value, weights, asset_returns, schedule, horizon, drawdown_splits, trust_return_overrides, rebalance_schedule)`**
  - `drawdown_splits`: `{onset_year: {trust: fraction}}` — target apportionment; unfunded spills STI→MTG→LTG
  - `trust_return_overrides`: `{year: {trust: net_return}}` — multi-year crisis overlay
  - `rebalance_schedule`: `{year: {trust: target_weight}}` — triggers end-of-year rebalance (after growth, after drawdown)
- **Intra-year sequence**: Growth → Drawdown → Rebalance (rebalance is post-drawdown, year-end)
- `YearState` fields used in app: `pre_drawdown_value`, `pre_drawdown_weights`, `redemption_amounts`, `rebalance_cost`, `ending_value`, `ending_weights`, `ending_holdings`, `pre_rebalance_weights`
- `build_drought_schedule(onset_year, total_relief, year_4_fraction, residual_split)` → `{year: aud_amount}`
- `post_drawdown_summary(result, onset_year)` → dict for onset card
- `monte_carlo(...)` → Module 5b fan chart

---

## Cross-Module Data Flow
```
cma-store  (returns, vols, corr, cpi — all decimals)
   ├─→ tc.trust_characteristics()          → Module 2 cards / CFO tables
   ├─→ tc.portfolio_net_return()           → Module 3 live metrics / optimiser
   ├─→ st.build_crisis_path()              → Module 4 m4-path-store / crisis chart
   ├─→ _full_scenario_trust_path()         → Module 4 stress sim, Module 5 stress branch, Module 6 overrides, Module 7, Module 8
   ├─→ dr.project()                        → Module 4 stress sim (3 calls), Module 5 (3 calls), Module 6 (3 calls), Module 7 (6 calls)
   └─→ ro.optimise_three_decision()        → Module 8 grid search

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
- **Rebalancing timing**: engine sequence is Growth → Drawdown → Rebalance within each year. `min_reb = onset` in both `sync_m5_year_bounds` and `sync_m6_year_bounds`. UI sub-label: "Occurs at year-end: after growth, after that year's drawdown." Drift display uses `YearState.pre_rebalance_weights` (post-drawdown, pre-rebalance), NOT `ending_weights`.
- `_projection_summary_table(result, table_id)` — always pass explicit `table_id` to avoid duplicate component ID errors across modules.
- **Liquidity constraint**: hard all-years gate everywhere (M4 stress sim, M5, M6, M8). All years including drought drawdown years must pass STI ≥ 10% and STI+MTG ≥ 25%. No pre-rebalance exemption.
- **Master Fund Return Summary 10Y Avg**: all columns use geometric mean — `(∏(1+annual_value))^(1/n)−1`. Pass/fail uses geometric net.
- **`m4-path-store` format**: `{"years": {str(yr): list[float]}, "scenario_name": str}` — values nested under `"years"`, NOT flat.
- **Grid lines**: all charts in Modules 4, 5, and 6 use `showgrid=False, zeroline=False` on both axes.
- **`_full_scenario_trust_path` vs `_scenario_trust_net_path`**: use `_full_scenario_trust_path` for all M4/M5/M6/M7/M8 stress overrides. `_scenario_trust_net_path` is only the raw crisis helper inside that full-path function.
- **M4 stress sim**: no recovery-start rebalance. `stressed` projection = no rebalancing at all. `rebalanced` = strategic rebalance only at `m4-reb-year`. Chart always shows BAU / Stressed (no rebalance) / Stressed + rebalanced.
- **M8 seed weights**: removed. All candidates come from `_candidate_grid` filtered by `trust_min`, `trust_max`, and return hurdle. Seeds previously injected from M3/M5/M6 inputs no longer used.
- **`trust_min`**: set via `m8-trust-min-select` (RadioItems, values `"0.05"/"0.10"/"0.15"`). Parsed with `float(trust_min_select or "0.05")`. Applied in `generate_grid` as STI floor = `max(LIQUIDITY_12M_MIN, trust_min)`, MTG floor = `trust_min`, LTG floor = `trust_min`. SLSQP bounds updated accordingly.
- **M5 stress toggle**: `m8-include-m5-stress` RadioItems (`"include"`/`"exclude"`). When `"exclude"`, `include_m5_stress=False` is passed and the stress `dr.project` call is skipped entirely — only `bau_eval` is computed per reb5, and `m5_stress` in the result is `None`.
- **M4 master scenario**: `update_module_5` uses `Input("m4-scenario","value")`. `update_module_6` uses `Input("m4-scenario","value")`. `update_module_7` maps one input to both m5/m6 scenario vars. `run_robust_optimiser` reads `State("m4-scenario")`.
- **LTG cap**: `LTG_MAX = 0.50` in `optimiser.py`. Change this one constant to adjust the policy cap globally.
- **`_master_fund_return_table` Post-reb Avg row**: appears when `rebalance_year is not None` and post-rebalance years exist. Uses `y.starting_weights` from the projection result.
