# NSWDF Portfolio Dashboard — Project Context

## Project Brief
NSW Drought Fund (AUD ~3 billion master fund) portfolio allocation dashboard.
Role: Master fund perspective. Objective: meet fund liquidity, returns, risk appetite, and drought response requirements across three unit trusts — STI (Short-Term Income), MTG (Medium-Term Growth), LTG (Long-Term Growth).

## Live File
`Project 2/FINC-3600-main/app.py` — single Dash app, all modules in one file (~6,200 lines).
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

### Panels
1. Current Holdings + Proposed Allocation sliders with auto-rebalance
2. Live metrics: Net Return, Vol, Sharpe + constraint pills
3. Optimiser — `max_sharpe` / `min_vol` / `max_return`, vol cap → result card + transaction cost
4. Feasible portfolio scatter
5. Sensitivity sweep tornado
6. Board Policy compliance table (`_board_compliance_table`) — includes domicile and asset type Info rows

### Stores
- `m3-opt-store`: serialised `OptimisationResult` dict
- `portfolio-allocation-store`: `{STI, MTG, LTG}` decimals

---

## Module 4 — Market Stress Testing

### Scenarios (`SCENARIO_ORDER`)
`GFC` | `COVID Crash` | `COVID Inflation Shock (2022)` | `AUD Depreciation Shock` | `Interest Rate Shock (+200bps)`

### Multi-year crisis path (`stress.build_crisis_path`)
Returns `{year_offset: asset_returns_array}`:
- GFC (21 months): 2 years
- COVID Crash (2 months): 1 year (cumulative, not annualised)
- COVID Inflation 2022: 1 year
- AUD Depreciation: 1 year (worst rolling 12m window)
- Rate Shock: 2 years (Y1 = CMA − duration×0.02, Y2 = 50% reversion)

### Stores
- `m4-shocked-store`: `list[float]` — 11 Year 1 asset returns in decimals
- `m4-path-store`: `{str(year_offset): list[float]}` — full multi-year asset return path

### Key Helpers
- `_scenario_defaults(name, cma_baseline)` → Year 1 shock (shock table + compare chart)
- `_scenario_trust_net_path(name, cma_returns)` → `{year_offset: {trust: net_return}}` — **consumed by Modules 5 and 6**

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
2. **Rebalance** — trades on the post-growth, pre-drawdown portfolio
3. **Drawdown** — drought redemption from the rebalanced holdings

This means rebalancing is **end-of-year on the grown portfolio**. The minimum rebalance year is **`onset`** (no lower bound beyond the drought start year). Rebalancing in a drought year means the portfolio is rebalanced first, then the drawdown is taken from the new weights.

### Controls
- Severity / total relief / onset year / year-onset fraction
- **Onset drawdown split** (`m5-onset-split-STI/MTG/LTG`): auto-populated from actual compounded pre-drawdown balances using the STI → MTG → LTG sequential redemption rule. `m5-predrawdown-balances` div shows 3 drought years with [fully drawn / partial / untouched] tags.
- **Post-drought rebalancing panel** (`_rebalancing_controls(onset)`):
  - Rebalance year input — `min=onset`, default = `min(onset+3, 9)`. Sub-label: "Occurs at year-end: after growth, before that year's drawdown."
  - New strategic allocation — auto-sums to 100 (`rebalance_m5_reb` callback)
  - Liquidity constraint checker + drifted weights display
  - Board Policy compliance table
  - Stress scenario dropdown + stress onset year

### Year-bound enforcement callbacks
- `sync_m5_year_bounds(onset)`: enforces `rebalance_year ≥ onset`; `stress_year > rebalance_year`. Fires on onset change.
- `sync_m6_year_bounds(onset)`: same minimum for M6 rebalance year.

### Layout Panels (order)
1. Controls + config summary
2. Portfolio value trajectory (BAU)
3. Year-onset outcome card + pre-drawdown balance panel
4. Post-drought rebalancing panel
5. Branch comparison chart (BAU teal vs stress orange dashed)
6. Trust composition over time (BAU/Stress toggle)
7. Year-by-year summary table (same toggle)
8. Master fund return summary (same toggle)

### Three projections in `update_module_5`
1. `result` — base BAU + drought, no rebalance
2. `bau_branch` — BAU + drought + rebalance (Branch a)
3. `stress_result` — same as (2) + multi-year stress starting at `stress_year` (Branch b)

### `_projection_summary_table(result, table_id)`
Always pass explicit `table_id` to avoid duplicate ID bugs when called from multiple modules.
- M5: `table_id="m5-projection-table"` (default)
- M6: `table_id="m6-projection-table"`

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

### Controls (`_m6_rebalancing_controls(onset)`)
- Rebalance year (`m6-rebalance-year`) — `min=onset`, default = `min(onset+3, 9)`. Same year-end note as M5.
- New STI / MTG / LTG % allocation — auto-sums to 100 (`rebalance_m6_reb` callback)
- Inherits drought params from Module 5 (same onset, severity, relief, fraction, onset split)

### Drawdown profile (`_m6_drawdown_profile(stressed, schedule)`)
Shows actual per-trust redemptions (`y.redemption_amounts`) from the stressed projection for each drought year. Uses [fully drawn / partial / untouched] tags per trust, drawn from `stressed.years[yr]`.

### Panels
1. Recovery trajectory chart (3 lines: drought-only / stressed / rebalanced)
2. Drawdown profile panel (stressed path redemptions)
3. Post-event rebalancing controls
4. Year-by-year summary table (`m6-projection-table`)
5. Return summary + joint impact cards

---

## Module 7 — Executive Summary

Side-by-side summary of both scenarios. Fires on any M5 or M6 input change.

### Sections
1. **Starting position** — fund value, allocation, trust metrics
2. **Drought configuration** — severity, relief, onset, drawdown schedule
3. **Scenario 1 (M5)** — drought impact, rebalancing (Year N, year-end), Branch (a)/(b) outcomes
4. **Scenario 2 (M6)** — stress overlay detail (per-trust per-year stressed vs CMA return), combined impact, post-event rebalancing (Year N, year-end), recovery outcomes
5. **Comparison table** — 5-column summary across both scenarios

Rebalance year rows display: `"Year N  (year-end: after growth, before drawdown)"`.

---

## Module Logic Connections (`modules/` → `app.py`)

### `trust_calcs.py` (as `tc`)
- `ASSET_CLASSES` (11-element list), `TRUST_NAMES`, `TRUST_RAW_WEIGHTS`, `TRUST_BUY_SPREADS`, `TRUST_SELL_SPREADS`, `TRUST_ONGOING_COSTS`
- `trust_net_return(trust, asset_returns)`, `trust_gross_return`, `trust_volatility`, `trust_sharpe`
- `trust_characteristics(returns, cov, cash, cpi)` → Module 2 cards
- `historical_trust_returns_monthly`, `historical_cumulative_wealth`

### `stress.py` (as `st`)
- `build_crisis_path(name, returns_df, cma_baseline)` → `{int: np.ndarray}` multi-year path
- **`_scenario_trust_net_path(name, cma_returns)`** → `{year_offset: {trust: net_return}}` — consumed by M5 stress branch and M6 overrides

### `drought.py` (as `dr`)
- **`project(initial_value, weights, asset_returns, schedule, horizon, drawdown_splits, trust_return_overrides, rebalance_schedule)`**
  - `drawdown_splits`: `{onset_year: {trust: fraction}}` — target apportionment; unfunded spills STI→MTG→LTG
  - `trust_return_overrides`: `{year: {trust: net_return}}` — multi-year crisis overlay
  - `rebalance_schedule`: `{year: {trust: target_weight}}` — triggers end-of-year rebalance (after growth, before drawdown)
- **Intra-year sequence**: Growth → Rebalance → Drawdown (engine comment: "Happens after growth, before any drought drawdown")
- `YearState` fields used in app: `pre_drawdown_value`, `pre_drawdown_weights`, `redemption_amounts`, `rebalance_cost`, `ending_value`, `ending_weights`, `ending_holdings`
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
   ├─→ _scenario_trust_net_path()          → Module 5 stress branch, Module 6 overrides
   └─→ dr.project()                        → Module 5 (3 calls), Module 6 (3 calls), Module 7 (6 calls)

portfolio-allocation-store  ({STI, MTG, LTG} decimals — set by Module 3)
   ├─→ Module 2 CFO Table 3
   ├─→ Module 4 compare chart + crisis path chart
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
- **Rebalancing timing**: `min_reb = onset` in both `sync_m5_year_bounds` and `sync_m6_year_bounds`. The engine sequence is Growth → Rebalance → Drawdown within each year. Default displayed value is still `min(onset+3, 9)` as a sensible starting point.
- `_projection_summary_table(result, table_id)` — always pass explicit `table_id` to avoid duplicate component ID errors across modules.
