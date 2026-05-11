# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the STI / MTG / LTG unit trusts.

_Last updated: 11 May 2026 by Sol_

## Status

| Module | Status |
| --- | --- |
| 1. CMA Inputs + EDA (returns, vols, 11x11 correlation, CPI) | Built |
| 2. Trust Characteristics + CFO Brief Tables | Built |
| 3. Optimisation (grid search + scipy refinement) | Built |
| 4. Market Stress Testing | Built |
| 5. Drought Scenario | Built |
| 6. Combined Stress | Built |

## Run

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050 in your browser. The app runs with `debug=True` so edits to `app.py` auto-reload — no restart needed.

If you see stale errors after a big change: `rm -rf __pycache__ && python app.py`

## Directory Structure

```
FINC-3600-main/
├── app.py                      # Single Dash app (all 6 modules)
├── requirements.txt
├── data/
│   ├── index_returns.csv       # Monthly returns Jan 2006–Feb 2026, 11 asset classes
│   └── macro_indicators.csv    # Monthly macro series Jan 2006–Feb 2026
├── modules/
│   ├── trust_calcs.py          # Core engine: trust weights, return, vol, Sharpe, PSD
│   ├── metrics.py
│   ├── optimiser.py
│   ├── stress.py
│   └── drought.py
└── CPI Forecast/               # Source data + notebooks for CPI forecasting
```

## Recent Changes (Module 1 EDA)

- **Interactive asset toggles** on Monthly Returns, Cumulative Returns, and Rolling Vol charts — show/hide individual asset classes with All / None shortcuts.
- **Monthly vs Annualised toggle** on the returns chart — annualised mode uses realized geometric calendar-year returns (not monthly × 12) with an annual x-axis.
- **GFC and COVID era shading** on time-series charts, clipped to the selected analysis period.
- **Global Analysis Period selector** at the top of Module 1 — a single From / To date range that drives all EDA charts, descriptive stats, correlation matrix, return distributions, and CMA historical columns simultaneously.
- **CPI Assumption** control moved into the same control bar as the period selector.
- **Interactive Risk-Return scatter** with period and asset class toggles.
- **CMA table enhancements**: historical return and vol columns shaded light grey; forecast vol column also shaded grey; delta column (Forecast − Hist Return) added at far right with green/red conditional shading by magnitude (light < 0.5%, mild 0.5–1.5%, dark > 1.5%).
- **Macro indicators data** (`data/macro_indicators.csv`) added — monthly AUS CPI YoY %, US CPI YoY %, and Fed Funds Rate % from Jan 2006 to Feb 2026, sourced from ABS and FRED.
- **Color palette** updated to richer trust tones across all charts and the CMA table.

## What's in this slice

- `modules/trust_calcs.py` — engine: 11 asset classes, fixed trust weight
  vectors, gross/net return, covariance-based volatility, Sharpe, CMA-to-
  covariance conversion, and nearest-PSD correction (eigenvalue-clipping
  approximation of Higham's method).
- `app.py` — Dash app with Module 1 fully wired:
  - Editable returns/volatility/delta CMA table (11 rows, pre-populated from historical period)
  - Editable 11x11 correlation matrix with auto-mirroring and nearest-PSD correction
  - Full EDA panel: time-series returns, cumulative growth, rolling vol, risk-return scatter, descriptive stats, return distributions, correlation heatmap — all period-reactive
  - Single `dcc.Store` (`cma-store`) that downstream modules read

## Key Conventions

- All returns and costs stored as **decimals** internally; UI displays percentages.
- `cma-store` schema: `{returns, vols, corr, cpi, psd_adjusted}` — all decimals.
- Cash gross return (`returns[0]`) is used as the Sharpe risk-free rate.
- Trust weight vectors in `trust_calcs.build_trust_weight_vector` encode the 50/50 Unhedged/Hedged split for global equity within MTG and LTG.
