# NSWDF Portfolio Dashboard

Interactive Plotly Dash dashboard for the NSW Drought Fund (AUD 3 billion)
allocation analysis across the STI / MTG / LTG unit trusts.

## Status

| Module | Status |
| --- | --- |
| 1. CMA Inputs (returns, vols, 11x11 correlation, CPI) | Built |
| 2. Trust Characteristics + CFO Brief Tables | Pending |
| 3. Optimisation (grid search + scipy refinement) | Pending |
| 4. Market Stress Testing | Pending |
| 5. Drought Scenario | Pending |
| 6. Combined Stress | Pending |

## Run

```
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

## What's in this slice

- `modules/trust_calcs.py` — engine: 11 asset classes, fixed trust weight
  vectors, gross/net return, covariance-based volatility, Sharpe, CMA-to-
  covariance conversion, and nearest-PSD correction (eigenvalue-clipping
  approximation of Higham's method).
- `app.py` — Dash app shell with Module 1 fully wired:
  - Editable returns/volatility table (11 rows, pre-populated from history)
  - Editable 11x11 correlation matrix with auto-mirroring of off-diagonal
    edits and silent nearest-PSD correction with an amber warning banner
  - CPI input (default 2.5%)
  - Single `dcc.Store` (`cma-store`) that downstream modules will read

The CSV at `data/index_returns.csv` is the Refinitiv monthly index series
(Jan 2006 to Feb 2026, 11 asset classes). Pre-population uses arithmetic
mean x 12 for expected returns, monthly std x sqrt(12) for vol, and the
sample correlation matrix.

## Key conventions (for downstream modules)

- All returns and costs are stored as decimals internally. The UI displays
  percentages, the engine and the `cma-store` payload work in decimals.
- `cma-store` schema:
  - `returns`: 11-element list, decimal expected returns
  - `vols`: 11-element list, decimal volatilities
  - `corr`: 11x11 list of lists, PSD-corrected
  - `cpi`: scalar decimal
  - `psd_adjusted`: bool
- Cash gross return = `returns[0]` is used as the Sharpe risk-free rate
  (per spec: gross, not net of cost).
- Trust weight vectors are constructed in `trust_calcs.build_trust_weight_vector`
  and already encode the 50/50 Unhedged/Hedged split for the global equity
  block within MTG and LTG.
