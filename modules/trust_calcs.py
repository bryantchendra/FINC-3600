"""
Trust-level computation engine for the NSWDF dashboard.

This module is the single source of truth for:
- The 11 asset classes (names match the Refinitiv CSV columns exactly)
- The fixed trust weight vectors (STI, MTG, LTG)
- Asset class costs, trust ongoing costs, buy/sell spreads
- Conversion from CMA inputs (returns, vols, correlations) to a covariance matrix
- Trust-level gross/net return, volatility, and Sharpe ratio
- PSD correction for user-supplied correlation matrices

Conventions:
- All return and cost figures are stored as DECIMALS (e.g. 0.045 = 4.5% p.a.)
- The Cash gross return from the CMA is used as the risk-free rate for Sharpe
  (per spec: "not net of cost")
- Buy/sell spreads are NOT applied here. They are event-driven costs handled in
  the optimiser, drought, and combined-stress modules only.
- The "30% Global Equities (50% hedged to AUD)" trust composition is encoded
  directly as 15% Unhedged + 15% Hedged in TRUST_RAW_WEIGHTS. There is no
  separate blending step; the table already reflects the split.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------

# Order MUST match the Refinitiv CSV column order. Any change here breaks the
# correlation matrix indexing and the trust weight vectors.
ASSET_CLASSES: list[str] = [
    "Cash",
    "Australian Short Duration Bond",
    "Australian Fixed Income",
    "Global Fixed Income (Hedged)",
    "Global Credit (Hedged)",
    "Australian Listed Equity",
    "Global Listed Equity (Unhedged)",
    "Global Listed Equity (Hedged)",
    "Australian Listed Property",
    "Global Infrastructure (Unhedged)",
    "Global Private Equity",
]

# Short labels for use in tight UI components (correlation matrix headers, etc.)
ASSET_CLASS_SHORT: dict[str, str] = {
    "Cash": "Cash",
    "Australian Short Duration Bond": "AU SD Bond",
    "Australian Fixed Income": "AU Fixed Inc",
    "Global Fixed Income (Hedged)": "Glb FI (H)",
    "Global Credit (Hedged)": "Glb Credit (H)",
    "Australian Listed Equity": "AU Equity",
    "Global Listed Equity (Unhedged)": "Glb Eq (U)",
    "Global Listed Equity (Hedged)": "Glb Eq (H)",
    "Australian Listed Property": "AU Property",
    "Global Infrastructure (Unhedged)": "Glb Infra (U)",
    "Global Private Equity": "Glb PE",
}

# Asset class management costs (% p.a., as decimals)
ASSET_COSTS: dict[str, float] = {
    "Cash": 0.0004,
    "Australian Short Duration Bond": 0.0007,
    "Australian Fixed Income": 0.0008,
    "Global Fixed Income (Hedged)": 0.0025,
    "Global Credit (Hedged)": 0.0030,
    "Australian Listed Equity": 0.0010,
    "Global Listed Equity (Unhedged)": 0.0010,
    "Global Listed Equity (Hedged)": 0.0020,
    "Australian Listed Property": 0.0015,
    "Global Infrastructure (Unhedged)": 0.0040,
    "Global Private Equity": 0.0068,
}


# ---------------------------------------------------------------------------
# Trust definitions (FIXED per the IM — must not be edited from the UI)
# ---------------------------------------------------------------------------

# Per-trust raw weight maps. The Unhedged/Hedged split for global equities is
# already encoded as two separate 15% lines per the spec.
TRUST_RAW_WEIGHTS: dict[str, dict[str, float]] = {
    "STI": {
        "Cash": 0.50,
        "Australian Short Duration Bond": 0.50,
    },
    "MTG": {
        "Cash": 0.075,
        "Global Credit (Hedged)": 0.125,
        "Australian Fixed Income": 0.15,
        "Global Fixed Income (Hedged)": 0.15,
        "Global Listed Equity (Unhedged)": 0.15,
        "Global Listed Equity (Hedged)": 0.15,
        "Australian Listed Equity": 0.15,
        "Australian Listed Property": 0.05,
    },
    "LTG": {
        "Cash": 0.05,
        "Global Credit (Hedged)": 0.05,
        "Global Fixed Income (Hedged)": 0.05,
        "Global Listed Equity (Unhedged)": 0.15,
        "Global Listed Equity (Hedged)": 0.15,
        "Australian Listed Equity": 0.30,
        "Global Private Equity": 0.15,
        "Global Infrastructure (Unhedged)": 0.10,
    },
}

TRUST_NAMES: list[str] = ["STI", "MTG", "LTG"]

TRUST_ONGOING_COSTS: dict[str, float] = {
    "STI": 0.00126,
    "MTG": 0.00358,
    "LTG": 0.00488,
}

TRUST_BUY_SPREADS: dict[str, float] = {
    "STI": 0.0003,
    "MTG": 0.0015,
    "LTG": 0.0018,
}

TRUST_SELL_SPREADS: dict[str, float] = {
    "STI": 0.0003,
    "MTG": 0.0016,
    "LTG": 0.0019,
}

# Trust-level CPI+ targets (used by Module 2 pass/fail flags)
# STI is Cash + 50bps; MTG is CPI + 200bps; LTG is CPI + 300bps.
# These are returned by `trust_target_return` given a CMA cash return and CPI.
def trust_target_return(trust_name: str, cash_return: float, cpi: float) -> float:
    if trust_name == "STI":
        return cash_return + 0.005
    if trust_name == "MTG":
        return cpi + 0.02
    if trust_name == "LTG":
        return cpi + 0.03
    raise ValueError(f"Unknown trust: {trust_name}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_trust_weights() -> None:
    """Sanity check: each trust's weights must sum to 1.0."""
    for trust, raw in TRUST_RAW_WEIGHTS.items():
        total = sum(raw.values())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(
                f"Trust {trust} weights sum to {total}, expected 1.0"
            )
        for asset in raw:
            if asset not in ASSET_CLASSES:
                raise ValueError(
                    f"Trust {trust} references unknown asset: {asset}"
                )


_validate_trust_weights()


# ---------------------------------------------------------------------------
# Weight vectors and trust-level metrics
# ---------------------------------------------------------------------------

def build_trust_weight_vector(trust_name: str) -> np.ndarray:
    """
    Return the 11-element weight vector for `trust_name`, ordered by
    ASSET_CLASSES.
    """
    raw = TRUST_RAW_WEIGHTS[trust_name]
    return np.array([raw.get(asset, 0.0) for asset in ASSET_CLASSES], dtype=float)


def asset_cost_vector() -> np.ndarray:
    """Return the 11-element cost vector ordered by ASSET_CLASSES."""
    return np.array([ASSET_COSTS[a] for a in ASSET_CLASSES], dtype=float)


def trust_gross_return(trust_name: str, asset_returns: np.ndarray) -> float:
    """Weighted average of asset class returns using the trust's fixed weights."""
    w = build_trust_weight_vector(trust_name)
    return float(w @ np.asarray(asset_returns, dtype=float))


def trust_weighted_asset_cost(trust_name: str) -> float:
    """Weighted asset class cost component of the trust net return."""
    w = build_trust_weight_vector(trust_name)
    return float(w @ asset_cost_vector())


def trust_net_return(trust_name: str, asset_returns: np.ndarray) -> float:
    """Net = gross return - weighted asset cost - trust ongoing cost."""
    return (
        trust_gross_return(trust_name, asset_returns)
        - trust_weighted_asset_cost(trust_name)
        - TRUST_ONGOING_COSTS[trust_name]
    )


def trust_volatility(trust_name: str, cov_matrix: np.ndarray) -> float:
    """sqrt(w^T Sigma w) using the trust's fixed weight vector."""
    w = build_trust_weight_vector(trust_name)
    cov = np.asarray(cov_matrix, dtype=float)
    var = float(w @ cov @ w)
    # Numerical safety: clip tiny negatives caused by floating-point noise
    return float(np.sqrt(max(var, 0.0)))


def trust_sharpe(
    trust_name: str,
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
) -> float:
    """
    Sharpe = (trust_net_return - cash_return) / trust_vol.
    `cash_return` is the GROSS Cash expected return from the CMA (per spec).
    """
    nr = trust_net_return(trust_name, asset_returns)
    vol = trust_volatility(trust_name, cov_matrix)
    if vol <= 0:
        return float("nan")
    return (nr - cash_return) / vol


# ---------------------------------------------------------------------------
# Portfolio-level metrics (used by Modules 3, 5, 6)
# ---------------------------------------------------------------------------

def portfolio_net_return(
    trust_weights: dict[str, float], asset_returns: np.ndarray
) -> float:
    """Weighted sum of trust net returns across STI/MTG/LTG."""
    return sum(
        trust_weights[t] * trust_net_return(t, asset_returns) for t in TRUST_NAMES
    )


def portfolio_covariance(cov_matrix: np.ndarray) -> np.ndarray:
    """
    3x3 covariance matrix between the three trusts, derived by projecting the
    11x11 asset covariance through the trust weight matrix W (3 x 11).
    Used for portfolio volatility computations in Module 3.
    """
    W = np.array([build_trust_weight_vector(t) for t in TRUST_NAMES])  # (3, 11)
    return W @ np.asarray(cov_matrix, dtype=float) @ W.T  # (3, 3)


def portfolio_volatility(
    trust_weights: dict[str, float], cov_matrix: np.ndarray
) -> float:
    """sqrt(x^T (W Sigma W^T) x) where x is the 3-element trust weight vector."""
    x = np.array([trust_weights[t] for t in TRUST_NAMES], dtype=float)
    trust_cov = portfolio_covariance(cov_matrix)
    var = float(x @ trust_cov @ x)
    return float(np.sqrt(max(var, 0.0)))


def portfolio_sharpe(
    trust_weights: dict[str, float],
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
) -> float:
    nr = portfolio_net_return(trust_weights, asset_returns)
    vol = portfolio_volatility(trust_weights, cov_matrix)
    if vol <= 0:
        return float("nan")
    return (nr - cash_return) / vol


# ---------------------------------------------------------------------------
# CMA -> covariance and PSD correction
# ---------------------------------------------------------------------------

def cma_to_covariance(
    vols: np.ndarray, corr_matrix: np.ndarray
) -> np.ndarray:
    """
    Build a covariance matrix from a vector of volatilities and a correlation
    matrix: Sigma[i,j] = vol[i] * vol[j] * corr[i,j].
    """
    vols = np.asarray(vols, dtype=float)
    corr = np.asarray(corr_matrix, dtype=float)
    return np.outer(vols, vols) * corr


def is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True iff `matrix` is symmetric and positive semi-definite."""
    m = np.asarray(matrix, dtype=float)
    if m.shape[0] != m.shape[1]:
        return False
    if not np.allclose(m, m.T, atol=1e-10):
        return False
    eigvals = np.linalg.eigvalsh((m + m.T) / 2)
    return bool(eigvals.min() >= -tol)


def nearest_psd_correlation(
    corr: np.ndarray, epsilon: float = 1e-8
) -> tuple[np.ndarray, bool]:
    """
    Return the nearest PSD correlation matrix using spectral eigenvalue
    clipping (a simple approximation of Higham's nearest-correlation method).

    Approach:
        1. Symmetrize the input.
        2. If already PSD (within numerical tolerance), return unchanged.
        3. Otherwise floor the eigenvalues at `epsilon`, reconstruct, and
           renormalise the diagonal to 1.

    Returns
    -------
    (corrected_matrix, was_adjusted)
        was_adjusted is True iff the input required correction.
    """
    corr = np.asarray(corr, dtype=float)
    # Symmetrize first
    corr_sym = (corr + corr.T) / 2

    eigvals = np.linalg.eigvalsh(corr_sym)
    if eigvals.min() >= -epsilon and np.allclose(np.diag(corr_sym), 1.0, atol=1e-9):
        # Already PSD and unit-diagonal, nothing to adjust
        return corr_sym, False

    # Clip eigenvalues
    eigvals_full, eigvecs = np.linalg.eigh(corr_sym)
    eigvals_clipped = np.maximum(eigvals_full, epsilon)
    psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Renormalise to unit diagonal (so it's a valid correlation matrix)
    d = np.sqrt(np.diag(psd))
    d[d == 0] = 1.0
    psd = psd / np.outer(d, d)

    # Force exact symmetry and unit diagonal to clean up floating-point noise
    psd = (psd + psd.T) / 2
    np.fill_diagonal(psd, 1.0)
    # Clip any tiny excursions outside [-1, 1]
    psd = np.clip(psd, -1.0, 1.0)
    np.fill_diagonal(psd, 1.0)
    return psd, True


# ---------------------------------------------------------------------------
# Convenience: full trust characteristics (used by Module 2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Historical backtest helpers (used by Module 2)
# ---------------------------------------------------------------------------

def historical_trust_returns_monthly(returns_df) -> "pd.DataFrame":
    """
    Compute monthly trust-level returns under monthly rebalancing to fixed
    weights.

    `returns_df` is a DataFrame whose columns are exactly ASSET_CLASSES in the
    canonical order. Returns a DataFrame indexed identically with one column
    per trust ('STI', 'MTG', 'LTG').
    """
    import pandas as pd  # local import keeps this module pandas-optional
    out = pd.DataFrame(index=returns_df.index)
    R = returns_df.values  # (T, 11)
    for t in TRUST_NAMES:
        w = build_trust_weight_vector(t)
        out[t] = R @ w
    return out


def historical_trust_returns_monthly_net(returns_df) -> "pd.DataFrame":
    """
    As `historical_trust_returns_monthly` but with monthly costs deducted:
    monthly_cost = (weighted_asset_cost + trust_ongoing_cost) / 12.
    """
    import pandas as pd
    gross = historical_trust_returns_monthly(returns_df)
    net = pd.DataFrame(index=gross.index)
    for t in TRUST_NAMES:
        annual_cost = trust_weighted_asset_cost(t) + TRUST_ONGOING_COSTS[t]
        net[t] = gross[t] - annual_cost / 12
    return net


def historical_cumulative_wealth(monthly_trust_returns) -> "pd.DataFrame":
    """Cumulative wealth path starting at 1.0 from monthly returns."""
    return (1.0 + monthly_trust_returns).cumprod()


# ---------------------------------------------------------------------------
# Convenience: full trust characteristics (used by Module 2)
# ---------------------------------------------------------------------------

def trust_characteristics(
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
    cpi: float,
) -> dict[str, dict[str, float]]:
    """
    Compute the full Module 2 metric set for all three trusts.

    Returns a dict keyed by trust name, with values:
        {
            "gross_return", "weighted_asset_cost", "ongoing_cost",
            "net_return", "volatility", "sharpe",
            "target_return", "cpi_plus_spread", "meets_target"
        }
    """
    out: dict[str, dict[str, float]] = {}
    for t in TRUST_NAMES:
        gross = trust_gross_return(t, asset_returns)
        wac = trust_weighted_asset_cost(t)
        ongoing = TRUST_ONGOING_COSTS[t]
        net = gross - wac - ongoing
        vol = trust_volatility(t, cov_matrix)
        sharpe = (net - cash_return) / vol if vol > 0 else float("nan")
        target = trust_target_return(t, cash_return, cpi)
        out[t] = {
            "gross_return": gross,
            "weighted_asset_cost": wac,
            "ongoing_cost": ongoing,
            "net_return": net,
            "volatility": vol,
            "sharpe": sharpe,
            "target_return": target,
            "cpi_plus_spread": net - cpi,
            "meets_target": net >= target,
        }
    return out
