"""
Market stress testing for the NSWDF dashboard.

Two kinds of scenarios:
    1. Historical-window scenarios — shocked asset returns are derived from
       cumulative returns over a specified date range in the Refinitiv CSV.
       Longer windows are annualised for CMA comparison. Short event shocks
       such as COVID Crash are kept as cumulative event-window changes, since
       annualising a two-month crash can overstate the economic interpretation.
    2. Analytical scenarios — currently just the +200bps interest rate shock,
       priced via approximate modified duration. Documented as a
       simplification.

For each scenario the same flow applies:
    user-supplied shocked asset returns (11-element decimal array)
        -> trust gross returns via fixed weight vectors
        -> trust net returns by subtracting weighted asset cost + ongoing cost
           (per spec: ongoing costs DO apply, spreads do NOT in Module 4)
        -> portfolio return given a 3-trust allocation

Conventions:
    - CSV index labels are "Mon YYYY" strings (e.g. "Aug 2008"). All window
      bounds use the same string convention so we can index directly without
      a date parser.
    - All returns are decimals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import trust_calcs as tc

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Approximate modified durations (years) for the +200bps rate shock.
# Per spec: equity / property / infra / PE are not directly impacted by the
# parallel shift, so duration = 0 for those rows.
APPROX_DURATIONS: dict[str, float] = {
    "Cash": 0.1,
    "Australian Short Duration Bond": 2.0,
    "Australian Fixed Income": 5.0,
    "Global Fixed Income (Hedged)": 7.0,
    "Global Credit (Hedged)": 4.0,
    "Australian Listed Equity": 0.0,
    "Global Listed Equity (Unhedged)": 0.0,
    "Global Listed Equity (Hedged)": 0.0,
    "Australian Listed Property": 0.0,
    "Global Infrastructure (Unhedged)": 0.0,
    "Global Private Equity": 0.0,
}

# Factor mapping for the Module 4 dominant-factor tags (per spec).
ASSET_FACTORS: dict[str, str] = {
    "Cash": "Duration",
    "Australian Short Duration Bond": "Duration",
    "Australian Fixed Income": "Duration",
    "Global Fixed Income (Hedged)": "Duration",
    "Global Credit (Hedged)": "Credit spread",
    "Australian Listed Equity": "Equity beta",
    "Global Listed Equity (Unhedged)": "Currency",
    "Global Listed Equity (Hedged)": "Equity beta",
    "Australian Listed Property": "Equity beta",
    "Global Infrastructure (Unhedged)": "Currency",
    "Global Private Equity": "Equity beta",
}

# Asset classes considered "unhedged" for the AUD shock scenario.
UNHEDGED_ASSETS: list[str] = [
    "Global Listed Equity (Unhedged)",
    "Global Infrastructure (Unhedged)",
]

# Historical window definitions. The CSV index is in "Mon YYYY" format.
SCENARIO_WINDOWS: dict[str, tuple[str, str]] = {
    "GFC": ("Nov 2007", "Jul 2009"),
    "COVID Crash": ("Feb 2020", "Mar 2020"),
    "COVID Inflation Shock (2022)": ("Jan 2022", "Dec 2022"),
}

# Short shock windows are better represented as discrete cumulative changes
# in value rather than annual rates.
EVENT_WINDOW_SCENARIOS: set[str] = {"COVID Crash"}


# ---------------------------------------------------------------------------
# Window arithmetic helpers
# ---------------------------------------------------------------------------

def _window_returns(returns_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice the CSV to the inclusive window."""
    if start not in returns_df.index or end not in returns_df.index:
        raise KeyError(f"Window bounds {start}..{end} not in CSV index")
    i = returns_df.index.get_loc(start)
    j = returns_df.index.get_loc(end)
    if i > j:
        i, j = j, i
    return returns_df.iloc[i:j + 1]


def cumulative_return(window: pd.DataFrame) -> pd.Series:
    """Cumulative return per asset class over the window: prod(1+r) - 1."""
    return (1.0 + window).prod() - 1.0


def annualise_window_return(cumulative: pd.Series, n_months: int) -> pd.Series:
    """
    Annualise a cumulative return computed over `n_months` months:
        ann = (1 + cum)^(12 / n) - 1
    Returns the SAME shape Series with values in decimal form.
    """
    if n_months <= 0:
        return cumulative
    return (1.0 + cumulative) ** (12.0 / n_months) - 1.0


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A named stress scenario producing an 11-element shocked return vector."""
    name: str
    description: str
    asset_returns: np.ndarray  # decimal, length 11, ordered per ASSET_CLASSES
    # Optional cumulative window return (decimal) for max-drawdown reporting.
    # None for analytical scenarios.
    window_cumulative: Optional[np.ndarray] = None
    window_label: Optional[str] = None
    n_months: Optional[int] = None
    return_basis: str = "annualised"
    is_analytical: bool = False


def build_historical_scenario(
    name: str,
    returns_df: pd.DataFrame,
    start: str,
    end: str,
) -> StressScenario:
    """
    Build a stress scenario from a historical window.

    The shocked return vector is usually the annualised cumulative return per
    asset class over the window. Short event shocks are retained as cumulative
    event-window returns. The window cumulative return is also retained so the
    UI can report a max-drawdown figure.
    """
    window = _window_returns(returns_df, start, end)
    cum = cumulative_return(window)
    n = len(window)
    annualised = annualise_window_return(cum, n)
    is_event_window = name in EVENT_WINDOW_SCENARIOS

    # Order by ASSET_CLASSES to be safe
    cum_arr = cum.reindex(tc.ASSET_CLASSES).values
    ann_arr = annualised.reindex(tc.ASSET_CLASSES).values
    asset_returns = cum_arr if is_event_window else ann_arr
    basis = "event_window" if is_event_window else "annualised"
    basis_text = (
        "cumulative event-window percentage changes"
        if is_event_window
        else "annualised cumulative return over the window"
    )

    return StressScenario(
        name=name,
        description=(
            f"Historical window: {start} to {end} ({n} months). "
            f"Shocked asset returns are {basis_text}."
        ),
        asset_returns=asset_returns,
        window_cumulative=cum_arr,
        window_label=f"{start} \u2013 {end}",
        n_months=n,
        return_basis=basis,
        is_analytical=False,
    )


def build_aud_shock_scenario(
    returns_df: pd.DataFrame,
    baseline_returns: np.ndarray,
) -> StressScenario:
    """
    AUD depreciation: identify the worst rolling 12-month period for the
    two unhedged asset classes (judged by the AVERAGE of their cumulative
    returns over the window) and apply those returns to the unhedged
    rows only. Hedged assets and everything else keep their CMA returns.
    """
    if len(returns_df) < 12:
        raise ValueError("Need at least 12 months of data for AUD shock")

    # Rolling 12-month cumulative return per unhedged series
    rolled = (1.0 + returns_df[UNHEDGED_ASSETS]).rolling(12).apply(np.prod, raw=True) - 1.0
    rolled = rolled.dropna()
    if rolled.empty:
        raise ValueError("Rolling 12-month series is empty")

    # Worst window = minimum of the average across the two unhedged series
    avg = rolled.mean(axis=1)
    end_idx = avg.idxmin()
    end_pos = returns_df.index.get_loc(end_idx)
    start_pos = end_pos - 11
    start_idx = returns_df.index[start_pos]

    window = returns_df.iloc[start_pos:end_pos + 1]
    cum = cumulative_return(window)
    ann = annualise_window_return(cum, 12)

    # Apply only to unhedged assets; hedged/other = CMA baseline
    out = baseline_returns.copy().astype(float)
    for ac in UNHEDGED_ASSETS:
        i = tc.ASSET_CLASSES.index(ac)
        out[i] = float(ann[ac])

    cum_full = np.full(len(tc.ASSET_CLASSES), np.nan)
    for ac in UNHEDGED_ASSETS:
        i = tc.ASSET_CLASSES.index(ac)
        cum_full[i] = float(cum[ac])

    return StressScenario(
        name="AUD Depreciation Shock",
        description=(
            f"Worst rolling 12-month window for the two unhedged asset "
            f"classes: {start_idx} to {end_idx}. Shocked returns applied "
            "to Global Listed Equity (Unhedged) and Global Infrastructure "
            "(Unhedged) only; all other asset classes hold their CMA "
            "expected return."
        ),
        asset_returns=out,
        window_cumulative=cum_full,
        window_label=f"{start_idx} \u2013 {end_idx}",
        n_months=12,
        return_basis="annualised",
        is_analytical=False,
    )


def build_rate_shock_scenario(baseline_returns: np.ndarray) -> StressScenario:
    """
    Analytical +200bps parallel shift, priced as -duration * 0.02 on top
    of the CMA baseline return. Equity / property / infra / PE keep CMA.
    """
    out = baseline_returns.copy().astype(float)
    for i, ac in enumerate(tc.ASSET_CLASSES):
        d = APPROX_DURATIONS[ac]
        if d > 0:
            # Price impact applied as a one-year hit on top of the CMA mean.
            out[i] = baseline_returns[i] - d * 0.02

    return StressScenario(
        name="Interest Rate Shock (+200bps)",
        description=(
            "Hypothetical instantaneous +200bps parallel shift in yields. "
            "Price impact computed as \u2212duration \u00d7 0.02 and added "
            "to each duration-sensitive asset's CMA return. Equity, "
            "property, infrastructure, and private equity are not directly "
            "shocked. Approximate modified durations used: "
            "Cash 0.1y, Aus Short Bond 2y, Aus Fixed Income 5y, "
            "Global FI Hedged 7y, Global Credit Hedged 4y. "
            "Limitation: a parallel shift ignores convexity and curve "
            "twists, and equities are assumed unaffected which is an "
            "approximation."
        ),
        asset_returns=out,
        window_cumulative=None,
        window_label=None,
        n_months=None,
        return_basis="annualised",
        is_analytical=True,
    )


def build_all_scenarios(
    returns_df: pd.DataFrame,
    baseline_returns: np.ndarray,
) -> dict[str, StressScenario]:
    """Return all five canonical scenarios keyed by name."""
    out: dict[str, StressScenario] = {}
    for name, (s, e) in SCENARIO_WINDOWS.items():
        out[name] = build_historical_scenario(name, returns_df, s, e)
    out["AUD Depreciation Shock"] = build_aud_shock_scenario(
        returns_df, baseline_returns
    )
    out["Interest Rate Shock (+200bps)"] = build_rate_shock_scenario(
        baseline_returns
    )
    return out


# ---------------------------------------------------------------------------
# Trust-level projection (Module 4 has no spreads)
# ---------------------------------------------------------------------------

def trust_returns_under_shock(asset_returns: np.ndarray) -> dict[str, float]:
    """
    Net trust returns under a shocked asset-return vector. Asset costs and
    trust ongoing costs ARE deducted (per spec); buy/sell spreads are NOT.
    """
    out: dict[str, float] = {}
    for t in tc.TRUST_NAMES:
        gross = tc.trust_gross_return(t, asset_returns)
        wac = tc.trust_weighted_asset_cost(t)
        ongoing = tc.TRUST_ONGOING_COSTS[t]
        out[t] = gross - wac - ongoing
    return out


def trust_returns_under_event_shock(asset_returns: np.ndarray, n_months: int) -> dict[str, float]:
    """
    Net trust return for a discrete event-window shock. Asset returns are
    cumulative over the event window and costs are pro-rated to the same
    number of months.
    """
    out: dict[str, float] = {}
    cost_fraction = max(n_months, 0) / 12.0
    for t in tc.TRUST_NAMES:
        gross = tc.trust_gross_return(t, asset_returns)
        wac = tc.trust_weighted_asset_cost(t)
        ongoing = tc.TRUST_ONGOING_COSTS[t]
        out[t] = gross - (wac + ongoing) * cost_fraction
    return out


def portfolio_return_under_shock(
    trust_weights: dict[str, float], asset_returns: np.ndarray
) -> float:
    """Weighted portfolio net return under a shock."""
    trust_nets = trust_returns_under_shock(asset_returns)
    return sum(trust_weights[t] * trust_nets[t] for t in tc.TRUST_NAMES)


def trust_drawdown_from_window(
    trust_name: str,
    returns_df: pd.DataFrame,
    start: str,
    end: str,
) -> float:
    """
    Trust-level peak-to-trough drawdown across the window, applying the
    fixed trust weight vector to the asset-class monthly returns and
    rebalancing monthly. Net of ongoing costs, no spreads.

    Returned as a non-negative number (0.30 = 30% drawdown).
    """
    from . import metrics as mt
    window = _window_returns(returns_df, start, end)
    w = tc.build_trust_weight_vector(trust_name)
    monthly_gross = window.values @ w
    annual_cost = tc.trust_weighted_asset_cost(trust_name) + tc.TRUST_ONGOING_COSTS[trust_name]
    monthly_net = monthly_gross - annual_cost / 12
    return mt.max_drawdown(monthly_net)


# ---------------------------------------------------------------------------
# Factor exposure tagging
# ---------------------------------------------------------------------------

def dominant_factor(
    trust_name: str,
    shocked_asset_returns: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """
    For a given trust under a shocked asset return vector, compute each
    asset class's contribution to the trust gross return as
        weight[i] * shocked_return[i]
    Group contributions by factor (Equity beta / Duration / Credit spread
    / Currency), take absolute values, and identify the largest factor.

    Returns
    -------
    (dominant_factor_label, factor_contributions_dict)
    """
    w = tc.build_trust_weight_vector(trust_name)
    contributions = w * shocked_asset_returns

    factor_totals: dict[str, float] = {}
    for i, ac in enumerate(tc.ASSET_CLASSES):
        factor = ASSET_FACTORS[ac]
        factor_totals[factor] = factor_totals.get(factor, 0.0) + contributions[i]

    # Drop factors with zero total exposure for this trust
    nonzero = {k: v for k, v in factor_totals.items() if abs(v) > 1e-12}
    if not nonzero:
        return ("None", {})

    # Use absolute magnitude to pick "dominant" — we care about which factor
    # drove the move, not its sign
    dom = max(nonzero, key=lambda k: abs(nonzero[k]))
    return dom, nonzero
