"""
Risk and performance metrics used across modules.

All functions accept a sequence of period returns (decimal, e.g. 0.025 = 2.5%).
None of these functions compound or annualise unless explicitly noted in the
function name. Callers pass the appropriate frequency.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(returns: Iterable[float]) -> float:
    """
    Maximum peak-to-trough drawdown from a sequence of period returns.

    Returns a non-negative number: 0.30 means a 30% drawdown.
    """
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative / running_max - 1.0
    out = float(-drawdown.min())
    return 0.0 if out == 0.0 else out  # normalises -0.0 artefact


def drawdown_series(returns: Iterable[float]) -> np.ndarray:
    """Drawdown at each step (non-positive values, 0 at new peaks)."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return arr
    cumulative = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(cumulative)
    return cumulative / running_max - 1.0


def time_to_recovery(returns: Iterable[float]) -> int | None:
    """
    Number of periods between the trough of the maximum drawdown and the
    first point at which cumulative wealth recovers to the prior peak.

    Returns None if the drawdown has not recovered by the end of the series.
    """
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return None
    cumulative = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(cumulative)
    dd = cumulative / running_max - 1.0
    trough_idx = int(np.argmin(dd))
    peak_value = float(running_max[trough_idx])
    after = cumulative[trough_idx:]
    recovered = np.where(after >= peak_value)[0]
    if recovered.size == 0:
        return None
    return int(recovered[0])


# ---------------------------------------------------------------------------
# VaR and CVaR
# ---------------------------------------------------------------------------

def var_historic(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Historical VaR at `confidence`, returned as a non-negative number."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return 0.0
    q = np.quantile(arr, 1.0 - confidence)
    return float(-q)


def cvar_historic(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Historical Expected Shortfall at `confidence`, non-negative."""
    arr = np.asarray(list(returns), dtype=float)
    if arr.size == 0:
        return 0.0
    threshold = np.quantile(arr, 1.0 - confidence)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------

def annualised_arithmetic(monthly_returns: Iterable[float]) -> float:
    """Arithmetic mean of monthly returns x 12."""
    arr = np.asarray(list(monthly_returns), dtype=float)
    return float(arr.mean() * 12) if arr.size else 0.0


def annualised_geometric(monthly_returns: Iterable[float]) -> float:
    """(prod(1 + r))^(12/N) - 1."""
    arr = np.asarray(list(monthly_returns), dtype=float)
    n = arr.size
    if n == 0:
        return 0.0
    return float(np.prod(1.0 + arr) ** (12.0 / n) - 1.0)


def annualised_vol(monthly_returns: Iterable[float]) -> float:
    """Sample std of monthly returns x sqrt(12)."""
    arr = np.asarray(list(monthly_returns), dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) * np.sqrt(12))


# ---------------------------------------------------------------------------
# Transaction costs (used in Modules 3, 5, 6)
# ---------------------------------------------------------------------------

def transaction_cost_aud(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value_aud: float,
    buy_spreads: dict[str, float],
    sell_spreads: dict[str, float],
) -> dict:
    """
    Round-trip transaction cost in AUD when reweighting current -> target.

    Sell spread applies to the absolute reduction in each trust; buy spread
    applies to the absolute increase. Returns a dict with totals and a
    per-trust breakdown.
    """
    sell_aud = buy_aud = sell_cost = buy_cost = 0.0
    per_trust: dict[str, dict] = {}
    for trust in current_weights:
        delta_w = target_weights.get(trust, 0.0) - current_weights[trust]
        delta_aud = delta_w * portfolio_value_aud
        if delta_aud > 0:
            cost = delta_aud * buy_spreads[trust]
            buy_aud += delta_aud
            buy_cost += cost
            side = "buy"
        elif delta_aud < 0:
            cost = -delta_aud * sell_spreads[trust]
            sell_aud += -delta_aud
            sell_cost += cost
            side = "sell"
        else:
            cost = 0.0
            side = "none"
        per_trust[trust] = {
            "delta_weight": float(delta_w),
            "delta_aud": float(delta_aud),
            "side": side,
            "cost_aud": float(cost),
        }
    return {
        "sell_aud": sell_aud,
        "buy_aud": buy_aud,
        "sell_cost_aud": sell_cost,
        "buy_cost_aud": buy_cost,
        "total_cost_aud": sell_cost + buy_cost,
        "per_trust": per_trust,
    }


# ---------------------------------------------------------------------------
# Liquidity coverage (used by Module 3)
# ---------------------------------------------------------------------------

def liquidity_coverage(weights: dict[str, float]) -> dict:
    """
    Liquidity coverage based on trust redemption frequency:
        STI = daily       -> within 12 months and 3 years
        MTG = monthly     -> within 3 years
        LTG = quarterly   -> not within 12 months
    """
    sti = weights.get("STI", 0.0)
    mtg = weights.get("MTG", 0.0)
    within_12m = sti
    within_3y = sti + mtg
    return {
        "within_12m": within_12m,
        "within_3y": within_3y,
        "meets_12m": within_12m >= 0.10,
        "meets_3y": within_3y >= 0.25,
    }
