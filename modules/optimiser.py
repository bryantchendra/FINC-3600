"""
Portfolio optimisation for the NSWDF allocation problem.

The decision variable is a 3-element vector (w_STI, w_MTG, w_LTG) that:
    1. sums to 1
    2. has all entries >= 0
    3. satisfies the liquidity constraints:
         w_STI >= 0.10                        (10% within 12 months)
         w_STI + w_MTG >= 0.25                (25% within 3 years)
    4. delivers a net portfolio return >= CPI + 2.5%

The dimensionality is small enough that a 1%-resolution grid search is the
primary search strategy; scipy.optimize.minimize is used only for final-step
refinement around the grid optimum.

Module conventions:
    - All returns and vols are in decimal form.
    - asset_returns is an 11-element numpy array, ordered per
      trust_calcs.ASSET_CLASSES.
    - cov_matrix is the 11x11 asset covariance, NOT the 3x3 trust covariance.
      Routines internally project to the trust level using trust_calcs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import trust_calcs as tc

GRID_STEP = 0.01           # 1% grid resolution
TARGET_SPREAD = 0.025      # CPI + 2.5%
LIQUIDITY_12M_MIN = 0.10
LIQUIDITY_3Y_MIN = 0.25
TRUST_MAX = 0.50           # Board policy cap on any single trust allocation
TRUST_MIN = 0.00           # Default diversification floor (no floor)


# ---------------------------------------------------------------------------
# Constraint helpers
# ---------------------------------------------------------------------------

def liquidity_feasible(w_sti: float, w_mtg: float) -> bool:
    """True iff (w_STI, w_MTG, .) satisfies the two liquidity constraints."""
    return (w_sti >= LIQUIDITY_12M_MIN - 1e-12
            and (w_sti + w_mtg) >= LIQUIDITY_3Y_MIN - 1e-12)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid(step: float = GRID_STEP, trust_max: float = TRUST_MAX,
                  trust_min: float = TRUST_MIN) -> np.ndarray:
    """
    Enumerate all (w_STI, w_MTG, w_LTG) triples on the simplex (sum to 1,
    each in [0, 1]) at the given step, that ALSO satisfy the liquidity
    constraints and the diversification floor. Each row is a 3-vector.

    `trust_max` caps each individual trust allocation (default: TRUST_MAX = 0.50).
    `trust_min` sets a minimum floor on each trust allocation (default: 0.0).
    """
    n = int(round(1 / step)) + 1
    grid = np.linspace(0, 1, n)
    sti_min = max(LIQUIDITY_12M_MIN, trust_min)
    out = []
    for w_sti in grid:
        if w_sti < sti_min - 1e-12 or w_sti > trust_max + 1e-12:
            continue
        for w_mtg in grid:
            if w_mtg < trust_min - 1e-12 or w_mtg > trust_max + 1e-12:
                continue
            w_ltg = 1.0 - w_sti - w_mtg
            if w_ltg < trust_min - 1e-12 or w_ltg > trust_max + 1e-12:
                continue
            if (w_sti + w_mtg) < LIQUIDITY_3Y_MIN - 1e-12:
                continue
            out.append([w_sti, w_mtg, max(0.0, w_ltg)])
    return np.array(out)


# ---------------------------------------------------------------------------
# Vectorised metric evaluation across the grid
# ---------------------------------------------------------------------------

def evaluate_grid(
    grid: np.ndarray,
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
) -> pd.DataFrame:
    """
    Evaluate net return, volatility, and Sharpe for every portfolio in the
    grid. Returns a DataFrame with columns: w_STI, w_MTG, w_LTG, net_return,
    volatility, sharpe.
    """
    # Trust net return vector: 3-element array, one per trust
    trust_net = np.array([
        tc.trust_net_return(t, asset_returns) for t in tc.TRUST_NAMES
    ])
    trust_cov = tc.portfolio_covariance(cov_matrix)  # 3x3

    net = grid @ trust_net  # (N,)
    # Vol: sqrt of diag(grid @ Sigma @ grid.T) — but we only want diagonal
    var = np.einsum("ij,jk,ik->i", grid, trust_cov, grid)
    var = np.clip(var, 0.0, None)
    vol = np.sqrt(var)

    sharpe = np.where(vol > 0, (net - cash_return) / vol, np.nan)

    df = pd.DataFrame({
        "w_STI": grid[:, 0],
        "w_MTG": grid[:, 1],
        "w_LTG": grid[:, 2],
        "net_return": net,
        "volatility": vol,
        "sharpe": sharpe,
    })
    return df


# ---------------------------------------------------------------------------
# Objective functions for scipy refinement
# ---------------------------------------------------------------------------

def _portfolio_metrics(w_xy: np.ndarray, trust_net, trust_cov, cash_return):
    """Helper: from (w_STI, w_MTG), compute (return, vol, sharpe)."""
    w = np.array([w_xy[0], w_xy[1], 1 - w_xy[0] - w_xy[1]])
    r = float(w @ trust_net)
    var = float(w @ trust_cov @ w)
    var = max(var, 0.0)
    vol = float(np.sqrt(var))
    sharpe = (r - cash_return) / vol if vol > 0 else 0.0
    return r, vol, sharpe, w


def _refine_max_sharpe(seed_w, trust_net, trust_cov, cash_return, cpi,
                       trust_max: float = TRUST_MAX, trust_min: float = TRUST_MIN):
    """scipy refinement around `seed_w` to maximise Sharpe."""
    target_return = cpi + TARGET_SPREAD

    def neg_sharpe(w_xy):
        _, _, s, _ = _portfolio_metrics(w_xy, trust_net, trust_cov, cash_return)
        return -s

    tm, tmin = trust_max, trust_min
    constraints = [
        {"type": "ineq", "fun": lambda w: w[0] - max(LIQUIDITY_12M_MIN, tmin)},
        {"type": "ineq", "fun": lambda w: w[0] + w[1] - LIQUIDITY_3Y_MIN},
        {"type": "ineq", "fun": lambda w, _tm=tm: w[0] + w[1] - (1 - _tm)},
        {"type": "ineq", "fun": lambda w, _min=tmin: 1 - _min - w[0] - w[1]},
        {"type": "ineq", "fun": lambda w: w[1] - tmin},
        {"type": "ineq",
         "fun": lambda w: _portfolio_metrics(w, trust_net, trust_cov, cash_return)[0] - target_return},
    ]
    bounds = [(max(0.0, tmin), tm), (max(0.0, tmin), tm)]
    res = minimize(
        neg_sharpe, x0=seed_w[:2], bounds=bounds,
        constraints=constraints, method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 200},
    )
    return res


def _refine_min_vol(seed_w, trust_net, trust_cov, cash_return, cpi,
                    trust_max: float = TRUST_MAX, trust_min: float = TRUST_MIN):
    target_return = cpi + TARGET_SPREAD

    def vol_sq(w_xy):
        w = np.array([w_xy[0], w_xy[1], 1 - w_xy[0] - w_xy[1]])
        return float(w @ trust_cov @ w)

    tm, tmin = trust_max, trust_min
    constraints = [
        {"type": "ineq", "fun": lambda w: w[0] - max(LIQUIDITY_12M_MIN, tmin)},
        {"type": "ineq", "fun": lambda w: w[0] + w[1] - LIQUIDITY_3Y_MIN},
        {"type": "ineq", "fun": lambda w, _tm=tm: w[0] + w[1] - (1 - _tm)},
        {"type": "ineq", "fun": lambda w, _min=tmin: 1 - _min - w[0] - w[1]},
        {"type": "ineq", "fun": lambda w: w[1] - tmin},
        {"type": "ineq",
         "fun": lambda w: _portfolio_metrics(w, trust_net, trust_cov, cash_return)[0] - target_return},
    ]
    bounds = [(max(0.0, tmin), tm), (max(0.0, tmin), tm)]
    res = minimize(
        vol_sq, x0=seed_w[:2], bounds=bounds,
        constraints=constraints, method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    return res


def _refine_max_return(seed_w, trust_net, trust_cov, cash_return, cpi, vol_cap,
                       trust_max: float = TRUST_MAX, trust_min: float = TRUST_MIN):
    target_return = cpi + TARGET_SPREAD

    def neg_ret(w_xy):
        w = np.array([w_xy[0], w_xy[1], 1 - w_xy[0] - w_xy[1]])
        return -float(w @ trust_net)

    tm, tmin = trust_max, trust_min
    constraints = [
        {"type": "ineq", "fun": lambda w: w[0] - max(LIQUIDITY_12M_MIN, tmin)},
        {"type": "ineq", "fun": lambda w: w[0] + w[1] - LIQUIDITY_3Y_MIN},
        {"type": "ineq", "fun": lambda w, _tm=tm: w[0] + w[1] - (1 - _tm)},
        {"type": "ineq", "fun": lambda w, _min=tmin: 1 - _min - w[0] - w[1]},
        {"type": "ineq", "fun": lambda w: w[1] - tmin},
        {"type": "ineq",
         "fun": lambda w: _portfolio_metrics(w, trust_net, trust_cov, cash_return)[0] - target_return},
        # vol <= cap  =>  cap^2 - vol^2 >= 0
        {"type": "ineq",
         "fun": lambda w: vol_cap ** 2 - float(np.array([w[0], w[1], 1 - w[0] - w[1]]) @ trust_cov @ np.array([w[0], w[1], 1 - w[0] - w[1]]))},
    ]
    bounds = [(max(0.0, tmin), tm), (max(0.0, tmin), tm)]
    res = minimize(
        neg_ret, x0=seed_w[:2], bounds=bounds,
        constraints=constraints, method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 200},
    )
    return res


# ---------------------------------------------------------------------------
# Optimisation result
# ---------------------------------------------------------------------------

@dataclass
class OptimisationResult:
    feasible: bool
    message: str
    weights: dict[str, float]      # always populated; default {} if infeasible
    net_return: float
    volatility: float
    sharpe: float
    cpi_plus_spread: float
    objective: str

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_result(objective: str, message: str) -> OptimisationResult:
    return OptimisationResult(
        feasible=False, message=message, weights={}, net_return=float("nan"),
        volatility=float("nan"), sharpe=float("nan"),
        cpi_plus_spread=float("nan"), objective=objective,
    )


def _result_from_w(
    w: np.ndarray, trust_net, trust_cov, cash_return, cpi, objective
) -> OptimisationResult:
    r = float(w @ trust_net)
    var = max(float(w @ trust_cov @ w), 0.0)
    vol = float(np.sqrt(var))
    sharpe = (r - cash_return) / vol if vol > 0 else float("nan")
    return OptimisationResult(
        feasible=True, message="", weights={t: float(w[i]) for i, t in enumerate(tc.TRUST_NAMES)},
        net_return=r, volatility=vol, sharpe=sharpe,
        cpi_plus_spread=r - cpi, objective=objective,
    )


# ---------------------------------------------------------------------------
# Top-level optimisation entry points
# ---------------------------------------------------------------------------

def optimise(
    objective: str,
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
    cpi: float,
    vol_cap: Optional[float] = None,
    grid_df: Optional[pd.DataFrame] = None,
    trust_max: float = TRUST_MAX,
    trust_min: float = TRUST_MIN,
) -> OptimisationResult:
    """
    Run grid search + scipy refinement for the requested objective.

    objective in {"max_sharpe", "min_vol", "max_return"}.
    For "max_return", `vol_cap` (decimal) is required.
    `grid_df` is the precomputed feasibility-evaluated grid; built fresh if None.
    `trust_max` caps each individual trust allocation; pass 1.0 to disable.
    `trust_min` sets a minimum floor on each trust (diversification constraint).
    """
    target_return = cpi + TARGET_SPREAD

    trust_net = np.array([
        tc.trust_net_return(t, asset_returns) for t in tc.TRUST_NAMES
    ])
    trust_cov = tc.portfolio_covariance(cov_matrix)

    if grid_df is None:
        grid = generate_grid(trust_max=trust_max, trust_min=trust_min)
        grid_df = evaluate_grid(grid, asset_returns, cov_matrix, cash_return)

    # Apply objective-specific feasibility filter
    if objective == "max_sharpe":
        feas = grid_df[grid_df["net_return"] >= target_return - 1e-12]
        if feas.empty:
            return _empty_result(
                objective,
                "No feasible allocation meets the CPI+2.5% target under "
                "current CMA assumptions.",
            )
        sharpe_valid = feas["sharpe"].dropna()
        if sharpe_valid.empty:
            sharpe_valid = feas["net_return"]
        idx = sharpe_valid.idxmax()
        seed = feas.loc[idx, ["w_STI", "w_MTG", "w_LTG"]].values
        res = _refine_max_sharpe(seed, trust_net, trust_cov, cash_return, cpi,
                                 trust_max=trust_max, trust_min=trust_min)
        if res.success:
            w = np.array([res.x[0], res.x[1], 1 - res.x[0] - res.x[1]])
            # Use refined point only if it still satisfies the return target
            r_refined = float(w @ trust_net)
            if r_refined >= target_return - 1e-9 and liquidity_feasible(w[0], w[1]):
                return _result_from_w(w, trust_net, trust_cov, cash_return, cpi, objective)
        # Fall back to the grid optimum
        return _result_from_w(seed, trust_net, trust_cov, cash_return, cpi, objective)

    elif objective == "min_vol":
        feas = grid_df[grid_df["net_return"] >= target_return - 1e-12]
        if feas.empty:
            return _empty_result(
                objective,
                "No feasible allocation meets the CPI+2.5% target under "
                "current CMA assumptions.",
            )
        vol_valid = feas["volatility"].dropna()
        if vol_valid.empty:
            vol_valid = feas["net_return"]
        idx = vol_valid.idxmin()
        seed = feas.loc[idx, ["w_STI", "w_MTG", "w_LTG"]].values
        res = _refine_min_vol(seed, trust_net, trust_cov, cash_return, cpi,
                              trust_max=trust_max, trust_min=trust_min)
        if res.success:
            w = np.array([res.x[0], res.x[1], 1 - res.x[0] - res.x[1]])
            r_refined = float(w @ trust_net)
            if r_refined >= target_return - 1e-9 and liquidity_feasible(w[0], w[1]):
                return _result_from_w(w, trust_net, trust_cov, cash_return, cpi, objective)
        return _result_from_w(seed, trust_net, trust_cov, cash_return, cpi, objective)

    elif objective == "max_return":
        if vol_cap is None:
            raise ValueError("vol_cap is required for max_return objective")
        feas = grid_df[
            (grid_df["volatility"] <= vol_cap + 1e-12)
            & (grid_df["net_return"] >= target_return - 1e-12)
        ]
        if feas.empty:
            # Distinguish two cases for a clearer message
            cap_only = grid_df[grid_df["volatility"] <= vol_cap + 1e-12]
            if cap_only.empty:
                msg = (
                    f"No feasible allocation has volatility \u2264 "
                    f"{vol_cap*100:.2f}%. Lower bound on portfolio "
                    "volatility under the liquidity constraints is "
                    f"{grid_df['volatility'].min()*100:.2f}%."
                )
            else:
                msg = (
                    f"No allocation with volatility \u2264 {vol_cap*100:.2f}% "
                    "also meets the CPI+2.5% return target. The maximum "
                    "achievable return at this volatility cap is "
                    f"{cap_only['net_return'].max()*100:.2f}%."
                )
            return _empty_result(objective, msg)
        ret_valid = feas["net_return"].dropna()
        if ret_valid.empty:
            return _empty_result(objective, "No feasible allocation with valid return found.")
        idx = ret_valid.idxmax()
        seed = feas.loc[idx, ["w_STI", "w_MTG", "w_LTG"]].values
        res = _refine_max_return(seed, trust_net, trust_cov, cash_return, cpi, vol_cap,
                                 trust_max=trust_max, trust_min=trust_min)
        if res.success:
            w = np.array([res.x[0], res.x[1], 1 - res.x[0] - res.x[1]])
            var_refined = float(w @ trust_cov @ w)
            r_refined = float(w @ trust_net)
            if (var_refined <= vol_cap ** 2 + 1e-12
                and r_refined >= target_return - 1e-9
                and liquidity_feasible(w[0], w[1])):
                return _result_from_w(w, trust_net, trust_cov, cash_return, cpi, objective)
        return _result_from_w(seed, trust_net, trust_cov, cash_return, cpi, objective)

    else:
        raise ValueError(f"Unknown objective: {objective}")


# ---------------------------------------------------------------------------
# Sensitivity sweep (used by Module 3 tornado chart)
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    objective: str,
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cash_return: float,
    cpi: float,
    vol_cap: Optional[float] = None,
    bumps: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01),
    trust_max: float = TRUST_MAX,
    trust_min: float = TRUST_MIN,
) -> pd.DataFrame:
    """
    For each asset class, perturb its expected return by each `bump` (in
    decimal, e.g. -0.01 = -100bps), re-run optimisation under the same
    objective, and record the resulting portfolio's net return, volatility,
    and CPI+2.5% feasibility.

    Returns a long-format DataFrame:
        asset_class, bump_bps, net_return, volatility, sharpe,
        meets_target, feasible
    """
    rows = []
    target_return = cpi + TARGET_SPREAD

    # Baseline (unperturbed)
    base_grid = generate_grid(trust_max=trust_max, trust_min=trust_min)
    base_eval = evaluate_grid(base_grid, asset_returns, cov_matrix, cash_return)
    base_res = optimise(
        objective, asset_returns, cov_matrix, cash_return, cpi,
        vol_cap=vol_cap, grid_df=base_eval, trust_max=trust_max, trust_min=trust_min,
    )

    for i, ac in enumerate(tc.ASSET_CLASSES):
        for bump in bumps:
            perturbed = asset_returns.copy()
            perturbed[i] = perturbed[i] + bump
            grid_eval = evaluate_grid(base_grid, perturbed, cov_matrix, cash_return)
            res = optimise(
                objective, perturbed, cov_matrix, cash_return, cpi,
                vol_cap=vol_cap, grid_df=grid_eval, trust_max=trust_max, trust_min=trust_min,
            )
            rows.append({
                "asset_class": ac,
                "bump_bps": int(round(bump * 10000)),
                "feasible": res.feasible,
                "net_return": res.net_return if res.feasible else np.nan,
                "volatility": res.volatility if res.feasible else np.nan,
                "sharpe": res.sharpe if res.feasible else np.nan,
                "meets_target": (
                    res.feasible and res.net_return >= target_return - 1e-9
                ),
            })

    df = pd.DataFrame(rows)
    df.attrs["baseline"] = base_res
    return df
