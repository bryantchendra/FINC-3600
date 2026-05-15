"""
Robust three-decision optimiser for the NSWDF dashboard.

The optimiser chooses:
    1. initial STI / MTG / LTG allocation,
    2. Module 5 post-drought rebalance allocation,
    3. Module 6 post-combined-stress rebalance allocation.

It is deliberately scenario-bound: a "guarantee" means every tested path
passes under the current CMA, drought, and stress assumptions supplied by the
app. If the searched grid cannot pass the tests, the correct result is
infeasible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from . import drought as dr
from . import metrics as mt
from . import optimiser as op
from . import trust_calcs as tc


TRUSTS = tc.TRUST_NAMES
TARGET_SPREAD = op.TARGET_SPREAD


@dataclass
class AllocationCandidate:
    weights: dict[str, float]
    net_return: float
    volatility: float
    liquidity_12m: float
    liquidity_3y: float
    return_surplus: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PathEvaluation:
    name: str
    passed: bool
    final_value: float
    worst_year_value: float
    liquidity_breaches: int
    post_rebalance_breaches: int
    exhausted: bool
    exhaustion_year: Optional[int]
    rebalance_cost: float
    spread_cost: float
    avg_annual_return: float
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RobustOptimisationResult:
    feasible: bool
    message: str
    grid_step: float
    candidates_tested: int
    score: float
    initial: Optional[AllocationCandidate]
    module5_rebalance: Optional[AllocationCandidate]
    module6_rebalance: Optional[AllocationCandidate]
    m5_bau: Optional[PathEvaluation]
    m5_stress: Optional[PathEvaluation]
    m6_recovery: Optional[PathEvaluation]

    def to_dict(self) -> dict:
        out = asdict(self)
        return out


def _weights_from_array(w: np.ndarray) -> dict[str, float]:
    return {t: float(w[i]) for i, t in enumerate(TRUSTS)}


def _allocation_candidate(
    weights: dict[str, float],
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float,
) -> AllocationCandidate:
    ret = tc.portfolio_net_return(weights, asset_returns)
    vol = tc.portfolio_volatility(weights, cov_matrix)
    liq = mt.liquidity_coverage(weights)
    return AllocationCandidate(
        weights=weights,
        net_return=ret,
        volatility=vol,
        liquidity_12m=liq["within_12m"],
        liquidity_3y=liq["within_3y"],
        return_surplus=ret - target_return,
    )


def _candidate_grid(
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cpi: float,
    step: float,
) -> list[AllocationCandidate]:
    target = cpi + TARGET_SPREAD
    out: list[AllocationCandidate] = []
    for row in op.generate_grid(step):
        weights = _weights_from_array(row)
        cand = _allocation_candidate(weights, asset_returns, cov_matrix, target)
        if cand.return_surplus >= -1e-12:
            out.append(cand)
    # Higher return first is a useful tie-breaker when scenario scores are close.
    out.sort(key=lambda c: (c.net_return, -c.volatility), reverse=True)
    return out


def _avg_annual_return(result: dr.ProjectionResult) -> float:
    """
    Geometric mean of the portfolio's annual investment return over the horizon.

    Each year's return is measured as pre_drawdown_value / starting_value - 1,
    i.e. the growth attributable to the investment strategy (net of rebalancing
    costs), before drought redemptions are taken.  Years after the fund is
    exhausted (starting_value == 0) are excluded from the calculation.
    """
    compound = 1.0
    count = 0
    for y in result.years:
        if y.starting_value <= 0:
            break
        compound *= max(0.0, y.pre_drawdown_value) / y.starting_value
        count += 1
    if count == 0:
        return float("-inf")
    return compound ** (1.0 / count) - 1.0


def _passes_liquidity(
    result: dr.ProjectionResult,
    rebalance_year: Optional[int],
    liquidity_mode: str,
) -> tuple[bool, int, int]:
    breaches = sum(1 for y in result.years if not (y.meets_12m and y.meets_3y))
    if liquidity_mode == "all_years":
        return breaches == 0, breaches, breaches

    if liquidity_mode == "final_only":
        final_ok = bool(result.years and result.years[-1].meets_12m and result.years[-1].meets_3y)
        return final_ok, breaches, 0 if final_ok else 1

    # Default: drought/stress years may temporarily deplete liquidity, but the
    # post-rebalance strategy must restore Board liquidity thresholds.
    if rebalance_year is None:
        post = result.years
    else:
        post = [y for y in result.years if y.year >= rebalance_year]
    post_breaches = sum(1 for y in post if not (y.meets_12m and y.meets_3y))
    return post_breaches == 0, breaches, post_breaches


def _evaluate_path(
    name: str,
    result: dr.ProjectionResult,
    rebalance_year: Optional[int],
    liquidity_mode: str,
    return_hurdle: float,
) -> PathEvaluation:
    liquidity_ok, breaches, post_breaches = _passes_liquidity(
        result, rebalance_year, liquidity_mode
    )
    avg_return = _avg_annual_return(result)
    return_ok = avg_return >= return_hurdle - 1e-6
    passed = (not result.fund_exhausted) and liquidity_ok and return_ok
    worst = min((y.ending_value for y in result.years), default=result.initial_value)
    rebalance_cost = sum(y.rebalance_cost for y in result.years)

    messages = []
    if result.fund_exhausted:
        messages.append(f"fund exhausted in Year {result.exhaustion_year}")
    if not liquidity_ok:
        messages.append(f"{post_breaches} post-test liquidity breach(es)")
    if not return_ok:
        messages.append(
            f"10Y avg return {avg_return*100:.2f}% below hurdle {return_hurdle*100:.2f}%"
        )
    if not messages:
        messages.append("passed")

    return PathEvaluation(
        name=name,
        passed=passed,
        final_value=result.final_value,
        worst_year_value=worst,
        liquidity_breaches=breaches,
        post_rebalance_breaches=post_breaches,
        exhausted=result.fund_exhausted,
        exhaustion_year=result.exhaustion_year,
        rebalance_cost=rebalance_cost,
        spread_cost=result.total_spread_cost,
        avg_annual_return=avg_return,
        message="; ".join(messages),
    )


def _path_score(*paths: PathEvaluation, surplus: float = 0.0) -> float:
    worst_final = min(p.final_value for p in paths)
    worst_value = min(p.worst_year_value for p in paths)
    total_cost = sum(p.rebalance_cost + p.spread_cost for p in paths)
    return worst_final + 0.15 * worst_value + 300_000_000 * surplus - 0.15 * total_cost


def optimise_three_decision(
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cpi: float,
    drought_schedule: dict[int, float],
    onset_split: dict[str, float],
    m5_rebalance_year: int,
    m5_stress_overrides: dict[int, dict[str, float]],
    m6_rebalance_year: int,
    m6_stress_overrides: dict[int, dict[str, float]],
    initial_value: float = 3_000_000_000,
    horizon: int = 10,
    grid_step: float = 0.05,
    liquidity_mode: str = "post_rebalance",
) -> RobustOptimisationResult:
    """
    Search the robust allocation grid and return the best passing policy.

    `grid_step=0.05` means 5 percentage-point increments. Smaller steps are
    more precise but slower because the optimiser tests scenario projections
    for each candidate allocation.

    Search order:
      1. Pre-compute the best M6 rebalance for each candidate initial allocation.
         M6 recovery depends only on (w0, reb6), so this is independent of M5.
      2. Skip the expensive M5 double-projection search for any initial that has
         no feasible M6 rebalance — this prunes the majority of infeasible initials
         before the costlier M5 work begins.
      3. For surviving initials, search M5 rebalance candidates.
      4. Combine the best (M5, M6) pair and track the overall winner by a master
         score that is the min worst-case Y10 value across all three paths.
    """
    return_hurdle = cpi + TARGET_SPREAD
    candidates = _candidate_grid(asset_returns, cov_matrix, cpi, grid_step)
    if not candidates:
        return RobustOptimisationResult(
            feasible=False,
            message="No allocation on the searched grid meets CPI + 2.5% and liquidity constraints.",
            grid_step=grid_step,
            candidates_tested=0,
            score=float("-inf"),
            initial=None,
            module5_rebalance=None,
            module6_rebalance=None,
            m5_bau=None,
            m5_stress=None,
            m6_recovery=None,
        )

    drawdown_splits = {min(drought_schedule): onset_split} if drought_schedule else {}
    best: Optional[tuple[float, AllocationCandidate, AllocationCandidate, AllocationCandidate,
                         PathEvaluation, PathEvaluation, PathEvaluation]] = None
    tested = 0

    # ------------------------------------------------------------------
    # Step 1: pre-compute the best M6 rebalance for each initial.
    # M6 recovery path depends only on (w0, reb6) — it does NOT depend on
    # which M5 rebalance is chosen later.  Computing this first lets us skip
    # the more expensive M5 search (2 projections per candidate) for any
    # initial allocation that has no feasible M6 rebalance.
    # ------------------------------------------------------------------
    BestM6 = Optional[tuple[float, AllocationCandidate, PathEvaluation]]
    m6_cache: list[BestM6] = []

    for initial in candidates:
        w0 = initial.weights
        best_m6_entry: BestM6 = None
        for reb6 in candidates:
            rebalance = {m6_rebalance_year: reb6.weights}
            recovery = dr.project(
                initial_value, w0, asset_returns, drought_schedule,
                horizon=horizon, drawdown_splits=drawdown_splits,
                rebalance_schedule=rebalance,
                trust_return_overrides=m6_stress_overrides,
            )
            recovery_eval = _evaluate_path(
                "M6 stress + drought -> rebalance -> BAU recovery",
                recovery, m6_rebalance_year, liquidity_mode, return_hurdle,
            )
            tested += 1
            if recovery_eval.passed:
                score = _path_score(recovery_eval, surplus=reb6.return_surplus)
                if best_m6_entry is None or score > best_m6_entry[0]:
                    best_m6_entry = (score, reb6, recovery_eval)
        m6_cache.append(best_m6_entry)

    # ------------------------------------------------------------------
    # Step 2: search M5 only for initials that have a feasible M6 rebalance.
    # ------------------------------------------------------------------
    for idx, initial in enumerate(candidates):
        if m6_cache[idx] is None:
            continue

        w0 = initial.weights
        best_m5: Optional[tuple[float, AllocationCandidate, PathEvaluation, PathEvaluation]] = None

        for reb5 in candidates:
            rebalance = {m5_rebalance_year: reb5.weights}
            bau = dr.project(
                initial_value, w0, asset_returns, drought_schedule,
                horizon=horizon, drawdown_splits=drawdown_splits,
                rebalance_schedule=rebalance,
            )
            stress = dr.project(
                initial_value, w0, asset_returns, drought_schedule,
                horizon=horizon, drawdown_splits=drawdown_splits,
                rebalance_schedule=rebalance,
                trust_return_overrides=m5_stress_overrides,
            )
            bau_eval = _evaluate_path(
                "M5 drought -> rebalance -> BAU",
                bau, m5_rebalance_year, liquidity_mode, return_hurdle,
            )
            stress_eval = _evaluate_path(
                "M5 drought -> rebalance -> late stress",
                stress, m5_rebalance_year, liquidity_mode, return_hurdle,
            )
            tested += 1
            if not (bau_eval.passed and stress_eval.passed):
                continue
            score = _path_score(bau_eval, stress_eval, surplus=reb5.return_surplus)
            if best_m5 is None or score > best_m5[0]:
                best_m5 = (score, reb5, bau_eval, stress_eval)

        if best_m5 is None:
            continue

        m6_score, reb6, m6_recovery = m6_cache[idx]
        m5_score, reb5, m5_bau, m5_stress_eval = best_m5
        surplus = min(initial.return_surplus, reb5.return_surplus, reb6.return_surplus)

        # Master score: worst Y10 value across all three paths drives the ranking;
        # surplus vs CPI+2.5% is a secondary tie-breaker.
        worst_final = min(m5_bau.final_value, m5_stress_eval.final_value, m6_recovery.final_value)
        master_score = worst_final + 300_000_000 * surplus

        if best is None or master_score > best[0]:
            best = (master_score, initial, reb5, reb6, m5_bau, m5_stress_eval, m6_recovery)

    if best is None:
        return RobustOptimisationResult(
            feasible=False,
            message=(
                "No three-allocation policy on the searched grid passed the selected "
                "drought, late-stress, and combined-stress tests. Try a coarser liquidity "
                "rule, lower Y10 floor, or revisit the CMA assumptions."
            ),
            grid_step=grid_step,
            candidates_tested=tested,
            score=float("-inf"),
            initial=None,
            module5_rebalance=None,
            module6_rebalance=None,
            m5_bau=None,
            m5_stress=None,
            m6_recovery=None,
        )

    master_score, initial, reb5, reb6, m5_bau, m5_stress_eval, m6_recovery = best
    return RobustOptimisationResult(
        feasible=True,
        message=(
            "Robust policy found. The guarantee is conditional on current CMA, "
            "drought, stress, and grid-search assumptions."
        ),
        grid_step=grid_step,
        candidates_tested=tested,
        score=master_score,
        initial=initial,
        module5_rebalance=reb5,
        module6_rebalance=reb6,
        m5_bau=m5_bau,
        m5_stress=m5_stress_eval,
        m6_recovery=m6_recovery,
    )
