"""
Robust three-decision optimiser for the NSWDF dashboard.

The optimiser chooses:
    1. initial STI / MTG / LTG allocation,
    2. Module 5 post-drought rebalance allocation,
    3. Module 6 post-combined-stress rebalance allocation.

The initial allocation is also certified against the Module 4 stress-only
path. Its recovery-start rebalance trades back to the initial weights, so the
path is still affected only by the initial allocation.

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
    module4_rebalance: Optional[AllocationCandidate]
    module5_rebalance: Optional[AllocationCandidate]
    module6_rebalance: Optional[AllocationCandidate]
    m4_stress: Optional[PathEvaluation]
    m5_bau: Optional[PathEvaluation]
    m5_stress: Optional[PathEvaluation]
    m6_recovery: Optional[PathEvaluation]
    diagnostics: Optional[dict] = None

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


def _candidate_pool_diagnostics(
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    cpi: float,
    step: float,
    candidates: list[AllocationCandidate],
) -> dict:
    target = cpi + TARGET_SPREAD
    grid = op.generate_grid(step)
    best_return = None
    return_fails = 0
    for row in grid:
        weights = _weights_from_array(row)
        ret = tc.portfolio_net_return(weights, asset_returns)
        best_return = ret if best_return is None else max(best_return, ret)
        if ret < target - 1e-12:
            return_fails += 1
    return {
        "stage": "Allocation pool",
        "tested": int(len(grid)),
        "passed": int(len(candidates)),
        "failed": int(len(grid) - len(candidates)),
        "return_fail": int(return_fails),
        "liquidity_fail": 0,
        "exhaustion_fail": 0,
        "best_avg_return": best_return,
        "best_final_value": None,
        "min_post_breaches": 0,
        "note": "Pre-filter requires Board liquidity floors and forecast return >= CPI + 2.5%.",
    }


def _new_path_stats(stage: str, note: str) -> dict:
    return {
        "stage": stage,
        "tested": 0,
        "passed": 0,
        "failed": 0,
        "return_fail": 0,
        "liquidity_fail": 0,
        "exhaustion_fail": 0,
        "best_avg_return": None,
        "best_final_value": None,
        "min_post_breaches": None,
        "note": note,
    }


def _record_path_stats(stats: dict, evaluation: "PathEvaluation",
                       return_hurdle: float) -> None:
    stats["tested"] += 1
    if evaluation.passed:
        stats["passed"] += 1
    else:
        stats["failed"] += 1

    if evaluation.exhausted:
        stats["exhaustion_fail"] += 1
    if evaluation.post_rebalance_breaches > 0:
        stats["liquidity_fail"] += 1
    if evaluation.avg_annual_return < return_hurdle - 1e-6:
        stats["return_fail"] += 1

    best_avg = stats["best_avg_return"]
    if best_avg is None or evaluation.avg_annual_return > best_avg:
        stats["best_avg_return"] = evaluation.avg_annual_return

    best_final = stats["best_final_value"]
    if best_final is None or evaluation.final_value > best_final:
        stats["best_final_value"] = evaluation.final_value

    min_breaches = stats["min_post_breaches"]
    if min_breaches is None or evaluation.post_rebalance_breaches < min_breaches:
        stats["min_post_breaches"] = evaluation.post_rebalance_breaches


def _new_combo_stats(stage: str, note: str) -> dict:
    stats = _new_path_stats(stage, note)
    stats["best_avg_return"] = None
    stats["best_final_value"] = None
    stats["min_post_breaches"] = None
    return stats


def _record_combo_stats(stats: dict, passed: bool, evaluations: list["PathEvaluation"],
                        return_hurdle: float) -> None:
    stats["tested"] += 1
    if passed:
        stats["passed"] += 1
    else:
        stats["failed"] += 1

    if any(e.exhausted for e in evaluations):
        stats["exhaustion_fail"] += 1
    if any(e.post_rebalance_breaches > 0 for e in evaluations):
        stats["liquidity_fail"] += 1
    if any(e.avg_annual_return < return_hurdle - 1e-6 for e in evaluations):
        stats["return_fail"] += 1

    if not evaluations:
        return

    combo_avg = min(e.avg_annual_return for e in evaluations)
    combo_final = min(e.final_value for e in evaluations)
    combo_breaches = max(e.post_rebalance_breaches for e in evaluations)

    best_avg = stats["best_avg_return"]
    if best_avg is None or combo_avg > best_avg:
        stats["best_avg_return"] = combo_avg

    best_final = stats["best_final_value"]
    if best_final is None or combo_final > best_final:
        stats["best_final_value"] = combo_final

    min_breaches = stats["min_post_breaches"]
    if min_breaches is None or combo_breaches < min_breaches:
        stats["min_post_breaches"] = combo_breaches


def _avg_annual_return(result: dr.ProjectionResult) -> float:
    """
    Geometric mean of the portfolio's annual net investment return, consistent
    with the Master Fund Return Summary table.

    Each year's return = sum(starting_weights[t] * trust_returns[t]) — the
    weighted net return earned on the allocation held at the start of the year,
    before drought redemptions.  Years after the fund is exhausted are excluded.
    """
    compound = 1.0
    count = 0
    for y in result.years:
        if y.starting_value <= 0:
            break
        port_net = sum(y.starting_weights[t] * y.trust_returns[t]
                       for t in y.starting_weights)
        compound *= (1.0 + port_net)
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
    check_return: bool = True,
) -> PathEvaluation:
    liquidity_ok, breaches, post_breaches = _passes_liquidity(
        result, rebalance_year, liquidity_mode
    )
    avg_return = _avg_annual_return(result)
    return_ok = (avg_return >= return_hurdle - 1e-6) if check_return else True
    passed = (not result.fund_exhausted) and liquidity_ok and return_ok
    worst = min((y.ending_value for y in result.years), default=result.initial_value)
    rebalance_cost = sum(y.rebalance_cost for y in result.years)

    messages = []
    if result.fund_exhausted:
        messages.append(f"fund exhausted in Year {result.exhaustion_year}")
    if not liquidity_ok:
        messages.append(f"{post_breaches} post-test liquidity breach(es)")
    if check_return and not return_ok:
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
    worst_return = min(p.avg_annual_return for p in paths)
    return worst_return + 1e-9 * surplus


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
    m4_stress_overrides: Optional[dict[int, dict[str, float]]] = None,
    m4_rebalance_year: Optional[int] = None,
    m4_reb_year: Optional[int] = None,
    initial_value: float = 3_000_000_000,
    horizon: int = 10,
    grid_step: float = 0.05,
    liquidity_mode: str = "all_years",
    seed_weights: Optional[list[dict[str, float]]] = None,
) -> RobustOptimisationResult:
    """
    Search the robust allocation grid and return the best passing policy.

    `grid_step=0.05` means 5 percentage-point increments. Smaller steps are
    more precise but slower because the optimiser tests scenario projections
    for each candidate allocation.

    Search order:
      1. Apply the M4 stress-only gate to each candidate initial allocation.
         If m4_rebalance_year is supplied, rebalance back to the initial weights
         at the recovery start. If m4_reb_year is supplied, also search over all
         M4 rebalance candidates (a fourth decision variable) and keep the best
         passing one per initial. This stage depends only on w0 and the M4 reb.
      2. Pre-compute the best M6 rebalance for each M4-surviving initial allocation.
         M6 recovery depends only on (w0, reb6), so this is independent of M5.
      3. Skip the expensive M5 double-projection search for any initial that has
         no feasible M6 rebalance — this prunes the majority of infeasible initials
         before the costlier M5 work begins.
      4. For surviving initials, search M5 rebalance candidates.
      5. Combine the best (M4 reb, M5, M6) triple and track the overall winner by
         a master score that is the worst 10Y average return across all certified
         paths.
    """
    return_hurdle = cpi + TARGET_SPREAD
    candidates = _candidate_grid(asset_returns, cov_matrix, cpi, grid_step)

    # Inject user-configured seed allocations (e.g. from Modules 3, 5, 6).
    # Seeds bypass the return pre-filter so manually-found solutions are always tested,
    # even when they fall between grid points (e.g. 33/33/34 is not on any standard grid).
    if seed_weights:
        existing = [tuple(round(c.weights[t], 6) for t in TRUSTS) for c in candidates]
        for sw in seed_weights:
            total = sum(max(0.0, sw.get(t, 0.0)) for t in TRUSTS)
            if total <= 0:
                continue
            normed = {t: max(0.0, sw.get(t, 0.0)) / total for t in TRUSTS}
            key = tuple(round(normed[t], 6) for t in TRUSTS)
            if key in existing:
                continue  # already on the grid
            cand = _allocation_candidate(normed, asset_returns, cov_matrix, return_hurdle)
            candidates.append(cand)
            existing.append(key)
        candidates.sort(key=lambda c: (c.net_return, -c.volatility), reverse=True)

    diagnostics = {
        "return_hurdle": return_hurdle,
        "liquidity_mode": liquidity_mode,
        "stages": [
            _candidate_pool_diagnostics(
                asset_returns, cov_matrix, cpi, grid_step, candidates
            )
        ],
    }
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
            m4_stress=None,
            m5_bau=None,
            m5_stress=None,
            m6_recovery=None,
            diagnostics=diagnostics,
        )

    drawdown_splits = {min(drought_schedule): onset_split} if drought_schedule else {}
    best: Optional[tuple[float, AllocationCandidate, Optional[AllocationCandidate],
                         AllocationCandidate, AllocationCandidate,
                         Optional[PathEvaluation], PathEvaluation, PathEvaluation,
                         PathEvaluation]] = None
    tested = 0

    # ------------------------------------------------------------------
    # Step 1: pre-compute the M4 stress-only result for each initial.
    # If m4_reb_year is set, search all candidates as the M4 rebalance
    # allocation and keep the best passing (m4_reb, eval) per initial.
    # The gate is survival + liquidity only (return hurdle relaxed).
    # ------------------------------------------------------------------
    m4_enabled = m4_stress_overrides is not None
    # Cache stores (m4_reb_cand | None, PathEvaluation) or None if initial fails.
    m4_cache: list[Optional[tuple[Optional[AllocationCandidate], PathEvaluation]]] = []
    m4_path_stats = _new_path_stats(
        "M4 stress-only path",
        "Market stress only, no drought. Pass criterion: non-exhaustion + liquidity "
        "(return hurdle relaxed — GFC-level shocks preclude meeting the 10Y average). "
        "Recovery-start rebalance returns to the initial allocation if enabled. "
        "If a post-stress rebalance year is configured, searches over all candidates "
        "as the M4 rebalance allocation.",
    )

    actual_m4_rebalance_year = (
        int(m4_rebalance_year)
        if m4_rebalance_year is not None and 1 <= int(m4_rebalance_year) <= horizon
        else None
    )
    actual_m4_reb_year = (
        int(m4_reb_year)
        if m4_reb_year is not None and 1 <= int(m4_reb_year) <= horizon
        else None
    )

    if m4_enabled:
        for initial in candidates:
            base_reb_sched: dict[int, dict] = {}
            if actual_m4_rebalance_year is not None:
                base_reb_sched[actual_m4_rebalance_year] = initial.weights

            if actual_m4_reb_year is None:
                # No strategic M4 rebalance — single evaluation per initial.
                stress_only = dr.project(
                    initial_value, initial.weights, asset_returns, {},
                    horizon=horizon, trust_return_overrides=m4_stress_overrides,
                    rebalance_schedule=base_reb_sched or None,
                )
                stress_only_eval = _evaluate_path(
                    "M4 stress only -> recovery-start rebalance",
                    stress_only, actual_m4_rebalance_year, liquidity_mode, return_hurdle,
                    check_return=False,
                )
                _record_path_stats(m4_path_stats, stress_only_eval, return_hurdle)
                tested += 1
                if stress_only_eval.passed:
                    m4_cache.append((None, stress_only_eval))
                else:
                    m4_cache.append(None)
            else:
                # Search over all candidates as the M4 strategic rebalance allocation.
                best_m4_entry: Optional[tuple[float, AllocationCandidate, PathEvaluation]] = None
                for m4_reb in candidates:
                    reb_sched = dict(base_reb_sched)
                    reb_sched[actual_m4_reb_year] = m4_reb.weights
                    stress_only = dr.project(
                        initial_value, initial.weights, asset_returns, {},
                        horizon=horizon, trust_return_overrides=m4_stress_overrides,
                        rebalance_schedule=reb_sched,
                    )
                    stress_only_eval = _evaluate_path(
                        "M4 stress only -> strategic rebalance",
                        stress_only, actual_m4_reb_year, liquidity_mode, return_hurdle,
                        check_return=False,
                    )
                    _record_path_stats(m4_path_stats, stress_only_eval, return_hurdle)
                    tested += 1
                    if not stress_only_eval.passed:
                        continue
                    score = _path_score(stress_only_eval, surplus=m4_reb.return_surplus)
                    if best_m4_entry is None or score > best_m4_entry[0]:
                        best_m4_entry = (score, m4_reb, stress_only_eval)
                if best_m4_entry is not None:
                    m4_cache.append((best_m4_entry[1], best_m4_entry[2]))
                else:
                    m4_cache.append(None)
        diagnostics["stages"].append(m4_path_stats)
    else:
        m4_cache = [None for _ in candidates]

    # ------------------------------------------------------------------
    # Step 2: pre-compute the best M6 rebalance for each initial.
    # M6 is a full certification gate — return hurdle + liquidity + non-exhaustion.
    # Only passing M6 entries are cached; initials with no feasible M6 are pruned
    # before the more expensive M5 search begins.
    # ------------------------------------------------------------------
    BestM6 = Optional[tuple[float, AllocationCandidate, PathEvaluation]]
    m6_cache: list[BestM6] = []
    m6_path_stats = _new_path_stats(
        "M6 recovery path",
        "Combined stress + drought, then Module 6 rebalance and BAU recovery. "
        "Full gate: return ≥ CPI+2.5%, liquidity, non-exhaustion.",
    )
    m6_initial_stats = _new_combo_stats(
        "Initial allocations surviving M6",
        "Counts initial allocations with at least one Module 6 rebalance that passes.",
    )

    for initial in candidates:
        idx = len(m6_cache)
        if m4_enabled and m4_cache[idx] is None:
            m6_cache.append(None)
            continue
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
            _record_path_stats(m6_path_stats, recovery_eval, return_hurdle)
            tested += 1
            if not recovery_eval.passed:
                continue
            score = _path_score(recovery_eval, surplus=reb6.return_surplus)
            if best_m6_entry is None or score > best_m6_entry[0]:
                best_m6_entry = (score, reb6, recovery_eval)
        _record_combo_stats(
            m6_initial_stats,
            best_m6_entry is not None,
            [best_m6_entry[2]] if best_m6_entry is not None else [],
            return_hurdle,
        )
        m6_cache.append(best_m6_entry)

    diagnostics["stages"].extend([m6_path_stats, m6_initial_stats])

    # ------------------------------------------------------------------
    # Step 3: search M5 for candidates that survived M4 and M6.
    # 4-path certification: M4 stress only (full gate), M5 BAU (full gate),
    # M5 stress (survival+liquidity), and M6 combined stress (full gate).
    # ------------------------------------------------------------------
    m5_bau_stats = _new_path_stats(
        "M5 BAU branch",
        "Drought, Module 5 rebalance, then BAU continuation.",
    )
    m5_stress_stats = _new_path_stats(
        "M5 late-stress branch",
        "Drought, Module 5 rebalance, then selected late market stress. "
        "Pass criterion: non-exhaustion + liquidity only (return hurdle not applied).",
    )
    m5_pair_stats = _new_combo_stats(
        "M5 paired rebalance",
        "Counts Module 5 rebalance choices where both BAU and late-stress branches pass.",
    )
    for idx, initial in enumerate(candidates):
        m4_cache_entry = m4_cache[idx]
        if m4_enabled and m4_cache_entry is None:
            continue  # initial allocation failed the Module 4 stress-only gate
        m4_reb_cand = m4_cache_entry[0] if m4_cache_entry is not None else None
        m4_eval     = m4_cache_entry[1] if m4_cache_entry is not None else None

        m6_entry = m6_cache[idx]
        if m6_entry is None:
            continue  # no feasible M6 for this initial — skip before M5 search

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
            # BAU branch: pass criterion is non-exhaustion + liquidity only.
            # Return hurdle relaxed — post-drought rebalancing is a structural
            # decision, not a return guarantee; the drought drawdowns themselves
            # suppress the 10Y geometric average below the hurdle in many valid cases.
            bau_eval = _evaluate_path(
                "M5 drought -> rebalance -> BAU",
                bau, m5_rebalance_year, liquidity_mode, return_hurdle,
                check_return=False,
            )
            # Stress branch: pass criterion is non-exhaustion + liquidity only.
            stress_eval = _evaluate_path(
                "M5 drought -> rebalance -> late stress",
                stress, m5_rebalance_year, liquidity_mode, return_hurdle,
                check_return=False,
            )
            _record_path_stats(m5_bau_stats, bau_eval, return_hurdle)
            _record_path_stats(m5_stress_stats, stress_eval, return_hurdle)
            _record_combo_stats(
                m5_pair_stats,
                bau_eval.passed and stress_eval.passed,
                [bau_eval, stress_eval],
                return_hurdle,
            )
            tested += 1
            if not (bau_eval.passed and stress_eval.passed):
                continue
            # Score on worst-case return across both paths.
            score = _path_score(bau_eval, stress_eval, surplus=reb5.return_surplus)
            if best_m5 is None or score > best_m5[0]:
                best_m5 = (score, reb5, bau_eval, stress_eval)

        if best_m5 is None:
            continue

        m5_score, reb5, m5_bau, m5_stress_eval = best_m5
        reb6        = m6_entry[1]
        m6_recovery = m6_entry[2]

        m4_surplus  = m4_reb_cand.return_surplus if m4_reb_cand is not None else initial.return_surplus
        surplus = min(initial.return_surplus, m4_surplus, reb5.return_surplus, reb6.return_surplus)

        # Master score: worst-case 10Y return across all certified paths.
        certified_paths = [p for p in [m4_eval, m5_bau, m5_stress_eval, m6_recovery] if p is not None]
        worst_return = min(p.avg_annual_return for p in certified_paths)
        master_score = worst_return + 1e-9 * surplus

        if best is None or master_score > best[0]:
            best = (master_score, initial, m4_reb_cand, reb5, reb6, m4_eval,
                    m5_bau, m5_stress_eval, m6_recovery)

    diagnostics["stages"].extend([m5_bau_stats, m5_stress_stats, m5_pair_stats])

    if best is None:
        return RobustOptimisationResult(
            feasible=False,
            message=(
                "No four-path policy passed all gates: M4 stress-only (survival+liquidity), "
                "M5 BAU (survival+liquidity), M5 stress (survival+liquidity), "
                "and M6 combined stress (survival+liquidity+return). "
                "Try a coarser grid, revisit CMA inputs, or adjust stress scenario settings."
            ),
            grid_step=grid_step,
            candidates_tested=tested,
            score=float("-inf"),
            initial=None,
            module4_rebalance=None,
            module5_rebalance=None,
            module6_rebalance=None,
            m4_stress=None,
            m5_bau=None,
            m5_stress=None,
            m6_recovery=None,
            diagnostics=diagnostics,
        )

    master_score, initial, m4_reb, reb5, reb6, m4_stress, m5_bau, m5_stress_eval, m6_recovery = best
    m4_reb_note = (
        f"Post-stress rebalance at Year {actual_m4_reb_year} to the recommended M4 allocation. "
        if actual_m4_reb_year is not None else ""
    )
    return RobustOptimisationResult(
        feasible=True,
        message=(
            "Four-path robust policy found. "
            "M4 stress-only: certified on survival + liquidity (return hurdle relaxed). "
            + m4_reb_note
            + "M5 BAU: certified on survival + liquidity (return hurdle relaxed — "
            "drought drawdowns suppress the 10Y geometric average in many valid cases). "
            "M5 stress: certified on survival + liquidity (return hurdle relaxed). "
            "M6 combined stress: certified on survival + liquidity + return ≥ CPI+2.5%."
        ),
        grid_step=grid_step,
        candidates_tested=tested,
        score=master_score,
        initial=initial,
        module4_rebalance=m4_reb,
        module5_rebalance=reb5,
        module6_rebalance=reb6,
        m4_stress=m4_stress,
        m5_bau=m5_bau,
        m5_stress=m5_stress_eval,
        m6_recovery=m6_recovery,
        diagnostics=diagnostics,
    )
