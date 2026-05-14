"""
Drought scenario projection engine for Modules 5 and 6.

The engine simulates the fund value year by year over a 10-year horizon
(default Year 0 to Year 10 inclusive). Each year:

    1. Grow each trust's holdings by its expected net return (or the user-
       supplied stressed return for that year, used by Module 6).
    2. If a drawdown is scheduled for that year, redeem in the order
       STI -> MTG -> LTG. Each redemption pays the trust's sell spread
       on the redeemed amount; the spread cost is deducted from the
       redemption proceeds (i.e. the trust must be redeemed by enough to
       both cover the relief AND the spread).
    3. After the redemption, trust holdings and weights are updated and
       carried into the next year.

Module 5 uses default (CMA-derived) net returns for every year.
Module 6 passes `trust_return_overrides[year_index] = {trust: shocked_net}`
to swap the normal CMA return for the stressed return in the shock year only.

Conventions:
    - All AUD figures are in dollars (decimals are not used for currency).
    - All return figures are decimals.
    - Holdings are tracked PER TRUST in AUD, not as weights, because the
      sequential redemption logic requires absolute amounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from . import trust_calcs as tc


# ---------------------------------------------------------------------------
# Drought severity bands (Appendix 3 of the assignment)
# ---------------------------------------------------------------------------

SEVERITY_BANDS: dict[str, tuple[float, float]] = {
    "Mild":     (50_000_000,    150_000_000),
    "Moderate": (300_000_000,   600_000_000),
    "Severe":   (1_000_000_000, 2_000_000_000),
}


def severity_midpoint(severity: str) -> float:
    lo, hi = SEVERITY_BANDS[severity]
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Per-year state and projection result
# ---------------------------------------------------------------------------

@dataclass
class YearState:
    year: int
    starting_value: float                  # portfolio value at start of year
    starting_weights: dict[str, float]
    trust_returns: dict[str, float]        # net return applied this year
    pre_drawdown_value: float              # after growth, before drawdown
    pre_drawdown_weights: dict[str, float]
    drawdown: float                        # 0 if no drought this year
    redemption_amounts: dict[str, float]   # AUD redeemed per trust (gross of spread)
    spread_costs: dict[str, float]         # AUD spread cost per trust
    ending_value: float
    ending_weights: dict[str, float]
    ending_holdings: dict[str, float]      # AUD per trust at year-end
    liquidity_within_12m: float            # fraction of fund accessible within 12m
    liquidity_within_3y: float
    meets_12m: bool
    meets_3y: bool
    rebalance_cost: float = 0.0           # AUD cost of rebalancing in this year (0 if no rebalance)


@dataclass
class ProjectionResult:
    years: list[YearState] = field(default_factory=list)
    total_drawdown: float = 0.0
    total_spread_cost: float = 0.0
    fund_exhausted: bool = False
    exhaustion_year: Optional[int] = None
    initial_value: float = 0.0
    final_value: float = 0.0


# ---------------------------------------------------------------------------
# Drought schedule construction (used by both deterministic and MC paths)
# ---------------------------------------------------------------------------

def build_drought_schedule(
    onset_year: int,
    total_relief: float,
    year_4_fraction: float = 0.5,
    residual_split: tuple[float, float] = (0.5, 0.5),
) -> dict[int, float]:
    """
    Distribute `total_relief` across the drought event:
        - onset_year: total_relief * year_4_fraction
        - onset_year + 1: total_relief * (1 - year_4_fraction) * residual_split[0]
        - onset_year + 2: total_relief * (1 - year_4_fraction) * residual_split[1]

    `residual_split` is normalised to sum to 1 if it doesn't.
    Returns a dict {year_index: aud_drawdown}.
    """
    schedule: dict[int, float] = {}
    schedule[onset_year] = total_relief * year_4_fraction
    rs = sum(residual_split)
    if rs > 0:
        a, b = residual_split[0] / rs, residual_split[1] / rs
    else:
        a, b = 0.5, 0.5
    residual = total_relief * (1 - year_4_fraction)
    schedule[onset_year + 1] = residual * a
    schedule[onset_year + 2] = residual * b
    # Drop zero entries
    return {y: v for y, v in schedule.items() if v > 0}


# ---------------------------------------------------------------------------
# Sequential redemption
# ---------------------------------------------------------------------------

def _redeem_sequential(
    holdings: dict[str, float],
    drawdown: float,
    sell_spreads: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """
    Redeem `drawdown` AUD from the holdings, in order STI -> MTG -> LTG.

    For each trust, the gross redemption needs to satisfy:
        (gross - gross * sell_spread) = relief_paid
    i.e. gross = relief_paid / (1 - sell_spread)
    The spread cost is deducted from the trust's holding alongside the
    relief, so the holding decreases by `gross`. The funds the drought
    receives is `relief_paid` = gross * (1 - sell_spread).

    If a trust's holding is exhausted before the relief is fully met, move
    to the next trust in sequence.

    Returns
    -------
    (new_holdings, redemption_amounts_gross, spread_costs, unmet_relief)

    `unmet_relief` is non-zero only if the entire fund is exhausted before
    the drawdown is met, in which case the caller should mark the fund as
    exhausted.
    """
    new_holdings = dict(holdings)
    redemptions = {t: 0.0 for t in tc.TRUST_NAMES}
    spreads = {t: 0.0 for t in tc.TRUST_NAMES}
    remaining_relief = drawdown

    for trust in tc.TRUST_NAMES:
        if remaining_relief <= 0:
            break
        avail = new_holdings.get(trust, 0.0)
        if avail <= 0:
            continue
        s = sell_spreads[trust]
        # Gross needed if we paid full remaining_relief from this trust
        full_gross = remaining_relief / (1 - s)
        if full_gross <= avail:
            # We can fully meet the relief from this trust
            new_holdings[trust] = avail - full_gross
            redemptions[trust] = full_gross
            spreads[trust] = full_gross * s
            remaining_relief = 0.0
        else:
            # Use the entire trust holding; relief delivered is avail*(1-s)
            relief_delivered = avail * (1 - s)
            new_holdings[trust] = 0.0
            redemptions[trust] = avail
            spreads[trust] = avail * s
            remaining_relief -= relief_delivered

    return new_holdings, redemptions, spreads, remaining_relief


def _redeem_with_target_split(
    holdings: dict[str, float],
    drawdown: float,
    target_split: dict[str, float],
    sell_spreads: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """
    Redeem a scheduled drawdown using a user-specified net relief split.

    `target_split` gives the share of net relief intended to come from each
    trust. If a trust cannot fund its assigned share, any unmet relief is
    redeemed through the standard STI -> MTG -> LTG sequence.
    """
    split_total = sum(max(0.0, target_split.get(t, 0.0)) for t in tc.TRUST_NAMES)
    if split_total <= 0:
        return _redeem_sequential(holdings, drawdown, sell_spreads)

    split = {
        t: max(0.0, target_split.get(t, 0.0)) / split_total
        for t in tc.TRUST_NAMES
    }
    new_holdings = dict(holdings)
    redemptions = {t: 0.0 for t in tc.TRUST_NAMES}
    spreads = {t: 0.0 for t in tc.TRUST_NAMES}
    unmet_relief = 0.0

    for trust in tc.TRUST_NAMES:
        assigned_relief = drawdown * split[trust]
        if assigned_relief <= 0:
            continue
        avail = new_holdings.get(trust, 0.0)
        if avail <= 0:
            unmet_relief += assigned_relief
            continue
        s = sell_spreads[trust]
        gross_needed = assigned_relief / (1 - s)
        if gross_needed <= avail:
            gross = gross_needed
            delivered = assigned_relief
        else:
            gross = avail
            delivered = gross * (1 - s)
            unmet_relief += assigned_relief - delivered
        new_holdings[trust] = avail - gross
        redemptions[trust] += gross
        spreads[trust] += gross * s

    if unmet_relief > 1e-9:
        new_holdings, spill_redemptions, spill_spreads, unmet_relief = _redeem_sequential(
            new_holdings, unmet_relief, sell_spreads
        )
        for trust in tc.TRUST_NAMES:
            redemptions[trust] += spill_redemptions[trust]
            spreads[trust] += spill_spreads[trust]

    return new_holdings, redemptions, spreads, unmet_relief


# ---------------------------------------------------------------------------
# Top-level deterministic projection
# ---------------------------------------------------------------------------

def project(
    initial_value: float,
    initial_weights: dict[str, float],
    asset_returns: np.ndarray,
    drought_schedule: dict[int, float],
    horizon: int = 10,
    trust_return_overrides: Optional[dict[int, dict[str, float]]] = None,
    drawdown_splits: Optional[dict[int, dict[str, float]]] = None,
    rebalance_schedule: Optional[dict[int, dict[str, float]]] = None,
) -> ProjectionResult:
    """
    Run a deterministic projection.

    Parameters
    ----------
    initial_value : starting AUD value at the end of Year 0 (i.e. before
        Year 1 begins). Default is 3,000,000,000.
    initial_weights : trust weights at start.
    asset_returns : 11-element CMA expected return vector. The default
        annual net return per trust is computed from this and the fixed
        trust weight vectors.
    drought_schedule : {year: drawdown_aud} for years where a drawdown
        is scheduled.
    horizon : final year index (inclusive). The projection covers years
        1..horizon (10 years by default).
    trust_return_overrides : optional {year_index: {trust_name: net_return}}
        used by Module 6 to inject a stressed trust return for one year.
    drawdown_splits : optional {year_index: {trust_name: share}}
        user-specified net relief split for scheduled drawdowns. Years not
        listed use the standard STI -> MTG -> LTG redemption sequence.

    Returns
    -------
    ProjectionResult covering years 1..horizon.

    Year 0 is implicit: it's the starting state, not simulated.
    """
    overrides = trust_return_overrides or {}
    splits = drawdown_splits or {}
    rebalances = rebalance_schedule or {}

    # Default trust net returns from the CMA
    default_trust_nets = {
        t: tc.trust_net_return(t, asset_returns) for t in tc.TRUST_NAMES
    }

    # Initial holdings in AUD
    holdings = {t: initial_weights.get(t, 0.0) * initial_value for t in tc.TRUST_NAMES}
    total = sum(holdings.values())
    if total <= 0:
        raise ValueError("Initial weights sum to zero")
    # Normalise in case weights were not exactly 1.0
    holdings = {t: v * (initial_value / total) for t, v in holdings.items()}

    result = ProjectionResult()
    result.initial_value = initial_value

    current_value = initial_value
    current_weights = {t: holdings[t] / current_value for t in tc.TRUST_NAMES}

    for year in range(1, horizon + 1):
        starting_value = current_value
        starting_weights = dict(current_weights)
        starting_holdings = dict(holdings)

        # Determine the trust net returns for this year
        if year in overrides:
            year_returns = {**default_trust_nets, **overrides[year]}
        else:
            year_returns = default_trust_nets

        # Apply growth
        for t in tc.TRUST_NAMES:
            holdings[t] = holdings[t] * (1 + year_returns[t])
        pre_drawdown_value = sum(holdings.values())
        if pre_drawdown_value > 0:
            pre_drawdown_weights = {
                t: holdings[t] / pre_drawdown_value for t in tc.TRUST_NAMES
            }
        else:
            pre_drawdown_weights = {t: 0.0 for t in tc.TRUST_NAMES}

        # Rebalance to new strategic allocation if scheduled for this year.
        # Happens after growth, before any drought drawdown.
        rebalance_cost = 0.0
        if year in rebalances and pre_drawdown_value > 0:
            target_w = rebalances[year]
            tw_sum = sum(max(0.0, target_w.get(t, 0.0)) for t in tc.TRUST_NAMES)
            if tw_sum > 0:
                normed = {t: max(0.0, target_w.get(t, 0.0)) / tw_sum for t in tc.TRUST_NAMES}
            else:
                normed = {t: 1 / len(tc.TRUST_NAMES) for t in tc.TRUST_NAMES}
            target_holdings = {t: normed[t] * pre_drawdown_value for t in tc.TRUST_NAMES}
            for t in tc.TRUST_NAMES:
                trade = target_holdings[t] - holdings[t]
                if trade < -1e-6:
                    rebalance_cost += abs(trade) * tc.TRUST_SELL_SPREADS[t]
                elif trade > 1e-6:
                    rebalance_cost += trade * tc.TRUST_BUY_SPREADS[t]
            # Move to target weights, then haircut all holdings proportionally for cost
            scale = max(0.0, (pre_drawdown_value - rebalance_cost) / pre_drawdown_value)
            holdings = {t: target_holdings[t] * scale for t in tc.TRUST_NAMES}
            pre_drawdown_value = sum(holdings.values())
            if pre_drawdown_value > 0:
                pre_drawdown_weights = {t: holdings[t] / pre_drawdown_value for t in tc.TRUST_NAMES}

        # Apply drawdown if any
        drawdown = drought_schedule.get(year, 0.0)
        if drawdown > 0:
            if year in splits:
                holdings, redemptions, spreads, unmet = _redeem_with_target_split(
                    holdings, drawdown, splits[year], tc.TRUST_SELL_SPREADS
                )
            else:
                holdings, redemptions, spreads, unmet = _redeem_sequential(
                    holdings, drawdown, tc.TRUST_SELL_SPREADS
                )
            if unmet > 1e-3:
                # Fund exhausted mid-drawdown
                result.fund_exhausted = True
                result.exhaustion_year = year
        else:
            redemptions = {t: 0.0 for t in tc.TRUST_NAMES}
            spreads = {t: 0.0 for t in tc.TRUST_NAMES}

        # End-of-year state
        ending_value = sum(holdings.values())
        if ending_value > 0:
            ending_weights = {t: holdings[t] / ending_value for t in tc.TRUST_NAMES}
        else:
            ending_weights = {t: 0.0 for t in tc.TRUST_NAMES}

        # Liquidity coverage from ending weights
        liq_12m = ending_weights["STI"]
        liq_3y = ending_weights["STI"] + ending_weights["MTG"]

        ystate = YearState(
            year=year,
            starting_value=starting_value,
            starting_weights=starting_weights,
            trust_returns=dict(year_returns),
            pre_drawdown_value=pre_drawdown_value,
            pre_drawdown_weights=pre_drawdown_weights,
            drawdown=drawdown,
            redemption_amounts=redemptions,
            spread_costs=spreads,
            ending_value=ending_value,
            ending_weights=ending_weights,
            ending_holdings=dict(holdings),
            liquidity_within_12m=liq_12m,
            liquidity_within_3y=liq_3y,
            meets_12m=(liq_12m >= 0.10) if ending_value > 0 else False,
            meets_3y=(liq_3y >= 0.25) if ending_value > 0 else False,
            rebalance_cost=rebalance_cost,
        )
        result.years.append(ystate)

        result.total_drawdown += drawdown
        result.total_spread_cost += sum(spreads.values())

        current_value = ending_value
        current_weights = ending_weights

        if ending_value <= 1.0:  # AUD 1 threshold for "exhausted"
            result.fund_exhausted = True
            if result.exhaustion_year is None:
                result.exhaustion_year = year
            # Don't break — let the trajectory show the flat zero line,
            # so users see the fund stays at zero
            for t in tc.TRUST_NAMES:
                holdings[t] = 0.0
            current_value = 0.0
            current_weights = {t: 0.0 for t in tc.TRUST_NAMES}

    result.final_value = current_value
    return result


def post_drawdown_summary(result: ProjectionResult, onset_year: int) -> dict:
    """
    Extract the key Year-(onset_year) outcome metrics for the UI panel:
        remaining_value, ending_weights, liquidity ratios, residual drawdown.
    """
    if onset_year < 1 or onset_year > len(result.years):
        return {}
    y = result.years[onset_year - 1]
    residual = sum(
        ys.drawdown for ys in result.years if ys.year > onset_year
    )
    # Can the post-drawdown portfolio meet the residual?
    can_sustain = y.ending_value >= residual

    return {
        "year": onset_year,
        "drawdown_this_year": y.drawdown,
        "remaining_value": y.ending_value,
        "ending_weights": y.ending_weights,
        "liquidity_12m": y.liquidity_within_12m,
        "liquidity_3y": y.liquidity_within_3y,
        "meets_12m": y.meets_12m,
        "meets_3y": y.meets_3y,
        "spread_cost_this_year": sum(y.spread_costs.values()),
        "residual_drawdown_to_come": residual,
        "can_sustain_residual": can_sustain,
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

# Per-year drought distribution (mutually exclusive, sum to 1.0):
DROUGHT_PROB = {
    "None":     0.525,
    "Mild":     0.300,
    "Moderate": 0.125,
    "Severe":   0.050,
}

# The order matters for sampling with np.random.choice
_DROUGHT_OUTCOMES = ["None", "Mild", "Moderate", "Severe"]
_DROUGHT_PROBS = np.array([DROUGHT_PROB[o] for o in _DROUGHT_OUTCOMES])


@dataclass
class MonteCarloResult:
    n_paths: int
    horizon: int
    # Shape (n_paths, horizon+1): value at end of each year, indexed
    # 0..horizon. Column 0 is the initial value.
    values: np.ndarray
    # Per-path total drawdown delivered, total spread cost, exhaustion year
    total_drawdowns: np.ndarray   # (n_paths,)
    total_spreads: np.ndarray     # (n_paths,)
    exhaustion_year: np.ndarray   # (n_paths,) int; -1 if never exhausted

    def percentile_bands(self) -> dict[str, np.ndarray]:
        """5th/25th/50th/75th/95th percentile value at each year."""
        pcts = [5, 25, 50, 75, 95]
        return {
            f"p{p}": np.percentile(self.values, p, axis=0) for p in pcts
        }

    def exhaustion_probability(self) -> float:
        """P(fund exhausted by horizon)."""
        return float((self.exhaustion_year >= 0).mean())


def _vectorised_redeem(
    holdings: np.ndarray,           # (n_paths, 3) AUD per trust
    drawdowns: np.ndarray,          # (n_paths,) per-path drawdown for this year
    sell_spreads: np.ndarray,       # (3,) sell spread per trust
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised version of `_redeem_sequential` operating on `n_paths` paths.

    Returns
    -------
    new_holdings : (n_paths, 3)
    spread_costs_total : (n_paths,) summed spread cost per path this year
    unmet_relief : (n_paths,) any unmet shortfall after exhausting all trusts
    """
    n_paths = holdings.shape[0]
    new_h = holdings.copy()
    remaining = drawdowns.astype(float).copy()
    spreads_total = np.zeros(n_paths)

    for ti in range(3):  # 0=STI, 1=MTG, 2=LTG
        s = sell_spreads[ti]
        active = remaining > 1e-6  # paths still needing to redeem
        if not np.any(active):
            break

        # Gross needed if we paid all remaining from this trust
        full_gross = np.where(active, remaining / (1 - s), 0.0)
        avail = new_h[:, ti]

        # If avail >= full_gross: fully covered from this trust
        fully_covered = active & (full_gross <= avail)
        partially_covered = active & (full_gross > avail)

        # Fully covered branch
        new_h[fully_covered, ti] = avail[fully_covered] - full_gross[fully_covered]
        spreads_total[fully_covered] += full_gross[fully_covered] * s
        remaining[fully_covered] = 0.0

        # Partial branch: deplete the trust
        if np.any(partially_covered):
            relief_delivered = avail[partially_covered] * (1 - s)
            spreads_total[partially_covered] += avail[partially_covered] * s
            new_h[partially_covered, ti] = 0.0
            remaining[partially_covered] -= relief_delivered

    return new_h, spreads_total, remaining


def monte_carlo(
    initial_value: float,
    initial_weights: dict[str, float],
    asset_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_paths: int = 10_000,
    horizon: int = 10,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Monte Carlo over portfolio paths.

    Each year for each path:
        1. Draw a drought severity from the 4-way categorical distribution.
        2. If a drought, draw the relief amount from Uniform within the
           severity band. Treat each year's drought as a single-year event:
           the entire relief amount is paid in that year (no multi-year
           splits). This is a simplification — the deterministic projection
           is the right tool for multi-year drought structures.
        3. Draw trust net returns from Normal(mean, vol) using the
           portfolio-projected (3x3) trust covariance and trust mean
           vectors derived from the CMA inputs.
        4. Apply growth then sequential redemption with sell spreads.

    Parameters
    ----------
    initial_value, initial_weights : starting portfolio.
    asset_returns : 11-element CMA expected return vector.
    cov_matrix : 11x11 asset covariance matrix.
    n_paths : number of simulation paths (default 10,000).
    horizon : final year (inclusive); paths cover 1..horizon.
    seed : RNG seed for reproducibility. None for non-deterministic.

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.default_rng(seed)

    # Derive trust mean and covariance from CMAs (one-time)
    trust_mean = np.array([
        tc.trust_net_return(t, asset_returns) for t in tc.TRUST_NAMES
    ])
    trust_cov = tc.portfolio_covariance(cov_matrix)
    # Cholesky for correlated normal draws. Add a tiny ridge for numerical
    # stability if cov is borderline-singular.
    try:
        L = np.linalg.cholesky(trust_cov + np.eye(3) * 1e-12)
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition with floored eigenvalues
        evals, evecs = np.linalg.eigh(trust_cov)
        evals = np.maximum(evals, 0.0)
        L = evecs @ np.diag(np.sqrt(evals))

    sell_spreads = np.array([tc.TRUST_SELL_SPREADS[t] for t in tc.TRUST_NAMES])

    # Initial holdings (n_paths copies of the same starting state)
    init_aud = np.array([initial_weights.get(t, 0.0) * initial_value
                         for t in tc.TRUST_NAMES])
    holdings = np.tile(init_aud, (n_paths, 1)).astype(float)  # (n_paths, 3)

    # Pre-allocate outputs
    values = np.zeros((n_paths, horizon + 1))
    values[:, 0] = initial_value
    total_drawdowns = np.zeros(n_paths)
    total_spreads = np.zeros(n_paths)
    exhaustion_year = np.full(n_paths, -1, dtype=int)

    # Pre-build severity band ranges
    band_lo = {k: dr_v[0] for k, dr_v in SEVERITY_BANDS.items()}
    band_hi = {k: dr_v[1] for k, dr_v in SEVERITY_BANDS.items()}

    for year in range(1, horizon + 1):
        # 1. Draw drought severity
        sev_idx = rng.choice(4, size=n_paths, p=_DROUGHT_PROBS)
        severities = np.array([_DROUGHT_OUTCOMES[i] for i in sev_idx])

        # 2. Draw relief amounts (zero where 'None')
        drawdowns = np.zeros(n_paths)
        for sev in ["Mild", "Moderate", "Severe"]:
            mask = severities == sev
            n = int(mask.sum())
            if n > 0:
                drawdowns[mask] = rng.uniform(band_lo[sev], band_hi[sev], size=n)

        # 3. Draw trust net returns: z ~ N(0, I); r = mean + L @ z
        z = rng.standard_normal(size=(n_paths, 3))
        trust_returns = trust_mean + z @ L.T  # (n_paths, 3)

        # Apply growth (only to non-exhausted paths)
        active = exhaustion_year < 0
        holdings[active] = holdings[active] * (1 + trust_returns[active])
        # Floor at 0 in case a trust posts a return < -100% from a fat tail
        holdings = np.maximum(holdings, 0.0)

        # 4. Apply redemption
        if np.any(drawdowns > 0):
            # Only paths that have something to redeem
            new_h, sp, unmet = _vectorised_redeem(holdings, drawdowns, sell_spreads)
            holdings = new_h
            total_spreads += sp
            # Track delivered drawdown (excluding unmet portion)
            delivered = drawdowns - np.maximum(unmet, 0.0)
            total_drawdowns += np.maximum(delivered, 0.0)
            # Exhaustion check
            newly_exhausted = (unmet > 1e-3) & (exhaustion_year < 0)
            exhaustion_year[newly_exhausted] = year
            # Once exhausted, hold at zero
            holdings[exhaustion_year >= 0] = 0.0

        # Record path values
        path_values = holdings.sum(axis=1)
        # Mark already-exhausted paths as zero too
        path_values[exhaustion_year >= 0] = 0.0
        values[:, year] = path_values

        # Catch any paths whose post-drawdown value is essentially zero
        becomes_zero = (path_values < 1.0) & (exhaustion_year < 0)
        exhaustion_year[becomes_zero] = year

    return MonteCarloResult(
        n_paths=n_paths,
        horizon=horizon,
        values=values,
        total_drawdowns=total_drawdowns,
        total_spreads=total_spreads,
        exhaustion_year=exhaustion_year,
    )
