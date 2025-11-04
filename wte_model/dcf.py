"""Discounted cash flow model assembly."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .costs import capex_schedule, opex_block
from .energy import energy_block
from .finance import debt_schedule, depreciation_schedule, tax_block
from .inputs import WTEMasterInputs
from .revenue import revenue_block


def _npv(rate: float, cashflows: list[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def _irr(cashflows: list[float], guess: float = 0.1) -> float:
    """Compute internal rate of return using Newton-Raphson with bisection fallback."""

    rate = guess
    for _ in range(50):
        npv = _npv(rate, cashflows)
        d = sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cashflows))
        if abs(d) < 1e-12:
            break
        new_rate = rate - npv / d
        if -0.9999 < new_rate < 10:
            rate = new_rate
        if abs(npv) < 1e-10:
            return rate
    lo, hi = -0.9, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        v = _npv(mid, cashflows)
        if abs(v) < 1e-8:
            return mid
        if v > 0:
            lo = mid
        else:
            hi = mid
    return rate


def cashflow_model(inp: WTEMasterInputs) -> Dict[str, np.ndarray | float]:
    """Run the full project cash flow model."""

    timeline = inp.timeline
    ppy = timeline.periods_per_year
    n = timeline.n

    energy = energy_block(inp)
    revenue = revenue_block(inp, energy)
    capex = capex_schedule(inp)
    opex = opex_block(inp, energy)
    debt = debt_schedule(inp, capex, 0)
    depr = depreciation_schedule(inp, capex)

    ebitda = revenue["total_revenue"] - opex["total_opex"]
    ebt = ebitda - debt["interest"] - depr
    tax = tax_block(inp, ebt, depr)
    cfads = ebitda - tax

    debt_service = debt["debt_service"]
    cfw = cfads - debt_service

    equity_invest = capex - debt["debt_draws"]
    equity_cf = -equity_invest + cfw - debt["fees"]

    irr_eq = _irr(list(equity_cf))
    proj_cf = list(-capex) + list(revenue["total_revenue"] - opex["total_opex"] - tax)
    irr_proj = _irr(proj_cf)
    with np.errstate(divide="ignore", invalid="ignore"):
        dscr = np.where(debt_service > 0, cfads / debt_service, np.nan)

    return {
        "energy": energy,
        "rev": revenue,
        "opex": opex,
        "capex": capex,
        "debt": debt,
        "depr": depr,
        "tax": tax,
        "ebitda": ebitda,
        "cfads": cfads,
        "equity_cf": equity_cf,
        "irr_eq": irr_eq,
        "irr_proj": irr_proj,
        "dscr": dscr,
    }
