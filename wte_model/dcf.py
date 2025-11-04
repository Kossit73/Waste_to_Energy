"""Discounted cash flow model assembly."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .costs import capex_schedule, opex_block
from .energy import energy_block
from .finance import (
    coverage_ratios,
    debt_schedule,
    depreciation_schedule,
    tax_block,
    working_capital_block,
)
from .inputs import WTEMasterInputs
from .revenue import revenue_block


def _npv(rate: float, cashflows: List[float]) -> float:
    return sum(cf / ((1 + rate) ** t) for t, cf in enumerate(cashflows))


def _irr(cashflows: List[float], guess: float = 0.1) -> float:
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


def cashflow_model(inp: WTEMasterInputs) -> Dict[str, Dict[str, np.ndarray] | np.ndarray | float]:
    """Run the full project cash flow model."""

    timeline = inp.timeline
    periods = timeline.n

    energy = energy_block(inp)
    revenue = revenue_block(inp, energy)
    capex = capex_schedule(inp)
    opex = opex_block(inp, energy)
    depr = depreciation_schedule(inp, capex)

    ebitda = revenue["total_revenue"] - opex["total_opex"]
    depreciation_total = depr["total"]

    debt = debt_schedule(inp, capex["total"])

    for iteration in range(2):
        interest_cash = debt["interest_cash"]
        taxable_income = ebitda - depreciation_total - interest_cash
        tax = tax_block(inp, taxable_income, revenue["total_revenue"], energy)
        wc = working_capital_block(inp, revenue, opex)
        cfads = ebitda - tax["cash_tax"] - wc["cash_effect"]

        if debt["needs_cfads"] and iteration == 0:
            debt = debt_schedule(inp, capex["total"], cfads=cfads)
            continue
        break

    # Recompute with final debt schedule to ensure consistency
    interest_cash = debt["interest_cash"]
    taxable_income = ebitda - depreciation_total - interest_cash
    tax = tax_block(inp, taxable_income, revenue["total_revenue"], energy)
    wc = working_capital_block(inp, revenue, opex)
    cfads = ebitda - tax["cash_tax"] - wc["cash_effect"]

    debt_service = debt["debt_service"]
    fees = debt["fees"]
    equity_invest = capex["total"] - debt["debt_draws"]

    withholding_rate = inp.finance.tax.withholding_rate
    equity_before_withholding = cfads - debt_service - fees - equity_invest
    withholding = np.where(equity_before_withholding > 0, equity_before_withholding * withholding_rate, 0.0)
    equity_cf = equity_before_withholding - withholding

    project_cash = revenue["total_revenue"] - opex["total_opex"] - tax["cash_tax"] - wc["cash_effect"]

    proj_cf = list(project_cash - capex["total"])
    irr_proj = _irr(proj_cf)
    irr_eq = _irr(list(equity_cf))

    coverage = coverage_ratios(debt, cfads, project_cash, timeline, inp.finance.discount_rate)
    debt["llcr"] = coverage["llcr"]
    debt["plcr"] = coverage["plcr"]

    return {
        "energy": energy,
        "rev": revenue,
        "capex": capex,
        "opex": opex,
        "depr": depr,
        "debt": debt,
        "tax": tax,
        "working_capital": wc,
        "ebitda": ebitda,
        "cfads": cfads,
        "equity_cf": equity_cf,
        "withholding": withholding,
        "irr_eq": irr_eq,
        "irr_proj": irr_proj,
        "dscr": np.where(debt_service > 0, cfads / debt_service, np.nan),
    }

