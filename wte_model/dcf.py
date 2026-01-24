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

    has_positive = any(cf > 0 for cf in cashflows)
    has_negative = any(cf < 0 for cf in cashflows)
    if not (has_positive and has_negative):
        return float("nan")

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
    f_lo = _npv(lo, cashflows)
    f_hi = _npv(hi, cashflows)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return float("nan")

    for _ in range(200):
        mid = (lo + hi) / 2
        v = _npv(mid, cashflows)
        if abs(v) < 1e-8:
            return mid
        if v > 0:
            lo = mid
        else:
            hi = mid
    return float("nan")


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

    wc = working_capital_block(inp, revenue, opex)
    debt = debt_schedule(inp, capex["total"])

    max_iter = 10
    tol = 1e-6
    tax = None
    cfads = None

    converged = False
    for iteration in range(max_iter):
        interest_cash = debt["interest_cash"]
        taxable_income = ebitda - depreciation_total - interest_cash
        tax = tax_block(inp, taxable_income, revenue["total_revenue"], energy)
        cfads = ebitda - tax["cash_tax"] - wc["cash_effect"]

        if not debt["needs_cfads"]:
            converged = True
            break

        updated = debt_schedule(inp, capex["total"], cfads=cfads)
        diff = np.max(np.abs(updated["debt_service"] - debt["debt_service"]))
        debt = updated
        if diff < tol:
            converged = True
            break

    # Ensure final cash flows align with converged debt schedule
    interest_cash = debt["interest_cash"]
    taxable_income = ebitda - depreciation_total - interest_cash
    tax = tax_block(inp, taxable_income, revenue["total_revenue"], energy)
    cfads = ebitda - tax["cash_tax"] - wc["cash_effect"]

    investment_total = capex["total"] + debt["interest_capitalised"]
    debt_funding = debt["funding_total"]

    debt_service = debt["debt_service"]
    fees = debt["fees"]
    equity_invest = investment_total - debt_funding

    withholding_rate = inp.finance.tax.withholding_rate
    equity_before_withholding = cfads - debt_service - fees - equity_invest
    withholding = np.where(equity_before_withholding > 0, equity_before_withholding * withholding_rate, 0.0)
    equity_cf = equity_before_withholding - withholding

    project_cash = revenue["total_revenue"] - opex["total_opex"] - tax["cash_tax"] - wc["cash_effect"]

    proj_cf = list(project_cash - investment_total)
    irr_proj = _irr(proj_cf)
    irr_eq = _irr(list(equity_cf))

    coverage = coverage_ratios(debt, cfads, project_cash, timeline, inp.finance.discount_rate)
    debt["llcr"] = coverage["llcr"]
    debt["plcr"] = coverage["plcr"]

    capex["interest_capitalised"] = debt["interest_capitalised"]
    capex["investment_total"] = investment_total

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
        "debt_iterations": iteration + 1,
        "debt_converged": converged,
        "dscr": np.where(debt_service > 0, cfads / debt_service, np.nan),
    }
