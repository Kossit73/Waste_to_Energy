"""Debt, depreciation and tax calculations."""
from __future__ import annotations

import numpy as np

from .inputs import WTEMasterInputs


def _pmt(rate: float, nper: int, pv: float) -> float:
    """Annuity payment helper."""

    if nper <= 0:
        return 0.0
    if rate == 0:
        return pv / nper
    return (rate * pv) / (1 - (1 + rate) ** (-nper))


def debt_draws_from_capex(capex: np.ndarray, debt_ratio: float) -> np.ndarray:
    """Proportion of capex financed by debt."""

    return capex * debt_ratio


def debt_schedule(
    inp: WTEMasterInputs, capex: np.ndarray, start_index: int = 0
) -> dict[str, np.ndarray]:
    """Generate debt drawdown, service, and balance arrays."""

    timeline = inp.timeline
    n = timeline.n
    ppy = timeline.periods_per_year

    debt_draws = debt_draws_from_capex(capex, inp.finance.debt_ratio)
    principal = np.zeros(n)
    interest = np.zeros(n)
    balance = np.zeros(n)
    fees = np.zeros(n)

    if n > 0:
        fees[start_index] = inp.finance.upfront_fee_pct * debt_draws.sum()

    for i in range(n):
        balance[i] = (balance[i - 1] if i > 0 else 0) + debt_draws[i]
        interest[i] = balance[i] * (inp.finance.interest_rate / ppy)

    cod = timeline.construction_periods
    first_amort = cod + inp.finance.grace_years * ppy
    if first_amort < n:
        remain = balance[first_amort - 1] if first_amort > 0 else balance[0]
        amort_periods = min(inp.finance.tenor_years * ppy, n - first_amort)
        if amort_periods > 0 and remain > 0:
            annuity = _pmt(inp.finance.interest_rate / ppy, amort_periods, remain)
            for k in range(amort_periods):
                idx = first_amort + k
                prev_balance = balance[idx - 1] if idx > 0 else remain
                interest[idx] = prev_balance * (inp.finance.interest_rate / ppy)
                principal[idx] = annuity - interest[idx]
                balance[idx] = prev_balance - principal[idx]

    debt_service = principal + interest
    return {
        "debt_draws": debt_draws,
        "interest": interest,
        "principal": principal,
        "debt_service": debt_service,
        "balance": balance,
        "fees": fees,
    }


def working_capital(
    revenue: np.ndarray, opex: np.ndarray, days: int, ppy: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simplified working capital model."""

    wc = np.zeros_like(revenue)
    factor = days / 365.0
    for i, _ in enumerate(wc):
        wc[i] = (revenue[i] - opex[i]) * factor
    cash_effect = np.concatenate([[wc[0]], np.diff(wc)])
    return wc, cash_effect


def tax_block(inp: WTEMasterInputs, ebt: np.ndarray, depr: np.ndarray) -> np.ndarray:
    """Corporate income tax calculation (no loss carry-forward)."""

    return (ebt > 0) * ebt * inp.finance.tax_rate


def depreciation_schedule(inp: WTEMasterInputs, capex: np.ndarray) -> np.ndarray:
    """Straight-line depreciation schedule."""

    n = inp.timeline.n
    ppy = inp.timeline.periods_per_year
    cod = inp.timeline.construction_periods
    depr = np.zeros(n)
    base = capex.sum()
    sl = base / (inp.finance.depr_years * ppy)
    for i in range(cod, min(n, cod + inp.finance.depr_years * ppy)):
        depr[i] = sl
    return depr
