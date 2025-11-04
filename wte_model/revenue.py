"""Revenue block calculations."""
from __future__ import annotations

import numpy as np

from .inputs import WTEMasterInputs


def _escalate(initial: float, rate: float, periods: int, per_year: int) -> np.ndarray:
    """Return an array with annual compounding escalation."""

    annual = [initial * (1 + rate) ** year for year in range(int(periods / per_year) + 1)]
    return np.array([annual[i // per_year] for i in range(periods)])


def revenue_block(inp: WTEMasterInputs, energy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build the revenue schedule arrays."""

    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year

    ppa = _escalate(inp.revenue.ppa_price_usd_per_mwh, inp.revenue.ppa_escalation, periods, ppy)
    gate = _escalate(inp.revenue.gate_fee_usd_per_t, inp.revenue.gate_fee_escalation, periods, ppy)
    other_rate = _escalate(
        inp.revenue.metal_recovery_usd_per_t + inp.revenue.ash_revenue_usd_per_t,
        inp.revenue.other_escalation,
        periods,
        ppy,
    )

    energy_revenue = energy["net_mwh"] * ppa
    gate_revenue = energy["tonnes"] * gate
    other_revenue = energy["tonnes"] * other_rate

    total = energy_revenue + gate_revenue + other_revenue
    return {
        "energy_revenue": energy_revenue,
        "gate_revenue": gate_revenue,
        "other_revenue": other_revenue,
        "total_revenue": total,
    }
