"""Capital and operating cost calculations."""
from __future__ import annotations

import numpy as np

from .inputs import WTEMasterInputs


def capex_schedule(inp: WTEMasterInputs) -> np.ndarray:
    """Generate the capital expenditure schedule across construction."""

    timeline = inp.timeline
    construction_periods = timeline.construction_periods
    profile = inp.costs.capex_spend_profile or [1 / construction_periods] * construction_periods
    if abs(sum(profile) - 1.0) > 1e-6:
        raise ValueError("Capex spend profile must sum to 1.")

    sched = np.zeros(timeline.n)
    sched[:construction_periods] = np.array(profile) * inp.costs.capex_total_usd
    return sched


def opex_block(inp: WTEMasterInputs, energy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calculate operating expenditure schedules."""

    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year
    esc = inp.costs.opex_escalation

    fixed = np.array(
        [inp.costs.fixed_om_usd_pa * ((1 + esc) ** (i // ppy)) / ppy for i in range(periods)]
    )
    variable = energy["tonnes"] * inp.costs.variable_om_usd_per_t
    disposal = energy["tonnes"] * inp.costs.landfill_disposal_usd_per_t
    insurance = np.array(
        [
            inp.costs.insurance_pct_of_capex_pa
            * inp.costs.capex_total_usd
            * ((1 + esc) ** (i // ppy))
            / ppy
            for i in range(periods)
        ]
    )
    maintenance = np.array(
        [
            inp.costs.maintenance_pct_of_capex_pa
            * inp.costs.capex_total_usd
            * ((1 + esc) ** (i // ppy))
            / ppy
            for i in range(periods)
        ]
    )

    total = fixed + variable + disposal + insurance + maintenance
    return {
        "fixed_om": fixed,
        "variable_om": variable,
        "disposal": disposal,
        "insurance": insurance,
        "maintenance": maintenance,
        "total_opex": total,
    }
