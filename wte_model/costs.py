"""Capital and operating cost calculations."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .inputs import CapexItem, OpexComponent, WTEMasterInputs
from .utils import expand_series, normalise_profile


def _capex_item_schedule(
    item: CapexItem,
    timeline,
    default_profile,
) -> np.ndarray:
    """Return the spend schedule for a single capex item."""

    periods = timeline.n
    construction_periods = timeline.construction_periods
    if item.spend_profile is not None:
        profile = expand_series(
            item.spend_profile,
            construction_periods,
            timeline.periods_per_year,
            label=f"{item.name}_profile",
            allow_partial=True,
        )[:construction_periods]
    elif default_profile is not None:
        profile = expand_series(
            default_profile,
            construction_periods,
            timeline.periods_per_year,
            label="capex_profile",
            allow_partial=True,
        )[:construction_periods]
    else:
        profile = np.full(construction_periods, 1 / max(construction_periods, 1))
    profile = normalise_profile(profile, label=f"{item.name} capex profile")

    schedule = np.zeros(periods)
    schedule[:construction_periods] = profile * item.amount

    if item.inflation_curve is not None and construction_periods:
        inflation = expand_series(
            item.inflation_curve,
            construction_periods,
            timeline.periods_per_year,
            label=f"{item.name}_inflation",
            allow_partial=True,
        )[:construction_periods]
        schedule[:construction_periods] *= inflation

    return schedule


def capex_schedule(inp: WTEMasterInputs) -> Dict[str, Dict[str, np.ndarray] | np.ndarray]:
    """Generate the capital expenditure schedule across construction."""

    timeline = inp.timeline
    total = np.zeros(timeline.n)
    items: Dict[str, np.ndarray] = {}

    if inp.costs.capex_items:
        for item in inp.costs.capex_items:
            schedule = _capex_item_schedule(item, timeline, inp.costs.capex_spend_profile)
            items[item.name] = schedule
            total += schedule
    else:
        construction_periods = timeline.construction_periods
        profile = inp.costs.capex_spend_profile or [1 / max(construction_periods, 1)] * construction_periods
        profile_arr = normalise_profile(profile, label="capex_profile")
        schedule = np.zeros(timeline.n)
        schedule[:construction_periods] = profile_arr[:construction_periods] * inp.costs.capex_total_usd
        items["total_capex"] = schedule
        total += schedule

    return {"total": total, "items": items}


def _component_schedule(
    component: OpexComponent,
    timeline,
    energy: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Return the fixed, variable and total arrays for a component."""

    periods = timeline.n
    ppy = timeline.periods_per_year

    escalation = np.array([(1 + component.escalation) ** (i // ppy) for i in range(periods)])
    inflation = (
        expand_series(component.inflation_curve, periods, ppy, label=f"{component.name}_inflation")
        if component.inflation_curve is not None
        else np.ones(periods)
    )
    seasonality = (
        expand_series(component.seasonality, periods, ppy, label=f"{component.name}_seasonality")
        if component.seasonality is not None
        else np.ones(periods)
    )

    fixed = np.full(periods, component.fixed_annual / ppy, dtype=float) * escalation * inflation * seasonality

    driver = component.driver.lower()
    if driver == "tonnes":
        quantity = energy["tonnes"]
    elif driver == "net_mwh":
        quantity = energy["net_mwh"]
    elif driver == "gross_mwh":
        quantity = energy["gross_mwh"]
    elif driver == "heat_mwh":
        quantity = energy["heat_mwh"]
    elif driver == "custom":
        quantity = expand_series(
            component.custom_quantity,
            periods,
            ppy,
            label=f"{component.name}_quantity",
        )
    else:
        raise ValueError(f"Unsupported opex driver '{component.driver}' for {component.name}.")

    variable = quantity * component.variable_per_unit * escalation * inflation
    total = fixed + variable
    return {"fixed": fixed, "variable": variable, "total": total}


def opex_block(inp: WTEMasterInputs, energy: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
    """Calculate operating expenditure schedules."""

    timeline = inp.timeline
    periods = timeline.n

    if inp.costs.opex_components:
        components: Dict[str, Dict[str, np.ndarray]] = {}
        category_totals: Dict[str, np.ndarray] = {}
        fixed_total = np.zeros(periods)
        variable_total = np.zeros(periods)
        total = np.zeros(periods)

        for component in inp.costs.opex_components:
            sched = _component_schedule(component, timeline, energy)
            components[component.name] = sched
            fixed_total += sched["fixed"]
            variable_total += sched["variable"]
            total += sched["total"]
            category_totals.setdefault(component.category, np.zeros(periods))
            category_totals[component.category] += sched["total"]

        return {
            "components": components,
            "category_totals": category_totals,
            "fixed_total": fixed_total,
            "variable_total": variable_total,
            "total_opex": total,
            "fixed_om": fixed_total,
            "variable_om": variable_total,
            "disposal": category_totals.get("disposal", np.zeros(periods)),
            "insurance": category_totals.get("insurance", np.zeros(periods)),
            "maintenance": category_totals.get("maintenance", np.zeros(periods)),
        }

    # Fallback to legacy simple build if no granular components are supplied
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
        "components": {},
        "category_totals": {},
        "fixed_total": fixed,
        "variable_total": variable,
        "total_opex": total,
        "fixed_om": fixed,
        "variable_om": variable,
        "disposal": disposal,
        "insurance": insurance,
        "maintenance": maintenance,
    }

