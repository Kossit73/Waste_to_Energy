"""Energy block calculations."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .inputs import WTEMasterInputs
from .utils import expand_series

MJ_PER_MWH = 3.6e3


def _resolve_profile(
    base: float,
    profile: "Sequence[float]" | None,
    periods: int,
    periods_per_year: int,
    label: str,
) -> np.ndarray:
    """Return a per-period array for the supplied profile."""

    if profile is None:
        return np.full(periods, base, dtype=float)
    return expand_series(profile, periods, periods_per_year, label=label)


def energy_block(inp: WTEMasterInputs) -> dict[str, np.ndarray]:
    """Compute throughput and net electricity production arrays."""

    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year
    tech = inp.tech

    if tech.tonnes_profile is not None:
        tonnes = expand_series(tech.tonnes_profile, periods, ppy, label="tonnes_profile")
    else:
        base_tonnes = tech.msw_tonnes_pa / ppy
        tonnes = np.full(periods, base_tonnes, dtype=float)

    availability = _resolve_profile(tech.availability, tech.availability_profile, periods, ppy, "availability")
    tonnes = tonnes * availability

    if tech.degradation_rate:
        degrade_curve = np.array([(1 - tech.degradation_rate) ** (i / ppy) for i in range(periods)])
        tonnes *= degrade_curve

    boiler_eff = _resolve_profile(tech.boiler_efficiency, tech.boiler_efficiency_profile, periods, ppy, "boiler_efficiency")
    electrical_eff = _resolve_profile(
        tech.electrical_efficiency,
        tech.electrical_efficiency_profile,
        periods,
        ppy,
        "electrical_efficiency",
    )
    parasitic = _resolve_profile(
        tech.parasitic_load_frac, tech.parasitic_profile, periods, ppy, "parasitic_load"
    )

    thermal_mwh = tonnes * 1000 * tech.lhv_mj_per_kg / MJ_PER_MWH
    steam_mwh = thermal_mwh * boiler_eff
    gross_mwh = steam_mwh * electrical_eff
    parasitic_mwh = gross_mwh * parasitic
    net_mwh = gross_mwh - parasitic_mwh
    heat_mwh = np.clip(thermal_mwh - steam_mwh, 0.0, None)

    return {
        "tonnes": tonnes,
        "availability": availability,
        "thermal_mwh": thermal_mwh,
        "steam_mwh": steam_mwh,
        "gross_mwh": gross_mwh,
        "parasitic_mwh": parasitic_mwh,
        "net_mwh": net_mwh,
        "heat_mwh": heat_mwh,
    }

