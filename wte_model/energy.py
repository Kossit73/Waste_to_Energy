"""Energy block calculations."""
from __future__ import annotations

import numpy as np

from .inputs import WTEMasterInputs

MJ_PER_MWH = 3.6e3


def energy_block(inp: WTEMasterInputs) -> dict[str, np.ndarray]:
    """Compute throughput and net electricity production arrays."""

    timeline = inp.timeline
    periods = timeline.n
    tonnes = np.full(
        periods,
        (inp.tech.msw_tonnes_pa * inp.tech.availability) / timeline.periods_per_year,
    )

    energy_mj = (
        tonnes
        * 1000
        * inp.tech.lhv_mj_per_kg
        * inp.tech.boiler_efficiency
    )
    net_mwh = (
        energy_mj / MJ_PER_MWH
        * inp.tech.electrical_efficiency
        * (1 - inp.tech.parasitic_load_frac)
    )
    return {
        "tonnes": tonnes,
        "net_mwh": net_mwh,
    }
