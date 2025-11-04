"""Revenue block calculations."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .inputs import PriceCurve, RevenueStream, WTEMasterInputs
from .utils import expand_series


def _price_array(curve: PriceCurve, periods: int, ppy: int, label: str) -> np.ndarray:
    """Construct a per-period price array from a :class:`PriceCurve`."""

    price = np.full(periods, curve.base, dtype=float)
    if curve.annual_escalation:
        annual_factor = np.array([(1 + curve.annual_escalation) ** (i // ppy) for i in range(periods)])
        price *= annual_factor
    if curve.periodic_multipliers is not None:
        price *= expand_series(curve.periodic_multipliers, periods, ppy, label=f"{label}_periodic")
    if curve.index_curve is not None:
        price *= expand_series(curve.index_curve, periods, ppy, label=f"{label}_index")
    if curve.fx_curve is not None:
        price *= expand_series(curve.fx_curve, periods, ppy, label=f"{label}_fx")
    if curve.adders is not None:
        price += expand_series(curve.adders, periods, ppy, label=f"{label}_adders")
    return price


def _quantity_for_stream(
    stream: RevenueStream,
    energy: Dict[str, np.ndarray],
    periods: int,
    ppy: int,
) -> np.ndarray:
    """Return the quantity driver for a stream."""

    driver = stream.driver.lower()
    if stream.quantity_profile is not None:
        quantity = expand_series(stream.quantity_profile, periods, ppy, label=f"{stream.name}_quantity")
    elif driver == "net_mwh":
        quantity = energy["net_mwh"] * stream.share
    elif driver == "gross_mwh":
        quantity = energy["gross_mwh"] * stream.share
    elif driver == "heat_mwh":
        quantity = energy["heat_mwh"] * stream.share
    elif driver == "tonnes":
        quantity = energy["tonnes"] * stream.share
    else:
        raise ValueError(f"Unsupported revenue driver '{stream.driver}' for {stream.name}.")

    if stream.seasonality is not None:
        quantity *= expand_series(stream.seasonality, periods, ppy, label=f"{stream.name}_seasonality")

    return quantity


def revenue_block(inp: WTEMasterInputs, energy: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | List[Dict[str, np.ndarray]]]:
    """Build the revenue schedule arrays."""

    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year

    streams_data: List[Dict[str, np.ndarray]] = []
    energy_revenue = np.zeros(periods)
    gate_revenue = np.zeros(periods)
    other_revenue = np.zeros(periods)
    total_revenue = np.zeros(periods)

    for stream in inp.revenue.streams:
        price = _price_array(stream.price_curve, periods, ppy, stream.name)
        quantity = _quantity_for_stream(stream, energy, periods, ppy)
        revenue = quantity * price
        streams_data.append({"name": stream.name, "price": price, "quantity": quantity, "revenue": revenue})
        total_revenue += revenue
        driver = stream.driver.lower()
        if driver in {"net_mwh", "gross_mwh", "heat_mwh"}:
            energy_revenue += revenue
        elif driver == "tonnes":
            gate_revenue += revenue
        else:
            other_revenue += revenue

    other_revenue += total_revenue - (energy_revenue + gate_revenue)

    return {
        "streams": streams_data,
        "energy_revenue": energy_revenue,
        "gate_revenue": gate_revenue,
        "other_revenue": other_revenue,
        "total_revenue": total_revenue,
    }

