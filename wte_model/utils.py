"""Utility helpers for the waste-to-energy model."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def expand_series(
    series: Sequence[float] | None,
    periods: int,
    periods_per_year: int,
    *,
    label: str = "series",
    allow_partial: bool = False,
) -> np.ndarray:
    """Expand a shorthand series to the full model horizon.

    The helper accepts a variety of common layouts:
    * exact-length arrays (already periodic)
    * annual values (len == periods / periods_per_year)
    * intra-year seasonality curves (len == periods_per_year)
    * factors whose length divides the total periods

    Parameters
    ----------
    series:
        Input values. ``None`` raises ``ValueError`` because the caller should
        guard optional inputs.
    periods:
        Total periods in the model horizon.
    periods_per_year:
        Model frequency (e.g. 4 for quarterly).
    label:
        Friendly label used in validation errors.
    allow_partial:
        If ``True`` and the provided sequence is shorter than the horizon, the
        tail is padded by repeating the final value. Useful for capex profiles
        that cover only the construction window.
    """

    if series is None:
        raise ValueError(f"{label} is None and cannot be expanded.")

    arr = np.asarray(list(series), dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value.")

    if arr.size == periods:
        return arr

    if periods % arr.size == 0:
        return np.tile(arr, periods // arr.size)

    annual_periods = periods // periods_per_year
    if annual_periods * periods_per_year == periods and arr.size == annual_periods:
        return np.repeat(arr, periods_per_year)[:periods]

    if arr.size == periods_per_year:
        reps = (periods + periods_per_year - 1) // periods_per_year
        return np.tile(arr, reps)[:periods]

    if allow_partial and arr.size <= periods:
        padded = np.full(periods, arr[-1], dtype=float)
        padded[: arr.size] = arr
        return padded

    raise ValueError(
        f"Cannot expand {label} of length {arr.size} to {periods} periods with"
        f" {periods_per_year} periods/year."
    )


def normalise_profile(profile: Iterable[float], *, label: str = "profile") -> np.ndarray:
    """Return a numpy array that sums to 1.0.

    Zero entries are permitted but the profile must contain at least one
    positive value.
    """

    arr = np.asarray(list(profile), dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} must contain at least one value.")
    total = arr.sum()
    if total <= 0:
        raise ValueError(f"{label} must sum to a positive value; got {total}.")
    return arr / total


def discount_factors(rate: float, periods: int) -> np.ndarray:
    """Return per-period discount factors at the supplied periodic rate."""

    if rate <= -1:
        raise ValueError("Discount rate must exceed -100%.")
    return np.array([1 / ((1 + rate) ** t) for t in range(periods)], dtype=float)

