"""Input data structures for the waste-to-energy model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Timeline:
    """Project timeline settings."""

    years: int = 25
    build_months: int = 24
    start_year: int = 2026
    periods_per_year: int = 4  # e.g. quarterly model

    @property
    def n(self) -> int:
        """Total number of model periods."""

        return self.years * self.periods_per_year

    @property
    def construction_periods(self) -> int:
        """Number of model periods in the construction phase."""

        import math

        return int(math.ceil(self.build_months / (12 / self.periods_per_year)))


@dataclass
class TechAssumptions:
    """Technical assumptions for the facility."""

    msw_tonnes_pa: float
    lhv_mj_per_kg: float
    boiler_efficiency: float
    electrical_efficiency: float
    availability: float
    parasitic_load_frac: float = 0.10


@dataclass
class RevenueAssumptions:
    """Commercial assumptions used in the revenue build."""

    ppa_price_usd_per_mwh: float
    gate_fee_usd_per_t: float
    heat_price_usd_per_mwh: float = 0.0
    metal_recovery_usd_per_t: float = 0.0
    ash_revenue_usd_per_t: float = 0.0
    ppa_escalation: float = 0.0
    gate_fee_escalation: float = 0.0
    other_escalation: float = 0.0


@dataclass
class CostAssumptions:
    """Capital and operating expenditure assumptions."""

    capex_total_usd: float
    capex_spend_profile: Optional[List[float]] = None
    fixed_om_usd_pa: float = 0.0
    variable_om_usd_per_t: float = 0.0
    landfill_disposal_usd_per_t: float = 0.0
    insurance_pct_of_capex_pa: float = 0.0
    maintenance_pct_of_capex_pa: float = 0.0
    opex_escalation: float = 0.0


@dataclass
class FinanceAssumptions:
    """Financing assumptions."""

    debt_ratio: float
    interest_rate: float
    tenor_years: int
    grace_years: int = 0
    upfront_fee_pct: float = 0.0
    dscr_min: float = 1.20
    tax_rate: float = 0.25
    depr_years: int = 15
    working_cap_days: int = 30
    discount_rate: float = 0.10


@dataclass
class WTEMasterInputs:
    """Container for all model inputs."""

    timeline: Timeline
    tech: TechAssumptions
    revenue: RevenueAssumptions
    costs: CostAssumptions
    finance: FinanceAssumptions
