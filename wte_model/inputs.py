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


def default_inputs() -> WTEMasterInputs:
    """Return a representative default input set for the model.

    The values mirror the fallback assumptions used by the Excel loader so
    they provide a sensible baseline when a workbook is not supplied.
    """

    return WTEMasterInputs(
        timeline=Timeline(),
        tech=TechAssumptions(
            msw_tonnes_pa=300_000.0,
            lhv_mj_per_kg=9.0,
            boiler_efficiency=0.85,
            electrical_efficiency=0.22,
            availability=0.90,
            parasitic_load_frac=0.10,
        ),
        revenue=RevenueAssumptions(
            ppa_price_usd_per_mwh=110.0,
            gate_fee_usd_per_t=25.0,
            heat_price_usd_per_mwh=0.0,
            metal_recovery_usd_per_t=0.0,
            ash_revenue_usd_per_t=0.0,
            ppa_escalation=0.0,
            gate_fee_escalation=0.0,
            other_escalation=0.0,
        ),
        costs=CostAssumptions(
            capex_total_usd=180_000_000.0,
            capex_spend_profile=None,
            fixed_om_usd_pa=10_000_000.0,
            variable_om_usd_per_t=15.0,
            landfill_disposal_usd_per_t=5.0,
            insurance_pct_of_capex_pa=0.0075,
            maintenance_pct_of_capex_pa=0.02,
            opex_escalation=0.02,
        ),
        finance=FinanceAssumptions(
            debt_ratio=0.70,
            interest_rate=0.08,
            tenor_years=12,
            grace_years=2,
            upfront_fee_pct=0.01,
            dscr_min=1.20,
            tax_rate=0.25,
            depr_years=15,
            working_cap_days=30,
            discount_rate=0.10,
        ),
    )
