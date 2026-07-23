"""Input data structures for the waste-to-energy model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


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
class PriceCurve:
    """Pricing helper capturing escalation, indexation, and FX adjustments."""

    base: float
    annual_escalation: float = 0.0
    periodic_multipliers: Optional[Sequence[float]] = None
    index_curve: Optional[Sequence[float]] = None
    fx_curve: Optional[Sequence[float]] = None
    adders: Optional[Sequence[float]] = None


@dataclass
class RevenueStream:
    """Generic revenue stream configuration."""

    name: str
    driver: str  # e.g. "net_mwh", "tonnes", "heat_mwh", "custom"
    price_curve: PriceCurve
    share: float = 1.0
    quantity_profile: Optional[Sequence[float]] = None
    seasonality: Optional[Sequence[float]] = None
    currency: str = "USD"
    notes: str | None = None


@dataclass
class TechAssumptions:
    """Technical assumptions for the facility with optional variability."""

    msw_tonnes_pa: float
    lhv_mj_per_kg: float
    boiler_efficiency: float
    electrical_efficiency: float
    availability: float
    parasitic_load_frac: float = 0.10
    tonnes_profile: Optional[Sequence[float]] = None
    availability_profile: Optional[Sequence[float]] = None
    boiler_efficiency_profile: Optional[Sequence[float]] = None
    electrical_efficiency_profile: Optional[Sequence[float]] = None
    parasitic_profile: Optional[Sequence[float]] = None
    degradation_rate: float = 0.0  # annual degradation applied to throughput


@dataclass
class CapexItem:
    """Individual capital expenditure line with its own depreciation settings."""

    name: str
    amount: float
    spend_profile: Optional[Sequence[float]] = None
    life_years: int = 15
    bonus_depreciation_pct: float = 0.0
    residual_value_pct: float = 0.0
    method: str = "straight_line"  # or "declining_balance"
    declining_balance_rate: Optional[float] = None
    inflation_curve: Optional[Sequence[float]] = None


@dataclass
class OpexComponent:
    """Operating expenditure component driven by throughput or a custom series."""

    name: str
    category: str = "opex"  # e.g. direct_costs, staff, other
    fixed_annual: float = 0.0
    variable_per_unit: float = 0.0
    driver: str = "tonnes"
    custom_quantity: Optional[Sequence[float]] = None
    escalation: float = 0.0
    inflation_curve: Optional[Sequence[float]] = None
    seasonality: Optional[Sequence[float]] = None
    notes: str | None = None


@dataclass
class RevenueAssumptions:
    """Collection of commercial streams feeding the revenue build."""

    streams: List[RevenueStream] = field(default_factory=list)

    def _match_stream(
        self,
        driver: str,
        *,
        keywords: Optional[Sequence[str]] = None,
    ) -> Optional[RevenueStream]:
        driver = driver.lower()
        keyword_list = [k.lower() for k in (keywords or []) if k]
        for stream in self.streams:
            if stream.driver.lower() != driver:
                continue
            if keyword_list and not any(k in stream.name.lower() for k in keyword_list):
                continue
            return stream
        for stream in self.streams:
            if stream.driver.lower() == driver:
                return stream
        return None

    def _ensure_stream(self, name: str, driver: str, *, keywords: Optional[Sequence[str]] = None) -> RevenueStream:
        stream = self._match_stream(driver, keywords=keywords or [name])
        if stream is None:
            stream = RevenueStream(name=name, driver=driver, price_curve=PriceCurve(base=0.0))
            self.streams.append(stream)
        return stream

    @property
    def ppa_price_usd_per_mwh(self) -> float:
        stream = self._match_stream("net_mwh", keywords=("ppa", "electricity"))
        return float(stream.price_curve.base) if stream else 0.0

    @ppa_price_usd_per_mwh.setter
    def ppa_price_usd_per_mwh(self, value: float) -> None:
        stream = self._ensure_stream("PPA", "net_mwh", keywords=("ppa", "electricity"))
        stream.price_curve.base = value

    @property
    def ppa_escalation(self) -> float:
        stream = self._match_stream("net_mwh", keywords=("ppa", "electricity"))
        return float(stream.price_curve.annual_escalation) if stream else 0.0

    @ppa_escalation.setter
    def ppa_escalation(self, value: float) -> None:
        stream = self._ensure_stream("PPA", "net_mwh", keywords=("ppa", "electricity"))
        stream.price_curve.annual_escalation = value

    @property
    def gate_fee_usd_per_t(self) -> float:
        stream = self._match_stream("tonnes", keywords=("gate", "tipping"))
        return float(stream.price_curve.base) if stream else 0.0

    @gate_fee_usd_per_t.setter
    def gate_fee_usd_per_t(self, value: float) -> None:
        stream = self._ensure_stream("Gate fees", "tonnes", keywords=("gate", "tipping"))
        stream.price_curve.base = value

    @property
    def gate_fee_escalation(self) -> float:
        stream = self._match_stream("tonnes", keywords=("gate", "tipping"))
        return float(stream.price_curve.annual_escalation) if stream else 0.0

    @gate_fee_escalation.setter
    def gate_fee_escalation(self, value: float) -> None:
        stream = self._ensure_stream("Gate fees", "tonnes", keywords=("gate", "tipping"))
        stream.price_curve.annual_escalation = value

    @property
    def heat_price_usd_per_mwh(self) -> float:
        stream = self._match_stream("heat_mwh", keywords=("heat", "steam"))
        return float(stream.price_curve.base) if stream else 0.0

    @heat_price_usd_per_mwh.setter
    def heat_price_usd_per_mwh(self, value: float) -> None:
        stream = self._ensure_stream("Heat offtake", "heat_mwh", keywords=("heat", "steam"))
        stream.price_curve.base = value

    def _byproduct_streams(self) -> List[RevenueStream]:
        return [
            stream
            for stream in self.streams
            if stream.driver.lower() == "tonnes" and "gate" not in stream.name.lower()
        ]

    def _ensure_byproduct(self, name: str, keyword: str) -> RevenueStream:
        stream = self._match_stream("tonnes", keywords=(keyword,))
        if stream is None:
            stream = RevenueStream(name=name, driver="tonnes", price_curve=PriceCurve(base=0.0))
            self.streams.append(stream)
        return stream

    @property
    def metal_recovery_usd_per_t(self) -> float:
        stream = self._match_stream("tonnes", keywords=("metal",))
        return float(stream.price_curve.base) if stream else 0.0

    @metal_recovery_usd_per_t.setter
    def metal_recovery_usd_per_t(self, value: float) -> None:
        stream = self._ensure_byproduct("Metals", "metal")
        stream.price_curve.base = value

    @property
    def ash_revenue_usd_per_t(self) -> float:
        stream = self._match_stream("tonnes", keywords=("ash",))
        return float(stream.price_curve.base) if stream else 0.0

    @ash_revenue_usd_per_t.setter
    def ash_revenue_usd_per_t(self, value: float) -> None:
        stream = self._ensure_byproduct("Ash", "ash")
        stream.price_curve.base = value

    @property
    def other_escalation(self) -> float:
        streams = self._byproduct_streams()
        return float(streams[0].price_curve.annual_escalation) if streams else 0.0

    @other_escalation.setter
    def other_escalation(self, value: float) -> None:
        for stream in self._byproduct_streams():
            stream.price_curve.annual_escalation = value


@dataclass
class CostAssumptions:
    """Capital and operating expenditure assumptions."""

    capex_total_usd: float = 0.0
    capex_spend_profile: Optional[List[float]] = None
    capex_items: List[CapexItem] = field(default_factory=list)
    fixed_om_usd_pa: float = 0.0
    variable_om_usd_per_t: float = 0.0
    landfill_disposal_usd_per_t: float = 0.0
    insurance_pct_of_capex_pa: float = 0.0
    maintenance_pct_of_capex_pa: float = 0.0
    opex_escalation: float = 0.0
    opex_components: List[OpexComponent] = field(default_factory=list)


@dataclass
class WorkingCapitalAssumptions:
    """Granular working capital configuration."""

    receivable_days: float = 45.0
    prepaid_days: float = 30.0
    prepaid_absolute: float = 0.0
    other_current_asset_pct_revenue: float = 0.0
    other_asset_absolute: float = 0.0
    inventory_days: float = 20.0
    payable_days: float = 45.0
    other_current_liability_pct_opex: float = 0.0
    accrued_expense_absolute: float = 0.0


@dataclass
class TaxAssumptions:
    """Corporate tax and incentive configuration."""

    corporate_rate: float = 0.25
    minimum_tax_rate: float = 0.0  # applied to revenue if higher than income tax
    withholding_rate: float = 0.0
    allow_loss_carryforward: bool = True
    carryforward_years: int = 20
    holiday_years: int = 0
    holiday_start_offset: int = 0  # periods after COD before holiday starts
    carbon_credit_per_mwh: float = 0.0
    other_incentives: Optional[Sequence[float]] = None


@dataclass
class DebtFacility:
    """Debt facility with configurable draw and amortisation behaviour."""

    name: str
    draw_ratio: float = 0.0  # share of project capex financed
    commitment: Optional[float] = None
    draw_profile: Optional[Sequence[float]] = None
    interest_rate: float = 0.08
    tenor_years: int = 12
    grace_years: int = 0
    amortization: str = "annuity"  # annuity, straight_line, custom, sculpted
    custom_amort_profile: Optional[Sequence[float]] = None
    target_dscr: float = 1.20
    sculpt_from_cfads: bool = False
    cash_sweep_pct: float = 0.0
    cash_sweep_trigger: float = 1.0
    upfront_fee_pct: float = 0.0
    interest_during_construction: bool = True


@dataclass
class FinanceAssumptions:
    """Financing assumptions."""

    debt_ratio: float = 0.70  # legacy single-facility fields remain for fallback
    interest_rate: float = 0.08
    tenor_years: int = 12
    grace_years: int = 0
    upfront_fee_pct: float = 0.0
    dscr_min: float = 1.20
    tax_rate: float = 0.25
    depr_years: int = 15
    working_cap_days: int = 30
    discount_rate: float = 0.10
    debt_facilities: List[DebtFacility] = field(default_factory=list)
    tax: TaxAssumptions = field(default_factory=TaxAssumptions)
    working_capital: WorkingCapitalAssumptions = field(default_factory=WorkingCapitalAssumptions)
    macro_indices: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tax is None:
            self.tax = TaxAssumptions(corporate_rate=self.tax_rate)
        else:
            if self.tax_rate and self.tax.corporate_rate != self.tax_rate:
                self.tax.corporate_rate = self.tax_rate


@dataclass
class WTEMasterInputs:
    """Container for all model inputs."""

    timeline: Timeline
    tech: TechAssumptions
    revenue: RevenueAssumptions
    costs: CostAssumptions
    finance: FinanceAssumptions


def default_inputs() -> WTEMasterInputs:
    """Return a representative default input set for the enhanced model."""

    timeline = Timeline()

    tech = TechAssumptions(
        msw_tonnes_pa=300_000.0,
        lhv_mj_per_kg=9.0,
        boiler_efficiency=0.85,
        electrical_efficiency=0.22,
        availability=0.90,
        parasitic_load_frac=0.10,
        degradation_rate=0.005,
    )

    revenue_streams = [
        RevenueStream(
            name="PPA",
            driver="net_mwh",
            share=1.0,
            price_curve=PriceCurve(base=110.0, annual_escalation=0.02),
        ),
        RevenueStream(
            name="Gate fees",
            driver="tonnes",
            share=1.0,
            price_curve=PriceCurve(base=25.0, annual_escalation=0.02),
        ),
        RevenueStream(
            name="Metals recovery",
            driver="tonnes",
            share=1.0,
            price_curve=PriceCurve(base=3.0, annual_escalation=0.02),
        ),
    ]

    capex_items = [
        CapexItem(name="Land", amount=10_000_000.0, life_years=0, residual_value_pct=1.0),
        CapexItem(name="Civil works", amount=40_000_000.0, life_years=25, residual_value_pct=0.05),
        CapexItem(
            name="Process equipment",
            amount=85_000_000.0,
            life_years=20,
            residual_value_pct=0.05,
            bonus_depreciation_pct=0.1,
        ),
        CapexItem(
            name="Power island",
            amount=35_000_000.0,
            life_years=20,
            residual_value_pct=0.05,
        ),
        CapexItem(name="Other assets", amount=15_000_000.0, life_years=10, residual_value_pct=0.1),
    ]

    opex_components = [
        OpexComponent(
            name="Operations staff",
            category="staff",
            fixed_annual=3_000_000.0,
            escalation=0.02,
        ),
        OpexComponent(
            name="Maintenance",
            category="maintenance",
            variable_per_unit=8.0,
            driver="tonnes",
            escalation=0.02,
        ),
        OpexComponent(
            name="Residue disposal",
            category="direct_costs",
            variable_per_unit=5.0,
            driver="tonnes",
            escalation=0.02,
        ),
        OpexComponent(
            name="Insurance",
            category="other",
            fixed_annual=1_350_000.0,
            escalation=0.02,
        ),
        OpexComponent(
            name="General administration",
            category="other",
            fixed_annual=1_200_000.0,
            escalation=0.02,
        ),
        OpexComponent(
            name="Energy cost",
            category="direct_costs",
            variable_per_unit=3.5,
            driver="net_mwh",
            escalation=0.02,
        ),
    ]

    costs = CostAssumptions(
        capex_total_usd=sum(item.amount for item in capex_items),
        capex_items=capex_items,
        fixed_om_usd_pa=10_000_000.0,
        variable_om_usd_per_t=15.0,
        landfill_disposal_usd_per_t=5.0,
        insurance_pct_of_capex_pa=0.0075,
        maintenance_pct_of_capex_pa=0.02,
        opex_escalation=0.02,
        opex_components=opex_components,
    )

    senior = DebtFacility(
        name="Senior debt",
        draw_ratio=0.65,
        interest_rate=0.07,
        tenor_years=15,
        grace_years=2,
        amortization="annuity",
        upfront_fee_pct=0.01,
        cash_sweep_pct=0.0,
        interest_during_construction=True,
    )
    mezz = DebtFacility(
        name="Mezzanine",
        draw_ratio=0.05,
        interest_rate=0.10,
        tenor_years=10,
        grace_years=3,
        amortization="straight_line",
        upfront_fee_pct=0.01,
        interest_during_construction=False,
    )

    finance = FinanceAssumptions(
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
        debt_facilities=[senior, mezz],
        tax=TaxAssumptions(corporate_rate=0.25, minimum_tax_rate=0.01, withholding_rate=0.05),
        working_capital=WorkingCapitalAssumptions(
            receivable_days=45.0,
            prepaid_days=30.0,
            inventory_days=20.0,
            payable_days=45.0,
            other_current_asset_pct_revenue=0.01,
            other_current_liability_pct_opex=0.01,
        ),
    )

    return WTEMasterInputs(
        timeline=timeline,
        tech=tech,
        revenue=RevenueAssumptions(streams=revenue_streams),
        costs=costs,
        finance=finance,
    )

