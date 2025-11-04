"""Interactive Streamlit interface for the waste-to-energy financial model."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from wte_model import (
    CostAssumptions,
    FinanceAssumptions,
    RevenueAssumptions,
    TechAssumptions,
    Timeline,
    WTEMasterInputs,
    cashflow_model,
    default_inputs,
)
from wte_model.io_excel import load_inputs_from_xlsm


st.set_page_config(page_title="Waste-to-Energy Model", layout="wide")
st.title("Waste-to-Energy Financial Model")
st.caption(
    "Upload an Excel workbook or fine-tune the controls to evaluate project-level "
    "economics, debt sizing, and equity returns."
)


def _load_inputs_from_upload(uploaded_file) -> WTEMasterInputs:
    """Persist an uploaded workbook to disk and load it with the Excel parser."""

    suffix = Path(uploaded_file.name or "inputs.xlsm").suffix or ".xlsm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        return load_inputs_from_xlsm(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _parse_capex_profile(text: str, fallback: Optional[List[float]]) -> Optional[List[float]]:
    """Convert a comma-separated text field into a normalised spend profile."""

    if not text.strip():
        return fallback
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        st.warning("Capex profile could not be parsed; using fallback values.")
        return fallback
    if not values:
        return fallback
    total = sum(values)
    if total <= 0:
        st.warning("Capex profile must sum to a positive value; using fallback values.")
        return fallback
    if abs(total - 1.0) > 1e-6:
        st.info("Capex profile normalised to sum to 1.0.")
        values = [v / total for v in values]
    return values


uploaded_workbook = st.file_uploader(
    "Upload Excel assumptions (optional)", type=["xlsm", "xlsx"], accept_multiple_files=False
)

source_label = "Default inputs"
inputs = default_inputs()
if uploaded_workbook is not None:
    try:
        inputs = _load_inputs_from_upload(uploaded_workbook)
        source_label = uploaded_workbook.name or "Uploaded workbook"
        st.success(f"Loaded inputs from {source_label}.")
    except Exception as exc:  # pragma: no cover - user feedback path
        st.error(f"Failed to read workbook: {exc}")

st.caption(f"Using assumptions from: {source_label}")

tabs = st.tabs(["Timeline", "Technology", "Revenue", "Costs", "Finance"])

with tabs[0]:
    st.write("Configure the overall schedule and model resolution.")
    years = st.number_input("Operating years", min_value=1, max_value=60, value=inputs.timeline.years, step=1)
    build_months = st.number_input(
        "Construction months", min_value=1, max_value=120, value=inputs.timeline.build_months, step=1
    )
    start_year = st.number_input(
        "Start year", min_value=2000, max_value=2100, value=inputs.timeline.start_year, step=1
    )
    periods_per_year = st.number_input(
        "Periods per year", min_value=1, max_value=12, value=inputs.timeline.periods_per_year, step=1
    )

with tabs[1]:
    st.write("Update throughput and conversion efficiencies for the facility.")
    msw_tonnes_pa = st.number_input(
        "MSW throughput (t/a)", min_value=10_000.0, max_value=1_000_000.0, value=float(inputs.tech.msw_tonnes_pa), step=10_000.0
    )
    lhv_mj_per_kg = st.number_input(
        "Lower heating value (MJ/kg)", min_value=4.0, max_value=18.0, value=float(inputs.tech.lhv_mj_per_kg), step=0.1
    )
    boiler_efficiency = st.number_input(
        "Boiler efficiency", min_value=0.3, max_value=1.0, value=float(inputs.tech.boiler_efficiency), step=0.01
    )
    electrical_efficiency = st.number_input(
        "Electrical efficiency", min_value=0.1, max_value=0.4, value=float(inputs.tech.electrical_efficiency), step=0.01
    )
    availability = st.number_input(
        "Availability", min_value=0.5, max_value=1.0, value=float(inputs.tech.availability), step=0.01
    )
    parasitic_load_frac = st.number_input(
        "Parasitic load fraction", min_value=0.0, max_value=0.3, value=float(inputs.tech.parasitic_load_frac), step=0.01
    )

with tabs[2]:
    st.write("Set commercial terms for power, waste, and by-product revenues.")
    ppa_price = st.number_input(
        "PPA price (USD/MWh)", min_value=0.0, max_value=500.0, value=float(inputs.revenue.ppa_price_usd_per_mwh), step=1.0
    )
    gate_fee = st.number_input(
        "Gate fee (USD/t)", min_value=0.0, max_value=200.0, value=float(inputs.revenue.gate_fee_usd_per_t), step=1.0
    )
    heat_price = st.number_input(
        "Heat price (USD/MWh)", min_value=0.0, max_value=200.0, value=float(inputs.revenue.heat_price_usd_per_mwh), step=1.0
    )
    metal_recovery = st.number_input(
        "Metal recovery (USD/t)", min_value=0.0, max_value=200.0, value=float(inputs.revenue.metal_recovery_usd_per_t), step=1.0
    )
    ash_revenue = st.number_input(
        "Ash revenue (USD/t)", min_value=0.0, max_value=200.0, value=float(inputs.revenue.ash_revenue_usd_per_t), step=1.0
    )
    ppa_escalation = st.number_input(
        "PPA escalation (pa)", min_value=0.0, max_value=0.10, value=float(inputs.revenue.ppa_escalation), step=0.005
    )
    gate_fee_escalation = st.number_input(
        "Gate fee escalation (pa)", min_value=0.0, max_value=0.10, value=float(inputs.revenue.gate_fee_escalation), step=0.005
    )
    other_escalation = st.number_input(
        "By-product escalation (pa)", min_value=0.0, max_value=0.10, value=float(inputs.revenue.other_escalation), step=0.005
    )

with tabs[3]:
    st.write("Adjust capital costs and operating expenditure assumptions.")
    capex_total = st.number_input(
        "Total CAPEX (USD)", min_value=50_000_000.0, max_value=600_000_000.0, value=float(inputs.costs.capex_total_usd), step=5_000_000.0
    )
    capex_profile_text = st.text_input(
        "Capex spend profile (comma separated)",
        value=", ".join(f"{v:.3f}" for v in inputs.costs.capex_spend_profile) if inputs.costs.capex_spend_profile else "",
        help="Enter fractions that sum to 1 over the construction periods. Leave blank for an even spread.",
    )
    fixed_om = st.number_input(
        "Fixed O&M (USD/a)", min_value=0.0, max_value=50_000_000.0, value=float(inputs.costs.fixed_om_usd_pa), step=500_000.0
    )
    variable_om = st.number_input(
        "Variable O&M (USD/t)", min_value=0.0, max_value=200.0, value=float(inputs.costs.variable_om_usd_per_t), step=1.0
    )
    landfill_disposal = st.number_input(
        "Residue disposal (USD/t)", min_value=0.0, max_value=200.0, value=float(inputs.costs.landfill_disposal_usd_per_t), step=1.0
    )
    insurance_pct = st.number_input(
        "Insurance (% of CAPEX/a)", min_value=0.0, max_value=0.05, value=float(inputs.costs.insurance_pct_of_capex_pa), step=0.001
    )
    maintenance_pct = st.number_input(
        "Maintenance (% of CAPEX/a)", min_value=0.0, max_value=0.10, value=float(inputs.costs.maintenance_pct_of_capex_pa), step=0.001
    )
    opex_escalation = st.number_input(
        "Opex escalation (pa)", min_value=0.0, max_value=0.10, value=float(inputs.costs.opex_escalation), step=0.005
    )

with tabs[4]:
    st.write("Define the financing structure and valuation parameters.")
    debt_ratio = st.number_input(
        "Debt ratio", min_value=0.0, max_value=1.0, value=float(inputs.finance.debt_ratio), step=0.05
    )
    interest_rate = st.number_input(
        "Interest rate (pa)", min_value=0.0, max_value=0.25, value=float(inputs.finance.interest_rate), step=0.005
    )
    tenor_years = st.number_input(
        "Debt tenor (years)", min_value=1, max_value=30, value=int(inputs.finance.tenor_years), step=1
    )
    grace_years = st.number_input(
        "Grace period (years)", min_value=0, max_value=10, value=int(inputs.finance.grace_years), step=1
    )
    upfront_fee_pct = st.number_input(
        "Upfront fee", min_value=0.0, max_value=0.05, value=float(inputs.finance.upfront_fee_pct), step=0.001
    )
    tax_rate = st.number_input(
        "Corporate tax rate", min_value=0.0, max_value=0.5, value=float(inputs.finance.tax_rate), step=0.01
    )
    depr_years = st.number_input(
        "Depreciation period (years)", min_value=1, max_value=30, value=int(inputs.finance.depr_years), step=1
    )
    working_cap_days = st.number_input(
        "Working capital days", min_value=0, max_value=180, value=int(inputs.finance.working_cap_days), step=5
    )
    discount_rate = st.number_input(
        "Discount rate (pa)", min_value=0.0, max_value=0.30, value=float(inputs.finance.discount_rate), step=0.01
    )

capex_profile = _parse_capex_profile(capex_profile_text, inputs.costs.capex_spend_profile)

user_inputs = WTEMasterInputs(
    timeline=Timeline(
        years=int(years),
        build_months=int(build_months),
        start_year=int(start_year),
        periods_per_year=int(periods_per_year),
    ),
    tech=TechAssumptions(
        msw_tonnes_pa=float(msw_tonnes_pa),
        lhv_mj_per_kg=float(lhv_mj_per_kg),
        boiler_efficiency=float(boiler_efficiency),
        electrical_efficiency=float(electrical_efficiency),
        availability=float(availability),
        parasitic_load_frac=float(parasitic_load_frac),
    ),
    revenue=RevenueAssumptions(
        ppa_price_usd_per_mwh=float(ppa_price),
        gate_fee_usd_per_t=float(gate_fee),
        heat_price_usd_per_mwh=float(heat_price),
        metal_recovery_usd_per_t=float(metal_recovery),
        ash_revenue_usd_per_t=float(ash_revenue),
        ppa_escalation=float(ppa_escalation),
        gate_fee_escalation=float(gate_fee_escalation),
        other_escalation=float(other_escalation),
    ),
    costs=CostAssumptions(
        capex_total_usd=float(capex_total),
        capex_spend_profile=capex_profile,
        fixed_om_usd_pa=float(fixed_om),
        variable_om_usd_per_t=float(variable_om),
        landfill_disposal_usd_per_t=float(landfill_disposal),
        insurance_pct_of_capex_pa=float(insurance_pct),
        maintenance_pct_of_capex_pa=float(maintenance_pct),
        opex_escalation=float(opex_escalation),
    ),
    finance=FinanceAssumptions(
        debt_ratio=float(debt_ratio),
        interest_rate=float(interest_rate),
        tenor_years=int(tenor_years),
        grace_years=int(grace_years),
        upfront_fee_pct=float(upfront_fee_pct),
        dscr_min=inputs.finance.dscr_min,
        tax_rate=float(tax_rate),
        depr_years=int(depr_years),
        working_cap_days=int(working_cap_days),
        discount_rate=float(discount_rate),
    ),
)

results = cashflow_model(user_inputs)


def _format_pct(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_ratio(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.2f}x"


def _safe_nanmin(array: np.ndarray) -> float:
    array = np.asarray(array, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmin(array))


col1, col2, col3 = st.columns(3)
col1.metric("Equity IRR", _format_pct(results.get("irr_eq")))
col2.metric("Project IRR", _format_pct(results.get("irr_proj")))
col3.metric("Min DSCR", _format_ratio(_safe_nanmin(results.get("dscr", np.array([])))))

st.subheader("Cash Flow Overview")
periods = np.arange(user_inputs.timeline.n)
years_axis = user_inputs.timeline.start_year + (periods // user_inputs.timeline.periods_per_year)
summary = pd.DataFrame(
    {
        "Period": periods + 1,
        "Calendar Year": years_axis,
        "Tonnes": results["energy"]["tonnes"],
        "Net MWh": results["energy"]["net_mwh"],
        "Total revenue": results["rev"]["total_revenue"],
        "Total opex": results["opex"]["total_opex"],
        "EBITDA": results["ebitda"],
        "CFADS": results["cfads"],
        "Debt service": results["debt"]["debt_service"],
        "Equity cash flow": results["equity_cf"],
        "Capex": results["capex"],
        "Debt balance": results["debt"]["balance"],
    }
)

st.dataframe(summary.round(2), use_container_width=True)

cash_csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download cash flow CSV",
    data=cash_csv,
    file_name="wte_cashflows.csv",
    mime="text/csv",
)

dscr_series = pd.Series(results.get("dscr"), index=periods + 1, dtype="float64")
dscr_series.replace([np.inf, -np.inf], np.nan, inplace=True)
st.subheader("Debt Service Coverage Ratio")
st.line_chart(dscr_series, height=260)

if uploaded_workbook is not None:
    mapped = getattr(WTEMasterInputs, "_MAPPED_DEBUG", None)
    if mapped:
        with st.expander("Excel named-range mapping"):
            st.json(mapped)

st.info(
    "Use the tabs above to tweak assumptions or upload an Excel workbook. "
    "The results above update instantly so you can iterate on project scenarios."
)
