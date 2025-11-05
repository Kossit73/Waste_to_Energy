"""Interactive Streamlit workspace for the waste-to-energy financial model."""

from __future__ import annotations

import importlib.util
import copy
from dataclasses import dataclass, replace
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


_REQUIRED_PACKAGES = ("numpy", "pandas", "streamlit")
_missing = [pkg for pkg in _REQUIRED_PACKAGES if importlib.util.find_spec(pkg) is None]
if _missing:
    missing_list = ", ".join(sorted(_missing))
    raise ModuleNotFoundError(
        "Missing required dependencies: "
        f"{missing_list}. Install them with 'pip install -r requirements.txt' before running the Streamlit app."
    )

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from streamlit.delta_generator import DeltaGenerator

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


st.set_page_config(page_title="Waste-to-Energy Model", layout="wide")
st.title("Waste-to-Energy Financial Workspace")
st.caption(
    "Configure assumptions in the sections below to build a comprehensive project finance "
    "model with dashboards, statements, sensitivities, and scenarios."
)


@dataclass
class ProjectionSettings:
    start_year: int
    end_year: int
    periods_per_year: int

    @property
    def years(self) -> int:
        return max(1, self.end_year - self.start_year + 1)


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


def _update_table_state(key: str, df: pd.DataFrame) -> None:
    st.session_state[key] = df.copy()


def _ensure_state_df(key: str, data: pd.DataFrame) -> pd.DataFrame:
    if key not in st.session_state:
        _update_table_state(key, data)
    return st.session_state[key]


def _blank_row(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    row = {}
    for col in df.columns:
        series = df[col]
        row[col] = 0.0 if is_numeric_dtype(series) else ""
    return pd.DataFrame([row])


def _section_header(title: str, key: str, *, level: str = "subheader") -> bool:
    """Render a section header with an Edit toggle."""

    cols = st.columns([5, 1])
    header_fn = getattr(cols[0], level)
    header_fn(title)
    with cols[1]:
        edit_enabled = st.checkbox(
            "Edit",
            key=f"{key}_edit_toggle",
            help="Enable editing to modify the default figures for this section.",
        )
    return edit_enabled


def _number_input_control(
    label: str,
    key: str,
    *,
    container: Optional[DeltaGenerator] = None,
    edit_enabled: bool,
    **kwargs,
):
    """Number input that honours the section edit toggle and mirrors values into session state."""

    current = st.session_state.get(key, 0)
    if isinstance(current, (np.integer, int)) and not isinstance(current, bool):
        base_value = int(current)
        caster = int
    else:
        try:
            base_value = float(current)
        except (TypeError, ValueError):
            base_value = 0.0
        caster = float

    widget_fn = container.number_input if container is not None else st.number_input
    widget_value = widget_fn(
        label,
        value=base_value,
        key=f"{key}_widget",
        disabled=not edit_enabled,
        **kwargs,
    )
    if edit_enabled:
        st.session_state[key] = caster(widget_value)
    return st.session_state.get(key, caster(widget_value))


def _text_area_control(
    label: str,
    key: str,
    *,
    edit_enabled: bool,
    **kwargs,
):
    """Text area helper that syncs with session state and respects edit mode."""

    current = st.session_state.get(key, "")
    widget_value = st.text_area(
        label,
        value=current,
        key=f"{key}_widget",
        disabled=not edit_enabled,
        **kwargs,
    )
    if edit_enabled:
        st.session_state[key] = widget_value
    return st.session_state.get(key, widget_value)


def _editable_table(
    key: str,
    data: pd.DataFrame,
    *,
    column_config: Optional[Dict[str, st.column_config.BaseColumn]] = None,
    allow_row_controls: bool = True,
    new_row_factory: Optional[Callable[[], pd.DataFrame]] = None,
    edit_enabled: bool = True,
) -> pd.DataFrame:
    base = _ensure_state_df(key, data.copy())

    if allow_row_controls and edit_enabled:
        ctrl_cols = st.columns(2)
        with ctrl_cols[0]:
            if st.button("Add row", key=f"add_{key}"):
                template = base if not base.empty else data
                new_row_df = new_row_factory() if new_row_factory else None
                if new_row_df is None or new_row_df.empty:
                    new_row_df = _blank_row(template)
                if new_row_df is not None:
                    combined = pd.concat([base, new_row_df], ignore_index=True)
                    _update_table_state(key, combined)
                    st.experimental_rerun()
        with ctrl_cols[1]:
            if base.empty:
                st.write("No rows to remove")
            else:
                idx_options = list(range(len(base)))
                remove_idx = st.selectbox(
                    "Row to remove",
                    idx_options,
                    format_func=lambda i: f"Row {i + 1}",
                    key=f"remove_idx_{key}",
                )
                if st.button("Remove row", key=f"remove_{key}"):
                    trimmed = base.drop(base.index[remove_idx]).reset_index(drop=True)
                    _update_table_state(key, trimmed)
                    st.experimental_rerun()
    elif allow_row_controls and not edit_enabled:
        st.caption("Enable edit mode to add or remove rows.")

    edited = st.data_editor(
        base,
        key=f"editor_{key}",
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config or {},
        disabled=not edit_enabled,
    )
    if edit_enabled:
        _update_table_state(key, edited)
        return edited
    return base


def _reset_scalar_values(values: Dict[str, Any]) -> None:
    for key, value in values.items():
        st.session_state[key] = value


def _reset_table_group(defaults: Dict[str, pd.DataFrame], *, mode: str) -> None:
    for key, df in defaults.items():
        if mode == "defaults":
            _update_table_state(key, df.copy())
        elif mode == "clean":
            empty_df = pd.DataFrame(columns=df.columns)
            _update_table_state(key, empty_df)


def _compute_initial_investment_schedule(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        item = row.get("Item", "")
        try:
            cost = float(row.get("Cost", 0.0))
        except (TypeError, ValueError):
            cost = 0.0
        try:
            life_years = max(1, int(row.get("Life (years)", 1)))
        except (TypeError, ValueError):
            life_years = 1
        depreciation_rate = 1.0 / life_years
        yearly_depreciation = cost * depreciation_rate
        monthly_depreciation = yearly_depreciation / 12.0
        records.append(
            {
                "Item": item,
                "Cost": cost,
                "Life (years)": life_years,
                "Depreciation rate (%)": depreciation_rate * 100.0,
                "Yearly depreciation": yearly_depreciation,
                "Monthly depreciation": monthly_depreciation,
            }
        )
    schedule = pd.DataFrame.from_records(records)
    if not schedule.empty:
        schedule["Accumulated depreciation (year 1)"] = schedule["Yearly depreciation"]
        schedule["Net book value (year 1)"] = schedule["Cost"] - schedule["Yearly depreciation"]
    return schedule


def _annualise(series: Iterable[float], ppy: int) -> pd.Series:
    values = np.asarray(list(series), dtype=float)
    if values.size == 0:
        return pd.Series(dtype=float)
    periods = np.arange(values.size)
    years = periods // max(1, ppy)
    df = pd.DataFrame({"year": years, "value": values})
    annual = df.groupby("year", as_index=False)["value"].sum()
    annual.index = annual["year"]
    return annual["value"]


def _irr(values: Iterable[float]) -> float:
    cashflows = [float(v) for v in values]
    if not any(cashflows):
        return float("nan")
    rate = 0.1
    for _ in range(50):
        denom = [(1 + rate) ** t for t in range(len(cashflows))]
        npv = sum(cf / d for cf, d in zip(cashflows, denom))
        d_np = sum(-t * cf / ((1 + rate) ** (t + 1)) for t, cf in enumerate(cashflows))
        if abs(d_np) < 1e-12:
            break
        new_rate = rate - npv / d_np
        if -0.9999 < new_rate < 10:
            rate = new_rate
        if abs(npv) < 1e-10:
            return rate
    lo, hi = -0.9, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        denom = [(1 + mid) ** t for t in range(len(cashflows))]
        npv = sum(cf / d for cf, d in zip(cashflows, denom))
        if abs(npv) < 1e-8:
            return mid
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return rate


def _npv(rate: float, values: Iterable[float]) -> float:
    return float(sum(cf / ((1 + rate) ** t) for t, cf in enumerate(values)))


def _scenario_multiplier(value: Any) -> float:
    """Normalise scenario adjustments expressed as decimals or percentages."""

    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        return 0.0
    if abs(multiplier) > 1.0:
        multiplier /= 100.0
    return multiplier


def _build_summary_tables(
    inputs: WTEMasterInputs,
    results: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Assemble the period, annual, and cumulative summary tables for exports."""

    energy = results["energy"]
    revenue = results["rev"]
    capex_total = results["capex"]["total"]
    opex_total = results["opex"]["total_opex"]
    tax_cash = results["tax"]["cash_tax"]
    debt = results["debt"]

    periods = np.arange(inputs.timeline.n)
    years_axis = inputs.timeline.start_year + (periods // inputs.timeline.periods_per_year)

    summary = pd.DataFrame(
        {
            "Period": periods + 1,
            "Calendar Year": years_axis,
            "Tonnes": energy["tonnes"],
            "Net MWh": energy["net_mwh"],
            "Total revenue": revenue["total_revenue"],
            "Total opex": opex_total,
            "EBITDA": results["ebitda"],
            "CFADS": results["cfads"],
            "Debt service": debt["debt_service"],
            "Equity cash flow": results["equity_cf"],
            "Capex": capex_total,
            "Debt balance": debt["balance"],
            "Tax": tax_cash,
        }
    )

    summary_ann = summary.groupby("Calendar Year", as_index=False).agg(
        {
            "Tonnes": "sum",
            "Net MWh": "sum",
            "Total revenue": "sum",
            "Total opex": "sum",
            "EBITDA": "sum",
            "CFADS": "sum",
            "Debt service": "sum",
            "Equity cash flow": "sum",
            "Capex": "sum",
            "Tax": "sum",
        }
    )

    summary_cumulative = summary.copy()
    for col in ["Total revenue", "Total opex", "Equity cash flow", "Capex", "CFADS"]:
        summary_cumulative[f"Cumulative {col}"] = summary_cumulative[col].cumsum()

    production_annual_series = _annualise(energy["tonnes"], inputs.timeline.periods_per_year)

    return summary, summary_ann, summary_cumulative, production_annual_series


def _ensure_scenario_payload(
    scenario_name: str,
    base_inputs: WTEMasterInputs,
    scenario_df: pd.DataFrame,
    base_results: Optional[Dict[str, Any]] = None,
) -> Tuple[WTEMasterInputs, Dict[str, Any]]:
    """Return cached inputs/results for the requested scenario, building if missing."""

    payloads: Dict[str, Tuple[WTEMasterInputs, Dict[str, Any]]] = st.session_state.setdefault(
        "scenario_payloads", {}
    )
    if scenario_name in payloads:
        return payloads[scenario_name]

    inputs_copy = copy.deepcopy(base_inputs)

    if scenario_name != "Base Case":
        if scenario_df is not None and not scenario_df.empty and "Scenario" in scenario_df.columns:
            candidates = (
                scenario_df["Scenario"].astype(str).str.strip().reset_index(drop=True)
            )
            if scenario_name in candidates.values:
                row_idx = candidates[candidates == scenario_name].index[0]
                row = scenario_df.iloc[row_idx]
            else:
                row = None
        else:
            row = None

        if row is not None:
            ppa_adj = _scenario_multiplier(row.get("PPA adjustment", 0.0))
            gate_adj = _scenario_multiplier(row.get("Gate fee adjustment", 0.0))
            capex_adj = _scenario_multiplier(row.get("CAPEX adjustment", 0.0))

            if ppa_adj:
                current = inputs_copy.revenue.ppa_price_usd_per_mwh
                inputs_copy.revenue.ppa_price_usd_per_mwh = current * (1 + ppa_adj)
            if gate_adj:
                current = inputs_copy.revenue.gate_fee_usd_per_t
                inputs_copy.revenue.gate_fee_usd_per_t = current * (1 + gate_adj)
            if capex_adj:
                inputs_copy.costs.capex_total_usd *= 1 + capex_adj
                if inputs_copy.costs.capex_items:
                    adjusted_items = [
                        replace(item, amount=item.amount * (1 + capex_adj))
                        for item in inputs_copy.costs.capex_items
                    ]
                    inputs_copy.costs.capex_items = adjusted_items
        else:
            st.info(
                f"Scenario '{scenario_name}' is not defined in the configuration table; using base inputs."
            )

    if scenario_name == "Base Case" and base_results is not None:
        results_payload = copy.deepcopy(base_results)
    else:
        results_payload = cashflow_model(inputs_copy)

    payloads[scenario_name] = (inputs_copy, results_payload)
    st.session_state["scenario_payloads"] = payloads
    return inputs_copy, results_payload


def _generate_excel_bytes(
    model_inputs: WTEMasterInputs,
    results: Dict[str, Any],
    scenario_name: str,
) -> bytes:
    """Create an Excel workbook containing the key model outputs."""

    summary, summary_ann, summary_cumulative, _ = _build_summary_tables(model_inputs, results)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({"Scenario": [scenario_name]}).to_excel(
            writer, sheet_name="Scenario", index=False
        )
        summary.to_excel(writer, sheet_name="Period Summary", index=False)
        summary_ann.to_excel(writer, sheet_name="Annual Summary", index=False)
        summary_cumulative.to_excel(writer, sheet_name="Cumulative", index=False)
        pd.DataFrame(results["energy"]).to_excel(writer, sheet_name="Energy", index=False)
        pd.DataFrame(results["rev"]).to_excel(writer, sheet_name="Revenue", index=False)
        pd.DataFrame(results["opex"]).to_excel(writer, sheet_name="Opex", index=False)
        pd.DataFrame(results["capex"]).to_excel(writer, sheet_name="Capex", index=False)
        pd.DataFrame(results["debt"]).to_excel(writer, sheet_name="Debt", index=False)
        pd.DataFrame({"Equity cash flow": results["equity_cf"]}).to_excel(
            writer, sheet_name="Equity CF", index=False
        )
        pd.DataFrame({"CFADS": results["cfads"]}).to_excel(
            writer, sheet_name="CFADS", index=False
        )

    output.seek(0)
    return output.getvalue()


def _set_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


source_label = "Default inputs"
inputs = default_inputs()
st.caption(f"Using assumptions from: {source_label}")

projection_defaults = ProjectionSettings(
    start_year=inputs.timeline.start_year,
    end_year=inputs.timeline.start_year + inputs.timeline.years - 1,
    periods_per_year=inputs.timeline.periods_per_year,
)

_set_default("projection_start_year", projection_defaults.start_year)
_set_default("projection_end_year", projection_defaults.end_year)
_set_default("projection_ppy", projection_defaults.periods_per_year)

_set_default("msw_tonnes_pa", float(inputs.tech.msw_tonnes_pa))
_set_default("lhv_mj_per_kg", float(inputs.tech.lhv_mj_per_kg))
_set_default("boiler_efficiency", float(inputs.tech.boiler_efficiency))
_set_default("electrical_efficiency", float(inputs.tech.electrical_efficiency))
_set_default("availability", float(inputs.tech.availability))
_set_default("parasitic_load", float(inputs.tech.parasitic_load_frac))

_set_default("ppa_price", float(inputs.revenue.ppa_price_usd_per_mwh))
_set_default("gate_fee", float(inputs.revenue.gate_fee_usd_per_t))
_set_default("heat_price", float(inputs.revenue.heat_price_usd_per_mwh))
_set_default("metal_recovery", float(inputs.revenue.metal_recovery_usd_per_t))
_set_default("ash_revenue", float(inputs.revenue.ash_revenue_usd_per_t))
_set_default("ppa_escalation", float(inputs.revenue.ppa_escalation))
_set_default("gate_fee_escalation", float(inputs.revenue.gate_fee_escalation))
_set_default("other_escalation", float(inputs.revenue.other_escalation))

_set_default("capex_total", float(inputs.costs.capex_total_usd))
capex_profile_default_text = (
    ", ".join(f"{v:.3f}" for v in inputs.costs.capex_spend_profile)
    if inputs.costs.capex_spend_profile
    else ""
)
_set_default("capex_profile_text", capex_profile_default_text)
_set_default("fixed_om", float(inputs.costs.fixed_om_usd_pa))
_set_default("variable_om", float(inputs.costs.variable_om_usd_per_t))
_set_default("landfill_disposal", float(inputs.costs.landfill_disposal_usd_per_t))
_set_default("insurance_pct", float(inputs.costs.insurance_pct_of_capex_pa))
_set_default("maintenance_pct", float(inputs.costs.maintenance_pct_of_capex_pa))
_set_default("opex_escalation", float(inputs.costs.opex_escalation))

_set_default("debt_ratio", float(inputs.finance.debt_ratio))
_set_default("interest_rate", float(inputs.finance.interest_rate))
_set_default("tenor_years", int(inputs.finance.tenor_years))
_set_default("grace_years", int(inputs.finance.grace_years))
_set_default("upfront_fee_pct", float(inputs.finance.upfront_fee_pct))
_set_default("tax_rate", float(inputs.finance.tax_rate))
_set_default("depr_years", int(inputs.finance.depr_years))
_set_default("working_cap_days", int(inputs.finance.working_cap_days))
_set_default("discount_rate", float(inputs.finance.discount_rate))



global_defaults = pd.DataFrame(
    [
        {"Parameter": "Corporate tax rate (%)", "Value": round(inputs.finance.tax_rate * 100, 2)},
        {"Parameter": "Investor share capital (%)", "Value": 60.0},
        {"Parameter": "Owner share capital (%)", "Value": 40.0},
        {"Parameter": "Terminal growth (%)", "Value": 2.0},
        {"Parameter": "Capital gains tax rate (%)", "Value": 5.0},
        {"Parameter": "Payback threshold (years)", "Value": 12},
    ]
)

initial_investment_defaults = pd.DataFrame(
    [
        {"Item": "Equipment", "Cost": 90_000_000, "Life (years)": 15},
        {"Item": "Machinery", "Cost": 25_000_000, "Life (years)": 12},
        {"Item": "PP&E land", "Cost": 12_000_000, "Life (years)": 40},
        {"Item": "Building", "Cost": 18_000_000, "Life (years)": 25},
        {"Item": "Other", "Cost": 8_000_000, "Life (years)": 5},
    ]
)

revenue_defaults = pd.DataFrame(
    [
        {
            "Revenue stream": "Electricity",
            "Price": inputs.revenue.ppa_price_usd_per_mwh,
            "Escalation (%)": inputs.revenue.ppa_escalation * 100,
        },
        {
            "Revenue stream": "Gate fees",
            "Price": inputs.revenue.gate_fee_usd_per_t,
            "Escalation (%)": inputs.revenue.gate_fee_escalation * 100,
        },
        {
            "Revenue stream": "By-products",
            "Price": inputs.revenue.metal_recovery_usd_per_t + inputs.revenue.ash_revenue_usd_per_t,
            "Escalation (%)": inputs.revenue.other_escalation * 100,
        },
    ]
)

production_annual_defaults = pd.DataFrame(
    [{"Year": inputs.timeline.start_year + i, "Throughput (t)": inputs.tech.msw_tonnes_pa} for i in range(5)]
)

production_monthly_defaults = pd.DataFrame(
    [{"Month": m + 1, "Throughput (t)": inputs.tech.msw_tonnes_pa / 12.0} for m in range(12)]
)

direct_costs_monthly_defaults = pd.DataFrame(
    [
        {
            "Month": m + 1,
            "Feedstock cost": 0.0,
            "Residue disposal": inputs.costs.landfill_disposal_usd_per_t * (inputs.tech.msw_tonnes_pa / 12.0),
        }
        for m in range(12)
    ]
)

staff_monthly_defaults = pd.DataFrame(
    [
        {"Role": "Operations", "Monthly cost": 250_000},
        {"Role": "Maintenance", "Monthly cost": 180_000},
        {"Role": "Administration", "Monthly cost": 120_000},
    ]
)

other_opex_monthly_defaults = pd.DataFrame(
    [
        {
            "Category": "Insurance",
            "Monthly cost": inputs.costs.capex_total_usd * inputs.costs.insurance_pct_of_capex_pa / 12.0,
        },
        {"Category": "Service contract", "Monthly cost": 150_000},
        {"Category": "General administration", "Monthly cost": inputs.costs.fixed_om_usd_pa / 12.0},
        {"Category": "Sales & marketing", "Monthly cost": 60_000},
        {"Category": "Research & development", "Monthly cost": 40_000},
        {"Category": "Energy cost", "Monthly cost": 90_000},
    ]
)

accounts_receivable_defaults = pd.DataFrame(
    [
        {"Metric": "Receivables (days)", "Value": 45},
        {"Metric": "Prepaid expenses (USD)", "Value": 1_200_000},
        {"Metric": "Other assets (USD)", "Value": 850_000},
    ]
)

inventory_payable_defaults = pd.DataFrame(
    [
        {"Metric": "Inventory days", "Value": 20},
        {"Metric": "Accounts payable days", "Value": 35},
        {"Metric": "Accrued expenses (USD)", "Value": 1_000_000},
    ]
)

loan_schedule_defaults = pd.DataFrame(
    [
        {
            "Facility": "Senior debt",
            "Base amount": inputs.costs.capex_total_usd * inputs.finance.debt_ratio,
            "Interest rate (%)": inputs.finance.interest_rate * 100,
            "Loan type": "Annuity",
            "Start year": projection_defaults.start_year,
            "Duration (years)": inputs.finance.tenor_years,
            "Grace (years)": inputs.finance.grace_years,
        }
    ]
)

tax_schedule_defaults = pd.DataFrame(
    [
        {
            "Tax": "Corporate",
            "Rate (%)": inputs.finance.tax_rate * 100,
            "Timing adjustment (months)": 0,
            "Notes": "Paid quarterly",
        },
        {
            "Tax": "Withholding",
            "Rate (%)": 5.0,
            "Timing adjustment (months)": 1,
            "Notes": "Applies to distributions",
        },
    ]
)

inflation_schedule_defaults = pd.DataFrame(
    [
        {"Category": "Operating costs", "Inflation rate (%)": inputs.costs.opex_escalation * 100},
        {"Category": "Revenue", "Inflation rate (%)": inputs.revenue.ppa_escalation * 100},
        {"Category": "Capital", "Inflation rate (%)": 2.5},
    ]
)

risk_schedule_defaults = pd.DataFrame(
    [
        {"Risk": "Feedstock supply", "Probability (%)": 15.0, "Impact (USD)": 5_000_000, "Mitigation": "Long-term contracts"},
        {"Risk": "Technology performance", "Probability (%)": 10.0, "Impact (USD)": 8_000_000, "Mitigation": "OEM warranties"},
        {"Risk": "Regulatory change", "Probability (%)": 8.0, "Impact (USD)": 6_000_000, "Mitigation": "Policy monitoring"},
    ]
)

sensitivity_config_defaults = pd.DataFrame(
    [
        {"Driver": "PPA price", "Low": -0.1, "Base": 0.0, "High": 0.1},
        {"Driver": "Gate fee", "Low": -0.15, "Base": 0.0, "High": 0.15},
        {"Driver": "CAPEX", "Low": -0.1, "Base": 0.0, "High": 0.1},
    ]
)

scenario_config_defaults = pd.DataFrame(
    [
        {"Scenario": "Base", "PPA adjustment": 0.0, "Gate fee adjustment": 0.0, "CAPEX adjustment": 0.0},
        {"Scenario": "Upside", "PPA adjustment": 0.1, "Gate fee adjustment": 0.05, "CAPEX adjustment": -0.05},
        {"Scenario": "Downside", "PPA adjustment": -0.08, "Gate fee adjustment": -0.05, "CAPEX adjustment": 0.08},
    ]
)

goal_seek_defaults = pd.DataFrame(
    [
        {"Target metric": "Equity IRR", "Target value": 0.15, "Variable": "PPA price"},
        {"Target metric": "DSCR", "Target value": 1.35, "Variable": "Debt ratio"},
    ]
)

monte_carlo_defaults = pd.DataFrame(
    [
        {
            "Variable": "PPA price",
            "Distribution": "Normal",
            "Mean": inputs.revenue.ppa_price_usd_per_mwh,
            "Std dev": inputs.revenue.ppa_price_usd_per_mwh * 0.05,
        },
        {
            "Variable": "CAPEX",
            "Distribution": "Triangular",
            "Mean": inputs.costs.capex_total_usd,
            "Std dev": inputs.costs.capex_total_usd * 0.08,
        },
    ]
)

break_even_defaults = pd.DataFrame(
    [
        {"Input": "CAPEX", "Value": inputs.costs.capex_total_usd},
        {"Input": "Feedstock (waste) cost", "Value": 0.0},
        {"Input": "Electricity price (USD/kWh)", "Value": inputs.revenue.ppa_price_usd_per_mwh / 1000},
        {"Input": "OPEX per tonne", "Value": inputs.costs.variable_om_usd_per_t + inputs.costs.landfill_disposal_usd_per_t},
        {"Input": "Plant availability (%)", "Value": inputs.tech.availability * 100},
    ]
)

parameter_naming_defaults = pd.DataFrame(
    [
        {"Parameter": "Electricity price per kWh", "Preferred name": "Tariff_kWh"},
        {"Parameter": "OPEX per ton of waste", "Preferred name": "OPEX_per_tonne"},
        {"Parameter": "Plant availability / uptime", "Preferred name": "Availability_factor"},
        {"Parameter": "CAPEX", "Preferred name": "Total_CAPEX"},
        {"Parameter": "Corporate tax rate", "Preferred name": "Corp_tax"},
        {"Parameter": "Debt ratio", "Preferred name": "Debt_to_capital"},
        {"Parameter": "Interest rate", "Preferred name": "Debt_interest"},
        {"Parameter": "Tenor (years)", "Preferred name": "Loan_tenor_years"},
        {"Parameter": "Grace (years)", "Preferred name": "Loan_grace"},
        {"Parameter": "Working capital days", "Preferred name": "WC_days"},
        {"Parameter": "Split CAPEX into multiple lines", "Preferred name": "Capex_components"},
        {"Parameter": "VAT / input credit timing", "Preferred name": "VAT_timing"},
        {"Parameter": "IDC during construction", "Preferred name": "Interest_during_construction"},
        {"Parameter": "Tariff escalation", "Preferred name": "Tariff_escalation"},
        {"Parameter": "Price indexation", "Preferred name": "Price_indexation"},
        {"Parameter": "FX", "Preferred name": "FX_rate"},
    ]
)





TABLE_LABELS: Dict[str, str] = {
    "global_inputs": "Global inputs",
    "initial_investment": "Initial investment",
    "revenue_inputs": "Revenue inputs",
    "production_annual": "Production annual",
    "production_monthly": "Production monthly",
    "direct_costs_monthly": "Direct costs monthly",
    "staff_monthly": "Staff monthly",
    "other_opex_monthly": "Other opex monthly",
    "accounts_receivable": "Accounts receivable",
    "inventory_payable": "Inventory & payables",
    "loan_schedule": "Loan schedule",
    "tax_schedule": "Tax schedule",
    "inflation_schedule": "Inflation schedule",
    "risk_schedule": "Risk schedule",
    "sensitivity_config": "Sensitivity configuration",
    "monte_carlo_config": "Monte Carlo configuration",
    "goal_seek": "Goal seek",
    "scenario_config": "Scenario configuration",
    "break_even_inputs": "Break-even inputs",
    "parameter_naming": "Parameter naming",
}


TABLE_DEFAULTS: Dict[str, pd.DataFrame] = {
    key: df for key, df in [
        ("global_inputs", global_defaults),
        ("initial_investment", initial_investment_defaults),
        ("revenue_inputs", revenue_defaults),
        ("production_annual", production_annual_defaults),
        ("production_monthly", production_monthly_defaults),
        ("direct_costs_monthly", direct_costs_monthly_defaults),
        ("staff_monthly", staff_monthly_defaults),
        ("other_opex_monthly", other_opex_monthly_defaults),
        ("accounts_receivable", accounts_receivable_defaults),
        ("inventory_payable", inventory_payable_defaults),
        ("loan_schedule", loan_schedule_defaults),
        ("tax_schedule", tax_schedule_defaults),
        ("inflation_schedule", inflation_schedule_defaults),
        ("risk_schedule", risk_schedule_defaults),
        ("sensitivity_config", sensitivity_config_defaults),
        ("monte_carlo_config", monte_carlo_defaults),
        ("goal_seek", goal_seek_defaults),
        ("scenario_config", scenario_config_defaults),
        ("break_even_inputs", break_even_defaults),
        ("parameter_naming", parameter_naming_defaults),
    ]
}


SCALAR_DEFAULTS: Dict[str, Any] = {
    "projection_start_year": projection_defaults.start_year,
    "projection_end_year": projection_defaults.end_year,
    "projection_ppy": projection_defaults.periods_per_year,
    "msw_tonnes_pa": float(inputs.tech.msw_tonnes_pa),
    "lhv_mj_per_kg": float(inputs.tech.lhv_mj_per_kg),
    "boiler_efficiency": float(inputs.tech.boiler_efficiency),
    "electrical_efficiency": float(inputs.tech.electrical_efficiency),
    "availability": float(inputs.tech.availability),
    "parasitic_load": float(inputs.tech.parasitic_load_frac),
    "ppa_price": float(inputs.revenue.ppa_price_usd_per_mwh),
    "gate_fee": float(inputs.revenue.gate_fee_usd_per_t),
    "heat_price": float(inputs.revenue.heat_price_usd_per_mwh),
    "metal_recovery": float(inputs.revenue.metal_recovery_usd_per_t),
    "ash_revenue": float(inputs.revenue.ash_revenue_usd_per_t),
    "ppa_escalation": float(inputs.revenue.ppa_escalation),
    "gate_fee_escalation": float(inputs.revenue.gate_fee_escalation),
    "other_escalation": float(inputs.revenue.other_escalation),
    "capex_total": float(inputs.costs.capex_total_usd),
    "capex_profile_text": capex_profile_default_text,
    "fixed_om": float(inputs.costs.fixed_om_usd_pa),
    "variable_om": float(inputs.costs.variable_om_usd_per_t),
    "landfill_disposal": float(inputs.costs.landfill_disposal_usd_per_t),
    "insurance_pct": float(inputs.costs.insurance_pct_of_capex_pa),
    "maintenance_pct": float(inputs.costs.maintenance_pct_of_capex_pa),
    "opex_escalation": float(inputs.costs.opex_escalation),
    "debt_ratio": float(inputs.finance.debt_ratio),
    "interest_rate": float(inputs.finance.interest_rate),
    "tenor_years": int(inputs.finance.tenor_years),
    "grace_years": int(inputs.finance.grace_years),
    "upfront_fee_pct": float(inputs.finance.upfront_fee_pct),
    "tax_rate": float(inputs.finance.tax_rate),
    "depr_years": int(inputs.finance.depr_years),
    "working_cap_days": int(inputs.finance.working_cap_days),
    "discount_rate": float(inputs.finance.discount_rate),
}


SCALAR_CLEAN_START: Dict[str, Any] = {
    "projection_start_year": projection_defaults.start_year,
    "projection_end_year": projection_defaults.start_year + 4,
    "projection_ppy": max(1, projection_defaults.periods_per_year),
    "msw_tonnes_pa": 10_000.0,
    "lhv_mj_per_kg": 4.0,
    "boiler_efficiency": 0.3,
    "electrical_efficiency": 0.1,
    "availability": 0.5,
    "parasitic_load": 0.0,
    "ppa_price": 0.0,
    "gate_fee": 0.0,
    "heat_price": 0.0,
    "metal_recovery": 0.0,
    "ash_revenue": 0.0,
    "ppa_escalation": 0.0,
    "gate_fee_escalation": 0.0,
    "other_escalation": 0.0,
    "capex_total": 50_000_000.0,
    "capex_profile_text": "",
    "fixed_om": 0.0,
    "variable_om": 0.0,
    "landfill_disposal": 0.0,
    "insurance_pct": 0.0,
    "maintenance_pct": 0.0,
    "opex_escalation": 0.0,
    "debt_ratio": 0.0,
    "interest_rate": 0.0,
    "tenor_years": 1,
    "grace_years": 0,
    "upfront_fee_pct": 0.0,
    "tax_rate": 0.0,
    "depr_years": 1,
    "working_cap_days": 0,
    "discount_rate": 0.0,
}

page_tabs = st.tabs(
    [
        "Input Landing",
        "Revenue & Production",
        "Operations & Working Capital",
        "Financing & Taxes",
        "Key Metrics Dashboard",
        "Financial Performance",
        "Financial Position",
        "Cash Flow Statement",
        "Sensitivity Analyses",
        "Scenario / Ifs",
        "Break-Even & Payback",
    ]
)


with page_tabs[0]:
    with st.expander("Editing guide", expanded=False):
        st.markdown(
            """
            1. **Enable edit mode** – toggle the *Edit* checkbox for the section you want to
               update. Inputs remain read-only until editing is enabled.
            2. **Adjust rows as needed** – once in edit mode, use the *Add row* and *Remove row*
               buttons above each table to manage its contents before typing directly into the
               cells.
            3. **Manage default sets** – restore the shipped defaults, start with empty tables,
               or save/load your own presets from the *Manage defaults & state* panel.
            4. **Apply structured growth** – the *Yearly increment helper* lets you apply a
               compound annual change to any numeric column while the relevant section is in
               edit mode.
            5. **Review downstream impact** – every edit flows automatically into the dashboards,
               statements, and analytics tabs so you can validate changes immediately.
            """
        )

    with st.expander("Manage defaults & state", expanded=False):
        col_md1, col_md2, col_md3 = st.columns(3)
        if col_md1.button("Restore defaults", key="restore_defaults"):
            _reset_scalar_values(SCALAR_DEFAULTS)
            _reset_table_group(TABLE_DEFAULTS, mode="defaults")
            st.success("Defaults restored.")
            st.experimental_rerun()
        if col_md2.button("Clean start", key="clean_start"):
            _reset_scalar_values(SCALAR_CLEAN_START)
            _reset_table_group(TABLE_DEFAULTS, mode="clean")
            st.success("Workspace cleared.")
            st.experimental_rerun()
        if col_md3.button("Save current as custom", key="save_custom_defaults"):
            st.session_state["custom_scalar_defaults"] = {
                key: st.session_state.get(key, value) for key, value in SCALAR_DEFAULTS.items()
            }
            st.session_state["custom_table_defaults"] = {
                key: _ensure_state_df(key, df.copy()).copy() for key, df in TABLE_DEFAULTS.items()
            }
            st.success("Saved current assumptions as custom defaults.")
        if st.session_state.get("custom_scalar_defaults"):
            if st.button("Load custom defaults", key="load_custom_defaults"):
                _reset_scalar_values(st.session_state["custom_scalar_defaults"])
                for key, df in st.session_state.get("custom_table_defaults", {}).items():
                    _update_table_state(key, df)
                st.success("Loaded custom defaults.")
                st.experimental_rerun()

    with st.expander("Yearly increment helper", expanded=False):
        available_tables = {
            key: _ensure_state_df(key, df.copy()) for key, df in TABLE_DEFAULTS.items()
        }
        options = [
            key for key, df in available_tables.items()
            if not df.empty and any(is_numeric_dtype(df[col]) for col in df.columns)
        ]
        if not options:
            st.info("Add rows to a table to enable yearly increments.")
        else:
            table_key = st.selectbox(
                "Table",
                options,
                format_func=lambda k: TABLE_LABELS.get(k, k.replace("_", " ").title()),
                key="increment_table_select",
            )
            table_df = available_tables[table_key].copy()
            numeric_cols = [col for col in table_df.columns if is_numeric_dtype(table_df[col])]
            column = st.selectbox(
                "Column",
                numeric_cols,
                key=f"increment_column_{table_key}",
            )
            edit_flag = st.session_state.get(f"{table_key}_edit_toggle", False)
            if not edit_flag:
                st.info("Enable edit mode for this table to apply increments.")
            else:
                default_base = float(table_df[column].fillna(0.0).iloc[0]) if not table_df.empty else 0.0
                base_value = st.number_input(
                    "Base value",
                    value=float(default_base),
                    key=f"increment_base_{table_key}",
                )
                increment_pct = st.number_input(
                    "Annual increment (%)",
                    value=0.0,
                    step=0.5,
                    key=f"increment_pct_{table_key}",
                )
                if st.button("Apply increment", key=f"apply_increment_{table_key}"):
                    if table_df.empty:
                        st.warning("Table has no rows to update.")
                    else:
                        growth = [(1 + increment_pct / 100.0) ** i for i in range(len(table_df))]
                        updated = table_df.copy()
                        updated[column] = [base_value * g for g in growth]
                        _update_table_state(table_key, updated)
                        st.success(
                            f"Applied {increment_pct:.2f}% annual increment to "
                            f"{TABLE_LABELS.get(table_key, column)}"
                        )
                        st.experimental_rerun()

    proj_edit = _section_header("Projection Horizon", "projection_horizon")
    col_proj1, col_proj2, col_proj3 = st.columns(3)
    start_year = _number_input_control(
        "Start year",
        "projection_start_year",
        container=col_proj1,
        min_value=2000,
        max_value=2100,
        step=1,
        edit_enabled=proj_edit,
    )
    _number_input_control(
        "End year",
        "projection_end_year",
        container=col_proj2,
        min_value=int(start_year) + 1,
        max_value=2150,
        step=1,
        edit_enabled=proj_edit,
    )
    _number_input_control(
        "Periods per year",
        "projection_ppy",
        container=col_proj3,
        min_value=1,
        max_value=12,
        step=1,
        edit_enabled=proj_edit,
    )
    st.info(
        "The projection horizon drives the calendar footprint of every table and report in the workspace, "
        "including the monthly and annual statements."
    )

    global_edit = _section_header("Global Inputs", "global_inputs")
    global_inputs = _editable_table(
        "global_inputs",
        global_defaults,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter"),
            "Value": st.column_config.NumberColumn("Value", help="Values are captured in native units or percent."),
        },
        edit_enabled=global_edit,
    )

    initial_edit = _section_header("Initial Investment Inputs", "initial_investment")
    initial_investment = _editable_table(
        "initial_investment",
        initial_investment_defaults,
        column_config={
            "Item": st.column_config.TextColumn("Item"),
            "Cost": st.column_config.NumberColumn("Cost", format="%0.0f"),
            "Life (years)": st.column_config.NumberColumn("Life (years)", min_value=1, step=1),
        },
        edit_enabled=initial_edit,
    )

    schedule = _compute_initial_investment_schedule(initial_investment)
    total_investment = schedule["Cost"].sum() if not schedule.empty else 0.0
    st.metric("Total investment", f"${total_investment:,.0f}")
    st.markdown("**Depreciation schedule snapshot**")
    if schedule.empty:
        st.info("Add rows to the investment table to generate depreciation schedules.")
    else:
        st.dataframe(schedule.round(2), use_container_width=True)


with page_tabs[1]:
    revenue_edit = _section_header("Revenue Inputs", "revenue_inputs")
    revenue_table = _editable_table(
        "revenue_inputs",
        revenue_defaults,
        column_config={
            "Revenue stream": st.column_config.TextColumn("Revenue stream"),
            "Price": st.column_config.NumberColumn("Price", format="%0.2f"),
            "Escalation (%)": st.column_config.NumberColumn("Escalation (%)", format="%0.2f"),
        },
        edit_enabled=revenue_edit,
    )

    st.subheader("Production Assumptions")
    prod_tabs = st.tabs(["Annual", "Monthly"])
    with prod_tabs[0]:
        prod_annual_edit = _section_header("Annual production", "production_annual")
        production_annual = _editable_table(
            "production_annual",
            production_annual_defaults,
            column_config={
                "Year": st.column_config.NumberColumn("Year", step=1),
                "Throughput (t)": st.column_config.NumberColumn("Throughput (t)", format="%0.0f"),
            },
            edit_enabled=prod_annual_edit,
        )
    with prod_tabs[1]:
        prod_monthly_edit = _section_header("Monthly production", "production_monthly")
        production_monthly = _editable_table(
            "production_monthly",
            production_monthly_defaults,
            column_config={
                "Month": st.column_config.NumberColumn("Month", step=1, min_value=1, max_value=12),
                "Throughput (t)": st.column_config.NumberColumn("Throughput (t)", format="%0.0f"),
            },
            edit_enabled=prod_monthly_edit,
        )

    tech_edit = _section_header("Technology Inputs", "technology_inputs")
    tech_cols = st.columns(3)
    _number_input_control(
        "MSW throughput (t/a)",
        "msw_tonnes_pa",
        container=tech_cols[0],
        min_value=10_000.0,
        max_value=1_000_000.0,
        step=10_000.0,
        edit_enabled=tech_edit,
    )
    _number_input_control(
        "Lower heating value (MJ/kg)",
        "lhv_mj_per_kg",
        container=tech_cols[1],
        min_value=4.0,
        max_value=18.0,
        step=0.1,
        edit_enabled=tech_edit,
    )
    _number_input_control(
        "Availability",
        "availability",
        container=tech_cols[2],
        min_value=0.5,
        max_value=1.0,
        step=0.01,
        edit_enabled=tech_edit,
    )

    tech_cols2 = st.columns(3)
    _number_input_control(
        "Boiler efficiency",
        "boiler_efficiency",
        container=tech_cols2[0],
        min_value=0.3,
        max_value=1.0,
        step=0.01,
        edit_enabled=tech_edit,
    )
    _number_input_control(
        "Electrical efficiency",
        "electrical_efficiency",
        container=tech_cols2[1],
        min_value=0.1,
        max_value=0.5,
        step=0.01,
        edit_enabled=tech_edit,
    )
    _number_input_control(
        "Parasitic load fraction",
        "parasitic_load",
        container=tech_cols2[2],
        min_value=0.0,
        max_value=0.3,
        step=0.01,
        edit_enabled=tech_edit,
    )

    commercial_edit = _section_header("Commercial Terms", "commercial_terms")
    comm_cols = st.columns(3)
    _number_input_control(
        "PPA price (USD/MWh)",
        "ppa_price",
        container=comm_cols[0],
        min_value=0.0,
        max_value=500.0,
        step=1.0,
        edit_enabled=commercial_edit,
    )
    _number_input_control(
        "Gate fee (USD/t)",
        "gate_fee",
        container=comm_cols[1],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        edit_enabled=commercial_edit,
    )
    _number_input_control(
        "Heat price (USD/MWh)",
        "heat_price",
        container=comm_cols[2],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        edit_enabled=commercial_edit,
    )

    comm_cols2 = st.columns(3)
    _number_input_control(
        "Metal recovery (USD/t)",
        "metal_recovery",
        container=comm_cols2[0],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        edit_enabled=commercial_edit,
    )
    _number_input_control(
        "Ash revenue (USD/t)",
        "ash_revenue",
        container=comm_cols2[1],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        edit_enabled=commercial_edit,
    )
    _number_input_control(
        "PPA escalation (pa)",
        "ppa_escalation",
        container=comm_cols2[2],
        min_value=0.0,
        max_value=0.15,
        step=0.005,
        edit_enabled=commercial_edit,
    )

    comm_cols3 = st.columns(2)
    _number_input_control(
        "Gate fee escalation (pa)",
        "gate_fee_escalation",
        container=comm_cols3[0],
        min_value=0.0,
        max_value=0.15,
        step=0.005,
        edit_enabled=commercial_edit,
    )
    _number_input_control(
        "By-product escalation (pa)",
        "other_escalation",
        container=comm_cols3[1],
        min_value=0.0,
        max_value=0.15,
        step=0.005,
        edit_enabled=commercial_edit,
    )


with page_tabs[2]:
    direct_cost_edit = _section_header("Direct Costs (Monthly)", "direct_costs_monthly")
    direct_costs_monthly = _editable_table(
        "direct_costs_monthly",
        direct_costs_monthly_defaults,
        column_config={
            "Month": st.column_config.NumberColumn("Month", min_value=1, max_value=12, step=1),
            "Feedstock cost": st.column_config.NumberColumn("Feedstock cost", format="%0.0f"),
            "Residue disposal": st.column_config.NumberColumn("Residue disposal", format="%0.0f"),
        },
        edit_enabled=direct_cost_edit,
    )

    staff_edit = _section_header("Staff Costs (Monthly)", "staff_monthly")
    staff_monthly = _editable_table(
        "staff_monthly",
        staff_monthly_defaults,
        column_config={
            "Role": st.column_config.TextColumn("Role"),
            "Monthly cost": st.column_config.NumberColumn("Monthly cost", format="%0.0f"),
        },
        edit_enabled=staff_edit,
    )

    other_opex_edit = _section_header("Other Opex (Monthly)", "other_opex_monthly")
    other_opex_monthly = _editable_table(
        "other_opex_monthly",
        other_opex_monthly_defaults,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Monthly cost": st.column_config.NumberColumn("Monthly cost", format="%0.0f"),
        },
        edit_enabled=other_opex_edit,
    )

    st.subheader("Working Capital Inputs")
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        receivable_edit = _section_header("Accounts receivable", "accounts_receivable", level="markdown")
        accounts_receivable = _editable_table(
            "accounts_receivable",
            accounts_receivable_defaults,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Value": st.column_config.NumberColumn("Value", format="%0.2f"),
            },
            edit_enabled=receivable_edit,
        )
    with col_wc2:
        payable_edit = _section_header("Inventory & payables", "inventory_payable", level="markdown")
        inventory_payable = _editable_table(
            "inventory_payable",
            inventory_payable_defaults,
            column_config={
                "Metric": st.column_config.TextColumn("Metric"),
                "Value": st.column_config.NumberColumn("Value", format="%0.2f"),
            },
            edit_enabled=payable_edit,
        )

    cost_edit = _section_header("Cost Structure Controls", "cost_structure")
    cost_cols = st.columns(3)
    _number_input_control(
        "Total CAPEX (USD)",
        "capex_total",
        container=cost_cols[0],
        min_value=50_000_000.0,
        max_value=600_000_000.0,
        step=5_000_000.0,
        edit_enabled=cost_edit,
    )
    _number_input_control(
        "Fixed O&M (USD/a)",
        "fixed_om",
        container=cost_cols[1],
        min_value=0.0,
        max_value=80_000_000.0,
        step=500_000.0,
        edit_enabled=cost_edit,
    )
    _number_input_control(
        "Variable O&M (USD/t)",
        "variable_om",
        container=cost_cols[2],
        min_value=0.0,
        max_value=250.0,
        step=1.0,
        edit_enabled=cost_edit,
    )

    cost_cols2 = st.columns(3)
    _number_input_control(
        "Residue disposal (USD/t)",
        "landfill_disposal",
        container=cost_cols2[0],
        min_value=0.0,
        max_value=200.0,
        step=1.0,
        edit_enabled=cost_edit,
    )
    _number_input_control(
        "Insurance (% of CAPEX/a)",
        "insurance_pct",
        container=cost_cols2[1],
        min_value=0.0,
        max_value=0.05,
        step=0.001,
        edit_enabled=cost_edit,
    )
    _number_input_control(
        "Maintenance (% of CAPEX/a)",
        "maintenance_pct",
        container=cost_cols2[2],
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        edit_enabled=cost_edit,
    )

    _number_input_control(
        "Opex escalation (pa)",
        "opex_escalation",
        min_value=0.0,
        max_value=0.15,
        step=0.005,
        edit_enabled=cost_edit,
    )

    _text_area_control(
        "Capex spend profile (comma separated)",
        "capex_profile_text",
        help="Enter fractions that sum to 1 over the construction periods. Leave blank for an even spread.",
        edit_enabled=cost_edit,
    )



with page_tabs[3]:
    finance_edit = _section_header("Financing Structure", "financing_structure")
    finance_cols = st.columns(3)
    _number_input_control(
        "Debt ratio",
        "debt_ratio",
        container=finance_cols[0],
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        edit_enabled=finance_edit,
    )
    _number_input_control(
        "Interest rate (pa)",
        "interest_rate",
        container=finance_cols[1],
        min_value=0.0,
        max_value=0.25,
        step=0.005,
        edit_enabled=finance_edit,
    )
    _number_input_control(
        "Upfront fee",
        "upfront_fee_pct",
        container=finance_cols[2],
        min_value=0.0,
        max_value=0.05,
        step=0.001,
        edit_enabled=finance_edit,
    )

    finance_cols2 = st.columns(3)
    _number_input_control(
        "Debt tenor (years)",
        "tenor_years",
        container=finance_cols2[0],
        min_value=1,
        max_value=30,
        step=1,
        edit_enabled=finance_edit,
    )
    _number_input_control(
        "Grace period (years)",
        "grace_years",
        container=finance_cols2[1],
        min_value=0,
        max_value=10,
        step=1,
        edit_enabled=finance_edit,
    )
    _number_input_control(
        "Discount rate (pa)",
        "discount_rate",
        container=finance_cols2[2],
        min_value=0.0,
        max_value=0.30,
        step=0.01,
        edit_enabled=finance_edit,
    )

    finance_cols3 = st.columns(2)
    _number_input_control(
        "Corporate tax rate",
        "tax_rate",
        container=finance_cols3[0],
        min_value=0.0,
        max_value=0.5,
        step=0.01,
        edit_enabled=finance_edit,
    )
    _number_input_control(
        "Working capital days",
        "working_cap_days",
        container=finance_cols3[1],
        min_value=0,
        max_value=180,
        step=5,
        edit_enabled=finance_edit,
    )

    _number_input_control(
        "Depreciation period (years)",
        "depr_years",
        min_value=1,
        max_value=30,
        step=1,
        edit_enabled=finance_edit,
    )

    loan_edit = _section_header("Loan Schedule", "loan_schedule")
    loan_schedule = _editable_table(
        "loan_schedule",
        loan_schedule_defaults,
        column_config={
            "Facility": st.column_config.TextColumn("Facility"),
            "Base amount": st.column_config.NumberColumn("Base amount", format="%0.0f"),
            "Interest rate (%)": st.column_config.NumberColumn("Interest rate (%)", format="%0.2f"),
            "Loan type": st.column_config.TextColumn("Loan type"),
            "Start year": st.column_config.NumberColumn("Start year", step=1),
            "Duration (years)": st.column_config.NumberColumn("Duration (years)", step=1),
            "Grace (years)": st.column_config.NumberColumn("Grace (years)", step=1),
        },
        edit_enabled=loan_edit,
    )

    tax_edit = _section_header("Tax Schedule", "tax_schedule")
    tax_schedule = _editable_table(
        "tax_schedule",
        tax_schedule_defaults,
        column_config={
            "Tax": st.column_config.TextColumn("Tax"),
            "Rate (%)": st.column_config.NumberColumn("Rate (%)", format="%0.2f"),
            "Timing adjustment (months)": st.column_config.NumberColumn("Timing adjustment (months)", step=1),
            "Notes": st.column_config.TextColumn("Notes"),
        },
        edit_enabled=tax_edit,
    )

    inflation_edit = _section_header("Inflation Schedule", "inflation_schedule")
    inflation_schedule = _editable_table(
        "inflation_schedule",
        inflation_schedule_defaults,
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Inflation rate (%)": st.column_config.NumberColumn("Inflation rate (%)", format="%0.2f"),
        },
        edit_enabled=inflation_edit,
    )

    risk_edit = _section_header("Risk Schedule", "risk_schedule")
    risk_schedule = _editable_table(
        "risk_schedule",
        risk_schedule_defaults,
        column_config={
            "Risk": st.column_config.TextColumn("Risk"),
            "Probability (%)": st.column_config.NumberColumn("Probability (%)", format="%0.1f"),
            "Impact (USD)": st.column_config.NumberColumn("Impact (USD)", format="%0.0f"),
            "Mitigation": st.column_config.TextColumn("Mitigation"),
        },
        edit_enabled=risk_edit,
    )


projection = ProjectionSettings(
    start_year=int(st.session_state["projection_start_year"]),
    end_year=int(st.session_state["projection_end_year"]),
    periods_per_year=int(st.session_state["projection_ppy"]),
)

capex_profile = _parse_capex_profile(st.session_state["capex_profile_text"], inputs.costs.capex_spend_profile)

revenue_inputs = RevenueAssumptions()
revenue_inputs.ppa_price_usd_per_mwh = float(st.session_state["ppa_price"])
revenue_inputs.ppa_escalation = float(st.session_state["ppa_escalation"])
revenue_inputs.gate_fee_usd_per_t = float(st.session_state["gate_fee"])
revenue_inputs.gate_fee_escalation = float(st.session_state["gate_fee_escalation"])
revenue_inputs.heat_price_usd_per_mwh = float(st.session_state["heat_price"])
revenue_inputs.metal_recovery_usd_per_t = float(st.session_state["metal_recovery"])
revenue_inputs.ash_revenue_usd_per_t = float(st.session_state["ash_revenue"])
revenue_inputs.other_escalation = float(st.session_state["other_escalation"])

user_inputs = WTEMasterInputs(
    timeline=Timeline(
        years=projection.years,
        build_months=inputs.timeline.build_months,
        start_year=projection.start_year,
        periods_per_year=projection.periods_per_year,
    ),
    tech=TechAssumptions(
        msw_tonnes_pa=float(st.session_state["msw_tonnes_pa"]),
        lhv_mj_per_kg=float(st.session_state["lhv_mj_per_kg"]),
        boiler_efficiency=float(st.session_state["boiler_efficiency"]),
        electrical_efficiency=float(st.session_state["electrical_efficiency"]),
        availability=float(st.session_state["availability"]),
        parasitic_load_frac=float(st.session_state["parasitic_load"]),
    ),
    revenue=revenue_inputs,
    costs=CostAssumptions(
        capex_total_usd=float(st.session_state["capex_total"]),
        capex_spend_profile=capex_profile,
        fixed_om_usd_pa=float(st.session_state["fixed_om"]),
        variable_om_usd_per_t=float(st.session_state["variable_om"]),
        landfill_disposal_usd_per_t=float(st.session_state["landfill_disposal"]),
        insurance_pct_of_capex_pa=float(st.session_state["insurance_pct"]),
        maintenance_pct_of_capex_pa=float(st.session_state["maintenance_pct"]),
        opex_escalation=float(st.session_state["opex_escalation"]),
    ),
    finance=FinanceAssumptions(
        debt_ratio=float(st.session_state["debt_ratio"]),
        interest_rate=float(st.session_state["interest_rate"]),
        tenor_years=int(st.session_state["tenor_years"]),
        grace_years=int(st.session_state["grace_years"]),
        upfront_fee_pct=float(st.session_state["upfront_fee_pct"]),
        dscr_min=inputs.finance.dscr_min,
        tax_rate=float(st.session_state["tax_rate"]),
        depr_years=int(st.session_state["depr_years"]),
        working_cap_days=int(st.session_state["working_cap_days"]),
        discount_rate=float(st.session_state["discount_rate"]),
    ),
)

results = cashflow_model(user_inputs)

summary, summary_ann, summary_cumulative, production_annual_series = _build_summary_tables(
    user_inputs, results
)

energy = results["energy"]
revenue = results["rev"]
capex_total = results["capex"]["total"]
opex_total = results["opex"]["total_opex"]
tax_cash = results["tax"]["cash_tax"]
depr_total = results["depr"]["total"]
debt = results["debt"]

base_signature = repr(user_inputs)
prev_signature = st.session_state.get("base_signature")
if prev_signature != base_signature:
    st.session_state["base_signature"] = base_signature
    st.session_state["input_snapshot"] = copy.deepcopy(user_inputs)
    st.session_state["results_snapshot"] = copy.deepcopy(results)
    st.session_state["scenario_payloads"] = {}
    st.session_state.pop("excel_bytes_map", None)
else:
    st.session_state["input_snapshot"] = copy.deepcopy(user_inputs)
    st.session_state["results_snapshot"] = copy.deepcopy(results)



def _get_global_value(parameter: str, default: float) -> float:
    df = st.session_state.get("global_inputs")
    if df is None or df.empty:
        return default
    mask = df["Parameter"].str.lower() == parameter.lower()
    if not mask.any():
        return default
    try:
        return float(df.loc[mask, "Value"].iloc[0])
    except (TypeError, ValueError, IndexError):
        return default


corp_tax_pct = _get_global_value("Corporate tax rate (%)", user_inputs.finance.tax_rate * 100)
investor_share_pct = _get_global_value("Investor share capital (%)", 50.0)
owner_share_pct = _get_global_value("Owner share capital (%)", 50.0)
terminal_growth_pct = _get_global_value("Terminal growth (%)", 2.0)
capital_gain_tax_pct = _get_global_value("Capital gains tax rate (%)", 5.0)
payback_threshold_years = _get_global_value("Payback threshold (years)", 12.0)

equity_cf = results["equity_cf"]
investor_cf = equity_cf * investor_share_pct / 100.0
owner_cf = equity_cf * owner_share_pct / 100.0

investor_irr = _irr(investor_cf)
owner_irr = _irr(owner_cf)

project_cashflows = -capex_total + (revenue["total_revenue"] - opex_total - tax_cash)
project_npv = _npv(user_inputs.finance.discount_rate, project_cashflows)

annual_revenue = _annualise(revenue["total_revenue"], user_inputs.timeline.periods_per_year)
annual_ebitda = _annualise(results["ebitda"], user_inputs.timeline.periods_per_year)
annual_equity_cf = _annualise(results["equity_cf"], user_inputs.timeline.periods_per_year)
annual_capex = _annualise(capex_total, user_inputs.timeline.periods_per_year)



with page_tabs[4]:
    st.subheader("Assumptions Snapshot")
    snapshot_cols = st.columns(4)
    snapshot_cols[0].metric("Corporate tax", f"{corp_tax_pct:.2f}%")
    snapshot_cols[1].metric("Investor share", f"{investor_share_pct:.1f}%")
    snapshot_cols[2].metric("Owner share", f"{owner_share_pct:.1f}%")
    snapshot_cols[3].metric("Terminal growth", f"{terminal_growth_pct:.1f}%")

    st.subheader("Global Overview")
    global_overview = pd.DataFrame(
        {
            "Metric": [
                "Corporate tax rate",
                "Investor share capital",
                "Owner share capital",
                "Terminal growth",
                "Capital gains tax",
                "Payback threshold",
            ],
            "Value": [
                corp_tax_pct / 100.0,
                investor_share_pct / 100.0,
                owner_share_pct / 100.0,
                terminal_growth_pct / 100.0,
                capital_gain_tax_pct / 100.0,
                payback_threshold_years,
            ],
        }
    )
    st.dataframe(global_overview, use_container_width=True)

    st.subheader("Latest Drivers")
    latest = summary.iloc[-1]
    latest_cols = st.columns(4)
    latest_cols[0].metric("Final month revenue", f"${latest['Total revenue']:,.0f}")
    latest_cols[1].metric("Final month EBITDA", f"${latest['EBITDA']:,.0f}")
    latest_cols[2].metric("Final month equity CF", f"${latest['Equity cash flow']:,.0f}")
    latest_cols[3].metric("Cumulative FCF", f"${summary_cumulative['Cumulative CFADS'].iloc[-1]:,.0f}")
    st.metric("Cumulative equity cash flow", f"${summary_cumulative['Cumulative Equity cash flow'].iloc[-1]:,.0f}")

    st.subheader("Headline Metrics")
    headline_cols = st.columns(3)
    headline_cols[0].metric("Project NPV", f"${project_npv:,.0f}")
    headline_cols[1].metric("Project IRR", f"{results['irr_proj'] * 100:.2f}%")
    headline_cols[2].metric("Equity IRR", f"{results['irr_eq'] * 100:.2f}%")
    irrs_cols = st.columns(3)
    irrs_cols[0].metric("Investor IRR", f"{investor_irr * 100:.2f}%")
    irrs_cols[1].metric("Owner IRR", f"{owner_irr * 100:.2f}%")
    payback_periods = summary_cumulative[summary_cumulative["Cumulative Equity cash flow"] >= 0]["Period"]
    payback_value = payback_periods.iloc[0] / user_inputs.timeline.periods_per_year if not payback_periods.empty else float("nan")
    irrs_cols[2].metric("Payback (years)", "n/a" if np.isnan(payback_value) else f"{payback_value:.2f}")

    st.subheader("Production of Nickel (Annual)")
    production_chart_df = pd.DataFrame(
        {
            "Year": production_annual_series.index + projection.start_year,
            "Nickel production (t)": production_annual_series.values,
        }
    )
    if not production_chart_df.empty:
        st.line_chart(production_chart_df.set_index("Year"))
    else:
        st.info("Add production assumptions to display the chart.")

    st.subheader("Cash Flow Overview")
    st.area_chart(summary.set_index("Period")["Equity cash flow"], height=260)

    st.subheader("Annual Operations & Production Summary")
    st.dataframe(summary_ann, use_container_width=True)

    st.subheader("Revenue Mix")
    revenue_mix = pd.DataFrame(
        {
            "Energy": _annualise(results["rev"]["energy_revenue"], user_inputs.timeline.periods_per_year),
            "Gate": _annualise(results["rev"]["gate_revenue"], user_inputs.timeline.periods_per_year),
            "Other": _annualise(results["rev"]["other_revenue"], user_inputs.timeline.periods_per_year),
        }
    )
    st.bar_chart(revenue_mix)

    st.subheader("Operating Costs")
    opex_mix = pd.DataFrame(
        {
            "Fixed": _annualise(results["opex"]["fixed_om"], user_inputs.timeline.periods_per_year),
            "Variable": _annualise(results["opex"]["variable_om"], user_inputs.timeline.periods_per_year),
            "Disposal": _annualise(results["opex"]["disposal"], user_inputs.timeline.periods_per_year),
            "Insurance": _annualise(results["opex"]["insurance"], user_inputs.timeline.periods_per_year),
            "Maintenance": _annualise(results["opex"]["maintenance"], user_inputs.timeline.periods_per_year),
        }
    )
    st.bar_chart(opex_mix)

    st.subheader("Cost Breakdown")
    cost_breakdown = pd.DataFrame(
        {
            "Category": ["CAPEX", "OPEX", "Debt service"],
            "Value": [
                float(capex_total.sum()),
                float(results["opex"]["total_opex"].sum()),
                float(debt["debt_service"].sum()),
            ],
        }
    )
    st.bar_chart(cost_breakdown.set_index("Category"))

    st.subheader("Capital Expenditure and Debt")
    capex_debt = pd.DataFrame(
        {
            "Capex": _annualise(capex_total, user_inputs.timeline.periods_per_year),
            "Debt draws": _annualise(debt["debt_draws"], user_inputs.timeline.periods_per_year),
        }
    )
    st.bar_chart(capex_debt)

    st.subheader("Fixed Asset Summary")
    st.dataframe(schedule.round(2), use_container_width=True)

    st.subheader("Debt Schedule")
    st.line_chart(pd.Series(debt["balance"], index=summary["Period"]))

    st.subheader("Cash Flow & Returns")
    cf_returns = pd.DataFrame(
        {
            "EBITDA": summary["EBITDA"],
            "CFADS": summary["CFADS"],
            "Equity CF": summary["Equity cash flow"],
        }
    )
    st.line_chart(cf_returns)

    st.subheader("Cumulative Cash Flows")
    cumulative_chart = summary_cumulative[["Cumulative CFADS", "Cumulative Equity cash flow"]]
    cumulative_chart.index = summary["Period"]
    st.line_chart(cumulative_chart)



with page_tabs[5]:
    st.subheader("Monthly Financial Performance")
    monthly_perf = summary[["Period", "Total revenue", "Total opex", "EBITDA", "Tax", "Equity cash flow"]].copy()
    monthly_perf.rename(
        columns={
            "Total revenue": "Revenue",
            "Total opex": "Operating costs",
            "Equity cash flow": "Equity CF",
        },
        inplace=True,
    )
    st.dataframe(monthly_perf.round(2), use_container_width=True)

    st.subheader("Annual Financial Performance")
    annual_perf = summary_ann[["Calendar Year", "Total revenue", "Total opex", "EBITDA", "Tax", "Equity cash flow"]].rename(
        columns={
            "Total revenue": "Revenue",
            "Total opex": "Operating costs",
            "Equity cash flow": "Equity CF",
        }
    )
    st.dataframe(annual_perf.round(2), use_container_width=True)

    st.subheader("Total Expense Schedule")
    expense_schedule = pd.DataFrame(
        {
            "Period": summary["Period"],
            "Fixed O&M": results["opex"]["fixed_om"],
            "Variable O&M": results["opex"]["variable_om"],
            "Disposal": results["opex"]["disposal"],
            "Insurance": results["opex"]["insurance"],
            "Maintenance": results["opex"]["maintenance"],
        }
    )
    st.dataframe(expense_schedule.round(2), use_container_width=True)


with page_tabs[6]:
    st.subheader("Monthly Statement of Financial Position")
    net_fixed_assets = np.cumsum(capex_total) - np.cumsum(depr_total)
    debt_balance = debt["balance"]
    equity_balance = np.cumsum(-capex_total + debt["debt_draws"] + results["equity_cf"])
    cash_balance = np.cumsum(results["equity_cf"])
    working_capital = np.full_like(net_fixed_assets, user_inputs.finance.working_cap_days)
    balance_sheet_monthly = pd.DataFrame(
        {
            "Period": summary["Period"],
            "Net fixed assets": net_fixed_assets,
            "Working capital": working_capital,
            "Cash": cash_balance,
            "Debt": debt_balance,
            "Equity": equity_balance,
        }
    )
    st.dataframe(balance_sheet_monthly.round(2), use_container_width=True)

    st.subheader("Annual Statement of Financial Position")
    balance_sheet_annual = balance_sheet_monthly.copy()
    balance_sheet_annual["Calendar Year"] = summary["Calendar Year"]
    balance_sheet_annual = balance_sheet_annual.groupby("Calendar Year", as_index=False).agg(
        {
            "Net fixed assets": "mean",
            "Working capital": "mean",
            "Cash": "mean",
            "Debt": "mean",
            "Equity": "mean",
        }
    )
    st.dataframe(balance_sheet_annual.round(2), use_container_width=True)


with page_tabs[7]:
    st.subheader("Monthly Cash Flow Statement")
    cash_flow_monthly = pd.DataFrame(
        {
            "Period": summary["Period"],
            "Operating cash flow": summary["CFADS"],
            "Investing cash flow": -capex_total,
            "Financing cash flow": debt["debt_draws"] - debt["debt_service"] + equity_cf,
            "Net cash flow": summary["CFADS"] - capex_total + debt["debt_draws"] - debt["debt_service"] + equity_cf,
            "Cumulative equity CF": summary_cumulative["Cumulative Equity cash flow"],
        }
    )
    st.dataframe(cash_flow_monthly.round(2), use_container_width=True)

    st.subheader("Annual Cash Flow Statement")
    cash_flow_annual = cash_flow_monthly.copy()
    cash_flow_annual["Calendar Year"] = summary["Calendar Year"]
    cash_flow_annual = cash_flow_annual.groupby("Calendar Year", as_index=False).agg(
        {
            "Operating cash flow": "sum",
            "Investing cash flow": "sum",
            "Financing cash flow": "sum",
            "Net cash flow": "sum",
            "Cumulative equity CF": "last",
        }
    )
    st.dataframe(cash_flow_annual.round(2), use_container_width=True)

    st.subheader("Cumulative Equity Cash Flow")
    st.line_chart(cash_flow_monthly.set_index("Period")["Cumulative equity CF"], height=260)
    st.dataframe(
        cash_flow_monthly[["Period", "Cumulative equity CF"]].round(2),
        use_container_width=True,
    )



with page_tabs[8]:
    sensitivity_edit = _section_header("Sensitivity Analysis Configuration", "sensitivity_config")
    sensitivity_config = _editable_table(
        "sensitivity_config",
        sensitivity_config_defaults,
        column_config={
            "Driver": st.column_config.TextColumn("Driver"),
            "Low": st.column_config.NumberColumn("Low", format="%0.2f"),
            "Base": st.column_config.NumberColumn("Base", format="%0.2f"),
            "High": st.column_config.NumberColumn("High", format="%0.2f"),
        },
        edit_enabled=sensitivity_edit,
    )

    st.subheader("Simulation Results")
    simulated = []
    for _, row in sensitivity_config.iterrows():
        driver = row.get("Driver", "")
        try:
            base_adj = float(row.get("Base", 0.0))
            low_adj = float(row.get("Low", 0.0))
            high_adj = float(row.get("High", 0.0))
        except (TypeError, ValueError):
            continue
        simulated.append(
            {
                "Driver": driver,
                "Low IRR": results["irr_eq"] + low_adj,
                "Base IRR": results["irr_eq"] + base_adj,
                "High IRR": results["irr_eq"] + high_adj,
            }
        )
    sensitivity_results = pd.DataFrame(simulated)
    st.dataframe(sensitivity_results.round(4), use_container_width=True)

    monte_edit = _section_header("Monte Carlo Simulation Configuration", "monte_carlo_config")
    monte_carlo_defaults_runtime = pd.DataFrame(
        [
            {
                "Variable": "PPA price",
                "Distribution": "Normal",
                "Mean": st.session_state["ppa_price"],
                "Std dev": st.session_state["ppa_price"] * 0.05,
            },
            {
                "Variable": "CAPEX",
                "Distribution": "Triangular",
                "Mean": st.session_state["capex_total"],
                "Std dev": st.session_state["capex_total"] * 0.08,
            },
        ]
    )
    monte_carlo_config = _editable_table(
        "monte_carlo_config",
        monte_carlo_defaults_runtime,
        column_config={
            "Variable": st.column_config.TextColumn("Variable"),
            "Distribution": st.column_config.TextColumn("Distribution"),
            "Mean": st.column_config.NumberColumn("Mean", format="%0.2f"),
            "Std dev": st.column_config.NumberColumn("Std dev", format="%0.2f"),
        },
        edit_enabled=monte_edit,
    )
    if not monte_carlo_config.empty:
        st.write(
            "Simulated IRR range (conceptual):",
            f"{(results['irr_eq'] - 0.02) * 100:.2f}% to {(results['irr_eq'] + 0.02) * 100:.2f}%",
        )


with page_tabs[9]:
    goal_edit = _section_header("Goal Seek Configuration", "goal_seek")
    goal_seek = _editable_table(
        "goal_seek",
        goal_seek_defaults,
        column_config={
            "Target metric": st.column_config.TextColumn("Target metric"),
            "Target value": st.column_config.NumberColumn("Target value", format="%0.2f"),
            "Variable": st.column_config.TextColumn("Variable"),
        },
        edit_enabled=goal_edit,
    )

    st.subheader("Goal Seek Results")
    if goal_seek.empty:
        st.info("Add goal seek configurations to calculate required adjustments.")
    else:
        results_rows = []
        for _, row in goal_seek.iterrows():
            metric = row.get("Target metric", "")
            try:
                target_value = float(row.get("Target value", 0.0))
            except (TypeError, ValueError):
                target_value = 0.0
            base_value = {
                "Equity IRR": results["irr_eq"],
                "Project IRR": results["irr_proj"],
                "DSCR": float(np.nanmin(results["dscr"])) if results["dscr"].size else float("nan"),
            }.get(metric, float("nan"))
            delta = target_value - base_value if not np.isnan(base_value) else float("nan")
            results_rows.append(
                {
                    "Metric": metric,
                    "Target": target_value,
                    "Base": base_value,
                    "Delta": delta,
                    "Suggested variable": row.get("Variable", ""),
                }
            )
        st.dataframe(pd.DataFrame(results_rows).round(4), use_container_width=True)

    scenario_edit = _section_header("Scenario / Is Configuration", "scenario_config")
    scenario_config = _editable_table(
        "scenario_config",
        scenario_config_defaults,
        column_config={
            "Scenario": st.column_config.TextColumn("Scenario"),
            "PPA adjustment": st.column_config.NumberColumn("PPA adjustment", format="%0.2f"),
            "Gate fee adjustment": st.column_config.NumberColumn("Gate fee adjustment", format="%0.2f"),
            "CAPEX adjustment": st.column_config.NumberColumn("CAPEX adjustment", format="%0.2f"),
        },
        edit_enabled=scenario_edit,
    )

    scenario_signature = scenario_config.to_csv(index=False) if not scenario_config.empty else ""
    if st.session_state.get("scenario_signature") != scenario_signature:
        st.session_state["scenario_signature"] = scenario_signature
        st.session_state["scenario_payloads"] = {}
        st.session_state.pop("excel_bytes_map", None)

    base_snapshot = st.session_state.get("input_snapshot", copy.deepcopy(user_inputs))
    base_results_snapshot = st.session_state.get("results_snapshot", copy.deepcopy(results))
    _ensure_scenario_payload("Base Case", base_snapshot, scenario_config, base_results_snapshot)

    scenario_names: List[str] = []
    if not scenario_config.empty and "Scenario" in scenario_config.columns:
        scenario_names = [
            name
            for name in scenario_config["Scenario"].astype(str).str.strip()
            if name and name.lower() != "nan"
        ]

    st.subheader("Scenario Tool Configuration")
    scenario_results: List[Dict[str, float]] = []
    if scenario_names:
        for scenario_name in scenario_names:
            scenario_model, scenario_payload = _ensure_scenario_payload(
                scenario_name,
                base_snapshot,
                scenario_config,
                base_results_snapshot,
            )
            scenario_results.append(
                {
                    "Scenario": scenario_name,
                    "Equity IRR": scenario_payload["irr_eq"],
                    "Project IRR": scenario_payload["irr_proj"],
                    "NPV": _npv(
                        scenario_model.finance.discount_rate,
                        scenario_payload["equity_cf"],
                    ),
                }
            )
        st.dataframe(pd.DataFrame(scenario_results).round(4), use_container_width=True)
    else:
        st.info("Add scenarios to compare outcomes.")

    st.subheader("Scenario Comparison")
    if scenario_results:
        comparison_chart = pd.DataFrame(scenario_results).set_index("Scenario")
        st.bar_chart(comparison_chart)
    else:
        st.info("Populate scenarios to view comparisons.")

    st.subheader("Excel Model Download")
    download_options = ["Base Case"]
    for name in scenario_names:
        if name not in download_options:
            download_options.append(name)
    selected_scenario = st.selectbox(
        "Scenario to export",
        download_options,
        key="excel_download_scenario",
    )
    download_container = st.container()

    snapshot = st.session_state.get("input_snapshot", copy.deepcopy(user_inputs))
    model, scenario_payload_results = _ensure_scenario_payload(
        selected_scenario,
        snapshot,
        scenario_config,
        st.session_state.get("results_snapshot", copy.deepcopy(results)),
    )
    st.session_state["model_results"] = (model, scenario_payload_results)

    excel_map: Dict[str, bytes] = st.session_state.setdefault("excel_bytes_map", {})
    excel_bytes = excel_map.get(selected_scenario)

    model.scenario = selected_scenario

    with download_container:
        if not excel_bytes:
            if st.button("Prepare Excel Model", key=f"prepare_excel_{selected_scenario.lower()}"):
                with st.spinner("Preparing Excel workbook..."):
                    excel_bytes = _generate_excel_bytes(model, scenario_payload_results, selected_scenario)
                excel_map[selected_scenario] = excel_bytes
                st.session_state.excel_bytes_map = excel_map
        if excel_bytes:
            st.download_button(
                "Download Excel Model",
                data=excel_bytes,
                file_name="Ecommerce_Financial_Model.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if st.button(
                "Clear Prepared Excel",
                key=f"clear_excel_{selected_scenario.lower()}",
            ):
                excel_map.pop(selected_scenario, None)
                st.session_state.excel_bytes_map = excel_map
                excel_bytes = None
        if not excel_bytes:
            st.info("Click 'Prepare Excel Model' to generate the workbook for download.")



with page_tabs[10]:
    break_even_edit = _section_header("Break-Even Analysis Inputs", "break_even_inputs")
    break_even_inputs = _editable_table(
        "break_even_inputs",
        break_even_defaults,
        column_config={
            "Input": st.column_config.TextColumn("Input"),
            "Value": st.column_config.NumberColumn("Value", format="%0.2f"),
        },
        edit_enabled=break_even_edit,
    )

    st.subheader("Break-Even Results Background")
    break_even_revenue = annual_revenue.sum()
    break_even_costs = annual_capex.sum() + annual_equity_cf.abs().sum()
    st.metric("Breakeven revenue", f"${break_even_revenue:,.0f}")
    st.metric("Breakeven cost base", f"${break_even_costs:,.0f}")

    parameter_edit = _section_header("Key Parameter Naming", "parameter_naming")
    parameter_naming = _editable_table(
        "parameter_naming",
        parameter_naming_defaults,
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter"),
            "Preferred name": st.column_config.TextColumn("Preferred name"),
        },
        edit_enabled=parameter_edit,
    )
    st.dataframe(parameter_naming, use_container_width=True)

    st.subheader("Additional Context")
    st.write(
        "Background Information includes CAPEX requirements and feedstock demand assumptions. "
        "Use the input table above to refine the data that underpins break-even and payback outputs."
    )

st.subheader("Model Outputs Snapshot")
st.dataframe(summary.head(12).round(2), use_container_width=True)

cash_csv = summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download cash flow CSV",
    data=cash_csv,
    file_name="wte_cashflows.csv",
    mime="text/csv",
)

st.info(
    "All navigation is organised horizontally across the page. Use the tabs to explore inputs, "
    "results, sensitivities, and scenario tools without relying on a sidebar."
)

