"""Utilities for exporting model results to Excel."""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .inputs import WTEMasterInputs


def _annualise(series: Iterable[float], periods_per_year: int) -> pd.Series:
    """Aggregate a periodic series into calendar-year totals."""
    arr = np.asarray(series, dtype=float)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    periods = np.arange(len(arr))
    years = periods // periods_per_year
    df = pd.DataFrame({"value": arr, "year": years})
    return df.groupby("year")[["value"]].sum().squeeze()


def build_summary_tables(
    inputs: WTEMasterInputs, results: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate summary tables used by both the UI and Excel export."""

    energy = results["energy"]
    revenue = results["rev"]
    capex = results["capex"]
    opex = results["opex"]
    tax = results["tax"]
    debt = results["debt"]

    capex_total = capex["total"]
    investment_total = capex.get("investment_total", capex_total)

    timeline = inputs.timeline
    periods = np.arange(timeline.n)
    years_axis = timeline.start_year + (periods // timeline.periods_per_year)

    summary = pd.DataFrame(
        {
            "Period": periods + 1,
            "Calendar Year": years_axis,
            "Tonnes": energy["tonnes"],
            "Net MWh": energy["net_mwh"],
            "Total revenue": revenue["total_revenue"],
            "Total opex": opex["total_opex"],
            "EBITDA": results["ebitda"],
            "CFADS": results["cfads"],
            "Debt service": debt["debt_service"],
            "Equity cash flow": results["equity_cf"],
            "Capex": investment_total,
            "Debt balance": debt["balance"],
            "Tax": tax["cash_tax"],
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

    production_annual_series = _annualise(
        energy["tonnes"], timeline.periods_per_year
    )

    return summary, summary_ann, summary_cumulative, production_annual_series


def generate_excel_bytes(
    model_inputs: WTEMasterInputs,
    results: Dict[str, Any],
    scenario_name: str,
) -> bytes:
    """Create an Excel workbook containing the key model outputs."""

    summary, summary_ann, summary_cumulative, _ = build_summary_tables(
        model_inputs, results
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame({"Scenario": [scenario_name]}).to_excel(
            writer, sheet_name="Scenario", index=False
        )
        summary.to_excel(writer, sheet_name="Period Summary", index=False)
        summary_ann.to_excel(writer, sheet_name="Annual Summary", index=False)
        summary_cumulative.to_excel(writer, sheet_name="Cumulative", index=False)
        pd.DataFrame(results["energy"]).to_excel(
            writer, sheet_name="Energy", index=False
        )
        pd.DataFrame(results["rev"]).to_excel(writer, sheet_name="Revenue", index=False)
        pd.DataFrame(results["opex"]).to_excel(writer, sheet_name="Opex", index=False)
        pd.DataFrame(results["capex"]).to_excel(writer, sheet_name="Capex", index=False)
        pd.DataFrame(results["debt"]).to_excel(writer, sheet_name="Debt", index=False)
        pd.DataFrame(results["tax"]).to_excel(writer, sheet_name="Tax", index=False)
        pd.DataFrame(results["working_capital"]).to_excel(
            writer, sheet_name="Working Capital", index=False
        )
        pd.DataFrame({"Equity cash flow": results["equity_cf"]}).to_excel(
            writer, sheet_name="Equity CF", index=False
        )
        pd.DataFrame({"CFADS": results["cfads"]}).to_excel(
            writer, sheet_name="CFADS", index=False
        )

    output.seek(0)
    return output.getvalue()
