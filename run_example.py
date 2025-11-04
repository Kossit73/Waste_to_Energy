"""Example runner for the waste-to-energy model."""
from __future__ import annotations

import numpy as np

from wte_model import (
    MonteCarloConfig,
    WTEMasterInputs,
    cashflow_model,
    default_inputs,
    run_monte_carlo,
    run_sensitivity,
)
from wte_model.io_excel import load_inputs_from_xlsm
from wte_model.scenario import DistributionSpec

EXCEL_PATH = "Waste_to_Energy_ Accra_Ghana.xlsm"


def _load_inputs() -> WTEMasterInputs:
    try:
        return load_inputs_from_xlsm(EXCEL_PATH)
    except FileNotFoundError:
        print("Workbook not found; falling back to packaged defaults.")
    except Exception as exc:
        print(f"Workbook load failed ({exc!r}); falling back to defaults.")
    return default_inputs()


def main() -> None:
    inputs = _load_inputs()

    mapped = getattr(type(inputs), "_MAPPED_DEBUG", None)
    if mapped:
        print("\n--- MAPPED INPUTS (from Excel named ranges where found) ---")
        for section, values in mapped.items():
            print(section, "=>", values)

    results = cashflow_model(inputs)

    print("\n--- RESULTS ---")
    print("Equity IRR:", round(100 * results["irr_eq"], 2), "%")
    print("Project IRR:", round(100 * results["irr_proj"], 2), "%")
    print("Min DSCR:", round(np.nanmin(results["dscr"]), 2))
    debt = results["debt"]
    print("LLCR:", round(debt.get("llcr", np.nan), 2))
    print("PLCR:", round(debt.get("plcr", np.nan), 2))

    print("\nWorking capital (first 6 periods):")
    wc = results["working_capital"]
    for key in ["receivables", "inventory", "payables", "cash_effect"]:
        print(f"  {key:>12}:", np.round(wc[key][:6], 2))

    print("\n--- Sensitivity: PPA price (USD/MWh) ---")
    sensitivity = run_sensitivity(
        inputs,
        "revenue.streams[0].price_curve.base",
        [90, 100, 110, 120, 130],
    )
    for row in sensitivity:
        print(
            f"PPA {row['value']:>6.0f} -> Equity IRR {row['irr_eq']*100:5.2f}%, "
            f"Project IRR {row['irr_proj']*100:5.2f}%",
        )

    print("\n--- Monte Carlo (10 iterations) ---")
    mc = run_monte_carlo(
        inputs,
        MonteCarloConfig(
            iterations=10,
            seed=1,
            distributions=[
                DistributionSpec(
                    path="tech.msw_tonnes_pa",
                    dist="normal",
                    params={"mean": inputs.tech.msw_tonnes_pa, "std": 10_000},
                    minimum=200_000,
                ),
                DistributionSpec(
                    path="revenue.streams[0].price_curve.base",
                    dist="uniform",
                    params={"low": 90, "high": 130},
                ),
            ],
        ),
    )
    for metric, stats in mc["summary"].items():
        print(metric, "=>", {k: round(v, 4) for k, v in stats.items()})


if __name__ == "__main__":
    main()
