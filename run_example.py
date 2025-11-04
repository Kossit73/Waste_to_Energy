"""Example runner for the waste-to-energy model."""
from __future__ import annotations

import numpy as np

from wte_model.dcf import cashflow_model
from wte_model.io_excel import load_inputs_from_xlsm

EXCEL_PATH = "Waste_to_Energy_ Accra_Ghana.xlsm"


def main() -> None:
    inputs = load_inputs_from_xlsm(EXCEL_PATH)

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

    for key in ["capex", "ebitda", "cfads", "equity_cf"]:
        arr = results.get(key)
        if arr is not None:
            print(key, ":", np.round(arr[:8], 2))


if __name__ == "__main__":
    main()
