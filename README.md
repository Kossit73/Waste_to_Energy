# Waste_to_Energy

This repository contains a simple Python implementation of a waste-to-energy
financial model. The code is organised as a small package (`wte_model`) that can
load assumptions from an Excel workbook and compute project-level metrics such
as EBITDA, CFADS, equity cash flows, IRRs, and DSCRs.

## Getting started

1. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Place the modelling workbook next to the code (the example runner expects a
   file named `Waste_to_Energy_ Accra_Ghana.xlsm`).

3. Execute the example script:

   ```bash
   python run_example.py
   ```

The script will display the mapped input values and headline financial metrics
calculated by the model.

## Streamlit dashboard

You can explore the model interactively with Streamlit. The app exposes the
full modelling workspace directly in the browser using the default
assumptions shipped with the repository.

```bash
streamlit run streamlit_app.py
```

Once the server starts, open the provided URL in your browser. Navigate across
the horizontal tabs to configure projection settings, edit inputs, review
metrics, and run sensitivities. Use the scenario download panel to export a
full Excel model whenever you need an offline copy.

### Editing the default figures

The workspace ships with illustrative assumptions but you can tailor every
section directly in the browser:

1. Toggle the **Edit** checkbox that appears in each section header to unlock
   its number inputs and data tables.
2. Use the **Add row**/**Remove row** buttons (available in edit mode) to adjust
   the schedule length, then press **Edit row** beside the line you want to
   update and submit the inline form to save your changes.
3. Open the **Manage defaults & state** panel on the *Input Landing* tab to
   restore the packaged defaults, clear all tables, or save/load your own
   custom preset.
4. Each schedule features a **Yearly increment** expander directly below the
   table. Open it while the section is in edit mode to apply compound annual
   changes to numeric columns without editing every row manually.
5. All edits cascade instantly through the dashboards, statements, and
   analytics tabs, so you can validate the impact without reloading the app.

### Troubleshooting Streamlit deployment

If the app fails to start with an error about missing packages (for example
`ModuleNotFoundError: No module named 'numpy'`), make sure the runtime has
installed the project requirements. Running `pip install -r requirements.txt`
inside the deployment environment resolves the issue. The Streamlit app now
performs a pre-flight dependency check and will point out any missing modules so
you can install them before redeploying.

## Package overview

- `wte_model/inputs.py` – dataclasses representing the model inputs.
- `wte_model/energy.py` – converts waste throughput into exported electricity.
- `wte_model/revenue.py` – computes revenues from power sales and waste gate fees.
- `wte_model/costs.py` – builds capex and operating cost schedules.
- `wte_model/finance.py` – handles debt schedules, depreciation, and tax.
- `wte_model/dcf.py` – assembles the full cash flow model and metrics.
- `wte_model/io_excel.py` – helper to read assumptions from a macro-enabled Excel workbook.
- `streamlit_app.py` – interactive dashboard built on top of the model.
- `run_example.py` – example entry point that ties everything together.

## Key capabilities

- **Operational variability** – model throughput, efficiency, and availability on a per-period basis to capture ramp-up, outages, and degradation.
- **Flexible commercial stack** – configure multiple revenue streams with independent drivers, indexation curves, and escalation rules (power, heat, gate fees, by-products, FX-linked tariffs, etc.).
- **Detailed capital assets** – track capex at the item level with distinct depreciation lives, bonus depreciation, and residual values for fixed-asset rollforwards.
- **Multi-facility financing** – represent senior, mezzanine, and working-capital facilities with interest-during-construction, sculpted or annuity amortisation, cash sweeps, and LLCR/PLCR coverage metrics.
- **Enhanced tax and working capital** – include loss carry-forwards, minimum taxes, withholding on distributions, and granular receivable/payable/inventory assumptions.
- **Risk analytics** – run one-click sensitivities, goal seeks, and Monte Carlo simulations to understand IRR/DSCR distributions and scenario comparisons.

Feel free to adapt the input mappings or extend the model logic to match your
specific project requirements.
