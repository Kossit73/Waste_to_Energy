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

## Package overview

- `wte_model/inputs.py` – dataclasses representing the model inputs.
- `wte_model/energy.py` – converts waste throughput into exported electricity.
- `wte_model/revenue.py` – computes revenues from power sales and waste gate fees.
- `wte_model/costs.py` – builds capex and operating cost schedules.
- `wte_model/finance.py` – handles debt schedules, depreciation, and tax.
- `wte_model/dcf.py` – assembles the full cash flow model and metrics.
- `wte_model/io_excel.py` – helper to read assumptions from a macro-enabled Excel workbook.
- `run_example.py` – example entry point that ties everything together.

Feel free to adapt the input mappings or extend the model logic to match your
specific project requirements.
