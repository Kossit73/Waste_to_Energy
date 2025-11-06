"""Interactive Streamlit workspace for the waste-to-energy financial model."""

from __future__ import annotations

import importlib.util
import copy
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype
from streamlit.delta_generator import DeltaGenerator

from wte_model import (
    CapexItem,
    CostAssumptions,
    DebtFacility,
    FinanceAssumptions,
    OpexComponent,
    PriceCurve,
    RevenueAssumptions,
    RevenueStream,
    TaxAssumptions,
    TechAssumptions,
    Timeline,
    WorkingCapitalAssumptions,
    WTEMasterInputs,
    build_summary_tables,
    cashflow_model,
    default_inputs,
    generate_excel_bytes,
    goal_seek,
    run_monte_carlo,
    run_sensitivity,
)
from wte_model.scenario import DistributionSpec, MonteCarloConfig


AI_PROVIDER_OPTIONS = ("OpenAI", "Azure OpenAI", "Anthropic", "Vertex AI", "Custom")

ML_METHOD_LABELS = {
    "linear_regression": "Linear regression",
    "random_forest": "Random forest",
    "xgboost": "Gradient boosted trees",
    "prophet": "Prophet trend model",
}

ML_LABEL_TO_CODE = {label: code for code, label in ML_METHOD_LABELS.items()}

GEN_AI_FEATURE_LABELS = {
    "summary": "Executive summary",
    "risk": "Risk highlights",
    "recommendations": "Recommendations",
    "variance_analysis": "Variance analysis",
}

GEN_AI_LABEL_TO_CODE = {label: code for code, label in GEN_AI_FEATURE_LABELS.items()}

DEFAULT_AI_SETTINGS = {
    "enabled": False,
    "provider": "OpenAI",
    "model": "gpt-4",
    "forecast_horizon": 3,
    "ml_methods": ["linear_regression"],
    "generative_features": ["summary"],
    "api_key": "",
}


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


def _working_capital_from_tables(
    accounts_df: Optional[pd.DataFrame],
    inventory_df: Optional[pd.DataFrame],
    defaults,
):
    """Map the editable working-capital tables into model assumptions."""

    cfg = replace(defaults)

    def _iter_rows(df: Optional[pd.DataFrame]):
        if df is None:
            return []
        if df.empty or "Metric" not in df.columns or "Value" not in df.columns:
            return []
        return df[["Metric", "Value"]].itertuples(index=False, name=None)

    for metric_raw, value in _iter_rows(accounts_df):
        metric = str(metric_raw).strip().lower()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if "receivable" in metric:
            if "day" in metric:
                cfg.receivable_days = numeric
        elif "prepaid" in metric:
            if "day" in metric:
                cfg.prepaid_days = numeric
            else:
                cfg.prepaid_absolute = numeric
                cfg.prepaid_days = 0.0
        elif "other" in metric and "asset" in metric:
            cfg.other_asset_absolute = numeric
            cfg.other_current_asset_pct_revenue = 0.0

    for metric_raw, value in _iter_rows(inventory_df):
        metric = str(metric_raw).strip().lower()
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if "inventory" in metric and "day" in metric:
            cfg.inventory_days = numeric
        elif "payable" in metric and "day" in metric:
            cfg.payable_days = numeric
        elif "accrued" in metric or "expense" in metric:
            cfg.accrued_expense_absolute = numeric
            cfg.other_current_liability_pct_opex = 0.0

    return cfg


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
        if is_bool_dtype(series):
            row[col] = False
        elif is_numeric_dtype(series):
            row[col] = 0.0
        else:
            row[col] = ""
    return pd.DataFrame([row])


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


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        value = cleaned
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    coerced = _coerce_float(value)
    return default if coerced is None else coerced


def _safe_int(value: Any, default: int = 0) -> int:
    coerced = _coerce_float(value)
    if coerced is None:
        return default
    try:
        return int(round(coerced))
    except (TypeError, ValueError):
        return default


def _safe_percentage(value: Any, default: float = 0.0) -> float:
    coerced = _coerce_float(value)
    if coerced is None:
        return default
    return coerced / 100.0 if abs(coerced) > 1.0 else coerced


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def _parse_series_cell(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        parts = list(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        parts = [part.strip() for part in re.split(r"[;,]", cleaned) if part.strip()]
    else:
        parts = [value]

    series: List[float] = []
    for part in parts:
        coerced = _coerce_float(part)
        if coerced is None:
            continue
        series.append(float(coerced))
    return series or None


def _normalise_label(label: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(label).strip().lower())
    return cleaned.strip("_")


PARAMETER_PATH_ALIASES: Dict[str, str] = {
    "ppa_price": "revenue.ppa_price_usd_per_mwh",
    "ppa_price_usd_per_mwh": "revenue.ppa_price_usd_per_mwh",
    "ppa_escalation": "revenue.ppa_escalation",
    "gate_fee": "revenue.gate_fee_usd_per_t",
    "gate_fee_usd_per_t": "revenue.gate_fee_usd_per_t",
    "gate_fee_escalation": "revenue.gate_fee_escalation",
    "heat_price": "revenue.heat_price_usd_per_mwh",
    "heat_price_usd_per_mwh": "revenue.heat_price_usd_per_mwh",
    "electricity_price_per_kwh": "revenue.ppa_price_usd_per_mwh",
    "electricity_price_usd_kwh": "revenue.ppa_price_usd_per_mwh",
    "metal_recovery": "revenue.metal_recovery_usd_per_t",
    "ash_revenue": "revenue.ash_revenue_usd_per_t",
    "capex": "costs.capex_total_usd",
    "capex_total": "costs.capex_total_usd",
    "capex_total_usd": "costs.capex_total_usd",
    "fixed_om": "costs.fixed_om_usd_pa",
    "fixed_o_m": "costs.fixed_om_usd_pa",
    "fixed_om_usd_pa": "costs.fixed_om_usd_pa",
    "variable_om": "costs.variable_om_usd_per_t",
    "variable_o_m": "costs.variable_om_usd_per_t",
    "variable_om_usd_per_t": "costs.variable_om_usd_per_t",
    "opex_per_ton_of_waste": "costs.variable_om_usd_per_t",
    "landfill_disposal": "costs.landfill_disposal_usd_per_t",
    "landfill_disposal_usd_per_t": "costs.landfill_disposal_usd_per_t",
    "insurance_pct": "costs.insurance_pct_of_capex_pa",
    "insurance_pct_of_capex_pa": "costs.insurance_pct_of_capex_pa",
    "maintenance_pct": "costs.maintenance_pct_of_capex_pa",
    "maintenance_pct_of_capex_pa": "costs.maintenance_pct_of_capex_pa",
    "opex_escalation": "costs.opex_escalation",
    "debt_ratio": "finance.debt_ratio",
    "interest_rate": "finance.interest_rate",
    "upfront_fee": "finance.upfront_fee_pct",
    "upfront_fee_pct": "finance.upfront_fee_pct",
    "tenor_years": "finance.tenor_years",
    "grace_years": "finance.grace_years",
    "discount_rate": "finance.discount_rate",
    "tax_rate": "finance.tax_rate",
    "corporate_tax_rate": "finance.tax_rate",
    "working_cap_days": "finance.working_cap_days",
    "depr_years": "finance.depr_years",
    "availability": "tech.availability",
    "plant_availability": "tech.availability",
    "msw_tonnes_pa": "tech.msw_tonnes_pa",
    "lhv_mj_per_kg": "tech.lhv_mj_per_kg",
    "boiler_efficiency": "tech.boiler_efficiency",
    "electrical_efficiency": "tech.electrical_efficiency",
    "parasitic_load": "tech.parasitic_load_frac",
}


def _parameter_alias_map() -> Dict[str, str]:
    alias_map = dict(PARAMETER_PATH_ALIASES)
    df = st.session_state.get("parameter_naming")
    if df is None or df.empty:
        return alias_map
    if "Parameter" not in df.columns or "Preferred name" not in df.columns:
        return alias_map
    for _, row in df.iterrows():
        base_label = _normalise_label(row.get("Parameter", ""))
        alias_label = _normalise_label(row.get("Preferred name", ""))
        if not base_label or not alias_label:
            continue
        path = PARAMETER_PATH_ALIASES.get(base_label)
        if path:
            alias_map[alias_label] = path
    return alias_map


def _resolve_parameter_path(label: str, alias_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    if label is None:
        return None
    label = str(label).strip()
    if not label:
        return None
    if "." in label or "[" in label:
        return label
    mapping = alias_map or _parameter_alias_map()
    normalised = _normalise_label(label)
    return mapping.get(normalised)


def _read_input_value(root: WTEMasterInputs, path: str) -> Optional[float]:
    try:
        current: Any = root
        for token in path.split("."):
            if "[" in token and token.endswith("]"):
                attr, idx = token[:-1].split("[")
                current = getattr(current, attr)[int(idx)]
            else:
                current = getattr(current, token)
        if isinstance(current, (int, float, np.number)):
            return float(current)
        return _coerce_float(current)
    except Exception:
        return None


def _apply_adjustment(base_value: float, adjustment: Any) -> Optional[float]:
    if adjustment is None:
        return None
    adj_value = _coerce_float(adjustment)
    if adj_value is None:
        return None
    if abs(base_value) <= 1e-9:
        return adj_value
    if abs(adj_value) <= 1.5:
        return float(base_value) * (1.0 + float(adj_value))
    return float(adj_value)


def _macro_indices_from_table(
    table: Optional[pd.DataFrame],
    timeline: Timeline,
) -> Dict[str, List[float]]:
    if table is None or table.empty:
        return {}

    macro: Dict[str, List[float]] = {}
    horizon_years = max(1, int(timeline.years))
    for _, row in table.iterrows():
        category_raw = row.get("Category") or row.get("Label")
        category = str(category_raw).strip() if category_raw is not None else ""
        if not category:
            continue
        key = _normalise_label(category)
        explicit_series = (
            _parse_series_cell(row.get("Series"))
            or _parse_series_cell(row.get("Inflation curve"))
        )
        if explicit_series:
            values = [float(val) for val in explicit_series if _coerce_float(val) is not None]
        else:
            rate = _coerce_float(row.get("Inflation rate (%)"))
            if rate is None:
                continue
            rate = rate / 100.0 if abs(rate) > 1.0 else rate
            values = [float(rate)]
        if not values:
            continue
        repeat = max(1, math.ceil(horizon_years / len(values)))
        expanded = (values * repeat)[:horizon_years]
        macro[key] = expanded
    return macro


def _risk_summary_from_table(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame()
    if "Probability (%)" not in table.columns or "Impact (USD)" not in table.columns:
        return pd.DataFrame()
    df = table.copy()
    df["Probability (%)"] = df["Probability (%)"].apply(lambda v: _safe_float(v, 0.0))
    df["Impact (USD)"] = df["Impact (USD)"].apply(lambda v: _safe_float(v, 0.0))
    df["Expected loss (USD)"] = df["Probability (%)"] / 100.0 * df["Impact (USD)"]
    return df


METRIC_ALIASES: Dict[str, str] = {
    "equity_irr": "irr_eq",
    "project_irr": "irr_proj",
    "dscr": "dscr_min",
    "irr_eq": "irr_eq",
    "irr_proj": "irr_proj",
    "dscr_min": "dscr_min",
}


def _resolve_metric_code(label: str) -> Optional[str]:
    if not label:
        return None
    normalised = _normalise_label(label)
    return METRIC_ALIASES.get(normalised)


def _sensitivity_results_from_table(
    table: Optional[pd.DataFrame],
    base_inputs: WTEMasterInputs,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if table is None or table.empty:
        return pd.DataFrame(), warnings
    if "Driver" not in table.columns:
        warnings.append("Sensitivity table is missing a 'Driver' column.")
        return pd.DataFrame(), warnings

    alias_map = _parameter_alias_map()
    records: List[Dict[str, Any]] = []

    for _, row in table.iterrows():
        driver_label = str(row.get("Driver", "")).strip()
        if not driver_label:
            continue
        path = _resolve_parameter_path(driver_label, alias_map)
        if not path:
            warnings.append(f"Could not map driver '{driver_label}' to a model input.")
            continue
        base_value = _read_input_value(base_inputs, path)
        if base_value is None:
            warnings.append(f"Unable to read the current value for '{driver_label}'.")
            continue

        case_labels: List[str] = []
        case_values: List[float] = []
        for label_name in ("Low", "Base", "High"):
            adjusted = _apply_adjustment(float(base_value), row.get(label_name))
            if label_name == "Base" and adjusted is None:
                adjusted = float(base_value)
            if adjusted is None:
                continue
            case_labels.append(label_name)
            case_values.append(float(adjusted))

        if not case_values:
            case_labels = ["Base"]
            case_values = [float(base_value)]

        unique_labels: List[str] = []
        unique_values: List[float] = []
        seen: Set[Tuple[str, float]] = set()
        for label_name, value in zip(case_labels, case_values):
            key = (label_name, round(float(value), 12))
            if key in seen:
                continue
            seen.add(key)
            unique_labels.append(label_name)
            unique_values.append(float(value))

        try:
            outputs = run_sensitivity(base_inputs, path, unique_values)
        except Exception as exc:
            warnings.append(f"Sensitivity run for '{driver_label}' failed: {exc}")
            continue

        for label_name, result in zip(unique_labels, outputs):
            records.append(
                {
                    "Driver": driver_label,
                    "Case": label_name,
                    "Test value": float(result.get("value", float("nan"))),
                    "Equity IRR": float(result.get("irr_eq", float("nan"))),
                    "Project IRR": float(result.get("irr_proj", float("nan"))),
                    "DSCR minimum": float(result.get("dscr_min", float("nan"))),
                }
            )

    return pd.DataFrame(records), warnings


def _monte_carlo_config_from_table(
    table: Optional[pd.DataFrame],
    base_inputs: WTEMasterInputs,
) -> Tuple[Optional[MonteCarloConfig], List[str]]:
    warnings: List[str] = []
    if table is None or table.empty:
        return None, warnings
    if "Variable" not in table.columns:
        warnings.append("Monte Carlo table is missing a 'Variable' column.")
        return None, warnings

    alias_map = _parameter_alias_map()

    iterations = 500
    if "Iterations" in table.columns:
        first_iter = table["Iterations"].dropna().iloc[0] if not table["Iterations"].dropna().empty else None
        if first_iter is not None:
            iterations = max(1, int(_safe_int(first_iter, iterations)))

    seed: Optional[int] = None
    if "Seed" in table.columns:
        first_seed = table["Seed"].dropna().iloc[0] if not table["Seed"].dropna().empty else None
        if first_seed is not None:
            seed = int(_safe_int(first_seed, 0))

    distributions: List[DistributionSpec] = []

    for _, row in table.iterrows():
        variable = str(row.get("Variable", "")).strip()
        if not variable:
            continue
        path = _resolve_parameter_path(variable, alias_map)
        if not path:
            warnings.append(f"Could not map Monte Carlo variable '{variable}' to a model input.")
            continue

        dist_raw = row.get("Distribution", "normal")
        dist = str(dist_raw).strip().lower() if isinstance(dist_raw, str) else "normal"
        base_value = _read_input_value(base_inputs, path) or 0.0

        params: Dict[str, float] = {}
        minimum = _coerce_float(row.get("Min"))
        maximum = _coerce_float(row.get("Max"))

        if dist == "normal":
            mean = _coerce_float(row.get("Mean"))
            std = _coerce_float(row.get("Std dev"))
            params["mean"] = float(base_value if mean is None else mean)
            params["std"] = float(abs(base_value) * 0.05 if std is None else std)
            if params["std"] <= 0:
                params["std"] = max(1e-6, abs(params["mean"]) * 0.01)
        elif dist == "triangular":
            left = _coerce_float(row.get("Left"))
            mode = _coerce_float(row.get("Mode"))
            right = _coerce_float(row.get("Right"))
            if mode is None:
                mode = _coerce_float(row.get("Mean"))
            if left is None or right is None:
                spread = _coerce_float(row.get("Std dev"))
                if spread is None:
                    spread = abs(base_value) * 0.1 or 1.0
                centre = mode if mode is not None else base_value
                left = centre - spread
                right = centre + spread
            if mode is None:
                mode = base_value
            params.update({
                "left": float(left),
                "mode": float(mode),
                "right": float(right),
            })
        elif dist == "uniform":
            low = _coerce_float(row.get("Low"))
            high = _coerce_float(row.get("High"))
            if low is None or high is None:
                width = _coerce_float(row.get("Std dev")) or abs(base_value) * 0.1 or 1.0
                centre = _coerce_float(row.get("Mean")) or base_value
                low = (centre or 0.0) - width
                high = (centre or 0.0) + width
            params.update({"low": float(low), "high": float(high)})
        elif dist == "lognormal":
            mean = _coerce_float(row.get("Mean"))
            sigma = _coerce_float(row.get("Std dev"))
            params["mean"] = float(math.log(max(base_value, 1e-6)) if mean is None else mean)
            params["sigma"] = float(abs(base_value) * 0.05 if sigma is None else sigma)
        else:
            warnings.append(f"Unsupported distribution '{dist}' for variable '{variable}'.")
            continue

        distributions.append(
            DistributionSpec(
                path=path,
                dist=dist,
                params=params,
                minimum=None if minimum is None else float(minimum),
                maximum=None if maximum is None else float(maximum),
            )
        )

    if not distributions:
        warnings.append("No valid Monte Carlo distributions configured.")
        return None, warnings

    config = MonteCarloConfig(iterations=iterations, seed=seed, distributions=distributions)
    return config, warnings


def _goal_seek_results_from_table(
    table: Optional[pd.DataFrame],
    base_inputs: WTEMasterInputs,
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if table is None or table.empty:
        return pd.DataFrame(), warnings
    if "Target metric" not in table.columns or "Variable" not in table.columns:
        warnings.append("Goal seek table requires 'Target metric' and 'Variable' columns.")
        return pd.DataFrame(), warnings

    alias_map = _parameter_alias_map()
    records: List[Dict[str, Any]] = []

    for _, row in table.iterrows():
        metric_label = str(row.get("Target metric", "")).strip()
        metric_code = _resolve_metric_code(metric_label)
        if not metric_code:
            warnings.append(f"Unsupported target metric '{metric_label}'.")
            continue

        variable_label = str(row.get("Variable", "")).strip()
        path = _resolve_parameter_path(variable_label, alias_map)
        if not path:
            warnings.append(f"Could not map goal seek variable '{variable_label}' to a model input.")
            continue

        target_value_raw = row.get("Target value")
        target_value = _coerce_float(target_value_raw)
        if target_value is None:
            warnings.append(f"Target value for '{metric_label}' is not numeric.")
            continue

        base_value = _read_input_value(base_inputs, path)
        if base_value is None:
            warnings.append(f"Unable to read the current value for '{variable_label}'.")
            continue

        min_override = _coerce_float(row.get("Min"))
        max_override = _coerce_float(row.get("Max"))
        span = abs(base_value) if abs(base_value) > 1e-9 else max(abs(target_value), 1.0)
        lower = float(min_override) if min_override is not None else float(base_value - span)
        upper = float(max_override) if max_override is not None else float(base_value + span)
        if lower == upper:
            upper = lower + (abs(lower) or 1.0)

        if lower > upper:
            lower, upper = upper, lower

        try:
            result = goal_seek(base_inputs, path, metric_code, float(target_value), (lower, upper))
        except Exception as exc:
            warnings.append(f"Goal seek for '{metric_label}' failed: {exc}")
            continue

        records.append(
            {
                "Metric": metric_label,
                "Variable": variable_label,
                "Target": float(target_value),
                "Solution": float(result.get("value", float("nan"))),
                "Metric achieved": float(result.get(metric_code, float("nan"))),
                "Bracket": f"[{lower:.4g}, {upper:.4g}]",
            }
        )

    return pd.DataFrame(records), warnings


def _table_differs(table: Optional[pd.DataFrame], template: Optional[pd.DataFrame]) -> bool:
    if table is None:
        return False
    if template is None:
        return True
    try:
        return not table.reset_index(drop=True).equals(template.reset_index(drop=True))
    except Exception:
        return True


def _capex_items_from_table(
    table: Optional[pd.DataFrame],
    template: Sequence[CapexItem],
) -> List[CapexItem]:
    if table is None or table.empty:
        return [replace(item) for item in template]

    items: List[CapexItem] = []
    for idx, row in table.reset_index(drop=True).iterrows():
        name_raw = row.get("Item") or row.get("Asset") or f"Item {idx + 1}"
        name = str(name_raw).strip() or f"Item {idx + 1}"
        amount = _safe_float(row.get("Cost"), 0.0)
        if amount == 0 and not name.strip():
            continue
        life_years = max(1, _safe_int(row.get("Life (years)"), 1))
        bonus_pct = _safe_percentage(row.get("Bonus depreciation (%)"), 0.0)
        residual_pct = _safe_percentage(row.get("Residual value (%)"), 0.0)
        method_raw = row.get("Depreciation method") or row.get("Method") or "straight_line"
        method = str(method_raw).strip().lower().replace(" ", "_")
        if method not in {"straight_line", "declining_balance"}:
            method = "straight_line"
        declining_rate = _parse_series_cell(row.get("Declining balance rate (%)"))
        declining_value = None
        if declining_rate:
            declining_value = _safe_percentage(declining_rate[0], 0.0)
        spend_profile = _parse_series_cell(row.get("Spend profile"))
        inflation_curve = _parse_series_cell(row.get("Inflation curve"))

        items.append(
            CapexItem(
                name=name,
                amount=amount,
                life_years=life_years,
                bonus_depreciation_pct=bonus_pct,
                residual_value_pct=residual_pct,
                method=method,
                declining_balance_rate=declining_value,
                spend_profile=spend_profile,
                inflation_curve=inflation_curve,
            )
        )

    if items:
        return items
    return [replace(item) for item in template]


def _infer_revenue_driver(name: str, raw_driver: Any) -> str:
    driver_map = {
        "net_mwh": "net_mwh",
        "gross_mwh": "gross_mwh",
        "heat_mwh": "heat_mwh",
        "tonnes": "tonnes",
        "tons": "tonnes",
        "custom": "custom",
    }
    if isinstance(raw_driver, str):
        candidate = raw_driver.strip().lower().replace(" ", "_")
        if candidate in driver_map:
            return driver_map[candidate]
        if candidate in {"electricity", "ppa", "power"}:
            return "net_mwh"
        if candidate in {"heat", "steam"}:
            return "heat_mwh"
        if candidate in {"waste", "gate", "tipping"}:
            return "tonnes"

    label = name.lower()
    if any(token in label for token in ("heat", "steam")):
        return "heat_mwh"
    if any(token in label for token in ("ppa", "electric", "power")):
        return "net_mwh"
    if any(token in label for token in ("gate", "tipping", "waste")):
        return "tonnes"
    return "tonnes"


def _revenue_assumptions_from_table(
    table: Optional[pd.DataFrame],
    template: RevenueAssumptions,
) -> RevenueAssumptions:
    base = copy.deepcopy(template)
    if table is None or table.empty:
        return base

    streams: List[RevenueStream] = []
    for idx, row in table.reset_index(drop=True).iterrows():
        name_raw = row.get("Revenue stream") or row.get("Stream") or f"Stream {idx + 1}"
        name = str(name_raw).strip() or f"Stream {idx + 1}"
        driver = _infer_revenue_driver(name, row.get("Driver"))
        price = _safe_float(row.get("Price"), 0.0)
        escalation = _safe_percentage(row.get("Escalation (%)"), 0.0)
        share = _safe_percentage(row.get("Share"), 1.0)
        if share == 0:
            share = _safe_percentage(row.get("Share (%)"), 1.0)
        quantity_profile = _parse_series_cell(row.get("Quantity profile"))
        seasonality = _parse_series_cell(row.get("Seasonality"))
        periodic = _parse_series_cell(row.get("Periodic multipliers"))
        index_curve = _parse_series_cell(row.get("Index curve"))
        fx_curve = _parse_series_cell(row.get("FX curve"))
        adders = _parse_series_cell(row.get("Adders"))
        currency_raw = row.get("Currency")
        currency = str(currency_raw).strip() if isinstance(currency_raw, str) else "USD"
        notes_raw = row.get("Notes")
        notes = str(notes_raw).strip() if isinstance(notes_raw, str) and notes_raw else None

        price_curve = PriceCurve(
            base=price,
            annual_escalation=escalation,
            periodic_multipliers=periodic,
            index_curve=index_curve,
            fx_curve=fx_curve,
            adders=adders,
        )

        streams.append(
            RevenueStream(
                name=name,
                driver=driver,
                price_curve=price_curve,
                share=share,
                quantity_profile=quantity_profile,
                seasonality=seasonality,
                currency=currency or "USD",
                notes=notes,
            )
        )

    if streams:
        base.streams = streams
    return base


def _aggregate_pattern(values: Sequence[float], periods_per_year: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or periods_per_year <= 0:
        return np.zeros(max(periods_per_year, 1), dtype=float)
    segment_edges = np.linspace(0.0, 1.0, arr.size + 1)
    period_edges = np.linspace(0.0, 1.0, periods_per_year + 1)
    totals = np.zeros(periods_per_year, dtype=float)
    for seg_idx in range(arr.size):
        seg_start, seg_end = segment_edges[seg_idx], segment_edges[seg_idx + 1]
        seg_value = arr[seg_idx]
        span = seg_end - seg_start
        if span <= 0:
            continue
        for period_idx in range(periods_per_year):
            period_start, period_end = period_edges[period_idx], period_edges[period_idx + 1]
            overlap = min(seg_end, period_end) - max(seg_start, period_start)
            if overlap <= 0:
                continue
            totals[period_idx] += seg_value * (overlap / span)
    return totals


def _components_from_monthly_table(
    table: Optional[pd.DataFrame],
    timeline: Timeline,
    *,
    category_label: str,
) -> List[OpexComponent]:
    if table is None or table.empty:
        return []

    ordered = table.reset_index(drop=True)
    numeric_cols = [
        col
        for col in ordered.columns
        if is_numeric_dtype(ordered[col]) and not str(col).lower().startswith("month")
    ]
    components: List[OpexComponent] = []
    if not numeric_cols:
        return components

    for col in numeric_cols:
        col_values: List[float] = []
        last_value = 0.0
        for _, row in ordered.iterrows():
            value = _coerce_float(row.get(col))
            if value is None:
                value = last_value if col_values else 0.0
            col_values.append(float(value))
            last_value = float(value)

        if not col_values:
            continue

        period_pattern = _aggregate_pattern(col_values, timeline.periods_per_year)
        if not period_pattern.size:
            continue

        if timeline.years > 1:
            reps = timeline.years
        else:
            reps = 1
        repeated = np.tile(period_pattern, reps)
        if repeated.size < timeline.n:
            reps_needed = (timeline.n + period_pattern.size - 1) // period_pattern.size
            repeated = np.tile(period_pattern, reps_needed)
        series = repeated[: timeline.n]

        category = category_label.lower().replace(" ", "_")
        name = f"{category_label} - {col}" if category_label else str(col)
        components.append(
            OpexComponent(
                name=name,
                category=category,
                fixed_annual=0.0,
                variable_per_unit=1.0,
                driver="custom",
                custom_quantity=series.tolist(),
            )
        )

    return components


def _assemble_opex_components(
    direct_df: Optional[pd.DataFrame],
    staff_df: Optional[pd.DataFrame],
    other_df: Optional[pd.DataFrame],
    timeline: Timeline,
) -> List[OpexComponent]:
    components: List[OpexComponent] = []
    components.extend(_components_from_monthly_table(direct_df, timeline, category_label="Direct costs"))
    components.extend(_components_from_monthly_table(staff_df, timeline, category_label="Staff"))
    components.extend(_components_from_monthly_table(other_df, timeline, category_label="Other opex"))
    return components


def _debt_facilities_from_table(
    table: Optional[pd.DataFrame],
    template: Sequence[DebtFacility],
) -> List[DebtFacility]:
    if table is None or table.empty:
        return [replace(item) for item in template]

    facilities: List[DebtFacility] = []
    for idx, row in table.reset_index(drop=True).iterrows():
        name_raw = row.get("Facility") or row.get("Lender") or f"Facility {idx + 1}"
        name = str(name_raw).strip() or f"Facility {idx + 1}"
        draw_ratio = _safe_percentage(row.get("Draw ratio (%)"), 0.0)
        commitment = _safe_float(row.get("Base amount"), 0.0)
        interest_rate = _safe_percentage(row.get("Interest rate (%)"), 0.0)
        tenor_years = max(0, _safe_int(row.get("Duration (years)"), 0))
        if tenor_years == 0:
            tenor_years = max(0, _safe_int(row.get("Tenor (years)"), 0))
        grace_years = max(0, _safe_int(row.get("Grace (years)"), 0))
        amort_raw = row.get("Loan type") or row.get("Amortization") or "annuity"
        amortization = str(amort_raw).strip().lower().replace(" ", "_")
        if amortization not in {"annuity", "straight_line", "custom", "sculpted"}:
            amortization = "annuity"
        custom_amort = _parse_series_cell(row.get("Amort profile"))
        draw_profile = _parse_series_cell(row.get("Draw profile"))
        target_dscr = _safe_float(row.get("Target DSCR"), 1.2)
        sweep_pct = _safe_percentage(row.get("Cash sweep (%)"), 0.0)
        sweep_trigger = _safe_float(row.get("Cash sweep trigger"), 1.0)
        upfront_fee = _safe_percentage(row.get("Upfront fee (%)"), 0.0)
        sculpt_flag = _safe_bool(row.get("Sculpt from CFADS?"), False)
        interest_cap = _safe_bool(row.get("Capitalise interest"), True)

        if draw_ratio <= 0 and commitment <= 0:
            continue

        facilities.append(
            DebtFacility(
                name=name,
                draw_ratio=draw_ratio,
                commitment=commitment if commitment > 0 else None,
                draw_profile=draw_profile,
                interest_rate=interest_rate,
                tenor_years=max(1, tenor_years),
                grace_years=max(0, grace_years),
                amortization=amortization,
                custom_amort_profile=custom_amort,
                target_dscr=target_dscr,
                sculpt_from_cfads=sculpt_flag,
                cash_sweep_pct=sweep_pct,
                cash_sweep_trigger=sweep_trigger,
                upfront_fee_pct=upfront_fee,
                interest_during_construction=interest_cap,
            )
        )

    if facilities:
        return facilities
    return [replace(item) for item in template]


def _tax_assumptions_from_table(
    table: Optional[pd.DataFrame],
    base_tax: TaxAssumptions,
    timeline: Timeline,
) -> TaxAssumptions:
    cfg = replace(base_tax)
    if table is None or table.empty:
        return cfg

    for _, row in table.iterrows():
        label_raw = row.get("Tax") or row.get("Parameter") or ""
        label = str(label_raw).strip().lower()
        rate_value = row.get("Rate (%)")
        if label:
            if "corporate" in label:
                cfg.corporate_rate = _safe_percentage(rate_value, cfg.corporate_rate)
            elif "withholding" in label:
                cfg.withholding_rate = _safe_percentage(rate_value, cfg.withholding_rate)
            elif "minimum" in label:
                cfg.minimum_tax_rate = _safe_percentage(rate_value, cfg.minimum_tax_rate)
            elif "carbon" in label:
                cfg.carbon_credit_per_mwh = _safe_float(row.get("Value"), cfg.carbon_credit_per_mwh)

        carry_val = row.get("Loss carryforward (years)") or row.get("Carryforward (years)")
        if carry_val is not None and not pd.isna(carry_val):
            cfg.carryforward_years = _safe_int(carry_val, cfg.carryforward_years)

        holiday_years = row.get("Holiday (years)") or row.get("Holiday years")
        if holiday_years is not None and not pd.isna(holiday_years):
            cfg.holiday_years = max(0, _safe_int(holiday_years, cfg.holiday_years))

        holiday_start = row.get("Holiday start (months)") or row.get("Holiday start")
        if holiday_start is not None and not pd.isna(holiday_start):
            months_per_period = 12 / max(1, timeline.periods_per_year)
            cfg.holiday_start_offset = max(0, _safe_int(float(holiday_start) / months_per_period, cfg.holiday_start_offset))

        allow_loss = row.get("Allow loss carryforward")
        if allow_loss is not None and not pd.isna(allow_loss):
            cfg.allow_loss_carryforward = _safe_bool(allow_loss, cfg.allow_loss_carryforward)

        incentives = row.get("Other incentives")
        parsed_incentives = _parse_series_cell(incentives)
        if parsed_incentives:
            cfg.other_incentives = parsed_incentives

    return cfg


def _payload_to_ai_settings(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    """Extract AI settings from an arbitrary payload dictionary."""

    settings = dict(DEFAULT_AI_SETTINGS)
    if isinstance(payload, dict):
        raw = payload.get("ai_settings") or payload.get("ai") or {}
        if isinstance(raw, dict):
            for key in settings:
                if key in raw:
                    settings[key] = raw[key]
    return settings


def _ai_settings_to_payload(settings: Dict[str, Any], payload: Dict[str, Any] | None) -> None:
    """Persist AI settings back onto the supplied payload dictionary."""

    if not isinstance(payload, dict):
        return
    stored = {
        "enabled": bool(settings.get("enabled", False)),
        "provider": settings.get("provider", DEFAULT_AI_SETTINGS["provider"]),
        "model": settings.get("model", DEFAULT_AI_SETTINGS["model"]),
        "forecast_horizon": int(settings.get("forecast_horizon", DEFAULT_AI_SETTINGS["forecast_horizon"])),
        "ml_methods": list(settings.get("ml_methods", DEFAULT_AI_SETTINGS["ml_methods"])),
        "generative_features": list(
            settings.get("generative_features", DEFAULT_AI_SETTINGS["generative_features"])
        ),
        "api_key": settings.get("api_key", ""),
    }
    payload["ai_settings"] = stored


def _rerun() -> None:
    """Trigger a logical refresh without calling the deprecated rerun API."""

    st.session_state["ai_settings_revision"] = (
        st.session_state.get("ai_settings_revision", 0) + 1
    )


def _render_ai_settings(payload: Dict[str, Any], container: Optional[DeltaGenerator] = None) -> None:
    target = container or st
    settings = st.session_state.setdefault("ai_settings", _payload_to_ai_settings(payload))
    st.session_state.setdefault("ai_api_key", settings.get("api_key", ""))

    provider_options = list(AI_PROVIDER_OPTIONS)
    if settings.get("provider") not in provider_options:
        provider_options.append(settings.get("provider"))

    current_provider = settings.get("provider", "OpenAI")
    try:
        provider_index = provider_options.index(current_provider)
    except ValueError:
        provider_index = 0

    ml_defaults = [
        ML_METHOD_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("ml_methods", ["linear_regression"])
    ]
    feature_defaults = [
        GEN_AI_FEATURE_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("generative_features", ["summary"])
    ]

    form = target.form("ai_settings_form")
    with form:
        enabled = form.checkbox(
            "Enable AI Enhancements",
            value=bool(settings.get("enabled", False)),
            help="Toggle machine-learning forecasts and generative commentary.",
        )
        provider = form.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            help="Select the API provider powering generative insights.",
        )
        model = form.text_input(
            "Model",
            value=settings.get("model", "gpt-4"),
            help="Name of the deployed model (for example `gpt-4o-mini`).",
        )
        horizon = form.number_input(
            "Forecast Horizon (years)",
            min_value=0,
            max_value=20,
            value=int(settings.get("forecast_horizon", 3)),
            step=1,
            help="Number of additional years used for machine-learning revenue forecasts.",
        )

        ml_selection = form.multiselect(
            "Machine Learning Methods",
            list(ML_METHOD_LABELS.values()),
            default=ml_defaults,
            help="Choose algorithms applied to projected net revenue.",
        )
        feature_selection = form.multiselect(
            "Generative Features",
            list(GEN_AI_FEATURE_LABELS.values()),
            default=feature_defaults,
            help="Pick the narrative focus areas generated by the AI summary.",
        )
        api_key = form.text_input(
            "API Key",
            value=st.session_state.get("ai_api_key", ""),
            type="password",
            help="Store your provider API key securely. Keys are retained only for the current session.",
        )

        submitted = form.form_submit_button("Save AI Configuration")

    if submitted:
        ml_codes = [ML_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower()) for label in ml_selection]
        feature_codes = [
            GEN_AI_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower())
            for label in feature_selection
        ]

        settings.update(
            {
                "enabled": enabled,
                "provider": provider,
                "model": model.strip() or "gpt-4",
                "forecast_horizon": int(horizon),
                "ml_methods": ml_codes or ["linear_regression"],
                "generative_features": feature_codes or ["summary"],
                "api_key": api_key.strip(),
            }
        )
        st.session_state["ai_settings"] = settings
        st.session_state["ai_api_key"] = settings.get("api_key", "")
        _ai_settings_to_payload(settings, payload)
        st.success("AI configuration updated. Rerunning the model with the new settings.")
        _rerun()


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


def _input_widget_for_column(
    *,
    column: str,
    value: Any,
    series: Optional[pd.Series],
    key: str,
    disabled: bool = False,
) -> Any:
    """Render an input widget appropriate for the column type."""

    bool_series = series is not None and is_bool_dtype(series)
    if bool_series:
        default_bool = False if value is None or pd.isna(value) else bool(value)
        return st.checkbox(
            column,
            value=default_bool,
            key=key,
            disabled=disabled,
        )

    numeric_series = series is not None and is_numeric_dtype(series)
    integer_series = series is not None and is_integer_dtype(series)

    if not numeric_series and isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        numeric_series = True
        integer_series = isinstance(value, (int, np.integer))

    if numeric_series:
        default_value = 0.0 if value is None or pd.isna(value) else float(value)
        if integer_series:
            widget_val = st.number_input(
                column,
                value=int(default_value),
                step=1,
                key=key,
                disabled=disabled,
            )
            return int(widget_val)
        widget_val = st.number_input(
            column,
            value=float(default_value),
            step=0.01,
            format="%.6f",
            key=key,
            disabled=disabled,
        )
        return float(widget_val)

    default_text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    return st.text_input(column, value=default_text, key=key, disabled=disabled)


def _editable_table(
    key: str,
    data: pd.DataFrame,
    *,
    column_config: Optional[Dict[str, st.column_config.BaseColumn]] = None,
    allow_row_controls: bool = True,
    new_row_factory: Optional[Callable[[], pd.DataFrame]] = None,
    edit_enabled: bool = True,
    row_edit_controls: bool = False,
    row_label_field: Optional[str] = None,
) -> pd.DataFrame:
    base = _ensure_state_df(key, data.copy())

    dialog_key = f"{key}_add_dialog_open"
    pending_key = f"{key}_pending_new_row"

    if not edit_enabled:
        st.session_state.pop(dialog_key, None)
        st.session_state.pop(pending_key, None)

    if allow_row_controls and edit_enabled:
        ctrl_cols = st.columns(2)
        with ctrl_cols[0]:
            if st.button("Add row", key=f"add_{key}"):
                template_source = base if not base.empty else data
                new_row_df = new_row_factory() if new_row_factory else None
                if new_row_df is None or new_row_df.empty:
                    new_row_df = _blank_row(template_source)
                if new_row_df is not None and not new_row_df.empty:
                    st.session_state[pending_key] = new_row_df.reset_index(drop=True)
                    st.session_state[dialog_key] = True
                else:
                    st.warning("Unable to create a template for the new row.")
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
                    base = trimmed
    elif allow_row_controls and not edit_enabled:
        st.caption("Enable edit mode to add or remove rows.")

    if edit_enabled:
        template_source = base if not base.empty else data
        updated_df = _handle_add_row_dialog(
            key=key,
            table=st.session_state[key],
            template_source=template_source,
            dialog_key=dialog_key,
            pending_key=pending_key,
        )
        if updated_df is not None:
            base = updated_df

    editor_disabled = not edit_enabled

    edited = st.data_editor(
        base,
        key=f"editor_{key}",
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config or {},
        disabled=editor_disabled,
    )

    if not edit_enabled:
        st.caption(
            "Defaults are read-only. Toggle the **Edit** checkbox to change cell values or manage rows."
        )
        return base

    _update_table_state(key, edited)

    if row_edit_controls:
        st.caption(
            "Update the values directly in the table or use the row controls below for step-by-step "
            "editing. Saved changes immediately refresh the schedules."
        )
        table = st.session_state[key].copy().reset_index(drop=True)
        label_field = row_label_field
        if label_field is None and not table.columns.empty:
            label_field = table.columns[0]

        active_key = f"{key}_active_row"
        active_row = st.session_state.get(active_key)
        if isinstance(active_row, int) and active_row >= len(table):
            st.session_state.pop(active_key, None)
            active_row = None

        if table.empty:
            st.session_state.pop(active_key, None)
            st.info("No rows available. Add a row above to begin editing.")
        else:
            for idx, row in table.iterrows():
                if label_field in table.columns:
                    descriptor = row.get(label_field, "")
                    descriptor = "" if pd.isna(descriptor) else str(descriptor)
                    title = f"Row {idx + 1}: {descriptor}"
                else:
                    title = f"Row {idx + 1}"

                row_cols = st.columns([5, 1])
                row_cols[0].markdown(f"**{title}**")
                is_active = active_row == idx
                edit_label = "Editing" if is_active else "Edit row"
                if row_cols[1].button(
                    edit_label,
                    key=f"{key}_edit_btn_{idx}",
                    disabled=is_active,
                    use_container_width=True,
                ):
                    st.session_state[active_key] = idx

        active_row = st.session_state.get(active_key)
        if isinstance(active_row, int) and 0 <= active_row < len(table):
            row = table.loc[active_row]
            if label_field in table.columns:
                descriptor = row.get(label_field, "")
                descriptor = "" if pd.isna(descriptor) else str(descriptor)
                active_title = f"Row {active_row + 1}: {descriptor}"
            else:
                active_title = f"Row {active_row + 1}"
            st.info(f"Editing {active_title}")
            with st.form(f"{key}_row_form_active"):
                updated_values: Dict[str, Any] = {}
                for col in table.columns:
                    cell_value = row[col]
                    safe_col = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col))
                    widget_key = f"{key}_row_active_{safe_col}"
                    series = table[col] if col in table.columns else None
                    new_val = _input_widget_for_column(
                        column=col,
                        value=cell_value,
                        series=series,
                        key=widget_key,
                    )
                    updated_values[col] = new_val

                action_cols = st.columns(2)
                save_clicked = action_cols[0].form_submit_button(
                    "Save changes", use_container_width=True
                )
                cancel_clicked = action_cols[1].form_submit_button(
                    "Cancel", use_container_width=True
                )

                if save_clicked:
                    for column_name, val in updated_values.items():
                        table.at[active_row, column_name] = val
                    _update_table_state(key, table)
                    st.session_state.pop(active_key, None)
                    st.success("Row updated.")
                elif cancel_clicked:
                    st.session_state.pop(active_key, None)
                    st.info("Row edit cancelled.")

    return st.session_state[key]


def _handle_add_row_dialog(
    *,
    key: str,
    table: pd.DataFrame,
    template_source: pd.DataFrame,
    dialog_key: str,
    pending_key: str,
) -> Optional[pd.DataFrame]:
    """Render the add-row dialog if requested and return the updated table."""

    if not st.session_state.get(dialog_key):
        return None

    pending = st.session_state.get(pending_key)
    if pending is None:
        st.session_state.pop(dialog_key, None)
        return None

    if isinstance(pending, pd.DataFrame):
        pending_df = pending.copy()
    elif isinstance(pending, dict):
        pending_df = pd.DataFrame([pending])
    else:
        pending_df = pd.DataFrame()

    if pending_df.empty:
        st.session_state.pop(dialog_key, None)
        st.session_state.pop(pending_key, None)
        st.warning("No template data available for the new row.")
        return None

    columns: List[str] = list(template_source.columns)
    for col in pending_df.columns:
        if col not in columns:
            columns.append(col)

    if not columns:
        st.session_state.pop(dialog_key, None)
        st.session_state.pop(pending_key, None)
        st.warning("No columns defined for this table; cannot add a new row.")
        return None

    type_table = template_source if not template_source.empty else pending_df

    def _render_form() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        with st.form(f"{key}_add_row_form"):
            inputs: Dict[str, Any] = {}
            for col in columns:
                safe_col = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col))
                widget_key = f"{key}_add_{safe_col}"
                default_val = (
                    pending_df.iloc[0][col]
                    if col in pending_df.columns
                    else None
                )
                series = None
                if col in type_table.columns:
                    series = type_table[col]
                new_val = _input_widget_for_column(
                    column=col,
                    value=default_val,
                    series=series,
                    key=widget_key,
                )
                inputs[col] = new_val

            action_cols = st.columns(2)
            submit = action_cols[0].form_submit_button(
                "Add row", use_container_width=True
            )
            cancel = action_cols[1].form_submit_button(
                "Cancel", use_container_width=True
            )
        if submit:
            return inputs, "submit"
        if cancel:
            return None, "cancel"
        return None, None

    modal_fn = getattr(st, "modal", None)
    if callable(modal_fn):
        with modal_fn("Add new row"):
            st.subheader("Add new row")
            values, action = _render_form()
    else:
        with st.expander("Add new row", expanded=True):
            values, action = _render_form()

    if action == "submit" and values is not None:
        new_row_df = pd.DataFrame([values], columns=columns)
        combined = pd.concat([table, new_row_df], ignore_index=True)
        _update_table_state(key, combined)
        st.session_state.pop(dialog_key, None)
        st.session_state.pop(pending_key, None)
        st.success("Row added.")
        return combined
    if action == "cancel":
        st.session_state.pop(dialog_key, None)
        st.session_state.pop(pending_key, None)
        st.info("Row addition cancelled.")

    return None


def _propagate_yearly_pattern(
    series: pd.Series,
    base_value: float,
    *,
    mode: str,
    rate_pct: float = 0.0,
) -> pd.Series:
    """Return a series of propagated values that respects the original dtype."""

    length = len(series)
    if length == 0:
        return series.copy()

    mode = mode.lower().strip()
    values: np.ndarray
    if mode == "copy":
        values = np.full(length, base_value, dtype=float)
    else:
        rate = rate_pct / 100.0
        if mode == "increase":
            factor = 1.0 + rate
        elif mode == "decrease":
            factor = 1.0 - rate
        else:
            raise ValueError(f"Unsupported propagation mode: {mode}")
        if factor <= 0:
            raise ValueError("Growth factor must be greater than zero.")
        exponent = np.arange(length, dtype=float)
        values = base_value * np.power(factor, exponent)

    propagated = pd.Series(values, index=series.index, dtype=float)

    dtype = series.dtype
    if is_integer_dtype(dtype):
        return propagated.round().astype(dtype)
    if is_numeric_dtype(dtype):
        return propagated.astype(float)
    return propagated


def _render_yearly_increment_helper(
    table_key: str,
    *,
    template: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """Render a yearly increment helper directly beneath a schedule."""

    template_df = template.copy() if template is not None else pd.DataFrame()
    table_df = _ensure_state_df(table_key, template_df)

    label = label or TABLE_LABELS.get(table_key, table_key.replace("_", " ").title())

    with st.expander("Yearly increment", expanded=False):
        st.caption(
            "Propagate updated values or apply compound adjustments across the production horizon."
        )
        st.markdown(
            "The helper overwrites values starting from the first year. Set a **Base value** and use"
            " the buttons to copy it forward or compound an annual percentage change; each subsequent"
            " year applies the chosen growth or reduction to the previous year's amount."
        )

        if table_df.empty:
            st.info("Add rows to the table to enable yearly increments.")
            return table_df

        numeric_cols = [col for col in table_df.columns if is_numeric_dtype(table_df[col])]
        if not numeric_cols:
            st.info("No numeric columns available for yearly increments.")
            return table_df

        edit_flag = st.session_state.get(f"{table_key}_edit_toggle", False)
        if not edit_flag:
            st.info("Enable edit mode for this table to apply increments.")
            return table_df

        column = st.selectbox(
            "Column",
            numeric_cols,
            key=f"{table_key}_increment_column",
        )

        column_series = table_df[column].fillna(0.0)
        base_default = float(column_series.iloc[0]) if not column_series.empty else 0.0

        base_value = st.number_input(
            "Base value",
            value=float(base_default),
            key=f"{table_key}_increment_base",
        )

        col_copy, col_increase, col_decrease = st.columns(3)

        with col_copy:
            if st.button("Copy forward", key=f"{table_key}_copy_forward"):
                updated = table_df.copy()
                updated[column] = _propagate_yearly_pattern(
                    updated[column],
                    base_value,
                    mode="copy",
                )
                _update_table_state(table_key, updated)
                table_df = updated
                st.success(
                    f"Copied the base value across the production horizon for {label} ({column})."
                )

        with col_increase:
            increase_pct = st.number_input(
                "Increase (%)",
                value=0.0,
                step=0.5,
                key=f"{table_key}_increase_pct",
            )
            if st.button("Apply increase", key=f"{table_key}_apply_increase"):
                updated = table_df.copy()
                try:
                    updated[column] = _propagate_yearly_pattern(
                        updated[column],
                        base_value,
                        mode="increase",
                        rate_pct=increase_pct,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    _update_table_state(table_key, updated)
                    table_df = updated
                    st.success(
                        f"Applied a {increase_pct:.2f}% annual increase across the production horizon for {label} ({column})."
                    )

        with col_decrease:
            decrease_pct = st.number_input(
                "Decrease (%)",
                value=0.0,
                step=0.5,
                min_value=0.0,
                max_value=99.5,
                key=f"{table_key}_decrease_pct",
            )
            if st.button("Apply decrease", key=f"{table_key}_apply_decrease"):
                updated = table_df.copy()
                try:
                    updated[column] = _propagate_yearly_pattern(
                        updated[column],
                        base_value,
                        mode="decrease",
                        rate_pct=decrease_pct,
                    )
                except ValueError:
                    st.error("Decrease must be less than 100% to maintain positive values.")
                else:
                    _update_table_state(table_key, updated)
                    table_df = updated
                    st.success(
                        f"Applied a {decrease_pct:.2f}% annual decrease across the production horizon for {label} ({column})."
                    )

    return table_df


def _sync_production_annual_with_projection(projection: ProjectionSettings) -> None:
    """Ensure the annual production schedule mirrors the current projection horizon."""

    table_key = "production_annual"
    template = production_annual_defaults
    current_table = _ensure_state_df(table_key, template)
    if not isinstance(current_table, pd.DataFrame):
        return

    columns = list(current_table.columns)
    if "Year" not in columns:
        return

    if "Throughput (t)" in columns:
        throughput_col: Optional[str] = "Throughput (t)"
    elif len(columns) > 1:
        throughput_col = columns[1]
    else:
        throughput_col = None

    if throughput_col is None:
        return

    target_years = list(range(projection.start_year, projection.end_year + 1))
    if not target_years:
        return

    ordered = current_table.sort_values(
        by="Year", kind="stable", na_position="last"
    ).reset_index(drop=True)

    ordered_values = ordered[throughput_col].tolist()
    base_throughput = float(st.session_state.get("msw_tonnes_pa", 0.0))
    if template is not None and not template.empty:
        template_value = template[throughput_col].iloc[0]
        if pd.notna(template_value):
            base_throughput = float(template_value)

    cleaned_values: List[float] = []
    if not ordered_values:
        cleaned_values = [base_throughput]
    else:
        fallback = base_throughput
        for value in ordered_values:
            if pd.isna(value):
                if cleaned_values:
                    cleaned_values.append(cleaned_values[-1])
                else:
                    cleaned_values.append(fallback)
            else:
                try:
                    cleaned_values.append(float(value))
                except (TypeError, ValueError):
                    cleaned_values.append(fallback if not cleaned_values else cleaned_values[-1])
        if all(pd.isna(val) for val in ordered_values):
            cleaned_values = [fallback]

    if not cleaned_values:
        cleaned_values = [base_throughput]

    if len(cleaned_values) < len(target_years):
        cleaned_values.extend([cleaned_values[-1]] * (len(target_years) - len(cleaned_values)))
    elif len(cleaned_values) > len(target_years):
        cleaned_values = cleaned_values[: len(target_years)]

    existing_by_year: Dict[int, float] = {}
    for idx, row in ordered.iterrows():
        year_raw = row.get("Year")
        if pd.isna(year_raw):
            continue
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        if idx < len(cleaned_values):
            existing_by_year[year] = cleaned_values[idx]

    new_values: List[float] = []
    for idx, year in enumerate(target_years):
        value = existing_by_year.get(year)
        if value is None:
            if idx < len(cleaned_values):
                value = cleaned_values[idx]
            elif new_values:
                value = new_values[-1]
            else:
                value = cleaned_values[-1]
        new_values.append(value)

    new_df = pd.DataFrame({"Year": target_years, throughput_col: new_values})

    year_dtype = current_table["Year"].dtype
    try:
        new_df["Year"] = new_df["Year"].astype(year_dtype)
    except (TypeError, ValueError):
        new_df["Year"] = new_df["Year"].astype(int)

    throughput_dtype = current_table[throughput_col].dtype
    throughput_series = pd.Series(new_values)
    if is_integer_dtype(throughput_dtype):
        new_df[throughput_col] = throughput_series.round().astype(throughput_dtype)
    elif is_numeric_dtype(throughput_dtype):
        try:
            new_df[throughput_col] = throughput_series.astype(throughput_dtype)
        except (TypeError, ValueError):
            new_df[throughput_col] = throughput_series.astype(float)
    else:
        new_df[throughput_col] = throughput_series.astype(float)

    for column in columns:
        if column not in new_df.columns:
            new_df[column] = current_table[column]
    new_df = new_df[columns]

    try:
        new_df = new_df.astype(current_table.dtypes.to_dict())
    except (TypeError, ValueError):
        pass

    if current_table.reset_index(drop=True).equals(new_df.reset_index(drop=True)):
        return

    _update_table_state(table_key, new_df)


def _reset_scalar_values(values: Dict[str, Any]) -> None:
    for key, value in values.items():
        st.session_state[key] = value


def _derive_production_schedule(
    projection: ProjectionSettings,
) -> pd.Series:
    """Return the user-configured annual throughput schedule."""

    table = st.session_state.get("production_annual")
    if not isinstance(table, pd.DataFrame) or table.empty:
        return pd.Series(dtype=float)

    if "Year" not in table.columns:
        return pd.Series(dtype=float)

    if "Throughput (t)" in table.columns:
        throughput_col: Optional[str] = "Throughput (t)"
    else:
        throughput_col = next(
            (col for col in table.columns if col != "Year" and is_numeric_dtype(table[col])),
            None,
        )

    if throughput_col is None:
        return pd.Series(dtype=float)

    target_years = list(range(projection.start_year, projection.end_year + 1))
    if not target_years:
        return pd.Series(dtype=float)

    ordered = table.sort_values(by="Year", kind="stable", na_position="last").reset_index(drop=True)

    default_value = float(st.session_state.get("msw_tonnes_pa", 0.0))
    if "Throughput (t)" in production_annual_defaults.columns and not production_annual_defaults.empty:
        template_value = production_annual_defaults["Throughput (t)"].iloc[0]
        if pd.notna(template_value):
            default_value = float(template_value)

    values_by_year: Dict[int, float] = {}
    for _, row in ordered.iterrows():
        year_raw = row.get("Year")
        if pd.isna(year_raw):
            continue
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            continue
        raw_value = row.get(throughput_col)
        if pd.isna(raw_value):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        values_by_year[year] = value

    if not values_by_year:
        return pd.Series(dtype=float)

    annual_values: List[float] = []
    last_value = default_value
    for year in target_years:
        value = values_by_year.get(year)
        if value is None:
            value = last_value
        else:
            last_value = value
        annual_values.append(value)

    if not annual_values:
        return pd.Series(dtype=float)

    schedule_index = pd.Index(target_years, name="Year")
    return pd.Series(annual_values, index=schedule_index, dtype=float)


def _resolve_production_throughput_profile(
    projection: ProjectionSettings,
) -> Optional[List[float]]:
    """Return a per-period throughput profile derived from the annual schedule."""

    schedule = _derive_production_schedule(projection)
    if schedule.empty:
        return None

    ppy = max(1, projection.periods_per_year)
    return list(schedule.values / ppy)


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

    results_payload["ai_settings"] = copy.deepcopy(
        st.session_state.get("ai_settings", DEFAULT_AI_SETTINGS)
    )

    payloads[scenario_name] = (inputs_copy, results_payload)
    st.session_state["scenario_payloads"] = payloads
    return inputs_copy, results_payload


def _set_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


source_label = "Default inputs"
inputs = default_inputs()
st.caption(f"Using assumptions from: {source_label}")

ai_payload = st.session_state.setdefault("ai_payload", {})
ai_payload.setdefault("ai_settings", dict(DEFAULT_AI_SETTINGS))

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
    [
        {
            "Year": inputs.timeline.start_year + i,
            "Throughput (t)": inputs.tech.msw_tonnes_pa,
        }
        for i in range(inputs.timeline.years)
    ]
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
            2. **Edit or extend rows** – once edit mode is active you can type directly into the
               table to update values or use the *Add row*/*Remove row* buttons to adjust the
               schedule length. The optional **Edit row** buttons let you work line by line if you
               prefer guided forms.
            3. **Manage default sets** – restore the shipped defaults, start with empty tables,
               or save/load your own presets from the *Manage defaults & state* panel.
            4. **Apply structured growth** – each schedule includes a *Yearly increment*
               expander directly beneath the table; open it while in edit mode to copy values
               forward or apply compound increases/decreases across the production horizon.
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
        if col_md2.button("Clean start", key="clean_start"):
            _reset_scalar_values(SCALAR_CLEAN_START)
            _reset_table_group(TABLE_DEFAULTS, mode="clean")
            st.success("Workspace cleared.")
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

    st.subheader("AI & ML Configuration")
    _render_ai_settings(ai_payload)
    st.markdown(
        """
        Configure machine-learning forecasts and generative summaries for the workbook and dashboards.
        Provide your preferred provider, model name, optional API credentials, and choose the analytics
        features to activate. Settings are stored per session and shared across scenarios and exports.
        """
    )

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

    _sync_production_annual_with_projection(
        ProjectionSettings(
            start_year=int(st.session_state["projection_start_year"]),
            end_year=int(st.session_state["projection_end_year"]),
            periods_per_year=int(st.session_state["projection_ppy"]),
        )
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
        row_edit_controls=True,
        row_label_field="Parameter",
    )
    global_inputs = _render_yearly_increment_helper(
        "global_inputs",
        template=global_defaults,
        label="Global inputs",
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
        row_edit_controls=True,
        row_label_field="Item",
    )
    initial_investment = _render_yearly_increment_helper(
        "initial_investment",
        template=initial_investment_defaults,
        label="Initial investment",
    )

    schedule = _compute_initial_investment_schedule(initial_investment)
    total_investment = schedule["Cost"].sum() if not schedule.empty else 0.0
    st.metric("Total investment", f"${total_investment:,.0f}")
    st.markdown("**Depreciation schedule snapshot**")
    if schedule.empty:
        st.info("Add rows to the investment table to generate depreciation schedules.")
    else:
        st.dataframe(schedule.round(2), use_container_width=True)


snapshot_placeholder = None

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
        row_edit_controls=True,
        row_label_field="Revenue stream",
    )
    revenue_table = _render_yearly_increment_helper(
        "revenue_inputs",
        template=revenue_defaults,
        label="Revenue inputs",
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
            row_edit_controls=True,
            row_label_field="Year",
        )
        production_annual = _render_yearly_increment_helper(
            "production_annual",
            template=production_annual_defaults,
            label="Annual production",
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
            row_edit_controls=True,
            row_label_field="Month",
        )
        production_monthly = _render_yearly_increment_helper(
            "production_monthly",
            template=production_monthly_defaults,
            label="Monthly production",
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

    st.subheader("Model Outputs Snapshot")
    snapshot_placeholder = st.empty()


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
        row_edit_controls=True,
        row_label_field="Month",
    )
    direct_costs_monthly = _render_yearly_increment_helper(
        "direct_costs_monthly",
        template=direct_costs_monthly_defaults,
        label="Direct costs monthly",
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
        row_edit_controls=True,
        row_label_field="Role",
    )
    staff_monthly = _render_yearly_increment_helper(
        "staff_monthly",
        template=staff_monthly_defaults,
        label="Staff costs monthly",
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
        row_edit_controls=True,
        row_label_field="Category",
    )
    other_opex_monthly = _render_yearly_increment_helper(
        "other_opex_monthly",
        template=other_opex_monthly_defaults,
        label="Other opex monthly",
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
            row_edit_controls=True,
            row_label_field="Metric",
        )
        accounts_receivable = _render_yearly_increment_helper(
            "accounts_receivable",
            template=accounts_receivable_defaults,
            label="Accounts receivable",
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
            row_edit_controls=True,
            row_label_field="Metric",
        )
        inventory_payable = _render_yearly_increment_helper(
            "inventory_payable",
            template=inventory_payable_defaults,
            label="Inventory & payables",
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
        row_edit_controls=True,
        row_label_field="Facility",
    )
    loan_schedule = _render_yearly_increment_helper(
        "loan_schedule",
        template=loan_schedule_defaults,
        label="Loan schedule",
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
        row_edit_controls=True,
        row_label_field="Tax",
    )
    tax_schedule = _render_yearly_increment_helper(
        "tax_schedule",
        template=tax_schedule_defaults,
        label="Tax schedule",
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
        row_edit_controls=True,
        row_label_field="Category",
    )
    inflation_schedule = _render_yearly_increment_helper(
        "inflation_schedule",
        template=inflation_schedule_defaults,
        label="Inflation schedule",
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
        row_edit_controls=True,
        row_label_field="Risk",
    )
    risk_schedule = _render_yearly_increment_helper(
        "risk_schedule",
        template=risk_schedule_defaults,
        label="Risk schedule",
    )

    risk_summary = _risk_summary_from_table(risk_schedule)
    if not risk_summary.empty:
        st.markdown("**Risk exposure summary**")
        st.dataframe(risk_summary.round(2), use_container_width=True)
        expected_loss = risk_summary["Expected loss (USD)"].sum()
        st.metric("Total expected loss", f"${expected_loss:,.0f}")
    else:
        st.info("Populate the risk table to calculate expected losses.")


projection = ProjectionSettings(
    start_year=int(st.session_state["projection_start_year"]),
    end_year=int(st.session_state["projection_end_year"]),
    periods_per_year=int(st.session_state["projection_ppy"]),
)

timeline_cfg = Timeline(
    years=projection.years,
    build_months=inputs.timeline.build_months,
    start_year=projection.start_year,
    periods_per_year=projection.periods_per_year,
)

capex_profile = _parse_capex_profile(
    st.session_state["capex_profile_text"], inputs.costs.capex_spend_profile
)

capex_items: List[CapexItem] = []
if _table_differs(initial_investment, initial_investment_defaults):
    capex_items = _capex_items_from_table(initial_investment, inputs.costs.capex_items)

capex_total_value = float(st.session_state["capex_total"])
if capex_items:
    capex_total_value = sum(item.amount for item in capex_items)

revenue_table_changed = _table_differs(revenue_table, revenue_defaults)
if revenue_table_changed:
    revenue_inputs = _revenue_assumptions_from_table(revenue_table, inputs.revenue)
else:
    revenue_inputs = RevenueAssumptions()
    revenue_inputs.ppa_price_usd_per_mwh = float(st.session_state["ppa_price"])
    revenue_inputs.ppa_escalation = float(st.session_state["ppa_escalation"])
    revenue_inputs.gate_fee_usd_per_t = float(st.session_state["gate_fee"])
    revenue_inputs.gate_fee_escalation = float(st.session_state["gate_fee_escalation"])
    revenue_inputs.heat_price_usd_per_mwh = float(st.session_state["heat_price"])
    revenue_inputs.metal_recovery_usd_per_t = float(st.session_state["metal_recovery"])
    revenue_inputs.ash_revenue_usd_per_t = float(st.session_state["ash_revenue"])
    revenue_inputs.other_escalation = float(st.session_state["other_escalation"])

opex_tables_changed = any(
    _table_differs(df, default)
    for df, default in (
        (direct_costs_monthly, direct_costs_monthly_defaults),
        (staff_monthly, staff_monthly_defaults),
        (other_opex_monthly, other_opex_monthly_defaults),
    )
)
opex_components: List[OpexComponent] = []
if opex_tables_changed:
    opex_components = _assemble_opex_components(
        direct_costs_monthly,
        staff_monthly,
        other_opex_monthly,
        timeline_cfg,
    )

loan_table_changed = _table_differs(loan_schedule, loan_schedule_defaults)
debt_facilities: List[DebtFacility] = []
if loan_table_changed:
    debt_facilities = _debt_facilities_from_table(loan_schedule, inputs.finance.debt_facilities)

tax_table_changed = _table_differs(tax_schedule, tax_schedule_defaults)

wc_cfg = _working_capital_from_tables(
    accounts_receivable,
    inventory_payable,
    inputs.finance.working_capital,
)

finance_inputs = FinanceAssumptions(
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
    working_capital=wc_cfg,
    debt_facilities=debt_facilities,
)

finance_inputs.macro_indices = _macro_indices_from_table(
    inflation_schedule, timeline_cfg
)

if tax_table_changed:
    finance_inputs.tax = _tax_assumptions_from_table(
        tax_schedule, finance_inputs.tax, timeline_cfg
    )

user_inputs = WTEMasterInputs(
    timeline=timeline_cfg,
    tech=TechAssumptions(
        msw_tonnes_pa=float(st.session_state["msw_tonnes_pa"]),
        lhv_mj_per_kg=float(st.session_state["lhv_mj_per_kg"]),
        boiler_efficiency=float(st.session_state["boiler_efficiency"]),
        electrical_efficiency=float(st.session_state["electrical_efficiency"]),
        availability=float(st.session_state["availability"]),
        parasitic_load_frac=float(st.session_state["parasitic_load"]),
        tonnes_profile=_resolve_production_throughput_profile(projection),
    ),
    revenue=revenue_inputs,
    costs=CostAssumptions(
        capex_total_usd=capex_total_value,
        capex_spend_profile=capex_profile,
        fixed_om_usd_pa=float(st.session_state["fixed_om"]),
        variable_om_usd_per_t=float(st.session_state["variable_om"]),
        landfill_disposal_usd_per_t=float(st.session_state["landfill_disposal"]),
        insurance_pct_of_capex_pa=float(st.session_state["insurance_pct"]),
        maintenance_pct_of_capex_pa=float(st.session_state["maintenance_pct"]),
        opex_escalation=float(st.session_state["opex_escalation"]),
        capex_items=capex_items,
        opex_components=opex_components,
    ),
    finance=finance_inputs,
)

results = cashflow_model(user_inputs)
results["ai_settings"] = copy.deepcopy(st.session_state.get("ai_settings", DEFAULT_AI_SETTINGS))

summary, summary_ann, summary_cumulative, production_annual_series = build_summary_tables(
    user_inputs, results
)
production_schedule_series = _derive_production_schedule(projection)
if not production_schedule_series.empty:
    summary_ann = summary_ann.merge(
        production_schedule_series.rename("Scheduled throughput (t)"),
        left_on="Calendar Year",
        right_index=True,
        how="left",
    )

if snapshot_placeholder is not None:
    snapshot_placeholder.dataframe(summary.head(12).round(2), use_container_width=True)

energy = results["energy"]
revenue = results["rev"]
capex_total = results["capex"]["total"]
investment_total = results["capex"].get("investment_total", capex_total)
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

project_cashflows = -investment_total + (revenue["total_revenue"] - opex_total - tax_cash)
project_npv = _npv(user_inputs.finance.discount_rate, project_cashflows)

annual_revenue = _annualise(revenue["total_revenue"], user_inputs.timeline.periods_per_year)
annual_ebitda = _annualise(results["ebitda"], user_inputs.timeline.periods_per_year)
annual_equity_cf = _annualise(results["equity_cf"], user_inputs.timeline.periods_per_year)
annual_investment = _annualise(investment_total, user_inputs.timeline.periods_per_year)



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
    modelled_series = production_annual_series.copy()
    if not modelled_series.empty:
        modelled_series.index = modelled_series.index + projection.start_year
        modelled_series.index.name = "Year"

    series_parts: List[pd.Series] = []
    if not production_schedule_series.empty:
        series_parts.append(production_schedule_series.rename("Scheduled throughput (t)"))
    if not modelled_series.empty:
        series_parts.append(modelled_series.rename("Modelled throughput (t)"))

    chart_df = pd.concat(series_parts, axis=1).dropna(how="all") if series_parts else pd.DataFrame()

    if not chart_df.empty:
        st.line_chart(chart_df)
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
                float(investment_total.sum()),
                float(results["opex"]["total_opex"].sum()),
                float(debt["debt_service"].sum()),
            ],
        }
    )
    st.bar_chart(cost_breakdown.set_index("Category"))

    st.subheader("Capital Expenditure and Debt")
    capex_debt = pd.DataFrame(
        {
            "Investment": _annualise(investment_total, user_inputs.timeline.periods_per_year),
            "Debt funding": _annualise(debt["funding_total"], user_inputs.timeline.periods_per_year),
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
    equity_balance = np.cumsum(-investment_total + debt["funding_total"] + results["equity_cf"])
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
            "Investing cash flow": -investment_total,
            "Financing cash flow": debt["funding_total"] - debt["debt_service"] + equity_cf,
            "Net cash flow": summary["CFADS"] - investment_total + debt["funding_total"] - debt["debt_service"] + equity_cf,
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
        row_edit_controls=True,
        row_label_field="Driver",
    )
    sensitivity_config = _render_yearly_increment_helper(
        "sensitivity_config",
        template=sensitivity_config_defaults,
        label="Sensitivity configuration",
    )

    st.subheader("Simulation Results")
    sensitivity_results_df, sensitivity_warnings = _sensitivity_results_from_table(
        sensitivity_config, user_inputs
    )
    for msg in sensitivity_warnings:
        st.warning(msg)
    if sensitivity_results_df.empty:
        st.info("Add drivers to the sensitivity table to evaluate alternative cases.")
    else:
        st.dataframe(sensitivity_results_df.round(4), use_container_width=True)
        pivot = sensitivity_results_df.pivot_table(
            index="Driver",
            columns="Case",
            values="Equity IRR",
        )
        if not pivot.empty:
            st.line_chart(pivot)

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
        row_edit_controls=True,
        row_label_field="Variable",
    )
    monte_carlo_config = _render_yearly_increment_helper(
        "monte_carlo_config",
        template=monte_carlo_defaults_runtime,
        label="Monte Carlo configuration",
    )
    mc_config, mc_warnings = _monte_carlo_config_from_table(monte_carlo_config, user_inputs)
    for msg in mc_warnings:
        st.warning(msg)
    if mc_config is None:
        st.info("Provide at least one valid distribution to run the Monte Carlo simulation.")
    else:
        try:
            mc_results = run_monte_carlo(user_inputs, mc_config)
        except Exception as exc:
            st.error(f"Monte Carlo simulation failed: {exc}")
        else:
            summary_df = pd.DataFrame(mc_results["summary"]).T
            summary_df.index.name = "Metric"
            st.markdown("**Monte Carlo summary (mean / percentiles)**")
            st.dataframe(summary_df.round(4), use_container_width=True)
            records_df = pd.DataFrame(mc_results.get("records", []))
            if not records_df.empty:
                st.markdown("**Sampled outcomes**")
                st.dataframe(records_df.head(20).round(4), use_container_width=True)


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
        row_edit_controls=True,
        row_label_field="Target metric",
    )
    goal_seek = _render_yearly_increment_helper(
        "goal_seek",
        template=goal_seek_defaults,
        label="Goal seek configuration",
    )

    st.subheader("Goal Seek Results")
    goal_seek_results_df, goal_seek_warnings = _goal_seek_results_from_table(
        goal_seek, user_inputs
    )
    for msg in goal_seek_warnings:
        st.warning(msg)
    if goal_seek_results_df.empty:
        st.info("Add goal seek configurations to calculate required adjustments.")
    else:
        st.dataframe(goal_seek_results_df.round(4), use_container_width=True)

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
        row_edit_controls=True,
        row_label_field="Scenario",
    )
    scenario_config = _render_yearly_increment_helper(
        "scenario_config",
        template=scenario_config_defaults,
        label="Scenario configuration",
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
                    excel_bytes = generate_excel_bytes(
                        model, scenario_payload_results, selected_scenario
                    )
                excel_map[selected_scenario] = excel_bytes
                st.session_state.excel_bytes_map = excel_map
        if excel_bytes:
            st.download_button(
                "Download Excel Model",
                data=excel_bytes,
                file_name="Waste_to_Energy_Financial_Model.xlsx",
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
        row_edit_controls=True,
        row_label_field="Input",
    )
    break_even_inputs = _render_yearly_increment_helper(
        "break_even_inputs",
        template=break_even_defaults,
        label="Break-even inputs",
    )

    st.subheader("Break-Even Results Background")
    break_even_revenue = annual_revenue.sum()
    break_even_costs = annual_investment.sum() + annual_equity_cf.abs().sum()
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
        row_edit_controls=True,
        row_label_field="Parameter",
    )
    parameter_naming = _render_yearly_increment_helper(
        "parameter_naming",
        template=parameter_naming_defaults,
        label="Parameter naming",
    )
    st.dataframe(parameter_naming, use_container_width=True)

    st.subheader("Additional Context")
    st.write(
        "Background Information includes CAPEX requirements and feedstock demand assumptions. "
        "Use the input table above to refine the data that underpins break-even and payback outputs."
    )

st.info(
    "All navigation is organised horizontally across the page. Use the tabs to explore inputs, "
    "results, sensitivities, and scenario tools without relying on a sidebar."
)

