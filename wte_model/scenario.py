"""Scenario and risk analytics utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import copy

import numpy as np

from .dcf import cashflow_model
from .inputs import WTEMasterInputs


@dataclass
class DistributionSpec:
    """Specification for a stochastic input used in Monte Carlo simulation."""

    path: str
    dist: str
    params: Dict[str, float]
    minimum: Optional[float] = None
    maximum: Optional[float] = None


@dataclass
class MonteCarloConfig:
    """Configuration for the Monte Carlo simulator."""

    iterations: int = 500
    seed: Optional[int] = None
    distributions: List[DistributionSpec] = field(default_factory=list)
    metrics: Sequence[str] = ("irr_eq", "irr_proj", "dscr_min")


def _resolve_path(root: WTEMasterInputs, path: str):
    """Return the parent object and attribute name for a dotted path."""

    current = root
    tokens = path.split(".")
    for token in tokens[:-1]:
        if "[" in token:
            attr, index = token[:-1].split("[")
            current = getattr(current, attr)[int(index)]
        else:
            current = getattr(current, token)
    final = tokens[-1]
    index = None
    if "[" in final:
        attr, idx = final[:-1].split("[")
        index = int(idx)
        final = attr
    return current, final, index


def _assign(root: WTEMasterInputs, path: str, value: float) -> None:
    parent, attr, index = _resolve_path(root, path)
    if index is None:
        setattr(parent, attr, value)
    else:
        seq = getattr(parent, attr)
        seq[index] = value


def _read(root: WTEMasterInputs, path: str) -> float:
    parent, attr, index = _resolve_path(root, path)
    if index is None:
        return getattr(parent, attr)
    seq = getattr(parent, attr)
    return seq[index]


def _sample(spec: DistributionSpec, rng: np.random.Generator) -> float:
    dist = spec.dist.lower()
    if dist == "normal":
        value = rng.normal(spec.params.get("mean", 0.0), spec.params.get("std", 1.0))
    elif dist == "lognormal":
        value = rng.lognormal(spec.params.get("mean", 0.0), spec.params.get("sigma", 1.0))
    elif dist == "uniform":
        value = rng.uniform(spec.params.get("low", 0.0), spec.params.get("high", 1.0))
    elif dist == "triangular":
        left = spec.params.get("left", 0.0)
        mode = spec.params.get("mode", left)
        right = spec.params.get("right", mode)
        value = rng.triangular(left, mode, right)
    else:
        raise ValueError(f"Unsupported distribution '{spec.dist}'.")

    if spec.minimum is not None:
        value = max(value, spec.minimum)
    if spec.maximum is not None:
        value = min(value, spec.maximum)
    return float(value)


def run_sensitivity(
    base_inputs: WTEMasterInputs,
    path: str,
    values: Iterable[float],
    metrics: Sequence[str] = ("irr_eq", "irr_proj", "dscr_min"),
) -> List[Dict[str, float]]:
    """Evaluate the model across a series of values for a single parameter."""

    results: List[Dict[str, float]] = []
    for value in values:
        scenario_inputs = copy.deepcopy(base_inputs)
        _assign(scenario_inputs, path, value)
        res = cashflow_model(scenario_inputs)
        record: Dict[str, float] = {"value": float(value)}
        for metric in metrics:
            if metric == "dscr_min":
                record[metric] = float(np.nanmin(res["dscr"]))
            else:
                record[metric] = float(res.get(metric, np.nan))
        results.append(record)
    return results


def run_monte_carlo(base_inputs: WTEMasterInputs, config: MonteCarloConfig) -> Dict[str, object]:
    """Run a Monte Carlo simulation and return the sampled metrics."""

    rng = np.random.default_rng(config.seed)
    metric_samples: Dict[str, List[float]] = {metric: [] for metric in config.metrics}
    records: List[Dict[str, float]] = []

    for _ in range(config.iterations):
        scenario_inputs = copy.deepcopy(base_inputs)
        for spec in config.distributions:
            draw = _sample(spec, rng)
            _assign(scenario_inputs, spec.path, draw)
        res = cashflow_model(scenario_inputs)
        record: Dict[str, float] = {}
        for metric in config.metrics:
            if metric == "dscr_min":
                value = float(np.nanmin(res["dscr"]))
            else:
                raw = res.get(metric)
                value = float(raw if not isinstance(raw, np.ndarray) else raw[-1])
            metric_samples[metric].append(value)
            record[metric] = value
        records.append(record)

    summary: Dict[str, Dict[str, float]] = {}
    for metric, samples in metric_samples.items():
        arr = np.array(samples)
        summary[metric] = {
            "mean": float(np.mean(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }

    return {"records": records, "summary": summary, "iterations": config.iterations}


def goal_seek(
    base_inputs: WTEMasterInputs,
    path: str,
    target_metric: str,
    target_value: float,
    bracket: Sequence[float],
    *,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> Dict[str, float]:
    """Binary-search goal seek for a scalar metric."""

    lo, hi = bracket
    if lo > hi:
        lo, hi = hi, lo

    def evaluate(value: float) -> float:
        scenario_inputs = copy.deepcopy(base_inputs)
        _assign(scenario_inputs, path, value)
        res = cashflow_model(scenario_inputs)
        if target_metric == "dscr_min":
            return float(np.nanmin(res["dscr"]))
        metric_value = res.get(target_metric)
        if isinstance(metric_value, np.ndarray):
            return float(metric_value[-1])
        return float(metric_value)

    f_lo = evaluate(lo) - target_value
    f_hi = evaluate(hi) - target_value
    if f_lo == 0:
        return {"value": lo, target_metric: target_value}
    if f_hi == 0:
        return {"value": hi, target_metric: target_value}
    if f_lo * f_hi > 0:
        raise ValueError("Goal seek bracket does not straddle the target value.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = evaluate(mid) - target_value
        if abs(f_mid) <= tol:
            return {"value": mid, target_metric: target_value}
        if f_mid * f_lo < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return {"value": 0.5 * (lo + hi), target_metric: target_value}


def apply_scenarios(
    base_inputs: WTEMasterInputs,
    scenarios: Dict[str, Dict[str, float]],
    metrics: Sequence[str] = ("irr_eq", "irr_proj", "dscr_min"),
) -> Dict[str, Dict[str, float]]:
    """Evaluate named scenarios with parameter overrides."""

    results: Dict[str, Dict[str, float]] = {}
    for name, overrides in scenarios.items():
        scenario_inputs = copy.deepcopy(base_inputs)
        for path, value in overrides.items():
            _assign(scenario_inputs, path, value)
        res = cashflow_model(scenario_inputs)
        record: Dict[str, float] = {}
        for metric in metrics:
            if metric == "dscr_min":
                record[metric] = float(np.nanmin(res["dscr"]))
            else:
                raw = res.get(metric)
                record[metric] = float(raw if not isinstance(raw, np.ndarray) else raw[-1])
        results[name] = record
    return results

