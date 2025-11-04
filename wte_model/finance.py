"""Debt, depreciation, tax, and working-capital calculations."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inputs import CapexItem, DebtFacility, FinanceAssumptions, WTEMasterInputs
from .utils import discount_factors, expand_series, normalise_profile


def _pmt(rate: float, nper: int, pv: float) -> float:
    """Annuity payment helper."""

    if nper <= 0 or pv <= 0:
        return 0.0
    if rate == 0:
        return pv / nper
    return (rate * pv) / (1 - (1 + rate) ** (-nper))


def _facility_draws(
    facility: DebtFacility,
    capex_total: np.ndarray,
    timeline,
    total_project_capex: float,
) -> np.ndarray:
    """Determine the draw profile for a facility."""

    periods = timeline.n
    construction_periods = timeline.construction_periods
    draws = np.zeros(periods)

    ratio = facility.draw_ratio
    commitment = facility.commitment
    if commitment is None and ratio > 0 and total_project_capex > 0:
        commitment = ratio * total_project_capex
    elif commitment is None:
        commitment = 0.0

    if commitment == 0.0:
        return draws

    if facility.draw_profile is not None:
        profile = expand_series(
            facility.draw_profile,
            construction_periods,
            timeline.periods_per_year,
            label=f"{facility.name}_draw_profile",
            allow_partial=True,
        )[:construction_periods]
        profile = normalise_profile(profile, label=f"{facility.name} draw profile")
        draws[:construction_periods] = profile * commitment
        return draws

    if ratio > 0:
        draws = capex_total * ratio
    else:
        profile = capex_total.copy()
        if profile.sum() <= 0:
            return draws
        draws = commitment * profile / profile.sum()
    return draws


def _apply_loss_queue(
    taxable: float,
    queue: List[Tuple[float, int]],
) -> Tuple[float, List[Tuple[float, int]]]:
    """Offset taxable income with the oldest available loss carry-forwards."""

    if taxable <= 0 or not queue:
        return taxable, queue

    new_queue: List[Tuple[float, int]] = []
    remaining = taxable
    for amount, life in queue:
        if remaining <= 0:
            new_queue.append((amount, life))
            continue
        offset = min(amount, remaining)
        remaining -= offset
        amount -= offset
        if amount > 1e-9 and life > 0:
            new_queue.append((amount, life))
    return remaining, new_queue


def _schedule_facility(
    facility: DebtFacility,
    draws: np.ndarray,
    timeline,
    cfads: Optional[np.ndarray],
) -> Dict[str, np.ndarray | float | bool]:
    """Build the detailed schedule for a single debt facility."""

    periods = timeline.n
    ppy = timeline.periods_per_year
    rate = facility.interest_rate / ppy

    balance = np.zeros(periods)
    interest_cash = np.zeros(periods)
    interest_capitalised = np.zeros(periods)
    principal = np.zeros(periods)
    debt_service = np.zeros(periods)
    fees = np.zeros(periods)

    if draws.sum() > 0:
        first_draw = int(np.argmax(draws > 0))
        fees[first_draw] = facility.upfront_fee_pct * draws.sum()

    amort_start = timeline.construction_periods + facility.grace_years * ppy
    amort_periods = max(0, min(facility.tenor_years * ppy, periods - amort_start))
    needs_cfads = facility.sculpt_from_cfads or facility.cash_sweep_pct > 0

    annuity_payment = 0.0
    straight_line_principal = 0.0
    custom_principal = np.zeros(amort_periods)
    balance_at_amort_start = 0.0

    for idx in range(periods):
        opening = balance[idx - 1] if idx > 0 else 0.0
        balance[idx] = opening + draws[idx]
        interest_amount = balance[idx] * rate

        capitalise = (
            facility.interest_during_construction
            and idx < timeline.construction_periods
            and idx < amort_start
        )
        if capitalise:
            interest_capitalised[idx] = interest_amount
            balance[idx] += interest_amount
        else:
            interest_cash[idx] = interest_amount

        if idx == amort_start and amort_periods > 0:
            balance_at_amort_start = balance[idx]
            if facility.amortization == "straight_line":
                straight_line_principal = balance_at_amort_start / amort_periods
            elif facility.amortization == "custom" and facility.custom_amort_profile is not None:
                profile = expand_series(
                    facility.custom_amort_profile,
                    amort_periods,
                    ppy,
                    label=f"{facility.name}_amort_profile",
                    allow_partial=True,
                )[:amort_periods]
                profile = normalise_profile(profile, label=f"{facility.name} amort profile")
                custom_principal = profile * balance_at_amort_start
            elif facility.amortization in {"sculpted", "annuity"} or facility.sculpt_from_cfads:
                annuity_payment = _pmt(rate, amort_periods, balance_at_amort_start)

        if idx >= amort_start and amort_periods > 0:
            period = idx - amort_start
            remaining_balance = balance[idx]
            scheduled_principal = 0.0
            scheduled_service = 0.0

            if facility.amortization == "straight_line":
                scheduled_principal = straight_line_principal
                scheduled_service = scheduled_principal + interest_cash[idx]
            elif facility.amortization == "custom" and custom_principal.size:
                scheduled_principal = custom_principal[min(period, custom_principal.size - 1)]
                scheduled_service = scheduled_principal + interest_cash[idx]
            elif (facility.amortization == "sculpted" or facility.sculpt_from_cfads) and cfads is not None:
                target_service = max(cfads[idx] / max(facility.target_dscr, 1e-6), 0.0)
                scheduled_service = min(target_service, remaining_balance + interest_cash[idx])
                scheduled_principal = max(scheduled_service - interest_cash[idx], 0.0)
            else:
                scheduled_service = annuity_payment
                scheduled_principal = max(scheduled_service - interest_cash[idx], 0.0)

            if facility.cash_sweep_pct and cfads is not None:
                excess = max(cfads[idx] - scheduled_service, 0.0) * facility.cash_sweep_pct
                sweep = min(excess, max(remaining_balance - scheduled_principal, 0.0))
                scheduled_principal += sweep
                scheduled_service += sweep

            scheduled_principal = min(scheduled_principal, remaining_balance)
            principal[idx] = scheduled_principal
            balance[idx] = remaining_balance - scheduled_principal
            debt_service[idx] = scheduled_principal + interest_cash[idx]

    return {
        "name": facility.name,
        "draws": draws,
        "balance": balance,
        "interest_cash": interest_cash,
        "interest_capitalised": interest_capitalised,
        "principal": principal,
        "debt_service": debt_service,
        "fees": fees,
        "needs_cfads": needs_cfads,
        "amort_start": amort_start,
        "amort_periods": amort_periods,
        "rate": rate,
        "balance_at_amort_start": balance_at_amort_start,
    }


def debt_schedule(
    inp: WTEMasterInputs,
    capex_total: np.ndarray,
    cfads: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | List[Dict[str, np.ndarray | float | bool]] | bool]:
    """Generate debt drawdown, service, and balance arrays for all facilities."""

    timeline = inp.timeline
    finance = inp.finance
    total_capex = float(capex_total.sum())

    facilities = finance.debt_facilities
    if not facilities:
        facilities = [
            DebtFacility(
                name="Senior debt",
                draw_ratio=finance.debt_ratio,
                interest_rate=finance.interest_rate,
                tenor_years=finance.tenor_years,
                grace_years=finance.grace_years,
                amortization="annuity",
                upfront_fee_pct=finance.upfront_fee_pct,
                interest_during_construction=True,
                target_dscr=finance.dscr_min,
            )
        ]

    facility_results: List[Dict[str, np.ndarray | float | bool]] = []
    total_draws = np.zeros(timeline.n)
    total_interest = np.zeros(timeline.n)
    total_interest_cap = np.zeros(timeline.n)
    total_principal = np.zeros(timeline.n)
    total_debt_service = np.zeros(timeline.n)
    total_balance = np.zeros(timeline.n)
    total_fees = np.zeros(timeline.n)
    needs_cfads = False

    for facility in facilities:
        draws = _facility_draws(facility, capex_total, timeline, total_capex)
        schedule = _schedule_facility(facility, draws, timeline, cfads)
        facility_results.append(schedule)
        total_draws += schedule["draws"]
        total_interest += schedule["interest_cash"]
        total_interest_cap += schedule["interest_capitalised"]
        total_principal += schedule["principal"]
        total_debt_service += schedule["debt_service"]
        total_balance += schedule["balance"]
        total_fees += schedule["fees"]
        needs_cfads = needs_cfads or bool(schedule["needs_cfads"])

    return {
        "facilities": facility_results,
        "debt_draws": total_draws,
        "interest_cash": total_interest,
        "interest_capitalised": total_interest_cap,
        "principal": total_principal,
        "debt_service": total_debt_service,
        "balance": total_balance,
        "fees": total_fees,
        "needs_cfads": needs_cfads,
    }


def working_capital_block(
    inp: WTEMasterInputs,
    revenue: Dict[str, np.ndarray],
    opex: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute detailed working capital balances and cash effects."""

    cfg = inp.finance.working_capital
    periods = inp.timeline.n

    revenue_total = revenue["total_revenue"]
    opex_total = opex["total_opex"]
    variable = opex.get("variable_total", opex_total)

    receivables = revenue_total * (cfg.receivable_days / 365.0)
    prepaid = opex_total * (cfg.prepaid_days / 365.0)
    inventory = variable * (cfg.inventory_days / 365.0)
    payables = opex_total * (cfg.payable_days / 365.0)
    other_assets = revenue_total * cfg.other_current_asset_pct_revenue
    other_liabilities = opex_total * cfg.other_current_liability_pct_opex

    net_wc = receivables + prepaid + inventory + other_assets - payables - other_liabilities
    cash_effect = np.concatenate(([net_wc[0]], np.diff(net_wc)))

    return {
        "receivables": receivables,
        "prepaid": prepaid,
        "inventory": inventory,
        "payables": payables,
        "other_assets": other_assets,
        "other_liabilities": other_liabilities,
        "net_working_capital": net_wc,
        "cash_effect": cash_effect,
    }


def tax_block(
    inp: WTEMasterInputs,
    taxable_income: np.ndarray,
    revenue_total: np.ndarray,
    energy: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Corporate income tax calculation with loss carry-forward and incentives."""

    tax_cfg = inp.finance.tax
    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year

    incentives = (
        expand_series(tax_cfg.other_incentives, periods, ppy, label="tax_incentives")
        if tax_cfg.other_incentives is not None
        else np.zeros(periods)
    )
    carbon_credit = np.zeros(periods)
    if tax_cfg.carbon_credit_per_mwh:
        carbon_credit = energy["net_mwh"] * tax_cfg.carbon_credit_per_mwh

    carryforward_periods = tax_cfg.carryforward_years * ppy if tax_cfg.carryforward_years else periods
    loss_queue: List[Tuple[float, int]] = []

    cash_tax = np.zeros(periods)
    taxable_after_losses = np.zeros(periods)
    loss_balance = np.zeros(periods)

    holiday_periods = tax_cfg.holiday_years * ppy
    holiday_start = timeline.construction_periods + tax_cfg.holiday_start_offset

    for idx in range(periods):
        taxable = taxable_income[idx] - incentives[idx] - carbon_credit[idx]

        if tax_cfg.allow_loss_carryforward:
            taxable, loss_queue = _apply_loss_queue(taxable, loss_queue)

        if taxable <= 0:
            if tax_cfg.allow_loss_carryforward and taxable < 0:
                loss_queue.append((-taxable, carryforward_periods))
            taxable = 0.0
        taxable_after_losses[idx] = taxable

        in_holiday = False
        if holiday_periods > 0 and holiday_start <= idx < holiday_start + holiday_periods:
            in_holiday = True

        income_tax = 0.0 if in_holiday else taxable * tax_cfg.corporate_rate
        minimum_tax = revenue_total[idx] * tax_cfg.minimum_tax_rate
        cash_tax[idx] = max(income_tax, minimum_tax)

        loss_balance[idx] = sum(amount for amount, _ in loss_queue)

        if tax_cfg.allow_loss_carryforward:
            loss_queue = [
                (amount, life - 1)
                for amount, life in loss_queue
                if life - 1 > 0 and amount > 1e-9
            ]

    return {
        "cash_tax": cash_tax,
        "taxable_after_losses": taxable_after_losses,
        "loss_carryforward_balance": loss_balance,
        "carbon_credit": carbon_credit,
    }


def depreciation_schedule(
    inp: WTEMasterInputs,
    capex: Dict[str, Dict[str, np.ndarray] | np.ndarray],
) -> Dict[str, np.ndarray | Dict[str, np.ndarray]]:
    """Straight-line or declining-balance depreciation per capex item."""

    timeline = inp.timeline
    periods = timeline.n
    ppy = timeline.periods_per_year
    cod = timeline.construction_periods

    total = np.zeros(periods)
    items: Dict[str, np.ndarray] = {}
    accumulated_by_item: Dict[str, np.ndarray] = {}
    net_book_by_item: Dict[str, np.ndarray] = {}

    if inp.costs.capex_items:
        for item in inp.costs.capex_items:
            spend = capex["items"].get(item.name)
            amount = float(spend.sum()) if spend is not None else item.amount
            schedule = np.zeros(periods)
            if item.life_years <= 0:
                # Land or non-depreciable asset
                items[item.name] = schedule
                accumulated_by_item[item.name] = np.zeros(periods)
                net_book_by_item[item.name] = np.full(periods, amount)
                continue

            life_periods = int(item.life_years * ppy)
            start = cod
            bonus = amount * item.bonus_depreciation_pct
            residual = amount * item.residual_value_pct
            depreciable = max(amount - residual, 0.0)
            remaining_basis = max(depreciable - bonus, 0.0)

            if bonus > 0 and start < periods:
                schedule[start] += bonus

            if item.method == "declining_balance":
                rate = item.declining_balance_rate or (2.0 / item.life_years)
                period_rate = rate / ppy
                book = remaining_basis
                for idx in range(start, min(periods, start + life_periods)):
                    depreciation = book * period_rate
                    next_book = book - depreciation
                    min_book = residual
                    if next_book < min_book:
                        depreciation = book - min_book
                        book = min_book
                    else:
                        book = next_book
                    schedule[idx] += max(depreciation, 0.0)
                    if book <= min_book + 1e-6:
                        break
            else:
                if life_periods > 0:
                    per_period = remaining_basis / life_periods
                    for idx in range(start, min(periods, start + life_periods)):
                        schedule[idx] += per_period

            total += schedule
            items[item.name] = schedule
            accumulated = np.cumsum(schedule)
            accumulated_by_item[item.name] = accumulated
            net = np.maximum(amount - accumulated, residual)
            net_book_by_item[item.name] = net
    else:
        life_periods = inp.finance.depr_years * ppy
        schedule = np.zeros(periods)
        per_period = inp.costs.capex_total_usd / max(life_periods, 1)
        for idx in range(cod, min(periods, cod + life_periods)):
            schedule[idx] = per_period
        total += schedule
        items["total_capex"] = schedule
        accumulated_by_item["total_capex"] = np.cumsum(schedule)
        net_book_by_item["total_capex"] = np.maximum(
            inp.costs.capex_total_usd - accumulated_by_item["total_capex"],
            0.0,
        )

    accumulated_total = np.cumsum(total)
    net_book_total = np.zeros(periods)
    for idx in range(periods):
        net_book_total[idx] = sum(net[idx] for net in net_book_by_item.values())

    return {
        "total": total,
        "items": items,
        "accumulated_total": accumulated_total,
        "net_book_total": net_book_total,
        "accumulated_by_item": accumulated_by_item,
        "net_book_by_item": net_book_by_item,
    }


def coverage_ratios(
    debt: Dict[str, np.ndarray | List[Dict[str, np.ndarray | float | bool]] | bool],
    cfads: np.ndarray,
    project_cash: np.ndarray,
    timeline,
    discount_rate: float,
) -> Dict[str, float]:
    """Compute LLCR and PLCR style coverage ratios using aggregate cash flows."""

    ppy = timeline.periods_per_year
    start = timeline.construction_periods
    amort_balances = debt["balance"]
    opening = amort_balances[start - 1] if start > 0 else amort_balances[start]
    if opening <= 0:
        return {"llcr": np.nan, "plcr": np.nan}

    disc = discount_rate / ppy
    mask = np.arange(timeline.n) >= start
    cfads_future = cfads[mask]
    project_future = project_cash[mask]
    discounts = discount_factors(disc, cfads_future.size)
    pv_cfads = float(np.dot(cfads_future, discounts))
    pv_project = float(np.dot(project_future, discounts))

    llcr = pv_cfads / opening if opening > 0 else np.nan
    plcr = pv_project / opening if opening > 0 else np.nan
    return {"llcr": llcr, "plcr": plcr}

