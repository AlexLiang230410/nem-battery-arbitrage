"""
Core battery arbitrage logic.

This module implements a simple threshold-based battery arbitrage model
that can be applied to real NEM spot price data (or any other price
time series with a regular time step).

All functions in this file are designed to be imported and used from
notebooks or other scripts. They do not depend on Google Colab.
"""

from typing import Dict, Any, Iterable, Tuple

import pandas as pd


def simulate_arbitrage(
    price_df: pd.DataFrame,
    dt_mins: float,
    power_mw: float,
    energy_mwh: float,
    charge_threshold: float,
    discharge_threshold: float,
    efficiency: float,
) -> pd.DataFrame:
    """
    Simulate a simple threshold-based arbitrage strategy for a single battery.

    The strategy is:
    - If price <= charge_threshold: charge at maximum power (subject to SoC limits).
    - If price >= discharge_threshold: discharge at maximum power (subject to SoC limits).
    - Otherwise: stay idle.

    The battery state of charge (SoC) is tracked in MWh. A simple round-trip
    efficiency model is used where efficiency is only applied when charging.

    Args:
        price_df: DataFrame with a DateTimeIndex and a 'price' column.
        dt_mins: Time step length in minutes (e.g., 5 for 5-minute data).
        power_mw: Maximum charging/discharging power in MW.
        energy_mwh: Battery energy capacity in MWh.
        charge_threshold: Charge when price is less than or equal to this value ($/MWh).
        discharge_threshold: Discharge when price is greater than or equal to this value ($/MWh).
        efficiency: Round-trip efficiency (0-1). Implemented as an efficiency loss on charging.

    Returns:
        A new DataFrame with the original 'price' column and additional columns:
        - 'action': string indicating 'charge', 'discharge', or 'idle'.
        - 'power_mw': signed power at the grid connection point (MW). Positive = discharge,
          negative = charge.
        - 'soc_mwh': state of charge in MWh.
        - 'revenue': revenue earned at each time step (AUD), using price * power * dt_hours.
        - 'cumulative_revenue': cumulative revenue over time (AUD).
    """
    if "price" not in price_df.columns:
        raise ValueError("price_df must contain a 'price' column.")

    if price_df.empty:
        raise ValueError("price_df is empty. Nothing to simulate.")

    if discharge_threshold <= charge_threshold:
        raise ValueError(
            "discharge_threshold must be strictly greater than charge_threshold."
        )

    dt_hours = dt_mins / 60.0

    soc = 0.0  # MWh
    actions = []
    powers = []
    socs = []
    revenues = []

    for _, row in price_df.iterrows():
        price = float(row["price"])
        action = "idle"
        grid_power_mw = 0.0  # positive = discharge, negative = charge

        # Maximum feasible charge/discharge power based on SoC constraints
        max_charge_mw = (energy_mwh - soc) / dt_hours if energy_mwh > 0 else 0.0
        max_charge_mw = max(max_charge_mw, 0.0)

        max_discharge_mw = soc / dt_hours
        max_discharge_mw = max(max_discharge_mw, 0.0)

        if price <= charge_threshold and max_charge_mw > 0.0:
            # Charge
            action = "charge"
            # Battery-side charging power (MW)
            battery_charge_mw = min(power_mw, max_charge_mw)
            # Grid sees negative power while charging
            grid_power_mw = -battery_charge_mw
            # Apply efficiency on charging: only a fraction of input energy is stored
            stored_energy_mwh = battery_charge_mw * dt_hours * efficiency
            soc += stored_energy_mwh

        elif price >= discharge_threshold and max_discharge_mw > 0.0:
            # Discharge
            action = "discharge"
            discharge_mw = min(power_mw, max_discharge_mw)
            grid_power_mw = discharge_mw
            # Energy taken from the battery (no additional loss modeled on discharge)
            discharged_energy_mwh = discharge_mw * dt_hours
            soc -= discharged_energy_mwh

        # Enforce numeric bounds on SoC
        if soc < 0.0:
            soc = 0.0
        if soc > energy_mwh:
            soc = energy_mwh

        revenue = price * grid_power_mw * dt_hours

        actions.append(action)
        powers.append(grid_power_mw)
        socs.append(soc)
        revenues.append(revenue)

    result = price_df.copy()
    result["action"] = actions
    result["power_mw"] = powers
    result["soc_mwh"] = socs
    result["revenue"] = revenues
    result["cumulative_revenue"] = result["revenue"].cumsum()

    return result


def summarize_performance(
    result_df: pd.DataFrame,
    dt_mins: float,
    power_mw: float,
    energy_mwh: float,
    print_summary: bool = True,
) -> Dict[str, float]:
    """
    Compute and optionally print key performance metrics from a simulation result.

    Args:
        result_df: Output DataFrame from simulate_arbitrage.
        dt_mins: Duration of each time step in minutes.
        power_mw: Battery maximum power in MW.
        energy_mwh: Battery energy capacity in MWh.
        print_summary: If True, print a human-readable summary to stdout.

    Returns:
        A dictionary containing:
            - 'power_mw'
            - 'energy_mwh'
            - 'total_revenue'
            - 'discharge_energy_mwh'
            - 'equivalent_full_cycles'
            - 'discharge_hours'
    """
    if result_df.empty:
        raise ValueError("result_df is empty. Cannot summarize performance.")

    dt_hours = dt_mins / 60.0

    total_revenue = float(result_df["revenue"].sum())

    # Total discharged energy is based on positive power (discharging)
    discharge_mask = result_df["power_mw"] > 0
    discharge_energy_mwh = float(
        (result_df.loc[discharge_mask, "power_mw"] * dt_hours).sum()
    )

    equivalent_full_cycles = (
        discharge_energy_mwh / energy_mwh if energy_mwh > 0 else 0.0
    )

    discharge_hours = discharge_energy_mwh / power_mw if power_mw > 0 else 0.0

    summary = {
        "power_mw": float(power_mw),
        "energy_mwh": float(energy_mwh),
        "total_revenue": total_revenue,
        "discharge_energy_mwh": discharge_energy_mwh,
        "equivalent_full_cycles": equivalent_full_cycles,
        "discharge_hours": discharge_hours,
    }

    if print_summary:
        print("=== Battery Arbitrage Performance Summary ===")
        print(f"Power (MW):                {summary['power_mw']:.2f}")
        print(f"Energy (MWh):              {summary['energy_mwh']:.2f}")
        print(f"Total revenue (AUD):       {summary['total_revenue']:.2f}")
        print(
            f"Discharged energy (MWh):   {summary['discharge_energy_mwh']:.2f}"
        )
        print(
            f"Equivalent full cycles:    {summary['equivalent_full_cycles']:.2f}"
        )
        print(f"Discharge hours (h):       {summary['discharge_hours']:.2f}")
        print("============================================")

    return summary


def run_scenarios(
    price_df: pd.DataFrame,
    dt_mins: float,
    scenarios: Dict[str, Dict[str, Any]],
    print_each_summary: bool = False,
) -> pd.DataFrame:
    """
    Run multiple battery arbitrage scenarios and compare their performance.

    Args:
        price_df: DataFrame with a DateTimeIndex and a 'price' column.
        dt_mins: Time step length in minutes.
        scenarios: A dictionary where keys are scenario names and values are dictionaries
            containing the parameters:
            'power_mw', 'energy_mwh', 'charge_threshold', 'discharge_threshold', 'efficiency'.
        print_each_summary: If True, print a summary for each scenario.

    Returns:
        A DataFrame where each row corresponds to one scenario and columns are
        the performance metrics returned by summarize_performance.
    """
    all_summaries: Dict[str, Dict[str, float]] = {}

    for name, params in scenarios.items():
        required_keys = {
            "power_mw",
            "energy_mwh",
            "charge_threshold",
            "discharge_threshold",
            "efficiency",
        }
        missing = required_keys - set(params.keys())
        if missing:
            raise ValueError(
                f"Scenario '{name}' is missing required parameters: {missing}"
            )

        result = simulate_arbitrage(
            price_df=price_df,
            dt_mins=dt_mins,
            power_mw=params["power_mw"],
            energy_mwh=params["energy_mwh"],
            charge_threshold=params["charge_threshold"],
            discharge_threshold=params["discharge_threshold"],
            efficiency=params["efficiency"],
        )

        summary = summarize_performance(
            result_df=result,
            dt_mins=dt_mins,
            power_mw=params["power_mw"],
            energy_mwh=params["energy_mwh"],
            print_summary=print_each_summary,
        )

        all_summaries[name] = summary

    comparison_df = pd.DataFrame.from_dict(all_summaries, orient="index")
    comparison_df.index.name = "scenario"

    return comparison_df


def grid_search_thresholds(
    price_df: pd.DataFrame,
    dt_mins: float,
    power_mw: float,
    energy_mwh: float,
    charge_threshold_range: Iterable[float],
    discharge_threshold_range: Iterable[float],
    efficiency: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform a grid search over charge and discharge thresholds.

    For each combination where charge_threshold < discharge_threshold,
    this function runs a simulation and records the total revenue.

    Args:
        price_df: DataFrame with a DateTimeIndex and a 'price' column.
        dt_mins: Time step length in minutes.
        power_mw: Battery maximum power in MW.
        energy_mwh: Battery energy capacity in MWh.
        charge_threshold_range: Iterable of charge threshold candidates.
        discharge_threshold_range: Iterable of discharge threshold candidates.
        efficiency: Round-trip efficiency (0-1).

    Returns:
        A tuple of:
            - grid_search_df: DataFrame with columns
              ['charge_threshold', 'discharge_threshold', 'total_revenue'].
            - best_params: dict containing the best combination and its revenue:
              {
                  'charge_threshold': ...,
                  'discharge_threshold': ...,
                  'total_revenue': ...
              }
    """
    records = []

    for ct in charge_threshold_range:
        for dt_ in discharge_threshold_range:
            if ct >= dt_:
                continue

            result = simulate_arbitrage(
                price_df=price_df,
                dt_mins=dt_mins,
                power_mw=power_mw,
                energy_mwh=energy_mwh,
                charge_threshold=float(ct),
                discharge_threshold=float(dt_),
                efficiency=efficiency,
            )

            summary = summarize_performance(
                result_df=result,
                dt_mins=dt_mins,
                power_mw=power_mw,
                energy_mwh=energy_mwh,
                print_summary=False,
            )

            records.append(
                {
                    "charge_threshold": float(ct),
                    "discharge_threshold": float(dt_),
                    "total_revenue": summary["total_revenue"],
                }
            )

    if not records:
        raise ValueError(
            "No valid threshold combinations were evaluated. "
            "Check that your ranges are not empty and allow charge_threshold < discharge_threshold."
        )

    grid_search_df = pd.DataFrame.from_records(records)

    best_row = grid_search_df.loc[grid_search_df["total_revenue"].idxmax()]

    best_params: Dict[str, Any] = {
        "charge_threshold": float(best_row["charge_threshold"]),
        "discharge_threshold": float(best_row["discharge_threshold"]),
        "total_revenue": float(best_row["total_revenue"]),
    }

    return grid_search_df, best_params


def adaptive_threshold_search(
    price_df: pd.DataFrame,
    dt_mins: float,
    power_mw: float,
    energy_mwh: float,
    charge_threshold_range: Iterable[float],
    discharge_threshold_range: Iterable[float],
    efficiency: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Placeholder for a more advanced threshold search algorithm.

    For now this function simply calls grid_search_thresholds with the
    same arguments. The function name is provided to give users a stable
    import path:

        from battery_core import adaptive_threshold_search

    In future versions you can replace the internals with a more
    sophisticated search (e.g. coarse-to-fine grid, Bayesian optimisation, etc.)
    without changing the notebook interface.

    Args:
        price_df: DataFrame with a DateTimeIndex and a 'price' column.
        dt_mins: Time step length in minutes.
        power_mw: Battery maximum power in MW.
        energy_mwh: Battery energy capacity in MWh.
        charge_threshold_range: Iterable of charge threshold candidates.
        discharge_threshold_range: Iterable of discharge threshold candidates.
        efficiency: Round-trip efficiency (0-1).

    Returns:
        Same as grid_search_thresholds.
    """
    return grid_search_thresholds(
        price_df=price_df,
        dt_mins=dt_mins,
        power_mw=power_mw,
        energy_mwh=energy_mwh,
        charge_threshold_range=charge_threshold_range,
        discharge_threshold_range=discharge_threshold_range,
        efficiency=efficiency,
    )
