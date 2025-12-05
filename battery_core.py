"""
Core battery arbitrage logic.

This module implements a simple threshold-based battery arbitrage model
that can be applied to real NEM spot price data (or any other price
time series with a regular time step).

All functions in this file are designed to be imported and used from
notebooks or other scripts. They do not depend on Google Colab.
"""

from typing import Dict, Any, Iterable, Tuple

import numpy as np
import pandas as pd

# Optional progress bar support. If tqdm is not available, we fall back
# to a no-op wrapper so that code still runs without errors.
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(0)


# ---------------------------------------------------------------------------
# Internal helper: fast revenue-only simulation using NumPy arrays
# ---------------------------------------------------------------------------


def _simulate_revenue_only(
    prices: np.ndarray,
    dt_hours: float,
    power_mw: float,
    energy_mwh: float,
    charge_threshold: float,
    discharge_threshold: float,
    efficiency: float,
) -> float:
    """
    Fast internal helper that simulates the battery strategy and returns only
    the total revenue. This avoids allocating large arrays or DataFrames and
    is therefore much faster when used inside grid search loops.

    Args:
        prices: 1D NumPy array of prices ($/MWh).
        dt_hours: Time step length in hours.
        power_mw: Maximum charging/discharging power in MW.
        energy_mwh: Battery energy capacity in MWh.
        charge_threshold: Charge when price <= this value.
        discharge_threshold: Discharge when price >= this value.
        efficiency: Round-trip efficiency (0-1), applied on charging.

    Returns:
        Total revenue over the entire time horizon (AUD).
    """
    soc = 0.0
    total_revenue = 0.0

    for price in prices:
        action_charge = price <= charge_threshold
        action_discharge = price >= discharge_threshold

        grid_power_mw = 0.0

        # Maximum feasible charge/discharge power based on SoC constraints
        if energy_mwh > 0.0:
            max_charge_mw = (energy_mwh - soc) / dt_hours
            max_charge_mw = max(max_charge_mw, 0.0)
        else:
            max_charge_mw = 0.0

        max_discharge_mw = soc / dt_hours
        max_discharge_mw = max(max_discharge_mw, 0.0)

        if action_charge and max_charge_mw > 0.0:
            # Charge
            battery_charge_mw = min(power_mw, max_charge_mw)
            grid_power_mw = -battery_charge_mw
            stored_energy_mwh = battery_charge_mw * dt_hours * efficiency
            soc += stored_energy_mwh

        elif action_discharge and max_discharge_mw > 0.0:
            # Discharge
            discharge_mw = min(power_mw, max_discharge_mw)
            grid_power_mw = discharge_mw
            discharged_energy_mwh = discharge_mw * dt_hours
            soc -= discharged_energy_mwh

        # Clamp SoC
        if soc < 0.0:
            soc = 0.0
        elif soc > energy_mwh:
            soc = energy_mwh

        total_revenue += price * grid_power_mw * dt_hours

    return float(total_revenue)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_arbitrage(
    price_df: pd.DataFrame,
    dt_mins: float,
    power_mw: float,
    energy_mwh: float,
    charge_threshold: float,
    discharge_threshold: float,
    efficiency: float,
    show_progress: bool = False,
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
        show_progress: If True, display a progress bar when iterating over time steps.
            This is mainly useful for very long time series.

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

    # Use NumPy array for fast iteration instead of DataFrame.iterrows()
    prices = price_df["price"].to_numpy(dtype=float)
    n_steps = prices.shape[0]

    soc = 0.0  # MWh
    actions: list[str] = []
    powers: list[float] = []
    socs: list[float] = []
    revenues: list[float] = []

    iterator = range(n_steps)
    if show_progress and n_steps > 50_000:
        iterator = tqdm(iterator, total=n_steps, desc="Simulating arbitrage")

    for i in iterator:
        price = prices[i]
        action = "idle"
        grid_power_mw = 0.0  # positive = discharge, negative = charge

        # Maximum feasible charge/discharge power based on SoC constraints
        if energy_mwh > 0.0:
            max_charge_mw = (energy_mwh - soc) / dt_hours
            max_charge_mw = max(max_charge_mw, 0.0)
        else:
            max_charge_mw = 0.0

        max_discharge_mw = soc / dt_hours
        max_discharge_mw = max(max_discharge_mw, 0.0)

        if price <= charge_threshold and max_charge_mw > 0.0:
            # Charge
            action = "charge"
            battery_charge_mw = min(power_mw, max_charge_mw)
            grid_power_mw = -battery_charge_mw
            stored_energy_mwh = battery_charge_mw * dt_hours * efficiency
            soc += stored_energy_mwh

        elif price >= discharge_threshold and max_discharge_mw > 0.0:
            # Discharge
            action = "discharge"
            discharge_mw = min(power_mw, max_discharge_mw)
            grid_power_mw = discharge_mw
            discharged_energy_mwh = discharge_mw * dt_hours
            soc -= discharged_energy_mwh

        # Clamp SoC
        if soc < 0.0:
            soc = 0.0
        elif soc > energy_mwh:
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
    show_progress: bool = True,
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
        show_progress: If True and multiple scenarios are provided, show a progress bar.

    Returns:
        A DataFrame where each row corresponds to one scenario and columns are
        the performance metrics returned by summarize_performance.
    """
    all_summaries: Dict[str, Dict[str, float]] = {}

    items = list(scenarios.items())
    iterator = items
    if show_progress and len(items) > 1:
        iterator = tqdm(items, total=len(items), desc="Running scenarios")

    for name, params in iterator:
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
            show_progress=False,
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
    show_progress: bool = True,
    min_spread: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform a grid search over charge and discharge thresholds.

    For each combination where charge_threshold < discharge_threshold,
    this function runs a simulation and records the total revenue.

    Compared with the naïve implementation, this version uses a fast
    internal helper that only computes total revenue, which makes the
    search significantly faster.

    Args:
        price_df: DataFrame with a DateTimeIndex and a 'price' column.
        dt_mins: Time step length in minutes.
        power_mw: Battery maximum power in MW.
        energy_mwh: Battery energy capacity in MWh.
        charge_threshold_range: Iterable of charge threshold candidates.
        discharge_threshold_range: Iterable of discharge threshold candidates.
        efficiency: Round-trip efficiency (0-1).
        show_progress: If True, display a progress bar from 0% to 100%.
        min_spread: Optional minimum spread between discharge and charge
            thresholds (discharge_threshold - charge_threshold >= min_spread).
            This can be used to prune nearly identical strategies and
            reduce the search space.

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
    dt_hours = dt_mins / 60.0
    prices = price_df["price"].to_numpy(dtype=float)

    # Build list of valid combinations up front (for progress bar)
    combos: list[Tuple[float, float]] = []
    for ct in charge_threshold_range:
        for dt_ in discharge_threshold_range:
            if ct >= dt_:
                continue
            if min_spread > 0.0 and (dt_ - ct) < min_spread:
                continue
            combos.append((float(ct), float(dt_)))

    if not combos:
        raise ValueError(
            "No valid threshold combinations were evaluated. "
            "Check that your ranges are not empty and allow "
            "charge_threshold < discharge_threshold (and min_spread if set)."
        )

    records: list[Dict[str, float]] = []

    iterator = combos
    if show_progress and len(combos) > 1:
        iterator = tqdm(iterator, total=len(combos), desc="Grid search")

    for ct, dt_ in iterator:
        total_revenue = _simulate_revenue_only(
            prices=prices,
            dt_hours=dt_hours,
            power_mw=power_mw,
            energy_mwh=energy_mwh,
            charge_threshold=ct,
            discharge_threshold=dt_,
            efficiency=efficiency,
        )

        records.append(
            {
                "charge_threshold": ct,
                "discharge_threshold": dt_,
                "total_revenue": total_revenue,
            }
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
    show_progress: bool = True,
    min_spread: float = 0.0,
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
        show_progress: If True, display a progress bar from 0% to 100%.
        min_spread: Optional minimum spread between thresholds.

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
        show_progress=show_progress,
        min_spread=min_spread,
    )
