"""
Data loading and preprocessing utilities for NEM PRICE_AND_DEMAND data.

This module contains helper functions to:
- Build file paths for AEMO PRICE_AND_DEMAND CSV files.
- Load a range of monthly CSVs into a single DataFrame.
- Convert the raw data into a standard price time series with a DateTimeIndex.

The functions are designed to work with a local directory where you have
already downloaded the AEMO PRICE_AND_DEMAND CSV files. By default we assume
files are named like:

    PRICE_AND_DEMAND_YYYYMM_REGION.csv

and stored under:

    <data_dir>/<region>/

For example:

    data/aemo_price_and_demand/VIC1/PRICE_AND_DEMAND_202412_VIC1.csv
"""

import os
from typing import List, Optional

import pandas as pd


def build_price_and_demand_filepath(
    region: str,
    year: int,
    month: int,
    data_dir: str,
) -> str:
    """
    Build the local filepath for a single AEMO PRICE_AND_DEMAND CSV file.

    Args:
        region: NEM region ID, e.g. 'VIC1', 'NSW1', 'QLD1'.
        year: Four-digit year (e.g. 2024).
        month: Month number from 1 to 12.
        data_dir: Root directory that contains region subfolders, e.g.
            'data/aemo_price_and_demand'.

    Returns:
        Absolute or relative path to the expected CSV file.
    """
    file_name = f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"
    region_dir = os.path.join(data_dir, region)
    file_path = os.path.join(region_dir, file_name)
    return file_path


def _iterate_months_inclusive(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[pd.Timestamp]:
    """
    Internal helper: generate a list of first-of-month timestamps
    from (start_year, start_month) to (end_year, end_month), inclusive.
    """
    current = pd.Timestamp(year=start_year, month=start_month, day=1)
    end = pd.Timestamp(year=end_year, month=end_month, day=1)

    months: List[pd.Timestamp] = []
    while current <= end:
        months.append(current)
        current = current + pd.DateOffset(months=1)
    return months


def load_price_and_demand_range(
    region: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    data_dir: str,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Load a range of PRICE_AND_DEMAND CSV files and concatenate them.

    Args:
        region: NEM region ID, e.g. 'VIC1', 'NSW1', 'QLD1'.
        start_year: First year to load.
        start_month: First month to load (1-12).
        end_year: Last year to load.
        end_month: Last month to load (1-12).
        data_dir: Root directory that contains region subfolders.
        strict: If True, raise FileNotFoundError when a file is missing.
            If False, skip missing files and only load the ones that exist.

    Returns:
        A concatenated DataFrame containing all loaded months.
        Columns will be whatever is present in the raw CSV files.

    Raises:
        FileNotFoundError: if strict=True and any expected file is missing.
        ValueError: if no files were loaded.
    """
    months = _iterate_months_inclusive(start_year, start_month, end_year, end_month)

    raw_dfs: List[pd.DataFrame] = []
    missing_files: List[str] = []

    for ts in months:
        year = ts.year
        month = ts.month
        path = build_price_and_demand_filepath(
            region=region,
            year=year,
            month=month,
            data_dir=data_dir,
        )

        if not os.path.exists(path):
            missing_files.append(path)
            if strict:
                raise FileNotFoundError(
                    f"Expected PRICE_AND_DEMAND file not found: {path}"
                )
            continue

        df_month = pd.read_csv(path)
        raw_dfs.append(df_month)

    if not raw_dfs:
        raise ValueError(
            "No PRICE_AND_DEMAND files were loaded. "
            "Check your region, date range, and data_dir."
        )

    raw = pd.concat(raw_dfs, ignore_index=True)
    return raw


def prepare_price_series(
    raw_df: pd.DataFrame,
    region: Optional[str] = None,
    time_col: str = "SETTLEMENTDATE",
    price_col: str = "RRP",
) -> pd.DataFrame:
    """
    Convert raw AEMO PRICE_AND_DEMAND data into a clean price time series.

    The output is a DataFrame with a DateTimeIndex and a single column 'price'.

    Args:
        raw_df: Raw DataFrame loaded from PRICE_AND_DEMAND CSV files.
        region: Optional NEM region to filter by using the 'REGIONID' column.
            If None, no region filtering is applied.
        time_col: Name of the timestamp column in the raw data.
        price_col: Name of the price column in the raw data.

    Returns:
        DataFrame with:
            - DateTimeIndex taken from time_col.
            - A single column 'price' in $/MWh.
    """
    df = raw_df.copy()

    if region is not None and "REGIONID" in df.columns:
        df = df[df["REGIONID"] == region]

    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in raw_df.")

    if price_col not in df.columns:
        raise ValueError(f"price column '{price_col}' not found in raw_df.")

    df[time_col] = pd.to_datetime(df[time_col])
    df = df[[time_col, price_col]].copy()
    df = df.set_index(time_col)
    df = df.rename(columns={price_col: "price"})
    df = df.sort_index()

    return df


def load_price_range(
    region: str,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    data_dir: str,
    strict: bool = True,
    time_col: str = "SETTLEMENTDATE",
    price_col: str = "RRP",
) -> pd.DataFrame:
    """
    High-level helper: load raw PRICE_AND_DEMAND data and return a clean price series.

    This function combines load_price_and_demand_range and prepare_price_series
    into a single call suitable for notebooks.

    Args:
        region: NEM region ID, e.g. 'VIC1', 'NSW1', 'QLD1'.
        start_year: First year to load.
        start_month: First month to load (1-12).
        end_year: Last year to load.
        end_month: Last month to load (1-12).
        data_dir: Root directory that contains region subfolders.
        strict: If True, raise FileNotFoundError when a file is missing.
        time_col: Name of the timestamp column in the raw data.
        price_col: Name of the price column in the raw data.

    Returns:
        DataFrame with a DateTimeIndex and a 'price' column.
    """
    raw = load_price_and_demand_range(
        region=region,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        data_dir=data_dir,
        strict=strict,
    )

    price_df = prepare_price_series(
        raw_df=raw,
        region=region,
        time_col=time_col,
        price_col=price_col,
    )

    return price_df


def download_price_and_demand(
    *args,
    **kwargs,
) -> None:
    """
    Placeholder for automated downloading of AEMO PRICE_AND_DEMAND data.

    This function is provided so that notebooks can safely import:

        from data_utils import download_price_and_demand

    without failing, even though the actual implementation is not yet provided.

    Current behaviour:
        - Raises NotImplementedError with a short explanation.

    In a future version you can implement HTTP downloads from the AEMO data portal
    or any other source you prefer.
    """
    raise NotImplementedError(
        "download_price_and_demand is not implemented yet. "
        "Please download PRICE_AND_DEMAND CSV files manually and place them under "
        "<data_dir>/<region>/ as described in the README."
    )
