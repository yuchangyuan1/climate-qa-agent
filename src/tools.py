"""
Climate data analysis tools.

This module provides computational tools for analyzing NetCDF climate data
from ERA5 reanalysis.
"""

import os
from typing import Dict, Any, Optional

import xarray as xr
import numpy as np

# Default data directory
DEFAULT_DATA_DIR = "./data/"

# Global datasets (lazy loaded)
_datasets: Dict[str, xr.Dataset] = {}

# Supported variables
SUPPORTED_VARIABLES = ["t2m", "d2m", "u10", "v10", "msl", "tp"]


def _get_data_paths(data_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Get paths to NetCDF data files.

    Args:
        data_dir: Directory containing .nc files.

    Returns:
        Dictionary mapping variable names to file paths.
    """
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

    return {
        "t2m": os.path.join(data_dir, "t2m.nc"),
        "d2m": os.path.join(data_dir, "d2m.nc"),
        "u10": os.path.join(data_dir, "u10.nc"),
        "v10": os.path.join(data_dir, "v10.nc"),
        "msl": os.path.join(data_dir, "msl.nc"),
        "tp": os.path.join(data_dir, "tp.nc"),
    }


def load_datasets(data_dir: Optional[str] = None) -> Dict[str, xr.Dataset]:
    """
    Load all available NetCDF datasets.

    Args:
        data_dir: Directory containing .nc files.

    Returns:
        Dictionary mapping variable names to xarray Datasets.
    """
    global _datasets

    if _datasets:
        return _datasets

    data_paths = _get_data_paths(data_dir)

    for var, path in data_paths.items():
        if os.path.exists(path):
            try:
                _datasets[var] = xr.open_dataset(path, engine="h5netcdf")
                print(f"Loaded {var} from {path}")
            except Exception as e:
                print(f"Warning: Could not load {var}: {e}")

    if not _datasets:
        print("Warning: No datasets loaded. Check data directory.")

    return _datasets


def get_datasets() -> Dict[str, xr.Dataset]:
    """
    Get loaded datasets, loading them if necessary.

    Returns:
        Dictionary of loaded datasets.
    """
    global _datasets
    if not _datasets:
        load_datasets()
    return _datasets


def _standardize_da(da: xr.DataArray) -> xr.DataArray:
    """
    Standardize ERA5 DataArray dimension names.

    Renames:
      - valid_time -> time
      - latitude -> lat
      - longitude -> lon

    Args:
        da: Input DataArray.

    Returns:
        Standardized DataArray.
    """
    rename = {}
    if "valid_time" in da.dims:
        rename["valid_time"] = "time"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    if "number" in da.dims:
        da = da.mean(dim="number")

    return da


def inspect_dataset(variable: str) -> Dict[str, Any]:
    """
    Inspect dataset metadata and time coverage.

    Args:
        variable: Variable name (e.g., 't2m', 'tp').

    Returns:
        Dictionary with dataset information.
    """
    datasets = get_datasets()

    if variable not in datasets:
        return {"error": f"Unknown variable '{variable}'. Available: {list(datasets.keys())}"}

    ds = datasets[variable]
    var_name = list(ds.data_vars)[0]
    da = _standardize_da(ds[var_name])

    return {
        "variable": variable,
        "dims": list(da.dims),
        "shape": tuple(da.shape),
        "time_start": str(da["time"].values[0]),
        "time_end": str(da["time"].values[-1]),
        "units": da.attrs.get("units", "unknown"),
    }


def compute_stat(
    variable: str,
    metric: str = "mean",
    spatial: str = "box_mean",
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    units: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute a statistic over climate data.

    Args:
        variable: Variable name (e.g., 't2m', 'tp').
        metric: Statistic to compute ('mean', 'max', 'min', 'sum').
        spatial: Spatial aggregation ('box_mean').
        lat: Latitude for point selection (not used with box_mean).
        lon: Longitude for point selection (not used with box_mean).
        units: Unit conversion ('C' for Celsius for temperature variables).

    Returns:
        Dictionary with computed statistic.
    """
    datasets = get_datasets()

    if variable not in datasets:
        return {"error": f"Unknown variable '{variable}'. Available: {list(datasets.keys())}"}

    ds = datasets[variable]
    var_name = list(ds.data_vars)[0]
    da = _standardize_da(ds[var_name])

    # Spatial reduction
    if spatial == "box_mean":
        if "lat" in da.dims:
            da = da.mean(dim="lat")
        if "lon" in da.dims:
            da = da.mean(dim="lon")

    # Compute metric
    if metric == "mean":
        value = float(da.mean())
    elif metric == "max":
        value = float(da.max())
    elif metric == "min":
        value = float(da.min())
    elif metric == "sum":
        value = float(da.sum())
    else:
        return {"error": f"Unsupported metric '{metric}'. Use: mean, max, min, sum"}

    unit_out = da.attrs.get("units", "native")

    # Temperature conversion (K -> C)
    if variable in ["t2m", "d2m"] and units == "C":
        value -= 273.15
        unit_out = "°C"

    # Precipitation special handling
    if variable == "tp":
        return {
            "variable": variable,
            "metric": metric,
            "value_m": value,
            "value_mm": value * 1000.0,
            "unit": "mm",
        }

    return {
        "variable": variable,
        "metric": metric,
        "value": value,
        "unit": unit_out,
    }
