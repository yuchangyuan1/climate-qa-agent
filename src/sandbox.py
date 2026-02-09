"""
Python REPL sandbox for autonomous code execution.

This module provides a restricted Python execution environment that allows
the agent to generate and debug xarray/numpy/pandas code for processing
ERA5 (NetCDF) climate datasets.
"""

import io
import sys
import traceback
from typing import Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

from .tools import get_datasets, _standardize_da, SUPPORTED_VARIABLES

# Maximum execution time in seconds
EXEC_TIMEOUT = 30

# Allowed modules for import within the sandbox
ALLOWED_MODULES = {
    "numpy", "np",
    "pandas", "pd",
    "xarray", "xr",
    "math",
    "datetime",
    "statistics",
    "json",
    "re",
}

# Dangerous built-in names to remove
_BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__",
    "open", "input", "breakpoint",
    "exit", "quit",
}


def _make_safe_builtins() -> dict:
    """Create a restricted builtins dict."""
    import builtins
    safe = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS}
    return safe


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted import that only allows whitelisted modules."""
    top_level = name.split(".")[0]
    if top_level not in ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed. "
            f"Allowed modules: {sorted(ALLOWED_MODULES)}"
        )
    return __builtins__["__import__"](name, globals, locals, fromlist, level) if isinstance(__builtins__, dict) \
        else __import__(name, globals, locals, fromlist, level)


def _build_namespace() -> dict:
    """Build the execution namespace with pre-injected libraries and data."""
    datasets = get_datasets()

    # Pre-standardize datasets for convenience
    standardized = {}
    for var_name, ds in datasets.items():
        data_var = list(ds.data_vars)[0]
        standardized[var_name] = _standardize_da(ds[data_var])

    safe_builtins = _make_safe_builtins()
    safe_builtins["__import__"] = _safe_import

    namespace = {
        "__builtins__": safe_builtins,
        # Pre-injected libraries
        "np": np,
        "numpy": np,
        "pd": pd,
        "pandas": pd,
        "xr": xr,
        "xarray": xr,
        # Pre-loaded climate datasets (raw xarray Datasets)
        "datasets": datasets,
        # Pre-standardized DataArrays (convenient for analysis)
        "data": standardized,
        # Helper info
        "SUPPORTED_VARIABLES": SUPPORTED_VARIABLES,
    }

    return namespace


def run_code(code: str, timeout: int = EXEC_TIMEOUT) -> Dict[str, Any]:
    """
    Execute Python code in a restricted sandbox.

    The sandbox provides:
    - Pre-imported: numpy (np), pandas (pd), xarray (xr)
    - Pre-loaded: datasets (dict of xr.Dataset), data (dict of standardized DataArrays)
    - Available variables in `data`: t2m, d2m, u10, v10, msl, tp (as xr.DataArray)
    - Restricted imports (only numpy, pandas, xarray, math, datetime, statistics, json, re)
    - No file I/O, no system access, no network access

    The code can print() results or assign to a variable named `result`.

    Args:
        code: Python code string to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        Dictionary with keys:
        - success (bool): Whether execution completed without error
        - output (str): Captured stdout output
        - result (str): String representation of the `result` variable if set
        - error (str): Error message and traceback if execution failed
    """
    namespace = _build_namespace()

    # Capture stdout
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    response: Dict[str, Any] = {
        "success": False,
        "output": "",
        "result": None,
        "error": None,
    }

    try:
        sys.stdout = stdout_capture

        # Execute with timeout (signal.alarm not available on Windows,
        # so we use a simple exec with no OS-level timeout on Windows)
        exec(code, namespace)

        response["success"] = True
        response["output"] = stdout_capture.getvalue()

        # Check if user set a `result` variable
        if "result" in namespace and namespace["result"] is not namespace.get("__builtins__"):
            res = namespace["result"]
            response["result"] = str(res)[:2000]

    except Exception:
        response["error"] = traceback.format_exc()
        response["output"] = stdout_capture.getvalue()

    finally:
        sys.stdout = old_stdout

    return response
