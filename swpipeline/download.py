"""
Data download pipeline for PSP, SolO, and ACE in-situ datasets.

The main entry point for science analysis is build_spacecraft_dataset(),
which orchestrates downloading, cleaning, resampling, and merging in one
call. Raw CDF files are cached locally by SunPy so repeated calls are fast.

Download strategy
-----------------
- Long time ranges are chunked into yearly segments to stay within CDAWeb
  server limits and to allow partial downloads on network interruptions.
- SunPy's Fido.fetch caches CDF files locally; re-running the script will
  not re-download files already on disk.
- ACE H2 data is provided at 1-hour cadence; no resampling is needed.
- PSP and SolO 1-minute (or 30-second) data is resampled to 1-hour means
  before merging.

Offline / testing usage
-----------------------
build_spacecraft_dataset() accepts optional ``mag_df`` and ``plasma_df``
parameters. When these are supplied the download step is skipped entirely,
making it easy to unit-test the cleaning and merging logic without any
network access.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from swpipeline.config import DATASETS, RAW_COLS

#: Number of threads for parallel CDF file reading.
#: 4 threads gives ~2× speedup on typical NVMe/SSD storage;
#: diminishing returns beyond that due to the Python GIL.
_N_READ_THREADS = 4
from swpipeline.process import (
    clean_psp_mag,
    clean_psp_spc,
    clean_solo_mag,
    clean_solo_swa_pas,
    clean_ace_mag,
    clean_ace_swe,
    resample_to_hourly,
    merge_mag_plasma,
)

# ---------------------------------------------------------------------------
# Internal configuration tables
# ---------------------------------------------------------------------------

#: Map spacecraft name → dataset keys for mag and plasma
_SC_DATASETS: dict[str, dict[str, str]] = {
    "psp":  {"mag": "psp_mag",      "plasma": "psp_spc"},
    "solo": {"mag": "solo_mag",     "plasma": "solo_swa_pas"},
    "ace":  {"mag": "ace_mag",      "plasma": "ace_swe"},
}

#: Cleaning function for each dataset key
_CLEAN_FN = {
    "psp_mag":      clean_psp_mag,
    "psp_spc":      clean_psp_spc,
    "solo_mag":     clean_solo_mag,
    "solo_swa_pas": clean_solo_swa_pas,
    "ace_mag":      clean_ace_mag,
    "ace_swe":      clean_ace_swe,
}

#: ACE H2 data are pre-averaged to 1-hour; skip the resample step
_NEEDS_RESAMPLE: dict[str, bool] = {
    "psp":  True,
    "solo": True,
    "ace":  False,
}


# ---------------------------------------------------------------------------
# Utility: year chunking
# ---------------------------------------------------------------------------

def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """
    Split [start, end] into year-long sub-intervals.

    Parameters
    ----------
    start, end : str
        ISO date strings (e.g. '2022-06-15').

    Returns
    -------
    list of (chunk_start, chunk_end) string pairs, one per calendar year.

    Examples
    --------
    >>> _year_chunks("2021-07-01", "2023-03-31")
    [('2021-07-01', '2021-12-31'), ('2022-01-01', '2022-12-31'), ('2023-01-01', '2023-03-31')]
    """
    t_start = pd.Timestamp(start)
    t_end   = pd.Timestamp(end)

    chunks: list[tuple[str, str]] = []
    current = t_start
    while current <= t_end:
        year_end = min(pd.Timestamp(f"{current.year}-12-31"), t_end)
        chunks.append((current.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
        current = pd.Timestamp(f"{current.year + 1}-01-01")

    return chunks


# ---------------------------------------------------------------------------
# CDF reading (cdflib — fast, linear scaling)
# ---------------------------------------------------------------------------

#: CDF fill values are large negative numbers (typically -1e31).
#: Any value at or below this threshold is replaced with NaN.
_FILL_THRESHOLD = -1e30


def _replace_fill(arr: np.ndarray) -> np.ndarray:
    """Replace CDF fill values (≤ -1e30) with NaN."""
    arr = np.asarray(arr, dtype=float)
    arr[arr <= _FILL_THRESHOLD] = np.nan
    return arr


def _read_psp_mag(files: list[str]) -> pd.DataFrame:
    """Read PSP FLD L2 MAG RTN 1-min CDF files using cdflib (threaded)."""
    import cdflib

    def _one(path: str) -> pd.DataFrame:
        cdf   = cdflib.CDF(path)
        times = cdflib.cdfepoch.to_datetime(cdf.varget("epoch_mag_RTN_1min"))
        mag   = _replace_fill(cdf.varget("psp_fld_l2_mag_RTN_1min"))
        return pd.DataFrame({
            "psp_fld_l2_mag_RTN_1min_0": mag[:, 0],
            "psp_fld_l2_mag_RTN_1min_1": mag[:, 1],
            "psp_fld_l2_mag_RTN_1min_2": mag[:, 2],
            # Quality flags live on a separate epoch in the same CDF.
            # Reading only the MAG epoch rows means the flag column is NaN
            # here; clean_psp_mag handles this via the B-component NaN filter.
            "psp_fld_l2_quality_flags":  np.nan,
        }, index=pd.DatetimeIndex(times))

    with ThreadPoolExecutor(max_workers=_N_READ_THREADS) as pool:
        dfs = list(pool.map(_one, files))
    return pd.concat(dfs) if dfs else pd.DataFrame()


def _read_psp_spc(files: list[str]) -> pd.DataFrame:
    """Read PSP SWEAP SPC L3i CDF files using cdflib (threaded)."""
    import cdflib

    def _one(path: str) -> pd.DataFrame:
        cdf   = cdflib.CDF(path)
        times = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
        v     = _replace_fill(cdf.varget("vp_moment_RTN"))
        return pd.DataFrame({
            "np_moment":       _replace_fill(cdf.varget("np_moment")),
            "wp_moment":       _replace_fill(cdf.varget("wp_moment")),
            "vp_moment_RTN_0": v[:, 0],
            "vp_moment_RTN_1": v[:, 1],
            "vp_moment_RTN_2": v[:, 2],
            "general_flag":    cdf.varget("general_flag").astype(float),
        }, index=pd.DatetimeIndex(times))

    with ThreadPoolExecutor(max_workers=_N_READ_THREADS) as pool:
        dfs = list(pool.map(_one, files))
    return pd.concat(dfs) if dfs else pd.DataFrame()


def _ts_col_to_cdf_var(ts_col: str) -> tuple[str, int | None]:
    """
    Map a SunPy-TimeSeries column name back to its CDF variable + index.

    TimeSeries expands 3-component vectors by appending ``_0``, ``_1``, ``_2``.
    Examples::

        'B_RTN_0'    → ('B_RTN', 0)
        'V_RTN_2'    → ('V_RTN', 2)
        'QUALITY_FLAG' → ('QUALITY_FLAG', None)
    """
    m = re.match(r"^(.+)_([012])$", ts_col)
    if m:
        return m.group(1), int(m.group(2))
    return ts_col, None


def _read_ace_mag(files: list[str]) -> pd.DataFrame:
    """Read ACE MFI H2 CDF files using cdflib (threaded).

    ACE CDFs use rVariables (not zVariables); variable names confirmed from
    CDF inspection: Epoch (rVar), Magnitude (scalar), Q_FLAG (scalar).
    """
    import cdflib

    def _one(path: str) -> pd.DataFrame:
        cdf   = cdflib.CDF(path)
        times = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
        return pd.DataFrame({
            "Magnitude": _replace_fill(cdf.varget("Magnitude")),
            "Q_FLAG":    cdf.varget("Q_FLAG").astype(float),
        }, index=pd.DatetimeIndex(times))

    with ThreadPoolExecutor(max_workers=_N_READ_THREADS) as pool:
        dfs = list(pool.map(_one, files))
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs) if dfs else pd.DataFrame()


def _read_ace_swe(files: list[str]) -> pd.DataFrame:
    """Read ACE SWE H2 CDF files using cdflib (threaded).

    ACE CDFs use rVariables (not zVariables); variable names confirmed from
    CDF inspection: Epoch, Np, Vp, Tpr (scalars), V_RTN (3-component vector).
    """
    import cdflib

    def _one(path: str) -> pd.DataFrame:
        cdf   = cdflib.CDF(path)
        times = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
        v_rtn = _replace_fill(cdf.varget("V_RTN"))
        return pd.DataFrame({
            "Np":      _replace_fill(cdf.varget("Np")),
            "Tpr":     _replace_fill(cdf.varget("Tpr")),
            "Vp":      _replace_fill(cdf.varget("Vp")),
            "V_RTN_0": v_rtn[:, 0],
            "V_RTN_1": v_rtn[:, 1],
            "V_RTN_2": v_rtn[:, 2],
        }, index=pd.DatetimeIndex(times))

    with ThreadPoolExecutor(max_workers=_N_READ_THREADS) as pool:
        dfs = list(pool.map(_one, files))
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs) if dfs else pd.DataFrame()


def _read_cdf_generic(files: list[str], dataset_key: str) -> pd.DataFrame:
    """
    Generic cdflib reader for datasets without a dedicated fast reader.

    Uses RAW_COLS[dataset_key] to determine which CDF variables to extract
    and how to name the output columns (matching SunPy TimeSeries convention).

    Handles both zVariables (PSP, SolO) and rVariables (ACE) transparently.
    """
    import cdflib

    # Build {cdf_variable: [(output_col, array_idx_or_None), ...]}
    var_requests: dict[str, list[tuple[str, int | None]]] = {}
    for ts_col in RAW_COLS[dataset_key].values():
        cdf_var, idx = _ts_col_to_cdf_var(ts_col)
        var_requests.setdefault(cdf_var, []).append((ts_col, idx))

    def _one(path: str) -> pd.DataFrame:
        cdf   = cdflib.CDF(path)
        info  = cdf.cdf_info()
        # ACE CDFs use rVariables exclusively; PSP/SolO use zVariables.
        all_vars = list(info.zVariables) + list(info.rVariables)
        avail    = set(all_vars)

        epoch_var = next(
            (v for v in all_vars if v.lower().startswith("epoch")), None
        )
        if epoch_var is None:
            return pd.DataFrame()
        times = cdflib.cdfepoch.to_datetime(cdf.varget(epoch_var))
        n = len(times)

        row: dict[str, np.ndarray] = {}
        for cdf_var, requests in var_requests.items():
            if cdf_var not in avail:
                for ts_col, _ in requests:
                    row[ts_col] = np.full(n, np.nan)
                continue

            arr = _replace_fill(np.asarray(cdf.varget(cdf_var), dtype=float))

            for ts_col, idx in requests:
                if arr.ndim == 1 and len(arr) == n:
                    row[ts_col] = arr
                elif arr.ndim == 2 and arr.shape[0] == n and idx is not None:
                    row[ts_col] = arr[:, idx]
                else:
                    row[ts_col] = np.full(n, np.nan)

        return pd.DataFrame(row, index=pd.DatetimeIndex(times))

    with ThreadPoolExecutor(max_workers=_N_READ_THREADS) as pool:
        dfs = list(pool.map(_one, files))
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs) if dfs else pd.DataFrame()


#: Dispatch table: dataset_key → fast cdflib reader function
_CDF_READERS = {
    "psp_mag": _read_psp_mag,
    "psp_spc": _read_psp_spc,
    "ace_mag": _read_ace_mag,
    "ace_swe": _read_ace_swe,
}


def _read_cdf_files(files: list[str], dataset_key: str) -> pd.DataFrame:
    """Read a list of CDF files for one dataset using cdflib."""
    reader = _CDF_READERS.get(dataset_key)
    if reader is not None:
        return reader(files)
    return _read_cdf_generic(files, dataset_key)


# ---------------------------------------------------------------------------
# Raw download
# ---------------------------------------------------------------------------

def fetch_raw(
    dataset_key: str,
    start: str,
    end: str,
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Download a CDAWeb dataset for [start, end] and return as a DataFrame.

    CDF files are downloaded in yearly chunks and cached locally by SunPy
    (under ``data_dir/{dataset_key}/``). Subsequent calls with the same
    arguments return quickly because SunPy detects the files on disk.

    Files are read with cdflib (not SunPy TimeSeries) for performance.
    Output column names match the SunPy TimeSeries convention so that the
    existing clean_* functions work unchanged.

    Parameters
    ----------
    dataset_key : str
        Key from ``swpipeline.config.DATASETS``
        (e.g. ``'psp_mag'``, ``'solo_swa_pas'``, ``'ace_swe'``).
    start, end : str
        ISO date strings for the desired time window.
    data_dir : str
        Root directory for CDF file cache.

    Returns
    -------
    pd.DataFrame
        Raw CDF data with column names matching SunPy TimeSeries output.
        Empty DataFrame if no data files are found.

    Raises
    ------
    KeyError
        If ``dataset_key`` is not in ``DATASETS``.
    """
    # Lazy imports so the module can be imported without SunPy installed
    # (useful for offline unit tests that never call fetch_raw directly).
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    ds_name = DATASETS[dataset_key]
    download_path = os.path.join(data_dir, dataset_key)
    os.makedirs(download_path, exist_ok=True)

    all_files: list[str] = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        result = Fido.search(
            a.Time(chunk_start, chunk_end),
            a.cdaweb.Dataset(ds_name),
        )
        if len(result) > 0:
            files = Fido.fetch(result, path=download_path)
            all_files.extend(files)

    if not all_files:
        return pd.DataFrame()

    return _read_cdf_files(sorted(all_files), dataset_key)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_spacecraft_dataset(
    sc: str,
    start: str,
    end: str,
    data_dir: str = "data",
    mag_df: pd.DataFrame | None = None,
    plasma_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Full download → clean → resample → merge pipeline for one spacecraft.

    Downloads both magnetic field and plasma data (unless pre-loaded
    DataFrames are supplied via ``mag_df`` / ``plasma_df``), applies
    instrument-specific quality filtering, resamples to 1-hour means
    (PSP/SolO only; ACE H2 is already hourly), and merges on the time index.

    Parameters
    ----------
    sc : str
        Spacecraft: ``'psp'``, ``'solo'``, or ``'ace'``.
    start, end : str
        ISO date strings for the desired time window.
    data_dir : str
        Root directory for CDF file cache (forwarded to ``fetch_raw``).
    mag_df : pd.DataFrame, optional
        Pre-loaded raw magnetic field DataFrame (as returned by
        ``SunPy TimeSeries.to_dataframe()``).  When supplied the download
        step for magnetic data is skipped.  Useful for offline testing.
    plasma_df : pd.DataFrame, optional
        Pre-loaded raw plasma DataFrame.  Same semantics as ``mag_df``.

    Returns
    -------
    pd.DataFrame
        Cleaned, hourly-averaged, merged DataFrame.

        PSP / SolO columns:
            ``B``, ``B_r``, ``B_t``, ``B_n`` [nT],
            ``N_p`` [cm⁻³], ``T_p`` [eV],
            ``V_r``, ``V_t``, ``V_n``, ``V_sw`` [km/s]

        ACE columns:
            ``B`` [nT], ``N_p`` [cm⁻³],
            ``T_p`` [eV, radial only], ``V_r``, ``V_t``, ``V_n``, ``V_sw`` [km/s]

        Returns an empty DataFrame if either mag or plasma data is empty
        after cleaning.

    Raises
    ------
    ValueError
        If ``sc`` is not ``'psp'``, ``'solo'``, or ``'ace'``.
    """
    if sc not in _SC_DATASETS:
        raise ValueError(
            f"Unknown spacecraft '{sc}'. Choose from: {sorted(_SC_DATASETS)}"
        )

    datasets = _SC_DATASETS[sc]

    # --- Download raw data (if not pre-supplied) ---
    if mag_df is None:
        mag_df = fetch_raw(datasets["mag"], start, end, data_dir)
    if plasma_df is None:
        plasma_df = fetch_raw(datasets["plasma"], start, end, data_dir)

    if mag_df.empty or plasma_df.empty:
        return pd.DataFrame()

    # --- Clean (instrument-specific quality filtering) ---
    mag_clean    = _CLEAN_FN[datasets["mag"]](mag_df)
    plasma_clean = _CLEAN_FN[datasets["plasma"]](plasma_df)

    if mag_clean.empty or plasma_clean.empty:
        return pd.DataFrame()

    # --- Resample to hourly (PSP / SolO only) ---
    if _NEEDS_RESAMPLE[sc]:
        mag_clean    = resample_to_hourly(mag_clean)
        plasma_clean = resample_to_hourly(plasma_clean)

    # --- Merge on time index (inner join) ---
    return merge_mag_plasma(mag_clean, plasma_clean)
