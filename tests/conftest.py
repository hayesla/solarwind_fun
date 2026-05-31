"""
Pytest fixtures: synthetic DataFrames that mimic the exact column structure
returned by SunPy TimeSeries for each CDAWeb dataset.

All mock values are physically plausible for solar wind at ~0.5 AU (PSP/SolO)
and ~1 AU (ACE), enabling physics calculation tests to be checked against
known analytical results.
"""

import numpy as np
import pandas as pd
import pytest


def _index(n, freq="1min", start="2022-06-15"):
    return pd.date_range(start, periods=n, freq=freq)


# ---------------------------------------------------------------------------
# PSP MAG  (PSP_FLD_L2_MAG_RTN_1MIN)
# Structure: Br, Bt, Bn in raw CDF names; quality flag column
# Note: real data has alternating NaN / value rows at 30-second cadence.
# ---------------------------------------------------------------------------
@pytest.fixture
def psp_mag_raw():
    """Raw PSP MAG DataFrame as returned by SunPy TimeSeries."""
    idx = _index(8, freq="30s")
    return pd.DataFrame({
        "psp_fld_l2_mag_RTN_1min_0": [-10.0, np.nan, -9.5, np.nan, -11.0, np.nan, np.nan, -8.0],
        "psp_fld_l2_mag_RTN_1min_1": [ 3.0,  np.nan,  3.2, np.nan,   2.8, np.nan, np.nan,  3.5],
        "psp_fld_l2_mag_RTN_1min_2": [ 1.0,  np.nan,  0.8, np.nan,   1.2, np.nan, np.nan,  0.9],
        # flag: 0=good, 256=quality concern, NaN for data rows
        "psp_fld_l2_quality_flags":  [np.nan, 0.0, np.nan, 256.0, np.nan, 0.0, 256.0, np.nan],
    }, index=idx)


# ---------------------------------------------------------------------------
# PSP SPC  (PSP_SWP_SPC_L3I)
# Structure: moments (np_moment, wp_moment, vp_moment_RTN_*), general_flag
# ---------------------------------------------------------------------------
@pytest.fixture
def psp_spc_raw():
    """Raw PSP SPC DataFrame as returned by SunPy TimeSeries."""
    idx = _index(6, freq="30s")
    return pd.DataFrame({
        # good rows: general_flag=0
        "np_moment":        [150.0, 120.0, 200.0, np.nan,  50.0,    0.005],
        "wp_moment":        [ 80.0,  75.0,  90.0, np.nan,  60.0,  800.0],
        "vp_moment_RTN_0":  [350.0, 320.0, 400.0, 380.0, 300.0,  380.0],
        "vp_moment_RTN_1":  [ 10.0,   8.0,  12.0,  11.0,   9.0,   10.0],
        "vp_moment_RTN_2":  [  3.0,   2.0,   4.0,   3.0,   2.5,    3.0],
        # flag: 0=good, 1=bad (but see config.py notes on SPC flag strategy)
        "general_flag":     [0, 0, 1, 0, 0, 0],
    }, index=idx)
    # Row 3: np_moment NaN  -> filtered by NaN check
    # Row 5: np_moment=0.005 below VALIDMIN=0.01 -> filtered by bounds
    # Row 5: wp_moment=800 above VALIDMAX=1000? No, 800 < 1000, keep
    # Row 2: general_flag=1 but np_moment valid (we use bounds, not flag)


# ---------------------------------------------------------------------------
# SolO MAG  (SOLO_L2_MAG-RTN-NORMAL-1-MINUTE)
# ---------------------------------------------------------------------------
@pytest.fixture
def solo_mag_raw():
    """Raw SolO MAG DataFrame as returned by SunPy TimeSeries."""
    idx = _index(5, freq="1min")
    return pd.DataFrame({
        "B_RTN_0":       [-5.0, -4.8, -5.2, -5.1, np.nan],
        "B_RTN_1":       [ 2.0,  2.1,  1.9,  2.0,  2.0],
        "B_RTN_2":       [ 0.5,  0.6,  0.4,  0.5,  0.5],
        "QUALITY_FLAG":  [3,     2,     1,    0,    3],
        "QUALITY_BITMASK": [168, 172, 136, 184, 168],
        "VECTOR_RANGE":      [0, 0, 0, 0, 0],
        "VECTOR_TIME_RESOLUTION": [0.016667]*5,
    }, index=idx)
    # Row 2: QUALITY_FLAG=1 (known problems) -> filtered out (< 2)
    # Row 3: QUALITY_FLAG=0 (bad) -> filtered out (< 2)


# ---------------------------------------------------------------------------
# SolO SWA-PAS  (SOLO_L2_SWA-PAS-GRND-MOM)
# ---------------------------------------------------------------------------
@pytest.fixture
def solo_swa_raw():
    """Raw SolO SWA-PAS DataFrame as returned by SunPy TimeSeries."""
    idx = _index(5, freq="4s")
    return pd.DataFrame({
        "N":               [5.0,   4.8,   6.0,  np.nan, 5.5],
        "T":               [20.0,  19.5,  22.0,   21.0,  18.0],  # eV
        "V_RTN_0":         [400.0, 390.0, 420.0,  410.0, 405.0],  # km/s
        "V_RTN_1":         [ 10.0,   8.0,  12.0,   11.0,  9.0],
        "V_RTN_2":         [  3.0,   2.0,   4.0,    3.0,  2.5],
        "TxTyTz_RTN_0":    [22.0,  21.0,  24.0,   23.0, 20.0],
        "TxTyTz_RTN_1":    [19.0,  18.5,  21.0,   20.0, 17.0],
        "TxTyTz_RTN_2":    [19.0,  18.5,  21.0,   20.0, 17.0],
        "quality_factor":  [0.0,   0.0,   0.005,   0.0,  0.5],  # row 4 bad
        "Half_interval":   [0.5]*5,
        "Info":            [0]*5,
        "total_count":     [17000]*5,
        "unrecovered_count": [18]*5,
    }, index=idx)
    # Row 3: N NaN -> filtered
    # Row 4: quality_factor=0.5 -> filtered (> 0.01 threshold)


# ---------------------------------------------------------------------------
# ACE MAG  (AC_H2_MFI) — already hourly, |B| only (no RTN components)
# ---------------------------------------------------------------------------
@pytest.fixture
def ace_mag_raw():
    """Raw ACE MAG DataFrame as returned by SunPy TimeSeries."""
    idx = _index(5, freq="1h")
    return pd.DataFrame({
        "BGSEc_0":    [-3.5, -3.2, -4.0, -3.8, -3.9],
        "BGSEc_1":    [ 0.6,  0.5,  0.7,  0.6,  0.8],
        "BGSEc_2":    [ 0.2,  0.3,  0.1,  0.2,  0.3],
        "BGSM_0":     [-3.4, -3.1, -3.9, -3.7, -3.8],
        "BGSM_1":     [ 0.7,  0.6,  0.8,  0.7,  0.9],
        "BGSM_2":     [ 0.2,  0.3,  0.1,  0.2,  0.3],
        "Magnitude":  [ 3.6,  3.3,  4.1,  3.9,  4.0],  # nT
        "Q_FLAG":     [0,     0,     1,    0,    2],     # v06: 1=maneuver, 2=bad; v07: packed bitmask
        "SC_pos_GSE_0": [1.5e6]*5,
        "SC_pos_GSE_1": [-5e4]*5,
        "SC_pos_GSE_2": [-1.5e5]*5,
        "SC_pos_GSM_0": [1.5e6]*5,
        "SC_pos_GSM_1": [-5e4]*5,
        "SC_pos_GSM_2": [-1.5e5]*5,
    }, index=idx)
    # Q_FLAG is no longer used for filtering (v07 bitmask encoding incompatible).
    # All 5 rows have positive Magnitude and should survive clean_ace_mag.


# ---------------------------------------------------------------------------
# ACE SWE  (AC_H2_SWE) — already hourly
# ---------------------------------------------------------------------------
@pytest.fixture
def ace_swe_raw():
    """Raw ACE SWE DataFrame as returned by SunPy TimeSeries."""
    idx = _index(5, freq="1h")
    return pd.DataFrame({
        "Np":       [5.0,   4.8,   6.0,  np.nan,  5.5],  # cm^-3
        "Tpr":      [7e4,   6.5e4, 8e4,   7.5e4,  6e4],  # Kelvin (radial)
        "Vp":       [450.0, 430.0, 480.0, 460.0,  0.5],  # km/s  (row 4 bad speed)
        "V_RTN_0":  [450.0, 430.0, 480.0, 460.0,  440.0],
        "V_RTN_1":  [  5.0,   4.0,   6.0,   5.0,    5.0],
        "V_RTN_2":  [  2.0,   1.5,   3.0,   2.0,    2.0],
        "alpha_ratio":  [0.04, 0.03, 0.05, 0.04, 0.04],
        "SC_pos_GSE_0": [1.5e6]*5,
        "SC_pos_GSE_1": [-5e4]*5,
        "SC_pos_GSE_2": [-1.5e5]*5,
        "SC_pos_GSM_0": [1.5e6]*5,
        "SC_pos_GSM_1": [-5e4]*5,
        "SC_pos_GSM_2": [-1.5e5]*5,
    }, index=idx)
    # Row 3: Np NaN -> filtered
    # Row 4: Vp=0.5 below V_sw_min=100 -> filtered


# ---------------------------------------------------------------------------
# Combined "clean" dataset fixture (as produced after process.py functions)
# Used for testing physics and stats modules.
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_df():
    """
    A fully cleaned, standardised DataFrame with trajectory info.
    Simulates the output of the full cleaning + trajectory pipeline.
    N_p in cm^-3, T_p in eV, B_* in nT, V_* in km/s, dist in AU.
    """
    n = 20
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h")
    dist = np.linspace(0.3, 0.9, n)  # AU
    return pd.DataFrame({
        "B_r":   -5.0 / dist**2 + rng.normal(0, 0.3, n),
        "B_t":    2.0 / dist**1.2 + rng.normal(0, 0.1, n),
        "B_n":    0.5 + rng.normal(0, 0.1, n),
        "B":      5.2 / dist**1.7 + rng.normal(0, 0.2, n),  # total |B|
        "N_p":    5.0 / dist**2 + rng.normal(0, 0.2, n),
        "T_p":   20.0 * dist**(-0.5) + rng.normal(0, 0.5, n),  # eV
        "V_r":  400.0 + rng.normal(0, 20, n),
        "V_t":   10.0 + rng.normal(0, 2, n),
        "V_n":    3.0 + rng.normal(0, 1, n),
        "V_sw":  400.0 + rng.normal(0, 20, n),
        "dist":  dist,
        "lat":   rng.uniform(-5, 5, n),
        "lon":   rng.uniform(0, 360, n),
    }, index=idx)
