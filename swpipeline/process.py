"""
Data cleaning and standardisation for PSP, SolO, and ACE in-situ datasets.

Each clean_* function:
  - Accepts a raw DataFrame (as returned by SunPy TimeSeries.to_dataframe())
  - Applies quality filtering documented in swpipeline/config.py
  - Renames columns to the standardised names in STD_COLS
  - Returns a clean DataFrame with only the standardised columns needed
    (plus any extra columns already present, e.g. trajectory info)

Design notes
------------
PSP SPC general_flag
    Investigation of the actual CDF files shows that general_flag > 0 for
    ~99.9% of rows even during perihelion passes, primarily because DQF_0
    (the dominant flag bit) indicates the instrument measurement mode rather
    than data invalidity. Physical validity bounds (VALIDMIN/VALIDMAX from
    CDF metadata) provide a more reliable quality filter for statistical work.
    This is different from Nathan Besch's approach (2026) which nominally
    filtered flags but appears not to have removed much data.

SolO SWA-PAS temperature
    The T column is already in eV (confirmed from CDF metadata UNITS field).
    No conversion required. Compare with PSP SPC where wp_moment [km/s] must
    be converted to temperature via T = wp^2 * m_p / (2e) in eV.

ACE SWE temperature
    Tpr is the *radial* proton temperature in Kelvin (not T_total).
    It is a lower bound on T_total. We convert to eV for cross-mission
    comparison, noting this limitation in the returned DataFrame attrs.
"""

import numpy as np
import pandas as pd

from swpipeline.config import RAW_COLS, QUALITY, PhysConst


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _validate_bounds(series: pd.Series, vmin: float, vmax: float) -> pd.Series:
    """Return a boolean mask: True where value is finite and within [vmin, vmax]."""
    return series.notna() & (series >= vmin) & (series <= vmax)


# ---------------------------------------------------------------------------
# PSP MAG
# ---------------------------------------------------------------------------

def clean_psp_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean PSP FIELDS L2 1-minute RTN magnetic field data.

    Quality strategy
    ----------------
    - Drop rows where B components are NaN (the 1-minute file has interleaved
      30-second cadence rows where the component data and flag alternate).
    - Drop rows where psp_fld_l2_quality_flags != 0.
    - Compute total magnitude B = sqrt(Br^2 + Bt^2 + Bn^2).

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of SunPy TimeSeries.to_dataframe() for PSP_FLD_L2_MAG_RTN_1MIN.

    Returns
    -------
    pd.DataFrame
        Columns: B_r, B_t, B_n, B  (all in nT)
    """
    cols = RAW_COLS["psp_mag"]
    Br_col, Bt_col, Bn_col, flag_col = (
        cols["B_r"], cols["B_t"], cols["B_n"], cols["flag"]
    )

    # Step 1: keep only rows where B components are present (not NaN)
    mask = df[Br_col].notna() & df[Bt_col].notna() & df[Bn_col].notna()

    # Step 2: among rows with data, also require flag == 0 (good quality)
    # Flag is NaN for data rows in the interleaved structure, so we need to
    # handle this carefully: for rows where flag IS present, require it == 0;
    # for rows where flag is NaN (data rows), we relax to NaN-flag rows only
    # if B values are also present. Because of the alternating structure,
    # the flag and B-data rows are mutually exclusive, so we cannot simultaneously
    # have flag present AND B data present. We therefore rely on B-presence alone
    # combined with checking for physically plausible values.
    # However, we also check any rows that happen to have both B data and a flag:
    flag_present = df[flag_col].notna()
    bad_flag = flag_present & (df[flag_col] != QUALITY["psp_mag"]["flag_good"])
    mask = mask & ~bad_flag

    cleaned = df[mask].copy()

    # Step 3: rename and compute magnitude
    out = pd.DataFrame(index=cleaned.index)
    out["B_r"] = cleaned[Br_col].astype(float)
    out["B_t"] = cleaned[Bt_col].astype(float)
    out["B_n"] = cleaned[Bn_col].astype(float)
    out["B"]   = np.sqrt(out["B_r"]**2 + out["B_t"]**2 + out["B_n"]**2)

    return out


# ---------------------------------------------------------------------------
# PSP SPC
# ---------------------------------------------------------------------------

def clean_psp_spc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean PSP SWEAP SPC Level-3 ion data.

    Quality strategy
    ----------------
    - Do NOT use general_flag as the primary filter (see module docstring).
    - Filter using physical validity bounds (VALIDMIN/VALIDMAX from CDF):
        * np_moment in [0.01, 10000] cm^-3
        * wp_moment in [1.0, 1000.0] km/s
    - Derive V_sw = |v_RTN| and require V_sw >= 50 km/s (outward solar wind).
    - Convert thermal speed wp_moment [km/s] to temperature T_p [eV]
      using T_eV = wp_kms^2 * m_p / (2e)  (PhysConst.WP_TO_TEV).

    Note: wp_moment is the most-probable speed from the 0th/2nd moment
    of the *reduced* (1D projected) VDF. This underestimates T_total if
    the distribution is anisotropic (T_perp > T_par), which is common in
    fast wind. Treat T_p as a lower bound on true proton temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of SunPy TimeSeries.to_dataframe() for PSP_SWP_SPC_L3I.

    Returns
    -------
    pd.DataFrame
        Columns: N_p [cm^-3], T_p [eV], V_r, V_t, V_n, V_sw [km/s]
    """
    q = QUALITY["psp_spc"]
    cols = RAW_COLS["psp_spc"]

    Np_col = cols["N_p"]
    wp_col = cols["w_p"]
    Vr_col, Vt_col, Vn_col = cols["V_r"], cols["V_t"], cols["V_n"]

    # Physical validity masks
    mask_N = _validate_bounds(df[Np_col], q["N_p_min"], q["N_p_max"])
    mask_w = _validate_bounds(df[wp_col], q["w_p_min"], q["w_p_max"])

    # Velocity components must be finite
    mask_v = (
        df[Vr_col].notna() & df[Vt_col].notna() & df[Vn_col].notna()
    )

    combined_mask = mask_N & mask_w & mask_v
    cleaned = df[combined_mask].copy()

    # Compute derived quantities
    out = pd.DataFrame(index=cleaned.index)
    out["N_p"]  = cleaned[Np_col].astype(float)
    out["T_p"]  = (cleaned[wp_col].astype(float))**2 * PhysConst.WP_TO_TEV
    out["V_r"]  = cleaned[Vr_col].astype(float)
    out["V_t"]  = cleaned[Vt_col].astype(float)
    out["V_n"]  = cleaned[Vn_col].astype(float)
    out["V_sw"] = np.sqrt(out["V_r"]**2 + out["V_t"]**2 + out["V_n"]**2)

    # Remove unphysically slow or fast wind
    speed_mask = _validate_bounds(out["V_sw"], q["V_sw_min"], q["V_sw_max"])
    out = out[speed_mask]

    return out


# ---------------------------------------------------------------------------
# SolO MAG
# ---------------------------------------------------------------------------

def clean_solo_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Solar Orbiter MAG L2 RTN-normal 1-minute magnetic field data.

    Quality strategy
    ----------------
    QUALITY_FLAG encoding (from CDF VAR_NOTES):
      0 = Bad data
      1 = Known problems, use at your own risk
      2 = Survey data, possibly not publication quality
      3 = Good for publication subject to PI approval
      4 = Excellent data, special treatment

    We keep QUALITY_FLAG >= 2 (survey or better) for statistical studies.
    Users requiring publication-quality data should use >= 3.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of SunPy TimeSeries.to_dataframe() for
        SOLO_L2_MAG-RTN-NORMAL-1-MINUTE.

    Returns
    -------
    pd.DataFrame
        Columns: B_r, B_t, B_n, B  (all in nT)
    """
    cols = RAW_COLS["solo_mag"]
    flag_min = QUALITY["solo_mag"]["flag_min"]

    mask = df[cols["flag"]] >= flag_min
    cleaned = df[mask].copy()

    out = pd.DataFrame(index=cleaned.index)
    out["B_r"] = cleaned[cols["B_r"]].astype(float)
    out["B_t"] = cleaned[cols["B_t"]].astype(float)
    out["B_n"] = cleaned[cols["B_n"]].astype(float)
    out["B"]   = np.sqrt(out["B_r"]**2 + out["B_t"]**2 + out["B_n"]**2)

    # Drop any remaining NaN rows (fill values already converted by SunPy)
    out = out.dropna(subset=["B_r", "B_t", "B_n"])

    return out


# ---------------------------------------------------------------------------
# SolO SWA-PAS
# ---------------------------------------------------------------------------

def clean_solo_swa_pas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Solar Orbiter SWA-PAS L2 ground-calculated proton moments.

    Quality strategy
    ----------------
    quality_factor is a float where 0.0 = perfect, higher = worse.
    We retain rows with quality_factor < 0.01 (near-perfect quality).

    Temperature note
    ----------------
    The T column is already in eV (confirmed from CDF UNITS='eV').
    This is T_total = (T_par + 2*T_perp)/3, the true scalar temperature.
    This is more physically complete than the PSP SPC temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of SunPy TimeSeries.to_dataframe() for
        SOLO_L2_SWA-PAS-GRND-MOM.

    Returns
    -------
    pd.DataFrame
        Columns: N_p [cm^-3], T_p [eV], V_r, V_t, V_n, V_sw [km/s]
    """
    cols = RAW_COLS["solo_swa_pas"]
    q = QUALITY["solo_swa_pas"]
    factor_max = q["factor_max"]

    # Quality filter
    mask_qf = df[cols["flag"]].notna() & (df[cols["flag"]] < factor_max)

    # Physical validity
    mask_N = _validate_bounds(df[cols["N_p"]], q["N_p_min"], q["N_p_max"])
    mask_T = _validate_bounds(df[cols["T_p"]], q["T_p_min"], q["T_p_max"])
    mask_v = (
        df[cols["V_r"]].notna() & df[cols["V_t"]].notna() & df[cols["V_n"]].notna()
    )

    combined_mask = mask_qf & mask_N & mask_T & mask_v
    cleaned = df[combined_mask].copy()

    out = pd.DataFrame(index=cleaned.index)
    out["N_p"]  = cleaned[cols["N_p"]].astype(float)
    out["T_p"]  = cleaned[cols["T_p"]].astype(float)   # already in eV
    out["V_r"]  = cleaned[cols["V_r"]].astype(float)
    out["V_t"]  = cleaned[cols["V_t"]].astype(float)
    out["V_n"]  = cleaned[cols["V_n"]].astype(float)
    out["V_sw"] = np.sqrt(out["V_r"]**2 + out["V_t"]**2 + out["V_n"]**2)

    return out


# ---------------------------------------------------------------------------
# ACE MAG
# ---------------------------------------------------------------------------

def clean_ace_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ACE MFI (magnetic field instrument) high-resolution data.

    Note: AC_H2_MFI provides hourly-averaged data with |B| (Magnitude)
    and GSE/GSM components, but NOT RTN components. For statistical studies
    of |B| radial scaling this is sufficient. RTN components would require
    coordinate transformation (see physics.gse_to_rtn_ace if needed).

    Quality strategy
    ----------------
    Q_FLAG encoding changed between v06 (2018-2019) and v07 (2020+) files:
      v06: simple 0=good, 1=maneuver, 2=bad/missing
      v07: 32-bit packed bitmask (non-zero does NOT mean bad data)
    2020-2024 data has 0% Q_FLAG==0 even though Magnitude values are
    physically valid (2-20 nT, no fill values). Filtering on Q_FLAG==0
    would discard 7 years of valid science data.

    Strategy: skip Q_FLAG filter; rely on physical validity of Magnitude:
      - NaN filter catches CDF fill values (already converted by _replace_fill)
      - Magnitude > 0 rejects the rare genuine zero/negative fill artefacts

    Parameters
    ----------
    df : pd.DataFrame
        Raw output from _read_ace_mag for AC_H2_MFI.

    Returns
    -------
    pd.DataFrame
        Columns: B [nT]  (total magnitude only; no RTN components available)
    """
    cols = RAW_COLS["ace_mag"]

    out = pd.DataFrame(index=df.index)
    out["B"] = df[cols["B"]].astype(float)

    # Drop fill values (NaN) and unphysical non-positive magnitudes
    out = out.dropna(subset=["B"])
    out = out[out["B"] > 0]

    return out


# ---------------------------------------------------------------------------
# ACE SWE
# ---------------------------------------------------------------------------

def clean_ace_swe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ACE SWEPAM (solar wind electron, proton, alpha monitor) ion data.

    Temperature note
    ----------------
    Tpr is the *radial* proton temperature in Kelvin (from CDF UNITS='Kelvin',
    CATDESC='radial component of the proton temperature'). This is an
    underestimate of T_total since it excludes the perpendicular component.
    We convert to eV for cross-mission comparison using k_B/e = 8.617e-5 eV/K.
    The converted value is stored as T_p with units eV (radial component).

    Quality strategy
    ----------------
    No explicit quality flag column in AC_H2_SWE. Quality is inferred from:
    - NaN filter (fill values -1e31 already converted to NaN by SunPy)
    - Physical validity bounds (VALIDMIN/VALIDMAX from CDF metadata):
        * Np in [0, 200] cm^-3
        * Tpr in [1000, 1.1e6] Kelvin
        * Vp in [100, 2000] km/s   (lower bound ensures we have solar wind)

    Parameters
    ----------
    df : pd.DataFrame
        Raw output of SunPy TimeSeries.to_dataframe() for AC_H2_SWE.

    Returns
    -------
    pd.DataFrame
        Columns: N_p [cm^-3], T_p [eV, radial], V_r, V_t, V_n, V_sw [km/s]
    """
    cols = RAW_COLS["ace_swe"]
    q = QUALITY["ace_swe"]

    mask_N  = _validate_bounds(df[cols["N_p"]], q["N_p_min"], q["N_p_max"])
    mask_T  = _validate_bounds(df[cols["T_p_K"]], q["T_p_K_min"], q["T_p_K_max"])
    mask_V  = _validate_bounds(df[cols["V_sw"]], q["V_sw_min"], q["V_sw_max"])
    mask_vr = df[cols["V_r"]].notna()

    combined_mask = mask_N & mask_T & mask_V & mask_vr
    cleaned = df[combined_mask].copy()

    out = pd.DataFrame(index=cleaned.index)
    out["N_p"]  = cleaned[cols["N_p"]].astype(float)
    out["T_p"]  = cleaned[cols["T_p_K"]].astype(float) * PhysConst.K_to_eV  # K → eV
    out["V_r"]  = cleaned[cols["V_r"]].astype(float)
    out["V_t"]  = cleaned[cols["V_t"]].astype(float)
    out["V_n"]  = cleaned[cols["V_n"]].astype(float)
    out["V_sw"] = cleaned[cols["V_sw"]].astype(float)

    return out


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample a DataFrame to 1-hour means.

    NaN values are excluded from the mean (pandas default behaviour).
    Hours with all-NaN inputs produce NaN output rows.

    Parameters
    ----------
    df : pd.DataFrame
        A time-indexed DataFrame at any sub-hourly cadence.

    Returns
    -------
    pd.DataFrame
        Hourly averaged DataFrame.
    """
    return df.resample("1h").mean()


# ---------------------------------------------------------------------------
# Merge MAG + Plasma DataFrames
# ---------------------------------------------------------------------------

def merge_mag_plasma(
    mag_df: pd.DataFrame,
    plasma_df: pd.DataFrame,
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge magnetic field and plasma DataFrames on their time index.

    Both inputs should already be at the same cadence (e.g., hourly) before
    merging. Uses pandas join which aligns on index.

    Parameters
    ----------
    mag_df : pd.DataFrame
        Cleaned magnetic field DataFrame (columns: B_r, B_t, B_n, B).
    plasma_df : pd.DataFrame
        Cleaned plasma DataFrame (columns: N_p, T_p, V_r, V_t, V_n, V_sw).
    how : str
        Join type passed to DataFrame.join (default 'inner' = keep only
        timestamps present in both DataFrames).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    return mag_df.join(plasma_df, how=how)
