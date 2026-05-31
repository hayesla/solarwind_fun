"""
Configuration: dataset IDs, column mappings, physical constants, quality thresholds.

All column names match exactly what SunPy TimeSeries returns for each CDAWeb dataset.
Physical constants use SI and are converted explicitly to avoid unit errors.
"""

import numpy as np

# ---------------------------------------------------------------------------
# CDAWeb Dataset IDs
# ---------------------------------------------------------------------------
DATASETS = {
    "psp_mag":      "PSP_FLD_L2_MAG_RTN_1MIN",
    "psp_spc":      "PSP_SWP_SPC_L3I",
    "solo_mag":     "SOLO_L2_MAG-RTN-NORMAL-1-MINUTE",
    "solo_swa_pas": "SOLO_L2_SWA-PAS-GRND-MOM",
    "ace_mag":      "AC_H2_MFI",
    "ace_swe":      "AC_H2_SWE",
}

# ---------------------------------------------------------------------------
# Raw CDF column names (as returned by SunPy TimeSeries)
# ---------------------------------------------------------------------------
RAW_COLS = {
    "psp_mag": {
        "B_r":   "psp_fld_l2_mag_RTN_1min_0",
        "B_t":   "psp_fld_l2_mag_RTN_1min_1",
        "B_n":   "psp_fld_l2_mag_RTN_1min_2",
        "flag":  "psp_fld_l2_quality_flags",
    },
    "psp_spc": {
        "N_p":   "np_moment",     # proton density, cm^-3
        "w_p":   "wp_moment",     # proton thermal speed (most probable), km/s
        "V_r":   "vp_moment_RTN_0",
        "V_t":   "vp_moment_RTN_1",
        "V_n":   "vp_moment_RTN_2",
        "flag":  "general_flag",
    },
    "solo_mag": {
        "B_r":   "B_RTN_0",
        "B_t":   "B_RTN_1",
        "B_n":   "B_RTN_2",
        "flag":  "QUALITY_FLAG",
    },
    "solo_swa_pas": {
        "N_p":   "N",
        "T_p":   "T",             # proton temperature in eV (directly!)
        "V_r":   "V_RTN_0",
        "V_t":   "V_RTN_1",
        "V_n":   "V_RTN_2",
        "flag":  "quality_factor",
    },
    "ace_mag": {
        "B":     "Magnitude",     # |B| in nT; RTN components not available
        "flag":  "Q_FLAG",
    },
    "ace_swe": {
        "N_p":   "Np",            # proton density, cm^-3
        "T_p_K": "Tpr",           # radial proton temperature, Kelvin (not total!)
        "V_sw":  "Vp",            # bulk speed magnitude, km/s
        "V_r":   "V_RTN_0",
        "V_t":   "V_RTN_1",
        "V_n":   "V_RTN_2",
    },
}

# Standardised column names used throughout the package
STD_COLS = {
    "B_r":   "B_r",      # nT
    "B_t":   "B_t",      # nT
    "B_n":   "B_n",      # nT
    "B":     "B",        # nT  (total magnitude)
    "N_p":   "N_p",      # cm^-3
    "T_p":   "T_p",      # eV  (always converted to eV)
    "V_r":   "V_r",      # km/s  (RTN radial, positive = away from Sun)
    "V_t":   "V_t",      # km/s
    "V_n":   "V_n",      # km/s
    "V_sw":  "V_sw",     # km/s  (bulk speed magnitude)
    "dist":  "dist",     # AU   (heliocentric distance, added from trajectory)
    "lat":   "lat",      # deg  (heliographic latitude)
    "lon":   "lon",      # deg  (heliographic longitude, Stonyhurst)
}

# ---------------------------------------------------------------------------
# Quality thresholds (from CDF metadata inspection, documented in config)
# ---------------------------------------------------------------------------
QUALITY = {
    "psp_mag": {
        # flag == 0 means all quality bits are clear (good data)
        # non-zero values (1, 2, 256, 257 ...) indicate quality concerns
        "flag_good": 0,
    },
    "psp_spc": {
        # general_flag: 0=good, >0=bad, -1=unknown (default for all rows)
        # IMPORTANT: Almost all rows have general_flag > 0 even at perihelion.
        # Investigation shows this is dominated by DQF_0 which indicates SPC
        # measurement mode, not necessarily data invalidity.
        # Strategy: use physical validity bounds + NaN filter instead of strict flag.
        # Physical bounds from CDF VALIDMIN/VALIDMAX:
        "N_p_min":  0.01,     # cm^-3
        "N_p_max":  10000.0,  # cm^-3
        "w_p_min":  1.0,      # km/s
        "w_p_max":  1000.0,   # km/s
        "V_sw_min": 50.0,     # km/s  (solar wind must be outward flowing)
        "V_sw_max": 2000.0,   # km/s
    },
    "solo_mag": {
        # QUALITY_FLAG: 0=Bad, 1=Known problems, 2=Survey, 3=Good, 4=Excellent
        # VAR_NOTES: "3: Good for publication subject to PI approval"
        # Use >= 2 for statistical studies; note that >= 3 is publication quality
        "flag_min": 2,
    },
    "solo_swa_pas": {
        # quality_factor: float 0.0=perfect, higher=worse
        # Inspected data shows most good observations have quality_factor ~ 0
        "factor_max": 0.01,
        # Physical bounds from CDF VALIDMIN/VALIDMAX:
        "N_p_min":  0.0,
        "N_p_max":  10000.0,
        "T_p_min":  0.0,      # eV
        "T_p_max":  1e5,      # eV (practical upper limit)
    },
    "ace_mag": {
        # Q_FLAG encoding changed between v06 (2018-2019) and v07 (2020+):
        # v06 uses 0=good/1=maneuver/2=bad; v07 uses a 32-bit packed bitmask.
        # Filtering on flag_good=0 drops all 2020-2024 data (physically valid).
        # clean_ace_mag therefore filters on Magnitude > 0 instead of Q_FLAG.
        # This entry is retained for reference only and is not used in cleaning.
        "flag_good": 0,
    },
    "ace_swe": {
        # No explicit flag column in AC_H2_SWE; use NaN + physical bounds
        # Physical bounds from CDF VALIDMIN/VALIDMAX:
        "N_p_min":  0.0,
        "N_p_max":  200.0,    # cm^-3
        "T_p_K_min": 1000.0,  # Kelvin
        "T_p_K_max": 1.1e6,   # Kelvin
        "V_sw_min": 100.0,    # km/s
        "V_sw_max": 2000.0,   # km/s
    },
}

# ---------------------------------------------------------------------------
# Physical constants (SI unless otherwise stated)
# ---------------------------------------------------------------------------
class PhysConst:
    m_p   = 1.67262192369e-27  # kg  proton mass
    k_B   = 1.380649e-23       # J/K Boltzmann constant
    e     = 1.602176634e-19    # C   elementary charge (also J/eV)
    mu_0  = 4.0 * np.pi * 1e-7 # T·m/A  permeability of free space

    # Conversion factors
    K_to_eV   = k_B / e        # 8.617333262e-5 eV/K
    eV_to_K   = e / k_B        # 11604.52 K/eV

    # Plasma beta prefactor: beta = BETA_PREFACTOR * N_cm3 * T_eV / B_nT^2
    # Derived from: beta = 2*mu_0 * N*1e6 * T_eV*e / (B*1e-9)^2
    BETA_PREFACTOR = (
        2.0 * mu_0 * 1e6 * e / 1e-18
    )  # ≈ 0.4022, dimensionless when N in cm^-3, T in eV, B in nT

    # Alfvén speed prefactor: V_A [km/s] = VA_PREFACTOR * B_nT / sqrt(N_cm3)
    # From: V_A = B / sqrt(mu_0 * rho), rho = N * m_p
    VA_PREFACTOR = (
        1e-9 / np.sqrt(mu_0 * m_p * 1e6)
    ) / 1e3  # convert m/s -> km/s  ≈ 21.8

    # SPC thermal speed to temperature: T_eV = WP_TO_TEV * w_p_kms^2
    # From: T = (wp * 1000)^2 * m_p / (2 * e)
    WP_TO_TEV = (1e3)**2 * m_p / (2.0 * e)  # ≈ 5.220e-3 eV / (km/s)^2
