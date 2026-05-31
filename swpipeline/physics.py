"""
Derived plasma physics quantities.

All functions accept numpy arrays or pandas Series (broadcasting applies).
Units are always explicit: nT for B, cm^-3 for N, eV for T, km/s for V.

Physical constants are imported from config.PhysConst to ensure consistency
across the package.
"""

import numpy as np
import pandas as pd

from swpipeline.config import PhysConst


def plasma_beta(
    N_cm3: float | np.ndarray | pd.Series,
    T_eV:  float | np.ndarray | pd.Series,
    B_nT:  float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute the proton plasma beta: β_p = thermal pressure / magnetic pressure.

    β_p = 2μ₀ n k_B T / B²

    With N in cm^-3, T in eV, B in nT this simplifies to:
        β_p = BETA_PREFACTOR × N × T / B²
    where BETA_PREFACTOR ≈ 0.4022 (derived in config.PhysConst).

    β_p < 1  →  magnetically dominated (typical near the Sun)
    β_p > 1  →  thermally dominated (typical at 1 AU in slow wind)

    Note: This uses only proton contributions. True total beta also includes
    alpha particles and electrons, which are not available for all datasets.

    Parameters
    ----------
    N_cm3 : array-like
        Proton number density in cm^-3.
    T_eV : array-like
        Proton temperature in eV.
    B_nT : array-like
        Total magnetic field magnitude in nT.

    Returns
    -------
    array-like
        Dimensionless plasma beta β_p.
    """
    return PhysConst.BETA_PREFACTOR * N_cm3 * T_eV / (B_nT**2)


def alfven_speed(
    N_cm3: float | np.ndarray | pd.Series,
    B_nT:  float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute the proton Alfvén speed.

    V_A = B / √(μ₀ ρ)  where  ρ = N × m_p

    With N in cm^-3 and B in nT, the result is returned in km/s.

    V_A [km/s] = VA_PREFACTOR × B_nT / √(N_cm3)
    where VA_PREFACTOR ≈ 21.8 (derived in config.PhysConst).

    Parameters
    ----------
    N_cm3 : array-like
        Proton number density in cm^-3.
    B_nT : array-like
        Total magnetic field magnitude in nT.

    Returns
    -------
    array-like
        Alfvén speed in km/s.
    """
    return PhysConst.VA_PREFACTOR * B_nT / np.sqrt(N_cm3)


def alfven_mach_number(
    V_sw_kms: float | np.ndarray | pd.Series,
    N_cm3:    float | np.ndarray | pd.Series,
    B_nT:     float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Compute the Alfvénic Mach number M_A = V_sw / V_A.

    M_A < 1  →  sub-Alfvénic (plasma co-rotates with Sun; rare, near PSP perihelia)
    M_A > 1  →  super-Alfvénic (typical solar wind conditions beyond ~20 R_Sun)

    Parameters
    ----------
    V_sw_kms : array-like
        Solar wind bulk speed in km/s.
    N_cm3 : array-like
        Proton number density in cm^-3.
    B_nT : array-like
        Total magnetic field magnitude in nT.

    Returns
    -------
    array-like
        Dimensionless Alfvén Mach number.
    """
    return V_sw_kms / alfven_speed(N_cm3, B_nT)


def thermal_speed_to_temp_eV(
    wp_kms: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Convert PSP SPC proton radial thermal (most-probable) speed to temperature.

    T_p [eV] = (wp [m/s])² × m_p / (2 e)
             = wp_kms² × WP_TO_TEV

    where WP_TO_TEV = (1e3)² × m_p / (2e) ≈ 5.220 × 10⁻³ eV / (km/s)².

    Note: wp is the most-probable speed from the *reduced* (1D) VDF.
    It approximately equals √(2 k_B T_par / m_p), so T here is the
    parallel temperature component. For isotropic distributions T_par = T_total;
    for anisotropic fast wind, T_total > T_par.

    Parameters
    ----------
    wp_kms : array-like
        Proton radial thermal (most-probable) speed in km/s.

    Returns
    -------
    array-like
        Proton temperature in eV.
    """
    return wp_kms**2 * PhysConst.WP_TO_TEV


def kelvin_to_eV(
    T_K: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """
    Convert temperature from Kelvin to eV.

    T [eV] = T [K] × k_B / e = T [K] × 8.617333262 × 10⁻⁵ eV/K

    Used to convert ACE SWEPAM Tpr (Kelvin) to eV for cross-mission comparison.

    Parameters
    ----------
    T_K : array-like
        Temperature in Kelvin.

    Returns
    -------
    array-like
        Temperature in eV.
    """
    return T_K * PhysConst.K_to_eV


def compute_all_physics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all derived physics columns to a clean merged DataFrame.

    Requires input DataFrame to have columns:
        B [nT], N_p [cm^-3], T_p [eV], V_sw [km/s]

    Adds columns:
        beta_p  — proton plasma beta (dimensionless)
        V_A     — Alfvén speed (km/s)
        M_A     — Alfvén Mach number (dimensionless)

    Does not modify the input DataFrame (returns a copy).

    Parameters
    ----------
    df : pd.DataFrame
        Clean merged DataFrame with B, N_p, T_p, V_sw columns.

    Returns
    -------
    pd.DataFrame
        Copy of df with additional physics columns.
    """
    out = df.copy()
    out["beta_p"] = plasma_beta(out["N_p"], out["T_p"], out["B"])
    out["V_A"]    = alfven_speed(out["N_p"], out["B"])
    out["M_A"]    = alfven_mach_number(out["V_sw"], out["N_p"], out["B"])
    return out
