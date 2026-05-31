"""
Statistical utilities for solar wind radial profile analysis.

Functions
---------
bin_by_distance          — median + std binning by heliocentric distance
fit_power_law            — y = a * r^b via scipy curve_fit
radial_profile           — end-to-end bin + fit convenience wrapper
block_bootstrap_exponent — correlation-aware uncertainty on fitted exponent
flag_icme_proxy          — simple in-situ ICME/transient proxy flag
split_solar_cycle        — split a time-indexed DataFrame by solar cycle phase
classify_wind_speed      — label rows as slow / intermediate / fast

Units throughout:  distance in AU, B in nT, N in cm^-3, T in eV, V in km/s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_by_distance(
    df: pd.DataFrame,
    param: str,
    n_bins: int = 20,
    dist_range: tuple[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bin a solar wind parameter by heliocentric distance.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'dist' (heliocentric distance in AU) and *param* columns.
    param : str
        Column name to summarise.
    n_bins : int
        Number of equal-width distance bins.
    dist_range : (r_min, r_max) tuple, optional
        Restrict binning to this distance range.  Defaults to the full range
        of df['dist'].

    Returns
    -------
    median_df : pd.DataFrame
        Columns 'dist' (bin centre) and *param* (median).
        Bins with no data have NaN in *param*.
    std_df : pd.DataFrame
        Column *param* only (sample std, NaN for single-point bins or empty bins).
        Shares the same integer index as median_df so the two can be concat-ed.
    """
    if len(df) == 0:
        empty_med = pd.DataFrame(
            {"dist": pd.Series(dtype=float), param: pd.Series(dtype=float)}
        )
        empty_std = pd.DataFrame({param: pd.Series(dtype=float)})
        return empty_med, empty_std

    r_min, r_max = (
        (df["dist"].min(), df["dist"].max()) if dist_range is None else dist_range
    )

    edges = np.linspace(r_min, r_max, n_bins + 1)

    # Integer bin label (NaN for out-of-range rows)
    bin_idx = pd.cut(df["dist"], bins=edges, labels=False, include_lowest=True)

    dist_meds = np.full(n_bins, np.nan)
    medians = np.full(n_bins, np.nan)
    stds = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = bin_idx == i
        dist_vals = df.loc[mask, "dist"].dropna()
        param_vals = df.loc[mask, param].dropna()
        if len(dist_vals) >= 1:
            dist_meds[i] = dist_vals.median()
        if len(param_vals) >= 1:
            medians[i] = param_vals.median()
        if len(param_vals) >= 2:
            stds[i] = float(param_vals.std(ddof=1))

    median_df = pd.DataFrame({"dist": dist_meds, param: medians})
    std_df = pd.DataFrame({param: stds})
    return median_df, std_df


# ---------------------------------------------------------------------------
# Power-law fitting
# ---------------------------------------------------------------------------

def _power_law_model(r: np.ndarray, a: float, b: float) -> np.ndarray:
    """Vectorised power-law: y = a * r^b."""
    return a * np.asarray(r, dtype=float) ** b


def fit_power_law(
    r: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
) -> dict:
    """
    Fit y = a * r^b using non-linear least squares (scipy.optimize.curve_fit).

    Initial guesses are derived from a log-log linear regression so that the
    optimisation converges reliably even for extreme exponents.

    Parameters
    ----------
    r : array-like
        Independent variable (heliocentric distance, AU).  Must be > 0.
    y : array-like
        Dependent variable (e.g., median |B| in nT).
    yerr : array-like, optional
        1-sigma uncertainties on *y*.  When supplied, used as absolute weights
        (absolute_sigma=True).

    Returns
    -------
    dict with keys
        amplitude      — fitted amplitude a
        exponent       — fitted exponent b
        amplitude_err  — 1-sigma uncertainty on a  (≥ 0)
        exponent_err   — 1-sigma uncertainty on b  (≥ 0)
        r_squared      — coefficient of determination R²

    Raises
    ------
    ValueError
        If fewer than 3 data points are supplied.
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(r) < 3:
        raise ValueError(
            f"fit_power_law requires at least 3 data points; got {len(r)}."
        )

    # --- Seed from log-log linear regression ---
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(r)
        log_y = np.log(np.abs(y))
        valid = np.isfinite(log_r) & np.isfinite(log_y)

    if valid.sum() >= 2:
        coeffs = np.polyfit(log_r[valid], log_y[valid], 1)
        b0 = float(coeffs[0])
        a0 = float(np.exp(coeffs[1]))
    else:
        b0 = -1.5
        a0 = float(np.nanmedian(y))

    # --- Non-linear least squares ---
    kwargs: dict = {}
    if yerr is not None:
        kwargs["sigma"] = np.asarray(yerr, dtype=float)
        kwargs["absolute_sigma"] = True

    popt, pcov = curve_fit(
        _power_law_model, r, y, p0=[a0, b0], maxfev=10_000, **kwargs
    )

    # Uncertainties: clamp non-finite values (e.g., perfectly flat profile)
    perr_raw = np.sqrt(np.diag(pcov))
    perr = np.where(np.isfinite(perr_raw), np.abs(perr_raw), 0.0)

    # --- R² ---
    y_pred = _power_law_model(r, *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return {
        "amplitude":     float(popt[0]),
        "exponent":      float(popt[1]),
        "amplitude_err": float(perr[0]),
        "exponent_err":  float(perr[1]),
        "r_squared":     float(r_squared),
    }


# ---------------------------------------------------------------------------
# End-to-end radial profile
# ---------------------------------------------------------------------------

def radial_profile(
    df: pd.DataFrame,
    param: str,
    n_bins: int = 20,
    dist_range: tuple[float, float] | None = None,
) -> dict:
    """
    Compute the radial profile of a solar wind parameter: bin then fit.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'dist' and *param* columns.
    param : str
        Column to profile.
    n_bins : int
        Number of equal-width distance bins.
    dist_range : tuple, optional
        Distance range forwarded to bin_by_distance.

    Returns
    -------
    dict with keys
        fit    — dict from fit_power_law (amplitude, exponent, …)
        median — binned median DataFrame (columns: 'dist', *param*)
        std    — binned std DataFrame (column: *param*)
    """
    median_df, std_df = bin_by_distance(
        df, param, n_bins=n_bins, dist_range=dist_range
    )
    valid = median_df.dropna(subset=["dist", param])
    fit_result = fit_power_law(valid["dist"].values, valid[param].values)
    return {"fit": fit_result, "median": median_df, "std": std_df}


# ---------------------------------------------------------------------------
# Block bootstrap uncertainty on power-law exponent
# ---------------------------------------------------------------------------

def block_bootstrap_exponent(
    df: pd.DataFrame,
    param: str,
    n_bins: int = 20,
    block_days: float = 5.0,
    n_iter: int = 500,
    ci: float = 0.68,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Block bootstrap estimate of power-law exponent and amplitude uncertainty.

    Solar wind data are temporally autocorrelated (~2–5 day correlation time),
    so bin-scatter underestimates the true uncertainty on fitted exponents.
    This function resamples contiguous temporal blocks (rather than individual
    hourly points) to preserve autocorrelation structure.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and contain 'dist' and *param* columns.
    param : str
        Column to fit.
    n_bins : int
        Number of distance bins passed to radial_profile.
    block_days : float
        Duration of each bootstrap block in days (default 5.0).
    n_iter : int
        Number of bootstrap iterations (default 500).
    ci : float
        Confidence interval probability (default 0.68 → ±1σ equivalent).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict with keys
        exponent_med   — median exponent across bootstrap samples
        exponent_lo    — lower CI bound
        exponent_hi    — upper CI bound
        exponent_err   — symmetric half-width (hi - lo) / 2
        amplitude_med  — median amplitude
        n_iter         — number of successful iterations
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("block_bootstrap_exponent requires a DatetimeIndex.")

    sub = df[["dist", param]].dropna()
    if len(sub) < 20:
        raise ValueError(f"Too few rows ({len(sub)}) for block bootstrap.")

    # Build block labels: each block covers block_days consecutive days
    t0 = sub.index.min()
    block_id = ((sub.index - t0).total_seconds() / (block_days * 86400)).astype(int)
    unique_blocks = block_id.unique()
    n_blocks = len(unique_blocks)

    exponents = []
    amplitudes = []

    for _ in range(n_iter):
        chosen = rng.choice(unique_blocks, size=n_blocks, replace=True)
        chunks = [sub[block_id == b] for b in chosen if (block_id == b).any()]
        if not chunks:
            continue
        boot_df = pd.concat(chunks)
        if len(boot_df) < n_bins * 3:
            continue
        try:
            res = radial_profile(boot_df, param, n_bins=n_bins)
            exponents.append(res["fit"]["exponent"])
            amplitudes.append(res["fit"]["amplitude"])
        except Exception:
            continue

    if len(exponents) < 10:
        raise RuntimeError("Too few successful bootstrap iterations.")

    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    exp_arr = np.array(exponents)
    amp_arr = np.array(amplitudes)

    return {
        "exponent_med": float(np.median(exp_arr)),
        "exponent_lo":  float(np.quantile(exp_arr, lo_q)),
        "exponent_hi":  float(np.quantile(exp_arr, hi_q)),
        "exponent_err": float((np.quantile(exp_arr, hi_q) - np.quantile(exp_arr, lo_q)) / 2),
        "amplitude_med": float(np.median(amp_arr)),
        "n_iter": len(exponents),
    }


# ---------------------------------------------------------------------------
# Simple in-situ ICME / transient proxy flag
# ---------------------------------------------------------------------------

def flag_icme_proxy(
    df: pd.DataFrame,
    b_col: str = "B",
    tp_col: str = "T_p",
    b_residual_thresh: float = 3.0,
    b_expected_slope: float = -1.75,
    b_expected_norm: float = 7.0,
    tp_col_expected_slope: float = -0.81,
    tp_col_expected_norm: float = 10.0,
    tp_frac_thresh: float = 0.5,
) -> pd.Series:
    """
    Conservative in-situ proxy for ICME / large transient intervals.

    Flags an hour as 'potential_transient' when either:
    (a) |B| > b_residual_thresh × B_expected(r)  (distance-detrended B enhancement)
    (b) T_p < tp_frac_thresh × T_expected(r)  (cold plasma, classic ICME signature)

    Criterion (a) uses a power-law B_expected(r) = b_expected_norm × r^b_expected_slope
    rather than a rolling window, to avoid sensitivity to PSP's rapidly changing r.
    Default parameters (norm=7.0, slope=-1.75) are chosen to match the observed median
    B across PSP+SolO. Adjust for your specific dataset.

    This is NOT a replacement for a validated ICME catalog (Richardson & Cane,
    Nieves-Chinchilla for PSP, Möstl/ICMECAT for SolO). It is intended only for
    sensitivity analysis: compare results with and without flagged intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and 'dist' column.
    b_col : str
        Column name for total |B| (nT).
    tp_col : str
        Column name for proton temperature (eV).
    b_residual_thresh : float
        Flag when |B| / B_expected(r) > this threshold (default 3.0).
    b_expected_slope : float
        Power-law slope of ambient B(r) (default -1.75).
    b_expected_norm : float
        B amplitude at 1 AU for expected profile (nT, default 7.0).
    tp_col_expected_slope : float
        Power-law slope of ambient T_p(r) (default -0.81 from SolO fit).
    tp_col_expected_norm : float
        T_p at 1 AU (eV, default 10.0).
    tp_frac_thresh : float
        T_p / T_expected must be below this to flag (default 0.5).

    Returns
    -------
    pd.Series of bool
        True where the observation is flagged as potential transient.
        Index matches df.index.
    """
    flag = pd.Series(False, index=df.index)

    if "dist" not in df.columns:
        return flag

    r = df["dist"].clip(lower=0.01)

    # Criterion A: distance-detrended B enhancement
    if b_col in df.columns:
        b_expected = b_expected_norm * r ** b_expected_slope
        b_ratio    = df[b_col] / b_expected.clip(lower=1e-3)
        flag |= (b_ratio > b_residual_thresh)

    # Criterion B: cold plasma relative to expected T_p(r)
    if tp_col in df.columns:
        t_expected = tp_col_expected_norm * r ** tp_col_expected_slope
        flag |= (df[tp_col] < tp_frac_thresh * t_expected)

    return flag


# ---------------------------------------------------------------------------
# Solar cycle splitting
# ---------------------------------------------------------------------------

_DEFAULT_CYCLE_RANGES: dict[str, tuple[str, str]] = {
    "minimum":    ("2019-01-01", "2021-12-31"),
    "transition": ("2022-01-01", "2022-12-31"),
    "maximum":    ("2023-01-01", "2025-12-31"),
}


def split_solar_cycle(
    df: pd.DataFrame,
    date_ranges: dict[str, tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Split a time-indexed DataFrame by solar cycle phase.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex.
    date_ranges : dict, optional
        Keys 'minimum', 'transition', 'maximum'.
        Values are (start, end) strings (inclusive, ISO 8601).
        Defaults to Solar Cycle 25 reference periods:

            minimum    2019-01-01 – 2021-12-31  (solar minimum)
            transition 2022-01-01 – 2022-12-31  (rising phase)
            maximum    2023-01-01 – 2025-12-31  (solar maximum)

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'minimum', 'transition', 'maximum'.
    """
    ranges = _DEFAULT_CYCLE_RANGES if date_ranges is None else date_ranges
    return {
        phase: df.loc[
            (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        ]
        for phase, (start, end) in ranges.items()
    }


# ---------------------------------------------------------------------------
# Wind-speed classification
# ---------------------------------------------------------------------------

def classify_wind_speed(
    df: pd.DataFrame,
    slow_max: float = 400.0,
    fast_min: float = 600.0,
) -> pd.DataFrame:
    """
    Classify each row by solar wind bulk speed.

    Labels
    ------
    'slow'         V_sw < slow_max
    'intermediate' slow_max ≤ V_sw ≤ fast_min
    'fast'         V_sw > fast_min

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'V_sw' column (km/s).
    slow_max : float
        Upper boundary for slow wind (default 400 km/s).
    fast_min : float
        Lower boundary for fast wind (default 600 km/s).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added 'wind_class' string column.
        The input DataFrame is not modified.
    """
    out = df.copy()
    v = out["V_sw"]
    conditions = [
        v < slow_max,
        (v >= slow_max) & (v <= fast_min),
        v > fast_min,
    ]
    out["wind_class"] = np.select(conditions, ["slow", "intermediate", "fast"],
                                  default="intermediate")
    return out
