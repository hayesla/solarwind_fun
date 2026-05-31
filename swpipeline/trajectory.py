"""
Spacecraft trajectory: heliocentric distance, latitude, and longitude.

Uses SPICE kernels (via SunPy) for PSP and Solar Orbiter.
ACE is fixed at L1 (~1.0 AU, 0° heliographic latitude).

SPICE kernel notes
------------------
Kernels are downloaded once and cached locally by SunPy (~50–200 MB total).
The first call to ``get_position()`` for a SPICE-based spacecraft triggers
the download; subsequent calls are fast (files already on disk).

To update to a newer trajectory kernel, edit the relevant URL in
``SPICE_KERNELS`` in this module and delete the cached file from
``~/.sunpy/data/``.

PSP kernel
    ``spp_nom_*_RO7.bsp`` — nominal (predicted) trajectory through 2025-08-31.
    For data beyond that date, replace with a newer predicted or reconstructed
    SPK file from:
    https://spdf.gsfc.nasa.gov/pub/data/psp/ephemeris/spice/ephemerides/

SolO kernel
    ``solo_ANC_soc-orbit-stp_20200210-20301120_280_V1_00288_V01.bsp`` —
    planned trajectory through 2030-11-20.
    Updated files available at ESA SPIFTP:
    http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SPICE kernel URLs  (override by editing these strings)
# ---------------------------------------------------------------------------

#: SPICE SPK kernel URLs for each spacecraft.
SPICE_KERNELS: dict[str, list[str]] = {
    "solo": [
        # Planetary ephemeris (needed to define the Sun's position)
        "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/de421.bsp",
        # SolO planned trajectory (2020-02-10 – 2030-11-20)
        (
            "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/"
            "solo_ANC_soc-orbit-stp_20200210-20301120_280_V1_00288_V01.bsp"
        ),
    ],
    "psp": [
        # Planetary ephemeris (shared with SolO; de421 is standard)
        "http://spiftp.esac.esa.int/data/SPICE/SOLAR-ORBITER/kernels/spk/de421.bsp",
        # PSP nominal trajectory (2018-08-12 – 2025-08-31)
        (
            "https://spdf.gsfc.nasa.gov/pub/data/psp/ephemeris/spice/ephemerides/"
            "spp_nom_20180812_20250831_v040_RO7.bsp"
        ),
    ],
}

#: SPICE body names (case-sensitive, must match kernel content).
_SPICE_BODY: dict[str, str] = {
    "solo": "Solar Orbiter",
    "psp":  "SOLAR PROBE PLUS",
}

#: Module-level set tracking which spacecraft kernels are loaded.
#: SPICE kernels persist for the lifetime of the Python process.
_kernels_loaded: set[str] = set()


# ---------------------------------------------------------------------------
# Kernel management
# ---------------------------------------------------------------------------

def load_spice_kernels(sc: str) -> None:
    """
    Download and initialise SPICE kernels for a spacecraft.

    Kernels are downloaded via SunPy's cache (stored in ``~/.sunpy/data/``).
    Already-loaded spacecraft are skipped — SPICE kernels persist in memory
    for the lifetime of the Python process, so duplicate loading is
    unnecessary and wasteful.

    Parameters
    ----------
    sc : str
        ``'psp'`` or ``'solo'``.
        ``'ace'`` is silently ignored (ACE uses a fixed L1 position).

    Raises
    ------
    ValueError
        If ``sc`` is not ``'ace'``, ``'psp'``, or ``'solo'``.
    """
    if sc == "ace":
        return  # ACE uses a fixed analytical position; no SPICE needed

    if sc in _kernels_loaded:
        return  # already initialised; skip to avoid duplicate loading

    if sc not in SPICE_KERNELS:
        raise ValueError(
            f"No SPICE kernels configured for '{sc}'. "
            f"Valid spacecraft: {sorted(SPICE_KERNELS)}"
        )

    # Lazy imports so the module can be used without SunPy in offline tests
    from sunpy.data import cache as sunpy_cache
    from sunpy.coordinates import spice

    kernel_files = [sunpy_cache.download(url) for url in SPICE_KERNELS[sc]]
    spice.initialize(kernel_files)
    _kernels_loaded.add(sc)


# ---------------------------------------------------------------------------
# Position calculation
# ---------------------------------------------------------------------------

def get_position(sc: str, times) -> pd.DataFrame:
    """
    Return heliocentric position for a spacecraft at the given times.

    For **PSP** and **SolO**: loads SPICE kernels on the first call and
    queries the Heliographic Stonyhurst (HGS) frame.

    For **ACE**: returns a fixed L1 position (1.0 AU, 0° lat, 0° lon).
    This is a reasonable approximation — ACE stays within ±0.01 AU of 1 AU
    on its Lissajous orbit around L1.

    Parameters
    ----------
    sc : str
        ``'psp'``, ``'solo'``, or ``'ace'``.
    times : array-like of datetime-like
        Timestamps at which to evaluate the position.

    Returns
    -------
    pd.DataFrame
        Columns:

        dist
            Heliocentric distance in AU.
        lat
            Heliographic latitude in degrees (HGS frame).
        lon
            Heliographic longitude in degrees (HGS, Stonyhurst).

        Index matches the input ``times`` (converted to ``DatetimeIndex``).
    """
    times_idx = pd.DatetimeIndex(times)
    n = len(times_idx)

    if sc == "ace":
        return pd.DataFrame(
            {
                "dist": np.ones(n),
                "lat":  np.zeros(n),
                "lon":  np.zeros(n),
            },
            index=times_idx,
        )

    load_spice_kernels(sc)

    # Lazy import of sunpy SPICE module
    import sunpy.coordinates.spice as sunpy_spice

    body = _SPICE_BODY[sc]
    sc_coord = sunpy_spice.get_body(body, times_idx)
    hgs = sc_coord.heliographic_stonyhurst

    return pd.DataFrame(
        {
            "dist": hgs.radius.to("AU").value,
            "lat":  hgs.lat.to("deg").value,
            "lon":  hgs.lon.to("deg").value,
        },
        index=times_idx,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def add_trajectory_columns(df: pd.DataFrame, sc: str) -> pd.DataFrame:
    """
    Add heliocentric distance, latitude, and longitude columns to a DataFrame.

    Calls ``get_position()`` using ``df.index`` as the observation times and
    appends the results as new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame with a ``DatetimeIndex``.
    sc : str
        ``'psp'``, ``'solo'``, or ``'ace'``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with three additional columns:

        dist
            Heliocentric distance in AU.
        lat
            Heliographic latitude in degrees.
        lon
            Heliographic longitude in degrees.

        The input DataFrame is not modified.
    """
    traj = get_position(sc, df.index)
    out = df.copy()
    out["dist"] = traj["dist"].values
    out["lat"]  = traj["lat"].values
    out["lon"]  = traj["lon"].values
    return out
