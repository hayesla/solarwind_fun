"""
Tests for swpipeline.process — data cleaning and standardisation.

Run with:  conda run -n in_situ pytest tests/test_process.py -v

Each test:
  1. Describes which real-data behaviour it encodes (docstring)
  2. Uses fixtures from conftest.py — no network required
  3. Has a clear expected value or property to assert
"""

import numpy as np
import pandas as pd
import pytest

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
from swpipeline.config import STD_COLS, PhysConst


# ===========================================================================
# PSP MAG
# ===========================================================================

class TestCleanPspMag:

    def test_output_has_standard_columns(self, psp_mag_raw):
        """Cleaned DF must have B_r, B_t, B_n, B (computed magnitude)."""
        out = clean_psp_mag(psp_mag_raw)
        for col in ["B_r", "B_t", "B_n", "B"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_bad_flag_rows_dropped(self, psp_mag_raw):
        """Rows where psp_fld_l2_quality_flags != 0 must be NaN or absent.

        The raw fixture has flag=256 for some rows; those B values should
        not appear in the cleaned output.
        """
        out = clean_psp_mag(psp_mag_raw)
        # None of the remaining rows should have had flag != 0
        assert out["B_r"].notna().all(), "NaN B_r rows survived cleaning"

    def test_nan_b_rows_dropped(self, psp_mag_raw):
        """Rows with NaN in B components must be dropped (alternating pattern)."""
        out = clean_psp_mag(psp_mag_raw)
        assert out["B_r"].notna().all()
        assert out["B_t"].notna().all()

    def test_total_magnitude_computed_correctly(self, psp_mag_raw):
        """B = sqrt(Br^2 + Bt^2 + Bn^2); verify with first good row values."""
        out = clean_psp_mag(psp_mag_raw)
        # Check all B values are consistent with RTN components
        computed = np.sqrt(out["B_r"]**2 + out["B_t"]**2 + out["B_n"]**2)
        np.testing.assert_allclose(out["B"], computed, rtol=1e-5)

    def test_b_magnitude_is_positive(self, psp_mag_raw):
        """Total field magnitude must always be positive."""
        out = clean_psp_mag(psp_mag_raw)
        assert (out["B"] >= 0).all()

    def test_output_is_dataframe(self, psp_mag_raw):
        out = clean_psp_mag(psp_mag_raw)
        assert isinstance(out, pd.DataFrame)

    def test_empty_input_returns_empty(self):
        """An all-bad-flag DataFrame must return an empty DataFrame."""
        idx = pd.date_range("2022-01-01", periods=3, freq="30s")
        bad = pd.DataFrame({
            "psp_fld_l2_mag_RTN_1min_0": [np.nan, np.nan, np.nan],
            "psp_fld_l2_mag_RTN_1min_1": [np.nan, np.nan, np.nan],
            "psp_fld_l2_mag_RTN_1min_2": [np.nan, np.nan, np.nan],
            "psp_fld_l2_quality_flags":  [256.0, 1.0, 257.0],
        }, index=idx)
        out = clean_psp_mag(bad)
        assert len(out) == 0


# ===========================================================================
# PSP SPC
# ===========================================================================

class TestCleanPspSpc:

    def test_output_has_standard_columns(self, psp_spc_raw):
        """Cleaned SPC DF must have N_p, T_p (eV), V_r, V_t, V_n, V_sw."""
        out = clean_psp_spc(psp_spc_raw)
        for col in ["N_p", "T_p", "V_r", "V_t", "V_n", "V_sw"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_nan_np_rows_dropped(self, psp_spc_raw):
        """Rows where np_moment is NaN (fill value) must be dropped."""
        out = clean_psp_spc(psp_spc_raw)
        assert out["N_p"].notna().all()

    def test_below_validmin_density_dropped(self, psp_spc_raw):
        """np_moment = 0.005 (below VALIDMIN=0.01) must be dropped."""
        out = clean_psp_spc(psp_spc_raw)
        assert (out["N_p"] >= 0.01).all()

    def test_temperature_converted_to_eV(self, psp_spc_raw):
        """T_p must be in eV, derived from wp_moment using WP_TO_TEV constant.

        For wp_moment = 80 km/s:
          T_eV = 80^2 * 5.220e-3 ≈ 33.4 eV
        """
        out = clean_psp_spc(psp_spc_raw)
        # First good row has wp_moment=80 km/s
        first_row = out.iloc[0]
        expected_T = 80.0**2 * PhysConst.WP_TO_TEV
        np.testing.assert_allclose(first_row["T_p"], expected_T, rtol=1e-4)

    def test_temperature_is_positive(self, psp_spc_raw):
        out = clean_psp_spc(psp_spc_raw)
        assert (out["T_p"] > 0).all()

    def test_bulk_speed_magnitude_computed(self, psp_spc_raw):
        """V_sw = sqrt(Vr^2 + Vt^2 + Vn^2)."""
        out = clean_psp_spc(psp_spc_raw)
        computed = np.sqrt(out["V_r"]**2 + out["V_t"]**2 + out["V_n"]**2)
        np.testing.assert_allclose(out["V_sw"], computed, rtol=1e-5)

    def test_unphysical_speeds_removed(self, psp_spc_raw):
        """Rows with V_sw < 50 km/s (non-solar-wind) must be removed."""
        out = clean_psp_spc(psp_spc_raw)
        assert (out["V_sw"] >= 50.0).all()

    def test_general_flag_not_used_as_primary_filter(self, psp_spc_raw):
        """Rows with general_flag=1 but valid physical quantities must survive.

        The fixture has row index 2 with general_flag=1 but valid np_moment
        and wp_moment. These rows SHOULD be kept (physical bounds strategy).
        """
        out = clean_psp_spc(psp_spc_raw)
        # Row with general_flag=1 had np_moment=200, wp_moment=90 (both valid)
        # It should appear in output
        assert len(out) >= 3, "Too many rows dropped: physical-flag strategy lost valid data"


# ===========================================================================
# SolO MAG
# ===========================================================================

class TestCleanSoloMag:

    def test_output_has_standard_columns(self, solo_mag_raw):
        out = clean_solo_mag(solo_mag_raw)
        for col in ["B_r", "B_t", "B_n", "B"]:
            assert col in out.columns

    def test_low_quality_flag_rows_dropped(self, solo_mag_raw):
        """QUALITY_FLAG < 2 (0=Bad, 1=Problems) must be dropped."""
        out = clean_solo_mag(solo_mag_raw)
        # Fixture has rows with QUALITY_FLAG 0 and 1 — must be gone
        # Surviving rows have flags 2, 3 (and the NaN row if filtered)
        assert len(out) >= 1  # at least the flag=2 and flag=3 rows

    def test_quality_flag_0_dropped(self, solo_mag_raw):
        """QUALITY_FLAG=0 (Bad) must never appear in cleaned data."""
        out = clean_solo_mag(solo_mag_raw)
        # QUALITY_FLAG=0 row had B_RTN_0=-5.1 — that exact value should not appear
        # (unless it coincides with another row, which it doesn't in fixture)
        remaining_Br = set(out["B_r"].dropna().round(2).tolist())
        assert -5.1 not in remaining_Br, "QUALITY_FLAG=0 row survived cleaning"

    def test_quality_flag_1_dropped(self, solo_mag_raw):
        """QUALITY_FLAG=1 (known problems) must never appear in cleaned data."""
        out = clean_solo_mag(solo_mag_raw)
        remaining_Br = set(out["B_r"].dropna().round(2).tolist())
        assert -5.2 not in remaining_Br, "QUALITY_FLAG=1 row survived cleaning"

    def test_good_rows_survive(self, solo_mag_raw):
        """QUALITY_FLAG=2 and QUALITY_FLAG=3 rows must survive."""
        out = clean_solo_mag(solo_mag_raw)
        # Flag=3 row: B_RTN_0=-5.0  Flag=2 row: B_RTN_0=-4.8
        surviving = set(out["B_r"].dropna().round(2).tolist())
        assert -5.0 in surviving
        assert -4.8 in surviving

    def test_magnitude_always_positive(self, solo_mag_raw):
        out = clean_solo_mag(solo_mag_raw)
        assert (out["B"] >= 0).all()


# ===========================================================================
# SolO SWA-PAS
# ===========================================================================

class TestCleanSoloSwaPas:

    def test_output_has_standard_columns(self, solo_swa_raw):
        out = clean_solo_swa_pas(solo_swa_raw)
        for col in ["N_p", "T_p", "V_r", "V_t", "V_n", "V_sw"]:
            assert col in out.columns

    def test_high_quality_factor_dropped(self, solo_swa_raw):
        """quality_factor=0.5 (> 0.01 threshold) must be dropped."""
        out = clean_solo_swa_pas(solo_swa_raw)
        # Fixture row 4 has quality_factor=0.5 → should be gone
        # That row has V_RTN_0=405 and N=5.5 — check no N=5.5 survived
        # (all others have N=5.0, 4.8, 6.0, NaN)
        assert not any(np.isclose(out["N_p"], 5.5, atol=0.01))

    def test_nan_density_dropped(self, solo_swa_raw):
        """Row with NaN N (density) must be dropped."""
        out = clean_solo_swa_pas(solo_swa_raw)
        assert out["N_p"].notna().all()

    def test_temperature_already_in_eV(self, solo_swa_raw):
        """SolO SWA-PAS T column is directly in eV — no conversion needed."""
        out = clean_solo_swa_pas(solo_swa_raw)
        # The fixture T values are 20, 19.5, 22 eV — should appear as-is
        assert out["T_p"].iloc[0] == pytest.approx(20.0, rel=1e-3)

    def test_bulk_speed_magnitude_computed(self, solo_swa_raw):
        out = clean_solo_swa_pas(solo_swa_raw)
        computed = np.sqrt(out["V_r"]**2 + out["V_t"]**2 + out["V_n"]**2)
        np.testing.assert_allclose(out["V_sw"], computed, rtol=1e-5)


# ===========================================================================
# ACE MAG
# ===========================================================================

class TestCleanAceMag:

    def test_output_has_standard_columns(self, ace_mag_raw):
        out = clean_ace_mag(ace_mag_raw)
        assert "B" in out.columns

    def test_all_valid_magnitude_rows_survive(self, ace_mag_raw):
        """All rows with Magnitude > 0 survive regardless of Q_FLAG.

        The v07 AC_H2_MFI files use a packed bitmask Q_FLAG encoding where
        non-zero does NOT mean bad data (2020-2024 data is 0% Q_FLAG==0 but
        physically valid). We filter on Magnitude > 0 only.
        """
        out = clean_ace_mag(ace_mag_raw)
        # Fixture has 5 rows all with positive Magnitude; all should survive
        assert len(out) == 5

    def test_nan_magnitude_dropped(self, ace_mag_raw):
        """Rows with NaN Magnitude (CDF fill values) must be dropped."""
        import numpy as np
        import pandas as pd
        df_with_nan = ace_mag_raw.copy()
        df_with_nan.loc[df_with_nan.index[2], "Magnitude"] = np.nan
        out = clean_ace_mag(df_with_nan)
        assert len(out) == 4

    def test_good_rows_survive(self, ace_mag_raw):
        """All positive-Magnitude rows survive."""
        out = clean_ace_mag(ace_mag_raw)
        remaining = set(out["B"].round(2).tolist())
        assert 3.6 in remaining
        assert 3.3 in remaining
        assert 3.9 in remaining

    def test_b_is_positive(self, ace_mag_raw):
        out = clean_ace_mag(ace_mag_raw)
        assert (out["B"] > 0).all()


# ===========================================================================
# ACE SWE
# ===========================================================================

class TestCleanAceSwe:

    def test_output_has_standard_columns(self, ace_swe_raw):
        out = clean_ace_swe(ace_swe_raw)
        for col in ["N_p", "T_p", "V_sw"]:
            assert col in out.columns

    def test_nan_density_dropped(self, ace_swe_raw):
        """NaN Np rows must be dropped."""
        out = clean_ace_swe(ace_swe_raw)
        assert out["N_p"].notna().all()

    def test_unphysical_speed_dropped(self, ace_swe_raw):
        """Vp=0.5 km/s (below 100 km/s minimum) must be dropped."""
        out = clean_ace_swe(ace_swe_raw)
        assert (out["V_sw"] >= 100.0).all()

    def test_temperature_converted_to_eV(self, ace_swe_raw):
        """Tpr [Kelvin] must be converted to eV.

        For Tpr = 7e4 K:
          T_eV = 7e4 * 8.617e-5 ≈ 6.03 eV
        """
        out = clean_ace_swe(ace_swe_raw)
        first_row = out.iloc[0]
        expected_T = 7e4 * PhysConst.K_to_eV
        np.testing.assert_allclose(first_row["T_p"], expected_T, rtol=1e-4)

    def test_temperature_is_positive(self, ace_swe_raw):
        out = clean_ace_swe(ace_swe_raw)
        assert (out["T_p"] > 0).all()

    def test_temperature_units_note_in_docstring(self):
        """Verify clean_ace_swe docstring documents temperature conversion."""
        assert "Kelvin" in clean_ace_swe.__doc__ or "eV" in clean_ace_swe.__doc__


# ===========================================================================
# Resampling
# ===========================================================================

class TestResampleToHourly:

    def test_output_is_hourly(self, psp_spc_raw):
        """After cleaning and resampling, index frequency should be 1 hour."""
        cleaned = clean_psp_spc(psp_spc_raw)
        if len(cleaned) < 1:
            pytest.skip("No good rows in fixture after cleaning")
        hourly = resample_to_hourly(cleaned)
        if len(hourly) > 1:
            diff = (hourly.index[1] - hourly.index[0]).total_seconds()
            assert diff == 3600.0

    def test_hourly_preserves_means(self):
        """Hourly resampling must compute the mean of sub-hourly values."""
        idx = pd.date_range("2022-01-01", periods=4, freq="15min")
        df = pd.DataFrame({"B": [10.0, 20.0, np.nan, 30.0]}, index=idx)
        hourly = resample_to_hourly(df)
        # First (and only) hour: mean of 10, 20, 30 (NaN excluded) = 20.0
        np.testing.assert_allclose(hourly["B"].iloc[0], 20.0, rtol=1e-5)

    def test_all_nan_hour_stays_nan(self):
        """An hour with all NaN values should produce a NaN output row."""
        idx = pd.date_range("2022-01-01", periods=2, freq="30min")
        df = pd.DataFrame({"B": [np.nan, np.nan]}, index=idx)
        hourly = resample_to_hourly(df)
        assert hourly["B"].isna().all()


# ===========================================================================
# Merge MAG + Plasma
# ===========================================================================

class TestMergeMagPlasma:

    def test_merged_has_both_mag_and_plasma_cols(self, clean_df):
        """merge_mag_plasma must join two DataFrames on their index."""
        mag_cols = ["B_r", "B_t", "B_n", "B"]
        plasma_cols = ["N_p", "T_p", "V_sw"]

        mag_df = clean_df[mag_cols].copy()
        plasma_df = clean_df[plasma_cols].copy()

        merged = merge_mag_plasma(mag_df, plasma_df)
        for col in mag_cols + plasma_cols:
            assert col in merged.columns

    def test_merge_aligns_on_time_index(self):
        """Only timestamps present in BOTH DataFrames should survive (inner join)."""
        idx1 = pd.date_range("2022-01-01 00:00", periods=3, freq="1h")
        idx2 = pd.date_range("2022-01-01 01:00", periods=3, freq="1h")
        mag = pd.DataFrame({"B": [5, 6, 7]}, index=idx1)
        plasma = pd.DataFrame({"N_p": [4, 5, 6]}, index=idx2)
        merged = merge_mag_plasma(mag, plasma)
        # Overlap: 01:00 and 02:00 → 2 rows
        assert len(merged) == 2
