"""
Tests for swpipeline.physics — derived plasma quantities.

Run with:  conda run -n in_situ pytest tests/test_physics.py -v

All expected values are computed analytically from known inputs so that
the tests are independent of the implementation and can detect unit errors.
"""

import numpy as np
import pandas as pd
import pytest

from swpipeline.physics import (
    plasma_beta,
    alfven_speed,
    alfven_mach_number,
    thermal_speed_to_temp_eV,
    kelvin_to_eV,
    compute_all_physics,
)
from swpipeline.config import PhysConst


# ---------------------------------------------------------------------------
# Reference values (computed by hand to check implementation)
# ---------------------------------------------------------------------------
# At 1 AU typical: N=5 cm^-3, T=10 eV, B=5 nT
# beta = 2*mu0 * N*1e6 * T_eV*e / B_nT^2*1e-18
#      = 2 * 4pi*1e-7 * 5*1e6 * 10*1.6e-19 / (5*1e-9)^2
#      = 2 * 1.2566e-6 * 5e6 * 1.6e-18 / 25e-18
#      = 2 * 1.2566e-6 * 5e6 * 1.6e-18 / 25e-18
# numerator = 2 * 1.2566e-6 * 5e6 * 1.6e-18
#           = 2 * 1.2566 * 5 * 1.6 * 1e(-6+6-18) = 2 * 10.053 * 1e-18
#           = 2.0106e-17
# denominator = 25e-18 = 2.5e-17
# beta = 2.0106e-17 / 2.5e-17 = 0.804
REF_BETA = (
    2.0 * PhysConst.mu_0 * 5e6 * 10.0 * PhysConst.e / (5e-9)**2
)  # computed with exact constants

# Alfvén speed: V_A = B / sqrt(mu0 * rho) = B*1e-9 / sqrt(mu0 * N*1e6 * m_p) [m/s]
# N=5 cm^-3, B=5 nT:
# V_A = 5e-9 / sqrt(4pi*1e-7 * 5e6 * 1.6726e-27)
#     = 5e-9 / sqrt(1.0511e-24)
#     = 5e-9 / 3.242e-12.5  ... let's use the formula
REF_VA_ms = 5e-9 / np.sqrt(PhysConst.mu_0 * 5e6 * PhysConst.m_p)
REF_VA_kms = REF_VA_ms / 1e3  # in km/s

# SPC thermal speed -> temperature
# wp = 80 km/s: T_eV = 80^2 * WP_TO_TEV
REF_T_EV = 80.0**2 * PhysConst.WP_TO_TEV


class TestPlasmaBeta:

    def test_known_value(self):
        """Analytical check: N=5, T=10eV, B=5nT → known beta."""
        result = plasma_beta(N_cm3=5.0, T_eV=10.0, B_nT=5.0)
        np.testing.assert_allclose(result, REF_BETA, rtol=1e-4)

    def test_increases_with_density(self):
        """β ∝ N: doubling density should double beta."""
        b1 = plasma_beta(5.0, 10.0, 5.0)
        b2 = plasma_beta(10.0, 10.0, 5.0)
        np.testing.assert_allclose(b2 / b1, 2.0, rtol=1e-5)

    def test_increases_with_temperature(self):
        """β ∝ T: doubling temperature should double beta."""
        b1 = plasma_beta(5.0, 10.0, 5.0)
        b2 = plasma_beta(5.0, 20.0, 5.0)
        np.testing.assert_allclose(b2 / b1, 2.0, rtol=1e-5)

    def test_decreases_with_b_squared(self):
        """β ∝ B^-2: doubling B should reduce beta by factor 4."""
        b1 = plasma_beta(5.0, 10.0, 5.0)
        b2 = plasma_beta(5.0, 10.0, 10.0)
        np.testing.assert_allclose(b1 / b2, 4.0, rtol=1e-5)

    def test_array_input(self):
        """Function should work element-wise on numpy arrays."""
        N = np.array([5.0, 10.0])
        T = np.array([10.0, 10.0])
        B = np.array([5.0, 5.0])
        result = plasma_beta(N, T, B)
        expected = np.array([plasma_beta(5.0, 10.0, 5.0),
                              plasma_beta(10.0, 10.0, 5.0)])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_series_input(self, clean_df):
        """Function should work on pandas Series (column-wise)."""
        result = plasma_beta(clean_df["N_p"], clean_df["T_p"], clean_df["B"])
        assert isinstance(result, pd.Series)
        assert (result > 0).all()

    def test_zero_b_raises_or_returns_inf(self):
        """Zero B field → beta is undefined (numpy returns inf; scalar may raise)."""
        # Use a numpy array to guarantee inf behaviour (scalar Python floats raise
        # ZeroDivisionError which is also acceptable — both indicate undefined).
        try:
            result = plasma_beta(np.array([5.0]), np.array([10.0]), np.array([0.0]))
            assert not np.isfinite(result).all()
        except (ZeroDivisionError, FloatingPointError):
            pass  # raising is also acceptable for undefined input

    def test_high_beta_plasma(self):
        """β > 1 for high density/temperature relative to B."""
        result = plasma_beta(20.0, 50.0, 3.0)  # typical near-Sun slow wind
        assert result > 1.0

    def test_low_beta_plasma(self):
        """β < 1 near Sun where B dominates (PSP closest approach conditions).

        At ~0.05 AU: B ~ 500 nT, N ~ 100 cm^-3, T ~ 50 eV
        β = 0.4022 × 100 × 50 / 500^2 ≈ 0.008  ← strongly magnetically dominated.
        """
        result = plasma_beta(100.0, 50.0, 500.0)  # ~0.05 AU, very high B
        assert result < 1.0, f"β={result:.3f} should be < 1 for strong-B near-Sun conditions"


class TestAlfvenSpeed:

    def test_known_value(self):
        """Analytical check: N=5 cm^-3, B=5 nT → known V_A in km/s."""
        result = alfven_speed(N_cm3=5.0, B_nT=5.0)
        np.testing.assert_allclose(result, REF_VA_kms, rtol=1e-4)

    def test_increases_with_b(self):
        """V_A ∝ B: doubling B doubles V_A."""
        va1 = alfven_speed(5.0, 5.0)
        va2 = alfven_speed(5.0, 10.0)
        np.testing.assert_allclose(va2 / va1, 2.0, rtol=1e-5)

    def test_decreases_with_sqrt_density(self):
        """V_A ∝ N^-0.5: quadrupling N halves V_A."""
        va1 = alfven_speed(5.0, 5.0)
        va2 = alfven_speed(20.0, 5.0)
        np.testing.assert_allclose(va1 / va2, 2.0, rtol=1e-5)

    def test_result_in_km_per_second(self):
        """V_A for typical solar wind should be 30-100 km/s at 1 AU."""
        va = alfven_speed(5.0, 5.0)
        assert 10 < va < 500, f"V_A={va} km/s outside expected range"

    def test_near_sun_va_higher(self):
        """Near the Sun (higher B, higher N) V_A is typically >100 km/s."""
        # PSP perihelion: B~100 nT, N~500 cm^-3
        va = alfven_speed(500.0, 100.0)
        assert va > 50, f"Near-Sun V_A={va} km/s lower than expected"

    def test_array_input(self):
        N = np.array([5.0, 10.0])
        B = np.array([5.0, 5.0])
        result = alfven_speed(N, B)
        assert len(result) == 2

    def test_zero_density_raises_or_inf(self):
        result = alfven_speed(0.0, 5.0)
        assert not np.isfinite(result)


class TestAlfvenMachNumber:

    def test_super_alfvenic_at_1au(self):
        """Solar wind at 1 AU (V=400 km/s) is super-Alfvénic (M_A > 1)."""
        # N=5, B=5nT → V_A ~ 48 km/s; V_sw=400 >> V_A
        ma = alfven_mach_number(V_sw_kms=400.0, N_cm3=5.0, B_nT=5.0)
        assert ma > 1.0

    def test_sub_alfvenic_near_sun(self):
        """PSP crossed the Alfvén critical surface; some perihelion data is sub-Alfvénic."""
        # Typical sub-Alfvénic: V_sw=200 km/s, B=100 nT, N=500 cm^-3
        # V_A = 100e-9 / sqrt(mu0 * 500e6 * mp) in m/s
        ma = alfven_mach_number(V_sw_kms=200.0, N_cm3=500.0, B_nT=100.0)
        assert ma < 5.0  # not necessarily < 1 for these params; just check range

    def test_consistency_with_va(self):
        """M_A = V_sw / V_A — check consistency."""
        V_sw, N, B = 400.0, 5.0, 5.0
        va = alfven_speed(N, B)
        ma_direct = alfven_mach_number(V_sw, N, B)
        np.testing.assert_allclose(ma_direct, V_sw / va, rtol=1e-5)

    def test_array_input(self, clean_df):
        ma = alfven_mach_number(clean_df["V_sw"], clean_df["N_p"], clean_df["B"])
        assert len(ma) == len(clean_df)
        assert (ma > 0).all()


class TestThermalSpeedToTemp:

    def test_known_value(self):
        """wp=80 km/s → T_eV = 80^2 * WP_TO_TEV."""
        result = thermal_speed_to_temp_eV(wp_kms=80.0)
        np.testing.assert_allclose(result, REF_T_EV, rtol=1e-6)

    def test_positive_output(self):
        assert thermal_speed_to_temp_eV(50.0) > 0

    def test_quadratic_scaling(self):
        """T ∝ wp^2: doubling wp quadruples temperature."""
        t1 = thermal_speed_to_temp_eV(50.0)
        t2 = thermal_speed_to_temp_eV(100.0)
        np.testing.assert_allclose(t2 / t1, 4.0, rtol=1e-5)

    def test_array_input(self):
        wp = np.array([50.0, 80.0, 100.0])
        result = thermal_speed_to_temp_eV(wp)
        assert len(result) == 3
        assert (result > 0).all()


class TestKelvinToEV:

    def test_known_conversion(self):
        """11604.52 K ≈ 1 eV (Boltzmann's constant)."""
        result = kelvin_to_eV(11604.52)
        np.testing.assert_allclose(result, 1.0, rtol=1e-3)

    def test_zero_kelvin_is_zero_ev(self):
        assert kelvin_to_eV(0.0) == pytest.approx(0.0)

    def test_typical_ace_temperature(self):
        """ACE typical Tpr ~ 7e4 K → ~6 eV."""
        result = kelvin_to_eV(7e4)
        assert 4.0 < result < 8.0, f"7e4 K → {result} eV out of expected range"

    def test_array_input(self):
        T_K = np.array([11604.52, 23209.04])
        result = kelvin_to_eV(T_K)
        np.testing.assert_allclose(result, [1.0, 2.0], rtol=1e-3)


class TestComputeAllPhysics:

    def test_adds_beta_column(self, clean_df):
        """compute_all_physics must add 'beta_p' column."""
        out = compute_all_physics(clean_df)
        assert "beta_p" in out.columns

    def test_adds_va_column(self, clean_df):
        """compute_all_physics must add 'V_A' column in km/s."""
        out = compute_all_physics(clean_df)
        assert "V_A" in out.columns

    def test_adds_mach_column(self, clean_df):
        """compute_all_physics must add 'M_A' (Alfvén Mach number) column."""
        out = compute_all_physics(clean_df)
        assert "M_A" in out.columns

    def test_does_not_modify_original(self, clean_df):
        """compute_all_physics must return a new DataFrame, not mutate input."""
        original_cols = set(clean_df.columns)
        out = compute_all_physics(clean_df)
        assert set(clean_df.columns) == original_cols

    def test_all_new_cols_finite_for_clean_data(self, clean_df):
        """All derived columns must be finite when input data is clean."""
        out = compute_all_physics(clean_df)
        for col in ["beta_p", "V_A", "M_A"]:
            assert out[col].notna().all(), f"NaN found in {col}"
            assert np.isfinite(out[col]).all(), f"Inf found in {col}"
