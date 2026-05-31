"""
Tests for swpipeline.stats — binning, power law fitting, radial profiles.

Run with:  conda run -n sw_pipeline pytest tests/test_stats.py -v
"""

import numpy as np
import pandas as pd
import pytest

from swpipeline.stats import (
    bin_by_distance,
    fit_power_law,
    radial_profile,
    split_solar_cycle,
    classify_wind_speed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_power_law_df():
    """DataFrame whose B column follows exactly B = 5 * r^-1.73 (no noise)."""
    r = np.linspace(0.1, 1.0, 50)
    df = pd.DataFrame({
        "dist": r,
        "B":    5.0 * r**(-1.73),
        "N_p":  6.0 * r**(-2.0),
        "T_p":  20.0 * r**(-0.5),
        "V_sw": 400.0 * np.ones(50),  # flat radial profile
    }, index=pd.date_range("2022-01-01", periods=50, freq="1h"))
    return df


@pytest.fixture
def noisy_df(simple_power_law_df):
    """Same as simple_power_law_df but with 10% Gaussian noise on B."""
    df = simple_power_law_df.copy()
    rng = np.random.default_rng(0)
    df["B"] *= 1.0 + rng.normal(0, 0.1, len(df))
    return df


@pytest.fixture
def solar_cycle_df():
    """DataFrame spanning 2019-2025 to test solar cycle splitting."""
    idx = pd.date_range("2019-01-01", "2025-12-31", freq="ME")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "dist": rng.uniform(0.1, 1.0, len(idx)),
        "B":    rng.uniform(2, 20, len(idx)),
        "V_sw": rng.uniform(300, 700, len(idx)),
        "N_p":  rng.uniform(1, 20, len(idx)),
        "T_p":  rng.uniform(5, 50, len(idx)),
    }, index=idx)


# ===========================================================================
# bin_by_distance
# ===========================================================================

class TestBinByDistance:

    def test_returns_median_and_std(self, simple_power_law_df):
        """bin_by_distance must return (median_df, std_df) tuple."""
        result = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        assert isinstance(result, tuple) and len(result) == 2
        med, std = result
        assert isinstance(med, pd.DataFrame)
        assert isinstance(std, pd.DataFrame)

    def test_median_has_dist_and_param_columns(self, simple_power_law_df):
        """Median DataFrame must have 'dist' and the parameter column."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        assert "dist" in med.columns
        assert "B" in med.columns

    def test_number_of_bins(self, simple_power_law_df):
        """Output should have at most n_bins rows (empty bins produce NaN)."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        assert len(med) <= 10

    def test_custom_distance_range(self, simple_power_law_df):
        """dist_range parameter restricts which data is binned."""
        med_full, _ = bin_by_distance(simple_power_law_df, "B", n_bins=5)
        med_inner, _ = bin_by_distance(
            simple_power_law_df, "B", n_bins=5, dist_range=(0.1, 0.5)
        )
        # Inner range should have fewer rows or lower max dist
        assert med_inner["dist"].max() <= 0.55  # within bin width tolerance

    def test_median_monotonically_decreasing_for_power_law(self, simple_power_law_df):
        """For B ∝ r^-1.73, binned medians must decrease with distance."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med["B"].dropna()
        assert (valid.diff().dropna() < 0).all(), "Binned medians should decrease"

    def test_std_is_nonnegative(self, noisy_df):
        """Standard deviations must always be ≥ 0."""
        _, std = bin_by_distance(noisy_df, "B", n_bins=10)
        valid = std["B"].dropna()
        assert (valid >= 0).all()

    def test_empty_dataframe_returns_empty(self):
        """Empty input must not raise — return empty DataFrames."""
        empty = pd.DataFrame({"dist": [], "B": []})
        med, std = bin_by_distance(empty, "B", n_bins=5)
        assert len(med) == 0 or med["B"].isna().all()

    def test_single_row_per_bin(self):
        """Single data point per bin: std should be NaN (undefined)."""
        df = pd.DataFrame({
            "dist": [0.15, 0.45, 0.75],
            "B":    [10.0, 5.0, 2.0],
        }, index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        _, std = bin_by_distance(df, "B", n_bins=3, dist_range=(0.0, 1.0))
        # With 1 point per bin, std is NaN
        assert std["B"].isna().any()


# ===========================================================================
# fit_power_law
# ===========================================================================

class TestFitPowerLaw:

    def test_recovers_exact_exponent(self, simple_power_law_df):
        """For noiseless data, fit_power_law must recover the true exponent."""
        med, std = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        # exponent should be close to -1.73
        np.testing.assert_allclose(result["exponent"], -1.73, atol=0.05)

    def test_recovers_exact_amplitude(self, simple_power_law_df):
        """Amplitude should be close to 5.0 for B = 5 * r^-1.73."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        np.testing.assert_allclose(result["amplitude"], 5.0, rtol=0.05)

    def test_returns_required_keys(self, simple_power_law_df):
        """Result dict must have: amplitude, exponent, amplitude_err,
        exponent_err, r_squared."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        for key in ("amplitude", "exponent", "amplitude_err", "exponent_err", "r_squared"):
            assert key in result, f"Missing key: {key}"

    def test_r_squared_near_one_for_clean_data(self, simple_power_law_df):
        """R² should be ≈ 1 for noiseless power-law data."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        assert result["r_squared"] > 0.999

    def test_r_squared_lower_for_noisy_data(self, noisy_df):
        """R² should be lower (but still high) for noisy data."""
        med, _ = bin_by_distance(noisy_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        assert 0.90 < result["r_squared"] < 1.0

    def test_flat_profile_exponent_near_zero(self):
        """A flat profile (V_sw) should give exponent ≈ 0."""
        r = np.linspace(0.1, 1.0, 20)
        y = np.full(20, 400.0)
        result = fit_power_law(r, y)
        np.testing.assert_allclose(result["exponent"], 0.0, atol=0.05)

    def test_errors_are_non_negative(self, simple_power_law_df):
        """Uncertainties on fit parameters must be ≥ 0."""
        med, _ = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = med.dropna(subset=["dist", "B"])
        result = fit_power_law(valid["dist"].values, valid["B"].values)
        assert result["amplitude_err"] >= 0
        assert result["exponent_err"] >= 0

    def test_with_yerr_uses_weights(self, simple_power_law_df):
        """Providing yerr should not raise and should still recover exponent."""
        med, std = bin_by_distance(simple_power_law_df, "B", n_bins=10)
        valid = pd.concat([med, std.rename(columns={"B": "B_std"})], axis=1).dropna()
        # Replace zero std (noiseless) with a small positive value
        yerr = valid["B_std"].replace(0, 0.01).values
        result = fit_power_law(valid["dist"].values, valid["B"].values, yerr=yerr)
        np.testing.assert_allclose(result["exponent"], -1.73, atol=0.1)

    def test_insufficient_data_raises(self):
        """Fewer than 3 points should raise ValueError."""
        with pytest.raises(ValueError):
            fit_power_law(np.array([0.3, 0.5]), np.array([10.0, 5.0]))


# ===========================================================================
# radial_profile (end-to-end: binning + fitting)
# ===========================================================================

class TestRadialProfile:

    def test_returns_fit_results_dict(self, simple_power_law_df):
        result = radial_profile(simple_power_law_df, "B", n_bins=10)
        assert "fit" in result
        assert "median" in result
        assert "std" in result

    def test_fit_exponent_correct(self, simple_power_law_df):
        result = radial_profile(simple_power_law_df, "B", n_bins=10)
        np.testing.assert_allclose(result["fit"]["exponent"], -1.73, atol=0.05)

    def test_median_df_has_expected_columns(self, simple_power_law_df):
        result = radial_profile(simple_power_law_df, "B", n_bins=10)
        assert "dist" in result["median"].columns
        assert "B" in result["median"].columns

    def test_multiple_params(self, simple_power_law_df):
        """radial_profile called for N_p should recover exponent ≈ -2."""
        result = radial_profile(simple_power_law_df, "N_p", n_bins=10)
        np.testing.assert_allclose(result["fit"]["exponent"], -2.0, atol=0.05)


# ===========================================================================
# split_solar_cycle
# ===========================================================================

class TestSplitSolarCycle:

    def test_returns_three_dataframes(self, solar_cycle_df):
        """split_solar_cycle must return a dict with min, transition, max keys."""
        result = split_solar_cycle(solar_cycle_df)
        assert set(result.keys()) == {"minimum", "transition", "maximum"}

    def test_minimum_period_correct_dates(self, solar_cycle_df):
        """Solar minimum: 2019-01-01 to 2021-12-31."""
        result = split_solar_cycle(solar_cycle_df)
        df_min = result["minimum"]
        assert df_min.index.min() >= pd.Timestamp("2019-01-01")
        assert df_min.index.max() <= pd.Timestamp("2021-12-31")

    def test_maximum_period_correct_dates(self, solar_cycle_df):
        """Solar maximum: 2023-01-01 to 2025-12-31."""
        result = split_solar_cycle(solar_cycle_df)
        df_max = result["maximum"]
        assert df_max.index.min() >= pd.Timestamp("2023-01-01")
        assert df_max.index.max() <= pd.Timestamp("2025-12-31")

    def test_transition_between_min_and_max(self, solar_cycle_df):
        """Transition: 2022-01-01 to 2022-12-31."""
        result = split_solar_cycle(solar_cycle_df)
        df_tr = result["transition"]
        assert df_tr.index.min() >= pd.Timestamp("2022-01-01")
        assert df_tr.index.max() <= pd.Timestamp("2022-12-31")

    def test_no_overlap_between_periods(self, solar_cycle_df):
        """No timestamp should appear in more than one period."""
        result = split_solar_cycle(solar_cycle_df)
        idx_min = set(result["minimum"].index)
        idx_tr  = set(result["transition"].index)
        idx_max = set(result["maximum"].index)
        assert len(idx_min & idx_tr)  == 0
        assert len(idx_min & idx_max) == 0
        assert len(idx_tr  & idx_max) == 0

    def test_union_covers_all_data(self, solar_cycle_df):
        """Union of the three periods must equal the full input index."""
        result = split_solar_cycle(solar_cycle_df)
        all_idx = (
            result["minimum"].index
            .union(result["transition"].index)
            .union(result["maximum"].index)
        )
        assert set(all_idx) == set(solar_cycle_df.index)

    def test_custom_date_ranges(self, solar_cycle_df):
        """Custom date ranges should override defaults."""
        custom = {
            "minimum":    ("2019-01-01", "2020-12-31"),
            "transition": ("2021-01-01", "2022-12-31"),
            "maximum":    ("2023-01-01", "2025-12-31"),
        }
        result = split_solar_cycle(solar_cycle_df, date_ranges=custom)
        assert result["minimum"].index.max() <= pd.Timestamp("2020-12-31")


# ===========================================================================
# classify_wind_speed
# ===========================================================================

class TestClassifyWindSpeed:

    def test_slow_wind_label(self):
        """V_sw < 400 km/s → 'slow'."""
        df = pd.DataFrame({"V_sw": [300.0, 350.0, 390.0]},
                          index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        result = classify_wind_speed(df)
        assert (result["wind_class"] == "slow").all()

    def test_fast_wind_label(self):
        """V_sw > 600 km/s → 'fast'."""
        df = pd.DataFrame({"V_sw": [650.0, 700.0, 750.0]},
                          index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        result = classify_wind_speed(df)
        assert (result["wind_class"] == "fast").all()

    def test_intermediate_wind_label(self):
        """400 ≤ V_sw ≤ 600 km/s → 'intermediate'."""
        df = pd.DataFrame({"V_sw": [400.0, 500.0, 600.0]},
                          index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        result = classify_wind_speed(df)
        assert (result["wind_class"] == "intermediate").all()

    def test_all_classes_present_in_mixed_data(self):
        """Mixed speeds should produce all three classes."""
        df = pd.DataFrame({"V_sw": [300.0, 500.0, 700.0]},
                          index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        result = classify_wind_speed(df)
        assert set(result["wind_class"].unique()) == {"slow", "intermediate", "fast"}

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        df = pd.DataFrame({"V_sw": [350.0, 550.0, 750.0]},
                          index=pd.date_range("2022-01-01", periods=3, freq="1h"))
        result = classify_wind_speed(df, slow_max=450.0, fast_min=650.0)
        classes = result["wind_class"].tolist()
        assert classes[0] == "slow"
        assert classes[1] == "intermediate"
        assert classes[2] == "fast"

    def test_does_not_modify_input(self):
        """classify_wind_speed must return a new DataFrame."""
        df = pd.DataFrame({"V_sw": [300.0, 600.0]},
                          index=pd.date_range("2022-01-01", periods=2, freq="1h"))
        original_cols = set(df.columns)
        _ = classify_wind_speed(df)
        assert set(df.columns) == original_cols
