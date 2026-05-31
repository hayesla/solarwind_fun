"""
Tests for swpipeline.trajectory — spacecraft position functions.

SPICE-dependent tests use monkeypatching to avoid downloading large kernel
files. The ACE tests require no mocking (fixed analytical position).

Run with:  conda run -n sw_pipeline pytest tests/test_trajectory.py -v
"""

import numpy as np
import pandas as pd
import pytest
from astropy import units as u

import swpipeline.trajectory as traj_module
from swpipeline.trajectory import (
    SPICE_KERNELS,
    add_trajectory_columns,
    get_position,
    load_spice_kernels,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_kernel_cache():
    """Clear the module-level kernel-loaded cache around every test."""
    traj_module._kernels_loaded.clear()
    yield
    traj_module._kernels_loaded.clear()


@pytest.fixture
def sample_times():
    return pd.date_range("2022-06-15", periods=5, freq="1h")


@pytest.fixture
def mock_spice(monkeypatch):
    """
    Patch load_spice_kernels and sunpy.coordinates.spice.get_body so that
    no network access or kernel files are required.

    The mock returns a constant synthetic position:
        dist = 0.5 AU,  lat = 5°,  lon = 120°
    """
    # --- Patch load_spice_kernels to be a no-op ---
    def fake_load(sc):
        traj_module._kernels_loaded.add(sc)

    monkeypatch.setattr(traj_module, "load_spice_kernels", fake_load)

    # --- Patch sunpy.coordinates.spice.get_body ---
    class FakeHGS:
        def __init__(self, n: int) -> None:
            self.radius = np.full(n, 0.5) * u.AU
            self.lat    = np.full(n, 5.0)  * u.deg
            self.lon    = np.full(n, 120.0) * u.deg

    class FakeSkyCoord:
        def __init__(self, n: int) -> None:
            self._hgs = FakeHGS(n)

        @property
        def heliographic_stonyhurst(self) -> FakeHGS:
            return self._hgs

    def fake_get_body(body: str, times) -> FakeSkyCoord:
        return FakeSkyCoord(len(pd.DatetimeIndex(times)))

    import sunpy.coordinates.spice as spice_mod
    monkeypatch.setattr(spice_mod, "get_body", fake_get_body)


# ===========================================================================
# get_position — ACE  (no SPICE needed)
# ===========================================================================

class TestGetPositionACE:

    def test_returns_dataframe(self, sample_times):
        result = get_position("ace", sample_times)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_times):
        result = get_position("ace", sample_times)
        assert set(result.columns) == {"dist", "lat", "lon"}

    def test_dist_is_one_au(self, sample_times):
        result = get_position("ace", sample_times)
        np.testing.assert_allclose(result["dist"].values, 1.0)

    def test_lat_is_zero(self, sample_times):
        result = get_position("ace", sample_times)
        np.testing.assert_allclose(result["lat"].values, 0.0)

    def test_lon_is_zero(self, sample_times):
        result = get_position("ace", sample_times)
        np.testing.assert_allclose(result["lon"].values, 0.0)

    def test_index_matches_input_times(self, sample_times):
        result = get_position("ace", sample_times)
        assert list(result.index) == list(sample_times)

    def test_length_matches_input(self, sample_times):
        result = get_position("ace", sample_times)
        assert len(result) == len(sample_times)

    def test_single_timestamp(self):
        t = pd.DatetimeIndex(["2022-06-15 12:00:00"])
        result = get_position("ace", t)
        assert len(result) == 1
        assert result["dist"].iloc[0] == pytest.approx(1.0)

    def test_all_values_finite(self, sample_times):
        result = get_position("ace", sample_times)
        assert result.notna().all().all()


# ===========================================================================
# get_position — PSP / SolO  (SPICE mocked)
# ===========================================================================

class TestGetPositionSPICE:

    def test_psp_returns_dataframe(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        assert isinstance(result, pd.DataFrame)

    def test_psp_has_expected_columns(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        assert set(result.columns) == {"dist", "lat", "lon"}

    def test_psp_length_matches_input(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        assert len(result) == len(sample_times)

    def test_psp_index_matches_input(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        assert list(result.index) == list(sample_times)

    def test_psp_dist_matches_mock_value(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        np.testing.assert_allclose(result["dist"].values, 0.5)

    def test_psp_lat_matches_mock_value(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        np.testing.assert_allclose(result["lat"].values, 5.0)

    def test_solo_returns_dataframe(self, sample_times, mock_spice):
        result = get_position("solo", sample_times)
        assert isinstance(result, pd.DataFrame)

    def test_solo_dist_matches_mock_value(self, sample_times, mock_spice):
        result = get_position("solo", sample_times)
        np.testing.assert_allclose(result["dist"].values, 0.5)

    def test_all_values_finite(self, sample_times, mock_spice):
        result = get_position("psp", sample_times)
        assert result.notna().all().all()


# ===========================================================================
# load_spice_kernels
# ===========================================================================

class TestLoadSpiceKernels:

    def test_raises_for_unknown_sc(self):
        with pytest.raises(ValueError, match="No SPICE kernels configured"):
            load_spice_kernels("voyager")

    def test_ace_is_silently_ignored(self):
        """ACE does not use SPICE; call must be a no-op (no error)."""
        load_spice_kernels("ace")  # must not raise

    def test_ace_does_not_add_to_kernels_loaded(self):
        load_spice_kernels("ace")
        assert "ace" not in traj_module._kernels_loaded

    def test_kernel_urls_defined_for_psp(self):
        assert "psp" in SPICE_KERNELS
        assert len(SPICE_KERNELS["psp"]) >= 1
        for url in SPICE_KERNELS["psp"]:
            assert url.startswith("http")

    def test_kernel_urls_defined_for_solo(self):
        assert "solo" in SPICE_KERNELS
        assert len(SPICE_KERNELS["solo"]) >= 1

    def test_spk_file_in_psp_kernels(self):
        """PSP kernel list must include an SPK file (.bsp)."""
        assert any(".bsp" in url for url in SPICE_KERNELS["psp"])

    def test_already_loaded_skips_download(self, monkeypatch):
        """If SC already in _kernels_loaded, sunpy_cache.download is never called."""
        traj_module._kernels_loaded.add("solo")

        download_calls = {"n": 0}

        def fake_download(url: str) -> str:
            download_calls["n"] += 1
            return "/fake/path.bsp"

        # sunpy.data.cache is a Cache instance on the sunpy.data module,
        # not a sub-module — patch via the parent module.
        import sunpy.data as sunpy_data_mod
        monkeypatch.setattr(sunpy_data_mod.cache, "download", fake_download)

        load_spice_kernels("solo")   # "solo" already loaded → must return early
        assert download_calls["n"] == 0, (
            "load_spice_kernels should not re-download already-loaded kernels"
        )

    def test_first_load_calls_initialize(self, monkeypatch):
        """First load for a SC must call spice.initialize exactly once."""
        init_calls = {"n": 0}

        def fake_download(url: str) -> str:
            return "/fake/path.bsp"

        def fake_initialize(kernel_files) -> None:
            init_calls["n"] += 1

        import sunpy.data as sunpy_data_mod
        import sunpy.coordinates.spice as spice_mod
        monkeypatch.setattr(sunpy_data_mod.cache, "download", fake_download)
        monkeypatch.setattr(spice_mod, "initialize", fake_initialize)

        load_spice_kernels("psp")
        assert init_calls["n"] == 1
        assert "psp" in traj_module._kernels_loaded


# ===========================================================================
# add_trajectory_columns
# ===========================================================================

class TestAddTrajectoryColumns:

    def test_adds_dist_lat_lon_ace(self, clean_df):
        """ACE: add_trajectory_columns must add dist, lat, lon without SPICE."""
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "ace")
        assert "dist" in result.columns
        assert "lat"  in result.columns
        assert "lon"  in result.columns

    def test_does_not_modify_input(self, clean_df):
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        original_cols = set(df_no_traj.columns)
        _ = add_trajectory_columns(df_no_traj, "ace")
        assert set(df_no_traj.columns) == original_cols

    def test_ace_dist_is_one_au(self, clean_df):
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "ace")
        np.testing.assert_allclose(result["dist"].values, 1.0)

    def test_ace_lat_is_zero(self, clean_df):
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "ace")
        np.testing.assert_allclose(result["lat"].values, 0.0)

    def test_index_is_preserved(self, clean_df):
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "ace")
        assert list(result.index) == list(df_no_traj.index)

    def test_other_columns_unchanged(self, clean_df):
        """Existing columns must be unchanged after adding trajectory."""
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "ace")
        for col in df_no_traj.columns:
            pd.testing.assert_series_equal(result[col], df_no_traj[col])

    def test_psp_dist_uses_spice_mock(self, clean_df, mock_spice):
        """With mocked SPICE, PSP dist should equal the mock value (0.5 AU)."""
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "psp")
        np.testing.assert_allclose(result["dist"].values, 0.5)

    def test_solo_lat_uses_spice_mock(self, clean_df, mock_spice):
        df_no_traj = clean_df.drop(columns=["dist", "lat", "lon"])
        result = add_trajectory_columns(df_no_traj, "solo")
        np.testing.assert_allclose(result["lat"].values, 5.0)
