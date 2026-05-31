"""
Tests for swpipeline.download — year chunking and the full build pipeline.

Network-dependent tests (fetch_raw) are marked with @pytest.mark.network
and are skipped by default. Run them explicitly with:

    conda run -n sw_pipeline pytest tests/test_download.py -v -m network

All other tests are fully offline: build_spacecraft_dataset() accepts
``mag_df`` and ``plasma_df`` keyword arguments that bypass the download step,
so the cleaning / resampling / merging logic can be tested with the synthetic
DataFrames from conftest.py.

Run with:  conda run -n sw_pipeline pytest tests/test_download.py -v
"""

import pandas as pd
import pytest

from swpipeline.download import _year_chunks, build_spacecraft_dataset


# ===========================================================================
# _year_chunks — pure Python, no network required
# ===========================================================================

class TestYearChunks:

    def test_single_full_year(self):
        chunks = _year_chunks("2022-01-01", "2022-12-31")
        assert len(chunks) == 1
        assert chunks[0] == ("2022-01-01", "2022-12-31")

    def test_two_full_years(self):
        chunks = _year_chunks("2022-01-01", "2023-12-31")
        assert len(chunks) == 2
        assert chunks[0] == ("2022-01-01", "2022-12-31")
        assert chunks[1] == ("2023-01-01", "2023-12-31")

    def test_three_years(self):
        chunks = _year_chunks("2021-01-01", "2023-12-31")
        assert len(chunks) == 3

    def test_partial_first_year(self):
        """Start mid-year → first chunk begins at start date."""
        chunks = _year_chunks("2022-06-15", "2023-12-31")
        assert chunks[0][0] == "2022-06-15"
        assert chunks[0][1] == "2022-12-31"
        assert chunks[1] == ("2023-01-01", "2023-12-31")

    def test_partial_last_year(self):
        """End mid-year → last chunk ends at end date."""
        chunks = _year_chunks("2022-01-01", "2023-03-31")
        assert chunks[-1][1] == "2023-03-31"

    def test_same_year_partial(self):
        """Start and end within the same year → exactly one chunk."""
        chunks = _year_chunks("2022-04-01", "2022-09-30")
        assert len(chunks) == 1
        assert chunks[0] == ("2022-04-01", "2022-09-30")

    def test_full_mission_psp(self):
        """PSP full mission (2018–2025) → 8 chunks."""
        chunks = _year_chunks("2018-01-01", "2025-12-31")
        assert len(chunks) == 8

    def test_chunk_boundaries_are_consecutive(self):
        """End of chunk i and start of chunk i+1 differ by exactly one day."""
        chunks = _year_chunks("2021-01-01", "2023-12-31")
        for i in range(len(chunks) - 1):
            end_i    = pd.Timestamp(chunks[i][1])
            start_i1 = pd.Timestamp(chunks[i + 1][0])
            assert start_i1 == end_i + pd.Timedelta(days=1)

    def test_start_equals_end(self):
        """Single-day range → one chunk with matching start and end."""
        chunks = _year_chunks("2022-06-15", "2022-06-15")
        assert len(chunks) == 1
        assert chunks[0] == ("2022-06-15", "2022-06-15")


# ===========================================================================
# build_spacecraft_dataset — offline tests using fixture DataFrames
# ===========================================================================

class TestBuildSpacecraftDataset:

    # --- Error handling ---

    def test_unknown_sc_raises_value_error(self, psp_mag_raw, psp_spc_raw):
        """Unknown spacecraft name must raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="Unknown spacecraft"):
            build_spacecraft_dataset(
                "voyager", "2022-01-01", "2022-12-31",
                mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
            )

    def test_empty_mag_returns_empty_dataframe(self, psp_spc_raw):
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=pd.DataFrame(), plasma_df=psp_spc_raw,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_plasma_returns_empty_dataframe(self, psp_mag_raw):
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=pd.DataFrame(),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    # --- PSP ---

    def test_psp_returns_nonempty_dataframe(self, psp_mag_raw, psp_spc_raw):
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_psp_has_magnetic_columns(self, psp_mag_raw, psp_spc_raw):
        """PSP output must include B, B_r, B_t, B_n (all in nT)."""
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        for col in ("B", "B_r", "B_t", "B_n"):
            assert col in result.columns, f"Missing magnetic column: {col}"

    def test_psp_has_plasma_columns(self, psp_mag_raw, psp_spc_raw):
        """PSP output must include N_p, T_p, V_sw, V_r, V_t, V_n."""
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        for col in ("N_p", "T_p", "V_sw", "V_r", "V_t", "V_n"):
            assert col in result.columns, f"Missing plasma column: {col}"

    def test_psp_index_is_datetime(self, psp_mag_raw, psp_spc_raw):
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_psp_index_is_hourly(self, psp_mag_raw, psp_spc_raw):
        """After resampling, successive timestamps must be 1 h apart."""
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        if len(result) >= 2:
            diffs = result.index.to_series().diff().dropna()
            assert (diffs == pd.Timedelta("1h")).all()

    def test_psp_b_is_positive(self, psp_mag_raw, psp_spc_raw):
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        assert (result["B"] > 0).all()

    def test_psp_all_values_finite(self, psp_mag_raw, psp_spc_raw):
        """No NaN or Inf should survive the cleaning + resampling pipeline."""
        result = build_spacecraft_dataset(
            "psp", "2022-01-01", "2022-12-31",
            mag_df=psp_mag_raw, plasma_df=psp_spc_raw,
        )
        assert result.notna().all().all(), "NaN found after build_spacecraft_dataset"

    # --- SolO ---

    def test_solo_returns_nonempty_dataframe(self, solo_mag_raw, solo_swa_raw):
        result = build_spacecraft_dataset(
            "solo", "2022-01-01", "2022-12-31",
            mag_df=solo_mag_raw, plasma_df=solo_swa_raw,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_solo_has_standard_columns(self, solo_mag_raw, solo_swa_raw):
        result = build_spacecraft_dataset(
            "solo", "2022-01-01", "2022-12-31",
            mag_df=solo_mag_raw, plasma_df=solo_swa_raw,
        )
        for col in ("B", "N_p", "T_p", "V_sw"):
            assert col in result.columns

    # --- ACE ---

    def test_ace_returns_nonempty_dataframe(self, ace_mag_raw, ace_swe_raw):
        result = build_spacecraft_dataset(
            "ace", "2022-01-01", "2022-12-31",
            mag_df=ace_mag_raw, plasma_df=ace_swe_raw,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_ace_has_standard_columns(self, ace_mag_raw, ace_swe_raw):
        result = build_spacecraft_dataset(
            "ace", "2022-01-01", "2022-12-31",
            mag_df=ace_mag_raw, plasma_df=ace_swe_raw,
        )
        for col in ("B", "N_p", "T_p", "V_sw"):
            assert col in result.columns

    def test_ace_index_is_hourly(self, ace_mag_raw, ace_swe_raw):
        """ACE data is already hourly; the merged index should stay hourly."""
        result = build_spacecraft_dataset(
            "ace", "2022-01-01", "2022-12-31",
            mag_df=ace_mag_raw, plasma_df=ace_swe_raw,
        )
        if len(result) >= 2:
            diffs = result.index.to_series().diff().dropna()
            assert (diffs == pd.Timedelta("1h")).all()

    def test_ace_b_is_positive(self, ace_mag_raw, ace_swe_raw):
        result = build_spacecraft_dataset(
            "ace", "2022-01-01", "2022-12-31",
            mag_df=ace_mag_raw, plasma_df=ace_swe_raw,
        )
        assert (result["B"] > 0).all()

    def test_ace_does_not_have_rtn_mag_components(self, ace_mag_raw, ace_swe_raw):
        """ACE MAG provides only |B|; RTN components must NOT appear."""
        result = build_spacecraft_dataset(
            "ace", "2022-01-01", "2022-12-31",
            mag_df=ace_mag_raw, plasma_df=ace_swe_raw,
        )
        for col in ("B_r", "B_t", "B_n"):
            assert col not in result.columns, f"ACE should not have {col}"
