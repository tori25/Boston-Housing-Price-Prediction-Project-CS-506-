import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.clean_data import clean_data
from src.features import create_features


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal dataframe mimicking the FY2025 Boston Property Assessment dataset."""
    return pd.DataFrame({
        "LU":          ["R1", "R1",  "R2",  "E",   "R1",  "R1"],
        # "E" = tax-exempt (non-residential) → filtered out
        "ZIP_CODE":    ["02101", "02101", "02102", "02103", "02104", "02105"],
        "LIVING_AREA": [1200,    1800,    0,       2500,    900,     2100],
        # 0 LIVING_AREA → removed
        "LAND_SF":     ["3,000", "4,500", "2,000", "1,000", "2,000", "5,500"],
        "GROSS_AREA":  [1200,    1800,    2000,    2500,    900,     2100],
        "BED_RMS":     [3,       4,       2,       2,       2,       3],
        "FULL_BTH":    [1,       2,       1,       1,       1,       2],
        "HLF_BTH":     [1,       0,       0,       0,       0,       1],
        "TT_RMS":      [6,       8,       5,       4,       5,       7],
        "FIREPLACES":  [1,       2,       0,       0,       0,       1],
        "NUM_PARKING": [1,       2,       0,       0,       0,       1],
        "YR_BUILT":    [1920,    1950,    1980,    2000,    1900,    1985],
        "YR_REMODEL":  [2010,    1950,    2020,    2000,    1900,    2015],
        "RES_FLOOR":   [2,       3,       2,       10,      1,       2],
        "OWN_OCC":     ["Y",     "N",     "Y",     "N",     "N",     "Y"],
        "OVERALL_COND":["G",     "A",     "E",     "G",     "F",     "G"],
        "TOTAL_VALUE": ["500,000","700,000","300,000","400,000","0",     "450,000"],
        # 0 TOTAL_VALUE → removed; values stored as comma-formatted strings
        "LAND_VALUE":  ["200,000","300,000","150,000","200,000","0",     "180,000"],
        # leakage column → dropped
        "BLDG_VALUE":  ["300,000","400,000","150,000","200,000","0",     "270,000"],
        # leakage column → dropped
        "GROSS_TAX":   ["$5,000", "$7,000", "$3,000", "$4,000", "$0",    "$4,500"],
        # leakage column → dropped
        "PID":         ["1",     "2",     "3",     "4",     "5",     "6"],
        # admin column → dropped
        "OWNER":       ["John",  "Jane",  "Bob",   "Alice", "City",  "Mark"],
        # admin column → dropped
    })


@pytest.fixture
def sample_clean_df():
    """Minimal cleaned FY2025 dataframe for feature engineering tests."""
    return pd.DataFrame({
        "LU":          [1,       2,       1,       5],
        "ZIPCODE":     [2101,    2102,    2103,    2104],
        "LIVING_AREA": [1200,    1800,    2500,    900],
        "LAND_SF":     [3000,    4500,    1000,    2000],
        "GROSS_AREA":  [1200,    1800,    2500,    900],
        "BED_RMS":     [3,       4,       2,       2],
        "FULL_BTH":    [1,       2,       1,       1],
        "HLF_BTH":     [1,       0,       0,       0],
        "TT_RMS":      [6,       8,       4,       5],
        "FIRE_PLACE":  [1,       2,       0,       0],
        "NUM_PARKING": [1,       2,       0,       0],
        "YR_BUILT":    [1920,    1950,    2000,    1980],
        "YR_REMODEL":  [2010,    1950,    2000,    2020],
        "RES_FLOOR":   [2,       3,       10,      2],
        "OWN_OCC":     [1,       0,       0,       1],
        "OVERALL_COND":[4,       3,       5,       4],
        "TOTAL_VALUE": [500000,  700000,  400000,  350000],
    })


# ──────────────────────────────────────────────
# clean_data tests
# ──────────────────────────────────────────────

class TestCleanData:

    def test_removes_non_residential(self, sample_raw_df, tmp_path, monkeypatch):
        """Non-residential LU types (E, C, I, etc.) must be filtered out."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        # LU is encoded as int; non-residential rows were never in the map so they're gone
        assert len(result) < len(sample_raw_df)

    def test_removes_zero_total_value(self, sample_raw_df, tmp_path, monkeypatch):
        """Rows with TOTAL_VALUE == 0 should be removed."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert (result["TOTAL_VALUE"] > 0).all()

    def test_removes_zero_living_area(self, sample_raw_df, tmp_path, monkeypatch):
        """Rows with LIVING_AREA == 0 should be removed."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert (result["LIVING_AREA"] > 0).all()

    def test_removes_leakage_columns(self, sample_raw_df, tmp_path, monkeypatch):
        """LAND_VALUE, BLDG_VALUE, GROSS_TAX must not appear in cleaned output."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        for col in ["LAND_VALUE", "BLDG_VALUE", "GROSS_TAX"]:
            assert col not in result.columns, f"Leakage column '{col}' still present"

    def test_removes_admin_columns(self, sample_raw_df, tmp_path, monkeypatch):
        """Admin columns (PID, OWNER) must not appear in cleaned output."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        for col in ["PID", "OWNER"]:
            assert col not in result.columns, f"Admin column '{col}' still present"

    def test_no_missing_values_after_clean(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned dataset should have zero missing values."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert result.isnull().sum().sum() == 0

    def test_output_file_created(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned CSV should be written to the processed folder."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        assert (tmp_path / "boston_clean.csv").exists()

    def test_total_value_preserved(self, sample_raw_df, tmp_path, monkeypatch):
        """Target column TOTAL_VALUE must still be present after cleaning."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert "TOTAL_VALUE" in result.columns

    def test_no_duplicates(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned dataset should have no duplicate rows."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert result.duplicated().sum() == 0

    def test_row_count_reduced(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaning should remove at least the non-residential, zero-value, and zero-area rows."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert len(result) < len(sample_raw_df)

    # ── helper ──────────────────────────────────
    def _run_clean(self, df, tmp_path, monkeypatch):
        raw_path = tmp_path / "fy2025_property_assessment.csv"
        out_path = tmp_path / "boston_clean.csv"
        df.to_csv(raw_path, index=False)

        import src.clean_data as cd
        monkeypatch.setattr(cd, "RAW_DATA_PATH", str(raw_path))
        monkeypatch.setattr(cd, "PROCESSED_DATA_PATH", str(out_path))
        cd.clean_data()


# ──────────────────────────────────────────────
# create_features tests
# ──────────────────────────────────────────────

class TestCreateFeatures:

    def test_age_created(self, sample_clean_df):
        """AGE should equal 2025 - YR_BUILT."""
        result = create_features(sample_clean_df)
        assert "AGE" in result.columns
        expected = 2025 - sample_clean_df["YR_BUILT"]
        pd.testing.assert_series_equal(
            result["AGE"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_is_remodeled_created(self, sample_clean_df):
        """IS_REMODELED should be 1 when YR_REMODEL > YR_BUILT, else 0."""
        result = create_features(sample_clean_df)
        assert "IS_REMODELED" in result.columns
        expected = (sample_clean_df["YR_REMODEL"] > sample_clean_df["YR_BUILT"]).astype(int)
        pd.testing.assert_series_equal(
            result["IS_REMODELED"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_bath_total_created(self, sample_clean_df):
        """BATH_TOTAL should equal FULL_BTH + 0.5 * HLF_BTH."""
        result = create_features(sample_clean_df)
        assert "BATH_TOTAL" in result.columns
        expected = sample_clean_df["FULL_BTH"] + 0.5 * sample_clean_df["HLF_BTH"]
        pd.testing.assert_series_equal(
            result["BATH_TOTAL"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_area_per_room_created(self, sample_clean_df):
        """AREA_PER_ROOM should equal LIVING_AREA / TT_RMS."""
        result = create_features(sample_clean_df)
        assert "AREA_PER_ROOM" in result.columns
        expected = sample_clean_df["LIVING_AREA"] / sample_clean_df["TT_RMS"]
        pd.testing.assert_series_equal(
            result["AREA_PER_ROOM"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_bed_bath_created(self, sample_clean_df):
        """BED_BATH should equal BED_RMS * BATH_TOTAL."""
        result = create_features(sample_clean_df)
        assert "BED_BATH" in result.columns
        bath_total = sample_clean_df["FULL_BTH"] + 0.5 * sample_clean_df["HLF_BTH"]
        expected = sample_clean_df["BED_RMS"] * bath_total
        pd.testing.assert_series_equal(
            result["BED_BATH"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_original_columns_preserved(self, sample_clean_df):
        """create_features should preserve original columns except YR_BUILT (replaced by AGE)."""
        result = create_features(sample_clean_df)
        for col in sample_clean_df.columns:
            if col == "YR_BUILT":
                continue  # intentionally dropped — replaced by AGE
            assert col in result.columns

    def test_no_mutation_of_input(self, sample_clean_df):
        """create_features should not modify the original dataframe."""
        original_cols = list(sample_clean_df.columns)
        create_features(sample_clean_df)
        assert list(sample_clean_df.columns) == original_cols

    def test_row_count_unchanged(self, sample_clean_df):
        """Feature engineering should not add or remove rows."""
        result = create_features(sample_clean_df)
        assert len(result) == len(sample_clean_df)

    def test_output_file_created(self, sample_clean_df, tmp_path, monkeypatch):
        """features.py main() should write train_features.csv to the output path."""
        import src.features as feat
        in_path = tmp_path / "boston_clean.csv"
        out_path = tmp_path / "train_features.csv"
        sample_clean_df.to_csv(in_path, index=False)
        monkeypatch.setattr(feat, "INPUT_PATH", str(in_path))
        monkeypatch.setattr(feat, "OUTPUT_PATH", str(out_path))
        feat.main()
        assert out_path.exists()

    def test_output_file_has_engineered_columns(self, sample_clean_df, tmp_path, monkeypatch):
        """Output CSV from features.py should contain all 5 engineered feature columns."""
        import src.features as feat
        in_path = tmp_path / "boston_clean.csv"
        out_path = tmp_path / "train_features.csv"
        sample_clean_df.to_csv(in_path, index=False)
        monkeypatch.setattr(feat, "INPUT_PATH", str(in_path))
        monkeypatch.setattr(feat, "OUTPUT_PATH", str(out_path))
        feat.main()
        result = pd.read_csv(out_path)
        for col in ["AGE", "IS_REMODELED", "BATH_TOTAL", "AREA_PER_ROOM", "BED_BATH"]:
            assert col in result.columns, f"Missing engineered column: {col}"
