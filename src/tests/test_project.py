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
    """Minimal dataframe that mimics the raw Boston Housing dataset."""
    return pd.DataFrame({
        "crim":    [0.00632, 0.02731, 15.0,   0.05023],  # 15.0 is a top-1% outlier
        "zn":      [18.0,    0.0,     0.0,    0.0],
        "indus":   [2.31,    7.07,    7.07,   2.18],
        "chas":    [0,       0,       0,      0],
        "nox":     [0.538,   0.469,   0.469,  0.458],
        "rm":      [6.575,   6.421,   7.185,  6.998],
        "age":     [65.2,    78.9,    61.1,   45.8],
        "dis":     [4.09,    4.9671,  4.9671, 6.0622],
        "rad":     [1,       2,       2,      3],
        "tax":     [296,     242,     242,    222],
        "ptratio": [15.3,    17.8,    17.8,   18.7],
        "b":       [396.9,   396.9,   392.83, 394.63],
        "lstat":   [4.98,    9.14,    4.03,   2.94],
        "medv":    [24.0,    21.6,    50.0,   33.4],  # 50.0 is censored
    })


@pytest.fixture
def sample_clean_df():
    """Minimal cleaned Boston dataframe for feature engineering tests."""
    return pd.DataFrame({
        "crim":    [0.00632, 0.02731, 0.05023, 0.08829],
        "zn":      [18.0,    0.0,     0.0,     12.5],
        "indus":   [2.31,    7.07,    2.18,    7.87],
        "chas":    [0,       0,       0,       0],
        "nox":     [0.538,   0.469,   0.458,   0.524],
        "rm":      [6.575,   6.421,   6.998,   6.012],
        "age":     [65.2,    78.9,    45.8,    66.6],
        "dis":     [4.09,    4.9671,  6.0622,  5.5605],
        "rad":     [1,       2,       3,       5],
        "tax":     [296,     242,     222,     311],
        "ptratio": [15.3,    17.8,    18.7,    15.2],
        "b":       [396.9,   396.9,   394.63,  395.6],
        "lstat":   [4.98,    9.14,    2.94,    12.43],
        "medv":    [24.0,    21.6,    33.4,    22.9],
    })


# ──────────────────────────────────────────────
# clean_data tests
# ──────────────────────────────────────────────

class TestCleanData:

    def test_removes_censored_medv(self, sample_raw_df, tmp_path, monkeypatch):
        """Rows where medv == 50 (censored values) should be removed."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert (result["medv"] < 50).all()

    def test_removes_extreme_crime(self, sample_raw_df, tmp_path, monkeypatch):
        """Top 1% crime rate outliers should be removed."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        original_99 = sample_raw_df["crim"].quantile(0.99)
        assert (result["crim"] <= original_99).all()

    def test_no_missing_values_after_clean(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned dataset should have zero missing values."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert result.isnull().sum().sum() == 0

    def test_output_file_created(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned CSV should be written to the processed folder."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        assert (tmp_path / "boston_clean.csv").exists()

    def test_medv_preserved(self, sample_raw_df, tmp_path, monkeypatch):
        """Target column medv must still be present after cleaning."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert "medv" in result.columns

    def test_no_duplicates(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned dataset should have no duplicate rows."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert result.duplicated().sum() == 0

    def test_row_count_reduced(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaning should remove at least the censored medv==50 row."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "boston_clean.csv")
        assert len(result) < len(sample_raw_df)

    # ── helper ──────────────────────────────────
    def _run_clean(self, df, tmp_path, monkeypatch):
        raw_path = tmp_path / "boston.csv"
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

    def test_crime_log_created(self, sample_clean_df):
        """CRIME_LOG should equal log1p(crim)."""
        result = create_features(sample_clean_df)
        assert "CRIME_LOG" in result.columns
        expected = np.log1p(sample_clean_df["crim"])
        pd.testing.assert_series_equal(
            result["CRIME_LOG"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_room_sq_created(self, sample_clean_df):
        """ROOM_SQ should equal rm squared."""
        result = create_features(sample_clean_df)
        assert "ROOM_SQ" in result.columns
        expected = sample_clean_df["rm"] ** 2
        pd.testing.assert_series_equal(
            result["ROOM_SQ"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_tax_per_room_created(self, sample_clean_df):
        """TAX_PER_ROOM should equal tax / rm."""
        result = create_features(sample_clean_df)
        assert "TAX_PER_ROOM" in result.columns
        expected = sample_clean_df["tax"] / sample_clean_df["rm"]
        pd.testing.assert_series_equal(
            result["TAX_PER_ROOM"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_lstat_per_room_created(self, sample_clean_df):
        """LSTAT_PER_ROOM should equal lstat / rm."""
        result = create_features(sample_clean_df)
        assert "LSTAT_PER_ROOM" in result.columns
        expected = sample_clean_df["lstat"] / sample_clean_df["rm"]
        pd.testing.assert_series_equal(
            result["LSTAT_PER_ROOM"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_pollution_proximity_created(self, sample_clean_df):
        """POLLUTION_PROXIMITY should equal nox / dis."""
        result = create_features(sample_clean_df)
        assert "POLLUTION_PROXIMITY" in result.columns
        expected = sample_clean_df["nox"] / sample_clean_df["dis"]
        pd.testing.assert_series_equal(
            result["POLLUTION_PROXIMITY"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_school_index_created(self, sample_clean_df):
        """SCHOOL_INDEX should equal ptratio * lstat."""
        result = create_features(sample_clean_df)
        assert "SCHOOL_INDEX" in result.columns
        expected = sample_clean_df["ptratio"] * sample_clean_df["lstat"]
        pd.testing.assert_series_equal(
            result["SCHOOL_INDEX"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_age_dist_created(self, sample_clean_df):
        """AGE_DIST should equal age * dis."""
        result = create_features(sample_clean_df)
        assert "AGE_DIST" in result.columns
        expected = sample_clean_df["age"] * sample_clean_df["dis"]
        pd.testing.assert_series_equal(
            result["AGE_DIST"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False
        )

    def test_original_columns_preserved(self, sample_clean_df):
        """create_features should not drop any original columns."""
        result = create_features(sample_clean_df)
        for col in sample_clean_df.columns:
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
        """Output CSV from features.py should contain all 7 engineered feature columns."""
        import src.features as feat
        in_path = tmp_path / "boston_clean.csv"
        out_path = tmp_path / "train_features.csv"
        sample_clean_df.to_csv(in_path, index=False)
        monkeypatch.setattr(feat, "INPUT_PATH", str(in_path))
        monkeypatch.setattr(feat, "OUTPUT_PATH", str(out_path))
        feat.main()
        result = pd.read_csv(out_path)
        for col in ["CRIME_LOG", "ROOM_SQ", "TAX_PER_ROOM", "LSTAT_PER_ROOM",
                    "POLLUTION_PROXIMITY", "SCHOOL_INDEX", "AGE_DIST"]:
            assert col in result.columns, f"Missing engineered column: {col}"
