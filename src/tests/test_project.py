import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clean_data import clean_data
from src.features import create_features


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal dataframe that mimics the raw Ames dataset."""
    return pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "GrLivArea": [1500, 2000, 5000, 800],   # 5000 is an outlier
        "SalePrice": [200000, 300000, 400000, 150000],
        "LotFrontage": [70.0, None, 60.0, None],
        "MasVnrArea": [100.0, None, 0.0, 50.0],
        "GarageYrBlt": [2000.0, None, 1990.0, None],
        "PoolQC": [None, None, None, None],
        "MiscFeature": [None, "Shed", None, None],
        "Alley": [None, None, None, None],
        "Fence": [None, "MnPrv", None, None],
        "BsmtQual": [None, "Gd", "TA", None],
        "GarageType": ["Attchd", None, "Detchd", None],
        "FireplaceQu": [None, "Gd", None, None],
        "GarageFinish": ["Fin", None, "Unf", None],
        "GarageQual": [None, "TA", None, None],
        "GarageCond": [None, "TA", None, None],
        "BsmtCond": [None, "TA", None, None],
        "BsmtExposure": [None, "No", None, None],
        "BsmtFinType1": [None, "GLQ", None, None],
        "BsmtFinType2": [None, "Unf", None, None],
        "MasVnrType": [None, "BrkFace", None, None],
        "Neighborhood": ["CollgCr", "Veenker", "Crawfor", "NoRidge"],
        "SaleCondition": ["Normal", "Normal", "Abnorml", "Normal"],
    })


@pytest.fixture
def sample_clean_df():
    """Minimal cleaned dataframe for feature engineering tests."""
    return pd.DataFrame({
        "TotalBsmtSF": [800, 1000, 600, 400],
        "1stFlrSF":    [1000, 1200, 900, 700],
        "2ndFlrSF":    [500, 600, 0, 300],
        "FullBath":    [2, 2, 1, 1],
        "HalfBath":    [1, 0, 0, 1],
        "BsmtFullBath":[1, 1, 0, 0],
        "BsmtHalfBath":[0, 0, 1, 0],
        "YrSold":      [2010, 2009, 2008, 2007],
        "YearBuilt":   [1990, 2000, 1970, 1985],
        "YearRemodAdd":[2005, 2000, 1980, 2000],
        "OpenPorchSF": [60, 0, 30, 20],
        "EnclosedPorch":[0, 40, 0, 0],
        "3SsnPorch":   [0, 0, 0, 0],
        "ScreenPorch": [0, 0, 50, 0],
        "SalePrice":   [200000, 300000, 150000, 180000],
    })


# ──────────────────────────────────────────────
# clean_data tests
# ──────────────────────────────────────────────

class TestCleanData:

    def test_removes_id_column(self, sample_raw_df, tmp_path, monkeypatch):
        """Id column should be dropped after cleaning."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        assert "Id" not in result.columns

    def test_removes_outliers(self, sample_raw_df, tmp_path, monkeypatch):
        """Rows with GrLivArea >= 4000 should be removed."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        assert (result["GrLivArea"] < 4000).all()

    def test_no_missing_values_after_clean(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned dataset should have zero missing values."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        assert result.isnull().sum().sum() == 0

    def test_drops_low_value_columns(self, sample_raw_df, tmp_path, monkeypatch):
        """Columns like PoolQC, MiscFeature, Alley, Fence should be dropped."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        for col in ["PoolQC", "MiscFeature", "Alley", "Fence"]:
            assert col not in result.columns

    def test_lotfrontage_filled_with_median(self, sample_raw_df, tmp_path, monkeypatch):
        """LotFrontage NaN values should be filled (no missing values remain)."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        assert result["LotFrontage"].isnull().sum() == 0

    def test_output_file_created(self, sample_raw_df, tmp_path, monkeypatch):
        """Cleaned CSV should be written to the processed folder."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        assert (tmp_path / "train_clean.csv").exists()

    def test_saleprice_preserved(self, sample_raw_df, tmp_path, monkeypatch):
        """SalePrice target column must still be present after cleaning."""
        self._run_clean(sample_raw_df, tmp_path, monkeypatch)
        result = pd.read_csv(tmp_path / "train_clean.csv")
        assert "SalePrice" in result.columns

    # ── helper ──────────────────────────────────
    def _run_clean(self, df, tmp_path, monkeypatch):
        raw_path = tmp_path / "train.csv"
        out_path = tmp_path / "train_clean.csv"
        df.to_csv(raw_path, index=False)

        import src.clean_data as cd
        monkeypatch.setattr(cd, "RAW_DATA_PATH", str(raw_path))
        monkeypatch.setattr(cd, "PROCESSED_DATA_PATH", str(out_path))
        cd.clean_data()


# ──────────────────────────────────────────────
# create_features tests
# ──────────────────────────────────────────────

class TestCreateFeatures:

    def test_total_sf_created(self, sample_clean_df):
        """TotalSF should equal TotalBsmtSF + 1stFlrSF + 2ndFlrSF."""
        result = create_features(sample_clean_df)
        assert "TotalSF" in result.columns
        expected = sample_clean_df["TotalBsmtSF"] + sample_clean_df["1stFlrSF"] + sample_clean_df["2ndFlrSF"]
        pd.testing.assert_series_equal(result["TotalSF"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)

    def test_total_bathrooms_created(self, sample_clean_df):
        """TotalBathrooms should be a weighted sum of all bathroom columns."""
        result = create_features(sample_clean_df)
        assert "TotalBathrooms" in result.columns
        assert (result["TotalBathrooms"] >= 0).all()

    def test_house_age_at_sale(self, sample_clean_df):
        """HouseAgeAtSale should equal YrSold - YearBuilt and be non-negative."""
        result = create_features(sample_clean_df)
        assert "HouseAgeAtSale" in result.columns
        expected = sample_clean_df["YrSold"] - sample_clean_df["YearBuilt"]
        pd.testing.assert_series_equal(result["HouseAgeAtSale"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)
        assert (result["HouseAgeAtSale"] >= 0).all()

    def test_years_since_remodel(self, sample_clean_df):
        """YearsSinceRemodel should equal YrSold - YearRemodAdd."""
        result = create_features(sample_clean_df)
        assert "YearsSinceRemodel" in result.columns
        expected = sample_clean_df["YrSold"] - sample_clean_df["YearRemodAdd"]
        pd.testing.assert_series_equal(result["YearsSinceRemodel"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)

    def test_total_porch_sf(self, sample_clean_df):
        """TotalPorchSF should be the sum of all porch columns."""
        result = create_features(sample_clean_df)
        assert "TotalPorchSF" in result.columns
        expected = (sample_clean_df["OpenPorchSF"] + sample_clean_df["EnclosedPorch"]
                    + sample_clean_df["3SsnPorch"] + sample_clean_df["ScreenPorch"])
        pd.testing.assert_series_equal(result["TotalPorchSF"].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)

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