import os
import pandas as pd

INPUT_PATH = "data/processed/train_clean.csv"
OUTPUT_PATH = "data/processed/train_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Total square footage
    if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(df.columns):
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Total bathrooms
    bathroom_cols = {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}
    if bathroom_cols.issubset(df.columns):
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

    # House age at sale
    if {"YrSold", "YearBuilt"}.issubset(df.columns):
        df["HouseAgeAtSale"] = df["YrSold"] - df["YearBuilt"]

    # Years since remodel at sale
    if {"YrSold", "YearRemodAdd"}.issubset(df.columns):
        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]

    # Total porch area
    porch_cols = {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}
    if porch_cols.issubset(df.columns):
        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["EnclosedPorch"]
            + df["3SsnPorch"]
            + df["ScreenPorch"]
        )

    return df


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    feature_df = create_features(df)

    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {feature_df.shape}")


if __name__ == "__main__":
    main()