import pandas as pd
import os

RAW_DATA_PATH = "data/raw/train.csv"
PROCESSED_DATA_PATH = "data/processed/train_clean.csv"

def clean_data():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop ID column
    df = df.drop(columns=["Id"], errors="ignore")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Drop columns with too many missing values / very low usefulness
    columns_to_drop = ["PoolQC", "MiscFeature", "Alley", "Fence"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Fill categorical columns where missing usually means "feature not present"
    none_cols = [
        "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond"
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Fill numeric columns
    if "LotFrontage" in df.columns:
        df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    # Fill any remaining categorical missing values with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fill any remaining numeric missing values with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Remove obvious outliers
    if "GrLivArea" in df.columns:
        df = df[df["GrLivArea"] < 4000]

    # Print cleaned data information
    print("\nColumn Data Types:")
    print(df.dtypes)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    categorical_cols = df.select_dtypes(include=["object"]).columns
    print("\nCategorical Column Unique Values:")
    print(df[categorical_cols].nunique())

    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nCleaned data saved to:", PROCESSED_DATA_PATH)
    print("Cleaned dataset shape:", df.shape)

if __name__ == "__main__":
    clean_data()