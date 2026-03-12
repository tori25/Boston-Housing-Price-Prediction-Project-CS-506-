import pandas as pd
import os

RAW_DATA_PATH = "data/raw/train.csv"
PROCESSED_DATA_PATH = "data/processed/train_clean.csv"

def clean_data():
    # Load dataset
    df = pd.read_csv(RAW_DATA_PATH)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(0)

    # Remove columns with too many missing values
    columns_to_drop = ["PoolQC", "Fence", "MiscFeature", "Alley"]
    df = df.drop(columns=columns_to_drop)

    # Remove obvious outliers (example: very large living area)
    if "GrLivArea" in df.columns:
        df = df[df["GrLivArea"] < 4000]

    # Ensure processed folder exists
    os.makedirs("data/processed", exist_ok=True)

    # Save cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Cleaned data saved to:", PROCESSED_DATA_PATH)

    # Print initial data information
    print("\nColumn Data Types:")
    print(df.dtypes)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    print("\nCategorical Column Unique Values:")
    print(df[categorical_cols].nunique())

if __name__ == "__main__":
    clean_data()