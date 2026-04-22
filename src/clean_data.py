import pandas as pd
import os

RAW_DATA_PATH = "data/raw/boston.csv"
PROCESSED_DATA_PATH = "data/processed/boston_clean.csv"


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)
    original_shape = df.shape

    df = df.drop_duplicates()
    print(f"After dropping duplicates: {df.shape}")

    # Drop unnamed index columns that appear when a CSV is saved with index=True
    id_cols = [col for col in df.columns if col.lower() in ("id", "unnamed: 0")]
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"Dropped ID columns: {id_cols}")

    # 'black' is a racially charged feature from the 1978 paper — excluded from all modeling
    if "black" in df.columns:
        df = df.drop(columns=["black"])
        print("Dropped column: 'black'")

    # chas/rad/tax loaded as float64 by pandas but are conceptually integers
    for col in ("chas", "rad", "tax"):
        if col in df.columns:
            df[col] = df[col].astype(int)
    print("Data types fixed: chas → int, rad → int, tax → int")

    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if missing_cols.empty:
        print("Missing values: none found.")
    else:
        for col in missing_cols.index:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled '{col}' with median ({median_val:.3f})")

    # medv == 50 is a censored cap — these are not real observations
    before = len(df)
    if "medv" in df.columns:
        df = df[df["medv"] < 50]
    print(f"Removed censored medv==50 rows: {before - len(df)} removed")

    # All Boston features are physically non-negative; negatives indicate data errors
    numeric_cols = df.select_dtypes(include="number").columns
    before = len(df)
    for col in numeric_cols:
        if col != "medv":
            df = df[df[col] >= 0]
    print(f"Removed rows with negative feature values: {before - len(df)} removed")

    # Top 1% crime tracts are orders of magnitude above the rest — distort KNN distances
    if "crim" in df.columns:
        crim_99 = df["crim"].quantile(0.99)
        before = len(df)
        df = df[df["crim"] <= crim_99]
        print(f"Removed top 1% crim outliers (>{crim_99:.2f}): {before - len(df)} removed")

    print(f"\nOriginal shape: {original_shape}")
    print(f"Cleaned shape:  {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nCleaned data saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    clean_data()
