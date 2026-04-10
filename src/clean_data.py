import pandas as pd
import os

RAW_DATA_PATH = "data/raw/boston.csv"
PROCESSED_DATA_PATH = "data/processed/boston_clean.csv"


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH)
    original_shape = df.shape

    # ── Drop duplicate rows ───────────────────────────────────────────────────
    df = df.drop_duplicates()
    print(f"After dropping duplicates: {df.shape}")

    # ── Remove irrelevant ID columns ──────────────────────────────────────────
    # Boston dataset has no ID column, but we drop any unnamed index columns
    # that may appear if the CSV was saved with an index
    id_cols = [col for col in df.columns if col.lower() in ("id", "unnamed: 0")]
    if id_cols:
        df = df.drop(columns=id_cols)
        print(f"Dropped ID columns: {id_cols}")

    # ── Fix data types ────────────────────────────────────────────────────────
    # chas is a binary dummy (0/1) — cast to int
    if "chas" in df.columns:
        df["chas"] = df["chas"].astype(int)

    # rad is an ordinal index (1–24) — cast to int
    if "rad" in df.columns:
        df["rad"] = df["rad"].astype(int)

    # tax is a whole-number rate — cast to int
    if "tax" in df.columns:
        df["tax"] = df["tax"].astype(int)

    print("Data types fixed: chas → int, rad → int, tax → int")

    # ── Handle missing values ─────────────────────────────────────────────────
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]

    if missing_cols.empty:
        print("Missing values: none found.")
    else:
        print(f"Missing values found:\n{missing_cols}")
        # Fill numeric columns with median
        for col in missing_cols.index:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled '{col}' with median ({median_val:.3f})")

    # ── Remove obviously broken rows ──────────────────────────────────────────
    # medv == 50 are censored values — the dataset artificially caps at $50k
    # These are not real observations and bias the regression ceiling
    before = len(df)
    if "medv" in df.columns:
        df = df[df["medv"] < 50]
    print(f"Removed censored medv==50 rows: {before - len(df)} removed")

    # Negative values are physically impossible for any feature
    numeric_cols = df.select_dtypes(include="number").columns
    before = len(df)
    for col in numeric_cols:
        if col != "medv":
            df = df[df[col] >= 0]
    print(f"Removed rows with negative feature values: {before - len(df)} removed")

    # ── Remove extreme outliers ───────────────────────────────────────────────
    # Top 1% of crime rate — a few tracts have rates orders of magnitude
    # above the rest and distort distance-based models
    if "crim" in df.columns:
        crim_99 = df["crim"].quantile(0.99)
        before = len(df)
        df = df[df["crim"] <= crim_99]
        print(f"Removed top 1% crim outliers (>{crim_99:.2f}): {before - len(df)} removed")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nOriginal shape: {original_shape}")
    print(f"Cleaned shape:  {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nCleaned data saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    clean_data()
