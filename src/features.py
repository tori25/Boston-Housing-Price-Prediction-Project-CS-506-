import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/boston_clean.csv"
OUTPUT_PATH = "data/processed/train_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    created_features = []

    # log(1 + crim) — crime is heavily right-skewed; log compresses the tail
    if "crim" in df.columns:
        df["CRIME_LOG"] = np.log1p(df["crim"])
        created_features.append("CRIME_LOG")

    # rm² — value premium for extra rooms is non-linear; squaring captures the curve
    if "rm" in df.columns:
        df["ROOM_SQ"] = df["rm"] ** 2
        created_features.append("ROOM_SQ")

    # tax / rm — ownership cost per unit of space
    if {"tax", "rm"}.issubset(df.columns):
        df["TAX_PER_ROOM"] = df["tax"] / df["rm"]
        created_features.append("TAX_PER_ROOM")

    # lstat / rm — combines the two strongest predictors (r = -0.76 and r = +0.69)
    if {"lstat", "rm"}.issubset(df.columns):
        df["LSTAT_PER_ROOM"] = df["lstat"] / df["rm"]
        created_features.append("LSTAT_PER_ROOM")

    # nox / dis — pollution per unit of distance from employment centers
    if {"nox", "dis"}.issubset(df.columns):
        df["POLLUTION_PROXIMITY"] = df["nox"] / df["dis"]
        created_features.append("POLLUTION_PROXIMITY")

    # ptratio × lstat — bad schools in poor areas compound each other
    if {"ptratio", "lstat"}.issubset(df.columns):
        df["SCHOOL_INDEX"] = df["ptratio"] * df["lstat"]
        created_features.append("SCHOOL_INDEX")

    # age × dis — old housing stock that is also far from employment
    if {"age", "dis"}.issubset(df.columns):
        df["AGE_DIST"] = df["age"] * df["dis"]
        created_features.append("AGE_DIST")

    print(f"Created {len(created_features)} features: {created_features}")
    return df


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Input dataset shape: {df.shape}")

    feature_df = create_features(df)

    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")
    print(f"Output dataset shape: {feature_df.shape}")


if __name__ == "__main__":
    main()
