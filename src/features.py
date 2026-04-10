import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/boston_clean.csv"
OUTPUT_PATH = "data/processed/train_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    created_features = []

    # ── 1. CRIME_LOG = log(1 + crim) ─────────────────────────────────────────
    # Crime rate is heavily right-skewed — a handful of high-crime tracts
    # dominate. Log-transform compresses the tail so models treat a jump
    # from 0.1 to 1.0 the same as a jump from 1.0 to 10.0.
    # Maps to: "crime rate" from the feature list.
    if "crim" in df.columns:
        df["CRIME_LOG"] = np.log1p(df["crim"])
        created_features.append("CRIME_LOG")

    # ── 2. ROOM_SQ = rm² ─────────────────────────────────────────────────────
    # The value premium for extra rooms is non-linear: going from 6 to 7 rooms
    # adds more value than going from 4 to 5. Squaring captures this curve.
    # Maps to: "number of rooms / total living area" from the feature list.
    if "rm" in df.columns:
        df["ROOM_SQ"] = df["rm"] ** 2
        created_features.append("ROOM_SQ")

    # ── 3. TAX_PER_ROOM = tax / rm ───────────────────────────────────────────
    # Ownership cost per unit of space. A high tax rate on a small home is
    # a worse deal than the same rate on a large home.
    # Maps to: "tax per square foot" from the feature list.
    if {"tax", "rm"}.issubset(df.columns):
        df["TAX_PER_ROOM"] = df["tax"] / df["rm"]
        created_features.append("TAX_PER_ROOM")

    # ── 4. LSTAT_PER_ROOM = lstat / rm ───────────────────────────────────────
    # Poverty density relative to home size. A small home in a high-poverty
    # neighborhood is penalized more than a large home in the same neighborhood.
    # Combines the two strongest individual predictors (r = -0.76 and r = +0.69).
    if {"lstat", "rm"}.issubset(df.columns):
        df["LSTAT_PER_ROOM"] = df["lstat"] / df["rm"]
        created_features.append("LSTAT_PER_ROOM")

    # ── 5. POLLUTION_PROXIMITY = nox / dis ───────────────────────────────────
    # Pollution concentration per unit of distance from employment centers.
    # A neighborhood that is close to industry AND heavily polluted is worse
    # than one that is far away with the same pollution level.
    # Maps to: "distance-related ratios" from the feature list.
    if {"nox", "dis"}.issubset(df.columns):
        df["POLLUTION_PROXIMITY"] = df["nox"] / df["dis"]
        created_features.append("POLLUTION_PROXIMITY")

    # ── 6. SCHOOL_INDEX = ptratio × lstat ────────────────────────────────────
    # School quality (pupil-teacher ratio) weighted by neighborhood poverty.
    # Bad schools in poor areas compound each other — both individually
    # have r ≈ -0.51 and -0.76 with medv. Their product captures the
    # double-penalty of low school quality in low-income neighborhoods.
    # Maps to: "school/district info" from the feature list.
    if {"ptratio", "lstat"}.issubset(df.columns):
        df["SCHOOL_INDEX"] = df["ptratio"] * df["lstat"]
        created_features.append("SCHOOL_INDEX")

    # ── 7. AGE_DIST = age × dis ──────────────────────────────────────────────
    # Old housing stock that is also far from employment centers.
    # Each factor alone is a moderate negative predictor; together they
    # describe the least desirable type of neighborhood: aging and remote.
    # Maps to: "age of house" + "distance to employment" from the feature list.
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
