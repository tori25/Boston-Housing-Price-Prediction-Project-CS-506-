import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/boston_clean.csv"
OUTPUT_PATH = "data/processed/train_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    created_features = []

    # 2025 - YR_BUILT — older buildings tend to command lower prices
    # YR_BUILT is dropped afterwards to avoid perfect collinearity with AGE
    if "YR_BUILT" in df.columns:
        df["AGE"] = 2025 - df["YR_BUILT"]
        df = df.drop(columns=["YR_BUILT"])
        created_features.append("AGE")

    # 1 if property was remodeled after original construction, 0 if never touched
    if {"AGE", "YR_REMODEL"}.issubset(df.columns):
        yr_built_equiv = 2025 - df["AGE"]
        df["IS_REMODELED"] = (df["YR_REMODEL"] > yr_built_equiv).astype(int)
        created_features.append("IS_REMODELED")

    # FULL_BTH + 0.5 * HLF_BTH — weighted bath count; half baths add less value
    if {"FULL_BTH", "HLF_BTH"}.issubset(df.columns):
        df["BATH_TOTAL"] = df["FULL_BTH"] + 0.5 * df["HLF_BTH"]
        created_features.append("BATH_TOTAL")

    # LIVING_AREA / TT_RMS — average room size; larger rooms signal higher quality
    if {"LIVING_AREA", "TT_RMS"}.issubset(df.columns):
        safe_rooms = df["TT_RMS"].replace(0, np.nan)
        df["AREA_PER_ROOM"] = (df["LIVING_AREA"] / safe_rooms).fillna(df["LIVING_AREA"])
        created_features.append("AREA_PER_ROOM")

    # BED_RMS * BATH_TOTAL — luxury homes score high on both; BATH_TOTAL created above
    if {"BED_RMS", "BATH_TOTAL"}.issubset(df.columns):
        df["BED_BATH"] = df["BED_RMS"] * df["BATH_TOTAL"]
        created_features.append("BED_BATH")

    # ZIP codes are nominal categories, not ordinal numbers — one-hot encode them
    # so each ZIP gets its own coefficient instead of implying 02199 > 02101
    if "ZIP_CODE" in df.columns:
        zip_dummies = pd.get_dummies(
            df["ZIP_CODE"].fillna(0).astype(int).astype(str),
            prefix="ZIP",
            drop_first=True,
        ).astype(int)
        df = df.drop(columns=["ZIP_CODE"])
        df = pd.concat([df, zip_dummies], axis=1)
        created_features.append(f"ZIP_dummies({len(zip_dummies.columns)})")

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
