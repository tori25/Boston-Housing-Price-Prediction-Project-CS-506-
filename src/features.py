import os
import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/boston_clean.csv"
OUTPUT_PATH = "data/processed/train_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    created_features = []

    # 2025 - YR_BUILT — older buildings tend to command lower prices
    if "YR_BUILT" in df.columns:
        df["AGE"] = 2025 - df["YR_BUILT"]
        created_features.append("AGE")

    # 1 if property was remodeled after original construction, 0 if never touched
    if {"YR_BUILT", "YR_REMODEL"}.issubset(df.columns):
        df["IS_REMODELED"] = (df["YR_REMODEL"] > df["YR_BUILT"]).astype(int)
        created_features.append("IS_REMODELED")

    # FULL_BTH + 0.5 * HLF_BTH — weighted bath count; half baths add less value
    if {"FULL_BTH", "HLF_BTH"}.issubset(df.columns):
        df["BATH_TOTAL"] = df["FULL_BTH"] + 0.5 * df["HLF_BTH"]
        created_features.append("BATH_TOTAL")

    # log(LIVING_AREA) — square footage is right-skewed; log compresses the tail
    if "LIVING_AREA" in df.columns:
        df["LOG_LIVING_AREA"] = np.log1p(df["LIVING_AREA"])
        created_features.append("LOG_LIVING_AREA")

    # log(LAND_SF) — lot size is right-skewed
    if "LAND_SF" in df.columns:
        df["LOG_LAND_SF"] = np.log1p(df["LAND_SF"])
        created_features.append("LOG_LAND_SF")

    # LIVING_AREA / TT_RMS — average room size; larger rooms signal higher quality
    if {"LIVING_AREA", "TT_RMS"}.issubset(df.columns):
        safe_rooms = df["TT_RMS"].replace(0, np.nan)
        df["AREA_PER_ROOM"] = (df["LIVING_AREA"] / safe_rooms).fillna(df["LIVING_AREA"])
        created_features.append("AREA_PER_ROOM")

    # BED_RMS * BATH_TOTAL — luxury homes score high on both; BATH_TOTAL created above
    if {"BED_RMS", "BATH_TOTAL"}.issubset(df.columns):
        df["BED_BATH"] = df["BED_RMS"] * df["BATH_TOTAL"]
        created_features.append("BED_BATH")

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
