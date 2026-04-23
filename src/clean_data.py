import pandas as pd
import os

RAW_DATA_PATH = "data/raw/fy2025_property_assessment.csv"
PROCESSED_DATA_PATH = "data/processed/boston_clean.csv"

# Residential land use codes — exclude commercial, industrial, tax-exempt, CD = Residential Condominium Unit , 73,394 records out of 183,445 totaletc.
RESIDENTIAL_LU = {"R1", "R2", "R3", "R4", "CD"}

# Leakage: LAND_VALUE + BLDG_VALUE = TOTAL_VALUE; GROSS_TAX and SFYI_VALUE are derived from it
LEAKAGE_COLS = ["LAND_VALUE", "BLDG_VALUE", "GROSS_TAX", "SFYI_VALUE"]

# Admin/ID/mailing columns with no predictive value
DROP_COLS = [
    "PID", "CM_ID", "GIS_ID", "ST_NUM", "ST_NUM2", "ST_NAME", "UNIT_NUM", "CITY",
    "BLDG_SEQ", "NUM_BLDGS", "LUC", "LU_DESC", "BLDG_TYPE",
    "OWNER", "MAIL_ADDRESSEE", "MAIL_STREET_ADDRESS", "MAIL_CITY", "MAIL_STATE", "MAIL_ZIP_CODE",
    # Condo-specific columns — sparse for non-condo properties
    "CD_FLOOR", "RES_UNITS", "COM_UNITS", "RC_UNITS", "ORIENTATION", "CORNER_UNIT",
    # Too granular / sparse style codes
    "BDRM_COND", "BTHRM_STYLE1", "BTHRM_STYLE2", "BTHRM_STYLE3",
    "KITCHEN_TYPE", "KITCHEN_STYLE1", "KITCHEN_STYLE2", "KITCHEN_STYLE3",
    "HEAT_TYPE", "HEAT_SYSTEM", "AC_TYPE", "ROOF_COVER", "ROOF_STRUCTURE",
    "INT_WALL", "EXT_FNISHED", "INT_COND", "EXT_COND", "PROP_VIEW",
]

# Ordinal condition encoding: first letter of "X - Description" format
CONDITION_MAP = {"E": 5, "G": 4, "A": 3, "F": 2, "P": 1}
LU_MAP = {"R1": 1, "R2": 2, "R3": 3, "R4": 4, "CD": 5}

# Columns stored as "1,234" strings in the raw CSV — need comma stripping before numeric ops
CURRENCY_COLS = ["LAND_SF", "TOTAL_VALUE"]


def _parse_currency(series: pd.Series) -> pd.Series:
    """Strip commas and dollar signs, then coerce to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
        errors="coerce"
    )


def clean_data():
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    original_shape = df.shape
    print(f"Raw data loaded: {original_shape}")

    # Strip whitespace from column names (GROSS_TAX has leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Parse currency-formatted columns before any filtering
    for col in CURRENCY_COLS:
        if col in df.columns:
            df[col] = _parse_currency(df[col])

    # Keep residential properties only
    if "LU" in df.columns:
        before = len(df)
        df = df[df["LU"].isin(RESIDENTIAL_LU)].copy()
        print(f"Filtered to residential (R1/R2/R3/R4/CD): {before:,} → {len(df):,} rows")

    # Drop admin, ID, and leakage columns
    cols_to_drop = [c for c in DROP_COLS + LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} admin/leakage columns")

    df = df.drop_duplicates()
    print(f"After deduplication: {df.shape}")

    # Must have a real assessed value
    if "TOTAL_VALUE" in df.columns:
        before = len(df)
        df = df[df["TOTAL_VALUE"] > 0]
        print(f"Removed zero TOTAL_VALUE: {before - len(df)} removed")

    # Must have a real living area — zero means missing data
    if "LIVING_AREA" in df.columns:
        before = len(df)
        df = df[df["LIVING_AREA"] > 0]
        print(f"Removed zero LIVING_AREA: {before - len(df)} removed")

    # Year built must be plausible
    if "YR_BUILT" in df.columns:
        before = len(df)
        df = df[(df["YR_BUILT"] > 1700) & (df["YR_BUILT"] <= 2025)]
        print(f"Removed invalid YR_BUILT: {before - len(df)} removed")

    # Encode LU as ordinal integer
    if "LU" in df.columns:
        df["LU"] = df["LU"].map(LU_MAP).fillna(0).astype(int)

    # OWN_OCC: Y = owner-occupied (1), everything else = 0
    if "OWN_OCC" in df.columns:
        df["OWN_OCC"] = (df["OWN_OCC"] == "Y").astype(int)

    # OVERALL_COND: "A - Average" → extract first letter → map to integer
    if "OVERALL_COND" in df.columns:
        first_letter = df["OVERALL_COND"].astype(str).str.strip().str[0]
        df["OVERALL_COND"] = first_letter.map(CONDITION_MAP).fillna(3).astype(int)

    # ZIP_CODE to integer
    if "ZIP_CODE" in df.columns:
        df["ZIP_CODE"] = pd.to_numeric(df["ZIP_CODE"], errors="coerce")

    # Fill remaining numeric missing values with column median
    numeric_cols = df.select_dtypes(include="number").columns
    missing = df[numeric_cols].isnull().sum()
    for col in missing[missing > 0].index:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  Filled '{col}' with median ({median_val:.1f})")

    # Drop any remaining non-numeric columns (can't use in modeling without encoding)
    non_numeric = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        df = df.drop(columns=non_numeric)
        print(f"Dropped remaining non-numeric columns: {non_numeric}")

    # Remove top 1% of TOTAL_VALUE — extreme luxury properties distort the model
    if "TOTAL_VALUE" in df.columns:
        cap = df["TOTAL_VALUE"].quantile(0.99)
        before = len(df)
        df = df[df["TOTAL_VALUE"] <= cap]
        print(f"Removed top 1% TOTAL_VALUE (>${cap:,.0f}): {before - len(df)} removed")

    print(f"\nOriginal shape: {original_shape}")
    print(f"Cleaned shape:  {df.shape}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nCleaned data saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    clean_data()
