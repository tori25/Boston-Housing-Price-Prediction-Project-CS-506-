import io
import os
import pandas as pd
import requests
import urllib3

# Suppress the InsecureRequestWarning from verify=False — intentional SSL bypass for macOS
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

RAW_DATA_PATH = "data/raw/fy2025_property_assessment.csv"
PROCESSED_DIR = "data/processed"
INSPECTION_PATH = "data/processed/data_inspection.txt"

FY2025_URL = (
    "https://data.boston.gov/dataset/e02c44d2-3c64-459c-8fe2-e1ce5f38a035"
    "/resource/6b7e460e-33f6-4e61-80bc-1bef2e73ac54"
    "/download/fy2025-property-assessment-data_12_30_2024.csv"
)


def collect_data():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_DATA_PATH):
        print("Downloading FY2025 Boston Property Assessment data...")
        # requests handles AWS S3 signed-URL redirects correctly; verify=False for macOS SSL
        response = requests.get(FY2025_URL, verify=False, timeout=120)
        response.raise_for_status()
        # io.BytesIO wraps raw bytes so pandas can read them without writing a temp file first
        df = pd.read_csv(io.BytesIO(response.content), low_memory=False)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"Saved to: {RAW_DATA_PATH}")
    else:
        print(f"Raw data already exists: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH, low_memory=False)

    print(f"Raw dataset shape: {df.shape}")
    inspect_raw_data(df)
    return df


def inspect_raw_data(df):
    target_col = "TOTAL_VALUE"
    leakage_cols = {"LAND_VALUE", "BLDG_VALUE", "GROSS_TAX"}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    usable_numeric = [c for c in numeric_cols if c not in leakage_cols and c != target_col]

    lines = [
        "Boston Property Assessment FY2025 — Raw Data Inspection",
        "=" * 60,
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Target column: {target_col}",
        "",
        "Land use breakdown (LU):",
    ]
    if "LU" in df.columns:
        for lu, count in df["LU"].value_counts().head(12).items():
            lines.append(f"  {lu:6s}: {count:>7,}")
    lines += [
        "",
        f"Missing values (total): {df.isnull().sum().sum():,}",
        "",
        f"Numeric columns in raw data, excluding leakage and target ({len(usable_numeric)}):",
    ]
    for col in usable_numeric:
        lines.append(f"  {col}")
    lines += [
        "",
        f"Columns excluded as data leakage: {sorted(leakage_cols)}",
        "",
        "Descriptive statistics:",
        df[usable_numeric + [target_col]].describe().to_string(),
    ]

    report = "\n".join(lines) + "\n"
    print(report)
    with open(INSPECTION_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Inspection saved to: {INSPECTION_PATH}")


if __name__ == "__main__":
    collect_data()
