import os
import ssl
import pandas as pd

RAW_DATA_PATH = "data/raw/boston.csv"
INSPECTION_PATH = "data/processed/data_inspection.txt"


def inspect_raw_data(df: pd.DataFrame) -> str:
    lines = []

    lines += [
        "=" * 60,
        "RAW DATA INSPECTION",
        "=" * 60,
        "",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "",
        "Columns:",
        *[f"  {col}" for col in df.columns],
        "",
        "Data types:",
    ]
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col:12s}  {dtype}")
    lines.append("")

    missing = df.isnull().sum()
    total_missing = missing.sum()
    lines += [f"Missing values (total: {total_missing}):"]
    if total_missing == 0:
        lines.append("  None — dataset is complete.")
    else:
        for col, n in missing[missing > 0].items():
            pct = 100 * n / len(df)
            lines.append(f"  {col:12s}  {n} ({pct:.1f}%)")
    lines.append("")

    lines += ["Summary statistics:", df.describe().to_string(), ""]

    target_col = "medv"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    unusable_cols = {
        "black": "Racially charged feature from the 1978 paper — dropped in modern applications.",
    }
    usable_numeric = [c for c in numeric_cols if c not in unusable_cols and c != target_col]

    lines += [
        "-" * 60,
        "Column notes:",
        "",
        f"  Target column : {target_col}",
        f"    → Median home value in $1,000s. This is what we predict.",
        "",
        f"  Numeric columns used in modeling ({len(usable_numeric)}):",
        *[f"    {col}" for col in usable_numeric],
        "",
    ]

    if categorical_cols:
        lines += [
            f"  Categorical columns ({len(categorical_cols)}):",
            *[f"    {col}" for col in categorical_cols],
            "",
        ]
    else:
        lines += [
            "  Categorical columns: None",
            "    → All features are numeric. No one-hot encoding needed.",
            "",
        ]

    lines += [f"  Columns flagged as unusable ({len(unusable_cols)}):"]
    for col, reason in unusable_cols.items():
        lines.append(f"    {col}: {reason}")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


def collect_data():
    # Required on macOS — statsmodels download fails SSL verification without this
    ssl._create_default_https_context = ssl._create_unverified_context

    import statsmodels.api as sm

    print("Downloading Boston Housing Dataset via statsmodels...")
    boston = sm.datasets.get_rdataset("Boston", "MASS").data

    report = inspect_raw_data(boston)

    os.makedirs("data/processed", exist_ok=True)
    with open(INSPECTION_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Inspection notes saved to: {INSPECTION_PATH}")

    os.makedirs("data/raw", exist_ok=True)
    boston.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")

    return boston


if __name__ == "__main__":
    collect_data()
