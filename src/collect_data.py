import os
import ssl
import pandas as pd

RAW_DATA_PATH = "data/raw/boston.csv"
INSPECTION_PATH = "data/processed/data_inspection.txt"


def inspect_raw_data(df: pd.DataFrame) -> str:
    """
    Inspect raw data before any cleaning.
    Prints and returns a report covering shape, columns, dtypes,
    missing values, summary stats, and column classification notes.
    """
    lines = []

    # ── Shape ─────────────────────────────────────────────────────────────────
    lines += [
        "=" * 60,
        "RAW DATA INSPECTION",
        "=" * 60,
        "",
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        "",
    ]

    # ── Columns ───────────────────────────────────────────────────────────────
    lines += [
        "Columns:",
        *[f"  {col}" for col in df.columns],
        "",
    ]

    # ── Data types ────────────────────────────────────────────────────────────
    lines += ["Data types:"]
    for col, dtype in df.dtypes.items():
        lines.append(f"  {col:12s}  {dtype}")
    lines.append("")

    # ── Missing values ────────────────────────────────────────────────────────
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

    # ── Summary statistics ────────────────────────────────────────────────────
    lines += [
        "Summary statistics:",
        df.describe().to_string(),
        "",
    ]

    # ── Column classification notes ───────────────────────────────────────────
    target_col = "medv"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    # Columns considered unusable or ethically problematic
    unusable_cols = {
        "black": "Racially charged feature from the 1978 paper — dropped in modern applications.",
    }

    lines += [
        "-" * 60,
        "Column notes:",
        "",
        f"  Target column : {target_col}",
        f"    → Median home value in $1,000s. This is what we predict.",
        "",
        f"  Numeric columns ({len(numeric_cols)}):",
        *[f"    {col}" for col in numeric_cols],
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
    """
    Download the Boston Housing Dataset via statsmodels (Harrison & Rubinfeld, 1978).
    Inspects raw data before saving to data/raw/boston.csv.
    Saves inspection notes to data/processed/data_inspection.txt.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    import statsmodels.api as sm

    print("Downloading Boston Housing Dataset via statsmodels...")
    boston = sm.datasets.get_rdataset("Boston", "MASS").data

    # Inspect before saving anything
    report = inspect_raw_data(boston)

    # Save inspection notes
    os.makedirs("data/processed", exist_ok=True)
    with open(INSPECTION_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Inspection notes saved to: {INSPECTION_PATH}")

    # Save raw data
    os.makedirs("data/raw", exist_ok=True)
    boston.to_csv(RAW_DATA_PATH, index=False)
    print(f"Raw dataset saved to: {RAW_DATA_PATH}")

    return boston


if __name__ == "__main__":
    collect_data()
