import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

ZILLOW_PATH = "data/raw/zillow.csv"
PLOTS_DIR = "plots"
PROCESSED_PATH = "data/processed/zillow_boston.csv"


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    zillow = pd.read_csv(ZILLOW_PATH)

    zillow_boston = zillow[zillow["RegionName"] == "Boston, MA"].copy()

    if zillow_boston.empty:
        raise ValueError("No rows found for 'Boston, MA' in Zillow dataset.")

    print(f"Boston row shape: {zillow_boston.shape}")

    # Metadata columns — everything else is a monthly date column
    non_date_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    date_cols = [col for col in zillow_boston.columns if col not in non_date_cols]

    # Reshape from wide (one row, many date columns) to long (one row per month)
    boston_long = zillow_boston.melt(
        id_vars=["RegionName"],
        value_vars=date_cols,
        var_name="Date",
        value_name="MedianSalePrice"
    )

    boston_long["Date"] = pd.to_datetime(boston_long["Date"], errors="coerce")
    boston_long["MedianSalePrice"] = pd.to_numeric(boston_long["MedianSalePrice"], errors="coerce")
    boston_long = boston_long.dropna(subset=["Date", "MedianSalePrice"])
    boston_long = boston_long.sort_values("Date").reset_index(drop=True)

    # Derived time columns saved to processed CSV for reference — Zillow data is context only, not used in modeling
    boston_long["Year"] = boston_long["Date"].dt.year
    boston_long["Month"] = boston_long["Date"].dt.month
    boston_long["Quarter"] = boston_long["Date"].dt.quarter
    boston_long["TimeIndex"] = range(len(boston_long))  # sequential integer for charting
    boston_long["PriceGrowth"] = boston_long["MedianSalePrice"].pct_change()
    boston_long["RollingMean3"] = boston_long["MedianSalePrice"].rolling(window=3).mean()
    boston_long["RollingMean12"] = boston_long["MedianSalePrice"].rolling(window=12).mean()

    print("\nBoston long format preview:")
    print(boston_long[["Date", "MedianSalePrice", "Year", "Month",
                        "TimeIndex", "PriceGrowth", "RollingMean3", "RollingMean12"]].head(15))

    boston_long.to_csv(PROCESSED_PATH, index=False)
    print(f"\nProcessed Zillow dataset saved to: {PROCESSED_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(boston_long["Date"], boston_long["MedianSalePrice"])
    plt.title("Boston Median Sale Price Over Time (Zillow)")
    plt.xlabel("Date")
    plt.ylabel("Median Sale Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/zillow_boston_trend.png", dpi=100)
    plt.close()
    print("Saved: zillow_boston_trend.png")

    return boston_long


if __name__ == "__main__":
    main()
