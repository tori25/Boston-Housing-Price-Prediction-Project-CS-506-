import os
import pandas as pd
import matplotlib.pyplot as plt

ZILLOW_PATH = "data/raw/zillow.csv"
PLOTS_DIR = "plots"

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    zillow = pd.read_csv(ZILLOW_PATH)

    # Keep only Boston
    zillow_boston = zillow[zillow["RegionName"] == "Boston, MA"].copy()

    if zillow_boston.empty:
        raise ValueError("No rows found for 'Boston, MA' in Zillow dataset.")

    print("\nBoston rows:")
    print(zillow_boston.head())

    print("\nBoston dataset shape:")
    print(zillow_boston.shape)

    # Columns that are metadata, not dates
    non_date_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]

    # Keep only date columns
    date_cols = [col for col in zillow_boston.columns if col not in non_date_cols]

    # Convert wide format to long format
    boston_long = zillow_boston.melt(
        id_vars=["RegionName"],
        value_vars=date_cols,
        var_name="Date",
        value_name="MedianSalePrice"
    )

    # Convert types
    boston_long["Date"] = pd.to_datetime(boston_long["Date"], errors="coerce")
    boston_long["MedianSalePrice"] = pd.to_numeric(
        boston_long["MedianSalePrice"], errors="coerce"
    )

    # Drop bad rows and sort by date
    boston_long = boston_long.dropna(subset=["Date", "MedianSalePrice"])
    boston_long = boston_long.sort_values("Date")

    print("\nBoston long format preview:")
    print(boston_long.head())

    # Plot Boston price trend
    plt.figure(figsize=(10, 5))
    plt.plot(boston_long["Date"], boston_long["MedianSalePrice"])
    plt.title("Boston Median Sale Price Over Time (Zillow)")
    plt.xlabel("Date")
    plt.ylabel("Median Sale Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/zillow_boston_trend.png")
    plt.show()

    return boston_long

if __name__ == "__main__":
    main()