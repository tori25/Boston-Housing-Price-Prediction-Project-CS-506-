import pandas as pd
import matplotlib.pyplot as plt

ZILLOW_PATH = "data/raw/zillow.csv"

def main():
    zillow = pd.read_csv(ZILLOW_PATH)

    # Keep only Boston
    zillow_boston = zillow[zillow["RegionName"] == "Boston, MA"]

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
    boston_long["Date"] = pd.to_datetime(boston_long["Date"])
    boston_long["MedianSalePrice"] = pd.to_numeric(
        boston_long["MedianSalePrice"], errors="coerce"
    )

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
    plt.show()


if __name__ == "__main__":
    main()