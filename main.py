import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/train.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    # Scatter plot 1
    plt.figure()
    sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])
    plt.title("GrLivArea vs SalePrice")
    plt.show()

    # Scatter plot 2
    plt.figure()
    sns.scatterplot(x=df["LotArea"], y=df["SalePrice"])
    plt.title("LotArea vs SalePrice")
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df)
    plt.title("SalePrice vs OverallQual")
    plt.show()


if __name__ == "__main__":
    main()