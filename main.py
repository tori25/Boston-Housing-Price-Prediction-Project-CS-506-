import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/train.csv"
FEATURE_PATH = "data/processed/train_features.csv"


def main():
    # Load raw dataset for exploratory analysis
    df = pd.read_csv(DATA_PATH)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    # Scatter plot: living area vs price
    plt.figure()
    sns.scatterplot(x=df["GrLivArea"], y=df["SalePrice"])
    plt.title("GrLivArea vs SalePrice")
    plt.xlabel("GrLivArea")
    plt.ylabel("SalePrice")
    plt.show()

    # Scatter plot: lot area vs price
    plt.figure()
    sns.scatterplot(x=df["LotArea"], y=df["SalePrice"])
    plt.title("LotArea vs SalePrice")
    plt.xlabel("LotArea")
    plt.ylabel("SalePrice")
    plt.show()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])

    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

    # Boxplot: quality vs price
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df)
    plt.title("SalePrice vs OverallQual")
    plt.show()

    # ---------------------------------------------------
    # Euclidean Distance Visualization
    # ---------------------------------------------------

    print("\nComputing Euclidean distances between houses...")

    df_features = pd.read_csv(FEATURE_PATH)

    features = ["GrLivArea", "TotalBsmtSF", "GarageArea", "TotalBathrooms"]

    X = df_features[features]

    # Scale features so distance is not dominated by large variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute distance from house 0 to all other houses
    distances = np.sqrt(np.sum((X_scaled - X_scaled[0]) ** 2, axis=1))

    # Plot histogram of distances
    plt.figure()
    plt.hist(distances, bins=30)
    plt.title("Distribution of Euclidean Distances from House 0")
    plt.xlabel("Distance")
    plt.ylabel("Number of Houses")
    plt.show()
 
    # Compute Manhattan distance
    manhattan_distances = np.sum(np.abs(X_scaled - X_scaled[0]), axis=1)

    plt.figure()
    plt.hist(manhattan_distances, bins=30)
    plt.title("Distribution of Manhattan Distances from House 0")
    plt.xlabel("Distance")
    plt.ylabel("Number of Houses")
    plt.show()

if __name__ == "__main__":
    main()