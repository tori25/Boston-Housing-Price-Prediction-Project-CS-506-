import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

INPUT_PATH = "data/processed/train_features.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.txt"
TARGET_COLUMN = "SalePrice"


def calculate_euclidean_distance(df: pd.DataFrame) -> float:
    # Features used to compare physical similarity between two houses
    features = ["GrLivArea", "TotalBsmtSF", "GarageArea", "TotalBathrooms"]

    # Check that all required columns exist
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required columns for Euclidean distance: {missing_features}"
        )

    # Select two example houses
    house1 = df.loc[0, features].values
    house2 = df.loc[1, features].values

    # Compute Euclidean distance
    return np.sqrt(np.sum((house1 - house2) ** 2))


def calculate_manhattan_distance(df: pd.DataFrame) -> float:
    # Features used to compare physical similarity between two houses
    features = ["GrLivArea", "TotalBsmtSF", "GarageArea", "TotalBathrooms"]

    # Check that all required columns exist
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required columns for Manhattan distance: {missing_features}"
        )

    # Select two example houses
    house1 = df.loc[0, features].values
    house2 = df.loc[1, features].values

    # Compute Manhattan distance
    return np.sum(np.abs(house1 - house2))


def evaluate_model(y_true, y_pred):
    # Return common regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main() -> None:
    # Load processed dataset with cleaned and engineered features
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    # Make sure target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # Calculate example distances between house 0 and house 1
    euclidean_distance = calculate_euclidean_distance(df)
    manhattan_distance = calculate_manhattan_distance(df)

    print(f"Euclidean distance between house 0 and house 1: {euclidean_distance:.2f}")
    print(f"Manhattan distance between house 0 and house 1: {manhattan_distance:.2f}")

    # Separate predictors and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Detect numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Number of numeric features: {len(numeric_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")

    # Use one train/test split for Linear Regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # -----------------------------
    # Linear Regression model
    # -----------------------------

    # Preprocess numeric columns
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Preprocess categorical columns
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing by column type
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline for Linear Regression
    linear_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    # Train and predict
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    # Evaluate Linear Regression
    linear_mae, linear_rmse, linear_r2 = evaluate_model(y_test, linear_predictions)

    # -----------------------------
    # KNN models
    # -----------------------------

    # KNN works best with numeric features only
    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()

    X_knn = numeric_df.drop(columns=[TARGET_COLUMN])
    y_knn = numeric_df[TARGET_COLUMN]

    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
        X_knn, y_knn, test_size=0.2, random_state=42
    )

    # KNN with Euclidean distance
    knn_euclidean_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5, metric="euclidean")),
        ]
    )

    knn_euclidean_model.fit(X_train_knn, y_train_knn)
    knn_euclidean_predictions = knn_euclidean_model.predict(X_test_knn)

    knn_euclidean_mae, knn_euclidean_rmse, knn_euclidean_r2 = evaluate_model(
        y_test_knn, knn_euclidean_predictions
    )

    # KNN with Manhattan distance
    knn_manhattan_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5, metric="manhattan")),
        ]
    )

    knn_manhattan_model.fit(X_train_knn, y_train_knn)
    knn_manhattan_predictions = knn_manhattan_model.predict(X_test_knn)

    knn_manhattan_mae, knn_manhattan_rmse, knn_manhattan_r2 = evaluate_model(
        y_test_knn, knn_manhattan_predictions
    )

    # -----------------------------
    # Clustering (K-Means)
    # -----------------------------

    # Use numeric features only
    X_cluster = df.select_dtypes(include=["int64", "float64"]).drop(columns=[TARGET_COLUMN])

    # Scale features (IMPORTANT for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    df["Cluster"] = clusters

    print("\nCluster distribution:")
    print(pd.Series(clusters).value_counts())

    print("\nAverage SalePrice per cluster:")
    print(df.groupby("Cluster")[TARGET_COLUMN].mean())

    # -----------------------------
    # Save results
    # -----------------------------

    results = (
        "Model Comparison Results\n\n"
        f"Euclidean distance between house 0 and house 1: {euclidean_distance:.2f}\n"
        f"Manhattan distance between house 0 and house 1: {manhattan_distance:.2f}\n\n"
        "Linear Regression Results\n"
        f"MAE: {linear_mae:.2f}\n"
        f"RMSE: {linear_rmse:.2f}\n"
        f"R^2: {linear_r2:.4f}\n\n"
        "KNN Regression (Euclidean) Results\n"
        f"MAE: {knn_euclidean_mae:.2f}\n"
        f"RMSE: {knn_euclidean_rmse:.2f}\n"
        f"R^2: {knn_euclidean_r2:.4f}\n\n"
        "KNN Regression (Manhattan) Results\n"
        f"MAE: {knn_manhattan_mae:.2f}\n"
        f"RMSE: {knn_manhattan_rmse:.2f}\n"
        f"R^2: {knn_manhattan_r2:.4f}\n"
        f"\nCluster count:\n{pd.Series(clusters).value_counts()}\n"
    )

    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Save metrics to a text file
    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(results)

    # Print results
    print("\n" + results)
    print(f"Results saved to: {MODEL_RESULTS_PATH}")


if __name__ == "__main__":
    main()