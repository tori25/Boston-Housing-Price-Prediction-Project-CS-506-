import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUT_PATH = "data/processed/train_features.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.txt"
TARGET_COLUMN = "SalePrice"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results = (
        "Linear Regression Results\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R^2: {r2:.4f}\n"
    )

    os.makedirs("data/processed", exist_ok=True)
    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(results)

    print(results)
    print(f"Results saved to: {MODEL_RESULTS_PATH}")


if __name__ == "__main__":
    main()