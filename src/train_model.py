import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "data/processed/train_clean.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.txt"
PLOTS_DIR = "plots"
TARGET_COLUMN = "SalePrice"


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    # Ensure folders exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load cleaned dataset
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    # Apply feature engineering
    from src.features import create_features
    df = create_features(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=False)

    # Split features and target
    X = df_encoded.drop(columns=[TARGET_COLUMN])
    y = df_encoded[TARGET_COLUMN]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train decision tree model
    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    mae, rmse, r2 = evaluate_model(y_test, y_pred)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Save results to text file
    results = (
        "Decision Tree Regression Results\n\n"
        f"Dataset shape: {df.shape}\n"
        f"Training set shape: {X_train.shape}\n"
        f"Test set shape: {X_test.shape}\n\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R^2: {r2:.4f}\n\n"
        "Top 10 Feature Importances\n"
        f"{importance_df.head(10).to_string(index=False)}\n"
    )

    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(results)

    print("\n" + results)
    print(f"Results saved to: {MODEL_RESULTS_PATH}")

    # -----------------------------
    # Plot 1: Actual vs Predicted
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title("Actual vs Predicted House Prices")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png")
    plt.show()

    # -----------------------------
    # Plot 2: Residuals
    # -----------------------------
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, linestyle="--")
    plt.xlabel("Predicted SalePrice")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/residual_plot.png")
    plt.show()

    # -----------------------------
    # Plot 3: Top 10 Feature Importances
    # -----------------------------
    top_features = importance_df.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importance.png")
    plt.show()

    # -----------------------------
    # Plot 4: Decision Tree
    # -----------------------------
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=2
    )
    plt.title("Decision Tree Visualization (Top Levels)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/decision_tree.png")
    plt.show()


if __name__ == "__main__":
    main()