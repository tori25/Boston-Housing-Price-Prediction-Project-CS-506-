import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = "data/processed/train_features.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.txt"
PLOTS_DIR = "plots"
TARGET_COLUMN = "medv"


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape:     {X_test.shape}")

    # Story order: linear baseline → regularized → non-linear → distance-based
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0))
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=5000))
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        "KNN (k=10)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=10, metric="euclidean"))
        ]),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae, rmse, r2 = evaluate_model(y_test, y_pred)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})
        predictions[name] = y_pred
        print(f"{name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    results_df = pd.DataFrame(results)
    best = results_df.sort_values("RMSE").iloc[0]

    interpretations = {
        "Linear Regression": "Baseline. Fits a global plane through all features. No regularization.",
        "Ridge Regression":  "Adds L2 penalty — shrinks correlated coefficients, reduces overfitting.",
        "Lasso Regression":  "Adds L1 penalty — forces some coefficients to zero (implicit feature selection).",
        "Decision Tree":     "Splits on single features. Captures non-linearity but high variance.",
        "KNN (k=10)":        "Predicts from the 10 nearest neighborhoods by Euclidean distance.",
    }

    report_lines = [
        "Boston Housing Price Prediction — Model Results",
        "=" * 60,
        "",
        f"Dataset shape: {df.shape}",
        f"Training set: {X_train.shape}   Test set: {X_test.shape}",
        "",
        "Story: linear baseline → regularization → non-linear → distance-based",
        "-" * 60,
    ]
    for _, row in results_df.iterrows():
        note = interpretations.get(row["Model"], "")
        report_lines.append(
            f"{row['Model']:25s}  MAE={row['MAE']:>6.2f}  RMSE={row['RMSE']:>6.2f}  R²={row['R²']:.4f}"
        )
        report_lines.append(f"  → {note}")
        report_lines.append("")

    report_lines += [
        "-" * 60,
        f"Best model (lowest RMSE): {best['Model']}",
        f"  MAE:  {best['MAE']:.2f}",
        f"  RMSE: {best['RMSE']:.2f}",
        f"  R²:   {best['R²']:.4f}",
    ]

    report = "\n".join(report_lines) + "\n"
    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Results saved to: {MODEL_RESULTS_PATH}")

    # blue = linear models, red = Decision Tree, green = KNN
    colors = ["#4878CF", "#4878CF", "#4878CF", "#D65F5F", "#6ACC65"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        ax.bar(results_df["Model"], results_df[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Model Comparison: Linear → Regularized → Decision Tree → KNN")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=100)
    plt.close()

    best_name = best["Model"]
    best_preds = predictions[best_name]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_preds, alpha=0.6)
    min_val = min(float(y_test.min()), float(best_preds.min()))
    max_val = max(float(y_test.max()), float(best_preds.max()))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    plt.xlabel("Actual Median Home Value ($1000s)")
    plt.ylabel("Predicted Median Home Value ($1000s)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png", dpi=100)
    plt.close()

    residuals = y_test - best_preds
    plt.figure(figsize=(8, 6))
    plt.scatter(best_preds, residuals, alpha=0.6)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Median Home Value ($1000s)")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/residual_plot.png", dpi=100)
    plt.close()

    dt_model = models["Decision Tree"]
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=8, max_depth=2)
    plt.title("Decision Tree Visualization (Top Levels)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/decision_tree.png", dpi=100)
    plt.close()

    # K-Means: exploration only — segments neighborhoods, not used for prediction
    cluster_features = ["rm", "lstat"]  # two strongest predictors
    df_cluster = df[cluster_features + [TARGET_COLUMN]].dropna().copy()

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster[cluster_features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_cluster)

    cluster_colors = ["#D65F5F", "#4878CF", "#6ACC65", "#E8A838"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id in range(4):
        mask = df_cluster["Cluster"] == cluster_id
        ax.scatter(
            df_cluster.loc[mask, "rm"],
            df_cluster.loc[mask, TARGET_COLUMN],
            c=cluster_colors[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.6,
            s=30,
        )
    ax.set_xlabel("Average Rooms per Dwelling (rm)")
    ax.set_ylabel("Median Home Value ($1000s)")
    ax.set_title("K-Means Clustering (k=4): Boston Neighborhood Segments")
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/kmeans_clusters.png", dpi=100)
    plt.close()
    print("K-Means cluster plot saved.")

    lr_pipeline = models["Linear Regression"]
    lr_coefs = lr_pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame({"Feature": X.columns.tolist(), "Coefficient": lr_coefs})
    coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoef", ascending=True)  # sorted by magnitude for readability

    colors_coef = ["#D65F5F" if c < 0 else "#4878CF" for c in coef_df["Coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value (standardized features)")
    ax.set_title("Linear Regression Coefficients\n(blue = positive effect, red = negative effect)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/coefficient_plot.png", dpi=100)
    plt.close()
    print("Coefficient plot saved.")

    return results_df


if __name__ == "__main__":
    main()
