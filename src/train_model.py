import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_PATH = "data/processed/train_features.csv"
MODEL_RESULTS_PATH = "data/processed/model_results.txt"
PLOTS_DIR = "plots"
TARGET_COLUMN = "TOTAL_VALUE"


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

    # Find optimal k for KNN via 5-fold CV on the training set only
    k_candidates = [3, 5, 7, 10, 15, 20, 30, 50]
    cv_rmses = []
    print("\nTuning KNN — 5-fold CV on training set:")
    for k in k_candidates:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=k, metric="euclidean"))
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        rmse = -scores.mean()
        cv_rmses.append(rmse)
        print(f"  k={k:3d}  CV RMSE={rmse:,.0f}")

    best_k = k_candidates[cv_rmses.index(min(cv_rmses))]
    print(f"Best k: {best_k}  (CV RMSE={min(cv_rmses):,.0f})\n")

    plt.figure(figsize=(8, 5))
    plt.plot(k_candidates, [r / 1_000 for r in cv_rmses], marker="o", color="#4878CF")
    plt.axvline(x=best_k, color="#D65F5F", linestyle="--", linewidth=1.5, label=f"Best k={best_k}")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("CV RMSE ($1,000s)")
    plt.title("KNN Hyperparameter Tuning: k vs CV RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/knn_tuning.png", dpi=100)
    plt.close()
    print("Saved: knn_tuning.png")

    knn_label = f"KNN (k={best_k})"

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
            ("model", Lasso(alpha=1000.0, max_iter=10000))
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_features="sqrt", random_state=42, n_jobs=-1
        ),
        knn_label: Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=best_k, metric="euclidean"))
        ]),
    }

    # 5-fold CV on training set — more reliable than a single split
    print("Cross-validating all models (5-fold on training set):")
    cv_rows = []
    for name, model in models.items():
        mae_scores  = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
        rmse_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        r2_scores   =  cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_rows.append({
            "Model":    name,
            "MAE":      mae_scores.mean(),  "MAE_std":  mae_scores.std(),
            "RMSE":     rmse_scores.mean(), "RMSE_std": rmse_scores.std(),
            "R²":       r2_scores.mean(),   "R²_std":   r2_scores.std(),
        })
        print(f"  {name:25s}  MAE={mae_scores.mean():,.0f}±{mae_scores.std():,.0f}"
              f"  RMSE={rmse_scores.mean():,.0f}±{rmse_scores.std():,.0f}"
              f"  R²={r2_scores.mean():.4f}±{r2_scores.std():.4f}")

    results_df = pd.DataFrame(cv_rows)
    best = results_df.sort_values("RMSE").iloc[0]

    # Train final models on full X_train so we have predictions for the diagnostic plots
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    interpretations = {
        "Linear Regression": "Baseline. Fits a global plane through all features. No regularization.",
        "Ridge Regression":  "Adds L2 penalty — shrinks correlated coefficients, reduces overfitting.",
        "Lasso Regression":  "Adds L1 penalty — forces some coefficients to zero (implicit feature selection).",
        "Decision Tree":     "Splits on single features. Captures non-linearity but high variance.",
        "Random Forest":     "200 trees, each on a random feature subset. Reduces variance via averaging.",
        knn_label:           f"Predicts from the {best_k} nearest properties by Euclidean distance (k tuned by 5-fold CV).",
    }

    report_lines = [
        "Boston Property Assessment FY2025 — Model Results",
        "=" * 60,
        "",
        f"Dataset shape: {df.shape}",
        f"Training set: {X_train.shape}   Test set: {X_test.shape}",
        "Evaluation: 5-fold cross-validation on training set (mean ± std)",
        "",
        "Story: linear baseline → regularization → non-linear → distance-based",
        "-" * 60,
    ]
    for _, row in results_df.iterrows():
        note = interpretations.get(row["Model"], "")
        report_lines.append(
            f"{row['Model']:25s}  "
            f"MAE={row['MAE']:>10,.0f}±{row['MAE_std']:>7,.0f}  "
            f"RMSE={row['RMSE']:>10,.0f}±{row['RMSE_std']:>7,.0f}  "
            f"R²={row['R²']:.4f}±{row['R²_std']:.4f}"
        )
        report_lines.append(f"  → {note}")
        report_lines.append("")

    report_lines += [
        "-" * 60,
        f"Best model (lowest CV RMSE): {best['Model']}",
        f"  MAE:  ${best['MAE']:,.0f} ± ${best['MAE_std']:,.0f}",
        f"  RMSE: ${best['RMSE']:,.0f} ± ${best['RMSE_std']:,.0f}",
        f"  R²:   {best['R²']:.4f} ± {best['R²_std']:.4f}",
    ]

    report = "\n".join(report_lines) + "\n"
    with open(MODEL_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Results saved to: {MODEL_RESULTS_PATH}")

    # blue = linear models, red = Decision Tree, purple = Random Forest, green = KNN
    colors = ["#4878CF", "#4878CF", "#4878CF", "#D65F5F", "#A87FD6", "#6ACC65"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        ax.bar(results_df["Model"], results_df[metric],
               yerr=results_df[f"{metric}_std"], color=colors,
               capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
        ax.set_title(metric)
        ax.set_ylabel(f"{metric} (mean ± std, 5-fold CV)")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Model Comparison: Linear → Regularized → Decision Tree → Random Forest → KNN")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison.png", dpi=100)
    plt.close()
    print("Saved: model_comparison.png")

    best_name = best["Model"]
    best_preds = predictions[best_name]

    # Red dashed diagonal = perfect predictions; points close to it = low error
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test / 1_000, best_preds / 1_000, alpha=0.4, s=10)
    min_val = min(float(y_test.min()), float(best_preds.min())) / 1_000
    max_val = max(float(y_test.max()), float(best_preds.max())) / 1_000
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    plt.xlabel("Actual Assessed Value ($1,000s)")
    plt.ylabel("Predicted Assessed Value ($1,000s)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/actual_vs_predicted.png", dpi=100)
    plt.close()
    print("Saved: actual_vs_predicted.png")

    residuals = y_test - best_preds
    plt.figure(figsize=(8, 6))
    plt.scatter(best_preds / 1_000, residuals / 1_000, alpha=0.4, s=10)
    plt.axhline(y=0, color="r", linestyle="--", linewidth=1.5)
    plt.xlabel("Predicted Assessed Value ($1,000s)")
    plt.ylabel("Residuals ($1,000s)")
    plt.title(f"Residual Plot — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/residual_plot.png", dpi=100)
    plt.close()
    print("Saved: residual_plot.png")

    # Decision Tree is not in a Pipeline so plot_tree receives it directly
    dt_model = models["Decision Tree"]
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=8, max_depth=2)
    plt.title("Decision Tree Visualization (Top Levels)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/decision_tree.png", dpi=100)
    plt.close()
    print("Saved: decision_tree.png")

    # K-Means: exploration only — segments properties by size and bedroom count
    # Cap at 5,000 sq ft so multi-family outliers don't collapse all clusters into the left edge
    print("Running K-Means clustering...")
    cluster_features = ["LIVING_AREA", "BED_RMS"]
    df_cluster = df[cluster_features + [TARGET_COLUMN]].dropna().copy()
    df_cluster = df_cluster[df_cluster["LIVING_AREA"] <= 5_000]
    df_cluster = df_cluster.sample(n=10000, random_state=42)

    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df_cluster[cluster_features])

    k_range = range(2, 11)
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(X_cluster)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_cluster, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, marker="o", color="#4878CF")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(list(k_range), sil_scores, marker="o", color="#6ACC65")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/kmeans_elbow_silhouette.png", dpi=100)
    plt.close()
    print("Saved: kmeans_elbow_silhouette.png")

    best_k = int(k_range.start + sil_scores.index(max(sil_scores)))
    print(f"Best k by silhouette score: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_cluster)

    cluster_colors = ["#D65F5F", "#4878CF", "#6ACC65", "#E8A838", "#9B59B6",
                      "#F39C12", "#1ABC9C", "#E74C3C", "#3498DB", "#2ECC71"]
    fig, ax = plt.subplots(figsize=(9, 6))
    for cluster_id in range(best_k):
        mask = df_cluster["Cluster"] == cluster_id
        ax.scatter(
            df_cluster.loc[mask, "LIVING_AREA"],
            df_cluster.loc[mask, TARGET_COLUMN] / 1_000,
            c=cluster_colors[cluster_id],
            label=f"Cluster {cluster_id}",
            alpha=0.4,
            s=10,
        )
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Total Assessed Value ($1,000s)")
    ax.set_title(f"K-Means Clustering (k={best_k}): Boston Property Segments")
    ax.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/kmeans_clusters.png", dpi=100)
    plt.close()
    print("Saved: kmeans_clusters.png")

    lr_pipeline = models["Linear Regression"]
    lr_coefs = lr_pipeline.named_steps["model"].coef_
    coef_df = pd.DataFrame({"Feature": X.columns.tolist(), "Coefficient": lr_coefs})
    coef_df["AbsCoef"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("AbsCoef", ascending=False).head(20).sort_values("AbsCoef", ascending=True)

    colors_coef = ["#D65F5F" if c < 0 else "#4878CF" for c in coef_df["Coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient Value (standardized features)")
    ax.set_title("Linear Regression — Top 20 Coefficients\n(blue = positive effect, red = negative effect)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/coefficient_plot.png", dpi=100)
    plt.close()
    print("Saved: coefficient_plot.png")
    print("K-Means finished.")

    # Random Forest feature importances — top 20 so ZIP dummies don't crowd out the signal
    rf_model = models["Random Forest"]
    importance_df = pd.DataFrame({
        "Feature":    X.columns.tolist(),
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#A87FD6")
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/rf_feature_importance.png", dpi=100)
    plt.close()
    print("Saved: rf_feature_importance.png")

    return results_df


if __name__ == "__main__":
    main()
