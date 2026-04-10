import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/boston_clean.csv"
PLOTS_DIR = "plots"


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded cleaned dataset: {df.shape}")

    # ── Plot 1: Target distribution ───────────────────────────────────────────
    # Claim: medv is approximately normally distributed after removing the
    # $50k censored cap, with a slight right skew toward higher-value neighborhoods.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["medv"], bins=30, edgecolor="black", color="#4878CF", alpha=0.8)
    ax.axvline(df["medv"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: ${df['medv'].mean():.1f}k")
    ax.axvline(df["medv"].median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median: ${df['medv'].median():.1f}k")
    ax.set_xlabel("Median Home Value ($1,000s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Boston Median Home Values (medv)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/target_distribution.png", dpi=100)
    plt.close()
    print("Saved: target_distribution.png")

    # ── Plot 2: Scatter — rm vs medv ──────────────────────────────────────────
    # Claim: More rooms strongly predicts higher home value.
    # rm is the single strongest positive predictor (correlation ~0.70).
    corr_rm = df["rm"].corr(df["medv"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["rm"], df["medv"], alpha=0.5, s=20, color="#4878CF")
    m, b = np.polyfit(df["rm"], df["medv"], 1)
    x_line = np.linspace(df["rm"].min(), df["rm"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, label="Trend line")
    ax.set_xlabel("Average Rooms per Dwelling (rm)")
    ax.set_ylabel("Median Home Value ($1,000s)")
    ax.set_title(f"Rooms vs Home Value  (r = {corr_rm:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_rm_vs_medv.png", dpi=100)
    plt.close()
    print("Saved: scatter_rm_vs_medv.png")

    # ── Plot 3: Scatter — lstat vs medv ──────────────────────────────────────
    # Claim: Higher % lower-status population strongly predicts lower home value.
    # lstat is the strongest negative predictor (correlation ~-0.74).
    corr_lstat = df["lstat"].corr(df["medv"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["lstat"], df["medv"], alpha=0.5, s=20, color="#D65F5F")
    m, b = np.polyfit(df["lstat"], df["medv"], 1)
    x_line = np.linspace(df["lstat"].min(), df["lstat"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5, label="Trend line")
    ax.set_xlabel("% Lower-Status Population (lstat)")
    ax.set_ylabel("Median Home Value ($1,000s)")
    ax.set_title(f"Poverty Rate vs Home Value  (r = {corr_lstat:.2f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_lstat_vs_medv.png", dpi=100)
    plt.close()
    print("Saved: scatter_lstat_vs_medv.png")

    # ── Plot 4: Boxplot — chas vs medv ────────────────────────────────────────
    # Claim: Neighborhoods bordering the Charles River (chas=1) have higher
    # median home values than those that don't (chas=0).
    chas_0 = df[df["chas"] == 0]["medv"]
    chas_1 = df[df["chas"] == 1]["medv"]
    median_0 = chas_0.median()
    median_1 = chas_1.median()

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [chas_0, chas_1],
        tick_labels=[f"Not River-Adjacent\n(chas=0, n={len(chas_0)})",
                     f"River-Adjacent\n(chas=1, n={len(chas_1)})"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    bp["boxes"][0].set_facecolor("#4878CF")
    bp["boxes"][1].set_facecolor("#6ACC65")
    ax.set_ylabel("Median Home Value ($1,000s)")
    ax.set_title("Charles River Adjacency vs Home Value (chas)")
    ax.annotate(f"Median: ${median_0:.1f}k", xy=(1, median_0), xytext=(1.15, median_0 + 1.5),
                fontsize=9, color="#4878CF")
    ax.annotate(f"Median: ${median_1:.1f}k", xy=(2, median_1), xytext=(2.05, median_1 + 1.5),
                fontsize=9, color="green")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/boxplot_chas_vs_medv.png", dpi=100)
    plt.close()
    print("Saved: boxplot_chas_vs_medv.png")

    # ── Plot 5: Correlation heatmap ───────────────────────────────────────────
    # Claim: rm and lstat are the strongest predictors of medv.
    # nox, ptratio, and indus also show notable negative correlation.
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    # Annotate each cell with the correlation value
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=100)
    plt.close()
    print("Saved: correlation_heatmap.png")

    # ── Feature correlations with target ─────────────────────────────────────
    print("\nFeature correlations with medv (sorted):")
    corr_target = df.corr()["medv"].drop("medv").sort_values()
    for feat, val in corr_target.items():
        bar = "█" * int(abs(val) * 20)
        direction = "+" if val > 0 else "-"
        print(f"  {feat:10s}  {direction}{abs(val):.3f}  {bar}")

    print("\nAll visualizations saved to plots/")


if __name__ == "__main__":
    main()
