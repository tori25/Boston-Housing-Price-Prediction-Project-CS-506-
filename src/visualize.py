import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

INPUT_PATH = "data/processed/boston_clean.csv"
PLOTS_DIR = "plots"

LU_LABELS = {1: "R1 (1-fam)", 2: "R2 (2-fam)", 3: "R3 (3-fam)", 4: "R4 (4+fam)", 5: "CD (condo)"}


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded cleaned dataset: {df.shape}")

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["TOTAL_VALUE"] / 1_000, bins=40, edgecolor="black", color="#4878CF", alpha=0.8)
    ax.axvline(df["TOTAL_VALUE"].mean() / 1_000, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: ${df['TOTAL_VALUE'].mean() / 1_000:.0f}k")
    ax.axvline(df["TOTAL_VALUE"].median() / 1_000, color="orange", linestyle="--", linewidth=1.5,
               label=f"Median: ${df['TOTAL_VALUE'].median() / 1_000:.0f}k")
    ax.set_xlabel("Total Assessed Value ($1,000s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Boston Property Assessed Values (FY2025)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/target_distribution.png", dpi=100)
    plt.close()
    print("Saved: target_distribution.png")

    # Living area vs value (strongest numeric predictor)
    # Cap at 5,000 sq ft — R4 multi-family buildings reach 43k sq ft (whole-building total),
    # which would compress 99% of residential data into an unreadable bottom-left cluster
    AREA_CAP = 5_000
    df_area = df[df["LIVING_AREA"] <= AREA_CAP]
    corr_area = df_area["LIVING_AREA"].corr(df_area["TOTAL_VALUE"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_area["LIVING_AREA"], df_area["TOTAL_VALUE"] / 1_000, alpha=0.3, s=8, color="#4878CF")
    m, b = np.polyfit(df_area["LIVING_AREA"], df_area["TOTAL_VALUE"] / 1_000, 1)
    x_line = np.linspace(df_area["LIVING_AREA"].min(), df_area["LIVING_AREA"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, label="Trend line")
    ax.set_xlabel("Living Area (sq ft)")
    ax.set_ylabel("Total Assessed Value ($1,000s)")
    ax.set_title(f"Living Area vs Assessed Value  (r = {corr_area:.2f}, LIVING_AREA ≤ {AREA_CAP:,} sq ft)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_area_vs_value.png", dpi=100)
    plt.close()
    print("Saved: scatter_area_vs_value.png")

    # Bedrooms vs value
    # Cap at 8 — R4 multi-family buildings record total bedrooms across all units,
    # so a 4-family with 5 beds each shows as 20, distorting the trend line
    BED_CAP = 8
    df_beds = df[df["BED_RMS"] <= BED_CAP]
    corr_beds = df["BED_RMS"].corr(df["TOTAL_VALUE"])  # full dataset, not the capped subset
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_beds["BED_RMS"], df_beds["TOTAL_VALUE"] / 1_000, alpha=0.3, s=8, color="#D65F5F")
    m, b = np.polyfit(df_beds["BED_RMS"], df_beds["TOTAL_VALUE"] / 1_000, 1)
    x_line = np.linspace(df_beds["BED_RMS"].min(), df_beds["BED_RMS"].max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.5, label="Trend line")
    ax.set_xlabel("Number of Bedrooms")
    ax.set_ylabel("Total Assessed Value ($1,000s)")
    ax.set_title(f"Bedrooms vs Assessed Value  (r = {corr_beds:.2f}, BED_RMS ≤ {BED_CAP})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/scatter_beds_vs_value.png", dpi=100)
    plt.close()
    print("Saved: scatter_beds_vs_value.png")

    # Boxplot: property type vs value — filter to LU codes actually present in data
    if "LU" in df.columns:
        labels = [LU_LABELS[k] for k in sorted(LU_LABELS) if k in df["LU"].values]
        lu_groups = [df[df["LU"] == k]["TOTAL_VALUE"] / 1_000 for k in sorted(LU_LABELS) if k in df["LU"].values]

        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(lu_groups, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        colors = ["#4878CF", "#6ACC65", "#D65F5F", "#E8A838", "#A87FD6"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Total Assessed Value ($1,000s)")
        ax.set_title("Assessed Value by Property Type (FY2025)")
        plt.tight_layout()
        plt.savefig(f"{PLOTS_DIR}/boxplot_lu_vs_value.png", dpi=100)
        plt.close()
        print("Saved: boxplot_lu_vs_value.png")

    # Correlation heatmap
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")  # white text on dark cells
    ax.set_title("Feature Correlation Heatmap — Original Cleaned Features (before ZIP encoding)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=100)
    plt.close()
    print("Saved: correlation_heatmap.png")

    print("\nFeature correlations with TOTAL_VALUE (sorted):")
    corr_target = corr["TOTAL_VALUE"].drop("TOTAL_VALUE").sort_values()
    for feat, val in corr_target.items():
        bar = "█" * int(abs(val) * 20)
        direction = "+" if val > 0 else "-"
        print(f"  {feat:20s}  {direction}{abs(val):.3f}  {bar}")

    print("\nAll visualizations saved to plots/")


if __name__ == "__main__":
    main()
