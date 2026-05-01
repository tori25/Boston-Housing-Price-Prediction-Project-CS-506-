# Boston Housing Price Prediction

> **Presentation video:** *(YouTube link to be added before 5/1 deadline)*

---

## How to Build and Run

**Prerequisites:** Python 3.9+

```bash
make install      # install all dependencies
make collect      # download FY2025 Boston Property Assessment dataset  →  data/raw/fy2025_property_assessment.csv
make clean        # clean raw data                                       →  data/processed/boston_clean.csv
make features     # engineer features                                    →  data/processed/train_features.csv
make visualize    # generate EDA plots                                   →  plots/
make train        # train all models                                     →  plots/ + data/processed/model_results.txt
```

Or run the entire pipeline at once:

```bash
make all
```

To run the test suite:

```bash
make test
# or
python3 -m pytest src/tests/test_project.py -v
```

A GitHub Actions workflow (`.github/workflows/test.yml`) runs all 22 tests automatically on every push and pull request to `main`.

### Output locations

| Output | Path |
|--------|------|
| Raw dataset | `data/raw/fy2025_property_assessment.csv` |
| Cleaned dataset | `data/processed/boston_clean.csv` |
| Engineered features | `data/processed/train_features.csv` |
| Model comparison results | `data/processed/model_results.txt` |
| Plots | `plots/` |

### Project structure

```
├── data/
│   ├── raw/
│   │   ├── fy2025_property_assessment.csv   # Boston FY2025 Property Assessment (downloaded by collect_data.py)
│   │   └── zillow.csv                       # Zillow Boston median sale price data (pre-downloaded)
│   └── processed/
│       ├── boston_clean.csv                 # Cleaned dataset
│       ├── train_features.csv               # Dataset with engineered features
│       ├── model_results.txt                # Model comparison output
│       └── data_inspection.txt              # Raw data inspection report
├── src/
│   ├── collect_data.py                      # Downloads FY2025 Boston Property Assessment data
│   ├── clean_data.py                        # Data cleaning pipeline
│   ├── features.py                          # Feature engineering
│   ├── visualize.py                         # EDA visualizations
│   ├── train_model.py                       # Model training and evaluation
│   ├── zillow_analysis.py                   # Zillow trend analysis
│   └── tests/
│       └── test_project.py                  # Unit tests (pytest)
├── plots/                                   # Generated visualizations
├── main.py                                  # Full pipeline entry point
├── Makefile
└── requirements.txt
```

---

## Project Goal

Predict **`TOTAL_VALUE`** — the total assessed value of a residential property in Boston — using the City of Boston's FY2025 Property Assessment dataset, downloaded from [data.boston.gov](https://data.boston.gov/dataset/property-assessment).

The dataset contains **183,445 property records** (as of December 2024) covering all taxable and non-taxable parcels in the City of Boston. After filtering to residential properties and cleaning, the modeling dataset contains **132,611 rows × 52 features** (16 original + 4 engineered + 32 ZIP one-hot encoded columns). Unlike the classic 1978 Boston Housing Dataset which described census-tract averages, this dataset contains actual individual property records — each row is one house, condo, or apartment building.

Performance is measured with **MAE**, **RMSE**, and **R²**. All error values are in dollars.

As supplemental real-world context, Zillow median sale price data for Boston, MA (2018–2026) is visualized alongside the modeled assessed values.

---

## Data Collection

Implemented in `src/collect_data.py`.

**Primary dataset:** City of Boston FY2025 Property Assessment data, downloaded via direct URL from [data.boston.gov](https://data.boston.gov/dataset/property-assessment). Saved to `data/raw/fy2025_property_assessment.csv`.

| Feature | Description |
|---------|-------------|
| `LIVING_AREA` | Living area square footage |
| `LAND_SF` | Parcel land area in square feet |
| `GROSS_AREA` | Gross floor area |
| `BED_RMS` | Number of bedrooms |
| `FULL_BTH` | Number of full bathrooms |
| `HLF_BTH` | Number of half bathrooms |
| `TT_RMS` | Total number of rooms |
| `FIREPLACES` | Number of fireplaces |
| `NUM_PARKING` | Number of parking spaces |
| `YR_BUILT` | Year property was built |
| `YR_REMODEL` | Year property was last remodeled |
| `RES_FLOOR` | Number of residential stories |
| `LU` | Land use type (R1/R2/R3/R4/CD) — encoded as 1–5 |
| `ZIP_CODE` | Zip code of parcel — one-hot encoded into 32 binary columns during feature engineering |
| `OWN_OCC` | Owner-occupied (Y=1, N=0) |
| `OVERALL_COND` | Overall condition (E=5, G=4, A=3, F=2, P=1) |
| `TOTAL_VALUE` | **Target** — Total assessed value in dollars |

**Excluded as data leakage:** `LAND_VALUE`, `BLDG_VALUE` (components of TOTAL_VALUE), `GROSS_TAX` (derived from TOTAL_VALUE).

**Secondary dataset:** Zillow Research Data — Median Sale Price by Metro Area (2018–2026). Used for a trend visualization only, not for modeling.

---

## Data Cleaning

Implemented in `src/clean_data.py`. Final cleaned dataset: **132,611 rows × 18 columns**, zero missing values.

### Step 1 — Parse currency-formatted columns
**What:** Strip commas and dollar signs from `LAND_SF`, `TOTAL_VALUE`, and other value columns stored as strings like `"1,150"` or `"$9,252.42"`.
**Why:** The raw CSV stores these as formatted strings, not numbers. Pandas reads them as objects and they must be converted to float before any filtering.

### Step 2 — Filter to residential properties
**What:** Keep only rows where `LU` is in {R1, R2, R3, R4, CD}. Drops 46,934 non-residential rows (commercial, industrial, tax-exempt, etc.).
**Why:** Commercial and industrial properties have different value drivers than residential. Mixing them would confuse every model.

### Step 3 — Drop admin, ID, and leakage columns
**What:** Drop 47 columns — admin (PID, OWNER, mailing address), ID columns (GIS_ID), condo-specific columns (CD_FLOOR, RES_UNITS), granular style codes (BTHRM_STYLE1, KITCHEN_TYPE), and leakage columns.
**Why:** Admin columns have no relationship to value; leakage columns (LAND_VALUE + BLDG_VALUE = TOTAL_VALUE) would give any model perfect information and produce meaningless results.

### Step 4 — Remove duplicate rows
**What:** Drop exact duplicate rows.
**Why:** Duplicates inflate training confidence in those data points.

### Step 5 — Remove zero assessed values
**What:** Remove rows where `TOTAL_VALUE == 0`.
**Why:** Tax-exempt or unassessed properties have no useful assessed value to predict.

### Step 6 — Remove zero living area
**What:** Remove 87 rows where `LIVING_AREA == 0`.
**Why:** Zero living area indicates missing data — no valid property has zero square footage.

### Step 7 — Remove invalid year built
**What:** Remove 76 rows where `YR_BUILT` is outside 1700–2025.
**Why:** Values like 0 or 20198 are data entry errors; year built is used to compute the AGE feature.

### Step 8 — Encode categorical columns
**What:** `LU` → integer map (R1=1 … CD=5); `OWN_OCC` → binary (Y=1); `OVERALL_COND` → ordinal (E=5 … P=1 from "X - Description" format).
**Why:** All sklearn models require numeric inputs. Ordinal encoding preserves the ordering of condition grades.

### Step 9 — Fill missing values with median
**What:** 10 numeric columns had missing values (e.g., `YR_REMODEL`, `LAND_SF`). Filled with column median.
**Why:** Median imputation is robust to outliers; preserves the distribution better than mean for skewed columns.

### Step 10 — Remove top 1% of TOTAL_VALUE
**What:** Remove 1,340 rows where `TOTAL_VALUE > $4,454,100`.
**Why:** Ultra-luxury properties (hotels, large apartment complexes) are in a different market segment and distort linear model coefficients and KNN distances.

---

## Feature Extraction

Implemented in `src/features.py`. Output: `data/processed/train_features.csv` (**132,611 rows × 53 columns** — 16 original + 4 engineered + 32 ZIP dummy columns + target).

### Feature correlations with TOTAL_VALUE (original features)

| Feature | Correlation | Notes |
|---------|------------|-------|
| `LIVING_AREA` | +0.437 | Strongest predictor — more space = more value |
| `FULL_BTH` | +0.424 | Full bathrooms strongly signal quality |
| `GROSS_AREA` | +0.324 | Total building footprint |
| `FIREPLACES` | +0.268 | Luxury amenity |
| `TT_RMS` | +0.262 | Total rooms |
| `BED_RMS` | +0.253 | Bedrooms |
| `RES_FLOOR` | +0.253 | Stories |
| `OVERALL_COND` | +0.236 | Condition grade |
| `HLF_BTH` | +0.226 | Half baths |
| `LAND_SF` | +0.101 | Lot size — weaker than interior space |
| `ZIP_CODE` | −0.051 | Weak as a raw integer — replaced by 32 one-hot dummies in feature engineering |

### Engineered features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE` | `2025 − YR_BUILT` | Older buildings tend to be worth less. `YR_BUILT` is dropped after this to eliminate perfect collinearity. |
| `IS_REMODELED` | `1 if YR_REMODEL > YR_BUILT` | Binary flag for renovation — remodeled properties command a premium |
| `BATH_TOTAL` | `FULL_BTH + 0.5 × HLF_BTH` | Weighted bath count; half baths add value but less than full baths |
| `AREA_PER_ROOM` | `LIVING_AREA / TT_RMS` | Average room size — larger rooms signal higher quality |
| `BED_BATH` | `BED_RMS × BATH_TOTAL` | Combined bedroom/bath count — luxury homes score high on both |
| `ZIP_XXXXX` | One-hot encoding of `ZIP_CODE` | 32 binary columns, one per ZIP code. Treats each neighborhood as a distinct category rather than an arbitrary number, giving linear models explicit neighborhood coefficients. `ZIP_CODE` is dropped after encoding. |

---

## Modeling Methods

Implemented in `src/train_model.py`. Data split 80/20 (`random_state=42`, 106,088 training / 26,523 test). All reported metrics use **5-fold cross-validation on the training set** (mean ± std), which gives more reliable estimates than a single split. Final models are then fit on the full training set for diagnostic plots. Linear models and KNN are wrapped in a `StandardScaler` pipeline; Decision Tree and Random Forest operate on raw values (tree-based models are scale-invariant).

### Linear Regression (baseline)
Fits a global hyperplane through all 55 features. No regularization.

### Ridge Regression
Adds L2 penalty to shrink correlated coefficients. Several features are correlated (e.g., `LIVING_AREA` and `GROSS_AREA`, `BED_RMS` and `TT_RMS`).

### Lasso Regression
Adds L1 penalty that forces some coefficients to zero — implicit feature selection. `alpha=1000`, `max_iter=10000`.

### Decision Tree
Splits on individual features recursively. `max_depth=5`, `min_samples_leaf=5`. Captures non-linear relationships that linear models miss.

### Random Forest
Ensemble of 200 decision trees, each trained on a bootstrap sample with a random subset of features (`max_features="sqrt"`). Averaging across trees reduces variance without increasing bias. Trained with `n_jobs=-1` (all CPU cores). No scaling required.

### KNN (k tuned by CV)
The optimal number of neighbors `k` is selected automatically by 5-fold cross-validation over `k ∈ {3, 5, 7, 10, 15, 20, 30, 50}` before training. The best k is chosen by lowest CV RMSE. Predicts by averaging the k most similar properties by Euclidean distance.

### K-Means clustering (exploration only)
Groups Boston properties into segments based on `LIVING_AREA` and `BED_RMS`. Not used for prediction — reveals natural market tiers. Features are standardized with `StandardScaler` before clustering so neither variable dominates by scale. A 10,000-property random sample (capped at 5,000 sq ft to prevent multi-family outliers from collapsing all clusters to the left edge) is used for speed.

**Choosing k — Elbow Method & Silhouette Score**

Both diagnostics are run over k ∈ {2, 3, …, 10}:

| Diagnostic | What it measures | How to read it |
|------------|-----------------|----------------|
| **Elbow Method** (inertia) | Total squared distance from each point to its cluster centroid | Plot inertia vs k; pick the k where the curve bends sharply ("the elbow") — adding more clusters past that point gives diminishing returns |
| **Silhouette Score** | How similar each point is to its own cluster vs the nearest other cluster. Range: [−1, 1] | Higher = tighter, better-separated clusters. Pick the k with the highest score |

The final k is chosen by **highest silhouette score**. Both diagnostic curves are saved to `plots/kmeans_elbow_silhouette.png`. The cluster scatter plot (Living Area vs Assessed Value, colored by cluster) is saved to `plots/kmeans_clusters.png`.

---

## Results

All metrics from **5-fold cross-validation on the training set** (mean ± std). Best per metric in bold.

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | $189,898 ± $2,079 | $297,010 ± $17,067 | 0.696 ± 0.039 |
| Ridge Regression | $189,867 ± $2,077 | $297,012 ± $17,072 | 0.696 ± 0.039 |
| Lasso Regression | $188,347 ± $1,950 | $296,636 ± $15,227 | 0.697 ± 0.035 |
| Decision Tree | $233,759 ± $872 | $359,124 ± $1,549 | 0.557 ± 0.011 |
| KNN (k=5, CV-tuned) | $108,283 ± $1,276 | $187,460 ± $2,740 | 0.879 ± 0.003 |
| **Random Forest** | **$102,208 ± $1,194** | **$176,096 ± $1,927** | **0.894 ± 0.003** |

*All error values in dollars.*

### Best model: Random Forest

**Random Forest** achieved the best result on all three metrics — lowest RMSE ($176k), lowest MAE ($102k), and highest R² (0.894). On average, predictions are within **$102,000** of the true assessed value. The tiny standard deviation (±0.003 R²) across folds confirms it is consistent and not sensitive to which subset of data it trains on.

### Why Random Forest wins

**Linear Regression (R²=0.696)** fits a single global plane: if Living Area increases by 100 sq ft, the model adds the same dollar amount everywhere, regardless of neighborhood or property type. Adding ZIP code one-hot encoding improved this significantly from R²=0.509, but a flat plane still cannot capture non-linear interactions.

**Decision Tree (R²=0.557)** captures non-linearity but is limited to depth=5 and a single tree — high variance on held-out folds.

**KNN (R²=0.879, k=5 auto-tuned)** outperforms linear models by finding the 5 most similar properties and averaging their values, naturally capturing local market variation. Optimal k was selected by cross-validation from {3, 5, 7, 10, 15, 20, 30, 50}.

**Random Forest (R²=0.894)** builds 200 decision trees, each on a different bootstrap sample and random feature subset. Averaging across trees eliminates individual tree variance while retaining the ability to model non-linear interactions and neighborhood-level effects. It implicitly learns that LIVING_AREA matters differently in Back Bay vs. Hyde Park without explicit interaction features.

### What this tells us about the data

Boston property values are **highly local and non-linear**. The same square footage is worth very different amounts depending on neighborhood and property type. Encoding ZIP codes as 32 one-hot features (rather than a single integer) gave linear models neighborhood-level coefficients and improved their R² from 0.51 → 0.70. Random Forest improved further to 0.894 by learning complex interactions between location, size, and property type.

---

## Visualizations

All plots saved to `plots/` after running `make visualize` and `make train`.

| File | Step | Description |
|------|------|-------------|
| `target_distribution.png` | EDA | Histogram of `TOTAL_VALUE` with mean and median lines |
| `scatter_area_vs_value.png` | EDA | Living area vs assessed value scatter (r = +0.44) |
| `scatter_beds_vs_value.png` | EDA | Bedrooms vs assessed value scatter |
| `boxplot_lu_vs_value.png` | EDA | Property type (R1/R2/R3/R4/CD) vs assessed value |
| `correlation_heatmap.png` | EDA | Full feature correlation matrix |
| `model_comparison.png` | Modeling | Bar chart (with ± std error bars) comparing MAE, RMSE, and R² across all 6 models |
| `actual_vs_predicted.png` | Modeling | Actual vs predicted assessed values (best model: Random Forest) |
| `residual_plot.png` | Modeling | Prediction error vs predicted value (best model: Random Forest) |
| `coefficient_plot.png` | Modeling | Linear Regression coefficients by feature |
| `decision_tree.png` | Modeling | Decision tree structure (top 2 levels) |
| `knn_tuning.png` | Modeling | KNN cross-validation: k vs CV RMSE curve with best k marked |
| `rf_feature_importance.png` | Modeling | Random Forest top 20 feature importances (mean decrease in impurity) |
| `kmeans_elbow_silhouette.png` | Modeling | Elbow Method (inertia) and Silhouette Score curves for k = 2–10, used to select optimal k |
| `kmeans_clusters.png` | Modeling | K-Means Boston property market segments (k chosen by highest silhouette score) |
| `zillow_boston_trend.png` | Context | Boston median sale price trend over time (Zillow 2018–2026) |

---

## Limitations

- **Assessed ≠ market value:** The City of Boston assesses property values for tax purposes. Assessed values typically lag behind actual sale prices and may not capture market fluctuations.
- **No sale price data:** The dataset contains assessed values, not actual transaction prices. A property could sell for significantly more or less than its assessed value.
- **Top 1% removed:** 1,340 ultra-luxury and large multi-unit properties were removed. All models will underpredict at the very high end.
- **Location encoded at ZIP code granularity:** ZIP codes are used as the neighborhood proxy (32 one-hot columns). Finer-grained location data such as latitude/longitude or official Boston neighborhood boundaries could improve predictions further.
- **KNN and dimensionality:** KNN uses Euclidean distance across 55 features including 32 ZIP dummies. Some features may add noise rather than signal, degrading neighbor quality. PCA or feature selection could improve results.

---

## Environment

- **Language:** Python 3.9+
- **Platform:** macOS, Linux, Windows (WSL recommended)

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations and feature transforms |
| `scikit-learn` | Models, pipelines, scaling, metrics |
| `matplotlib` | All visualizations |
| `statsmodels` | (retained as dependency) |
| `requests` | HTTP download with redirect handling |
| `pytest` | Unit test framework |

```bash
pip install -r requirements.txt
```

## Conclusion 

My project shows that data scxiuence can predict Boston assessed property values with good accuracy. The biggest improvements came from cleaning the data carefully, removing leakage columns, adding useful engineered features, one-hot encoding ZIP codes, using cross-validation, and testing stronger models like Random Forest.

My final best model was Random Forest, with an R squared of 0.894.