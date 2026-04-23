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

The dataset contains **183,445 property records** (as of December 2024) covering all taxable and non-taxable parcels in the City of Boston. After filtering to residential properties and cleaning, the modeling dataset contains **132,611 rows × 24 features**. Unlike the classic 1978 Boston Housing Dataset which described census-tract averages, this dataset contains actual individual property records — each row is one house, condo, or apartment building.

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
| `ZIP_CODE` | Zip code of parcel |
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

Implemented in `src/features.py`. Output: `data/processed/train_features.csv` (**132,611 rows × 25 columns** — 18 original + 7 engineered).

### Feature correlations with TOTAL_VALUE

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
| `ZIP_CODE` | −0.051 | Slight negative (lower zip codes in higher-value neighborhoods) |

### Engineered features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `AGE` | `2025 − YR_BUILT` | Older buildings tend to be worth less; computable and interpretable |
| `IS_REMODELED` | `1 if YR_REMODEL > YR_BUILT` | Binary flag for renovation — remodeled properties command a premium |
| `BATH_TOTAL` | `FULL_BTH + 0.5 × HLF_BTH` | Weighted bath count; half baths add value but less than full baths |
| `LOG_LIVING_AREA` | `log(1 + LIVING_AREA)` | Living area is right-skewed — log compresses the tail |
| `LOG_LAND_SF` | `log(1 + LAND_SF)` | Lot size is right-skewed |
| `AREA_PER_ROOM` | `LIVING_AREA / TT_RMS` | Average room size — larger rooms signal higher quality |
| `BED_BATH` | `BED_RMS × BATH_TOTAL` | Combined bedroom/bath count — luxury homes score high on both |

---

## Modeling Methods

Implemented in `src/train_model.py`. All models trained on an **80/20 train/test split** (`random_state=42`, 106,088 training / 26,523 test). Linear models and KNN wrapped in a `StandardScaler` pipeline; Decision Tree operates on raw values.

### Linear Regression (baseline)
Fits a global hyperplane through all 24 features. No regularization.

### Ridge Regression
Adds L2 penalty to shrink correlated coefficients. Several features are correlated (e.g., `LIVING_AREA` and `GROSS_AREA`, `BED_RMS` and `TT_RMS`).

### Lasso Regression
Adds L1 penalty that forces some coefficients to zero — implicit feature selection. `alpha=1000`, `max_iter=10000`.

### Decision Tree
Splits on individual features recursively. `max_depth=5`, `min_samples_leaf=5`. Captures non-linear relationships that linear models miss.

### KNN (k=10)
Predicts by averaging the 10 most similar properties by Euclidean distance. No parametric assumptions — purely local similarity. Naturally captures neighborhood effects since nearby properties have similar features.

### K-Means clustering (exploration only)
Groups Boston properties into 4 segments based on `LIVING_AREA` and `BED_RMS`. Not used for prediction — reveals natural market tiers.

---

## Results

| Model | MAE | RMSE | R² | vs. Baseline (RMSE) |
|-------|-----|------|----|---------------------|
| Linear Regression | $253,069 | $388,393 | 0.509 | — baseline |
| Ridge Regression | $253,046 | $388,391 | 0.509 | ≈ same |
| Lasso Regression | $253,068 | $388,388 | 0.509 | ≈ same |
| Decision Tree | $203,991 | $304,670 | 0.698 | −$83,723 |
| **KNN (k=10)** | **$173,639** | **$293,096** | **0.720** | **−$95,297** |

*All error values in dollars. Best per metric in bold.*

### Best model: KNN (k=10)

**KNN (k=10)** achieved the best result on all three metrics — lowest RMSE ($293k), lowest MAE ($174k), and highest R² (0.720). On average, predictions are within **$174,000** of the true assessed value.

### Why KNN beats linear regression on this dataset

**Linear Regression (R²=0.51)** fits a single global plane: if Living Area increases by 100 sq ft, the model always adds the same dollar amount, regardless of the property's location or type. That assumption is wrong for real estate — an extra 100 sq ft in Beacon Hill adds far more value than in East Boston.

**KNN (R²=0.72)** has no such assumption. It finds the 10 most similar properties (same zip code range, similar size, similar bedrooms) and averages their values. This naturally captures local market variation without needing to model it explicitly.

**Decision Tree (R²=0.70)** also outperforms linear models by creating different prediction rules for different property segments — e.g., "if LIVING_AREA > 2,000 and FULL_BTH > 2 then predict $X". But it's slightly noisier than KNN at depth=5.

### What this tells us about the data

Boston property values are **highly local and non-linear** — the same square footage is worth very different amounts depending on neighborhood and property type. Linear models cannot capture this without neighborhood-level dummy variables. KNN captures it implicitly by finding similar properties.

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
| `model_comparison.png` | Modeling | Bar chart comparing MAE, RMSE, and R² across all 5 models |
| `actual_vs_predicted.png` | Modeling | Actual vs predicted assessed values (best model: KNN) |
| `residual_plot.png` | Modeling | Prediction error vs predicted value (best model: KNN) |
| `coefficient_plot.png` | Modeling | Linear Regression coefficients by feature |
| `decision_tree.png` | Modeling | Decision tree structure (top 2 levels) |
| `kmeans_clusters.png` | Modeling | K-Means (k=4) Boston property market segments |
| `zillow_boston_trend.png` | Context | Boston median sale price trend over time (Zillow 2018–2026) |

---

## Limitations

- **Assessed ≠ market value:** The City of Boston assesses property values for tax purposes. Assessed values typically lag behind actual sale prices and may not capture market fluctuations.
- **No sale price data:** The dataset contains assessed values, not actual transaction prices. A property could sell for significantly more or less than its assessed value.
- **Top 1% removed:** 1,340 ultra-luxury and large multi-unit properties were removed. All models will underpredict at the very high end.
- **Location encoded as zip code only:** `ZIP_CODE` has weak correlation with value (r = −0.05) because it's a single integer. With neighborhood dummy variables or latitude/longitude, the models would likely perform much better.
- **KNN and dimensionality:** KNN uses Euclidean distance across 24 features. Some features may add noise rather than signal, degrading neighbor quality. PCA or feature selection could improve results.

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
