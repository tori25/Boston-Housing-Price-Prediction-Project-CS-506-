# Boston Housing Price Prediction

> **Presentation video:** *(YouTube link to be added before 5/1 deadline)*

---

## How to Build and Run

**Prerequisites:** Python 3.9+

```bash
make install      # install all dependencies
make collect      # download Boston Housing dataset  →  data/raw/boston.csv
make clean        # clean raw data                   →  data/processed/boston_clean.csv
make features     # engineer features                →  data/processed/train_features.csv
make visualize    # generate EDA plots               →  plots/
make train        # train all models                 →  plots/ + data/processed/model_results.txt
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

A GitHub Actions workflow (`.github/workflows/test.yml`) runs all 19 tests automatically on every push and pull request to `main`.

### Output locations

| Output | Path |
|--------|------|
| Raw dataset | `data/raw/boston.csv` |
| Cleaned dataset | `data/processed/boston_clean.csv` |
| Engineered features | `data/processed/train_features.csv` |
| Model comparison results | `data/processed/model_results.txt` |
| Plots | `plots/` |

### Project structure

```
├── data/
│   ├── raw/
│   │   ├── boston.csv              # Boston Housing Dataset (downloaded by collect_data.py)
│   │   └── zillow.csv              # Zillow Boston median sale price data (pre-downloaded)
│   └── processed/
│       ├── boston_clean.csv        # Cleaned dataset
│       ├── train_features.csv      # Dataset with engineered features
│       └── model_results.txt       # Model comparison output
├── src/
│   ├── collect_data.py             # Downloads Boston Housing Dataset
│   ├── clean_data.py               # Data cleaning pipeline
│   ├── features.py                 # Feature engineering
│   ├── visualize.py                # EDA visualizations
│   ├── train_model.py              # Model training and evaluation
│   ├── zillow_analysis.py          # Zillow trend analysis
│   └── tests/
│       └── test_project.py         # Unit tests (pytest)
├── plots/                          # Generated visualizations
├── main.py                         # Full pipeline entry point
├── Makefile
└── requirements.txt
```

---

## Project Goal

Predict **`medv`** — the median value of owner-occupied homes in $1,000s — for each of 506 Boston neighborhoods using the Boston Housing Dataset (Harrison & Rubinfeld, 1978).

The dataset contains 13 features including crime rate, number of rooms, property tax rate, and access to employment centers. The project applies the full data science lifecycle: collection → cleaning → feature engineering → visualization → modeling → evaluation.

Performance is measured with **MAE**, **RMSE**, and **R²**.

As supplemental real-world context, Zillow median sale price data for Boston, MA is also collected and visualized to show how the market has evolved since the original 1978 dataset.

---

## Data Collection

Implemented in `src/collect_data.py`.

**Primary dataset:** Boston Housing Dataset loaded via `statsmodels.datasets.get_rdataset("Boston", "MASS")`. Saved to `data/raw/boston.csv`.

| Feature | Description |
|---------|-------------|
| `crim` | Per capita crime rate by town |
| `zn` | Proportion of residential land zoned for large lots |
| `indus` | Proportion of non-retail business acres |
| `chas` | Charles River dummy (1 if tract bounds river) |
| `nox` | Nitric oxide concentration |
| `rm` | Average number of rooms per dwelling |
| `age` | Proportion of units built before 1940 |
| `dis` | Weighted distance to five employment centers |
| `rad` | Accessibility index to radial highways |
| `tax` | Property tax rate per $10,000 |
| `ptratio` | Pupil-teacher ratio by town |
| `b` | Racial composition index |
| `lstat` | % lower-status population |
| `medv` | **Target** — Median home value in $1,000s |

**Secondary dataset:** Zillow Research Data — Median Sale Price by Metro Area. Used for a trend visualization showing the Boston real estate market from 2010 to present.

The `collect_data.py` script also produces `data/processed/data_inspection.txt` with a full summary of the raw data: shape, dtypes, missing values, and descriptive statistics.

---

## Data Cleaning

Implemented in `src/clean_data.py`. Final cleaned dataset: **485 rows × 14 columns**, zero missing values.

### Step 1 — Drop duplicate rows
**What:** Remove any rows that appear more than once.
**Why:** Duplicates cause the model to see the same observation multiple times during training, artificially inflating confidence in those data points.

### Step 2 — Remove irrelevant ID columns
**What:** Drop any column named `id` or `unnamed: 0` (index columns accidentally saved to CSV).
**Why:** ID columns are arbitrary identifiers with no relationship to home value.

### Step 3 — Fix data types
**What:** Cast `chas`, `rad`, and `tax` to `int`.
**Why:** These columns were loaded as `float64` by pandas but are conceptually integers — `chas` is a 0/1 dummy, `rad` is an ordinal index 1–24, `tax` is a whole-number rate. Correct types prevent floating-point noise in distance calculations (KNN).

### Step 4 — Handle missing values
**What:** Detect columns with missing values and fill with the column median.
**Why:** The classic Boston dataset has no missing values, but the check is explicit so the script is safe if the source data changes. Median imputation preserves the distribution better than mean imputation when outliers are present.

### Step 5 — Remove censored values
**What:** Remove all rows where `medv == 50`.
**Why:** The original dataset artificially caps home values at $50,000. These 16 rows are not real observations — they represent homes worth *at least* $50k but recorded as exactly $50k. Keeping them teaches the model a false ceiling, causing it to systematically underpredict expensive neighborhoods.

### Step 6 — Remove rows with negative feature values
**What:** Drop any row where a non-target numeric feature is negative.
**Why:** All Boston Housing features (crime rate, rooms, distance, etc.) are physically non-negative. A negative value indicates a data entry error.

### Step 7 — Remove extreme outliers (crime rate)
**What:** Remove rows in the top 1% of `crim`, dropping 5 rows.
**Why:** A small number of Boston tracts have crime rates orders of magnitude above the rest. These extreme values distort Euclidean distances in KNN and skew linear model coefficients. Log-transforming `crim` in feature engineering further tames this distribution.

---

## Feature Extraction

Implemented in `src/features.py`. Output: `data/processed/train_features.csv` (**485 rows × 21 columns** — 14 original + 7 engineered).

Before engineering, all raw features were ranked by correlation with `medv` to identify the strongest candidates.

### Feature correlations with medv

| Feature | Correlation | Notes |
|---------|------------|-------|
| `lstat` | −0.758 | Strongest predictor — poverty rate is a direct proxy for neighborhood desirability |
| `rm` | +0.690 | More rooms = more space = higher value |
| `indus` | −0.595 | Industrial land use reduces residential desirability |
| `tax` | −0.562 | Higher taxes increase ownership cost |
| `nox` | −0.517 | Air pollution proxy |
| `ptratio` | −0.514 | School quality proxy |
| `crim` | −0.506 | Crime directly reduces safety and desirability |
| `age` | −0.485 | Older housing stock correlates with lower value |
| `rad` | −0.462 | Higher highway exposure → more noise → lower value |
| `zn` | +0.403 | More residential zoning → suburban, higher-value areas |
| `dis` | +0.358 | Farther from industry → quieter, cleaner neighborhoods |
| `chas` | +0.072 | Kept for EDA only — only 35 river-adjacent tracts |
| `b` | — | **Excluded** — racially charged feature from 1978 |

### Engineered features

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `CRIME_LOG` | `log(1 + crim)` | Crime is heavily right-skewed — log compresses the tail so a jump from 0.1 to 1.0 is treated the same as 1.0 to 10.0 |
| `ROOM_SQ` | `rm²` | The value premium for extra rooms is non-linear — going from 6 to 7 rooms adds more value than 4 to 5 |
| `TAX_PER_ROOM` | `tax / rm` | Ownership cost per unit of space — a high tax rate on a small home is a worse deal than the same rate on a large home |
| `LSTAT_PER_ROOM` | `lstat / rm` | Poverty density relative to home size — combines the two strongest predictors |
| `POLLUTION_PROXIMITY` | `nox / dis` | Pollution concentration per unit of distance from employment centers |
| `SCHOOL_INDEX` | `ptratio × lstat` | School quality weighted by neighborhood poverty — bad schools in poor areas compound each other |
| `AGE_DIST` | `age × dis` | Old housing stock that is also far from employment — the least desirable combination |

All original columns are preserved so models have access to both raw and engineered features.

---

## Modeling Methods

Implemented in `src/train_model.py`. All models are trained on an **80/20 train/test split** (`random_state=42`). Linear models and KNN are wrapped in a `StandardScaler` pipeline; the Decision Tree operates on raw feature values.

### Linear Regression (baseline)
Fits a global hyperplane through all 21 features. No regularization. This is the simplest possible model and sets the performance floor.

### Ridge Regression
Adds an L2 penalty to shrink correlated coefficients. Several Boston features are correlated (e.g. `tax` and `rad`, `nox` and `indus`), so Ridge is a natural second step.

### Lasso Regression
Adds an L1 penalty that forces some coefficients to exactly zero — implicit feature selection across 21 features. `alpha=0.1`, `max_iter=5000`.

### Decision Tree
Splits on individual features recursively. Constrained to `max_depth=5`, `min_samples_leaf=5` to limit overfitting. Can capture non-linear relationships but suffers from high variance on small datasets.

### KNN (k=10)
Predicts by averaging the 10 most similar neighborhoods by Euclidean distance. No parametric assumptions — purely local similarity. `k=10` reduces the variance of single-neighbor predictions.

### K-Means clustering (exploration only)
Groups Boston neighborhoods into 4 segments based on `rm` and `lstat`. Not used for prediction — reveals natural market tiers (wealthy suburbs vs inner-city tracts) in an unsupervised way.

---

## Results

| Model | MAE | RMSE | R² | vs. Baseline (RMSE) |
|-------|-----|------|----|---------------------|
| **Linear Regression** | 2.30 | **2.95** | **0.844** | — baseline |
| Ridge Regression | 2.37 | 3.00 | 0.839 | +0.05 |
| Lasso Regression | 2.43 | 3.06 | 0.832 | +0.11 |
| Decision Tree | 2.67 | 3.80 | 0.740 | +0.85 |
| KNN (k=10) | **2.24** | 3.00 | 0.838 | +0.05 |

*All error values in $1,000s. Best per metric in bold.*

### Best model: Linear Regression

**Linear Regression** achieved the lowest RMSE (2.95) and highest R² (0.844). Typical predictions are within **$2,300** of the true median neighborhood value.

**KNN had the lowest MAE (2.24)** — individual predictions are slightly closer on average — but it does not generalize as well overall (RMSE 3.00).

### Why no model beat the linear baseline on RMSE

**Ridge and Lasso** scored slightly below Linear Regression. Regularization penalizes large coefficients to reduce overfitting, but with only 21 features and ~388 training rows there is not much overfitting to correct. The penalty ends up shrinking genuinely useful coefficients, slightly hurting performance.

**Decision Tree** was the clear worst (RMSE 3.80, R² 0.740). A single tree makes locally optimal splits but misses the smooth global linear trend between `rm`, `lstat`, and `medv`. Deeper trees would overfit on only 388 training rows.

### What this tells us about the data

The fact that the simplest model wins is the key finding: **Boston neighborhood prices have a largely linear relationship with the features.** Good feature engineering (`CRIME_LOG`, `ROOM_SQ`, `SCHOOL_INDEX`, etc.) allowed the linear model to capture non-linear patterns without requiring a non-linear model.

---

## Visualizations

All plots are saved to `plots/` after running `make visualize` and `make train`.

| File | Step | Description |
|------|------|-------------|
| `target_distribution.png` | EDA | Histogram of `medv` with mean and median lines |
| `scatter_rm_vs_medv.png` | EDA | Rooms vs home value scatter with trend line (r = +0.69) |
| `scatter_lstat_vs_medv.png` | EDA | Poverty rate vs home value scatter (r = −0.74) |
| `boxplot_chas_vs_medv.png` | EDA | River-adjacent vs non-river-adjacent home values |
| `correlation_heatmap.png` | EDA | Full feature correlation matrix |
| `model_comparison.png` | Modeling | Bar chart comparing MAE, RMSE, and R² across all 5 models |
| `actual_vs_predicted.png` | Modeling | Actual vs predicted home values (best model) |
| `residual_plot.png` | Modeling | Prediction error vs predicted value (best model) |
| `coefficient_plot.png` | Modeling | Linear Regression coefficients by feature (magnitude + direction) |
| `decision_tree.png` | Modeling | Decision tree structure (top 2 levels) |
| `kmeans_clusters.png` | Modeling | K-Means (k=4) Boston neighborhood segments |
| `zillow_boston_trend.png` | Context | Boston median sale price trend over time (Zillow) |

### Key insights

- **Actual vs. Predicted:** Points cluster tightly along the 45° line. Slight underprediction at the high end — expected, since `medv==50` censoring means few high-value examples survived cleaning.
- **Residuals:** Centered on zero with no clear pattern — no systematic bias.
- **Coefficient plot:** `LSTAT_PER_ROOM`, `ROOM_SQ`, and `SCHOOL_INDEX` have the largest magnitudes, confirming that rooms and poverty rate are the dominant price drivers — and that the engineered features captured the non-linear signal.
- **Decision Tree top splits:** First split on `rm`, second split on `lstat`, confirming these two features dominate.
- **K-Means clusters:** Four clear market tiers — high-rooms/low-poverty (wealthy suburbs) vs low-rooms/high-poverty (inner-city tracts), with two mid-range segments.
- **Zillow trend:** Boston median prices have risen from ~$380k (2010) to ~$750k (2022+), contextualizing how much the 1978 dataset understates current values.

---

## Limitations

- **1978 dollars:** All prices are in 1978 $1,000s and are not comparable to modern values. The Zillow trend plot provides modern context.
- **`medv` ceiling:** The dataset caps home values at $50,000. After removing these 16 censored rows, expensive neighborhoods are underrepresented — all models will underpredict high-value areas.
- **`b` column excluded:** The original dataset includes a racially charged feature (`b`) derived from the 1978 paper. It is excluded from all modeling and is not an appropriate variable for any modern application.
- **Small dataset:** 485 rows limits how well complex models can generalize. Random Forest, Gradient Boosting, or neural networks would need significantly more data to outperform the linear baseline here.
- **KNN and dimensionality:** KNN degrades as the number of features grows (curse of dimensionality). With 21 features it performs well, but adding more features would require PCA or feature selection first.
- **Geographic granularity:** The dataset describes census tracts — individual properties within a tract can vary significantly from the tract median.

---

## How to Contribute

1. Fork the repository and create a feature branch.
2. Install dependencies: `make install`
3. Make your changes and ensure all tests pass: `make test`
4. Submit a pull request with a clear description of the change.

The test suite covers the two most critical pipeline steps — data cleaning and feature engineering. Any new transformation should have a corresponding test in `src/tests/test_project.py`.

---

## Environment

- **Language:** Python 3.9+
- **Platform:** macOS, Linux, Windows (WSL recommended)
- **Key libraries:** pandas, numpy, scikit-learn, matplotlib, statsmodels, pytest

Install all dependencies:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations and feature transforms |
| `scikit-learn` | Models, pipelines, scaling, metrics |
| `matplotlib` | All visualizations |
| `statsmodels` | Boston Housing Dataset download |
| `pytest` | Unit test framework |
