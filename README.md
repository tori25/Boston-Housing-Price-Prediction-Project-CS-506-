# Boston Housing Price Prediction

## Project Description

This project applies the full data science lifecycle to predict housing prices in the Boston area. The goal is to use data science and machine learning methods to analyze housing-related features and build accurate models that estimate median property values.

The project will include data collection, data cleaning, feature engineering, visualization, model training, and evaluation. The scope of the project is two months.

The input features I plan to use include variables such as:

- **ZN** – Proportion of residential land zoned for lots over 25,000 sq.ft.  
- **INDUS** – Proportion of non-retail business acres per town  
- **CHAS** – Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
- **NOX** – Nitric oxides concentration (parts per 10 million)  
- **RM** – Average number of rooms per dwelling  
- **AGE** – Proportion of owner-occupied units built prior to 1940  
- **DIS** – Weighted distances to five Boston employment centers  
- **RAD** – Index of accessibility to radial highways  
- **TAX** – Full-value property-tax rate per $10,000  
- **PTRATIO** – Pupil-teacher ratio by town  
- **B** – 1000(Bk − 0.63)² where Bk is the proportion of Black residents by town  
- **LSTAT** – Percentage of lower-status population  
- **MEDV** – Median value of owner-occupied homes (target variable, in $1000s)

---

## Project Timeline (Two-Month Plan)

- **Weeks 1–2:** Data collection and initial data exploration  
- **Weeks 3–4:** Data cleaning and preprocessing. First Project check-in. 
- **Week 5:** Feature engineering  
- **Weeks 6–7:** Model training and evaluation. Second project check-in. 
- **Week 8:** Final analysis, visualization, report preparation, and repository cleanup  

This timeline reflects a structured and realistic plan for completing the project within two months.

---

## Project Goals

The primary goal of this project is:

- Predict median house values using structured housing and neighborhood features such as crime rate, average number of rooms, property tax rate, distance to employment centers, and accessibility to highways.

This goal is specific and measurable. Model performance will be evaluated using regression metrics including:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² (coefficient of determination)  

We will compare a baseline linear regression model with more advanced models and determine which approach achieves the lowest prediction error.

---

## Data Collection Plan

- **Primary Dataset:** House Prices – Advanced Regression Techniques  
- **Source:** Kaggle (public dataset)  
  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data  
- **Data Type:** Tabular CSV files  
- **Collection Method:** Downloading the dataset directly from Kaggle and storing it locally in the project repository  

As an alternative reference dataset, the classic Boston Housing dataset may also be used for comparison:

- https://www.kaggle.com/datasets/vikrishnan/boston-house-prices  

The primary dataset contains structured housing and neighborhood features including:

- Lot area  
- Overall material and finish quality  
- Year built  
- Total living area  
- Garage size  
- Basement area  
- Neighborhood  
- Property tax information  
- Sale price (target variable)  

The dataset will be downloaded once and used consistently throughout the project to ensure reproducibility.

---

## Data Cleaning Plan

Data cleaning will include:

- Checking for missing values  
- Handling outliers  
- Verifying feature ranges and distributions  
- Normalizing or scaling numerical features if needed  
- Splitting data into training and testing subsets  

---

## Feature Extraction

Feature engineering may include:

- Creating interaction features (e.g., rooms × location-related factors)  
- Scaling or transforming skewed variables  
- Exploring polynomial features if appropriate  

Feature importance analysis will be used to understand which variables most strongly influence house prices.

---

## Modeling Approach (Preliminary)

We plan to explore several regression models, including:

- Linear Regression (baseline)  
- Ridge or Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

The final model selection may evolve as we analyze the data and learn additional methods during the course.

---

## Visualization Plan (Preliminary)

Visualizations will include:

- Distribution plots of housing prices  
- Correlation heatmaps  
- Scatter plots of key features vs. price  
- Residual plots  
- Feature importance plots  

These visualizations will support interpretation of both the data and model performance.

---

## Test Plan (Preliminary)

- Split dataset into training and testing sets (e.g., 80/20 split)  
- Evaluate models using MAE, RMSE, and R²  
- Compare baseline and more advanced models  
- Analyze prediction errors and residual patterns  