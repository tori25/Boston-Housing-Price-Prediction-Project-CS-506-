# House Price Prediction Using the Ames Housing Dataset

## Project Description

This project applies the full data science lifecycle to predict housing prices in the Boston area. The goal is to use data science and machine learning methods to analyze housing-related features and build accurate models that estimate median property values.

The project will include data collection, data cleaning, feature engineering, visualization, model training, and evaluation. The scope of the project is two months.

The input features I plan to use include variables such as:

Data fields

MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale

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

- Successfuly predict the sales price for each house. For each Id in the test set,  must predict the value of the SalePrice variable. 


This goal is specific and measurable. Model performance will be evaluated using regression metrics including:

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- R² (coefficient of determination)  

We will compare a baseline linear regression model with more advanced models and determine which approach achieves the lowest prediction error.

---

## Data Collection Plan

- **Primary Dataset:** House Prices – Advanced Regression Techniques, Ames Housing Data Set 
- **Source:** Kaggle (public dataset)  
  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data  
- **Data Type:** Tabular CSV files  
- **Collection Method:** Downloading the dataset directly from Kaggle and storing it locally in the project repository  

The Ames Housing dataset is widely used in data science education because it contains a large number of detailed housing features, making it suitable for exploratory data analysis, feature engineering, and predictive modeling.

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

## Data cleaning will include:

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

## How to run the project 
- make install
- make clean
- make run


## Check-In nr. 1
The main dataset used is the Ames Housing dataset from Kaggle, which contains detailed information about residential homes.

In addition, external Zillow data for Boston, MA was explored to provide real-world market context.


## Project Structure 

data/
  raw/
    train.csv
    zillow.csv
  processed/
    train_clean.csv
    train_features.csv
    model_results.txt

src/
  clean_data.py
  features.py
  train_model.py
  zillow_analysis.py
  main.py

## Main Script
`main.py` is used for exploratory data analysis and visualization, including:
- Scatter plots
- Correlation heatmap
- Boxplots

## Data Processing

Steps Performed
	1.	Loaded raw dataset from data/raw/train.csv
	2.	Removed duplicate rows
	3.	Dropped columns with many missing or low-value features:
	•	PoolQC, Fence, MiscFeature, Alley, Utilities, LandSlope
	4.	Handled missing values by filling with default values (e.g., 0)
	5.	Removed outliers (e.g., very large houses with GrLivArea > 4000)
	6.	Inspected data types (numeric vs categorical)
	7.	Analyzed categorical features (number of unique values)
	8.	Saved cleaned dataset to data/processed/train_clean.csv

Purpose

These steps ensure the dataset is clean, consistent, and ready for modeling.

## Exploratory Data Analysis (EDA)

Several visualizations were created to understand relationships in the data:
	•	Living Area vs Price
Larger houses tend to have higher prices (positive relationship).

	•	House Quality vs Price
Higher-quality houses have significantly higher prices.

	•	Correlation Matrix
Strong predictors identified:

	•	OverallQual
	•	GrLivArea
	•	TotalBsmtSF
	•	GarageArea
	•	YearBuilt / YearRemodAdd

These insights guided feature selection for modeling.

Data Modeling

## Models Used
	•	Linear Regression (baseline model)
Captures overall relationships between features and price.

	•	K-Nearest Neighbors (KNN)
Predicts house prices based on similar houses using distance metrics.

	•	K-Means Clustering
Groups similar houses to identify patterns in the data (used for exploration, not prediction).

## Distance Metrics
	•	Euclidean Distance
	•	Manhattan Distance

These are used in KNN to measure similarity between houses.

## Preprocessing for Modeling
	•	Missing values handled using imputation
	•	Categorical features encoded using OneHotEncoder
	•	Features scaled using StandardScaler (important for KNN and clustering)

  Evaluation Metrics

## Models are evaluated using:
	•	MAE (Mean Absolute Error)
	•	RMSE (Root Mean Squared Error)
	•	R² Score

  ## Preliminary Results
	•	Models are able to capture general trends in the data
	•	Larger and higher-quality houses tend to have higher predicted prices
	•	KNN predicts based on similarity, while Linear Regression captures global patterns
	•	Predictions are not fully accurate yet

## Next Improvements
	•	Feature selection (use most important variables)
	•	Hyperparameter tuning (e.g., number of neighbors in KNN)
	•	Further model optimization

## Zillow Data Integration
In addition to the Ames dataset, Zillow data for Boston, MA was explored.
	•	The data was originally in wide format and reshaped into long format
	•	Used to analyze monthly housing price trends in Boston

In the future: 
	•	Integrate Zillow data into the modeling pipeline
	•	Use it to improve prediction accuracy and real-world relevance


