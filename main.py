import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_filename = 'train.csv'
df = pd.read_csv(data_filename)

print(df.head())
print(df.info())

# Print one value
print(df.iloc[7,2])

# Select predictor and response
x_true = df["LotArea"].iloc[5:13]
y_true = df["SalePrice"].iloc[5:13]

# Scatter plot 1
plt.scatter(df["GrLivArea"], df["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()

# Scatter plot 2
plt.scatter(df["LotArea"], df["SalePrice"])
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.title("LotArea vs SalePrice")
plt.show()

# # Correlation heatmap
# numeric_df = df.select_dtypes(include=[np.number])

# plt.figure(figsize=(10,8))
# sns.heatmap(numeric_df.corr(), cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# Boxplot
sns.boxplot(x="OverallQual", y="SalePrice", data=df)
plt.title("SalePrice vs OverallQual")
plt.show()