import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Exploratory Data Analysis (EDA)
df.info()
df.describe()

# Visualize correlations with target
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()


# Check for missing values
df.isnull().sum()

# If missing values exist, impute or drop
df.fillna(df.mean(), inplace=True)


from sklearn.preprocessing import StandardScaler

# Scaling numerical features
scaler = StandardScaler()
numerical_features = df.drop(columns=['target'])
scaled_features = scaler.fit_transform(numerical_features)

# Convert the scaled features back to a DataFrame
df_scaled = pd.DataFrame(scaled_features, columns=numerical_features.columns)
df_scaled['target'] = df['target']


