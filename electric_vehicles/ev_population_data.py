import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

# If missing data points are less than 10, drop the rows
for column in df.columns:
    if df[column].isnull().sum() < 10:
        df = df.dropna(subset=[column])

# For columns with more than 10 missing values, impute the data
# For numeric columns, use median imputation
numeric_columns = df.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())

# For categorical columns, use mode imputation
categorical_columns = df.select_dtypes(exclude=[np.number]).columns
for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

range_stats = df.groupby(['Make', 'Model'])['Electric Range'].agg(['mean', 'median', 'std']).reset_index()
print(f"\n{range_stats}\n")

import re

def extract_coordinates(location):
    match = re.search(r'POINT \(([-\d.]+) ([-\d.]+)\)', location)
    if match:
        return float(match.group(2)), float(match.group(1))  # lat, lon
    return None, None

# For simplicity and demonstration purposes, we test only the first 500 rows
df_first_500 = df.head(500).copy()
df_first_500['Latitude'], df_first_500['Longitude'] = zip(*df_first_500['Vehicle Location'].map(extract_coordinates))

plt.figure(figsize=(12, 8))
sns.kdeplot(data=df_first_500, x='Longitude', y='Latitude', cmap='YlOrRd', fill=True)
plt.title('Density of Electric Vehicles')
plt.show()
print("\n")

# Count vehicles per utility
utility_counts = df['Electric Utility'].value_counts()

# Create a scatter plot
plt.figure(figsize=(24, 20))
plt.scatter(utility_counts.values, utility_counts.index)
plt.xscale('log')
plt.title('Number of Vehicles vs Electric Utility')
plt.xlabel('Number of Vehicles (log scale)')
plt.ylabel('Electric Utility')
plt.show()

cafv_eligible = df[df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] == 'Clean Alternative Fuel Vehicle Eligible']['Electric Range']
cafv_not_eligible = df[df['Clean Alternative Fuel Vehicle (CAFV) Eligibility'] != 'Clean Alternative Fuel Vehicle Eligible']['Electric Range']

t_stat, p_value = stats.ttest_ind(cafv_eligible, cafv_not_eligible)

print(f"\nT-statistic: {t_stat}")
print(f"\nP-value: {p_value}")

if p_value < 0.05:
    print("\nThere is a significant difference in electric range between CAFV eligible and non-eligible vehicles.\n")
else:
    print("\nThere is no significant difference in electric range between CAFV eligible and non-eligible vehicles.\n")

district_counts = df['Legislative District'].value_counts()

plt.figure(figsize=(12, 8))
district_counts.plot(kind='bar')
plt.title('Distribution of Electric Vehicles Across Legislative Districts')
plt.xlabel('Legislative District')
plt.ylabel('Number of Vehicles')
plt.show()
print("\n")

# Remove rows with Base MSRP of 0 (likely missing data)
df_msrp = df[df['Base MSRP'] > 0]

# Regression analysis
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X = df_msrp[['Electric Range', 'Model Year', 'Make']]
y = df_msrp['Base MSRP']

# One-hot encode the 'Make' column
ct = ColumnTransformer([('encoder', OneHotEncoder(drop='first'), ['Make'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

model = LinearRegression()
model.fit(X_encoded, y)

print(f"R-squared score: {model.score(X_encoded, y)}\n")
print(f"Coefficients: {model.coef_}\n")
print(f"Intercept: {model.intercept_}")
