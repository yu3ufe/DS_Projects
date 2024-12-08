import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the data
df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

# Convert DATE OCC to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %H:%M:%S %p')

# Drop null values in DATE OCC
df_clean = df.dropna(subset=['DATE OCC'])

# Set the 'DATE OCC' column as the index
df_clean = df_clean.set_index('DATE OCC')

# Resample the data by month and count the number of occurrences
monthly_crime_data = df_clean.resample('ME').size()

# Plot the time series data
plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_crime_data)
plt.title('Monthly Crime Occurrences from 2020 to Present')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
print("\n")
plt.show()
print("\n")

# Perform seasonal decomposition of time series data
result = seasonal_decompose(monthly_crime_data, model='additive')
result.plot()
plt.show()
print("\n")

# Detect anomalies using rolling mean and standard deviation
rolling_mean = monthly_crime_data.rolling(window=12).mean()
rolling_std = monthly_crime_data.rolling(window=12).std()

plt.figure(figsize=(14, 7))
sns.lineplot(data=monthly_crime_data, label='Monthly Crime Data')
sns.lineplot(data=rolling_mean, label='Rolling Mean (12 months)')
sns.lineplot(data=rolling_std, label='Rolling Std (12 months)')
plt.title('Crime Data with Rolling Mean and Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()
print("\n")

# Identify significant changes in crime rates during and after the COVID-19 pandemic
pre_covid_period = monthly_crime_data['2020-01':'2020-02']
covid_period = monthly_crime_data['2020-03':'2023-05']
post_covid_period = monthly_crime_data['2023-06':]

mean_pre_covid = pre_covid_period.mean()
mean_covid = covid_period.mean()
mean_post_covid = post_covid_period.mean()

print(f"Mean crime rate before COVID-19: {mean_pre_covid}")
print(f"Mean crime rate during COVID-19: {mean_covid}")
print(f"Mean crime rate after COVID-19: {mean_post_covid}\n")

# Drop null values in LAT and LON
heatmap_cleaned = df.dropna(subset=['LAT', 'LON'])

# Extract the relevant columns for clustering
crime_data = heatmap_cleaned[['LAT', 'LON']]

# Standardize the data
scaler = StandardScaler()
crime_data_scaled = scaler.fit_transform(crime_data)

# Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(crime_data_scaled)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
dbscan_labels = dbscan.fit_predict(crime_data_scaled)

# Add the cluster labels to the original dataframe
heatmap_cleaned['KMeans_Cluster'] = kmeans_labels
heatmap_cleaned['DBSCAN_Cluster'] = dbscan_labels

# Plot the K-means clustering results
plt.figure(figsize=(14, 7))
sns.scatterplot(x='LON', y='LAT', hue='KMeans_Cluster', palette='viridis', data=heatmap_cleaned, legend='full')
plt.title('Crime Hotspots Identified by K-means Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()
print("\n")

# Plot the DBSCAN clustering results
plt.figure(figsize=(14, 7))
sns.scatterplot(x='LON', y='LAT', hue='DBSCAN_Cluster', palette='viridis', data=heatmap_cleaned, legend='full')
plt.title('Crime Hotspots Identified by DBSCAN Clustering')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()
print("\n")

# Drop null values in relevant columns
chi_Square = df.dropna(subset=['Vict Age', 'Vict Sex', 'Vict Descent', 'Crm Cd Desc'])
chi_Square = chi_Square.copy()

# Categorize age
chi_Square['Age_Group'] = pd.cut(chi_Square['Vict Age'], bins=[0, 18, 30, 50, 70, 100], labels=['0-18', '19-30', '31-50', '51-70', '71+'])

# Create a contingency table for each demographic characteristic and crime type
age_crime_table = pd.crosstab(chi_Square['Vict Age'], chi_Square['Crm Cd Desc'])
sex_crime_table = pd.crosstab(chi_Square['Vict Sex'], chi_Square['Crm Cd Desc'])
descent_crime_table = pd.crosstab(chi_Square['Vict Descent'], chi_Square['Crm Cd Desc'])

# Perform chi-square test for each contingency table
age_chi2, age_p, age_dof, age_expected = chi2_contingency(age_crime_table)
sex_chi2, sex_p, sex_dof, sex_expected = chi2_contingency(sex_crime_table)
descent_chi2, descent_p, descent_dof, descent_expected = chi2_contingency(descent_crime_table)

# Print the results
print(f"Chi-square test for Victim Age and Crime Type: chi2= {age_chi2}, p-value= {age_p}")
print(f"Chi-square test for Victim Sex and Crime Type: chi2= {sex_chi2}, p-value= {sex_p}")
print(f"Chi-square test for Victim Descent and Crime Type: chi2= {descent_chi2}, p-value= {descent_p}\n")

# Visualize the relationship between age groups and crime types
plt.figure(figsize=(15, 6))
sns.countplot(data=chi_Square, x='Age_Group', hue='Crm Cd Desc')
plt.title('Crime Types by Age Group')
plt.xticks()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
print("\n")

# Visualize the relationship between sex and crime types
plt.figure(figsize=(15, 6))
sns.countplot(data=chi_Square, x='Vict Sex', hue='Crm Cd Desc')
plt.title('Crime Types by Victim Sex')
plt.xticks()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
print("\n")

# Visualize the relationship between descent and crime types
plt.figure(figsize=(15, 6))
sns.countplot(data=chi_Square, x='Vict Descent', hue='Crm Cd Desc')
plt.title('Crime Types by Victim Descent')
plt.xticks()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
print("\n")

# Extract the relevant columns for correlation analysis
crime_codes = df[['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']]

# Drop NaN values
crime_codes = crime_codes.dropna()

# Calculate the correlation matrix
correlation_matrix = crime_codes.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title('Correlation Heatmap of Crime Types')
plt.show()
print("\n")

# Drop rows with null values in the columns that are going to be used
weapons = df.dropna(subset=['Crm Cd Desc', 'LOCATION', 'TIME OCC', 'Weapon Used Cd'])

# Count the most common weapons
weapon_counts = df['Weapon Desc'].value_counts()
print("Most common weapons used in crimes:")
print(weapon_counts.head(10))
print("\n")

# Extract relevant features for the logistic regression model
features = weapons[['Crm Cd Desc', 'LOCATION', 'TIME OCC']]
target = weapons['Weapon Used Cd']

# One-hot encode categorical features
features_encoded = pd.get_dummies(features, columns=['Crm Cd Desc', 'LOCATION'])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f"Accuracy of the logistic regression model: {accuracy}\n")
print("Classification Report:\n")
print(report)

# Print feature importances
feature_importance = pd.DataFrame({'Feature': features_encoded.columns, 'Importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importance)
