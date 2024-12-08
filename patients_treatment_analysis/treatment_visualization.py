import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from itertools import combinations
from collections import Counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

# Load the synthetic dataset
df = pd.read_csv('patient_data.csv')

# Step 1: Data Cleaning
# Remove rows with negative values in 'treatment_cost'
df = df[df['treatment_cost'] >= 0]

# Handle missing values by filling with median values
df.fillna(df.median(numeric_only=True), inplace=True)
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['diagnosis'] = df['diagnosis'].fillna(df['diagnosis'].mode()[0])
df['outcome'] = df['outcome'].fillna(df['outcome'].mode()[0])

# Convert date columns to datetime
df['admission_date'] = pd.to_datetime(df['admission_date'])
df['discharge_date'] = pd.to_datetime(df['discharge_date'])

# Appropriately fill the missing dates
df['admission_date'] = df['admission_date'].fillna(df['discharge_date'] - pd.Timedelta(days=10))
df['discharge_date'] = df['discharge_date'].fillna(df['admission_date'] + pd.Timedelta(days=10))

# Delete the completely missing dates data
df.dropna(subset=['admission_date', 'discharge_date'], inplace=True)

# Step 2: Temporal Aggregation Anomaly
def calculate_rolling_cost(group):
    group = group.sort_values('admission_date')
    group['cumulative_cost'] = group['treatment_cost'].cumsum()
    group['rolling_3month_cost'] = group['cumulative_cost'] - group['cumulative_cost'].shift(90, fill_value=0)
    return group

df = df.sort_values('admission_date')
df = df.groupby('hospital_id').apply(calculate_rolling_cost, include_groups=False).reset_index()

rolling_sales = df[['admission_date', 'hospital_id', 'rolling_3month_cost']]

# Step 3: Seasonal Segmentation
df.set_index('admission_date', inplace=True)
df.sort_index(inplace=True)
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df.index.to_series().apply(get_season)
seasonal_sales = df.groupby(['hospital_id', 'season'])['treatment_cost'].mean().reset_index()

# Step 4: Treatment Temporal Growth Rate
df['month'] = df.index.to_period('M')
# Count the number of treatments per month for each treatment type
monthly_treatment_counts = df.groupby(['treatment_id', 'month']).size().reset_index(name='treatment_count')
# Handle division by zero
def calculate_growth_rate(x):
    if x == 0:
        return 0  # Or another appropriate value, e.g., np.nan
    else:
        return (x - x.shift(1)) / x.shift(1)
# Calculate the MoM growth rate in treatment counts
monthly_treatment_counts['growth_rate'] = monthly_treatment_counts.groupby('treatment_id')['treatment_count'].pct_change()
# Identify the top 5 treatments with the highest average growth rate
top_growth_treatments = monthly_treatment_counts.groupby('treatment_id')['growth_rate'].mean().nlargest(5)

# Step 5: Patient Temporal Loyalty
patient_months = df.groupby('patient_id')['month'].nunique()
loyal_patients = patient_months[patient_months >= 3].index
loyal_sales = df[df['patient_id'].isin(loyal_patients)].groupby('hospital_id')['treatment_cost'].mean().reset_index()

# Step 6: Anomalous Treatment Detection
hospital_mean_cost = df.groupby('hospital_id')['treatment_cost'].mean()
hospital_std_cost = df.groupby('hospital_id')['treatment_cost'].std()
df.dropna(subset=['hospital_id'], inplace=True)
df['anomalous'] = df.apply(lambda row: np.abs(row['treatment_cost'] - hospital_mean_cost[row['hospital_id']]) > 3 * hospital_std_cost[row['hospital_id']], axis=1)

# Step 7: Frequent Diagnosis Pairing
def get_diagnosis_pairs(transactions):
    pairs = []
    for transaction in transactions:
        pairs.extend(combinations(transaction, 2))
    return pairs

transactions = df.groupby('patient_id')['diagnosis'].apply(list)
diagnosis_pairs = get_diagnosis_pairs(transactions)
pair_counts = Counter(diagnosis_pairs)
pair_lift = {pair: count / (df['diagnosis'].value_counts()[pair[0]] * df['diagnosis'].value_counts()[pair[1]]) for pair, count in pair_counts.items()}
top_pairs = sorted(pair_lift.items(), key=lambda x: x[1], reverse=True)[:5]

# Step 8: Day-of-Week Admission Patterns
df['day_of_week'] = df.index.day_name()
day_sales = df.groupby('day_of_week')['treatment_cost'].mean().reset_index()

# Step 9: Hospital Performance Clustering
monthly_sales_pivot = df.groupby(['hospital_id', 'month'])['treatment_cost'].sum().unstack(fill_value=0)
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(monthly_sales_pivot)
monthly_sales_pivot['cluster'] = clusters

# Step 10: Dynamic Treatment Cost Sensitivity
# Count the number of admissions for each treatment
treatment_counts = df.groupby(['treatment_id', df.index]).size().reset_index(name='treatment_count')
# Merge the treatment counts with the original DataFrame
df = df.merge(treatment_counts, on=['treatment_id', 'admission_date'])
# Calculate the price elasticity for each treatment
price_elasticity = df.groupby('treatment_id').apply(lambda x: np.corrcoef(x['treatment_cost'], x['treatment_count'])[0, 1], include_groups=False).reset_index(name='price_elasticity')
# Identify the top 5 treatments with the highest price elasticity
top_elastic_treatments = price_elasticity.nlargest(5, 'price_elasticity')

# Step 11: Predictive Admission Forecasting
forecast_results = {}
mae_results = {}
for hospital_id, group in df.groupby('hospital_id'):
    # Count admissions (assuming admission_date is the index)
    admission_counts = group.groupby(group.index)['admission_date'].size().reset_index(name='admission_count')
    # Build and fit the model (adjust parameters if needed)
    model = ExponentialSmoothing(admission_counts['admission_count'], trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    # Forecast admissions for the next three months
    forecast = fit.forecast(3)
    # Store results
    forecast_results[hospital_id] = forecast
    # Calculate MAE (assuming we have actual data for the forecast period)
    actual = admission_counts['admission_count'][-3:]
    mae = mean_absolute_error(actual, forecast[:len(actual)])
    mae_results[hospital_id] = mae

# Display results
print("Rolling Sales:\n", rolling_sales.head())
print("Seasonal Sales:\n", seasonal_sales.head())
print("Top Growth Treatments:\n", top_growth_treatments)
print("Loyal Sales:\n", loyal_sales.head())
print("Anomalous Treatments:\n", df[df['anomalous']].head())
print("Top Diagnosis Pairs:\n", top_pairs)
print("Day Sales:\n", day_sales)
print("Hospital Clusters:\n", monthly_sales_pivot.head())
print("Top Elastic Treatments:\n", top_elastic_treatments)
print("Sales Forecast:\n", forecast_results)

# Visualizations

# Visualize rolling 3-month cost for a specific hospital
hospital_id = df['hospital_id'].unique()[0]
plt.figure(figsize=(12, 6))
plt.plot(df[df['hospital_id'] == hospital_id]['admission_date'], df[df['hospital_id'] == hospital_id]['rolling_3month_cost'])
plt.title(f'Rolling 3-Month Treatment Cost for Hospital {hospital_id}')
plt.xlabel('Date')
plt.ylabel('Rolling 3-Month Cost')
plt.show()

# Seasonal Sales Visualization
seasonal_pivot = seasonal_sales.pivot(index='hospital_id', columns='season', values='treatment_cost')
seasonal_pivot.plot(kind='bar', figsize=(10, 6))
plt.title('Mean Treatment Cost per Season for Each Hospital')
plt.xlabel('Hospital ID')
plt.ylabel('Mean Treatment Cost')
plt.show()

# Day-of-Week Sales Visualization
day_sales.set_index('day_of_week').reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).plot(kind='bar', figsize=(10, 6))
plt.title('Mean Treatment Cost by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Mean Treatment Cost')
plt.show()
