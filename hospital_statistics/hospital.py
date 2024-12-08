import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("hospital_data.csv")

def standardize_time(time_str):
    if 'AM' in time_str or 'PM' in time_str:
        return pd.to_datetime(time_str, format='%I:%M:%S %p').strftime('%H:%M:%S')
    else:
        return time_str

df['arrival_time'] = df['arrival_time'].apply(standardize_time)

def normalize_to_minutes(value, unit):
    if unit == 'hours':
        return value * 60
    else:
        return value

df['treatment_duration_minutes'] = df.apply(lambda row: normalize_to_minutes(row['treatment_duration'], row['treatment_duration_unit']), axis=1)
df['equipment_usage_minutes'] = df.apply(lambda row: normalize_to_minutes(row['equipment_usage'], row['equipment_usage_unit']), axis=1)

# For numerical columns
num_imputer = IterativeImputer(estimator=RandomForestRegressor(), max_iter=10, random_state=0)
df[['treatment_duration_minutes', 'equipment_usage_minutes']] = num_imputer.fit_transform(df[['treatment_duration_minutes', 'equipment_usage_minutes']])

# Function to assign random True/False to NaNs
def impute_random_boolean(column):
    return column.apply(lambda x: np.random.choice([True, False]) if pd.isna(x) else x)

# Apply the function to the column
df['doctor_availability'] = impute_random_boolean(df['doctor_availability'])

doctor_available = df[df['doctor_availability'] == True]['treatment_duration_minutes']
doctor_unavailable = df[df['doctor_availability'] == False]['treatment_duration_minutes']

t_stat, p_value = stats.ttest_ind(doctor_available, doctor_unavailable)

print(f"\nT-statistic: {t_stat}")
print(f"\nP-value: {p_value}\n")

# Plotting
plt.figure(figsize=(10, 6))
plt.boxplot([doctor_available, doctor_unavailable], labels=['Doctor Available', 'Doctor Unavailable'])
plt.title('Treatment Duration by Doctor Availability')
plt.ylabel('Treatment Duration (minutes)')
plt.show()

correlation = df['treatment_duration_minutes'].corr(df['equipment_usage_minutes'])
print(f"\nCorrelation coefficient: {correlation}\n")

# Visualize the correlation
plt.figure(figsize=(10, 6))
plt.scatter(df['treatment_duration_minutes'], df['equipment_usage_minutes'])
plt.xlabel('Treatment Duration (minutes)')
plt.ylabel('Equipment Usage (minutes)')
plt.title('Treatment Duration vs Equipment Usage')
plt.show()
print("\n")

df['arrival_hour'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S').dt.hour

plt.figure(figsize=(12, 6))
df['arrival_hour'].hist(bins=24, range=(0, 24))
plt.title('Distribution of Patient Arrivals by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Arrivals')
plt.xticks(range(0, 24))
plt.show()
