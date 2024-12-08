import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Air_Quality.csv')

# Requirement 1: Analyze 2019 citywide air quality
df_2019 = df[(df['Geo Type Name'] == 'Citywide') & (df['Time Period'].str.contains('2019'))]
avg_quality_2019 = df_2019.groupby('Name')['Data Value'].mean().sort_values(ascending=False).reset_index()
print("Average Air Quality Indicators for Citywide in 2019:\n", avg_quality_2019)

# Requirement 2: Time-based analysis
df['Start_Date'] = pd.to_datetime(df['Start_Date'])
df['Year'] = df['Start_Date'].dt.year
df_trend = df[(df['Year'] >= 2010) & (df['Year'] <= 2020)]

# Requirement 3: Extract numeric part from Geo Join ID
df['Geo Join Numeric'] = df['Geo Join ID'].astype(str).str.extract('(\d+)', expand=False).astype(float)

# Requirement 4: Standardize Measure Info
df['Measure Info'] = df['Measure Info'].str.lower().str.strip()

# Requirement 5: Create pivot table for top 5 Geo Place Names with highest total Data Value
pivot = df.pivot_table(values='Data Value', index='Geo Place Name', columns='Name', aggfunc='sum')
top_5_places = pivot.sum(axis=1).nlargest(5).index
pivot_top_5 = pivot.loc[top_5_places]
print("Top 5 Geo Place Names with Highest Total Data Value:")
print(pivot_top_5)
print("\n")

# Requirement 6: Count missing values in each column
missing_values = df.isnull().sum()
print("Missing Values in Each Column:")
print(missing_values)
print("\n")

# Requirement 7: Fill missing values in Data Value
df['Data Value'] = df['Data Value'].fillna(df['Data Value'].median())

# Requirement 8: Filter for specific indicators and find highest average Data Value
filtered_df = df[(df['Indicator ID'].isin([640, 365])) & (df['Measure'] == 'Mean')]
highest_pollution_area = filtered_df.groupby('Geo Place Name')['Data Value'].mean().sort_values(ascending=False).head(1)
print("\nGeo Place Name with the highest average Data Value for critical pollutants:\n", highest_pollution_area)

# Requirement 9: Group by Geo Type Name and Time Period
grouped = df.groupby(['Geo Type Name', 'Time Period', 'Name'])['Data Value'].median().reset_index()
so2_emissions = grouped[grouped['Name'] == 'Boiler Emissions- Total SO2 Emissions']
highest_so2 = so2_emissions.loc[so2_emissions['Data Value'].idxmax()]
print("\nGeo Type Name and Time Period with highest median SO2 Emissions:")
print(highest_so2[['Geo Type Name', 'Time Period', 'Data Value']])

# Requirement 10: Adjust Data Value for measures with 'per'
def split_measure_info(row):
    if 'per' in row['Measure Info']:
        parts = row['Measure Info'].split('per')
        if len(parts) > 1:
            denominator = ''.join(filter(str.isdigit, parts[1]))
            if denominator:
                return float(denominator)
    return 1  # Default multiplier if 'per' not found or no numeric denominator

df['Multiplier'] = df.apply(split_measure_info, axis=1)
df['Adjusted Data Value'] = df['Data Value'] * df['Multiplier']
high_adjusted_values = df[df['Adjusted Data Value'] > 1000]
print("\nRows where Adjusted Data Value > 1000:\n", high_adjusted_values)

# Requirement 11: Merge Geo Join ID
unique_geo = df[['Geo Place Name', 'Geo Join ID']].drop_duplicates()
df_merged = pd.merge(df, unique_geo, on='Geo Place Name', suffixes=('', '_merged'))
df_merged = df_merged[df_merged['Geo Join ID_merged'].notnull()]
total_by_geo = df_merged.groupby('Geo Join ID_merged')['Data Value'].sum().reset_index()
print("\nTotal Data Value for each Geo Join ID:")
print(total_by_geo.head())

# Requirement 12: Normalize Data Value
def normalize_data_value(group):
    mean = group['Data Value'].mean()
    std = group['Data Value'].std()
    group['Normalized Data Value'] = (group['Data Value'] - mean) / std
    return group

df_normalized = df.groupby('Indicator ID').apply(normalize_data_value, include_groups=False)
df_filtered = df_normalized[df_normalized['Normalized Data Value'].abs() <= 1]
print("\nNumber of rows within one standard deviation:", len(df_filtered))
