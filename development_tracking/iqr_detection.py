import pandas as pd

# Load the dataset
df = pd.read_csv('Development_Tracking.csv')

# Group by 'ZONING_TO' and filter groups with more than 5 entries
grouped = df.groupby('ZONING_TO').filter(lambda x: len(x) > 5)

# Sum the 'SHAPESTArea' for each 'ZONING_TO'
result = grouped.groupby('ZONING_TO')['SHAPESTArea'].sum()

# Print the result
print(result)

# Identify columns with missing values
missing_values = df.isnull().sum()
print("\nColumns with missing values:\n", missing_values[missing_values > 0])

# Replace missing values in the AGE_RESTRI column with the mode of the column
age_restri_mode = df['AGE_RESTRI'].mode()[0]
df['AGE_RESTRI'] = df['AGE_RESTRI'].fillna(age_restri_mode)

# Replace missing values in the HOUSING_UN column with the median of the column
housing_un_median = df['HOUSING_UN'].median()
df['HOUSING_UN'] = df['HOUSING_UN'].fillna(housing_un_median)

# Replace missing values in AGE_REST_2 with the mean of the column if AGE_RESTRI is "YES"
age_rest_2_mean = df['AGE_REST_2'].mean()
df.loc[df['AGE_RESTRI'] == 'YES', 'AGE_REST_2'] = df.loc[df['AGE_RESTRI'] == 'YES', 'AGE_REST_2'].fillna(age_rest_2_mean)

# Print the updated DataFrame to verify changes
print()
print(df.head())

# Filter the dataset to include only rows where FINALAPPRO is "Approved"
filtered_df = df[df['FINALAPPRO'] == 'Approved']

# Sort the filtered dataset by DATEOFAPPL in descending order
sorted_df = filtered_df.sort_values(by='DATEOFAPPL', ascending=False)

# Select the top 10 entries based on the sorted order
top_10_df = sorted_df.head(10).copy()

# Convert DATEOFAPPL and FINALACTIONDATE columns to datetime, handling timezone differences
top_10_df['DATEOFAPPL'] = pd.to_datetime(top_10_df['DATEOFAPPL'], errors='coerce').dt.tz_localize(None)
top_10_df['FINALACTIONDATE'] = pd.to_datetime(top_10_df['FINALACTIONDATE'], errors='coerce').dt.tz_localize(None)

# Calculate the time difference in days between DATEOFAPPL and FINALACTIONDATE
top_10_df['APPROVAL_TIME_DAYS'] = (top_10_df['FINALACTIONDATE'] - top_10_df['DATEOFAPPL']).dt.days

# Print the resulting DataFrame
print()
print(top_10_df[['DATEOFAPPL', 'FINALACTIONDATE', 'APPROVAL_TIME_DAYS']])

# Filter the dataset to include only rows where RESIDENTIAL is "YES"
residential_df = df[df['RESIDENTIAL'] == 'YES']

# Calculate the average CALCACREAG for these residential projects (exclude outliers)
Q1 = residential_df['CALCACREAG'].quantile(0.25)
Q3 = residential_df['CALCACREAG'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
filtered_residential_df = residential_df[(residential_df['CALCACREAG'] >= lower_bound) & (residential_df['CALCACREAG'] <= upper_bound)]

# Calculate the average CALCACREAG for these residential projects
average_calcacreag = filtered_residential_df['CALCACREAG'].mean()

# Group the residential projects by TYPE_ and calculate the average CALCACREAG for each type
grouped_avg_calcacreag = filtered_residential_df.groupby('TYPE_')['CALCACREAG'].mean()

# Print the results
print(f"\nAverage CALCACREAG for residential projects (excluding outliers): {average_calcacreag}")
print("\nAverage CALCACREAG for each type of residential project:")
print(grouped_avg_calcacreag)

# Create a new column ZONING_CHANGE that indicates whether the zoning has changed
df['ZONING_CHANGE'] = df['ZONING_FRO'] != df['ZONING_TO']

# Count the number of projects where the zoning has changed
zoning_change_count = df['ZONING_CHANGE'].sum()

# Calculate the percentage of total projects that have undergone a zoning change
total_projects = len(df)
zoning_change_percentage = (zoning_change_count / total_projects) * 100

# Analyze the ZONING_CHANGE column to find the most common zoning change and the number of occurrences of this change
common_zoning_change = df[df['ZONING_CHANGE']].groupby(['ZONING_FRO', 'ZONING_TO']).size().idxmax()
common_zoning_change_count = df[df['ZONING_CHANGE']].groupby(['ZONING_FRO', 'ZONING_TO']).size().max()

# Print the results
print(f"\nNumber of projects where the zoning has changed: {zoning_change_count}")
print(f"Percentage of total projects that have undergone a zoning change: {zoning_change_percentage:.2f}%")
print(f"Most common zoning change: {common_zoning_change} with {common_zoning_change_count} occurrences")
