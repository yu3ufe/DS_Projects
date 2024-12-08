import pandas as pd
from geopy.distance import geodesic
import numpy as np
import gdown

gdown.download('https://drive.google.com/uc?id=1yyIp0TN-btqMjWhMKPRaQYvlcnTNYPeg', 'Real_Estate_Sales.csv', quiet=False)

# Load the dataset
file_path = 'Real_Estate_Sales.csv'
df = pd.read_csv(file_path, low_memory=False)

# Filter the dataset to include only properties listed in the year 2020
df_2020 = df[df['List Year'] == 2020]

# Group the filtered data by Town and calculate the average Sale Amount and Assessed Value for each town
grouped_df = df_2020.groupby('Town').agg({'Sale Amount': 'mean', 'Assessed Value': 'mean'}).reset_index()

# Identify the town with the highest average Sale Amount
highest_avg_sale_amount_town = grouped_df.loc[grouped_df['Sale Amount'].idxmax()]

# Identify the town with the highest average Assessed Value
highest_avg_assessed_value_town = grouped_df.loc[grouped_df['Assessed Value'].idxmax()]

print("Town with the highest average Sale Amount:")
print(highest_avg_sale_amount_town)

print("\nTown with the highest average Assessed Value:")
print(highest_avg_assessed_value_town)

# Convert the Date Recorded column to a datetime format
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'], format='%m/%d/%Y')

# Extract the month and year from the Date Recorded column and create two new columns: Recorded Month and Recorded Year
df['Recorded Month'] = df['Date Recorded'].dt.month
df['Recorded Year'] = df['Date Recorded'].dt.year

# Filter the dataset to include only sales recorded in the year 2021
df_2021 = df[df['Recorded Year'] == 2021]

# Calculate the total number of sales recorded in each month of the year 2021
monthly_sales_2021 = df_2021.groupby('Recorded Month').size().reset_index(name='Total Sales')

print(f"\n{monthly_sales_2021}")

# Parse the Location column to extract the latitude and longitude values into two separate columns
def parse_location(location):
    try:
        # Extract the latitude and longitude from the location string
        lon, lat = location.replace('POINT (', '').replace(')', '').split()
        return float(lat), float(lon)
    except:
        return np.nan, np.nan

df['Latitude'], df['Longitude'] = zip(*df['Location'].apply(parse_location))

# Coordinates of the state capital (Hartford, CT)
state_capital_coords = (41.7658, -72.6734)

# Calculate the distance between each property and the state capital
df['Distance to Capital'] = df.apply(lambda row: geodesic((row['Latitude'], row['Longitude']), state_capital_coords).miles if not np.isnan(row['Latitude']) and not np.isnan(row['Longitude']) else np.nan, axis=1)

# Identify the property that is the farthest from the central point
farthest_property = df.loc[df['Distance to Capital'].idxmax()]

print("\nProperty farthest from the state capital:")
print(farthest_property)

# Create a new column Sale to Assessed Ratio by dividing the Sale Amount by the Assessed Value
df['Sale to Assessed Ratio'] = df['Sale Amount'] / df['Assessed Value']

# Identify properties where the Sale to Assessed Ratio is greater than 1.5
high_ratio_properties = df[df['Sale to Assessed Ratio'] > 1.5]

# Calculate the percentage of such properties out of the total number of properties
percentage_high_ratio_properties = (len(high_ratio_properties) / len(df)) * 100

print(f"\nPercentage of properties with Sale to Assessed Ratio greater than 1.5: {percentage_high_ratio_properties:.2f}%")

# Perform a keyword search in the Assessor Remarks and OPM remarks columns to identify properties that mention "renovation"
renovation_mask = df['Assessor Remarks'].str.contains('renovation', case=False, na=False) | df['OPM remarks'].str.contains('renovation', case=False, na=False)

# Filter the dataset to include only properties that mention "renovation"
renovated_properties = df[renovation_mask]

# Count the number of properties that have undergone renovation according to the remarks
num_renovated_properties = len(renovated_properties)

# Create a summary table showing the count of renovated properties by Property Type
summary_table = renovated_properties.groupby('Property Type').size().reset_index(name='Count of Renovated Properties')

print(f"Number of properties that have undergone renovation: {num_renovated_properties}")
print("\nSummary table showing the count of renovated properties by Property Type:")
print(summary_table)
