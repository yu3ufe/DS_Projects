import pandas as pd
from geopy.distance import geodesic
import seaborn as sns
import matplotlib.pyplot as plt
import gdown

gdown.download('https://drive.google.com/uc?id=1S8M5VswfpFzBNcxZ36hRqtHn8zLe1X8-', 'NYPD_Arrest_Data__Year_to_Date_.csv', quiet=False)

# Load the dataset
df = pd.read_csv('NYPD_Arrest_Data__Year_to_Date_.csv')

# Convert ARREST_DATE to datetime format
df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], format='%m/%d/%Y')

# Extract month and year and create a new column ARREST_MONTH_YEAR
df['ARREST_MONTH_YEAR'] = df['ARREST_DATE'].dt.to_period('M')

# Filter the dataset to include only arrests made in the first quarter (January to March) of 2024
filtered_df = df[(df['ARREST_DATE'].dt.year == 2024) & (df['ARREST_DATE'].dt.month <= 3)]

print()
print(filtered_df.head())
print()

# Define the coordinates for Times Square
times_square_coords = (40.7580, -73.9855)

# Function to calculate distance to Times Square
def calculate_distance(lat, lon):
    arrest_coords = (lat, lon)
    return geodesic(arrest_coords, times_square_coords).kilometers

# Calculate the distance and create a new column DISTANCE_TO_TIMES_SQUARE
df['DISTANCE_TO_TIMES_SQUARE'] = df.apply(lambda row: calculate_distance(row['Latitude'], row['Longitude']), axis=1)

print(df['DISTANCE_TO_TIMES_SQUARE'].head())
print()

# One-hot encode the LAW_CAT_CD column
df_encoded = pd.get_dummies(df, columns=['LAW_CAT_CD'])

# Group by ARREST_BORO and AGE_GROUP and calculate the average DISTANCE_TO_TIMES_SQUARE for each group
grouped_df = df_encoded.groupby(['ARREST_BORO', 'AGE_GROUP'])['DISTANCE_TO_TIMES_SQUARE'].mean().reset_index()

print(grouped_df.head())
print()

# Replace missing or null values in the OFNS_DESC column with the mode
mode_ofns_desc = df['OFNS_DESC'].mode()[0]
df['OFNS_DESC'] = df['OFNS_DESC'].fillna(mode_ofns_desc)

# Define typical range for New York City coordinates
x_coord_min, x_coord_max = 912500, 1067380
y_coord_min, y_coord_max = 120000, 272000

# Detect and remove rows where X_COORD_CD or Y_COORD_CD values are outside the typical range
df_filtered = df[(df['X_COORD_CD'] >= x_coord_min) & (df['X_COORD_CD'] <= x_coord_max) &
                 (df['Y_COORD_CD'] >= y_coord_min) & (df['Y_COORD_CD'] <= y_coord_max)]

print(df_filtered.head())
print()

# Create a pivot table that shows the count of arrests for each combination of PERP_RACE and PERP_SEX, broken down by LAW_CAT_CD
pivot_table = pd.pivot_table(df, values='ARREST_KEY', index=['PERP_RACE', 'PERP_SEX'], columns='LAW_CAT_CD', aggfunc='count', fill_value=0)

# Generate a heatmap to visualize this pivot table
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
plt.title('Count of Arrests by PERP_RACE, PERP_SEX, and LAW_CAT_CD')
plt.xlabel('LAW_CAT_CD')
plt.ylabel('PERP_RACE and PERP_SEX')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Show the heatmap
plt.show()
