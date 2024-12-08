import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
from scipy.stats import ttest_ind

# Load the dataset
file_path = 'Youth_Tobacco_Survey__YTS__Data.csv'
df = pd.read_csv(file_path)

# Handle NaNs in Data_Value by filling them with the mean of the column
df['Data_Value'] = df['Data_Value'].fillna(df['Data_Value'].mean())

# Group the data by YEAR and Gender, and calculate the average Data_Value for each group
grouped_df = df.groupby(['YEAR', 'Gender'])['Data_Value'].mean().reset_index()

# Print the resulting dataframe
print()
print(grouped_df)
print()

# Filter for current cigarette use
cig_use_df = df[(df['TopicDesc'] == 'Cigarette Use (Youth)') & (df['Response'] == 'Current')]

# Group by year to find max and min
def find_extremes(group):
    max_state = group.loc[group['Data_Value'].idxmax()]
    min_state = group.loc[group['Data_Value'].idxmin()]
    return pd.Series({
        'Max_State': max_state['LocationDesc'],
        'Max_Data_Value': max_state['Data_Value'],
        'Max_Low_Confidence': max_state['Low_Confidence_Limit'],
        'Max_High_Confidence': max_state['High_Confidence_Limit'],
        'Min_State': min_state['LocationDesc'],
        'Min_Data_Value': min_state['Data_Value'],
        'Min_Low_Confidence': min_state['Low_Confidence_Limit'],
        'Min_High_Confidence': min_state['High_Confidence_Limit']
    })

extremes_by_year = cig_use_df.groupby('YEAR').apply(find_extremes, include_groups=False).reset_index()
print(extremes_by_year)
print()

# Filter the data for MeasureDesc "Quit Attempt in Past Year Among Current Cigarette Smokers"
filtered_df = df[df['MeasureDesc'] == 'Quit Attempt in Past Year Among Current Cigarette Smokers']

# Group the data by YEAR and LocationDesc, and calculate the average Data_Value for each group
grouped_df = filtered_df.groupby(['YEAR', 'LocationDesc'])['Data_Value'].mean().reset_index()

# Plot the trend for different states using seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(data=grouped_df, x='YEAR', y='Data_Value', hue='LocationDesc')
plt.xticks(grouped_df['YEAR'].unique())
plt.xlabel('Year')
plt.ylabel('Average Data Value')
plt.title('Trend of Quit Attempt in Past Year Among Current Cigarette Smokers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Filter the data for TopicDesc "Smokeless Tobacco Use (Youth)"
filtered_df = df[df['TopicDesc'] == 'Smokeless Tobacco Use (Youth)']

# Calculate the correlation between Sample_Size and Data_Value
correlation = filtered_df['Sample_Size'].corr(filtered_df['Data_Value'])

print(f"\nThe correlation between Sample_Size and Data_Value for the TopicDesc 'Smokeless Tobacco Use (Youth)' is {correlation:.10f}.")

# Filter the data for TopicDesc "Cigarette Use (Youth)"
filtered_df = df[df['TopicDesc'] == 'Cigarette Use (Youth)'].copy()

# Extract latitude and longitude from GeoLocation
filtered_df['Latitude'] = filtered_df['GeoLocation'].apply(lambda x: float(x.strip('()').split(', ')[0]))
filtered_df['Longitude'] = filtered_df['GeoLocation'].apply(lambda x: float(x.strip('()').split(', ')[1]))

# Group the data by LocationDesc and calculate the average Data_Value for each state
grouped_df = filtered_df.groupby(['LocationDesc', 'Latitude', 'Longitude'])['Data_Value'].mean().reset_index()

# Create a map centered around the United States
m = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

# Create a color map
colormap = cm.linear.YlOrRd_09.scale(grouped_df['Data_Value'].min(), grouped_df['Data_Value'].max())

# Add a marker cluster to the map
marker_cluster = MarkerCluster().add_to(m)

# Add markers to the map with color based on Data_Value
for idx, row in grouped_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=f"State: {row['LocationDesc']}<br>Data Value: {row['Data_Value']}",
        color=colormap(row['Data_Value']),
        fill=True,
        fill_color=colormap(row['Data_Value'])
    ).add_to(marker_cluster)

# Add the color map to the map
colormap.add_to(m)

# Save the map to an HTML file
m.save('cigarette_use_map.html')

print("\nMap has been created and saved as 'cigarette_use_map.html'.")
print()

# Filter the data for TopicDesc "Cigarette Use (Youth)"
filtered_df = df[df['TopicDesc'] == 'Cigarette Use (Youth)']

plt.figure(figsize=(12, 8))
sns.boxplot(x='Education', y='Data_Value', data=filtered_df)
plt.title('Youth Cigarette Use by Education Level')
plt.show()

# Separate the data by Education level
middle_school_df = filtered_df[filtered_df['Education'] == 'Middle School']
high_school_df = filtered_df[filtered_df['Education'] == 'High School']

# Extract the Data_Value for each education level
middle_school_values = middle_school_df['Data_Value'].dropna()
high_school_values = high_school_df['Data_Value'].dropna()

# Perform a t-test to compare the means of the two education levels
t_stat, p_value = ttest_ind(middle_school_values, high_school_values)

print(f"\nT-statistic: {t_stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("\nThere is a significant difference between Middle School and High School Data_Values for Cigarette Use (Youth).")
else:
    print("\nThere is no significant difference between Middle School and High School Data_Values for Cigarette Use (Youth).")
