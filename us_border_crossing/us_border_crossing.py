import pandas as pd
import matplotlib.pyplot as plt
import gdown

gdown.download('https://drive.google.com/uc?id=1RptLxTHgJ3Pn1FrLjUUd9Xju9AGdRh29', 'Border_Crossing_Entry_Data.csv', quiet=False)

# Load the dataset
file_path = 'Border_Crossing_Entry_Data.csv'
df = pd.read_csv(file_path)

# Parse the Date column to extract month and year
df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Group by Port Name, Measure, Year, and Month to calculate the average number of entries (Value)
monthly_avg_entries = df.groupby(['Port Name', 'Measure', 'Year', 'Month'])['Value'].mean().reset_index()

# Display the result
print(monthly_avg_entries.head())
print()

# Group by Port Name and Measure to calculate the total number of entries (Value)
total_entries = df.groupby(['Port Name', 'Measure'])['Value'].sum().reset_index()

# Sort the results by Measure and Value in descending order
sorted_entries = total_entries.sort_values(by=['Measure', 'Value'], ascending=[True, False])

# Identify the top 5 ports with the highest total entries for each Measure
top_5_ports_per_measure = sorted_entries.groupby('Measure').head(5)

# Display the result
print(top_5_ports_per_measure)
print()

# Filter the dataset to include only the entries within Latitude between 25 and 50 and Longitude between -125 and -70
filtered_df = df[(df['Latitude'] >= 25) & (df['Latitude'] <= 50) & (df['Longitude'] >= -125) & (df['Longitude'] <= -70)]

# Display the result
print(filtered_df.head())
print()

# Parse the Date column to extract year
df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
df['Year'] = df['Date'].dt.year

# Group by Port Name, Measure, and Year to calculate the total number of entries (Value)
yearly_entries = df.groupby(['Port Name', 'Measure', 'Year'])['Value'].sum().reset_index()

# Calculate the year-over-year growth
yearly_entries['YoY_Growth'] = yearly_entries.groupby(['Port Name', 'Measure'])['Value'].pct_change() * 100

# Filter the results to highlight ports with a growth rate of over 50%
high_growth_ports = yearly_entries[yearly_entries['YoY_Growth'] > 50]

# Display the result
print(high_growth_ports)
print()

# Parse the Date column to extract month and year
df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Group by Port Name, Measure, and Month to calculate the mean and standard deviation of entries (Value)
grouped = df.groupby(['Port Name', 'Measure', 'Month'])['Value']
mean_values = grouped.transform('mean')
std_values = grouped.transform('std')

# Detect anomalies where the number of entries is more than three standard deviations away from the mean
anomalies = df[(df['Value'] > mean_values + 3 * std_values) | (df['Value'] < mean_values - 3 * std_values)]

# Display the result
print(anomalies)
print()

# Aggregate the total number of entries (Value) by State and Measure
state_measure_agg = df.groupby(['State', 'Measure'])['Value'].sum().reset_index()

# Rank the states based on the total number of entries for each Measure
state_measure_agg['Rank'] = state_measure_agg.groupby('Measure')['Value'].rank(ascending=False, method='dense')

# Display the result
print(state_measure_agg.sort_values(by=['Measure', 'Rank']).head())
print()

# Parse the Date column to extract month and year
df['Date'] = pd.to_datetime(df['Date'], format='%b %Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Group by Measure and Month to calculate the average number of entries (Value)
monthly_avg = df.groupby(['Measure', 'Month'])['Value'].mean().reset_index()

# Pivot the data for better visualization
pivot_table = monthly_avg.pivot(index='Month', columns='Measure', values='Value')

# Plot the results
pivot_table.plot(kind='line', figsize=(12, 8), marker='o')
plt.title('Average Number of Entries by Measure and Month')
plt.xlabel('Month')
plt.ylabel('Average Number of Entries')
plt.legend(title='Measure')
plt.grid(True)
plt.show()
