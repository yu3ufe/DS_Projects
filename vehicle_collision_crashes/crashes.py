import pandas as pd
import gdown

gdown.download('https://drive.google.com/uc?id=11nbMNXCbvc0RWg8yR1D_CJipSJqruvtS', 'Motor_Vehicle_Collisions_Crashes.csv', quiet=False)

# Load the dataset
df = pd.read_csv("Motor_Vehicle_Collisions_Crashes.csv", low_memory=False)

# Calculate the total number of collisions for each unique combination of LATITUDE and LONGITUDE
collision_counts = df.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='TOTAL_COLLISIONS')

# Determine the top 5 locations with the highest number of collisions
top_5_locations = collision_counts.nlargest(5, 'TOTAL_COLLISIONS')

# For these top 5 locations, calculate the average number of persons injured and killed per collision
top_5_data = df.merge(top_5_locations, on=['LATITUDE', 'LONGITUDE'])
average_injured_killed = top_5_data.groupby(['LATITUDE', 'LONGITUDE']).agg({
    'NUMBER OF PERSONS INJURED': 'mean',
    'NUMBER OF PERSONS KILLED': 'mean'
}).reset_index()

# Merge the results to get the final output
final_output = top_5_locations.merge(average_injured_killed, on=['LATITUDE', 'LONGITUDE'])

print(final_output)
print("\n")

# Convert CRASH DATE and CRASH TIME to datetime
df['CRASH_DATETIME'] = pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'])

# Extract hour from datetime
df['CRASH_HOUR'] = df['CRASH_DATETIME'].dt.hour

# Group by BOROUGH and CRASH_HOUR, count collisions
borough_hour_counts = df.groupby(['BOROUGH', 'CRASH_HOUR']).size().reset_index(name='collision_count')

# Find peak hour for each borough
peak_hours = borough_hour_counts.loc[borough_hour_counts.groupby('BOROUGH')['collision_count'].idxmax()]

# Find most common contributing factor during peak hour for each borough
for index, row in peak_hours.iterrows():
    borough = row['BOROUGH']
    peak_hour = row['CRASH_HOUR']

    borough_peak_data = df[(df['BOROUGH'] == borough) & (df['CRASH_HOUR'] == peak_hour)]
    top_factor = borough_peak_data['CONTRIBUTING FACTOR VEHICLE 1'].mode().iloc[0]

    print(f"Borough: {borough}")
    print(f"Peak hour: {peak_hour}:00")
    print(f"Most common contributing factor: {top_factor}")
    print()

# Calculate the total number of collisions involving each vehicle type in VEHICLE TYPE CODE 1
vehicle_collision_counts = df['VEHICLE TYPE CODE 1'].value_counts().reset_index(name='TOTAL_COLLISIONS')
vehicle_collision_counts.columns = ['VEHICLE_TYPE', 'TOTAL_COLLISIONS']

# Identify the vehicle type most frequently involved in collisions where at least one person was killed
fatal_collisions = df[df['NUMBER OF PERSONS KILLED'] > 0]
most_frequent_fatal_vehicle = fatal_collisions['VEHICLE TYPE CODE 1'].value_counts().idxmax()

# For this vehicle type, analyze the most common contributing factor and the average number of persons injured per collision
most_frequent_fatal_vehicle_data = df[df['VEHICLE TYPE CODE 1'] == most_frequent_fatal_vehicle]
most_common_contributing_factor = most_frequent_fatal_vehicle_data['CONTRIBUTING FACTOR VEHICLE 1'].mode().iloc[0]
average_persons_injured = most_frequent_fatal_vehicle_data['NUMBER OF PERSONS INJURED'].mean()

# Print the results
print(f"\nTotal number of collisions involving each vehicle type:\n{vehicle_collision_counts}\n")
print(f"Vehicle type most frequently involved in fatal collisions: {most_frequent_fatal_vehicle}")
print(f"Most common contributing factor for {most_frequent_fatal_vehicle}: {most_common_contributing_factor}")
print(f"Average number of persons injured per collision for {most_frequent_fatal_vehicle}: {average_persons_injured}")
