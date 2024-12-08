import pandas as pd

# Load the dataset
df = pd.read_csv('Death_rates_for_suicide__US.csv')

# Filter the data for age-adjusted rates
df_age_adjusted = df[df['UNIT'] == 'Deaths per 100,000 resident population, age-adjusted']

# Filter the data for "Male: White" and "Female: Black or African American"
df_male_white = df_age_adjusted[df_age_adjusted['STUB_LABEL'] == 'Male: White']
df_female_black = df_age_adjusted[df_age_adjusted['STUB_LABEL'] == 'Female: Black or African American']

# Calculate the average annual suicide rate for each demographic
average_male_white = df_male_white['ESTIMATE'].mean()
average_female_black = df_female_black['ESTIMATE'].mean()

print(f"\nAverage annual suicide rate for Male: White: {average_male_white:.2f}%")
print(f"Average annual suicide rate for Female: Black or African American: {average_female_black:.2f}%")

# Filter the data for the age group "15-24 years"
df_age_group = df[df['AGE'] == '15-24 years']

# Check if the filtered DataFrame is empty
if df_age_group.empty:
    print("No data available for age group '15-24 years'.")
else:
    # Group by year and calculate the mean suicide rate for each year
    df_yearly = df_age_group.groupby('YEAR')['ESTIMATE'].mean().reset_index()

    # Sort by year to ensure correct YOY calculation
    df_yearly = df_yearly.sort_values(by='YEAR')

    # Check if the yearly DataFrame is empty
    if df_yearly.empty:
        print("No data available after grouping by year.")
    else:
        # Calculate the year-over-year change in suicide rates
        df_yearly['YOY_CHANGE'] = df_yearly['ESTIMATE'].diff()

        # Check if the YOY_CHANGE column is empty
        if df_yearly['YOY_CHANGE'].isnull().all():
            print("No year-over-year change data available.")
        else:
            # Identify the year with the highest year-over-year increase
            max_increase_year = df_yearly.loc[df_yearly['YOY_CHANGE'].idxmax()]

            print(f"\nYear with the highest year-over-year increase: {int(max_increase_year['YEAR'])}")
            print(f"Increase in suicide rate: {max_increase_year['YOY_CHANGE']:.2f}%\n")

# Filter the data for the age groups "10-14 years" and "25-34 years"
df_filtered = df[df['AGE'].isin(['10-14 years', '25-34 years'])].copy()

# Create a new column for the decade
df_filtered['DECADE'] = (df_filtered['YEAR'] // 10) * 10

# Group by decade and age group, and calculate the mean suicide rate for each group
df_decade = df_filtered.groupby(['DECADE', 'AGE'])['ESTIMATE'].mean().reset_index()

# Pivot the table to compare the two age groups side by side
df_pivot = df_decade.pivot(index='DECADE', columns='AGE', values='ESTIMATE').reset_index()

# Determine which age group had higher rates on average for each decade
df_pivot['Higher_Rate_Age_Group'] = df_pivot.apply(
    lambda row: '10-14 years' if row['10-14 years'] > row['25-34 years'] else '25-34 years', axis=1
)

# Format the columns to 2 decimal places
df_pivot['10-14 years'] = df_pivot['10-14 years'].map('{:.2f}'.format)
df_pivot['25-34 years'] = df_pivot['25-34 years'].map('{:.2f}'.format)

# Print the summary
print(df_pivot[['DECADE', '10-14 years', '25-34 years', 'Higher_Rate_Age_Group']])
