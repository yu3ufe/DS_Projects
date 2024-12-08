import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Read the data
df = pd.read_csv('electronics_store_data.csv')


# Step 2: Implement outlier detection methods
# Method 1 - Using Z-score
# Calculate mean and standard deviation
mean_expenses = df['Expenses'].mean()
std_expenses = df['Expenses'].std()

# Calculate Z-scores
df['Z_score'] = (df['Expenses'] - mean_expenses) / std_expenses

# Define a threshold for Z-scores to identify outliers
threshold = 3
outliers_z = df[np.abs(df['Z_score']) > threshold]
print(f"Number of outliers detected by Z-Score: {len(outliers_z)}")

# Method 2 - Using IQR
Q1 = df['Expenses'].quantile(0.25)
Q3 = df['Expenses'].quantile(0.75)
IQR = Q3 - Q1

# Define a range to identify outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['Expenses'] < lower_bound) | (df['Expenses'] > upper_bound)]
print(f"Number of outliers detected by IQR: {len(outliers_iqr)}")

# Step 3: Visualize outliers using box plots
# Box plot to visualize outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Expenses', data=df)
plt.title('Box plot of Expenses by Category')
plt.show()

# Step 4: Winsorize Outliers - Winsorization will cap extreme values at a specified percentile.
# Define the percentile limits for Winsorization
lower_percentile = 0.05
upper_percentile = 0.95

# Calculate the lower and upper limits
lower_limit = df['Expenses'].quantile(lower_percentile)
upper_limit = df['Expenses'].quantile(upper_percentile)

# Apply Winsorization
df['Expenses_winsorized'] = df['Expenses']
df['Expenses_winsorized'] = np.where(df['Expenses_winsorized'] < lower_limit, lower_limit, df['Expenses_winsorized'])
df['Expenses_winsorized'] = np.where(df['Expenses_winsorized'] > upper_limit, upper_limit, df['Expenses_winsorized'])

# Step 5: Compare Summary Statistics
# Summary statistics before outlier treatment
summary_before = df['Expenses'].describe()

# Summary statistics after outlier treatment
summary_after = df['Expenses_winsorized'].describe()

# Display the results
print('Summary Statistics Before Outlier Treatment:')
print(summary_before)
print('\nSummary Statistics After Outlier Treatment:')
print(summary_after)
# Visual comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(df['Expenses'], ax=ax1, kde=True)
ax1.set_title('Distribution Before Outlier Treatment')

sns.histplot(df['Expenses_winsorized'], ax=ax2, kde=True)
ax2.set_title('Distribution After Outlier Treatment')

plt.show()

print("\nSummary of Outlier Impact:")
print(f"- The Z-Score method identified {len(outliers_z)} outliers.")
print(f"- The IQR method identified {len(outliers_iqr)} outliers.")
print("- After winsorization, the max and min values are adjusted, potentially reducing skewness and kurtosis.")
print("- The mean might move closer to the median, indicating a less skewed distribution.")
print("- Standard deviation typically decreases, reflecting less variability after treatment.")
