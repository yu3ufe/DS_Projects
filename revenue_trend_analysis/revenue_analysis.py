import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv("product_sales.csv")

# 1. Fill missing values and ensure non-negative
df['price'] = df['price'].fillna(df['price'].median()).abs()
df['quantity'] = df['quantity'].fillna(df['quantity'].median()).abs()

# 2. Calculate revenue
df['revenue'] = df['price'] * df['quantity']

# 3. Extract month from sale_date
df['sale_month'] = pd.to_datetime(df['sale_date']).dt.month

# 4. Apply discount in December
df['price'] = df.apply(lambda row: row['price'] * 0.9 if row['sale_month'] == 12 else row['price'], axis=1)
df['revenue'] = df['price'] * df['quantity']

# Group by Category and Month and calculate total_revenue and average price for each group
grouped_data = df.groupby(['category', 'sale_month']).agg({'revenue': 'sum', 'price': 'mean'}).reset_index()

# Find the month with the highest revenue for each category
highest_revenue_month = grouped_data.groupby('category')['revenue'].idxmax()
highest_revenue_months = grouped_data.loc[highest_revenue_month]
print(f"\n\nHighest Revenue Months:\n{highest_revenue_months}")

# Convert 'sale_date' to datetime
df['sale_date'] = pd.to_datetime(df['sale_date'])
# Filter out February sales
df_without_february = df[df['sale_date'].dt.month != 2].copy()
# Calculate cumulative revenue by product and date
df_without_february['cumulative_revenue'] = df_without_february.groupby(['ProductID', 'sale_date'])['revenue'].cumsum()
# Sort by cumulative revenue and category
df_without_february = df_without_february.sort_values(by=['category', 'cumulative_revenue'], ascending=False)
# Get top 3 products by cumulative revenue in each category
top_3_products = df_without_february.groupby('category').head(3)
# Print the results
print(f"\n\nTop 3 Products:\n{top_3_products[['category', 'ProductID', 'sale_date', 'cumulative_revenue']]}")

# Create a pivot table to get monthly revenue by category
pivot_table = df.pivot_table(index='category', columns='sale_month', values='revenue', aggfunc='sum')
# Calculate the percentage change from the previous month
pivot_table_percentage_change = pivot_table.pct_change(axis=1).shift(-1) * 100
pivot_table_percentage_change.columns = [f'Percent Change {col}' for col in pivot_table_percentage_change.columns]
pivot_table = pd.concat([pivot_table, pivot_table_percentage_change], axis=1)

# Plot revenue trend for each category
plt.figure(figsize=(12, 6))
for category in pivot_table.index:
    plt.plot(pivot_table.columns[:12], pivot_table.loc[category, :12], label=category)

# Highlight the month with highest revenue for each category
for category, month in highest_revenue_months[['category', 'sale_month']].itertuples(index=False):
    plt.scatter(month, pivot_table.loc[category, month], color='red')

plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.title('Revenue Trend by Category')
plt.legend()
plt.xticks()
plt.show()
