import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from sklearn.cluster import KMeans

# Load the datasets
sales_data = pd.read_csv('sales_data.csv')
promotions = pd.read_csv('promotions.csv')
economic_indicators = pd.read_csv('economic_indicators.csv')

# Convert date columns to datetime
sales_data['date'] = pd.to_datetime(sales_data['date'])
promotions['start_date'] = pd.to_datetime(promotions['start_date'])
promotions['end_date'] = pd.to_datetime(promotions['end_date'])
economic_indicators['month'] = pd.to_datetime(economic_indicators['month'])

# Handle missing values
sales_data = sales_data.dropna()
promotions = promotions.dropna()
economic_indicators = economic_indicators.dropna()

# Handle outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

sales_data = remove_outliers(sales_data, 'units_sold')
sales_data = remove_outliers(sales_data, 'revenue')

# Group sales data by date and category
daily_sales = sales_data.groupby(['date', 'category'])['revenue'].sum().reset_index()
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['year'] = daily_sales['date'].dt.year

# Calculate average monthly revenue for each category
monthly_avg = daily_sales.groupby(['category', 'month'])['revenue'].mean().reset_index()

# Visualize seasonal trends
plt.figure(figsize=(12, 6))
for category in monthly_avg['category'].unique():
    category_data = monthly_avg[monthly_avg['category'] == category]
    plt.plot(category_data['month'], category_data['revenue'], label=category)

plt.xlabel('Month')
plt.ylabel('Average Revenue')
plt.title('Seasonal Trends in Sales by Category')
plt.legend()
plt.show()

# Quantify the impact of seasonal fluctuations
for category in monthly_avg['category'].unique():
    category_data = monthly_avg[monthly_avg['category'] == category]
    max_revenue = category_data['revenue'].max()
    min_revenue = category_data['revenue'].min()
    fluctuation = (max_revenue - min_revenue) / min_revenue * 100
    print(f"{category}: Seasonal fluctuation impact: {fluctuation:.2f}%")

# Merge sales data with promotions on product_id
sales_with_promos = pd.merge(sales_data, promotions, on='product_id', how='left')

# Filter rows where the sale date is within the promotion period
sales_with_promos['date'] = pd.to_datetime(sales_with_promos['date'])
sales_with_promos['start_date'] = pd.to_datetime(sales_with_promos['start_date'])
sales_with_promos['end_date'] = pd.to_datetime(sales_with_promos['end_date'])

sales_with_promos['is_promo'] = sales_with_promos.apply(
    lambda row: row['start_date'] <= row['date'] <= row['end_date'] if pd.notnull(row['start_date']) else False,
    axis=1
)

# Compare sales during promotion and non-promotion periods
promo_effectiveness = sales_with_promos.groupby('is_promo').agg({
    'units_sold': 'mean',
    'revenue': 'mean'
}).reset_index()

print("Average sales during promotion vs. non-promotion periods:")
print(promo_effectiveness)

# Calculate correlation between discount percentage and sales increase
sales_with_promos['sales_increase'] = sales_with_promos.groupby('product_id')['units_sold'].pct_change()
correlation = sales_with_promos[sales_with_promos['is_promo']]['discount_percentage'].corr(sales_with_promos[sales_with_promos['is_promo']]['sales_increase'])

print(f"Correlation between discount percentage and sales increase: {correlation:.2f}")

# Merge sales data with economic indicators
sales_with_econ = pd.merge(sales_data, economic_indicators, left_on=sales_data['date'].dt.to_period('M').astype(str),
                           right_on=economic_indicators['month'].dt.to_period('M').astype(str))

# Perform regression analysis
model = ols('revenue ~ gdp_growth + unemployment_rate + consumer_confidence_index', data=sales_with_econ).fit()
print(model.summary())

# Visualize the relationship
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(data=sales_with_econ, x='gdp_growth', y='revenue', ax=axes[0])
sns.scatterplot(data=sales_with_econ, x='unemployment_rate', y='revenue', ax=axes[1])
sns.scatterplot(data=sales_with_econ, x='consumer_confidence_index', y='revenue', ax=axes[2])
plt.tight_layout()
plt.show()

# Calculate average sales and promotion frequency for each product
product_performance = sales_data.groupby('product_id').agg({
    'units_sold': 'mean',
    'revenue': 'mean'
}).reset_index()

product_promo_freq = promotions.groupby('product_id').size().reset_index(name='promo_frequency')

product_segments = pd.merge(product_performance, product_promo_freq, on='product_id')

# Use K-means clustering to segment products
kmeans = KMeans(n_clusters=4, random_state=42)
product_segments['cluster'] = kmeans.fit_predict(product_segments[['units_sold', 'revenue', 'promo_frequency']])

# Identify top-performing and underperforming products
top_products = product_segments.nlargest(10, 'revenue')
underperforming_products = product_segments.nsmallest(10, 'revenue')

print("Top-performing products:")
print(top_products)
print("\nUnderperforming products:")
print(underperforming_products)

# Merge sales data with promotions and economic indicators
sales_promo_econ = pd.merge(sales_with_promos, economic_indicators, left_on=sales_with_promos['date'].dt.to_period('M').astype(str),
                            right_on=economic_indicators['month'].dt.to_period('M').astype(str))

# Calculate promotion effectiveness for each month
monthly_promo_effectiveness = sales_promo_econ[sales_promo_econ['is_promo']].groupby(sales_promo_econ['date'].dt.month).agg({
    'units_sold': 'mean',
    'revenue': 'mean',
    'gdp_growth': 'mean',
    'unemployment_rate': 'mean',
    'consumer_confidence_index': 'mean'
}).reset_index()

# Visualize promotion effectiveness by month
plt.figure(figsize=(12, 6))
sns.barplot(x='date', y='revenue', data=monthly_promo_effectiveness)
plt.xlabel('Month')
plt.ylabel('Average Revenue During Promotions')
plt.title('Promotion Effectiveness by Month')
plt.show()

# Recommend optimal promotion periods
optimal_months = monthly_promo_effectiveness.nlargest(3, 'revenue')
print("Recommended months for promotions:")
print(optimal_months[['date', 'revenue', 'gdp_growth', 'unemployment_rate', 'consumer_confidence_index']])
