import pandas as pd
import dateutil.parser

# Step 1: Load and Preprocess Data
def load_and_preprocess_data(file_path):

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Handle the missing values
    df.dropna(subset=['customer_id'], inplace=True)

    def convert_to_datetime(date_str):
        if isinstance(date_str, str):
            try:
                # Try different date formats
                return dateutil.parser.parse(date_str)
            except ValueError:
                return None  # Mark invalid dates as None
        else:
            return None

    # Apply the conversion function to the 'purchase_date' column
    df['purchase_date'] = df['purchase_date'].apply(convert_to_datetime)

    # Drop rows with invalid dates (now None)
    df.dropna(subset=['purchase_date'], inplace=True)

    # Handle negative values
    df['purchase_amount'] = df['purchase_amount'].abs()

    # Group by 'customer_id' and 'product_category', then sum 'purchase_amount'
    df = df.groupby(['customer_id', 'purchase_date', 'product_category'])['purchase_amount'].sum().reset_index()

    return df

# Step 2: Filter and Analyze Data
def analyze_purchases(df):

    # Filter data for summer months of 2023
    summer_df = df[(df['purchase_date'].dt.month >= 6) &
                   (df['purchase_date'].dt.month <= 8) &
                   (df['purchase_date'].dt.year == 2023)]

    # Calculate total spending per customer per category in the summer
    summer_spending = summer_df.groupby(['customer_id', 'product_category'])['purchase_amount'].sum()

    # Identify customers meeting the criteria (at least 3 purchases, 2 categories)
    customer_counts = df.groupby('customer_id')['product_category'].nunique()
    valid_customers = customer_counts[customer_counts >= 2].index
    purchase_counts = df.groupby('customer_id').size()
    valid_customers = purchase_counts[purchase_counts >= 3].index.intersection(valid_customers)

    # Filter summer spending to include only valid customers
    valid_summer_spending = summer_spending[summer_spending.index.get_level_values('customer_id').isin(valid_customers)]

    # Calculate average spending per customer per category
    avg_spending = valid_summer_spending.groupby('product_category').mean()

    # Get top 5 categories
    top_5_categories = avg_spending.nlargest(5)

    return top_5_categories

# Main execution
def main():
    file_path = 'customer_data.csv'
    df = load_and_preprocess_data(file_path)

    if df.empty:
        print("No valid data after preprocessing.")
        return

    top_categories = analyze_purchases(df)

    print("Top 5 product categories by average spending per customer in summer 2023:")
    for idx, (category, avg_spend) in enumerate(top_categories.items(), 1):
        print(f"{idx}. {category.capitalize()}: ${avg_spend:.2f}")

if __name__ == "__main__":
    main()
