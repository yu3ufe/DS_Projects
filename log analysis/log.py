import pandas as pd
from urllib.parse import urlparse, parse_qs
from collections import Counter, defaultdict
import csv

# Constants
CHUNK_SIZE = 100000 # Adjust based on memory availability

# Function to extract product ID and category from URL
def extract_product_info(url):

    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    product_id = query.get('product_id', [None])[0]
    category = query.get('category', [None])[0]
    return product_id, category

# Function to clean and process each chunk
def process_chunk(chunk):
    # Clean data: remove entries with missing user_id or known bot user agents
    chunk = chunk[chunk['user_id'].notna()]
    chunk = chunk[~chunk['user_agent'].str.contains('bot|spider|crawl', case=False, na=False)]

    # Extract product info
    chunk[['product_id', 'category']] = chunk['URL'].apply(extract_product_info).tolist()

    return chunk

# Function to analyze daily data
def analyze_daily_data(df):
    daily_data = defaultdict(lambda: {'products': Counter(), 'referrers': Counter(), 'sequences': defaultdict(Counter)})

    # Sort by user_id and timestamp to ensure correct sequence tracking
    df = df.sort_values(by=['user_id', 'timestamp'])
    for i, row in df.iterrows():
        date = row['timestamp'].date()
        product_id = row['product_id']
        referrer = row['referrer']
        user_id = row['user_id']

        # Count product views
        daily_data[date]['products'][product_id] += 1

        # Count referrers
        if referrer:
            daily_data[date]['referrers'][referrer] += 1

        # Track user navigation sequences
        if i > 0 and df.iloc[i-1]['user_id'] == user_id:
            last_url = df.iloc[i-1]['URL']
            sequence = f"{last_url} -> {row['URL']}"
            daily_data[date]['sequences'][user_id][sequence] += 1

    return daily_data

# Main processing loop
def process_log_files(file_path):
    all_daily_stats = {} # Dictionary to store all daily stats
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE, sep=',', parse_dates=['timestamp']):
        cleaned_chunk = process_chunk(chunk)
        daily_stats = analyze_daily_data(cleaned_chunk)

        # Accumulate daily stats
        for date, data in daily_stats.items():
            if date not in all_daily_stats:
                all_daily_stats[date] = data
            else:
                # Combine data for the same date
                all_daily_stats[date]['products'].update(data['products'])
                all_daily_stats[date]['referrers'].update(data['referrers'])
                for user_id, sequences in data['sequences'].items():
                    all_daily_stats[date]['sequences'][user_id].update(sequences)

    # Save all accumulated data to a single file
    with open('all_daily_reports.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Date', 'Metric', 'Value']) # Modified header
        for date, data in all_daily_stats.items():
            for product_id, count in data['products'].items():
                writer.writerow([date, 'Product Views', f'{product_id}: {count}'])
            for referrer, count in data['referrers'].items():
                writer.writerow([date, 'Referrer', f'{referrer}: {count}'])
            for user_id, sequences in data['sequences'].items():
                for sequence, count in sequences.items():
                    writer.writerow([date, 'User Sequence', f'User {user_id}: {sequence} ({count})'])
            top_referrers = data['referrers'].most_common(5)  # Get top 5
            for referrer, count in top_referrers:
                writer.writerow([date, 'Top Referrer', f'{referrer}: {count}'])

# Run the process
process_log_files('web_data.csv')
