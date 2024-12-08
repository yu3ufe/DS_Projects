import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
# Assuming the CSV file is named 'movie_ratings.csv'
df = pd.read_csv('movie_ratings.csv')

# Step 2: Data Cleaning
# Remove users with fewer than 5 ratings
user_counts = df['user_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

# Handle missing values - for simplicity, we'll drop rows with any NaN values
df.dropna(inplace=True)

# Convert timestamp to datetime for time-based analysis
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Step 3: Outlier Detection with Z-scores
ratings = df['rating']
# Calculate mean and standard deviation
mean_rating = np.mean(ratings)
std_rating = np.std(ratings)
# Calculate z-scores
z_scores = (ratings - mean_rating) / std_rating
# Add z-scores to the DataFrame if needed
df['z_score'] = z_scores
# Filter out outliers (e.g., keeping ratings within 3 standard deviations)
df_cleaned = df[np.abs(z_scores) < 3]

# Step 4: Plot Distribution of Ratings
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['rating'], bins=20, edgecolor='black')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Step 5: Top 10 Movies by Average Rating
top_movies = df_cleaned.groupby('movie_id').agg({'rating': 'mean'}).sort_values(by='rating', ascending=False).head(10)
print("Top 10 Movies by Average Rating:\n", top_movies)

# Step 6: Consistent Raters
user_std = df_cleaned.groupby('user_id')['rating'].std()
consistent_users = user_std[user_std < 1].index
consistent_raters_avg = df_cleaned[df_cleaned['user_id'].isin(consistent_users)]['rating'].mean()
print(f"Average rating by consistent raters: {consistent_raters_avg:.2f}")

# Step 7: Average Rating per Year
df_cleaned['year'] = df_cleaned['timestamp'].dt.year
yearly_avg = df_cleaned.groupby('year')['rating'].mean()
plt.figure(figsize=(10, 6))
yearly_avg.plot(kind='line')
plt.title('Average Movie Rating Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

# Step 8: Most Rated Movies
most_rated = df_cleaned['movie_id'].value_counts().head(10)
print("10 Most Rated Movies:\n", most_rated)

# Step 9: Number of Movies Rated by Users
user_movie_count = df_cleaned.groupby('user_id')['movie_id'].nunique()
user_movie_count = user_movie_count[user_movie_count > 20]
variance = user_movie_count.var()
print(f"Variance of movies rated by users who rated more than 20 movies: {variance:.2f}")

# Step 10: Time of Day Analysis
df_cleaned['time_of_day'] = df_cleaned['timestamp'].dt.hour.apply(lambda x: 'morning' if 5 <= x < 12 else
                                                                 'afternoon' if 12 <= x < 18 else
                                                                 'evening' if 18 <= x < 22 else 'night')
time_avg = df_cleaned.groupby('time_of_day')['rating'].mean()
print("Average ratings by time of day:\n", time_avg)

# Step 11: Average Rating by Genre
genre_avg = df_cleaned.groupby('genre')['rating'].mean().sort_values(ascending=False)
print("Top 3 genres by average rating:\n", genre_avg.head(3))

# Step 12: Pairwise Comparison for Top 5 Most Rated Movies
top_5_movies = df_cleaned['movie_id'].value_counts().index[:5]
for i in range(len(top_5_movies)):
    for j in range(i+1, len(top_5_movies)):
        movie1 = df_cleaned[df_cleaned['movie_id'] == top_5_movies[i]]['rating']
        movie2 = df_cleaned[df_cleaned['movie_id'] == top_5_movies[j]]['rating']

        def calculate_t_statistic_and_p_value(mv1, mv2):

            n1 = len(mv1)
            n2 = len(mv2)

            # Calculate sample means and standard deviations
            mean1 = np.mean(mv1)
            mean2 = np.mean(mv2)
            std1 = np.std(mv1, ddof=1)
            std2 = np.std(mv2, ddof=1)

            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

            # Calculate t-statistic
            t_statistic = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))

            # Approximate p-value using t-distribution (assuming normal distribution)
            df = n1 + n2 - 2

            p_value = 2 * (1 - (1 / (1 + (t_statistic**2 / df)))**((df + 1) / 2))

            return t_statistic, p_value
        t_stat, p_value = calculate_t_statistic_and_p_value(movie1, movie2)
        if p_value < 0.05:
            print(f"There is a significant difference between {top_5_movies[i]} and {top_5_movies[j]}")
