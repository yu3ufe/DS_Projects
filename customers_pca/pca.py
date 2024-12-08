import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 1. Data Loading and Preprocessing
df = pd.read_csv("customers.csv")

# Function to clean numeric columns
def clean_numeric_column(column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column] = df[column].fillna(df[column].median())
    df.loc[df[column] < 0, column] = df[column].median()
    df[column] = df[column].astype(int)

# Clean CustomerID and Age columns
clean_numeric_column('CustomerID')
clean_numeric_column('Age')

# Correct invalid entries for Gender and Location
valid_genders = ['Male', 'Female']
valid_genders_unique = df['Gender'][df['Gender'].isin(valid_genders)].unique()
df['Gender'] = df['Gender'].apply(lambda x: np.random.choice(valid_genders_unique) if x not in valid_genders else x)

valid_locations = df['Location'][(df['Location'] != "123") & df['Location'].notna()].unique()
df['Location'] = df['Location'].apply(lambda x: np.random.choice(valid_locations) if x not in valid_locations else x)

# Clean other numeric columns
for column in ['TotalSpending', 'NumPurchases', 'WebsiteVisits', 'CustomerServiceCalls']:
    clean_numeric_column(column)

# 2. Encode Categorical Variables
df['Gender'] = df['Gender'].astype(str)
df['Location'] = df['Location'].astype(str)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(df[['Gender', 'Location']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Gender', 'Location']))
df = pd.concat([df, encoded_df], axis=1)
df.drop(['Gender', 'Location'], axis=1, inplace=True)

# 3. Standardize numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'TotalSpending', 'NumPurchases', 'WebsiteVisits', 'CustomerServiceCalls']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 4. Apply PCA to reduce dimensionality
pca = PCA()
principal_components = pca.fit_transform(df)

# Determine the optimal number of principal components
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Plot the cumulative explained variance ratio to determine the optimal number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Principal Components')
plt.grid(True)
plt.show()

# Print the cumulative explained variance ratio for each number of components
for i, ratio in enumerate(cumulative_variance_ratio):
    print(f"Number of Components: {i+1}, Cumulative Explained Variance Ratio: {ratio:.4f}")

# Determine the optimal number of components (e.g., where cumulative variance ratio exceeds 95%)
optimal_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nThe optimal number of components where cumulative variance ratio exceeds 95% is: {optimal_components}\n")

# Apply PCA with the optimal number of components (assuming 2 for visualization)
pca_optimal = PCA(n_components=2)
principal_components = pca_optimal.fit_transform(df)
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Visualize the reduced data in a 2D scatter plot, color-coding points based on "TotalSpending"
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=df['TotalSpending'], cmap='viridis')
plt.xlabel(f"Principal Component 1 ({pca_optimal.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca_optimal.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA Visualization of Customer Segmentation")
plt.colorbar(scatter, label="Total Spending")

# Annotate the plot with the explained variance ratio for each principal component
plt.annotate(f"Explained Variance: {sum(pca_optimal.explained_variance_ratio_[:2])*100:.1f}%",
             xy=(0.05, 0.95), xycoords='axes fraction')

plt.show()
