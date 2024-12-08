import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# Load the dataset
file_path = "SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv"
data = pd.read_csv(file_path)

# Calculate correlation and p-value
correlation, p_value = stats.pearsonr(data['Supply Chain Emission Factors without Margins'],
                                      data['Margins of Supply Chain Emission Factors'])

# Print the correlation coefficient and p-value
print(f"Correlation coefficient: {correlation:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Supply Chain Emission Factors without Margins', y='Margins of Supply Chain Emission Factors')
plt.title('SEF vs MEF')
plt.xlabel('Supply Chain Emission Factors without Margins')
plt.ylabel('Margins of Supply Chain Emission Factors')
plt.show()

# Define the predictor variables (X) and the target variable (y)
X = data[['Supply Chain Emission Factors without Margins', 'Margins of Supply Chain Emission Factors']]
y = data['Supply Chain Emission Factors with Margins']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant to the predictor variables (required for statsmodels)
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the multiple linear regression model on the training data
model = sm.OLS(y_train, X_train_const).fit()

# Predict on the test data
y_pred = model.predict(X_test_const)

# Calculate R-squared and Adjusted R-squared values on the test data
r2 = r2_score(y_test, y_pred)
adjusted_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

# Print the R-squared and Adjusted R-squared values
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adjusted_r2:.4f}")

# Print the model summary to interpret the coefficients
print(model.summary())

# Identify and analyze outliers in the "Supply Chain Emission Factors with Margins" column using IQR method
def identify_outliers_iqr(data, column):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    # Calculate IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    return outliers, lower_bound, upper_bound

# Identify outliers in the "Supply Chain Emission Factors with Margins" column
outliers, lower_bound, upper_bound = identify_outliers_iqr(data, 'Supply Chain Emission Factors with Margins')

# Print the results
print(f"Lower bound for outliers: {lower_bound:.4f}")
print(f"Upper bound for outliers: {upper_bound:.4f}")
print(f"Number of outliers: {len(outliers)}")
print("Outliers:")
print(outliers)

def categorize_naics(title):
    if 'Farming' in title or 'Agriculture' in title:
        return 'Farming'
    elif 'Manufacturing' in title:
        return 'Manufacturing'
    else:
        return 'Other'

data['Category'] = data['2017 NAICS Title'].apply(categorize_naics)

farming = data[data['Category'] == 'Farming']['Supply Chain Emission Factors with Margins']
manufacturing = data[data['Category'] == 'Manufacturing']['Supply Chain Emission Factors with Margins']
other = data[data['Category'] == 'Other']['Supply Chain Emission Factors with Margins']

f_statistic, p_value = stats.f_oneway(farming, manufacturing, other)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Post-hoc test (Tukey's HSD) if ANOVA shows significant differences
if p_value < 0.05:
    tukey_results = pairwise_tukeyhsd(data['Supply Chain Emission Factors with Margins'], data['Category'])
    print(tukey_results)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose optimal k and perform clustering
optimal_k = 3  # Example value, adjust based on elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster characteristics
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data[['Supply Chain Emission Factors without Margins',
                        'Margins of Supply Chain Emission Factors']].describe())
    print("\n")

# Define the predictor variables (X) and the target variable (y)
X = data[['Supply Chain Emission Factors without Margins', 'Margins of Supply Chain Emission Factors']]
y = data['Supply Chain Emission Factors with Margins'] + data['Margins of Supply Chain Emission Factors']

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict using the polynomial regression model
y_pred = model.predict(X_poly)

# Calculate R-squared value
r2 = r2_score(y, y_pred)

# Print the R-squared value and model coefficients
print(f"R-squared: {r2:.4f}")
print("Model Coefficients:")
print(model.coef_)
print("Intercept:")
print(model.intercept_)

# Select the features for PCA
features = data[['Supply Chain Emission Factors without Margins', 'Margins of Supply Chain Emission Factors']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA to retain at least 95% of the variance
pca = PCA(n_components=0.95)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])

# Print the explained variance ratio and the principal components
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print("\nPrincipal Components:")
print(principal_df.head())

# Add the principal components to the original dataset for further analysis
data_pca = pd.concat([data, principal_df], axis=1)
print("\nDataset with Principal Components:")
print(data_pca.head())
