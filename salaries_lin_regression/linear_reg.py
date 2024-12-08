import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("salary_data.csv")

# Step 1: Age column
df = df.dropna(subset=['Job Title'])
# Calculate the median age for each job title
median_ages = df.groupby('Job Title')['Age'].median()

# Define a function to fill missing ages with the median age for the same job title
def fill_missing_ages(row):
    if pd.isna(row['Age']):
        return median_ages[row['Job Title']]
    else:
        return row['Age']

# Apply the function to fill missing ages
df['Age'] = df.apply(fill_missing_ages, axis=1)

# Step 2: Gender column
df = df.dropna(subset=['Education Level'])
# Calculate the most frequent gender for each education level
most_frequent_gender = df.groupby('Education Level')['Gender'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# Define a function to fill missing genders with the most frequent gender for the same education level
def fill_missing_genders(row):
    if pd.isna(row['Gender']):
        return most_frequent_gender[row['Education Level']]
    else:
        return row['Gender']

# Apply the function to fill missing genders
df['Gender'] = df.apply(fill_missing_genders, axis=1)

# Step 3: Education Level column
# Prepare data for KNN imputation
le_gender = LabelEncoder()
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])

le_edu = LabelEncoder()
df['Education_Level_encoded'] = le_edu.fit_transform(df['Education Level'].astype(str))

knn_features = ['Age', 'Gender_encoded', 'Years of Experience']
knn_imputer = KNNImputer(n_neighbors=5)
imputed_data = knn_imputer.fit_transform(df[knn_features + ['Education_Level_encoded']])
# Ensure no NaN values are left in the imputed data
imputed_data = np.nan_to_num(imputed_data, nan=-1) # Replacing NaN with a placeholder
df[knn_features + ['Education_Level_encoded']] = imputed_data

# Decode Gender back to original values
df['Gender'] = le_gender.inverse_transform(df['Gender_encoded'].astype(int))
df['Education Level'] = df['Education_Level_encoded'].apply(lambda x: le_edu.inverse_transform([int(x)])[0] if x != -1 else np.nan)
df = df.drop(['Gender_encoded', 'Education_Level_encoded'], axis=1)

# Step 4: Job Title column
def fill_job_title(row):
    if pd.isnull(row['Job Title']):
        if row['Years of Experience'] < 2:
            return "Entry Level"
        elif row['Years of Experience'] >= 5:
            return df[df['Years of Experience'] >= 5]['Job Title'].mode().iloc[0]
    return row['Job Title']

df['Job Title'] = df.apply(fill_job_title, axis=1)

# Step 5: Years of Experience column
yoe_mean = df['Years of Experience'].mean()
yoe_std = df['Years of Experience'].std()

def generate_positive_yoe():
    yoe = np.random.normal(yoe_mean, yoe_std)
    while yoe <= 0:  # Keep generating until a positive value is obtained
        yoe = np.random.normal(yoe_mean, yoe_std)
    return int(yoe)  # Convert to integer

df['Years of Experience'] = df['Years of Experience'].fillna(
    df['Years of Experience'].apply(lambda x: generate_positive_yoe() if pd.isnull(x) else x)
)

# Step 6: Salary column
# Prepare data for linear regression
df.dropna(subset=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'], inplace=True)
X = pd.get_dummies(df.drop('Salary', axis=1), columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
y = df['Salary']

# Fit linear regression model
model = LinearRegression()
model.fit(X[y.notnull()], y[y.notnull()])

# Predict missing salaries
missing_salary = y.isnull()
df.loc[missing_salary, 'Salary'] = model.predict(X[missing_salary])

# Save the cleaned dataset
df.to_csv("cleaned_salary_data.csv", index=False)

print("Data cleaning and filtering completed. Cleaned data saved to 'cleaned_salary_data.csv'.")
