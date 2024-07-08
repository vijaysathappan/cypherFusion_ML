import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Specify the file location
file_location = "D:\\Downloads\\Clement SIr Proj\\DDoSsampledata.csv"

# Try reading the data with different encodings
try:
    df = pd.read_csv(file_location, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_location, encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv(file_location, encoding='iso-8859-1')

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns
print("\nNon-numeric columns:")
print(non_numeric_columns)

# Convert numeric columns to numeric, replacing non-numeric values with NaN
df_numeric = df.drop(columns=non_numeric_columns).apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the median of the column
for col in df_numeric.columns:
    median = df_numeric[col].median()
    df_numeric[col].fillna(median, inplace=True)

# Display the cleaned DataFrame with numeric data only
print("\nCleaned Numeric DataFrame:")
print(df_numeric)

# Perform polynomial feature extraction on numeric data
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_numeric)

# Create a DataFrame with the polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df_numeric.columns))

# Display the DataFrame with polynomial features
print("\nDataFrame with Polynomial Features:")
print(poly_df)

# If needed, merge back the non-numeric columns with the polynomial features
df_final = pd.concat([df[non_numeric_columns].reset_index(drop=True), poly_df], axis=1)

# Display the final DataFrame
print("\nFinal DataFrame with Non-Numeric Columns and Polynomial Features:")
print(df_final)
