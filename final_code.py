import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Specify the file location
file_location = "D:\\Downloads\\NetflixOriginals_1.csv"
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

# Identify the target column as the last column in the DataFrame
target_column = df.columns[-1]
print(target_column)

# Check if the target column is non-numeric and apply label encoding if necessary
if df[target_column].dtype == 'object':
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

# Convert numeric columns to numeric, replacing non-numeric values with NaN
df_numeric = df.drop(columns=non_numeric_columns).apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the median of the column
for col in df_numeric.columns:
    median = df_numeric[col].median()
    df_numeric[col].fillna(median, inplace=True)

# Add the target column back to the cleaned numeric DataFrame
df_numeric[target_column] = df[target_column]

# Display the cleaned DataFrame with numeric data only
print("\nCleaned Numeric DataFrame:")
print(df_numeric)

# Perform polynomial feature extraction on numeric data
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_numeric.drop(columns=[target_column]))
print("Polynomial Features:",poly_features)

# Create a DataFrame with the polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(df_numeric.drop(columns=[target_column]).columns))

# If needed, merge back the non-numeric columns with the polynomial features
df_final = pd.concat([poly_df, df[target_column].reset_index(drop=True)], axis=1)

# Display the final DataFrame
print("\nFinal DataFrame with Polynomial Features and Target Column:")
print(df_final)

# Define the DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}

    def _build_tree(self, X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return np.mean(y)
        best_feature, best_value = self._find_best_split(X, y)
        left_indices = X[:, best_feature] < best_value
        right_indices = ~left_indices
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth - 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth - 1)
        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def _find_best_split(self, X, y):
        best_score = float('inf')
        best_feature = None
        best_value = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = X[:, feature] < value
                right_indices = ~left_indices
                left_std = np.std(y[left_indices])
                right_std = np.std(y[right_indices])
                score = left_std + right_std
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_value = value
        return best_feature, best_value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, self.max_depth)

    def predict_instance(self, instance):
        tree = self.tree
        while isinstance(tree, dict):
            if instance[tree['feature']] < tree['value']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree

    def predict(self, X):
        return np.array([self.predict_instance(instance) for instance in X])

# Example usage of the DecisionTree class
if __name__ == "__main__":
    # Example data preparation steps
    # Replace with your actual data loading, cleaning, and feature engineering steps
    X = df_final.drop(columns=[target_column]).values  # Adjust target_column as needed
    print("X COlumn:",X)
    y = df_final[target_column].values  # Adjust target_column as needed
    print("y Column",y)

    # Example of splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Example of training the DecisionTree model
    model = DecisionTree(max_depth=3)
    print("Model:",model)
    model.fit(X_train, y_train)

    # Example of predicting with the trained model
    y_pred = model.predict(X_test)
    print(y_pred)

    # Example of evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R^2) Score: {r2}")
    print("\nExample of predicted output:")
    for i in range(len(y_test)):  # Print the first 5 predictions as an example
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")
