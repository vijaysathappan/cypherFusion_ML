import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

class CypherFusion:
    def __init__(self, num_trees=100, max_depth=3):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def _fit_tree(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrapped = X[indices]
        y_bootstrapped = y[indices]
        tree = DecisionTreeRegressor(max_depth=self.max_depth)
        tree.fit(X_bootstrapped, y_bootstrapped)
        return tree

    def _fit_xgboost(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth,
            'eval_metric': 'rmse'
        }
        bst = xgb.train(params, dtrain, num_boost_round=self.num_trees)
        return bst

    def fit(self, X, y, use_xgboost=False):
        self.trees = []
        if use_xgboost:
            self.trees = self._fit_xgboost(X, y)
        else:
            for _ in range(self.num_trees):
                tree = self._fit_tree(X, y)
                self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        if isinstance(self.trees, list):  # If using Decision Trees
            for tree in self.trees:
                predictions += tree.predict(X)
            return predictions / self.num_trees
        else:  # If using XGBoost
            dtest = xgb.DMatrix(X)
            return self.trees.predict(dtest)

def load_csv(file_path):
    return pd.read_csv(file_path, encoding='latin-1')  # Try latin-1 encoding

def encode_categorical(df):
    return pd.get_dummies(df)

def clean_data(df):
    # Fill missing values with the median
    df_filled = df.fillna(df.median())
    return df_filled

def hash_encode(df):
    # Use hash encoding for categorical features
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: hash(x) % 10**5)  # Hash and limit size to avoid large numbers
    return df

def convert_string_to_numeric(df):
    # Convert string columns to numeric using hash encoding
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: hash(x) % 10**5)  # Same hash encoding as in hash_encode function
    return df

def convert_target_to_numeric(y):
    # Convert categorical target to numeric using LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

def main():
    file_path = "D:\\Downloads\\Clement SIr Proj\\DDoSdata(1).csv"
    df = load_csv(file_path)

    # Step 1: Data Cleaning
    df_cleaned = clean_data(df)

    X = df_cleaned.iloc[:, :-1]
    y = df_cleaned.iloc[:, -1]

    # Step 2: Convert String Columns to Numeric
    X_numeric = convert_string_to_numeric(X)

    # Step 2.1: Apply Polynomial Feature Extraction
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_numeric)
    print(X_poly)

    # Step 3: Convert Target Variable to Numeric
    y_numeric = convert_target_to_numeric(y)

    # Step 4: Splitting the Data
    X_train, X_test, y_train, y_test = split_data(X_poly, y_numeric)

    # Step 5: Training and Evaluating the Model with CypherFusion (Decision Trees)
    print("Evaluation for Cypher Fusion (Decision Trees):")
    model = CypherFusion(num_trees=100, max_depth=3)
    model.fit(X_train, y_train, use_xgboost=False)  # Note: y_train is now numeric
    evaluate_model(model, X_test, y_test)

    # Step 6: Training and Evaluating the Model with CypherFusion (XGBoost)
    print("\nEvaluation for Cypher Fusion (XGBoost):")
    model_xgboost = CypherFusion(num_trees=100, max_depth=3)
    model_xgboost.fit(X_train, y_train, use_xgboost=True)  # Note: y_train is now numeric
    evaluate_model(model_xgboost, X_test, y_test)

if __name__ == "__main__":
    main()
