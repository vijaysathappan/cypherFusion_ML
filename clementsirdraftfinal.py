import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class CypherFusion:
    def __init__(self, num_trees=100, max_depth=3):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def _fit_tree(self, X, y):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_bootstrapped = X[indices]
        y_bootstrapped = y[indices]
        tree = DecisionTree(max_depth=self.max_depth)
        tree.fit(X_bootstrapped, y_bootstrapped)
        return tree

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = self._fit_tree(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for tree in self.trees:
            predictions += tree.predict(X)
        return predictions / self.num_trees

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

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

def main():
    file_path ="D:\\Downloads\\recruitment_data.csv"
    df = load_csv(file_path)

    # Step 1: Data Cleaning
    df_cleaned = clean_data(df)

    X = df_cleaned.iloc[:, :-1]
    y = df_cleaned.iloc[:, -1]

    # Step 2: Hash Encoding (if needed)
    X_encoded = hash_encode(X)

    # Step 3: Splitting the Data
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)

    # Step 4: Training the Model
    model = CypherFusion(num_trees=100, max_depth=3)
    model.fit(X_train.values, y_train.values)

    # Step 5: Evaluating the Model
    print("Evaluation for Cypher Fusion:")
    evaluate_model(model, X_test.values, y_test.values)

if __name__ == "__main__":
    main()
