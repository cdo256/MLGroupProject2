import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict

import warnings
warnings.filterwarnings("ignore")

def t_test_feature_selection(X, y, threshold=0.05):
    """
    Perform t-test for feature selection (binary classification).

    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target vector (n_samples)
    - threshold: p-value threshold for feature selection

    Returns:
    - selected_features: List of feature names of selected features
    """
    selected_features = []

    # Get unique classes (for binary classification)
    classes = np.unique(y)

    for i in range(X.shape[1]):
        feature = X.iloc[:, i]

        # Split data into two groups based on class
        group1 = feature[y == classes[0]]
        group2 = feature[y == classes[1]]

        # Perform t-test
        _, p_value = ttest_ind(group1, group2)

        # If the p-value is less than the threshold, select the feature
        if p_value < threshold:
            selected_features.append(X.columns[i])  # Use the column name
            print(f"Feature: {X.columns[i]} which has a p-value: {p_value} for t-test")

    return selected_features

def lasso_feature_selection(X, y, alpha=0.1):
    """
    Perform feature selection using Lasso regularization.

    Parameters:
    - X: Features DataFrame (n_samples x n_features)
    - y: Target variable
    - alpha: Regularization strength (default=0.1)

    Returns:
    - selected_features: List of feature names of selected features
    """
    # Standardize the data (important for Lasso to work well)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Lasso model
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_scaled, y)

    # Get the coefficients of the model (those close to zero are eliminated)
    selected_features = X.columns[lasso.coef_ != 0].tolist()

    print(f"Selected Features (names): {selected_features}")
    print(f"Lasso Coefficients: {lasso.coef_}")

    return selected_features

def rfe_wrapper(X, y, top_n_features, estimator=None, step=1):
    """
    Perform feature selection using a wrapper method (RFE).

    Parameters:
    - X: Features DataFrame or numpy array (n_samples x n_features)
    - y: Target variable
    - top_n_features: Number of features to select
    - estimator: Estimator model (e.g., RandomForestClassifier, LogisticRegression)
    - step: The number of features to remove at each iteration (default=1)

    Returns:
    - Selected features (list of feature names)
    """
    # Default estimator if not provided
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    # Initialize the RFE model
    selector = RFE(estimator, n_features_to_select=top_n_features, step=step)

    # Fit the RFE selector to the data
    selector.fit(X, y)

    # Get the mask of the best selected features
    selected_features = X.columns[selector.support_].tolist()

    print(f"Selected Features: {selected_features}")

    return selected_features

def anova_feature_selection(X, y, threshold=0.05):
    """
    Perform ANOVA (f_classif) for feature selection.

    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target vector (n_samples)
    - threshold: p-value threshold for feature selection

    Returns:
    - selected_features: List of feature names of selected features
    """
    # Perform ANOVA F-test (f_classif)
    f_values, p_values = f_classif(X, y)

    print("Feature Scores (F-values):")
    for i, f_val in enumerate(f_values):
        print(f"Feature {X.columns[i]}: F-value = {f_val:.4f}, p-value = {p_values[i]:.4f}")

    # Select features with p-value less than threshold
    selected_features = [X.columns[i] for i in range(len(p_values)) if p_values[i] < threshold]

    print("\nSelected Features (based on p-value < threshold):")
    print(selected_features)

    return selected_features

def chi_square_feature_selection(X, y, threshold=0.05):
    """
    Perform feature selection using the Chi-Square test.
    :param X: Feature matrix (n_samples x n_features)
    :param y: Target vector (n_samples)
    :param threshold: p-value threshold for feature selection
    :return: List of feature names of selected features
    """
    # Scale the data to non-negative values using Min-Max scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform Chi-Square test for feature selection
    chi2_selector = SelectKBest(score_func=chi2, k='all')
    chi2_selector.fit(X_scaled, y)
    p_values = chi2_selector.pvalues_

    # Select features with p-value less than the threshold
    selected_features = [X.columns[i] for i in range(len(p_values)) if p_values[i] < threshold]

    for i in range(len(p_values)):
        if p_values[i] < threshold:
            print(f"Selected feature: {X.columns[i]}, p-value: {p_values[i]:.4f}")

    return selected_features

def compute_entropy(y):
    """
    Compute entropy for classification labels.
    :param y: Target variable
    :return: entropy value
    """
    # Compute the value of each class probability
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    # Compute entropy: H(y) = -sum(p(x) * log2(p(x)))
    entropy = -np.sum(probs * np.log2(probs))

    return entropy

def forward_selection(X, y, estimators=None, scoring=None, cv=5):
    """
    Forward Selection Feature Selection method that selects the best model based on scoring.

    Parameters:
    - X: Features DataFrame (n_samples x n_features)
    - y: Target variable
    - estimators: List of Estimator models (default: LogisticRegression, SVC, etc.)
    - scoring: Scoring function (either 'entropy' or a custom function)
    - cv: Number of cross-validation folds (default: 5)

    Returns:
    - selected_features: List of selected feature indices
    """
    # If no estimators are provided, use default models
    if estimators is None:
        estimators = [
            LogisticRegression(solver='liblinear'),
            RandomForestClassifier(),
            DecisionTreeClassifier()
        ]

    # Set scoring function based on the user's choice
    if scoring == 'entropy':
        scoring_function = compute_entropy
    elif scoring is None:
        scoring_function = accuracy_score  # Default to accuracy if no scoring function is passed
    else:
        scoring_function = scoring  # Use the provided custom scoring function

    remaining_features = list(range(X.shape[1]))
    best_score = -float('inf')  # Start with a very low score
    last_best_score = -float('inf')  # Track the last best score to stop the process
    best_model = None
    selected_features = []

    # Dictionary to store performance of each model
    model_scores = defaultdict(list)

    # Stepwise addition of features across all models
    while remaining_features:
        scores_with_candidates = []

        # Try adding each feature to the selected set and calculate score for each model
        for feature in remaining_features:
            candidate_features = selected_features + [feature]
            X_train_selected = X.iloc[:, candidate_features]
            # Evaluate each model
            for estimator in estimators:
                score = cross_val_score(estimator, X_train_selected, y, cv=cv, scoring='accuracy').mean()
                # If using entropy, we need to invert the score (higher is better for entropy)
                if scoring_function == compute_entropy:
                    score = -score  # Invert the accuracy score (higher is better for entropy)
                scores_with_candidates.append((score, feature, estimator))

        # Sort and pick the best feature for the best model
        scores_with_candidates.sort(reverse=True, key=lambda x: x[0])  # Sort by score
        best_score, best_feature, best_model_for_iteration = scores_with_candidates[0]

        # If the best score improves, keep the feature
        if best_score > last_best_score:  # Only keep the feature if it improves performance
            print(f"best score: {best_score}")
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            last_best_score = best_score  # Update last_best_score to the new best
            best_model = best_model_for_iteration  # Update best_model to the current best model

            # Print the name of the feature being added
            if isinstance(X, pd.DataFrame):  # Check if X is a DataFrame
                feature_name = X.columns[best_feature]
            else:  # If X is a numpy array, use the index as feature name
                feature_name = f"Feature_{best_feature}"

            print(f"Added feature: {feature_name} | Score: {best_score:.4f} | Best Model: {best_model_for_iteration}")
        else:
            break  # Stop if no improvement is made

    # Print the results
    print("Selected Features:", selected_features)
    print("Best Model:", best_model)

    return selected_features


def evaluate_model(X_train, X_test, y_train, y_test, selected_features):
    # If selected_features contains names (strings), convert them to numeric indices
    if isinstance(selected_features[0], str):  # Check if the selected features are names
        selected_features = [X_train.columns.get_loc(col_name) for col_name in selected_features]

    # Now select the features from the original dataset using numeric indices
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    # Train a logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_selected, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_selected)

    # Evaluate the model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main(dataframe_name, top_n_features, predicted_column, outputdataframe_name):
    # Load the dataset
    df = pd.read_excel(dataframe_name)

    # Convert DataFrame Columns from String to Integer
    for column in df.columns:
        try:
            # Attempt to convert the column to numeric (integer)
            df[column] = pd.to_numeric(df[column], errors='coerce')  # Invalid values will be NaN
            df[column] = df[column].fillna(0).astype(int)  # Fill NaNs with 0 and convert to int
        except Exception as e:
            print(f"Skipping column {column}: {e}")

    # Predicted column
    y = df[predicted_column]
    df.drop([predicted_column], axis=1, inplace=True)
    X = df.dropna()

    # Print and compare results
    print("Comparison of Feature Selection Methods:\n")
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Compare the feature selection methods
    methods = ['t-test', 'ANOVA', 'Chi-Square', 'rfe', 'lasso', 'forward']

    selected_features_dict = {}
    best_method = None
    best_accuracy = -float('inf')  # Start with a very low accuracy

    # Dictionary to store feature selection and accuracy from all methods
    method_accuracies = {}

    # Perform feature selection using each method and store the accuracies and selected features
    for method in methods:
        if method == 't-test':
            selected_features_dict[method] = t_test_feature_selection(X_train, y_train)
        elif method == 'ANOVA':
            selected_features_dict[method] = anova_feature_selection(X_train, y_train)
        elif method == 'Chi-Square':
            selected_features_dict[method] = chi_square_feature_selection(X_train, y_train)
        elif method == 'rfe':
            selected_features_dict[method] = rfe_wrapper(X_train, y_train, top_n_features)
        elif method == 'lasso':
            selected_features_dict[method] = lasso_feature_selection(X_train, y_train)
        elif method == 'forward':
            selected_features_dict[method] = forward_selection(X_train, y_train, scoring="entropy")

        # Evaluate model performance with the selected features
        selected_features = selected_features_dict[method]
        accuracy = evaluate_model(X_train, X_test, y_train, y_test, selected_features)

        # Store the method's accuracy
        method_accuracies[method] = accuracy

        # Print results
        print(f"{method}:")
        print(f"  Selected features: {selected_features}")
        print(f"  Number of selected features: {len(selected_features)}")
        print(f"  Model accuracy: {accuracy:.4f}\n")

        # Update best performing method if necessary
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = method

    # After evaluating all methods, find the top-performing features across all methods
    all_selected_features = []

    for method, features in selected_features_dict.items():
        all_selected_features.extend(features)

    # Count how many times each feature was selected across all methods
    feature_counts = Counter(all_selected_features)

    # Sort features by the number of times they were selected
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Select the top N features based on their selection count across all methods
    top_selected_features = [feature for feature, count in sorted_features[:top_n_features]]

    # Output the results
    print(f"Best performing feature selection method: {best_method}")
    print(f"  With an accuracy of: {best_accuracy:.4f}")
    print(f"  Selected features: {selected_features_dict[best_method]}")

    print(f"\nTop {top_n_features} selected features from all methods:")
    print(f"  {top_selected_features}")

    # Combine the top selected features from X with y (target variable)
    top_selected_df = pd.concat([X[top_selected_features], y], axis=1)

    # Output the new DataFrame with top selected features
    print(f"\nNew DataFrame with top {top_n_features} selected features:")
    print(top_selected_df.head())  # Show first few rows of the new DataFrame

    # Optionally, return the top_selected_df DataFrame and create csv
    top_selected_df.to_csv(outputdataframe_name)

    return top_selected_df

if __name__ == '__main__':
    predicted_column_name = 'pCR (outcome)'
    dataframe_name = "preprocessed.xlsx"
    outputdataframe_name = "TrainDataset2024.csv"
    top_n_features = 10
    main(dataframe_name, top_n_features, predicted_column_name,outputdataframe_name)