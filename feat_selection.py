import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso
from scipy.stats import ttest_ind
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from collections import Counter, defaultdict
from preprocess import Preprocessor

from BaseClasses import modelType
from enum import Enum

import warnings
warnings.filterwarnings("ignore")

class fsMethod(Enum):
    ANOVA = 0,
    RFE = 1,
    FORWARD = 2,
    LASSO = 3,
    T_TEST = 4,
    CHISQUARE = 5

verbose = False

def debug_print(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)

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
        group1 = feature.loc[(y == classes[0]).index[((y == classes[0])[0])==True].tolist()]
        group2 = feature.loc[(y == classes[0]).index[((y == classes[1])[0])==True].tolist()]

        # Perform t-test
        _, p_value = ttest_ind(group1, group2)

        # If the p-value is less than the threshold, select the feature
        if p_value < threshold:
            selected_features.append(X.columns[i])  # Use the column name
            debug_print(f"Feature: {X.columns[i]} which has a p-value: {p_value} for t-test")

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

    debug_print(f"Selected Features (names): {selected_features}")
    debug_print(f"Lasso Coefficients: {lasso.coef_}")

    return selected_features

def rfe_wrapper(X, y, top_n_features, task, estimator=None, step=1):
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
        match task:
            case modelType.CLASSIFICATION:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            case modelType.REGRESSION:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
            case _:
                raise ValueError('Invalid task')

    # Initialize the RFE model
    selector = RFE(estimator, n_features_to_select=top_n_features, step=step)

    # Fit the RFE selector to the data
    selector.fit(X, y)

    # Get the mask of the best selected features
    selected_features = X.columns[selector.support_].tolist()

    debug_print(f"Selected Features: {selected_features}")

    return selected_features

def anova_feature_selection(X, y,task,top_n_features, threshold=0.05):
    """
    Perform ANOVA (f_classif) for feature selection.

    Parameters:
    - X: Feature matrix (n_samples x n_features)
    - y: Target vector (n_samples)
    - threshold: p-value threshold for feature selection
    - top_n_features - select the top

    Returns:
    - selected_features: List of feature names of selected features
    """

    debug_print(f"task:{task}")


    match task:
        case modelType.CLASSIFICATION:
            # Perform ANOVA F-test (f_classif)
            f_values, p_values = f_classif(X, y)

            debug_print("Feature Scores (F-values):")
            for i, f_val in enumerate(f_values):
                debug_print(f"Feature {X.columns[i]}: F-value = {f_val:.4f}, p-value = {p_values[i]:.4f}")

            # Select features with p-value less than threshold
            selected_features = [X.columns[i] for i in range(len(p_values)) if p_values[i] < threshold]

            #debug_print("\nSelected Features (based on p-value < threshold):")
            #debug_print(selected_features)

            selector = SelectKBest(f_classif, k = top_n_features)
            selector.fit_transform(X,y)
            selected_features = selector.get_feature_names_out()
        case modelType.REGRESSION:
            selector = SelectKBest(f_regression, k = top_n_features)
            selector.fit_transform(X,y)
            selected_features = selector.get_feature_names_out()
        case _:
            raise ValueError('Invalid task')


    return selected_features

def chi_square_feature_selection(X, y,top_n_features, threshold=0.05):
    """
    Perform feature selection using the Chi-Square test.
    :param X: Feature matrix (n_samples x n_features)
    :param y: Target vector (n_samples)
    :param threshold: p-value threshold for feature selection
    :return: List of feature names of selected features
    """

    # Scale the data to non-negative values using Min-Max scaling
    scaler = MinMaxScaler().set_output(transform = "pandas")
    X_scaled = scaler.fit_transform(X)

    # Perform Chi-Square test for feature selection
    chi2_selector = SelectKBest(chi2, k=top_n_features)
    chi2_selector.fit_transform(X_scaled, y)
    
    #Change back to this if wanting to use threshold based selection (instead of top X)
    #p_values = chi2_selector.pvalues_

    # Select features with p-value less than the threshold
    # selected_features = [X.columns[i] for i in range(len(p_values)) if p_values[i] < threshold]

    # for i in range(len(p_values)):
    #     if p_values[i] < threshold:
    #         debug_print(f"Selected feature: {X.columns[i]}, p-value: {p_values[i]:.4f}")

    return chi2_selector.get_feature_names_out()

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

def forward_selection(X, y, task, estimators=None, scoring=None, cv=5):
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
        match task:
            case modelType.CLASSIFICATION:
                estimators = [
                    LogisticRegression(solver='liblinear'),
                    RandomForestClassifier(),
                    DecisionTreeClassifier()
                ]
            case modelType.REGRESSION:
                estimators = [
                    LinearRegression(),
                    RandomForestRegressor(),
                    DecisionTreeRegressor()
                ]
            case _:
                raise ValueError('Invalid task')

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
            debug_print(f"best score: {best_score}")
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            last_best_score = best_score  # Update last_best_score to the new best
            best_model = best_model_for_iteration  # Update best_model to the current best model

            # debug_print the name of the feature being added
            if isinstance(X, pd.DataFrame):  # Check if X is a DataFrame
                feature_name = X.columns[best_feature]
            else:  # If X is a numpy array, use the index as feature name
                feature_name = f"Feature_{best_feature}"

            debug_print(f"Added feature: {feature_name} | Score: {best_score:.4f} | Best Model: {best_model_for_iteration}")
        else:
            break  # Stop if no improvement is made

    # Print the results
    debug_print("Selected Features:", selected_features)
    debug_print("Best Model:", best_model)

    return selected_features


def evaluate_model(X_train, X_test, y_train, y_test, selected_features,task):
    # If selected_features contains names (strings), convert them to numeric indices
    if(len(selected_features) == 0):
        debug_print("No features selected, skipping model evaluation")
        return 0
    if isinstance(selected_features[0], str):  # Check if the selected features are names
        selected_features = [X_train.columns.get_loc(col_name) for col_name in selected_features]

    # Now select the features from the original dataset using numeric indices
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]


    match task:
        case modelType.CLASSIFICATION:
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_selected, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test_selected)
            # Evaluate the model accuracy
            accuracy = accuracy_score(y_test, y_pred)
        case modelType.REGRESSION:
            model = LinearRegression()
            model.fit(X_train_selected, y_train)
            # Predict on the test set
            y_pred = model.predict(X_test_selected)
            # Evaluate the model accuracy
            accuracy = model.score(X_test_selected, y_test)
        case _:
            raise ValueError('Invalid task')
        

    return accuracy

def main(X, y, top_n_features, task):
    # Print and compare results

    debug_print("Comparison of Feature Selection Methods:\n")
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, = \
        train_test_split(X, y, test_size=0.3, random_state=42)

    # Compare the feature selection methods
    methods = [
        # task, method
        (task, fsMethod.ANOVA),
        (task, fsMethod.RFE),
        (task, fsMethod.FORWARD),
    ]

    match task:
        case modelType.REGRESSION:
            methods.append((task, fsMethod.LASSO))
        case modelType.CLASSIFICATION:
            methods.append((task, fsMethod.T_TEST))
            methods.append((task, fsMethod.CHISQUARE))
        case _:
            raise ValueError('Invalid task')

    # Dictionary from methods to (accuracy, features)
    results = {}

    selected_features_dict = {}
    best_method = None
    best_accuracy = -float('inf')  # Start with a very low accuracy

    # Dictionary to store feature selection and accuracy from all methods
    method_accuracies = {}

    # Perform feature selection using each method and store the accuracies and selected features
    for t, method in methods:
        # Only use methods that apply to the current task
        if t != task:
            continue

        print(f'Running task, method: {t.name, method.name}')
        match method:
            case fsMethod.T_TEST:
                selected_features = t_test_feature_selection(X_train, y_train)
            case fsMethod.ANOVA:
                selected_features = anova_feature_selection(X_train, y_train, t, top_n_features)
            case fsMethod.CHISQUARE:
                selected_features = chi_square_feature_selection(X_train, y_train,top_n_features)
            case fsMethod.RFE:
                selected_features = rfe_wrapper(X_train, y_train, top_n_features, t)
            case fsMethod.LASSO:
                selected_features = lasso_feature_selection(X_train, y_train)
            case fsMethod.FORWARD:
                selected_features = forward_selection(X_train, y_train, t, scoring="entropy")
            case _:
                raise ValueError('Invalid feature selction method')

        # Evaluate model performance with the selected feature
        accuracy = evaluate_model(X_train, X_test, y_train, y_test, selected_features, t)

        # Store the method's accuracy
        results[(t, method)] = (accuracy, selected_features)

        # Print results
        debug_print(f"{method}:")
        debug_print(f"  Selected features: {selected_features}")
        debug_print(f"  Number of selected features: {len(selected_features)}")
        debug_print(f"  Model accuracy: {accuracy:.4f}\n")

        # Update best performing method if necessary
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_method = (t, method)

    # After evaluating all methods, find the top-performing features across all methods
    all_selected_features = []

    for method, features in results.items():
        all_selected_features.extend(features[1])

    # Count how many times each feature was selected across all methods
    feature_counts = Counter(all_selected_features)

    # Sort features by the number of times they were selected
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Select the top N features based on their selection count across all methods
    top_selected_features = [feature for feature, count in sorted_features[:top_n_features]]

    # Output the results
    print(f"Best performing feature selection method: {best_method}")
    print(f"  With an accuracy of: {best_accuracy:.4f}")

    print(f"  Selected features: {results[best_method]}")

    print(f"\nTop {top_n_features} selected features from all methods:")
    print(f"  {top_selected_features}")

    return top_selected_features