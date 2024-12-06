import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


class PCAvsLDAComparison:
    def __init__(self, data, target, n_components):
        """
        Initialize the class with dataset and target variable
        Args:
        - data: Feature matrix (X)
        - target: Labels (y)
        - n_components: Number of components for PCA and LDA
        """
        self.data = data
        self.target = target
        self.n_components = n_components

        # # Standardize the features
        # self.scaler = StandardScaler()
        # self.data_scaled = self.scaler.fit_transform(self.data)

        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42)

        # Get the number of features in the original data
        self.n_features = self.data.shape[1]
        print(f"Original number of features: {self.n_features}")

        # Get the number of classes in the target variable for LDA
        self.n_classes = len(np.unique(self.target))
        print(f"Number of classes in target variable: {self.n_classes}")

        # Ensure that LDA's n_components is within the valid range
        self.lda_components = min(self.n_components, self.n_classes - 1)
        print(f"Adjusted number of components for LDA: {self.lda_components}")

    def apply_pca(self):
        """Apply PCA and reduce the feature dimensionality."""
        pca = PCA(n_components=self.n_components)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        return X_train_pca, X_test_pca

    def apply_lda(self):
        """Apply LDA and reduce the feature dimensionality."""
        lda = LDA(n_components=self.lda_components)  # Use the adjusted component count
        X_train_lda = lda.fit_transform(self.X_train, self.y_train)
        X_test_lda = lda.transform(self.X_test)
        return X_train_lda, X_test_lda

    def evaluate_classifiers(self, X_train, X_test, y_train, y_test):
        """
        Train multiple classifiers and evaluate the performance.
        Args:
        - X_train: Transformed feature matrix for training
        - X_test: Transformed feature matrix for testing
        - y_train: Training labels
        - y_test: Test labels
        """
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(),
            'ANN': MLPClassifier(random_state=42)
        }

        # Grid search for hyperparameter tuning of the ANN model
        ann_param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],  # Different sizes of hidden layers
            'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
            'solver': ['adam', 'sgd'],  # Optimizers
            'max_iter': [1000],  # Number of iterations for training
            'alpha': [0.0001, 0.001]  # Regularization parameter
        }

        # Perform grid search on ANN to find the best parameters
        ann_grid_search = GridSearchCV(MLPClassifier(random_state=42), ann_param_grid, cv=3)

        results = {}

        for model_name, model in classifiers.items():
            if model_name == 'ANN':
                print(f"Tuning {model_name} with GridSearchCV...")
                ann_grid_search.fit(X_train, y_train)
                best_ann_model = ann_grid_search.best_estimator_
                y_pred = best_ann_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[model_name] = accuracy
                print(f"Best ANN Parameters: {ann_grid_search.best_params_}")
                print(f"Accuracy with {model_name}: {accuracy:.4f}")
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[model_name] = accuracy
                print(f"Accuracy with {model_name}: {accuracy:.4f}")

        return results

    def compare_performance(self):
        """Compare performance of PCA, LDA, and no DR using multiple models and return results as DataFrame."""
        # Apply PCA and LDA
        X_train_pca, X_test_pca = self.apply_pca()
        X_train_lda, X_test_lda = self.apply_lda()

        # Evaluate classifiers on PCA and LDA
        pca_results = self.evaluate_classifiers(X_train_pca, X_test_pca, self.y_train, self.y_test)
        lda_results = self.evaluate_classifiers(X_train_lda, X_test_lda, self.y_train, self.y_test)

        # Evaluate classifiers without any dimensionality reduction
        no_dr_results = self.evaluate_classifiers(self.X_train, self.X_test, self.y_train, self.y_test)

        # Combine results into a DataFrame for each dimensionality reduction technique
        result_df = pd.DataFrame({
            'Classifier': list(pca_results.keys()),
            'No_DR_Accuracy': list(no_dr_results.values()),
            'PCA_Accuracy': list(pca_results.values()),
            'LDA_Accuracy': list(lda_results.values())
        })

        # Calculate the mean accuracy for No DR, PCA, and LDA across all classifiers
        no_dr_mean_accuracy = np.mean(list(no_dr_results.values()))
        pca_mean_accuracy = np.mean(list(pca_results.values()))
        lda_mean_accuracy = np.mean(list(lda_results.values()))

        # Add column to indicate the top performing DR technique based on mean accuracy
        top_dr_technique = 'PCA' if pca_mean_accuracy > lda_mean_accuracy else 'LDA'
        top_dr_technique = 'No DR' if no_dr_mean_accuracy > max(pca_mean_accuracy,
                                                                lda_mean_accuracy) else top_dr_technique
        result_df['Top_Performing_DR'] = top_dr_technique

        # Print out results
        print("Performance Comparison of No DR, PCA, and LDA:")
        print(f"Mean No DR Accuracy: {no_dr_mean_accuracy:.4f}")
        print(f"Mean PCA Accuracy: {pca_mean_accuracy:.4f}")
        print(f"Mean LDA Accuracy: {lda_mean_accuracy:.4f}")
        print(f"Top Performing DR Technique: {top_dr_technique}")
        print(result_df)

        return result_df

    def apply_best_dr_technique(self):
        """Check the mean accuracy of No DR, PCA, and LDA, and apply the best DR technique to the entire dataset."""
        # Apply PCA and LDA
        X_train_pca, X_test_pca = self.apply_pca()
        X_train_lda, X_test_lda = self.apply_lda()

        # Evaluate classifiers
        pca_results = self.evaluate_classifiers(X_train_pca, X_test_pca, self.y_train, self.y_test)
        lda_results = self.evaluate_classifiers(X_train_lda, X_test_lda, self.y_train, self.y_test)
        no_dr_results = self.evaluate_classifiers(self.X_train, self.X_test, self.y_train, self.y_test)

        # Calculate the mean accuracy for No DR, PCA, and LDA across all classifiers
        no_dr_mean_accuracy = np.mean(list(no_dr_results.values()))
        pca_mean_accuracy = np.mean(list(pca_results.values()))
        lda_mean_accuracy = np.mean(list(lda_results.values()))

        # Apply the best DR technique (No DR, PCA, or LDA) to the whole dataset
        if no_dr_mean_accuracy > max(pca_mean_accuracy, lda_mean_accuracy):
            print(
                f"No DR is the top performing technique with mean accuracy {no_dr_mean_accuracy:.4f}. Using original data.")
            return self.data  # Return the original dataset
        elif pca_mean_accuracy > lda_mean_accuracy:
            print(
                f"PCA is the top performing technique with mean accuracy {pca_mean_accuracy:.4f}. Applying PCA to the entire dataset.")
            pca = PCA(n_components=self.n_components)
            X_pca_full = pca.fit_transform(self.data)
            pca_df = pd.DataFrame(X_pca_full, columns=[f'PC{i + 1}' for i in range(self.n_components)])
            return pca_df
        else:
            print(
                f"LDA is the top performing technique with mean accuracy {lda_mean_accuracy:.4f}. Applying LDA to the entire dataset.")
            lda = LDA(n_components=self.lda_components)
            X_lda_full = lda.fit_transform(self.data, self.target)
            lda_df = pd.DataFrame(X_lda_full, columns=[f'LD{i + 1}' for i in range(self.lda_components)])
            return lda_df

    def evaluate_no_dr(self):
        """Evaluate the performance of classifiers without any dimensionality reduction."""
        print("Evaluating without any dimensionality reduction...")
        return self.evaluate_classifiers(self.X_train, self.X_test, self.y_train, self.y_test)

    def main(self, classifier):
        # Load the dataset
        # df = pd.read_csv("TrainDataset2024.csv")
        # df = pd.read_excel("preprocessed.xlsx")

        # # Convert DataFrame Columns from String to Integer
        # for column in df.columns:
        #     try:
        #         # Attempt to convert the column to numeric (integer)
        #         df[column] = pd.to_numeric(df[column], errors='coerce')  # Invalid values will be NaN
        #         df[column] = df[column].fillna(0).astype(int)  # Fill NaNs with 0 and convert to int
        #     except Exception as e:
        #         print(f"Skipping column {column}: {e}")

        ## y = self.data[classifier]
        y = self.target
        ## X = self.data.drop([classifier], axis=1)
        X = self.data
        
        # Get the number of features in X
        top_n_features = X.shape[1]  # This will give the number of columns (features) in X

        # Initialize the comparison class
        comparison = PCAvsLDAComparison(X, y, n_components=top_n_features)

        # Compare performance of No DR, PCA, and LDA
        results_df = comparison.compare_performance()

        # Apply the best DR technique to the whole dataset
        best_dr_df = comparison.apply_best_dr_technique()

        # Print out the resulting transformed dataset
        print("Transformed Data with Best DR Technique:")
        print(best_dr_df.head())

        return best_dr_df