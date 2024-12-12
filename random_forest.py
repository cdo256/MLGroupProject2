from model import ClassificationModel, RegressionModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pandas as pd
from model import *

class RandomForestClassifierModel(ClassificationModel):
    def __init__(self, **hyperparams):
        self.name = 'RandomForestClassifierModel'
        self.param_grid = {
            'n_estimators': [100, 500],
            'max_depth': [3, 5, 10, None],
            'criterion': ['gini', 'entropy', 'log_loss'],
        }
        self.hyperparams = hyperparams
        self.model = RandomForestClassifier(**hyperparams)

    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)  # Return accuracy as the performance metric

    def test(self, X, y):
        y_pred = self.model.predict(X)
        return accuracy_score(y, y_pred)  # Return accuracy for testing
    
    def predict(self,X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")


class RandomForestRegressorModel(RegressionModel):
    def __init__(self, **hyperparams):
        self.name = 'RandomForestRegressorModel'
        self.param_grid = {
            'n_estimators': [100, 500],
            'max_depth': [3, 5, 10, None],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
        }
        self.hyperparams = hyperparams
        self.model = RandomForestRegressor(**hyperparams)

    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        # Ensure y is a 1D numpy array for fitting
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy)
        y_pred = self.model.predict(X)
        return r2_score(y_numpy, y_pred)  # Return R² score as the evaluation metric

    def test(self, X, y):
        # Make predictions for the test data
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        y_pred = self.model.predict(X)
        return r2_score(y_numpy, y_pred)  # Return R² score for testing


    def predict(self,X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")
