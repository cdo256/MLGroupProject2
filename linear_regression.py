from model import *
import sklearn.linear_model

class LinearRegression(RegressionModel):
    def __init__(self, **hyperparams):
        self.name = 'LinearRegression'
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = sklearn.linear_model.LinearRegression(**hyperparams)
    
    def param_search(self, x, y):
        pass # no hyperparams to tune

    def train(self, x, y):
        self.model.fit(x, y)
        return self.model.score(x, y)

    def test(self, x, y):
        return self.model.score(x, y)
    

    def predict(self,x):
        return self.model.predict(x)
        print(f"prediction: {self.model.predict(x)}")
        print(f"prediction probability: {self.model.predict_proba}")

class ElasticNetRegressor(RegressionModel):
    def __init__(self, **hyperparams):
        self.name = 'ElasticNetRegressor'
        self.param_grid = {
            'alpha': [5000e-3, 1000e-3, 500e-3, 200e-3, 100e-3, 10e-3, 1e-3, 100e-6, 10e-6],
            'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        }
        self.hyperparams = hyperparams
        self.model = sklearn.linear_model.ElasticNet(**hyperparams)
    
    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def test(self, X, y):
        return self.model.score(X, y)
    

    def predict(self, X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")

