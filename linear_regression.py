from model import *
import sklearn.linear_model

class LinearRegression(RegressionModel):
    self.param_grid = {}
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = linear_model.LinearRegression(**hyperparams)
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return clf.score(X, y)

    def test(self, X, y):
        return clf.score(X, y)
