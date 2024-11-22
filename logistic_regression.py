from model import *
import sklearn.linear_model

class LinearRegression(ClassificationModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = sklearn.linear_model.LinearRegression(**hyperparams)
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def test(self, X, y):
        return self.model.score(X, y)
