from model import *
import sklearn.linear_model

class LogisticRegression(ClassificationModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = sklearn.linear_model.LogisticRegression(**hyperparams)
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def test(self, X, y):
        return self.model.score(X, y)
    
    def predict(self,X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")