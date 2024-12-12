from model import *
import sklearn.linear_model

class LogisticRegression(ClassificationModel):
    def __init__(self, **hyperparams):
        self.name = 'LogisticRegression'
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = sklearn.linear_model.LogisticRegression(**hyperparams)
    
    def param_search(self, X, y):
        pass # No hyperparameters to tune.

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def test(self, X, y):
        return self.model.score(X, y)
    
    def predict(self,X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")