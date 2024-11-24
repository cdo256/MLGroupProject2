import keras.layers
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from model import *
import pandas as pd
import keras

class ANNClassifier(ClassificationModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = MLPClassifier(**hyperparams)
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return self.model.score(X, y)

    def test(self, X, y):
        return self.model.score(X, y)


class ANNRegressor(RegressionModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = MLPRegressor(**hyperparams)
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy)
        return self.model.score(X, y_numpy)

    def test(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        return self.model.score(X, y_numpy)
    

class KerasANN(ClassificationModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(10,input_dim = 118, activation = "relu"))
        self.model.add(keras.layers.Dense(1,activation = "sigmoid"))
        self.model.compile(loss = "binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
    
    def param_search(self, X, y):
        self.model, self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
    
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy,epochs=10)
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]

    def test(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]