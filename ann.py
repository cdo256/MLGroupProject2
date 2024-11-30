import keras.layers
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from model import *
import pandas as pd
import keras

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


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
    

class KerasClassANN(ClassificationModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.hyperparams["input_dim"],input_dim = self.hyperparams["input_dim"], activation = "relu"))

        for i in self.hyperparams["hid_size"]:
            self.model.add(keras.layers.Dense(i, activation = "relu"))

        self.model.add(keras.layers.Dense(self.hyperparams["output_size"],activation = "sigmoid"))
        self.model.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    
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
    
class KerasRegANN(RegressionModel):
    def __init__(self, **hyperparams):
        self.param_grid = {}
        self.hyperparams = hyperparams

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.hyperparams["input_dim"],input_dim = self.hyperparams["input_dim"], activation = "relu"))

        for i in self.hyperparams["hid_size"]:
            self.model.add(keras.layers.Dense(i, activation = "relu"))

        self.model.add(keras.layers.Dense(self.hyperparams["output_size"]))
        self.model.compile(loss = "mean_squared_error",optimizer="adam",metrics=[r_square])
    
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