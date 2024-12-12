import keras.layers
import keras.metrics
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
        self.name = 'ANNClassifier'
        self.param_grid = {
            #'activation': ['relu', 'logistic'],
            # Strength of L2 regularization
            'alpha': [5e-3, 1e-3, 1e-4, 1e-5],
            'hidden_layer_sizes': [
                (100,),
                (100,20),
                (40,20,5),
                (40,20,5,5),
                (40,20,20,5,5),
                (100,40,40,20,5,5),
                (100,100,100),
            ],
        }
        self.hyperparams = hyperparams
        self.model = MLPClassifier(**hyperparams)
    
    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
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



class ANNRegressor(RegressionModel):
    def __init__(self, **hyperparams):
        self.name = 'ANNRegressor'
        self.param_grid = {
            #'activation': ['relu', 'logistic'],
            # Strength of L2 regularization
            #'alpha': [5e-2, 1e-2, 5e-3, 1e-3],
            'alpha': [2e-2, 1e-2], # Evaluate larger regularization parameters
            'hidden_layer_sizes': [
                (100,),
                (100,20),
                (40,20,5),
                (40,20,5,5),
                (40,20,20,5,5),
                (100,40,40,20,5,5),
                #(100,100,100),
            ],
        }
        self.hyperparams = hyperparams
        self.model = MLPRegressor(**hyperparams)
    
    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy)
        return self.model.score(X, y_numpy)

    def test(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        return self.model.score(X, y_numpy)
    
    
    def predict(self,X):
        return self.model.predict(X)
        print(f"Prediction: {self.model.predict(X)}")
        print(f"Prediction probability: {self.model.predict_proba}")

    

class KerasClassANN(ClassificationModel):
    def __init__(self, **hyperparams):
        self.name = 'KerasClassANN'
        self.param_grid = {}
        self.hyperparams = hyperparams

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.hyperparams["input_dim"],input_dim = self.hyperparams["input_dim"], activation = "relu"))

        for i in self.hyperparams["hid_size"]:
            self.model.add(keras.layers.Dense(i, activation = "relu"))

        self.model.add(keras.layers.Dense(self.hyperparams["output_size"],activation = "sigmoid"))
        self.model.compile(loss = "binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    
    def param_search(self, X, y):
        pass # Disabled

    def train(self, X, y):
    
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy,epochs=10,verbose = 0)
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]

    def test(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]
    
    def predict(self,X):
        return ((self.model.predict(X) > 0.5).astype(int)).flatten()
        print(f"Prediction: {(self.model.predict(X) > 0.5).astype(int)}")

        
    
class KerasRegANN(RegressionModel):
    def __init__(self, **hyperparams):
        self.name = 'KerasRegANN'
        self.param_grid = {}
        self.hyperparams = hyperparams

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.hyperparams["input_dim"],input_dim = self.hyperparams["input_dim"], activation = "relu"))

        for i in self.hyperparams["hid_size"]:
            self.model.add(keras.layers.Dense(i, activation = "relu"))

        self.model.add(keras.layers.Dense(self.hyperparams["output_size"]))
        self.model.compile(loss = "mean_squared_error",optimizer="adam",metrics=[r_square])
    
    def param_search(self, X, y):
        pass # Disabled
        
    def train(self, X, y):
    
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        self.model.fit(X, y_numpy,epochs=10)
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]

    def test(self, X, y):
        y_numpy = pd.DataFrame(y).to_numpy().flatten()
        return self.model.evaluate(X, y_numpy,verbose = 0)[1]
    
    def predict(self,X):
        return self.model.predict(X).flatten()
        print(f"Prediction: {(self.model.predict(X) > 0.5).astype(int)}")