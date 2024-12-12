# Overall Project Architecture

## preprocess.py

preprocess.py should contain a class Preprocessor with the following methods:

 - `load(filename)`:
   Returns filename as a data-frame without doing any pre-processing.
 - `preprocess_fit(df)`:
   Takes an unprocessed dataframe and performs the following:
    * Imputes numeric data.
    * Normalizes numeric data.
    * One-hot encodes categorical data.
    * Splits off output columns.
   It should return 3 different data-frames:
    - `y_clf` - the classifier columns, as a data-frame (obtained from
      hot_encoder.get_feature_names_out).
    - `y_reg` - the regression column out as a data-frame.
    - `X` - the remaining columns.
   The transformers for each of the steps should be stored in the Preprocessor object.
 - `preprocess_transform(df)`:
   This function takes a data-frame that may or may not have the output columns. It should perform the same transformations as `preprocess_fit`, with the parameters saved in the Preprocessor object.

## feat_selection.py` features as a list of strings.
This file should expose a single function
 - `compute_top_clf(X, y_clf, n=25)`:
   Takes `X`, `y_clf` - dataframes containing the pre-processed input data, and `n` number of features.
   Returns the best `n` features as a list of strings. 
 - `compute_top_reg(X, y_reg, n=25)`:
   Takes `X`, `y_reg` - dataframes containing the pre-processed input data, and `n` number of features.
   Returns the best `n` features as a list of strings. 

## dimensionality_reduction.py
This file should expose classes PCA and LDA with the following methods:
 - `fit(X, y_clf)`:
   Returns a transformed version of `X` with smaller dimension. `y_clf` is ignored by PCA. 
 - `transform(X)`:
   Transforms `X` using the saved tranform from `fit`.

## linear_regression.py, logistic_regression.py, ann.py, svm.py, decision_tree.py
These files should contain classes such as `LinearRegression`, `ANNRegressor` classes, which are subclasses of `RegressionModel`. The aim is that they should wrap Scikit's models, but specify a param-grid and default parameters. Additionally classes such as `LogisticRegression` and `ANNClassifier`

### **Attributes:**

- `param_grid`: A dictionary that defines the hyperparameters to be searched during model tuning. Currently, it is initialized as an empty dictionary.
- `hyperparams`: A dictionary of hyperparameters passed to the scikit-learn `LinearRegression` model. This is set during initialization.
- `model`: An instance of one of scikit-learn's models, initialized with the provided hyperparameters.

#### **Methods:**

- **`__init__(**hyperparams)`**:
  
  Initializes the model class with optional hyperparameters for the scikit-learn model.

  - **Parameters**:
    - `**hyperparams`: These are passed directly to the model during initialization.

- **`param_search(X, y)`**:
  
  Performs hyperparameter tuning using a grid search over the defined parameter grid (`param_grid`). This method updates both the model and its hyperparameters based on the best results from the search.

  - **Parameters**:
    - `X`: The input features as a data-frame.
    - `y`: The target values data-frame.
  
  - **Returns**: None. Updates the internal model and hyperparameters with the best configuration found during the search.

  - **Example**:
    ```python
    lr.param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
    lr.param_search(X_train, y_train)
    ```

- **`train(X, y)`**:
  
  Trains the regression/classification model on the provided data.

  - **Parameters**:
    - `X`: The input features as a DataFrame.
    - `y`: The target values as a DataFrame.

  - **Returns**: The training score (R^2) of the model on the provided data.

  - **Example**:
    ```python
    score = lr.train(X_train, y_train)
    ```

- **`test(X, y)`**:
  
  Tests the trained linear regression model on new data and returns its performance score.

  - **Parameters**:
    - `X`: The input features as a DataFrame.
    - `y`: The target values as a DataFrame.

  - **Returns**: The test score (R^2) of the model on the provided data.

  - **Example**:
    ```python
    test_score = lr.test(X_test, y_test)
    ```

class LinearRegression(RegressionModel):
    self.param_grid = {}
    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = linear_model.LinearRegression(**hyperparams)
    
    def param_search(self, X, y):
        self.hyperparams = sklearn_param_search(
            self.model, self.param_grid, X, y)

    def train(self, X, y):
        self.model.fit(X, y)
        return clf.score(X, y)

    def test(self, X, y):
        return clf.score(X, y)
