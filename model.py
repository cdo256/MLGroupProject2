
def sklearn_param_search(model, param_grid, X, y)
    grid_search = sklearn.model_selection.GridSearchCV(
        model,
        param_grid
    ).fit(X, y)
    return grid_search.best_, grid_search.best_params_

class MLModel:
    pass

class RegressionModel(MLModel):
    pass

class ClassificationModel(MLModel):
    pass