import sklearn

def sklearn_param_search(model, param_grid, X, y):
    grid = sklearn.model_selection.ParameterGrid(param_grid)
    print(f'Grid searching through {len(grid)} possibilities:')
    for combo in list(grid):
        print(combo)
    print()
    
    grid_search = sklearn.model_selection.GridSearchCV(
        model,
        param_grid,
        verbose=3 # Print scores
    ).fit(X, y)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')
    return grid_search.best_params_

class MLModel:
    pass

class RegressionModel(MLModel):
    pass

class ClassificationModel(MLModel):
    pass