import pandas as pd
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import feat_selection as fs
from dimensionality_reduction import PCAvsLDAComparison


from ann import ANNClassifier, ANNRegressor,KerasClassANN, KerasRegANN
from BaseClasses import modelType



def init(toggle):
    global features, pp, base_df
    # Here we want to preprocess the train data first then use the same parameters for the test data.
    # To do this we will use the class Preprocessor which is to be implemented.
    # For the time being we will preprocess train and test separately.
    top_n_features = 10
    pp = Preprocessor()
    base_df = pp.load('TrainDataset2024.xls')

    X, y_clf, y_reg = pp.preprocess_fit(base_df)

    if toggle == modelType.REGRESSION:
        y = y_reg
    elif toggle == modelType.CLASSIFICATION:
        y = y_clf
    else:
        print("ERRROR")

    print(f"Y Columns:\n{y.columns}")
    #reducer = PCAvsLDAComparison(base_df, y, top_n_features) ### This does not work currently, related error - "Data needs to be imputed so there are no NAN values"
    #best_df = reducer.main(y.columns[0])
    #print(best_df)
    features = fs.main(X, y, top_n_features, toggle)

# Perform cross-validation
def evaluate(model, k, task):
    global features, pp, base_df
    print(f'{model}:')

    # Shuffle by sampling all the data and drop the original indexes.
    shuffled_df = base_df.sample(frac=1.0).reset_index(drop=True)
    # Assign a fold based on the new shuffled indexes
    shuffled_df['fold'] = shuffled_df.index % k
    # folds is a list of k groups of roughly equal size
    folds = [fold_df for _, fold_df in shuffled_df.groupby('fold')]
    for test_fold in range(k):
        # leave out fold with index `test_fold` 
        test_df = folds[test_fold]
        train_df = pd.concat([folds[i] for i in range(k) if i != test_fold])
        y_train = {}
        y_test = {}
        X_train_full, y_train[modelType.CLASSIFICATION] , y_train[modelType.REGRESSION] = pp.preprocess_fit(train_df)
        X_test_full, y_test[modelType.CLASSIFICATION], y_test[modelType.REGRESSION] = pp.preprocess_transform(test_df)
        X_train = X_train_full[features.columns]
        X_test = X_test_full[features.columns]

        train_accuracy = model.train(X_train, y_train[task])
        test_accuracy = model.test(X_test, y_test[task])
        print(f'{train_accuracy:.03}, {test_accuracy:.03}')

if __name__ == '__main__':
    init(modelType.REGRESSION)
    evaluate(LinearRegression(), k=10, task=modelType.REGRESSION)
    evaluate(ANNRegressor(random_state = 1, max_iter = 5000),k = 10, task = modelType.REGRESSION)
    evaluate(KerasRegANN(input_dim = len(features.columns),output_size = 1,hid_size = (100,100,100)),k = 10,task = modelType.REGRESSION)


    init(modelType.CLASSIFICATION)
    evaluate(LogisticRegression(max_iter=1000), k=10, task=modelType.CLASSIFICATION)
    evaluate(ANNClassifier(random_state = 1, max_iter = 2000),k = 10, task = modelType.CLASSIFICATION)
    evaluate(KerasClassANN(input_dim = len(features.columns),output_size = 1,hid_size = (100,100,100)),k = 10,task = modelType.CLASSIFICATION)
    
