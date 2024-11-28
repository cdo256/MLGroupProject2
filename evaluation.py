import pandas as pd
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import feat_selection as fs

from ann import ANNClassifier, ANNRegressor,KerasANN

def init():
    global features, pp, base_df
    # Here we want to preprocess the train data first then use the same parameters for the test data.
    # To do this we will use the class Preprocessor which is to be implemented.
    # For the time being we will preprocess train and test separately.
    top_n_features = 10
    pp = Preprocessor()
    base_df = pp.load('TrainDataset2024.xls')
    X, y_clf, y_reg = pp.preprocess_fit(base_df)
    print(type(X))
    print(type(y_clf))
    print(type(y_reg))
    features = fs.main(X, y_clf, y_reg, top_n_features)

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
        X_train_full, y_train['clf'] , y_train['reg'] = pp.preprocess_fit(train_df)
        X_test_full, y_test['clf'], y_test['reg'] = pp.preprocess_transform(test_df)
        X_train = X_test_full[features]
        X_test = X_test_full[features]

        train_accuracy = model.train(X_train, y_train[task])
        test_accuracy = model.test(X_test, y_test[task])
        print(f'{train_accuracy:.03}, {test_accuracy:.03}')

if __name__ == '__main__':
    init()
    evaluate(LinearRegression(), k=10, task='reg')
    #evaluate(LogisticRegression(max_iter=1000), k=10, task='clf')
    #evaluate(ANNClassifier(random_state = 1, max_iter = 2000),k = 10, task = "clf")
    evaluate(ANNRegressor(random_state = 1, max_iter = 5000),k = 10, task = "reg")
    #evaluate(KerasANN(),k = 10,task = "clf")