import pandas as pd
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from preprocess import load, preprocess #, Preprocessor
from sklearn.model_selection import train_test_split

# Perform cross-validation
def evaluate(model, k, task):
    base_df = load('TrainDataset2024.xls')
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
        # Here we want to preprocess the train data first then use the same parameters for the test data.
        # To do this we will use the class Preprocessor which is to be implemented.
        # For the time being we will preprocess train and test separately.
        #preprocessor = Preprocessor()
        y_train = {}
        y_test = {}
        X_train, y_train['clf'] , y_train['reg'] = preprocess(train_df) #preprocessor.fit(train_df)
        X_test, y_test['clf'], y_test['reg'] = preprocess(test_df) #preprocessor.transform(test_df)

        # An example
        train_accuracy = model.train(X_train, y_train[task])
        test_accuracy = model.test(X_test, y_test[task])
        print(train_accuracy, test_accuracy)

if __name__ == '__main__':
    evaluate(LinearRegression(), k=5, task='reg')
    evaluate(LogisticRegression(max_iter=1000), k=5, task='clf')