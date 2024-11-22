import pandas as pd
from linear_regression import LinearRegression
from preprocess import load, preprocess #, Preprocessor
from sklearn.model_selection import train_test_split

#def evaluate_clf(base_df):

#X, y_clf, y_reg = preprocess(base_df)
#X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = \
#    train_test_split(X, y_clf, y_reg, shuffle=True, test_size=0.3)
#model = LinearRegression()
#train_accuracy = model.train(X_train, y_reg_train)
#test_accuracy = model.test(X_test, y_reg_test)
#print(train_accuracy, test_accuracy)

def evaluate():
    k = 5
    base_df = load('TrainDataset2024.xls')
    shuffled_df = base_df.sample(frac=1.0).reset_index(drop=True)
    shuffled_df['fold'] = shuffled_df.index % k
    folds = [fold_df for _, fold_df in shuffled_df.groupby('fold')]
    for test_fold in range(k):
        test_df = folds[test_fold]
        train_df = pd.concat([folds[i] for i in range(k) if i != test_fold])
        #preprocessor = Preprocessor()
        X_train, _, y_reg_train = preprocess(train_df) #preprocessor.fit(train_df)
        X_test, _, y_reg_test = preprocess(test_df) #preprocessor.fit(train_df)

        model = LinearRegression()
        train_accuracy = model.train(X_train, y_reg_train)
        test_accuracy = model.test(X_test, y_reg_test)
        print(train_accuracy, test_accuracy)

if __name__ == '__main__':
    evaluate()