import pandas as pd
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from preprocess import Preprocessor, clf_output_col, reg_output_col
from sklearn.model_selection import train_test_split
import feat_selection as fs

from ann import ANNClassifier, ANNRegressor,KerasClassANN, KerasRegANN
from BaseClasses import modelType
from random_forest import RandomForestClassifierModel, RandomForestRegressorModel

enable_feature_selection = True
load_features = True
save_features = True
do_prediction = False

def write_features(features, filename):
    with open(filename, 'w') as file:
        for feature in features:
            print(feature, file=file)

def read_features(filename):
    features = []
    try:
        with open(filename) as file:
            for line in file.readlines():
                features.append(line.strip())
        return features
    except FileNotFoundError:
        return None

def init(task):
    global features, pp, base_df, retained_features

    # Here we want to preprocess the train data first then use the same parameters for the test data.
    # To do this we will use the class Preprocessor which is to be implemented.
    # For the time being we will preprocess train and test separately.
    top_n_features = 10
    pp = Preprocessor()
    base_df = pp.load('TrainDataset2024.xls')
    X, y = pp.preprocess_fit(base_df, task=task)

    match task:
        case modelType.REGRESSION:
            filename = 'features-reg.txt'
        case modelType.CLASSIFICATION:
            filename = 'features-clf.txt'
        case _:
            raise ValueError('Invalid task')

    #print(f"Y Columns:\n{y.columns}")
    #reducer = PCAvsLDAComparison(base_df, y, top_n_features) ### This does not work currently, related error - "Data needs to be imputed so there are no NAN values"
    #best_df = reducer.main(y.columns[0])
    #print(best_df)
    retained_features = ['PgR', 'Gene', 'HER2']

    if load_features:
        print(f'Loading features from {filename}...')
        features = read_features(filename)
        if features is None:
            print(f'Warning: Loading features failed')
    if features is None:
        print(f'Generating features...')
        features = fs.main(X, y, top_n_features, task, retained_features=retained_features)
        if save_features:
            print('Saving features...')
            write_features(features, filename)
    return X[features], y

# Perform cross-validation
def evaluate(model, k, task, param_search=False):
    global features, pp, base_df, retained_features
    print(f'{model}:')

   # Shuffle by sampling all the data and drop the original indexes.
    shuffled_df = base_df.sample(frac=1.0).reset_index(drop=True)
    shuffled_df['fold'] = shuffled_df.index % k   
    
    pp = Preprocessor()

    doPrediction = False

    testData = pp.load("TestDatasetExample.xls",dropIDs = False)
    print(testData)
    pptestData     = pp.preprocess_predict(testData)

    # annReg = ANNRegressor(random_state = 1, max_iter = 5000)
    #
    # init
    folds = [fold_df for _, fold_df in shuffled_df.groupby('fold')]
    for test_fold in range(k):
        # leave out fold with index `test_fold`
        test_df = folds[test_fold]
        train_df = pd.concat([folds[i] for i in range(k) if i != test_fold])
        y_train = {}
        y_test = {}
        X_train_full, y_train = pp.preprocess_fit(train_df, task=task)
        X_test_full, y_test = pp.preprocess_transform(test_df, task=task)
        X_train = X_train_full[features]
        X_test = X_test_full[features]

        if param_search:
            model.param_search(X_train, y_train)

        train_accuracy = model.train(X_train, y_train)
        test_accuracy = model.test(X_test, y_test)
        print(f'{train_accuracy:.03}, {test_accuracy:.03}')

#Returns a numpy array of the predictions for the model that was passed in
def predict(model,X_pred):
    predictions = model.predict(X_pred)
    return predictions

def get_models(features):
    return {
        'LinearRegression': (modelType.REGRESSION, LinearRegression()),
        'ANNRegressor': (modelType.REGRESSION, ANNRegressor(random_state = 1, max_iter = 5000)),
        'KerasRegANN': (modelType.REGRESSION, KerasRegANN(input_dim = len(features),output_size = 1,hid_size = (100,100,100))), 
        'RandomForestRegressorModel': (modelType.REGRESSION, RandomForestRegressorModel()), 
        'LogisticRegression': (modelType.CLASSIFICATION, LogisticRegression(max_iter=1000)), 
        'ANNClassifier': (modelType.CLASSIFICATION, ANNClassifier(random_state = 1, max_iter = 2000)), 
        'KerasClassANN': (modelType.CLASSIFICATION, KerasClassANN(input_dim = len(features),output_size = 1,hid_size = (100,100,100))), 
        'RandomForestClassifierModel': (modelType.CLASSIFICATION, RandomForestClassifierModel()),
    }

if __name__ == '__main__':
    mode = 'predict'
    match mode:
        case 'predict':
            predictions = {}
            for task in modelType:
                X_train, y_train = init(task)
                models = get_models(features)
                match task:
                    case modelType.REGRESSION:
                        _, model = models['ANNRegressor']
                    case modelType.CLASSIFICATION:
                        _, model = models['KerasClassANN']
                    case _:
                        raise ValueError('Invalid model type')
                test_df = pp.load('TestDatasetExample.xls', dropIDs=False)
                X_test = pp.preprocess_predict(test_df)
                X_test = X_test[features]
                print('train', X_train.shape)
                print('test', X_test.shape)
                model.train(X_train, y_train)
                predictions = model.predict(X_test)
                print(predictions)
                match task:
                    case modelType.REGRESSION:
                        output_label = reg_output_col
                        output_filename = 'RFSPrediction.csv'
                    case modelType.CLASSIFICATION:
                        output_label = clf_output_col
                        output_filename = 'PCRPrediction.csv'
                    case _:
                        raise ValueError('Invalid model type')
                dictPredict = {
                    'ID': test_df['ID'],
                    output_label: predictions,
                }
                df_predictions = pd.DataFrame(dictPredict) 
                print(f'Writing predictions to {output_filename}')
                df_predictions.to_csv(output_filename, index = False)
        case 'evaluate':
            for task in modelType:
                X, y = init(modelType.REGRESSION)
                models = get_models(features)
                for t, model in models.values():
                    if t != task:
                        continue
                    evaluate(model, k=10, task=task)