import pandas as pd
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import feat_selection as fs
from sklearn.model_selection import KFold
from ann import ANNClassifier, ANNRegressor,KerasClassANN, KerasRegANN
from BaseClasses import modelType
from random_forest import RandomForestClassifierModel, RandomForestRegressorModel

def init(toggle):
    global features, pp, base_df, retained_features
    # Here we want to preprocess the train data first then use the same parameters for the test data.
    # To do this we will use the class Preprocessor which is to be implemented.
    # For the time being we will preprocess train and test separately.
    top_n_features = 10
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
    retained_features = ['PgR', 'Gene', 'HER2']
    features = fs.main(X, y, top_n_features, toggle,retained_features=retained_features)

# Perform cross-validation
def evaluate(model, k, task):
    global features, pp, base_df
    print(f'{model}:')

    # KFold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Process each fold
    for train_index, test_index in kf.split(base_df):
        # Split data into training and testing sets
        train_df = base_df.iloc[train_index]
        test_df = base_df.iloc[test_index]
        y_train = {}
        y_test = {}
        X_train_full, y_train[modelType.CLASSIFICATION] , y_train[modelType.REGRESSION] = pp.preprocess_fit(train_df)
        X_test_full, y_test[modelType.CLASSIFICATION], y_test[modelType.REGRESSION] = pp.preprocess_transform(test_df)
        X_train = X_train_full[features.columns]
        X_test = X_test_full[features.columns]

        train_accuracy = model.train(X_train, y_train[task])
        test_accuracy = model.test(X_test, y_test[task])
        print(f'{train_accuracy:.03}, {test_accuracy:.03}')

#Returns a numpy array of the predictions for the model that was passed in
def predict(model,X_pred):
    predictions = model.predict(X_pred)
    return predictions

if __name__ == '__main__':
    pp = Preprocessor()

    doPrediction = False

    testData = pp.load("TestDatasetExample.xls",dropIDs = False)
    print(testData)
    pptestData = pp.preprocess_predict(testData)

    # annReg = ANNRegressor(random_state = 1, max_iter = 5000)
    #
    # init(modelType.REGRESSION)
    # regData = pptestData[features.columns]
    #
    # #evaluate(LinearRegression(), k=10, task=modelType.REGRESSION)
    # evaluate(annReg,k = 10, task = modelType.REGRESSION)
    # #evaluate(KerasRegANN(input_dim = len(features.columns),output_size = 1,hid_size = (100,100,100)),k = 10,task = modelType.REGRESSION)
    #  # evaluate(RandomForestRegressorModel(), k=10, task=modelType.REGRESSION)
    
    init(modelType.CLASSIFICATION)
    classData = pptestData[features.columns]
    #evaluate(LogisticRegression(max_iter=1000), k=10, task=modelType.CLASSIFICATION)
    kerasClass = KerasClassANN(input_dim = len(features.columns),output_size = 1,hid_size = (100,100,100))

    #evaluate(ANNClassifier(random_state = 1, max_iter = 2000, hidden_layer_sizes = (50,)),k = 10, task = modelType.CLASSIFICATION)
    evaluate(kerasClass,k = 10,task = modelType.CLASSIFICATION)
    #evaluate(RandomForestClassifierModel(), k=10, task=modelType.CLASSIFICATION)

    if doPrediction:
        regPredict = predict(annReg,regData)
        classPredict = predict(kerasClass,classData)

        dictRegPredict = {"ID": testData["ID"], "RelapseFreeSurvival": regPredict}
        dictClassPredict = {"ID" : testData["ID"], "pCR (Predicted)": classPredict}
        df_predictions = pd.DataFrame(dictRegPredict)
        df_predictions.to_csv("RFSPrediction.csv",index = False)
        df_predictions = pd.DataFrame(dictClassPredict)
        df_predictions.to_csv("PCRPrediction.csv",index = False)
