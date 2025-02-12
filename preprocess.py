import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from BaseClasses import modelType
#to_excel required openpyxl api

base_df = None
file_name = "TrainDataset2024.xls"

all_df = pd.read_excel(file_name, index_col = False)
all_df = all_df.drop(columns=['ID'])

clf_output_col = 'pCR (outcome)'
reg_output_col = 'RelapseFreeSurvival (outcome)'
output_cols = [clf_output_col, reg_output_col]

# Boolean columns
bool_cols = ['ER', 'PgR', 'HER2', 'TrippleNegative', 'HistologyType', 'LNStatus', 'Gene']

# Categorical columns including boolean
cat_cols = ['ChemoGrade', 'Proliferation', 'TumourStage'] + bool_cols

input_cols = [col for col in all_df.columns if col not in output_cols]

# Numeric columns (continuous)
num_cols = [col for col in all_df.columns if col not in cat_cols and col not in output_cols]

class Preprocessor:
    def __init__(self):
        # Initialize transformers to be used in the pipeline
        self.mean_imputer = SimpleImputer(strategy="mean").set_output(transform="pandas")
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.one_hot_encoder = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([0.0, 1.0, float('nan')])

    # Method to load the dataset
    def load(self, filename, dropIDs = True):
        df = pd.read_excel(filename)
        df.replace(999, np.nan, inplace=True)  # Replace 999 with NaN
        if dropIDs:
            df = df.drop(columns=["ID"], errors="ignore")  # Drop the ID column if it exists
        return df

    def get_output_col(self, task):
        match task:
            case modelType.CLASSIFICATION:
                return clf_output_col
            case modelType.REGRESSION:
                return reg_output_col
            case _:
                raise ValueError('Invalid task')

    # Fit and transform the data for pre-processing
    def preprocess_fit(self, df, cat_cols=cat_cols, num_cols=num_cols, clf_output_col=clf_output_col, reg_output_col=reg_output_col, *, task):
        output_col = self.get_output_col(task)

        # Exclude rows with NA's in the output column.
        df = df.dropna(subset=[output_col])
        output = df[output_col]

        # 1. Impute numeric data (mean for continuous, median for categorical numerics)
        imputed_df = self.mean_imputer.fit_transform(df[input_cols])

        # 2. Normalize numeric data
        scaled_df = self.scaler.fit_transform(imputed_df)
        X = scaled_df

        match task:
            case modelType.CLASSIFICATION:
                label_encoded_clf_np = self.label_encoder.transform(output)
                #label_encoded_clf_df = pd.DataFrame(label_encoded_clf_np, columns=[clf_output_col])

                label_encoded_clf_df = pd.DataFrame(label_encoded_clf_np)
                # 4. Split output columns
                y_clf = label_encoded_clf_df
                return X, y_clf
            case modelType.REGRESSION:
                y_reg = output
                return X, y_reg

    # Transform new data using the fitted parameters
    def preprocess_transform(self, df, cat_cols=cat_cols, num_cols=num_cols, *, task):
        output_col = self.get_output_col(task)

        # Exclude rows with NA's in the output column.
        df = df.dropna(subset=[output_col])
        output = df[output_col]

        # Impute numeric data
        imputed_df = self.mean_imputer.transform(df[input_cols])

        # Normalize numeric data
        scaled_df = self.scaler.transform(imputed_df)

        # One-hot encode categorical data
        # hot_encoded = self.one_hot_encoder.transform(df[cat_cols])
        # hot_encoded_df = pd.DataFrame(
        #     hot_encoded,
        #     columns=self.one_hot_encoder.get_feature_names_out(cat_cols),
        #     index=df.index,
        # )

        X = scaled_df

        match task:
            case modelType.CLASSIFICATION:
                label_encoded_clf_np = self.label_encoder.transform(output)
                #label_encoded_clf_df = pd.DataFrame(label_encoded_clf_np, columns=[clf_output_col])

                label_encoded_clf_df = pd.DataFrame(label_encoded_clf_np)
                # 4. Split output columns
                y_clf = label_encoded_clf_df
                return X, y_clf
            case modelType.REGRESSION:
                y_reg = output
                return X, y_reg
 
    def preprocess_predict(self, df):
        #We don't drop the ID column on load so we want to drop it before preprocssing
        df = df.drop(columns=["ID"], errors="ignore") 
        imputed_df = self.mean_imputer.fit_transform(df)
        scaled_df = self.scaler.fit_transform(imputed_df)
        return scaled_df

# Example usage
if __name__ == "__main__":
    # Define constants for column groups
    clf_output_col = "pCR (outcome)"
    reg_output_col = "RelapseFreeSurvival (outcome)"
    output_cols = [clf_output_col, reg_output_col]

    # Load dataset to identify numeric columns
    file_name = "TrainDataset2024.xls"
    initial_df = pd.read_excel(file_name, index_col=False)
    initial_df = initial_df.drop(columns=["ID"], errors="ignore")

    # Boolean columns
    bool_cols = ["ER", "PgR", "HER2", "TrippleNegative", "HistologyType", "LNStatus", "Gene"]

    # Categorical columns including boolean
    cat_cols = ["ChemoGrade", "Proliferation", "TumourStage"] + bool_cols

    # Numeric columns except boolean and categorical
    num_cols = [col for col in initial_df.columns if col not in cat_cols and col not in output_cols]

    # Initialize and fit the preprocessor
    preprocessor = Preprocessor()
    loaded_df = preprocessor.load(file_name)
    X, y_clf, y_reg = preprocessor.preprocess_fit(loaded_df, cat_cols, num_cols, clf_output_col, reg_output_col)

    # Save preprocessed data
    preprocessed = pd.concat([X, y_clf, y_reg], axis=1)
    preprocessed.to_excel("preprocessed.xlsx", index=False)

    print("Preprocessing completed. Preprocessed data saved to 'preprocessed.xlsx'.")
    

'''
    # Take a data-frame and replace 999 with the mean value for each "continuous" numerical column.
    # Returns an imputed data-frame.
    def impute_mean(dataframe):
        mean_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
        mean_imputed = mean_imputer.fit_transform(dataframe[num_cols])
        return mean_imputed

    # Take a data-frame and replace 999 with the median value for each "categorical" numerical column.
    # Returns an imputed data-frame.
    def impute_median(dataframe):
        median_imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
        median_imputed = median_imputer.fit_transform(dataframe[num_cols])
        return median_imputed

    #TODO: More imputation methods?

    def normalize(dataframe):
        scaler = StandardScaler()
        normalized = scaler.fit_transform(dataframe)
        return normalized

    def one_hot_encode(imputed_median):
        HotEncoder = OneHotEncoder(handle_unknown = "ignore", sparse_output = False)
        hotDataSet = HotEncoder.fit_transform(imputed_median)
        hot_encoded_df = pd.DataFrame(
            hotDataSet, 
            columns=HotEncoder.get_feature_names_out(imputed_median.columns), 
            index=imputed_median.index  # Keep the same index as the input DataFrame
        )
        return hot_encoded_df

    def label_encode_binary(df):
        le = LabelEncoder()
        le.fit([0.0, 1.0, float('nan')])
        return le.transform(df)

    # Utility function to get pre-processed data.
    def preprocess(file_name):
        loaded_df = load(file_name)
        impute_mean_df = impute_mean(loaded_df)
        impute_median_df = impute_median(loaded_df)
        hot_encoded_df = one_hot_encode(impute_median_df)
        preprocessed = pd.concat([impute_mean_df, hot_encoded_df], axis=1)
        preprocessed.to_excel("preprocessed.xlsx", index=False)
        return preprocessed

preprocess(file_name)
'''