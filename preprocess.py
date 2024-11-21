import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
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

# Numeric columns (continuous)
num_cols = [col for col in all_df.columns if col not in cat_cols and col not in output_cols]


# Load returns the dataframe frm the loaded data.
# This sets self.base_df to a this dataframe and returns it.
def load(filename):
    global base_df
    base_df = pd.read_excel(filename)
    base_df.replace(999, np.nan, inplace=True) 
    base_df = base_df.drop(columns=['ID'])
    return base_df

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
def preprocess(base_df):
    imputed_df = impute_mean(base_df[num_cols])
    normalized_df = normalize(imputed_df)
    hot_encoded_df = one_hot_encode(base_df[cat_cols])
    X = pd.concat([imputed_df, hot_encoded_df], axis=1)
    y_clf = label_encode_binary(base_df[clf_output_col])
    y_reg = base_df[reg_output_col]
    return X, y_clf, y_reg
