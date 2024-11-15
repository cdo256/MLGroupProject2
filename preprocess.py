import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
#to_excel required openpyxl api

base_df = None
file_name = "TrainDataset2024.xls"

all_df = pd.read_excel(file_name, index_col = False)
all_df = all_df.drop(columns=['ID'])

# Boolean columns
bool_cols = ['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'HistologyType', 'LNStatus', 'Gene']

# Categorical columns including boolean
cat_cols = ['ChemoGrade', 'Proliferation', 'TumourStage'] + bool_cols

# Numeric columns (continuous)
num_cols = [col for col in all_df.columns if col not in cat_cols]

'''
all_df = pd.read_excel(file_name, index_col = False)
all_df.replace(999, np.nan, inplace=True) 
all_df = all_df.drop(columns=['ID'])
print("info")
print(all_df.max())
bool_df = all_df[bool_cols]
cat_df = all_df[cat_cols]
num_df = all_df[num_cols]
# Impute mean for continuous numeric columns
mean_imputer = SimpleImputer(strategy='mean').set_output(transform='pandas')
mean_imputed = mean_imputer.fit_transform(num_df)
mean_imputed.to_excel("mean_imputed.xlsx", index=False)

# Impute median for categorical columns
median_imputer = SimpleImputer(strategy='median').set_output(transform='pandas')
median_imputed = median_imputer.fit_transform(cat_df)
median_imputed.to_excel("median_imputed.xlsx", index=False)

# Combine the imputed columns with the rest of the original data
all_df[num_cols] = mean_imputed
all_df[cat_cols] = median_imputed
all_df.to_excel("all_df.xlsx", index=False)
'''

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
    median_imputed = median_imputer.fit_transform(dataframe[cat_cols])
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