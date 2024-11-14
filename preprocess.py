import pandas as pd
import numpy as np
import scipy
import sklearn

base_df = None

# Load returns the dataframe frm the loaded data.
# This sets self.base_df to a this dataframe and returns it.
def load(filename):
    raw_df = pd.read_excel(filename)
    return raw_df

# Take a data-frame and replace 999 with the mean value for each numerical column.
# Returns an imputed data-frame.
def impute_mean(df):
    pass

# Take a data-frame and replace 999 with the median value for each numerical column.
# Returns an imputed data-frame.
def impute_median(df):
    pass

#TODO: More imputation methods?

def normalize(df):
    pass

def one_hot_encode(df):
    pass

# Utility function to get pre-processed data.
def preprocess():
    df = load()

# Returns a numpy array from a data-frame.
def as_numpy(df):
    pass