import pandas as pd

base_df = None
file_name = "TrainDataset2024.xls"

# Load returns the dataframe frm the loaded data.
# This sets self.base_df to a this dataframe and returns it.
def load(filename):
    global base_df
    base_df = pd.read_excel(filename)
    base_df = base_df.drop(columns=['ID'])
    return base_df

load(file_name)

'''
print(base_df.head)
print(base_df.columns)
for col in base_df.columns:
    print(col, base_df[col].unique().size)
'''

# Boolean columns
bool_cols = ['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'HistologyType', 'LNStatus', 'Gene']
bool_df = base_df[bool_cols]

# Categorical columns including boolean
cat_cols = ['ChemoGrade', 'Proliferation', 'TumourStage'] + bool_cols
cat_df = base_df[cat_cols]

# Numeric columns
num_cols = [col for col in base_df.columns if col not in cat_cols]
#print(num_cols)
num_df = base_df[num_cols]

print(base_df['pCR (outcome)'].unique())
print(base_df[base_df['pCR (outcome)'] == 999]['pCR (outcome)'])
#print(base_df[base_df['pCR (outcome)'] == 999]['pCR (outcome)'].count())
print(f"Column 'pCR (outcome)' has {base_df[base_df['pCR (outcome)'] == 999]['pCR (outcome)'].count()} entries with a value of 999")
print(f"mean of column 'pCR (outcome)' , excluding 999 values, is {base_df[base_df['pCR (outcome)'] != 999]['pCR (outcome)'].mean()}")

# Take a data-frame and replace 999 with the mean value for each numerical column.
# Returns an imputed data-frame.
def impute_mean():
    imputed_mean_df = base_df
    for col in num_df:
        if(base_df[base_df[col] == 999][col].count() != 0):
            print(f"Column {col} has {base_df[base_df[col] != 999][col].count()} entries that are not 999")
            mean_value = base_df[base_df[col] != 999][col].mean()
            imputed_mean_df[col] = base_df[col].replace(999, mean_value)
    return imputed_mean_df  

impute_mean()

# Take a data-frame and replace 999 with the median value for each numerical column.
# Returns an imputed data-frame.
def impute_median():
    imputed_median_df = base_df
    for col in num_df:
        if(base_df[base_df[col] == 999][col].count() != 0):
            print(f"Column {col} has {base_df[base_df[col] != 999][col].count()} entries that are not 999")
            median_value = base_df[base_df[col] != 999][col].median()
            imputed_median_df[col] = base_df[col].replace(999, median_value)
    return imputed_median_df 

impute_median()

#TODO: More imputation methods?

def normalize():
    pass

def one_hot_encode():
    pass

# Utility function to get pre-processed data.
def preprocess(df):
    pass