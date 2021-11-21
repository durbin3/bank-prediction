from math import dist
import pandas as pd
import numpy as np
import os
from joblib import dump,load
from sklearn.metrics import matthews_corrcoef


def categorize_col(df,col,columns):
    cat_col = pd.get_dummies(df[col])
    cols = columns
    for x in df[col].unique(): 
        if (type(x)==float):
            print(x)
        else:
            cols.append(x)
    return cat_col,cols

def trim_data(raw,mode="train"):
    print("Trimming Data")
    columns = [
        'requested_amnt','annual_income',
        'revolving_balance','debt_to_income_ratio','total_revolving_limit',
        'months_since_last_delinq'
    ]
    data = pd.DataFrame(raw[columns])
    data.fillna(0)

    data['loan_duration'] = np.where(raw['loan_duration']==' 60 months',1,0)
    data['employment_length'] = raw.apply(parse_string,axis=1)
    data['income_to_balance'] = (data['revolving_balance'] + 1)/(data['annual_income']+1)
    data['balance_to_limit'] = (data['revolving_balance'] + 1)/(data['total_revolving_limit']+1)
    data['requ_to_income'] = (data['requested_amnt'] + 1)/(data['annual_income']+1)
    data['income_to_limit'] = (data['annual_income']+1)/(data['total_revolving_limit']+1)
    columns.append('loan_duration')
    columns.append('employment_length')
    columns.append('income_to_balance')
    columns.append('balance_to_limit')
    columns.append('requ_to_income')
    columns.append('income_to_limit')

    if (mode == "train"):
        data = clense_df(data)
    data,columns = add_categorical(data,raw,columns)
    print(data)
    return data,columns

def split_by_race(raw):
    data_sets = {}
    for race in raw['race'].unique():
        data_sets[race] = raw.loc[raw['race']==race]
    return data_sets

def parse_string(row):
    x = row['employment_length']
    if (x == '1 year'):
        return 1
    if (x == '2 years'):
        return 2
    if (x == '3 years'):
        return 3
    if (x == '4 years'):
        return 4
    if (x == '5 years'):
        return 5
    if (x == '6 years'):
        return 6
    if (x == '7 years'):
        return 7
    if (x == '8 years'):
        return 8
    if (x == '9 years'):
        return 9
    if (x == '10+ years'):
        return 10
    else:
        return 0

def add_categorical(data,raw,columns):
    
    category_cols = ['home_ownership_status']
    for col in category_cols:
        cat_col, columns = categorize_col(raw,col,columns)
        data = data.join(cat_col)
            
    data = data.fillna(0) 
    if ('loan_paid' in raw.columns):
        data['loan_paid'] = raw.loc[:,'loan_paid']

    if ("NONE" not in data.columns):
        data['NONE'] = 0
        columns.append("NONE")
        
    if ("OTHER" not in data.columns):
        data['OTHER'] = 0
        columns.append("OTHER")

    return data,columns
    
def normalize_col(col):
    return (col-col.mean())/col.std()
    # return (col-col.min())/(col.max()-col.min())

def normalize_df(df):
    norm = pd.DataFrame([])
    for col in df:
        norm[col] = normalize_col(df[col])
    return norm

def save_model(model,name):
    path = "./model/"+name+".joblib"
    s = dump(model,path)

def load_model(name):
    path = "./model/"+name+".joblib"
    print("Loading Model: ", path)
    return load(path)

def create_dir_if_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def remove_outliers(arr):
    mean = np.mean(arr)
    standard_deviation = np.std(arr)
    distance_from_mean = abs(arr - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    trimmed = arr[not_outlier]
    return trimmed
    
def clense_df(data):
    print("Removing Outliers")
    outlier_cols = []
    for col in data.columns:
        data[col+'_outliers']= calc_if_outlier(data,col)
        outlier_cols.append(col+'_outliers')
    
    data['is_outlier'] = 0
    for col in outlier_cols:
        data['is_outlier'] += data[col]
        
    data = data.loc[data['is_outlier']==0]
    outlier_cols.append('is_outlier')
    data = data.drop(columns=outlier_cols)
    return data

def calc_if_outlier(df,column):
    arr = df[column]
    mean = np.mean(arr)
    standard_deviation = np.std(arr)
    distance_from_mean = abs(arr - mean)
    max_deviations = 2
    col = np.where(distance_from_mean < (max_deviations*standard_deviation),0,1)
    return col

def calc_mcc(pred,actual):
    mcc = matthews_corrcoef(pred,actual)
    return mcc