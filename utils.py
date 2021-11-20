import pandas as pd
import numpy as np
import os
from joblib import dump,load

def categorize_col(df,col,columns):
    cat_col = pd.get_dummies(df[col])
    cols = columns
    for x in df[col].unique(): 
        if (type(x)==float):
            print(x)
        else:
            cols.append(x)
    return cat_col,cols

def trim_data(raw):
    columns = [
        'requested_amnt','annual_income',
        'revolving_balance','debt_to_income_ratio','total_revolving_limit'
    ]
    data = pd.DataFrame(raw[columns])
    category_cols = ['home_ownership_status','employment_verified']
    for col in category_cols:
        cat_col, columns = categorize_col(raw,col,columns)
        data = data.join(cat_col)
    
    data = data.fillna(0)
    data['loan_duration'] = np.where(raw['loan_duration']==' 36 months',0,1)
    data['employment_length'] = raw.apply(parse_string,axis=1)
    data['income_to_balance'] = (data['revolving_balance'] + 1)/(data['annual_income']+1)
    data['balance_to_limit'] = (data['revolving_balance'] + 1)/(data['total_revolving_limit']+1)
    data['requ_to_income'] = (data['requested_amnt'] + 1)/(data['annual_income']+1)
    columns.append('loan_duration')
    columns.append('employment_length')
    columns.append('income_to_balance')
    columns.append('balance_to_limit')
    columns.append('requ_to_income')
    if ('loan_paid' in raw.columns):
        data['loan_paid'] = raw.loc[:,'loan_paid']
    if ("NONE" not in data.columns):
        data['NONE'] = 0
        columns.append("NONE")
        
    if ("OTHER" not in data.columns):
        data['OTHER'] = 0
        columns.append("OTHER")

    print(data.head())
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

