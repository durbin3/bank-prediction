# import tensorflow as tf
# import tensorflow.keras.models as models
# import tensorflow.keras.losses as losses
import numpy as np
from pandas.core.indexes import category
import sklearn
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import os
import math

###
# TODO
# Try to split dataset categorically like a 
# decision tree, and train a different model for each
# leaf of the tree
# e.g. split into white/black, train different model for each
#
# Look into using boosted decision trees as well...
###
np.set_printoptions(suppress=True)

# columns that lock acc at .5: 'debt_to_income_ratio, total_revolving_limit

def main():
    print("Getting Data")
    raw = get_raw_data()
    print("Preprocessing Data")
    x_train,x_test,y_train,y_test = preprocessData(raw)
    # x_train,x_test,y_train,y_test = preprocessDataTree(raw)
    # model = trainModel(x_train,y_train)
    print("Constructing Model")
    model = makeTreeModel(x_train.values,y_train)
    print("Model Predict:")

    predictions = predict(model,x_test)
    score = score_model(predictions,y_test)
    print("\n\nScore =",score,"\n\n")
    final_preds(model)
    print("Done")


def get_raw_data():
    df = pd.read_csv('./data/lending_train.csv')
    # data = df[columns]
    return df

def preprocessData(raw):
    data,columns = trim_data(raw)
    data['fico_diff'] = data['fico_score_range_high'] - data['fico_score_range_low']
    data['income_to_balance'] = data['revolving_balance']/data['annual_income']
    data['balance_to_limit'] = data['revolving_balance']/data['total_revolving_limit']
    data['requ_to_income'] = data['requested_amnt']/data['annual_income']
    train_paid = data.loc[data['loan_paid'] == 1].sample(n=10000)
    train_nopay =  data.loc[data['loan_paid'] == 0].sample(n=10000)

    Train = train_paid.append(train_nopay)
    xTrain = Train[columns]

    yTrain = Train[['loan_paid']]

    data = data.drop(Train.index)
    xTest = np.array(data[columns].values.tolist())
    yTest = np.array(data['loan_paid'].tolist()).reshape(len(data['loan_paid']),1)

    return (xTrain,xTest,yTrain,yTest)


def makeTreeModel(x,y):
    y = np.array(y.values.tolist()).reshape(len(y),)
    print(x.shape,y.shape)
    model = GradientBoostingClassifier(n_estimators=1000,learning_rate=.001,max_depth=1,verbose=True)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    model.fit(x,y)
    return model


# def makeTFModel(x_train,y_train):
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(512,activation='sigmoid'),
#         tf.keras.layers.Dropout(.2),
#         tf.keras.layers.Dense(256,activation='sigmoid'),
#         tf.keras.layers.Dropout(.2),
#         tf.keras.layers.Dense(128,activation='sigmoid'),
#         tf.keras.layers.Dropout(.2),
#         tf.keras.layers.Dense(64,activation='sigmoid'),
#         tf.keras.layers.Dropout(.2),
#         tf.keras.layers.Dense(32,activation='sigmoid'),
#         tf.keras.layers.Dense(1)])

#     print("Model Created")
#     loss = losses.BinaryCrossentropy()
#     adam = tf.keras.optimizers.Adam()
#     model.compile(optimizer=adam, loss=loss,metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=4)
#     print("Model Done Training")
#     return model

def predict(model,xVals):
    preds = model.predict(xVals)
    print(preds)
    return np.where(preds > .5, 1,0)

def score_model(y_preds,y_actual):
    if (len(y_preds)!= len(y_actual)): return -1
    y_preds = y_preds.reshape(len(y_preds),)
    y_actual = y_actual.reshape(len(y_actual),)
    return (np.where(y_preds==y_actual,1,0).sum())/len(y_preds)

def normalize_col(col):
    return (col-col.mean())/col.std()
    # return (col-col.min())/(col.max()-col.min())

def normalize_df(df):
    norm = pd.DataFrame([])
    for col in df:
        norm[col] = normalize_col(df[col])
    return norm

def final_preds(model):
    predict_df = pd.read_csv('./data/lending_topredict.csv')
    data,columns = trim_data(predict_df)
    predictions = model.predict(data[columns].values)

    preds_df = pd.DataFrame(predict_df['ID'])
    preds_df = preds_df.set_index('ID')
    preds_df['loan_paid'] = predictions
    preds_df.to_csv('predictions.csv')

def categorize_col(df,col,columns):
    # raw = raw.join(pd.get_dummies(raw['race']))
    cat_col = pd.get_dummies(df[col])
    cols = columns
    for x in df[col].unique(): 
        # if (np.isnan(x) is False):
        if (type(x)==float):
            print(x)
        else:
            cols.append(x)
    return cat_col,cols

def trim_data(raw):
    columns = [
        'requested_amnt','annual_income','fico_score_range_low','fico_score_range_high',
        'revolving_balance','debt_to_income_ratio','total_revolving_limit'
    ]
    data = pd.DataFrame(raw[columns])
    category_cols = ['race','loan_duration','home_ownership_status','employment_length','reason_for_loan','employment_verified']
    # category_cols = ['race']
    for col in category_cols:
        cat_col, columns = categorize_col(raw,col,columns)
        data = data.join(cat_col)
    
    data = data.fillna(0)
    print(data.head())
    if ('loan_paid' in raw.columns):
        data['loan_paid'] = raw.loc[:,'loan_paid']
    return data,columns


main()