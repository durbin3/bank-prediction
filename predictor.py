import numpy as np
import pandas as pd
import os
from model import makeTreeModel
from utils import *

np.set_printoptions(suppress=True)

def main():
    print("Getting Data")
    raw = get_raw_data()
    data_races = split_by_race(raw)
    
    models = {}
    for race in data_races.keys():
        print("\nModel for race=",race)
        if os.path.exists("./model/"+race+".joblib"):
            model = load_model(race)
            models[race]= model
        else:
            model = construct_model(data_races[race])
            models[race]=model
            save_model(model,race)
    
    final_preds(models)
    print("Done")

def construct_model(raw):
    print("Preprocessing Data")
    x_train,x_test,y_train,y_test = preprocessData(raw)
    print("Constructing Model")
    model = makeTreeModel(x_train.values,y_train)
    print("Model Predict:")
    predictions = predict(model,x_test)
    score = score_model(predictions,y_test)
    print("\n\nScore =",score,"\n\n")
    return model

def get_raw_data():
    df = pd.read_csv('./data/lending_train.csv')
    return df

def predict(model,xVals):
    preds = model.predict(xVals)
    return np.where(preds > .5, 1,0)

def score_model(y_preds,y_actual):
    if (len(y_preds)!= len(y_actual)): return -1
    y_preds = y_preds.reshape(len(y_preds),)
    y_actual = y_actual.reshape(len(y_actual),)
    return (np.where(y_preds==y_actual,1,0).sum())/len(y_preds)


def final_preds(models):
    print("Final Predictions")
    predict_df = pd.read_csv('./data/lending_topredict.csv')
    data,columns = trim_data(predict_df)
    data = data.join(predict_df['race'])
    def predict_row(row):
        pred = models[row['race']].predict(row[columns].values.reshape(1,-1))
        return pred[0]
    predictions = data.apply(predict_row,axis=1)
    preds_df = pd.DataFrame(predict_df['ID'])
    preds_df = preds_df.set_index('ID')
    preds_df['loan_paid'] = predictions.values
    preds_df.to_csv('predictions.csv')

def preprocessData(raw):
    data,columns = trim_data(raw)
    train_nopay =  data.loc[data['loan_paid'] == 0].sample(frac=.8)
    train_paid = data.loc[data['loan_paid'] == 1].sample(n=len(train_nopay.index))
    Train = train_paid.append(train_nopay)
    xTrain = Train[columns]

    yTrain = Train[['loan_paid']]

    data = data.drop(Train.index)
    xTest = np.array(data[columns].values.tolist())
    yTest = np.array(data['loan_paid'].tolist()).reshape(len(data['loan_paid']),1)

    return (xTrain,xTest,yTrain,yTest)


if __name__ == "__main__":
    main()