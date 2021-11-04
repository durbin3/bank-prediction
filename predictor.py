import tensorflow as tf
import tensorflow.keras.losses as losses
import tensorflow.keras.callbacks
import numpy as np
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
columns = [
        'requested_amnt','annual_income','fico_score_range_low','fico_score_range_high',
        'revolving_balance'
]
load_model = True

def main():
    raw = get_raw_data()
    x_train,x_test,y_train,y_test = preprocessData(raw)

    model = trainModel(x_train,y_train)
    if (load_model is False):
        print("Model Predict:")

        predictions = predict(model,x_test)
        score = score_model(predictions,y_test)
        print("\n\nScore =",score,"\n\n")
    final_preds(model)


def get_raw_data():
    print("Loading Data")
    df = pd.read_csv('./lending_train.csv')
    # data = df[columns]
    return df

def preprocessData(raw):
    print("Preprocessing Data")
    data = pd.DataFrame(normalize_df(raw[columns]))
    data = data.join(categorize_col(raw,'race'))
    data = data.join(categorize_col(raw,'loan_duration'))
    data = data.join(categorize_col(raw,'home_ownership_status'))
    data = data.join(categorize_col(raw,'employment_length'))
    # data = data.join(categorize_col(raw,'extended_reason'),rsuffix='ext_')
    extended_reason =raw['extended_reason'].astype('category').cat.codes.to_frame()
    data = data.join(extended_reason)
    data = data.join(categorize_col(raw,'employment_verified'))
    data = data.join(categorize_col(raw,'state'))
    
    data['loan_paid'] = raw.loc[:,'loan_paid']

    train_paid = data.loc[data['loan_paid'] == 1].sample(n=15000)
    train_nopay =  data.loc[data['loan_paid'] == 0].sample(n=15000)

    Train = train_paid.append(train_nopay)
    Train = Train.sample(frac=1)
    print(Train.head())
    xTrain = Train[columns]

    yTrain = Train[['loan_paid']]

    data = data.drop(Train.index)
    xTest = np.array(data[columns].values.tolist())
    yTest = np.array(data['loan_paid'].tolist()).reshape(len(data['loan_paid']),1)
    # yTest = yTest.reshape(len(yTest),1)

    return (xTrain,xTest,yTrain,yTest)



def trainModel(x_train,y_train):
    print("Training Model")
    checkpoint_path = "./model/model.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_freq=5*625,
        verbose=1)

    model = tf.keras.models.Sequential([
        # tf.keras.layers.Dense(1024,activation='sigmoid'),
        # tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512,activation='sigmoid'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(256,activation='sigmoid'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(128,activation='sigmoid'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64,activation='sigmoid'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(32,activation='sigmoid'),
        tf.keras.layers.Dense(1)])

    print("Model Created")
    if (load_model and os.path.exists(checkpoint_dir)):
        print("Loading Model")
        model.load_weights(checkpoint_path).expect_partial()
        return model
    
    loss = losses.BinaryCrossentropy()
    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam, loss=loss,metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20,callbacks=[cp_callback])
    return model

def predict(model,xVals):
    print("Predicting")
    preds = model.predict(xVals)
    return np.where(preds > .5, 1,0)

def score_model(y_preds,y_actual):
    if (len(y_preds)!= len(y_actual)): return -1
    return (np.where(y_preds==y_actual,1,0).sum())/len(y_preds)

def normalize_col(col):
    return (col-col.mean())/col.std()
    # return (col-col.min())/(col.max()-col.min())

def normalize_df(df):
    norm = pd.DataFrame([])
    for col in df:
        norm[col] = normalize_col(df[col])
    return norm

def analysis():
    print("Analysis")
    df = pd.read_csv('./bank-lending-prediction-task/lending_train.csv')
    df = df.drop(columns=['employment','zipcode'])
    graph_data = df.drop(columns=['loan_duration','employment_length','race','reason_for_loan','extended_reason',
    'employment_verified','state','home_ownership_status','type_of_application'])
    dat = graph_data.loc[df['loan_paid']==1].append(df.loc[df['loan_paid']==0])

    i = 0
    j = 0
    max_col = len(graph_data.columns)
    width = math.floor(math.sqrt(max_col))+1
    fix,axs = plt.subplots(width,width)
    print("size = ",width, ' by ', width, "||max= ", max_col)
    print("Boxplotting")
    for col in graph_data.columns:
        print('col: ', col, '||i=',i,' j=',j)
        graph_data.boxplot(column=col,by='loan_paid',ax=axs[i,j])
        i+=1
        if (i==width):
            i = 0
            j+=1
    colors = {'1':'green', '0':'red'}
    # y = dat['loan_paid'].map(colors)
    # print(len(y),len(dat))
    print("Plotting!")
    # pd.plotting.scatter_matrix(dat,alpha=.2)
    plt.show()
    print("Done!")


def final_preds(model):
    predict_df = pd.read_csv('./lending_topredict.csv')
    columns = [
            'requested_amnt','annual_income','fico_score_range_low','fico_score_range_high',
            'revolving_balance'
    ]
    data = pd.DataFrame(normalize_df(predict_df[columns]))
    data = data.join(categorize_col(predict_df,'race'))
    data = data.join(categorize_col(predict_df,'loan_duration'))
    data = data.join(categorize_col(predict_df,'home_ownership_status'))
    data = data.join(categorize_col(predict_df,'employment_length'))
    extended_reason = predict_df['extended_reason'].astype('category').cat.codes.to_frame()
    data = data.join(extended_reason)
    data = data.join(categorize_col(predict_df,'employment_verified'))
    columns = ['requested_amnt', 'annual_income', 'fico_score_range_low', 'fico_score_range_high', 'revolving_balance', 'W', 'B', 'P', 'A', 'N', ' 60 months', ' 36 months', 'RENT', 'MORTGAGE', 'OWN', 'OTHER', 'ANY', 'NONE', '4 years', '10+ years', '2 years', '7 years', '8 years', '3 years', '< 1 year', '1 year', '9 years', '5 years', '6 years', 'Verified', 'Source Verified', 'Not Verified', 'FL', 'AR', 'NY', 'MI', 'MN', 'NJ', 'MA', 'SD', 'MD', 'IN', 'AZ', 'VA', 'TX', 'WI', 'NM', 'NH', 'CA', 'OH', 'WV', 'CO', 'WA', 'NC', 'AL', 'SC', 'OK', 'OR', 'NV', 'PA', 'KS', 'MO', 'MT', 'TN', 'GA', 'IL', 'AK', 'KY', 'CT', 'LA', 'RI', 'MS', 'DC', 'DE', 'ID', 'WY', 'UT', 'HI', 'ND', 'NE', 'VT', 'ME', 'IA'] 
    data = data.join(categorize_col(predict_df,'state'))
    xPred = data[columns]
    preds_df = pd.DataFrame(predict_df['ID'])
    preds_df = preds_df.set_index('ID')
    predictions = predict(model, xPred)

    preds_df['loan_paid'] = predictions
    preds_df.to_csv('predictions.csv')

def categorize_col(df,col):
    print("\tCategorizing Column: ", col)
    cat_col = pd.get_dummies(df[col])
    for x in df[col].unique(): 
        if (type(x)!=float):            
            columns.append(x)
    return cat_col


# analysis()
main()