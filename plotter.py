import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.decomposition import PCA
import umap

def analysis():
    print("Analysis")
    print("Loading Data Sets")
    train_df = pd.read_csv('./data/lending_train.csv')
    test_df = pd.read_csv('./data/lending_topredict.csv')

    num_entries = len(train_df['ID'])
    # data = train_df['requested_amnt']
    # t_data = test_df['requested_amnt']
    # mu = np.average(data)
    # mut = np.average(t_data)
    # std = np.std(data)
    # stdt = np.std(t_data)
    # print("Plotting!")
    # for b in range(10,20):
    #     plt.hist(data,b,color='blue',alpha=.5)
    #     plt.hist(t_data,b,color='red', alpha=.5)
    #     plt.savefig(f'./plots/histogram_{b}.jpg')
    #     plt.clf()

    # print(f'mu: {mu}, std: {std}')
    # print(f'mu: {mut}, std: {stdt}')

    
    data = train_df[['race','requested_amnt']]
    t_data = test_df[['race','requested_amnt']]

    cl,_ = categorize_col(data,'race',[])
    tcl,_ = categorize_col(t_data,'race',[])

    print(cl.head())

    totals = [cl[col].sum() for col in cl.columns]
    t_totals = [tcl[col].sum() for col in tcl.columns]

    print("Training:\t", np.array(totals)/num_entries)
    print("Test:\t", np.array(t_totals)/len(test_df['ID']))
    print("Done!")


def cluster():
    print("Clustering")
    print("Loading Data Sets")
    train_df = pd.read_csv('./data/lending_train.csv')
    test_df = pd.read_csv('./data/lending_topredict.csv')

    train_data = preprocess(train_df)
    test_data = preprocess(test_df)
    print(train_data.head())
    key = pd.Series(np.zeros(len(train_data.index)))
    key = key.append(pd.Series(np.ones(len(test_data.index))))
    
    all_data = train_data.append(test_data)
    print(len(train_data.index),len(test_data.index),len(all_data.index),len(key.index))
    plot_pca(all_data,"both",key)
    
    print("UMAP Clustering")
    reducer = umap.UMAP(n_neighbors=30,min_dist=0.0,n_components=2,random_state=42)
    embedding = reducer.fit_transform(all_data)
    print(f'Dimensionality of the embedding: {embedding.shape}')
    plot_embedding(embedding,key,"both","UMAP")
    print("Done")
    return

def plot_pca(df,data_type,key):
    print("PCA Reduction")
    pca = PCA(n_components=2)
    pca.fit(df)
    embedding = pca.transform(df)
    
    plot_embedding(embedding,key,data_type,"PCA")
    
def plot_embedding(embedding,key,data_type,embedding_type):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=key, cmap='Spectral', s=.1,alpha=.6)
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(2))
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{embedding_type} projection of the {data_type} dataset', fontsize=24)
    plt.savefig(f'./images/{embedding_type}_{data_type}.jpg')
    plt.clf()
    

def preprocess(raw):
    data, cols = trim_data(raw)
    
    data['fico_diff'] = data['fico_score_range_high'] - data['fico_score_range_low']
    data['income_to_balance'] = (data['revolving_balance']+1)/(data['annual_income']+1)
    data['balance_to_limit'] = (data['revolving_balance']+1)/(data['total_revolving_limit']+1)
    data['requ_to_income'] = (data['requested_amnt']+1)/(data['annual_income']+1)
    
    data = data.fillna(0)
    data = data.sample(frac=.1)
    return data 

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
    # if ('loan_paid' in raw.columns):
    #     data['loan_paid'] = raw.loc[:,'loan_paid']
    return data,columns


def categorize_col(df,col,columns):
    cat_col = pd.get_dummies(df[col])
    cols = columns
    for x in df[col].unique(): 
        if (type(x)==float):
            print(x)
        else:
            cols.append(x)
    return cat_col,cols

# analysis()
cluster()