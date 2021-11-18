import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    pca = PCA(n_components=2)
    pca.fit(train_df)
    embedding = pca.transform(train_df)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('PCA projection of the Digits dataset', fontsize=24)
    return


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