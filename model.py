# import tensorflow as tf
# import tensorflow.keras.models as models
# import tensorflow.keras.losses as losses
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

def makeModel(data):
    race_models = {}
    for race in data['race'].unique():
        print("Making model for race=",race)
        x = data.loc[data['race']==race]
        y = x["loan_paid"]
        race_models[race] = makeTreeModel(x,y)

    return race_models

def makeTreeModel(x,y):
    y = np.array(y.values.tolist()).reshape(len(y),)
    print(x.shape,y.shape)
    model = GradientBoostingClassifier(n_estimators=1000,learning_rate=.0001,max_depth=2,verbose=True,random_state=42)
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
