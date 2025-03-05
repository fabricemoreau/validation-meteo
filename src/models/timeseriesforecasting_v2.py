""" tsforecastodeur pour un plusieurs paramètres """
import numpy as np
import pandas as pd

import joblib

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

periode = '2010-2024'
path    = './data/processed'

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
n_features = len(parametres)
seed = 42



X_train = np.load(path + '/np_tsforecast2_xtrain.npy')
X_test = np.load(path + '/np_tsforecast2_xtest.npy')
y_train = np.load(path + '/np_tsforecast2_ytrain.npy')
y_test = np.load(path + '/np_tsforecast2_ytest.npy')
scaler = joblib.load(path + '/joblib_tsforecast2_scaler.gz')

nb_jours = X_train.shape[1]

# plus petit jeu de données
X_train = X_train[:10000]
y_train = y_train[:10000]

# define model
""" 
inputs = Input(shape=(nb_jours, n_features))
m = Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'valid')(inputs)
m = AveragePooling1D(pool_size=2)(m)
m = Flatten()(m)
m = Dense(256, activation='relu')(m)
#m = Dropout(0.2)(m)
#m = Dense(16, activation='relu')(m)
outputs = Dense(n_features, activation = 'linear')(m)
model = Model(inputs=inputs, outputs=outputs) 
model.compile(optimizer= Adam(learning_rate=0.1), loss='mse')
"""
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(nb_jours, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(n_features, activation = 'linear'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 0.0001,
                           patience = 3,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 0.1,
                               patience = 2,
                               factor = 0.1, 
                               #cooldown = 2,						
                               verbose = 1)
checkpoint = ModelCheckpoint('data/models/tsforecast_v2.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min')

history = model.fit(X_train, y_train, batch_size=32, validation_data= (X_test, y_test), epochs=1,
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint])
# batch_size 128, nb_jours = 10:  64 filtres, 50 dense {'loss': [0.006158497184514999, 0.006046920083463192, 0.005969455931335688, 0.005799451377242804], 'val_loss': [0.006460063625127077, 0.006599133834242821, 0.006427753251045942, 0.006444086786359549], 'learning_rate': [0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.00010000000474974513]}
# batch_size 32, nb_jours = 30, 64 filtres, 128 dense {'loss': [0.009463822469115257, 0.007374047767370939, 0.0071603842079639435, 0.006700814701616764, 0.006681920029222965, 0.006613755598664284, 0.00661190040409565], 'val_loss': [0.0073448303155601025, 0.00772728631272912, 0.006957297213375568, 0.0067716194316744804, 0.006718783639371395, 0.006710231304168701, 0.006709347479045391], 'learning_rate': [0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.00010000000474974513, 0.00010000000474974513, 1.0000000656873453e-05, 1.0000000656873453e-05]}
# à tester sur 30 jours: plusieurs convolutions 

y_pred_test = model.predict(X_test)


# rmse 3 pour nb_jours = 5, filters = 16 dense = 20
y_test_unscaled = scaler.inverse_transform(y_test)
y_pred_test_unscaled = scaler.inverse_transform(y_pred_test)
loss = pd.DataFrame({'ligne': [1]})
for i, param in enumerate(parametres):
    loss[param + '_rmse'] = root_mean_squared_error(y_test_unscaled[:, i], y_pred_test_unscaled[:, i])
loss = loss.drop(columns = 'ligne')

#### A FAIRE
# prédire sur param_origine
# prédire sur jeu de test équilibré entre anomalie/absence anomalie
# dénormaliser les sorties


# anomalies détectées
#### FAUX : il faut prédire sur param + '_origine'
""" ydf = pd.concat([pd.DataFrame(y_test_unscaled.reshape(-1, n_features)),
                pd.DataFrame(y_pred_test_unscaled.reshape(-1, n_features))],
                axis = 1)
ydf.columns = [param + '_true' for param in parametres] + [param + '_pred' for param in parametres]

for i, param in enumerate(parametres):
    print(param)
    anomalies_true = np.where(anomalies_threshold.loc[indexes_test, param + '_anomalie_threshold'], 1, 0)
    ydf[param + '_anomalie_pred'] = np.where(np.abs(ydf[param + '_true'] - ydf[param + '_pred']) > loss[param + '_rmse'][0], 1, 0)
    print(pd.crosstab(anomalies_true, ydf[param + '_anomalie_pred']))
    print(classification_report(anomalies_true, ydf[param + '_anomalie_pred']))
 """