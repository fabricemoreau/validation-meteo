""" tsforecastodeur pour un plusieurs paramètres """
import numpy as np
import pandas as pd

import joblib

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
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



X_train = np.load(path + '/np_tsforecast3_xtrain.npy')
X_test = np.load(path + '/np_tsforecast3_xtest.npy')
X_meta_train = np.load(path + '/np_tsforecast3_xtrain_meta.npy')
X_meta_test = np.load(path + '/np_tsforecast3_xtest_meta.npy')
y_train = np.load(path + '/np_tsforecast3_ytrain.npy')
y_test = np.load(path + '/np_tsforecast3_ytest.npy')
indexes_test = np.load(path + '/np_tsforecast3_indexestrain.npy')

X_val_TN = np.load(path + '/np_tsforecast3_xval_tn.npy')
y_val_TN = np.load(path + '/np_tsforecast3_yval_tn.npy', )
indexes_val_TN = np.load(path + '/np_tsforecast3_indexesval_tn.npy')

scaler = joblib.load(path + '/joblib_tsforecast3_scaler.gz')

nb_jours = X_train.shape[1]

# plus petit jeu de données
X_train = X_train[:10000]
X_meta_train = X_meta_train[:10000]
y_train = y_train[:10000]

# define model
""" inputs = Input(shape=(nb_jours, n_features))
m = Conv1D(filters=32, kernel_size=3, activation='relu', padding = 'valid')(inputs)
m = AveragePooling1D(pool_size=2)(m)
m = Flatten()(m)
m = Dense(256, activation='relu')(m)
#m = Dropout(0.2)(m)
#m = Dense(16, activation='relu')(m)
outputs = Dense(n_features, activation = 'linear')(m)
model = Model(inputs=inputs, outputs=outputs) """
inputs_meta = Input(shape= (X_meta_train.shape[1]))
inputs = Input(shape=(nb_jours, n_features))
m = Conv1D(filters=64, kernel_size=2, activation='relu', padding = 'valid')(inputs)
m = MaxPooling1D(pool_size=2)(m)
m = Flatten()(m)
m = Concatenate()([m, inputs_meta])
m = Dense(128, activation='relu')(m)
#m = Dropout(0.2)(m)
#m = Dense(16, activation='relu')(m)
outputs = Dense(n_features, activation = 'linear')(m)
model = Model(inputs=[inputs, inputs_meta], outputs=outputs) 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#model.compile(optimizer= Adam(learning_rate=0.1), loss='mse')

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
# ne pas utiliser X_test dans validationdata
history = model.fit([X_train, X_meta_train], y_train, batch_size=32, epochs=10, validation_split = 0.2,
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint])
# {'loss': [0.007375519722700119, 0.007274068426340818, 0.007137983571738005, 0.006986081134527922, 0.00698351813480258, 0.006858995649963617, 0.0068119424395263195, 0.006916255224496126, 0.00686273816972971, 0.0067613571882247925], 'mae': [0.06399612128734589, 0.06351082772016525, 0.0627354085445404, 0.06191467121243477, 0.06179816275835037, 0.061247825622558594, 0.06091780960559845, 0.061398640275001526, 0.061342865228652954, 0.06079984828829765], 'learning_rate': [0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513]}

y_pred_test = model.predict([X_test, X_meta_test])


# rmse 3 pour nb_jours = 5, filters = 16 dense = 20
y_test_unscaled = scaler.inverse_transform(y_test)
y_pred_test_unscaled = scaler.inverse_transform(y_pred_test)
loss = pd.DataFrame({'ligne': [1]})
for i, param in enumerate(parametres):
    loss[param + '_rmse'] = root_mean_squared_error(y_test_unscaled[:, i], y_pred_test_unscaled[:, i])
loss = loss.drop(columns = 'ligne')

loss.to_csv(path + '/df_tsforecast3_rmsep.csv', sep=';', index = False)

df_pred = pd.DataFrame({'index': })

#### A FAIRE
# prédire sur param_origine
# prédire sur jeu de test équilibré entre anomalie/absence anomalie
# dénormaliser les sorties


# anomalies détectées
#### FAUX : il faut prédire sur param + '_origine'
ydf = pd.concat([pd.DataFrame(y_test_unscaled.reshape(-1, n_features)),
                pd.DataFrame(y_pred_test_unscaled.reshape(-1, n_features))],
                axis = 1)
ydf.columns = [param + '_true' for param in parametres] + [param + '_pred' for param in parametres]

for i, param in enumerate(parametres):
    print(param)
    anomalies_true = np.where(anomalies_threshold.loc[indexes_test, param + '_anomalie_threshold'], 1, 0)
    ydf[param + '_anomalie_pred'] = np.where(np.abs(ydf[param + '_true'] - ydf[param + '_pred']) > loss[param + '_rmse'][0], 1, 0)
    print(pd.crosstab(anomalies_true, ydf[param + '_anomalie_pred']))
    print(classification_report(anomalies_true, ydf[param + '_anomalie_pred']))
