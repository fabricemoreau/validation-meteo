""" autoencodeur pour un plusieurs paramètres en série temporelle"""
# en suivant https://keras.io/examples/timeseries/timeseries_anomaly_detection/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model 

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

periode = '2010-2024'
path    = './data/processed/autoenc'

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
n_features = len(parametres)
seed = 42



X_train = np.load(path + '/np_xtrain.npy')
X_test = np.load(path + '/np_xtest.npy')
#X_meta_train = np.load(path + '/np_xtrain_meta.npy')
X_meta_test = np.load(path + '/np_xtest_meta.npy')
y_train = np.load(path + '/np_ytrain.npy')
y_test = np.load(path + '/np_ytest.npy')
indexes_test = np.load(path + '/np_indexestest.npy')

#X_val_TN = np.load(path + '/np_xval_tn.npy')
#y_val_TN = np.load(path + '/np_yval_tn.npy')
#indexes_val_TN = np.load(path + '/np_indexesval_tn.npy')

scaler = joblib.load(path + '/joblib_scaler.gz')

nb_jours = X_train.shape[1]

# plus petit jeu de données
X_train = X_train[:(X_train.shape[0] // 2)]
#X_meta_train = X_meta_train[:10000]
#y_train = y_train[:10000]

# define model
#inputs_meta = Input(shape= (X_meta_train.shape[1],))
inputs = Input(shape=(nb_jours, n_features), name = 'input')
m = Conv1D(filters=32, kernel_size=2, strides = 2, activation='relu', padding = 'same', name = 'conv1')(inputs)
m = Dropout(rate = 0.2, name = 'drop1')(m)
m = Conv1D(filters=16, kernel_size=2, strides = 2, activation='relu', padding = 'same', name  = 'conv2')(m)
#m = Flatten()(m)
#m = Concatenate()([m, inputs_meta])
#m = Dense(128, activation='relu')(m)
m = Conv1DTranspose(filters=16, kernel_size=2, strides = 2, activation='relu', padding = 'same', name = 'convtransp1')(m)
m = Dropout(rate = 0.2)(m)
m = Conv1DTranspose(filters=32, kernel_size=2, strides = 2, activation='relu', padding = 'same', name = 'convtransp2')(m)
outputs = Conv1DTranspose(filters=4, kernel_size= 2, padding = 'same', name = 'output')(m)
#model = Model(inputs=[inputs, inputs_meta], outputs=outputs) 
model = Model(inputs=inputs, outputs=outputs) 
model.compile(optimizer=Adam(learning_rate=1E-3), loss='mse', metrics=['mae'])
model.summary()

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 1E-5,
                           patience = 3,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 0.001,
                               patience = 2,
                               factor = 0.1, 
                               #cooldown = 2,						
                               verbose = 1)
checkpoint = ModelCheckpoint(path + '/autoencodeur_checkpoint.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min')
history = model.fit(X_train, X_train, 
                    validation_data = (X_test, X_test),
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint],
                    batch_size=32, epochs=10)
# {'loss': [0.0017516941297799349, 0.001152091776020825, 0.0011082716519013047, 0.0010469603585079312, 0.001044145436026156, 0.0010346017079427838], 'mae': [0.030261138454079628, 0.025109004229307175, 0.02452913299202919, 0.02371225692331791, 0.023683100938796997, 0.023556610569357872], 'val_loss': [0.007493358105421066, 0.007624099031090736, 0.007091599982231855, 0.007809476461261511, 0.008102193474769592, 0.008179611526429653], 'val_mae': [0.06855253130197525, 0.0708676353096962, 0.06901231408119202, 0.07322104275226593, 0.0748029500246048, 0.07523316890001297], 'learning_rate': [0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.00010000000474974513, 0.00010000000474974513, 1.0000000656873453e-05]}
model = load_model(path + '/autoencodeur_checkpoint.keras')
# prend trop de mémoire de calculer tout X_train, on prend la moitié
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
# Get reconstruction loss threshold.
# question: faudrait-il un axis = 0?
threshold = np.max(train_mae_loss)
#0.2433545042611586
#np.max(train_mae_loss, axis = 0)
#array([0.2433545 , 0.19970821, 0.16277871, 0.17496798])

plt.figure()
plt.hist(train_mae_loss, bins=50)
#plt.plot(threshold[0], [300], label = "threshold 0")
#plt.plot(threshold[1], [300], label = "threshold 1")
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/train_mae_loss.png')
plt.show()

# reconstruction 1ere séquence
graph_sequence(X_train[0], "sequence_1.png", X_train_pred[0])
graph_sequence(scaler.inverse_transform(X_train[0]), 'sequence_1_denormalise.png', scaler.inverse_transform(X_train_pred[0]))

X_test_pred = model.predict(X_test)
test_mae_loss_param = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_mae_loss = test_mae_loss_param.reshape((-1))

plt.figure()
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/test_mae_loss.png')
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anomalies_dict = {}
for i, param in enumerate(parametres):
    anomalies_dict[param + '_anomalie_pred'] = indexes_test[np.where(test_mae_loss_param[:][:,i])].tolist()
    np.save(path + '/anomalies_pred_test_' + param + '.npy', indexes_test[np.where(test_mae_loss_param[:][:,i])])

np.save(path + '/np_xval_pred.npy', X_test_pred)
    
graph_sequence(scaler.inverse_transform(X_test[0]), 'test_pred0.png', scaler.inverse_transform(X_test_pred[0]))
graph_sequence(scaler.inverse_transform(X_test[100]), 'test_pred100.png', scaler.inverse_transform(X_test_pred[100]))
