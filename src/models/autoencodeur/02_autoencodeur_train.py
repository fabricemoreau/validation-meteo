
""" 
Entrainement autoencodeur

Pour utiliser ce script
définissez FICHIER_SUFFIXE en reprenant ce qui a été défini dans le script précédent de préprocessing
choisissez les PARAMETRES à utiliser
Les données seront écrite dans le dossier PATH spécifié

"""
METHOD  = 'autoencodeur'
PATH    = './data/processed/' + METHOD
PARAMETRES = ['ETP', 'GLOT', 'TN', 'TX']

FICHIER_SUFFIXE = '-'.join(PARAMETRES) + '_' + str(2010) + '-' + str(2022) + '-' + str(0.1)

import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# les colonnes doivent être dans l'ordre: parametres + meta_features
X_train = np.load(PATH + '/np_xtrain_' + FICHIER_SUFFIXE + '.npy')
X_val = np.load(PATH + '/np_xval_' + FICHIER_SUFFIXE + '.npy')

nfeatures = X_train.shape[1]
noutputs = len(PARAMETRES)

# on ne veut prédire que les paramètres, pas les meta_features
y_train = X_train[:,0:noutputs]
y_val = X_val[:,0:noutputs]

# define model
inputs = Input(shape=(nfeatures, ), name = 'input')
#e = MyGaussianNoise([0.037, 0.034, 0.020, 0.017, 0, 0, 0, 0, 0])(inputs)
e = Dense(nfeatures *2, name = 'encoder_l1')(inputs) 
e = BatchNormalization(name = 'batchnorm1')(e)
e = LeakyReLU(name = 'leakyrelu1')(e)
e = Dense(nfeatures * 4, name = 'encoder2')(e)
e = BatchNormalization(name = 'batchnorm2')(e)
e = LeakyReLU(name = 'leakyrelu2')(e)
bottleneck = Dense(nfeatures * 6, name = 'output')(e)
d = Dense(noutputs * 4)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(noutputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
outputs = Dense(noutputs, activation='linear')(d)
model = Model(inputs=inputs, outputs=outputs) 
model.compile(optimizer=Adam(learning_rate=1E-4), loss='mse', metrics=['mae'])

model.summary()

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 1E-7,
                           patience = 4,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 1E-5, 
                               patience = 3,
                               factor = 0.1, 
                               min_lr = 1E-7,						
                               verbose = 1)
checkpoint = ModelCheckpoint(PATH + '/autoencodeur_checkpoint_' + FICHIER_SUFFIXE + '.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min',
                                verbose = 1)
history = model.fit(X_train, y_train, 
                    validation_data = (X_val, y_val),
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint],
                    batch_size=256, epochs=20)
pd.DataFrame(history.history).to_csv(PATH + '/autoencodeur_history_' + FICHIER_SUFFIXE + '.csv', sep = ';', index = False)

# Génération du graphique d'entrainement (à déplacer dans un autre fichier)
import matplotlib.pyplot as plt
plt.figure()
# on n'affiche pas le premier EPOCH qui a des valeurs très hautes par rapport aux autres
plt.plot(history.history['loss'][1:], label = 'loss')
plt.plot(history.history['val_loss'][1:], label = 'val_loss')
plt.yscale("log")
plt.legend()
plt.ylabel("échelle logarithmique")
plt.xlabel("Epoch")
plt.xticks(range(1, len(history.history['loss'])))
plt.savefig(PATH + '/autoencodeur_train_history_' + FICHIER_SUFFIXE + '.png')
#plt.show()