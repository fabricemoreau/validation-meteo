""" autoencodeur pour un plusieurs paramètres"""
import numpy as np
import pandas as pd

form layer_mygaussiannoise import MyGaussianNoise

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


method  = 'autoencsimple'
path    = './data/processed/' + method

parametres = ['ETP', 'GLOT', 'TN', 'TX'] 

fichier_suffixe = '-'.join(parametres) + '_' + str(2010) + '-' + str(2022)

train = pd.read_csv(path + '/train_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
test = pd.read_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'month_sin', 'month_cos', 'jourjulien']

X_train = train[parametres + meta_features].values
X_test = test[parametres + meta_features].values

nfeatures = X_train.shape[1]
noutputs = len(parametres)

# on ne veut prédire que les paramètres, pas les meta_features
y_train = X_train[:,0:noutputs]
y_test = X_test[:,0:noutputs]

#@keras_export("keras.layers.GaussianNoise")


# define model
inputs = Input(shape=(nfeatures, ), name = 'input')
e = BatchNormalization(name = 'batchnorm0')(inputs)
e = MyGaussianNoise([0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0])(e) # 0.03
e = Dense(nfeatures *2, name = 'encoder_l1')(e)
#e = Dropout(rate = 0.2, name = 'drop1')(e)
e = BatchNormalization(name = 'batchnorm1')(e)
e = LeakyReLU(name = 'leakyrelu1')(e)
e = Dense(nfeatures, name = 'encoder2')(e)
#e = Dropout(rate = 0.2)(e)
e = BatchNormalization(name = 'batchnorm2')(e)
e = LeakyReLU(name = 'leakyrelu2')(e)
bottleneck = Dense(nfeatures, name = 'output')(e)
#d = Dropout(rate = 0.2)(bottleneck)
d = Dense(noutputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# decoder level 2
d = Dense(noutputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
outputs = Dense(noutputs, activation='linear')(d)
model = Model(inputs=inputs, outputs=outputs) 
model.compile(optimizer=Adam(learning_rate=1E-3), loss='mse', metrics=['mae'])
#model.compile(optimizer=Adam(learning_rate=1E-2), loss='mse', metrics=['mae'])

model.summary()

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 1E-7,
                           patience = 3,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 1E-5, # essayer 1E-6 au lieu de 1E-5
                               patience = 2,
                               factor = 0.1, 
                               min_lr = 1E-6,# ajouté
                               #cooldown = 2,						
                               verbose = 1)
checkpoint = ModelCheckpoint(path + '/autoencodeur_checkpoint_' + fichier_suffixe + '.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min',
                                verbose = 1)
history = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test),
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint],
                    batch_size=128, epochs=20)
pd.DataFrame(history.history).to_csv(path + '/autoencodeur_history_' + fichier_suffixe + '.csv', sep = ';', index = False)
