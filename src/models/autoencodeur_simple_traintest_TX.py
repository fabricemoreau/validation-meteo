""" autoencodeur pour un plusieurs paramètres"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, BatchNormalization
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
path    = './data/processed/autoencsimple'

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
parametres_origine = [ param + '_origine' for param in parametres ]

X_train = np.load(path + '/np_xtrain.npy')
X_test = np.load(path + '/np_xtest.npy')

nfeatures = X_train.shape[1]
noutputs = 1 

# on ne veut prédire que les paramètres
y_train = X_train[:,3]
y_test = X_test[:,3]

# define model
inputs = Input(shape=(nfeatures, ), name = 'input')
e = Dense(nfeatures *2, name = 'encoder_l1')(inputs)
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

#from tensorflow.keras.utils import plot_model
#plot_model(model, 'autoencoder_no_compress.png', show_shapes=True)

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 1E-7,
                           patience = 3,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 1E-5,
                               patience = 2,
                               factor = 0.1, 
                               #cooldown = 2,						
                               verbose = 1)
checkpoint = ModelCheckpoint(path + '/autoencodeur_checkpoint_TX.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min')
history = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test),
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint],
                    batch_size=128, epochs=10)
# {'loss': [0.0101701645180583, 0.00024426658637821674, 0.00023483976838178933, 0.00022908119717612863, 0.0002036402584053576, 0.00020271488756407052, 0.00019912216521333903, 0.0002007012808462605, 0.00020152806246187538], 'mae': [0.025244874879717827, 0.012297571636736393, 0.012052100151777267, 0.011858215555548668, 0.01112479716539383, 0.011071891523897648, 0.010977499186992645, 0.011039038188755512, 0.011032248847186565], 'val_loss': [0.0007970977458171546, 7.2510342761233915e-06, 0.0001416994637111202, 1.344024894933682e-05, 6.422848855436314e-06, 2.1580267457466107e-06, 3.859433945763158e-06, 2.9801872187817935e-06, 6.723072601744207e-06], 'val_mae': [0.025884319096803665, 0.0020621162839233875, 0.009204559028148651, 0.0029124843422323465, 0.0022207044530659914, 0.001071290229447186, 0.0015736239729449153, 0.0013886259403079748, 0.0021840352565050125], 'learning_rate': [0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.0010000000474974513, 0.00010000000474974513, 0.00010000000474974513, 1.0000000656873453e-05, 1.0000000656873453e-05, 1.0000001111620804e-06]}
model = load_model(path + '/autoencodeur_checkpoint_TX.keras')

X_train_pred = model.predict(X_train)
train_mae_loss = np.abs(X_train_pred - y_train.reshape(-1, 1))

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss, axis = 0)
#array([0.02504609, 0.03185712, 0.01667732, 0.03510014])
# on accepte 10% de faux positifs
threshold = np.quantile(train_mae_loss, 0.75, axis = 0)

plt.figure()
plt.hist(train_mae_loss, bins=50)
#plt.plot(threshold[0], [300], label = "threshold 0")
#plt.plot(threshold[1], [300], label = "threshold 1")
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/train_mae_loss_TX.png')
plt.show()

# reconstruction 1ere séquence
X_test_pred = model.predict(X_test)
test_mae_loss_param = np.abs(X_test_pred - y_test.reshape(-1, 1))


plt.figure()
plt.hist(test_mae_loss_param, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/test_mae_loss_TX.png')
plt.show()


# Detect all the samples which are anomalies.
anomalies = test_mae_loss_param > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anom_test = [[False] * noutputs] *  X_test.shape[0]

pd.crosstab([False] * X_test.shape[0], anomalies[:, 0].tolist())
pd.crosstab([False] * X_test.shape[0], anomalies[:, 1].tolist())
pd.crosstab([False] * X_test.shape[0], anomalies[:, 2].tolist())
pd.crosstab([False] * X_test.shape[0], anomalies[:, 3].tolist())


meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'month_sin', 'month_cos', 'jourjulien']

### problème standardisation lambert, altitude, joursjuliens
i =3
param = 'TX'

print(param)
balanced_df_param = pd.read_csv(path +  '/pd_balanced_df_' + param + '.csv', sep = ';')
X_test_pred_param= model.predict(balanced_df_param[parametres_origine + meta_features])
test_mae_loss_balanced = np.abs(X_test_pred_param - balanced_df_param[parametres_origine])
balanced_df_param[param + '_anomalie_pred'] = np.where( test_mae_loss_balanced[param + '_origine'] > threshold[0] / 2, 1, 0)
balanced_df_param[param + '_pred'] = X_test_pred_param[:,0]
print(pd.crosstab(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))
print(classification_report(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))    
