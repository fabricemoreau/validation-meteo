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
X_valid = np.load(path + '/np_xvalid.npy')

scaler = joblib.load(path + '/joblib_scaler.gz')

nfeatures = X_train.shape[1]
noutputs = len(parametres)

# on ne veut prédire que les paramètres
y_train = X_train[:,0:noutputs]
y_test = X_test[:,0:noutputs]
y_valid = X_valid[:,0:noutputs]

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
checkpoint = ModelCheckpoint(path + '/autoencodeur_checkpoint.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min')
history = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test),
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint],
                    batch_size=128, epochs=10)
# {'loss': [0.0008082717540673912, 0.0003462800523266196, 0.0003326582664158195, 0.00021125722560100257, 0.00017357846081722528, 0.00014328108227346092, 0.00014004007971379906, 0.0001383357448503375, 0.0001377734588459134, 0.00013651311746798456], 'mae': [0.016431264579296112, 0.014181971549987793, 0.014050103724002838, 0.01128819677978754, 0.01025001984089613, 0.009327085688710213, 0.009226457215845585, 0.009161440655589104, 0.00914173573255539, 0.009113244712352753], 'val_loss': [6.899613072164357e-05, 8.045606955420226e-05, 0.00039169786032289267, 1.1312834431009833e-05, 1.3247892638901249e-05, 5.759347004641313e-06, 6.653756372543285e-06, 5.6466637943231035e-06, 6.190084150148323e-06, 5.667076493409695e-06], 'val_mae': [0.006345141679048538, 0.007597834337502718, 0.01580004021525383, 0.0026154748629778624, 0.0027375533245503902, 0.001679902197793126, 0.0019450184190645814, 0.0017003501998260617, 0.0018145621288567781, 0.0016685528680682182], 'learning_rate': [0.009999999776482582, 0.009999999776482582, 0.009999999776482582, 0.0009999999310821295, 0.0009999999310821295, 9.99999901978299e-05, 9.99999901978299e-05, 9.99999883788405e-06, 9.99999883788405e-06, 9.99999883788405e-07]}
model = load_model(path + '/autoencodeur_checkpoint.keras')
#model = load_model(path + '/best_training.keras')

X_train_pred = model.predict(X_train)
train_mae_loss = np.abs(X_train_pred - y_train)

# Get reconstruction loss threshold.
from sklearn.metrics import mean_absolute_error
threshold = mean_absolute_error(y_train, X_train_pred)

threshold = np.max(train_mae_loss, axis = 0)
#array([0.02504609, 0.03185712, 0.01667732, 0.03510014])
# on accepte 25 d'erreur
threshold = np.quantile(train_mae_loss, 0.75, axis = 0)

plt.figure()
plt.hist(train_mae_loss, bins=50)
#plt.plot(threshold[0], [300], label = "threshold 0")
#plt.plot(threshold[1], [300], label = "threshold 1")
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/train_mae_loss.png')
plt.show()

# reconstruction 1ere séquence

# affichage d'une séquence temporelle
def graph_sequence(data: np.ndarray, filename: str, data_predicted : np.ndarray = None, data_fixed : np.ndarray = None):
    fig = plt.figure(figsize = (10, 10))
    for i, param in enumerate(parametres):
        ax1 = fig.add_subplot(2,2,i + 1)
        ax1.plot(data[i], label = parametres[i] + ' observé')
        if data_predicted is not None:
            ax1.plot(data_predicted[i], linestyle = 'dashed', marker='o', label = parametres[i] + ' prédit')
        if data_fixed is not None:
            ax1.plot(data_fixed[i], linestyle = 'dashed', marker='o', label = parametres[i] + ' corrigé')
        ax1.set_xlabel("temps (jour)")
        ax1.set_ylabel(parametres[i])
        ax1.set_title(parametres[i])
    fig.legend()
    fig.savefig(path + '/' + filename)
    fig.show();

graph_sequence(y_train[0], "sequence_1.png", X_train_pred[0])
#graph_sequence(scaler.inverse_transform(X_train[0]), 'sequence_1_denormalise.png', scaler.inverse_transform(X_train_pred[0]))

X_test_pred = model.predict(X_test)
test_mae_loss_param = np.abs(X_test_pred - y_test)


plt.figure()
plt.hist(test_mae_loss_param, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.savefig(path + '/test_mae_loss.png')
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
for i, param in enumerate(parametres):
    print(param)
    balanced_df_param = pd.read_csv(path +  '/pd_balanced_df_' + param + '.csv', sep = ';')
    X_test_pred_param= model.predict(balanced_df_param[parametres_origine + meta_features])
    test_mae_loss_balanced = np.abs(X_test_pred_param - balanced_df_param[parametres_origine])
    balanced_df_param[param + '_anomalie_pred'] = np.where( test_mae_loss_balanced[param + '_origine'] > threshold[i], 1, 0)
    balanced_df_param[param + '_pred'] = X_test_pred_param[:,i]
    print(pd.crosstab(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))
    print(classification_report(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))
    
    

##### validation
X_valid_pred = model.predict(X_valid)
valid_mae_loss_param = np.abs(X_valid_pred - y_valid)
anomalie_valid = valid_mae_loss_param > threshold
X_valid_pred_unscaled = scaler.inverse_transform( np.concatenate((X_valid_pred, X_valid[:, noutputs:]), axis = 1) )
valid_df = pd.read_csv(path +  '/meteobydate_valid.csv', sep = ';')

for i, param in enumerate(parametres):
    valid_df[param + '_anomalie_pred'] = np.where( valid_mae_loss_param[:,i] > threshold[i], 1, 0)
    valid_df[param + '_pred'] = X_valid_pred_unscaled[:,i]
    print(pd.crosstab(valid_df[param + '_anomalie_threshold'], valid_df[param + '_anomalie_pred']))
    print(classification_report(valid_df[param + '_anomalie_threshold'], valid_df[param + '_anomalie_pred']))
   
valid_df.to_csv(path + '/meteobydate_valid_pred.csv', sep = ';', index = False)

## A continuer:
# - normaliser avec valeurs min max par paramètre absolue: (X_test_pred_param < 0).sum(axis = 0)
#- confronter à anomalies réelles: définir seuil: rmse, plusieurs fois le threshold 
# analyser les FP FN: quelle différence entre prédiction et valeur origine, valeur corrigée
#- ajouter pluie?
#- regarder années 2023-2024
#-optimiser: ajouter des dropouts?
#-modèle pour détecter anomalie à partir de sortie autoencodeur!
