
""" 
Test pour autoencodeur

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
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score


# les colonnes doivent être dans l'ordre: parametres + meta_features
X_train = np.load(PATH + '/np_xtrain_' + FICHIER_SUFFIXE + '.npy')
X_val = np.load(PATH + '/np_xval_' + FICHIER_SUFFIXE + '.npy')
X_test = np.load(PATH + '/np_xtest_' + FICHIER_SUFFIXE + '.npy')

nfeatures = X_train.shape[1]
noutputs = len(PARAMETRES)

# on ne veut prédire que les paramètres, pas les meta_features
y_train = X_train[:,0:noutputs]
y_val = X_val[:,0:noutputs]
y_test = X_test[:,0:noutputs]


scaler_param = joblib.load(PATH + '/joblib_scaler_param_' + FICHIER_SUFFIXE + '.gz')
model = load_model(PATH + '/autoencodeur_checkpoint_' + FICHIER_SUFFIXE + '.keras')

plot_model(model, PATH + '/autoencoder_schema_ ' + FICHIER_SUFFIXE + '.png', show_shapes=True)

X_train_pred = model.predict(X_train)
train_ae_loss = np.abs(X_train_pred - y_train)
# seuil sécuritaire
threshold = np.max(train_ae_loss, axis = 0)
print("seuil d'anomalie", threshold)

# graphique des résidus en fonction du seuil sécuritaire et du seuil décile 9
plt.figure(figsize=(15, 15))
for i, param in enumerate(PARAMETRES):
    ax = plt.subplot(2, 2, i + 1)
    ax.hist(train_ae_loss[:,i], bins=50, label = param)
    ax.axvline(threshold[i], ymax = ax.get_ylim()[1] * 0.9, linestyle = ':', color = 'red')
    ax.text(threshold[i], ax.get_ylim()[1] * 0.9, 'Seuil anomalie max', size = 11, color = 'red')
    ax.axvline(np.quantile(train_ae_loss[:,i], 0.9, axis = 0), ymax = ax.get_ylim()[1] * 0.7, linestyle = ':', color = 'purple')
    ax.text(np.quantile(train_ae_loss[:,i], 0.98, axis = 0), ax.get_ylim()[1] * 0.7, 'Seuil anomalie décile 9', size = 11, color = 'purple')
    
    ax.set_xlabel("Train Absolute Error loss")
    ax.set_ylabel("No of samples")
    ax.set_title("seuils définis par rapport aux résidus du jeu d'entrainement")
plt.savefig(PATH + '/train_ae_loss_' + FICHIER_SUFFIXE + '.png')
#plt.show()

# résidus
plt.figure(figsize=(15,15))
for i, param in enumerate(PARAMETRES):
    ax = plt.subplot(2, 2, i +1)
    ax.plot(y_train[:, i], X_train_pred[:, i], '.')
    ax.title.set_text(param)
plt.savefig(PATH + '/residus_' + FICHIER_SUFFIXE + '.png')
#plt.show()


# affichage d'une séquence temporelle
def graph_sequence(data: np.ndarray, filename: str, data_predicted : np.ndarray = None, data_fixed : np.ndarray = None):
    fig = plt.figure(figsize = (10, 10))
    for i, param in enumerate(PARAMETRES):
        ax1 = fig.add_subplot(2,2,i + 1)
        ax1.plot(data[i], linestyle = 'dashed', marker='o', label = param + ' observé')
        if data_predicted is not None:
            ax1.plot(data_predicted[i], linestyle = 'dashed', marker='x', label = param + ' prédit')
        if data_fixed is not None:
            ax1.plot(data_fixed[i], linestyle = 'dashed', marker='^', label = param + ' corrigé')
        ax1.set_xlabel("temps (jour)")
        ax1.set_ylabel(param)
        ax1.set_title(param)
    fig.legend()
    fig.savefig(PATH + '/' + filename)
    #fig.show();

# A améliorer: faire graph sur toutes les données!
graph_sequence(y_train[0], 'sequence_1_' + FICHIER_SUFFIXE + '.png', X_train_pred[0])
graph_sequence(scaler_param.inverse_transform(y_train[0].reshape(1, -1)).reshape(-1), 'sequence_1_denormalise_' + FICHIER_SUFFIXE  +'.png', scaler_param.inverse_transform(X_train_pred[0].reshape(1, -1)).reshape(-1))
# affichage du point d'erreur absolue maximale pour l'ETP
ae_etp_max = np.where(train_ae_loss[:, 0] >= threshold[0])[0]
graph_sequence(y_train[ae_etp_max].reshape(-1), 'sequence_ETP_aemax_' + FICHIER_SUFFIXE + '.png', X_train_pred[ae_etp_max].reshape(-1))
graph_sequence(scaler_param.inverse_transform(y_train[ae_etp_max].reshape(1, -1)).reshape(-1), 'sequence_ETP_aemax_denormalise_' + FICHIER_SUFFIXE  +'.png', scaler_param.inverse_transform(X_train_pred[ae_etp_max].reshape(1, -1)).reshape(-1))

# vérification sur jeu de validation
X_val_pred = model.predict(X_val)
val_ae_loss_param = np.abs(X_val_pred - y_val)

# calcul RMSE dénormalisée
rmse = sum((scaler_param.inverse_transform(y_val) - scaler_param.inverse_transform(X_val_pred))**2) / len(y_test)
print("rmse = ", np.round(rmse, 2))

plt.figure(figsize=(15,15))
for i, param in enumerate(PARAMETRES):
    ax = plt.subplot(2, 2, i + 1)
    ax.hist(val_ae_loss_param[:,1], bins=50, label = param)
    ax.axvline(threshold[i], ymax = ax.get_ylim()[1] * 0.9, linestyle = ':', color = 'red')
    ax.text(threshold[i], ax.get_ylim()[1] * 0.9, 'Seuil anomalie max', size = 11, color = 'red')
    ax.axvline(np.quantile(train_ae_loss, 0.9, axis = 0)[i], ymax = ax.get_ylim()[1] * 0.7, linestyle = ':', color = 'purple')
    ax.text(np.quantile(train_ae_loss, 0.98, axis = 0)[i], ax.get_ylim()[1] * 0.7, 'Seuil anomalie décile 9', size = 11, color = 'purple')
    ax.set_xlim()
    ax.set_xlabel("Train Absolute Error loss")
    ax.set_ylabel("No of samples")
    ax.set_title("seuils définis par rapport aux résidus du jeu de validation")
plt.savefig(PATH + '/val_ae_loss_' + FICHIER_SUFFIXE + '.png')
#plt.show()


# Detect all the samples which are anomalies.
anomalies = val_ae_loss_param > np.quantile(train_ae_loss, 0.9, axis = 0)
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
for i, param in enumerate(PARAMETRES):
    print(param)
    print(pd.crosstab([False] * X_val.shape[0], anomalies[:, i].tolist()))
    
##### test
X_test_pred = model.predict(X_test)
test_ae_loss_param = np.abs(X_test_pred - y_test)
for i in np.arange(1, 0.895, -0.005): # [1]:
    threshold = np.quantile(train_ae_loss, i, axis = 0)
    anomalie_test = pd.DataFrame(np.where(test_ae_loss_param > threshold, 1, 0), columns = [param + '_anomaly_pred' for param in PARAMETRES])
    anomalie_test['anomaly_pred'] = np.where(anomalie_test[[param + '_anomaly_pred' for param in PARAMETRES]].sum(axis = 1) > 0, 1, 0)
    X_test_pred_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_test_pred), columns = [param + '_pred' for param in PARAMETRES])
    test = pd.read_csv(PATH + '/test_preprocessed_' + FICHIER_SUFFIXE + '.csv', sep = ';')
    test = pd.concat([test, anomalie_test, X_test_pred_unscaled], axis = 1)
    test['anomaly'] = np.where(test[[param + '_anomaly' for param in PARAMETRES]].sum(axis = 1) > 0, 1, 0)
    print("quantile seuil:", i)
    print(pd.crosstab(test['anomaly'], test['anomaly_pred']))
    print(classification_report(test['anomaly'], test['anomaly_pred']))
    print('accuracy', accuracy_score(test['anomaly'], test['anomaly_pred']))
    print('recall', recall_score(test['anomaly'], test['anomaly_pred'], zero_division=0))
    print(f1_score(test['anomaly'], test['anomaly_pred']))
    #test.to_csv(PATH + '/test_result_' + FICHIER_SUFFIXE + '_' + str(i) + '.csv', sep = ';', index = False)

# sauvegarde des résultats
test.rename(columns = dict(zip(PARAMETRES, [param + '_origine' for param in PARAMETRES]))).to_csv(PATH + '/test_result_' + FICHIER_SUFFIXE + '_' + str(i) + '.csv', index = False) # sep = ';', retiré pour compatibilité avec meteo_model

FN = test[(test.anomaly == 1) & (test.anomaly_pred == 0)][[param + '_pred' for param in PARAMETRES] + PARAMETRES + [param + '_difference' for param in PARAMETRES]]
FN_residus = np.abs(FN[PARAMETRES].values - FN[[param + '_pred' for param in PARAMETRES]].values)
FN_residus = pd.DataFrame(FN_residus, columns = PARAMETRES)
print("l'autoencodeur reproduit les anomalies!!")
print(FN_residus.max(), np.abs(FN[[param + '_difference' for param in PARAMETRES]]).max())

FP = test[(test.anomaly == 0) & (v.anomaly_pred == 1)][[param + '_pred' for param in PARAMETRES] + PARAMETRES + [param + '_difference' for param in PARAMETRES]]
FP_residus = np.abs(FP[PARAMETRES].values - FP[[param + '_pred' for param in PARAMETRES]].values)
FP_residus = pd.DataFrame(FP_residus, columns = PARAMETRES)
print("l'autoencodeur reproduit les anomalies!!")
print(FP_residus.max(), np.abs(FP[[param + '_difference' for param in PARAMETRES]]).max())