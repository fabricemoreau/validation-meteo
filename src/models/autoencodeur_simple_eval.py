import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import plot_model


from src.models.layer_mygaussiannoise import MyGaussianNoise

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support

method  = 'autoencoder'
path    = './data/processed/' + method

parametres = ['ETP', 'GLOT', 'TN', 'TX'] 

fichier_suffixe = '-'.join(parametres) + '_' + str(2010) + '-' + str(2022) + '-' + str(0.1)

# les colonnes doivent être dans l'ordre: parametres + meta_features
X_train = np.load(path + '/np_xtrain_' + fichier_suffixe + '.npy')
X_val = np.load(path + '/np_xval_' + fichier_suffixe + '.npy')
X_test = np.load(path + '/np_xtest_' + fichier_suffixe + '.npy')
X_test_recentes = np.load(path + '/np_xtest_recentes_' + fichier_suffixe + '.npy')

nfeatures = X_train.shape[1]
noutputs = len(parametres)

# on ne veut prédire que les paramètres, pas les meta_features
y_train = X_train[:,0:noutputs]
y_val = X_val[:,0:noutputs]
y_test = X_test[:,0:noutputs]
y_test_recentes = X_test_recentes[:,0:noutputs]


scaler_param = joblib.load(path + '/joblib_scaler_param_' + fichier_suffixe + '.gz')
# attention à /autoencodeur_reduce_checkpoint_
fichier_suffixe_ori = fichier_suffixe
fichier_suffixe += '_gaussiannoise'
model = load_model(path + '/autoencodeur_checkpoint_' + fichier_suffixe + '.keras')

plot_model(model, path + '/autoencoder_schema_ ' + fichier_suffixe + '.png', show_shapes=True)

X_train_pred = model.predict(X_train)
train_ae_loss = np.abs(X_train_pred - y_train)
#train_se_loss = (X_train_pred - y_train)**2

# Get reconstruction loss threshold.
#from sklearn.metrics import mean_absolute_error
#threshold = mean_absolute_error(y_train, X_train_pred)

# objectif, avoir un seuil inférieur à l'écart type
print("ecart type", y_train.std(axis = 0))
# seuil sécuritaire
threshold = np.max(train_ae_loss, axis = 0)
print("seuil d'anomalie", threshold)
#array([0.02504609, 0.03185712, 0.01667732, 0.03510014])
# on accepte 10 d'erreur
#threshold = np.quantile(train_ae_loss, 0.98, axis = 0)

plt.figure(figsize=(15, 15))
for i, param in enumerate(parametres):
    ax = plt.subplot(2, 2, i + 1)
    ax.hist(train_ae_loss, bins=50, label = parametres)
    ax.axvline(threshold[i], ymax = ax.get_ylim()[1] * 0.9, linestyle = ':', color = 'red')
    ax.text(threshold[i], ax.get_ylim()[1] * 0.9, 'Seuil anomalie max', size = 11, color = 'red')
    ax.axvline(np.quantile(train_ae_loss, 0.9, axis = 0)[i], ymax = ax.get_ylim()[1] * 0.7, linestyle = ':', color = 'purple')
    ax.text(np.quantile(train_ae_loss, 0.98, axis = 0)[i], ax.get_ylim()[1] * 0.7, 'Seuil anomalie décile 9', size = 11, color = 'purple')
    
    ax.set_xlabel("Train Absolute Error loss")
    ax.set_ylabel("No of samples")
    ax.set_title("seuils définis par rapport aux résidus du jeu d'entrainement")
plt.savefig(path + '/train_ae_loss_' + fichier_suffixe + '.png')
#plt.show()

plt.figure(figsize=(15,15))
for i, param in enumerate(parametres):
    ax = plt.subplot(2, 2, i +1)
    ax.plot(y_train[:, i], X_train_pred[:, i], '.')
    ax.title.set_text(param)
plt.savefig(path + '/residus_' + fichier_suffixe + '.png')
#plt.show()

"""
### alternative: calculer erreur cumulée de chaque séquence
# https://www.tensorflow.org/tutorials/generative/autoencoder?hl=fr
# https://anomagram.fastforwardlabs.com/#/
import tensorflow as tf
train_loss = tf.keras.losses.mae(y_train, X_train_pred)

# choisir un seuil
#threshold_total = np.mean(train_loss) + np.std(train_loss)
threshold_total = np.quantile(train_loss, 0.9) # on accepte 10% de faux positifs
print("Threshold total: ", threshold_total)

plt.figure()
plt.hist(train_loss[None,:], bins=50)
plt.axvline(threshold_total, ymax = 10000, linestyle = ':', color = 'red')
plt.text(threshold_total, 10000, 'Seuil anomalie', size = 11, color = 'red')
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.legend()
plt.savefig(path + '/train_total_loss_' + fichier_suffixe + '.png')
plt.show()
"""



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

# A améliorer: faire graph sur toutes les données!
graph_sequence(y_train[0], 'sequence_1_' + fichier_suffixe + '.png', X_train_pred[0])
graph_sequence(y_train[38440], 'sequence_38441_' + fichier_suffixe + '.png', X_train_pred[38440])
graph_sequence(scaler_param.inverse_transform(y_train[0].reshape(1, -1)).reshape(-1), 'sequence_1_denormalise_' + fichier_suffixe  +'.png', scaler_param.inverse_transform(X_train_pred[0].reshape(1, -1)).reshape(-1))

X_val_pred = model.predict(X_val)
val_ae_loss_param = np.abs(X_val_pred - y_test)


plt.figure(figsize=(15,15))
for i, param in enumerate(parametres):
    ax = plt.subplot(2, 2, i + 1)
    ax.hist(val_ae_loss_param, bins=50, label = parametres)
    ax.axvline(threshold[i], ymax = ax.get_ylim()[1] * 0.9, linestyle = ':', color = 'red')
    ax.text(threshold[i], ax.get_ylim()[1] * 0.9, 'Seuil anomalie max', size = 11, color = 'red')
    ax.axvline(np.quantile(train_ae_loss, 0.9, axis = 0)[i], ymax = ax.get_ylim()[1] * 0.7, linestyle = ':', color = 'purple')
    ax.text(np.quantile(train_ae_loss, 0.98, axis = 0)[i], ax.get_ylim()[1] * 0.7, 'Seuil anomalie décile 9', size = 11, color = 'purple')
    ax.set_xlabel("Train Absolute Error loss")
    ax.set_ylabel("No of samples")
    ax.set_title("seuils définis par rapport aux résidus du jeu de validation")

plt.savefig(path + '/val_ae_loss_' + fichier_suffixe + '.png')
#plt.show()


# Detect all the samples which are anomalies.
anomalies = val_ae_loss_param > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
for i, param in enumerate(parametres):
    print(param)
    print(pd.crosstab([False] * X_val.shape[0], anomalies[:, i].tolist()))
    
"""
## anomalies from total threshold
val_loss = tf.keras.losses.mae(y_test, X_val_pred)
plt.figure()
plt.hist(val_loss, bins=50)
plt.axvline(threshold_total, ymax = 10000, linestyle = ':', color = 'red')
plt.text(threshold_total, 10000, 'Seuil anomalie', size = 11, color = 'red')
plt.xlabel("Val loss")
plt.ylabel("No of samples")
plt.savefig(path + '/val_total_loss_' + fichier_suffixe + '.png')
plt.show()

# test seuils
for quant in [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
    print("quantile", quant)
    threshold_total = np.quantile(val_loss, quant)
    anomalies_total = train_loss > threshold_total
    print("Number of anomaly samples: ", np.sum(anomalies_total))
    print("Indices of anomaly samples: ", np.where(anomalies_total))
    print(pd.crosstab([False] * X_val.shape[0], anomalies_total.numpy().tolist()))
    #print(pd.crosstab([False] * X_val.shape[0], anomalies_total.numpy().tolist()))
"""    



##### validation
X_test_pred = model.predict(X_test)
test_ae_loss_param = np.abs(X_test_pred - y_val)
for i in np.arange(1, 0.895, -0.005): # [1]:
    threshold = np.quantile(train_ae_loss, i, axis = 0)
    anomalie_test = pd.DataFrame(np.where(test_ae_loss_param > threshold, 1, 0), columns = [param + '_anomaly_pred' for param in parametres])
    anomalie_test['anomaly_pred'] = np.where(anomalie_test[[param + '_anomaly_pred' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    X_test_pred_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_test_pred), columns = [param + '_pred' for param in parametres])
    test = pd.read_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
    test = pd.concat([test, anomalie_test, X_test_pred_unscaled], axis = 1)
    test['anomaly'] = np.where(test[[param + '_anomaly' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    print("quantile seuil:", i)
    print(pd.crosstab(test['anomaly'], test['anomaly_pred']))
    print(classification_report(test['anomaly'], test['anomaly_pred']))
    print('precision', precision_score(test['anomaly'], test['anomaly_pred'], zero_division=0))
    print('recall', recall_score(test['anomaly'], test['anomaly_pred'], zero_division=0))
    print(f1_score(test['anomaly'], test['anomaly_pred']))
    #test.to_csv(path + '/test_result_' + fichier_suffixe + '_' + str(i) + '.csv', sep = ';', index = False)


test['is_test'] = 1

test.rename(columns = dict(zip(parametres, [param + '_origine' for param in parametres]))).to_csv(path + '/test_result_' + fichier_suffixe + '_' + str(i) + '.csv', index = False) # sep = ';', retiré pour compatibilité avec meteo_model

### l'autoencodeur ne prédit pas les anomalies! Il semble s'appuyer trop sur les données fournies
FN = test.loc[(test.anomaly == 1) & (test.anomaly_pred == 0), parametres + [param +'_pred' for param in parametres] + [param + '_corrige' for param in parametres]]


"""
## validation totale
test_loss = tf.keras.losses.mae(y_test, X_test_pred)
anomalie_test_total = np.where(test_loss > threshold_total, 1, 0)
test = pd.read_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
#test['anomaly'] = np.where(test[[param + '_anomaly' for param in parametres]].sum(axis = 1) > 0, 1, 0)
test['anomaly_pred'] = anomalie_test_total
print(pd.crosstab(test['anomaly'], test['anomaly_pred']))
print(classification_report(test['anomaly'], test['anomaly_pred']))
print('precision', precision_score(test['anomaly'], test['anomaly_pred'], zero_division=0))
print('recall', recall_score(test['anomaly'], test['anomaly_pred'], zero_division=0))
print(f1_score(test['anomaly'], test['anomaly_pred']))
"""

##validation_recentes
#threshold = np.max(train_ae_loss, axis = 0)
X_test_recentes_pred = model.predict(X_test_recentes)
test_recentes_ae_loss_param = np.abs(X_test_recentes_pred - y_test_recentes)

for i in np.arange(1, 0.895, -0.005): # [1]:
    threshold = np.quantile(train_ae_loss, i, axis = 0)
    anomalie_test_recentes = pd.DataFrame(np.where(test_recentes_ae_loss_param > threshold, 1, 0), columns = [param + '_anomaly_pred' for param in parametres])
    anomalie_test_recentes['anomaly_pred'] = np.where(anomalie_test_recentes[[param + '_anomaly_pred' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    X_test_pred_recentes_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_test_recentes_pred), columns = [param + '_pred' for param in parametres])
    test_recentes = pd.read_csv(path + '/test_recentes_preprocessed_' + fichier_suffixe_ori + '.csv', sep = ';')
    test_recentes = pd.concat([test_recentes, anomalie_test_recentes, X_test_pred_recentes_unscaled], axis = 1)
    #test_recentes['anomaly'] = np.where(test_recentes[[param + 'anomaly' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    print("quantile seuil:", i)
    print(pd.crosstab(test_recentes['anomaly'], test_recentes['anomaly_pred']))
    print(classification_report(test_recentes['anomaly'], test_recentes['anomaly_pred']))

## etude des faux négatifs: on prend le seuil à 0.95
threshold = np.quantile(train_ae_loss, 0.95, axis = 0)
anomalie_test_recentes = pd.DataFrame(np.where(test_recentes_ae_loss_param > threshold, 1, 0), columns = [param + '_anomaly_pred' for param in parametres])
anomalie_test_recentes['anomaly_pred'] = np.where(anomalie_test_recentes[[param + '_anomaly_pred' for param in parametres]].sum(axis = 1) > 0, 1, 0)
X_test_pred_recentes_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_test_recentes_pred), columns = [param + '_pred' for param in parametres])
test_recentes = pd.read_csv(path + '/test_recentes_preprocessed_' + fichier_suffixe_ori + '.csv', sep = ';')
test_recentes = pd.concat([test_recentes, anomalie_test_recentes, X_test_pred_recentes_unscaled], axis = 1)

FN = test_recentes[(test_recentes.anomaly == 1) & (test_recentes.anomaly_pred == 0)][[param + '_pred' for param in parametres] + parametres + [param + '_difference' for param in parametres]]
FN_residus = np.abs(FN[parametres].values - FN[[param + '_pred' for param in parametres]].values)
FN_residus = pd.DataFrame(FN_residus, columns = parametres)
print("l'autoencodeur reproduit les anomalies!!")
FN_residus.max()
np.abs(FN[[param + '_difference' for param in parametres]]).max()