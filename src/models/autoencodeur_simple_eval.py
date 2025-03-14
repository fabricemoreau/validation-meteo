import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 
from tensorflow.keras.utils import plot_model


#from layer_mygaussiannoise import MyGaussianNoise

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support

method  = 'autoencsimple'
path    = './data/processed/' + method

parametres = ['ETP', 'GLOT', 'TN', 'TX'] 

fichier_suffixe = '-'.join(parametres) + '_' + str(2010) + '-' + str(2022)

# les colonnes doivent être dans l'ordre: parametres + meta_features
X_train = np.load(path + '/np_xtrain_' + fichier_suffixe + '.npy')
X_test = np.load(path + '/np_xtest_' + fichier_suffixe + '.npy')
X_val = np.load(path + '/np_xval_' + fichier_suffixe + '.npy')
X_val_recentes = np.load(path + '/np_xval_recentes_' + fichier_suffixe + '.npy')

nfeatures = X_train.shape[1]
noutputs = len(parametres)

# on ne veut prédire que les paramètres, pas les meta_features
y_train = X_train[:,0:noutputs]
y_test = X_test[:,0:noutputs]
y_val = X_val[:,0:noutputs]
y_val_recentes = X_val_recentes[:,0:noutputs]


scaler_param = joblib.load(path + '/joblib_scaler_param_' + fichier_suffixe + '.gz')
# attention à /autoencodeur_reduce_checkpoint_
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

plt.figure()
plt.hist(train_ae_loss, bins=50, label = parametres)
#plt.plot(threshold[0], [300], label = "threshold 0")
#plt.plot(threshold[1], [300], label = "threshold 1")
plt.xlabel("Train Absolute Error loss")
plt.ylabel("No of samples")
plt.legend()
plt.savefig(path + '/train_ae_loss_' + fichier_suffixe + '.png')
plt.show()

### alternative: calculer erreur cumulée de chaque séquence
# https://www.tensorflow.org/tutorials/generative/autoencoder?hl=fr
# https://anomagram.fastforwardlabs.com/#/
import tensorflow as tf
train_loss = tf.keras.losses.mae(y_train, X_train_pred)

plt.figure()
plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.legend()
plt.savefig(path + '/train_total_loss_' + fichier_suffixe + '.png')
plt.show()
# choisir un seuil
threshold_total = np.mean(train_loss) + np.std(train_loss)
print("Threshold total: ", threshold_total)



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

X_test_pred = model.predict(X_test)
test_ae_loss_param = np.abs(X_test_pred - y_test)


plt.figure()
plt.hist(test_ae_loss_param, bins=50)
plt.xlabel("test Absolute Error loss")
plt.ylabel("No of samples")
plt.savefig(path + '/test_ae_loss_' + fichier_suffixe + '.png')
plt.show()


# Detect all the samples which are anomalies.
anomalies = test_ae_loss_param > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
for i, param in enumerate(parametres):
    print(param)
    print(pd.crosstab([False] * X_test.shape[0], anomalies[:, i].tolist()))
    
## anomalies from total threshold
train_loss = tf.keras.losses.mae(y_test, X_test_pred)
# test seuils
for quant in [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]:
    print("quantile", quant)
    threshold_total = np.quantile(train_loss, quant)
    anomalies_total = train_loss > threshold_total
    print("Number of anomaly samples: ", np.sum(anomalies_total))
    print("Indices of anomaly samples: ", np.where(anomalies_total))
    print(pd.crosstab([False] * X_test.shape[0], anomalies_total.numpy().tolist()))
    #print(pd.crosstab([False] * X_test.shape[0], anomalies_total.numpy().tolist()))
    
    


##### validation
X_val_pred = model.predict(X_val)
val_ae_loss_param = np.abs(X_val_pred - y_val)
for i in [1]: #np.arange(1, 0.9, -0.005): # [1]:
    threshold = np.quantile(train_ae_loss, i, axis = 0)
    anomalie_val = pd.DataFrame(np.where(val_ae_loss_param > threshold, 1, 0), columns = [param + '_anomalie_pred' for param in parametres])
    anomalie_val['anomalie_pred'] = np.where(anomalie_val[[param + '_anomalie_pred' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    X_val_pred_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_val_pred), columns = [param + '_pred' for param in parametres])
    val = pd.read_csv(path + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
    val = pd.concat([val, anomalie_val, X_val_pred_unscaled], axis = 1)
    val['anomalie_threshold'] = np.where(val[[param + '_anomalie_threshold' for param in parametres]].sum(axis = 1) > 0, 1, 0)
    print("quantile seuil:", i)
    print(pd.crosstab(val['anomalie_threshold'], val['anomalie_pred']))
    print(classification_report(val['anomalie_threshold'], val['anomalie_pred']))
    #val.to_csv(path + '/val_result_' + fichier_suffixe + '_' + str(i) + '.csv', sep = ';', index = False)

# analyse
FN = val.loc[(val.anomalie == 1) & (val.anomalie_pred == 0), parametres + [param +'_pred' for param in parametres] + [param + '_corrige' for param in parametres]]
### l'autoencodeur ne prédit pas les anomalies! Il semble s'appuyer trop sur les données fournies

## validation totale
val_loss = tf.keras.losses.mae(y_val, X_val_pred)
anomalie_val_total = np.where(val_loss > threshold_total, 1, 0)
val = pd.read_csv(path + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
val['anomalie_threshold'] = np.where(val[[param + '_anomalie_threshold' for param in parametres]].sum(axis = 1) > 0, 1, 0)
val['anomalie_pred'] = anomalie_val_total
print(pd.crosstab(val['anomalie_threshold'], val['anomalie_pred']))
print(classification_report(val['anomalie_threshold'], val['anomalie_pred']))


##validation_recentes
threshold = np.max(train_ae_loss, axis = 0)
X_val_recentes_pred = model.predict(X_val_recentes)
val_recentes_ae_loss_param = np.abs(X_val_recentes_pred - y_val_recentes)

anomalie_val_recentes = pd.DataFrame(np.where(val_recentes_ae_loss_param > threshold, 1, 0), columns = [param + '_anomalie_pred' for param in parametres])
anomalie_val_recentes['anomalie_pred'] = np.where(anomalie_val_recentes[[param + '_anomalie_pred' for param in parametres]].sum(axis = 1) > 0, 1, 0)
X_val_pred_recentes_unscaled = pd.DataFrame(scaler_param.inverse_transform(X_val_recentes_pred), columns = [param + '_pred' for param in parametres])
val_recentes = pd.read_csv(path + '/val_recentes_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
val_recentes = pd.concat([val_recentes, anomalie_val_recentes, X_val_pred_recentes_unscaled], axis = 1)
val_recentes['anomalie_threshold'] = np.where(val_recentes[[param + '_anomalie_threshold' for param in parametres]].sum(axis = 1) > 0, 1, 0)
print(pd.crosstab(val_recentes['anomalie'], val_recentes['anomalie_pred']))
print(classification_report(val_recentes['anomalie'], val_recentes['anomalie_pred']))


## A continuer:
# - normaliser avec valeurs min max par paramètre absolue: (X_test_pred_param < 0).sum(axis = 0)
#- confronter à anomalies réelles: définir seuil: rmse, plusieurs fois le threshold 
# analyser les FP FN: quelle différence entre prédiction et valeur origine, valeur corrigée
#- ajouter pluie?
#- regarder années 2023-2024
#-optimiser: ajouter des dropouts?
#-modèle pour détecter anomalie à partir de sortie autoencodeur!
# calculer RMSE

""""sur jeux équilibrés

meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'month_sin', 'month_cos', 'jourjulien']

### problème standardisation lambert, altitude, joursjuliens
parametres_origine = [ param + '_origine' for param in parametres ]
for i, param in enumerate(parametres):
    print(param)
    balanced_df_param = pd.read_csv(path +  '/pd_balanced_df_' + param + '.csv', sep = ';')
    X_test_pred_param= model.predict(balanced_df_param[parametres_origine + meta_features])
    test_mae_loss_balanced = np.abs(X_test_pred_param - balanced_df_param[parametres_origine])
    balanced_df_param[param + '_anomalie_pred'] = np.where( test_mae_loss_balanced[param + '_origine'] > threshold[i], 1, 0)
    balanced_df_param[param + '_pred'] = X_test_pred_param[:,i]
    print(pd.crosstab(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))
    print(classification_report(balanced_df_param[param + '_anomalie_threshold'], balanced_df_param[param + '_anomalie_pred']))
    
    
"""