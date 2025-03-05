import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
#from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import joblib

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 


parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
parametres_origine = [ param + '_origine' for param in parametres ]
nb_jours = 12 # il faut qu'il soit divisible par 4
seed = 42

path    = './data/processed/autoenc'
balanced_df_TN = pd.read_csv(path +  '/pd_balanced_df.csv', sep = ';', parse_dates = True)
balanced_df_TN.datemesure = pd.to_datetime(balanced_df_TN.datemesure).round('d')
# on ne fait pas de : cela sera géré dans la construction du jeu : meteobydate = meteobydate.dropna()
balanced_df_TN = balanced_df_TN.sort_values(['codearvalis', 'datemesure'])

X_test = np.load(path + '/np_xtest.npy')
indexes_test = np.load(path + '/np_indexestest.npy')
scaler = joblib.load(path + '/joblib_scaler.gz')

anomalies_dict = {}
for i, param in enumerate(parametres):
    anomalies_dict[param + '_anomalie_pred'] = np.load(path + '/anomalies_pred_test_' + param + '.npy').tolist()

model = load_model(path + '/autoencodeur_checkpoint.keras')

n_features = len(parametres)
nb_stations = len(balanced_df_TN.codearvalis.unique())


balanced_df_TN.loc[:,parametres_origine] = scaler.transform(balanced_df_TN[parametres_origine].rename(columns = dict(zip(parametres_origine, parametres))))


# affichage d'une séquence temporelle
def graph_sequence(data: np.ndarray, filename: str, data_predicted : np.ndarray = None, data_fixed : np.ndarray = None):
    fig = plt.figure(figsize = (10, 10))
    for i, param in enumerate(parametres):
        ax1 = fig.add_subplot(2,2,i + 1)
        ax1.plot(data[:,i], label = parametres[i] + ' observé')
        if data_predicted is not None:
            ax1.plot(data_predicted[:,i], linestyle = 'dashed', marker='o', label = parametres[i] + ' prédit')
        if data_fixed is not None:
            ax1.plot(data_fixed[:,i], linestyle = 'dashed', marker='o', label = parametres[i] + ' corrigé')
        ax1.set_xlabel("temps (jour)")
        ax1.set_ylabel(parametres[i])
        ax1.set_title(parametres[i])
    fig.legend()
    fig.savefig(path + '/' + filename)
    fig.show();
#graph_sequence(X_train[0], 'nom.png', X_train[0])

################################

# modifier la dernière valeur de chaque matrice de X_test par les valeurs de  param _origine:
#  boucle sur X_test. 
for i, matrice in enumerate(X_test):
#     On récupère indexes_test correspondant
#     On recherche, si elle existe, la valeur correspondante dans balanced_df_TN
#     On remplace par _origine
    if (balanced_df_TN.indexes_test == indexes_test[i]).sum() > 0:
        X_test[i][-1] = balanced_df_TN.loc[balanced_df_TN.indexes_test == indexes_test[i], parametres_origine]

# on fait une prédiction
X_test_pred = model.predict(X_test)

# on contrôle les prédictions > seuil => anomalies_pred
test_mae_loss_param = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_mae_loss = test_mae_loss_param.reshape((-1))

threshold = 0.024

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# on ajoute à balanced_df_TN

# on analyse en comparant balanced_df_TN.param_threshold avec balanced_df_TN.colonneajoutée

