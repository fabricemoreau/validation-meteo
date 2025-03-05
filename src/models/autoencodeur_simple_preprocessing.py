import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import joblib

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
parametres_origine = [ param + '_origine' for param in parametres ]

periode = '2010-2024'
path    = './data/processed/autoencsimple'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')

## On se limite dans un premier temps à 2010-2022: éviter les nouvelles stations météo
meteobydate = meteobydate[meteobydate.datemesure.dt.year.isin(range(2010, 2023))]

#meteobydate['timestamp'] = meteobydate.datemesure.astype(int)

meteobydate["month_sin"] = np.sin(2 * np.pi * meteobydate.datemesure.dt.month / 12)
meteobydate["month_cos"] = np.cos(2 * np.pi * meteobydate.datemesure.dt.month / 12)


meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'month_sin', 'month_cos', 'jourjulien']

# séparation train-test: par les années: on garde les années les plus récentes pour le test
# séparation train-test: par les stations : on garde 10% pour le test
stations_rand = meteobydate.codearvalis.unique()
np.random.shuffle(stations_rand)
meteobydate['test'] = meteobydate.codearvalis.isin(stations_rand[(len(stations_rand) * 9 // 10): ]) | meteobydate.datemesure.dt.year.isin([2021, 2022])
print("on veut max 30% de données de test")
print(meteobydate.test.value_counts(normalize = True))

# On normalise avant transformation du jeu de données: c'est plus simple. Cela normalise aussi les variables à prédire!
scaler = MinMaxScaler()
meteobydate.loc[meteobydate.test == False, parametres + meta_features] = scaler.fit_transform(meteobydate.loc[meteobydate.test == False, parametres + meta_features])
meteobydate.loc[meteobydate.test == True, parametres + meta_features]  = scaler.transform    (meteobydate.loc[meteobydate.test == True, parametres + meta_features])

anomalies_true = (meteobydate[[param + '_anomalie' for param in parametres]] > 0)
threshold = meteobydate[[param + '_origine' for param in parametres]].std() * 0.1

for param in parametres:
    meteobydate[param + '_anomalie_threshold'] = np.where(np.abs(meteobydate[param + '_difference']) > threshold[param + '_origine'], 1, 0)

X_train = meteobydate.loc[meteobydate.test == False, parametres + meta_features]
X_test  = meteobydate.loc[meteobydate.test == True, parametres + meta_features]
# ordre aléatoire pour train
X_train = shuffle(X_train)

################################

## jeux d'évaluation
## A terminer
balanced_df = meteobydate[meteobydate.test == True]
balanced_df = balanced_df.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
# on doit normaliser les paramètres d'origine, mais les meta_features sont déjà normalisées.
# on les normalise une seconde fois, sans les utiliser
balanced_normalized = scaler.transform(balanced_df[parametres_origine + meta_features].rename(columns = dict(zip(parametres_origine, parametres))))
balanced_df.loc[:,parametres_origine] = balanced_normalized[:,0:len(parametres)]
for param in parametres:
    df_param  = balanced_df[balanced_df[param + '_anomalie_threshold'] == 1]
    nb_anomalies = (df_param[param + '_anomalie_threshold'] == 1).sum()
    balanced_df_param = pd.concat(
        [balanced_df[balanced_df[param + '_anomalie_threshold'] == 0].sample(nb_anomalies),
         df_param
        ])
    balanced_df_param.to_csv(path +  '/pd_balanced_df_' + param + '.csv', sep = ';', index = False)


# A REVOIR:
# fichier de validation
data_valid = meteobydate.loc[meteobydate.test == True]
data_valid.loc[:,parametres + meta_features] = scaler.inverse_transform(data_valid[parametres + meta_features])
data_valid[data_valid.test == True].to_csv(path + '/meteobydate_valid.csv', sep = ';', index = False)

X_valid = scaler.transform(data_valid[parametres_origine + meta_features].rename(columns = dict(zip(parametres_origine, parametres))))



np.save(path + '/np_xtrain.npy', X_train)
np.save(path + '/np_xtest.npy', X_test)
np.save(path + '/np_xvalid.npy', X_valid)
joblib.dump(scaler, path + '/joblib_scaler.gz')
