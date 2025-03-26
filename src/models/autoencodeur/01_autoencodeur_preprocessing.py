import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

#import matplotlib.pyplot as plt
import joblib

# vocabulaire
# train: jeu d'entrainement de l'autoencodeur : sur données corrigées
# test: jeu d'évaluation de l'autoencodeur: sur donnés corrigées: objectif: vérifier que l'autoencodeur ne détecte pas d'anomalie
# val : jeu d'évaluation de la détection d'anomalie: sur données brutes non utilisées pendant l'entrainement
# balanced_val_param : jeu d'évaluation de la détection d'anomalie équilibré. Sur données brutest non utilisées pendant l'entrainement
# val_recentes : jeu d'évaluation non utilisé par train, test et val: pour vérifier le comportement de l'autoencodeur sur des données non vues plus récentes

method  = 'autoencoder'
path    = './data/processed/' + method
SEUIL_ANOMALIE = 0.1
fichier = f"data/processed/meteo_pivot_cleaned_2010-2024_{SEUIL_ANOMALIE}.csv" 
parametres =  ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant # ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
## On se limite dans un premier temps à 2010-2022: éviter les nouvelles stations météo
train_years = [a for a in range(2010, 2021)]
# séparation train-test: par les années: on garde les années les plus récentes pour le test
test_years = [a for a in range(2021, 2023)]
# proportion de station météo complètes pour le test: on garde 10% pour le test
stations_split_test = 0.1

fichier_suffixe = '-'.join(parametres) + '_' + str(np.min(train_years + test_years)) + '-' + str(np.max(train_years + test_years)) + '-' + str(SEUIL_ANOMALIE)

meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']

parametres_origine = [ param + '_origine' for param in parametres ]


meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = ['datemesure'])

# fin de à déplacer en preprocessing
meteobydate['anomaly']  = np.where(meteobydate['anomaly'] > 0, 1, 0)
# fin de à déplacer en preprocessing

test_recentes = meteobydate[~meteobydate.datemesure.dt.year.isin(train_years + test_years)]
meteobydate = meteobydate[meteobydate.datemesure.dt.year.isin(train_years + test_years)]

# séparation données de test
stations_rand = meteobydate.codearvalis.unique()
# on tire un échantillon alétoire des stations pour le test
np.random.shuffle(stations_rand)
nb_stations_test = int(np.round(len(stations_rand) * stations_split_test))
meteobydate['test'] = meteobydate.codearvalis.isin(stations_rand[:nb_stations_test]) | meteobydate.datemesure.dt.year.isin(test_years)
print("Proportions de données de test:")
print(meteobydate.test.value_counts(normalize = True))

val = meteobydate[meteobydate.test].drop(columns = 'test')
# on mélange les données d'entrainement pour éviter d'avoir trop d'homogénéité dans les batchs
train = meteobydate[~meteobydate.test].sample(frac=1).drop(columns = 'test')
# pour faciliter les traitements, on renomme les colonnes des données val pour correspondre à celles utilisées pour l'entrainement
# les colonnes 'param' sont renommées en 'param_corrige
# les colonnes 'param_origine' sont renommées en 'param'
test = val
test = test.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
test = test.rename(columns = dict(zip(parametres_origine, parametres)))
test_recentes = test_recentes.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
test_recentes = test_recentes.rename(columns = dict(zip(parametres_origine, parametres)))

# on enregistre
train.to_csv(path + '/train_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
val.to_csv(path + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
test.to_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
test_recentes.to_csv(path + '/test_recentes_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)

# On normalise avant transformation du jeu de données: c'est plus simple. Cela normalise aussi les variables à prédire!
scaler_meta = MinMaxScaler()
train[meta_features] = scaler_meta.fit_transform(train[meta_features])
val[meta_features] = scaler_meta.transform(val[meta_features])
test[meta_features] = scaler_meta.transform(test[meta_features])
test_recentes[meta_features] = scaler_meta.transform(test_recentes[meta_features])

scaler_param = MinMaxScaler()
train[parametres] = scaler_param.fit_transform(train[parametres])
val[parametres] = scaler_param.transform(val[parametres])
test[parametres] = scaler_param.transform(test[parametres])
test_recentes[parametres] = scaler_param.transform(test_recentes[parametres])

X_train = train[parametres + meta_features]
X_val  = val[parametres + meta_features]
X_test   = test[parametres + meta_features]
X_test_recentes = test_recentes[parametres + meta_features]

np.save(path + '/np_xtrain_' + fichier_suffixe + '.npy', X_train)
np.save(path + '/np_xval_' + fichier_suffixe + '.npy', X_test)
np.save(path + '/np_xtest_' + fichier_suffixe + '.npy', X_val)
np.save(path + '/np_xtest_recentes_' + fichier_suffixe + '.npy', X_test_recentes)
joblib.dump(scaler_param, path + '/joblib_scaler_param_' + fichier_suffixe + '.gz')


test = pd.read_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
test[parametres] = scaler_param.transform(test[parametres])

# on recherche les valeurs minimales de correction (décile 1 pour retirer les cas extrêmes)
test[[param + '_corrige' for param in parametres]] = scaler_param.transform(test[[param + '_corrige' for param in parametres]].rename(columns = dict(zip([param + '_corrige' for param in parametres], parametres))))
quantiles_corrections = []
for param in parametres:
    correction = np.abs(test[param + '_corrige'] - val[param])
    quantiles_corrections.append(correction.loc[correction > 0].quantile([0.1])[0.1])
print(quantiles_corrections)
# [0.031959629941127, 0.015307964473969246, 0.007352941176470562, 0.0034602076124566894]
# le 25/03/2025: [0.02607232968881415, 0.036175897176691396, 0.022058823529411797, 0.029616724738675937]

################################


