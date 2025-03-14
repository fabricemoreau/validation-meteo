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

method  = 'autoencsimple'
path    = './data/processed/' + method
fichier = './data/processed/meteo_pivot_cleaned_2010-2024.csv' # doit contenir datetime sans heure, minute ou seconde

parametres =  ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant # ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
## On se limite dans un premier temps à 2010-2022: éviter les nouvelles stations météo
train_years = [a for a in range(2010, 2021)]
# séparation train-test: par les années: on garde les années les plus récentes pour le test
test_years = [a for a in range(2021, 2023)]
# proportion de station météo complètes pour le test: on garde 10% pour le test
stations_split_test = 0.1

fichier_suffixe = '-'.join(parametres) + '_' + str(np.min(train_years + test_years)) + '-' + str(np.max(train_years + test_years))

parametres_feature = ['GLOT', 'TN', 'TX'] # pas RR pour l'instant['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'month_sin', 'month_cos', 'jourjulien']
for param in parametres_feature:
    if param not in parametres:
        meta_features.append(param)

parametres_origine = [ param + '_origine' for param in parametres ]


meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = ['datemesure'])

# a déplacer en preprocessing
meteobydate["month_sin"] = np.sin(2 * np.pi * meteobydate.datemesure.dt.month / 12)
meteobydate["month_cos"] = np.cos(2 * np.pi * meteobydate.datemesure.dt.month / 12)

meteobydate['anomalie']  = np.where(meteobydate['anomalie'] > 0, 1, 0)
# on détermine un seuil mino d'anomalie par paramètre
threshold = meteobydate[[param + '_origine' for param in parametres]].std() * 0.1
for param in parametres:
    meteobydate[param + '_anomalie_threshold'] = np.where(np.abs(meteobydate[param + '_difference']) > threshold[param + '_origine'], 1, 0)
# fin de à déplacer en preprocessing

val_recentes = meteobydate[~meteobydate.datemesure.dt.year.isin(train_years + test_years)]
meteobydate = meteobydate[meteobydate.datemesure.dt.year.isin(train_years + test_years)]

# séparation données de test
stations_rand = meteobydate.codearvalis.unique()
# on tire un échantillon alétoire des stations pour le test
np.random.shuffle(stations_rand)
nb_stations_test = int(np.round(len(stations_rand) * stations_split_test))
meteobydate['test'] = meteobydate.codearvalis.isin(stations_rand[:nb_stations_test]) | meteobydate.datemesure.dt.year.isin(test_years)
print("Proportions de données de test:")
print(meteobydate.test.value_counts(normalize = True))

test = meteobydate[meteobydate.test].drop(columns = 'test')
# on mélange les données d'entrainement pour éviter d'avoir trop d'homogénéité dans les batchs
train = meteobydate[~meteobydate.test].sample(frac=1).drop(columns = 'test')
# pour faciliter les traitements, on renomme les colonnes des données val pour correspondre à celles utilisées pour l'entrainement
# les colonnes 'param' sont renommées en 'param_corrige
# les colonnes 'param_origine' sont renommées en 'param'
val = test
val = val.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
val = val.rename(columns = dict(zip(parametres_origine, parametres)))
val_recentes = val_recentes.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
val_recentes = val_recentes.rename(columns = dict(zip(parametres_origine, parametres)))

# on enregistre
train.to_csv(path + '/train_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
test.to_csv(path + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
val.to_csv(path + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
val_recentes.to_csv(path + '/val_recentes_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)

# On normalise avant transformation du jeu de données: c'est plus simple. Cela normalise aussi les variables à prédire!
scaler_meta = MinMaxScaler()
train[meta_features] = scaler_meta.fit_transform(train[meta_features])
test[meta_features] = scaler_meta.transform(test[meta_features])
val[meta_features] = scaler_meta.transform(val[meta_features])
val_recentes[meta_features] = scaler_meta.transform(val_recentes[meta_features])

scaler_param = MinMaxScaler()
train[parametres] = scaler_param.fit_transform(train[parametres])
test[parametres] = scaler_param.transform(test[parametres])
val[parametres] = scaler_param.transform(val[parametres])
val_recentes[parametres] = scaler_param.transform(val_recentes[parametres])

X_train = train[parametres + meta_features]
X_test  = test[parametres + meta_features]
X_val   = val[parametres + meta_features]
X_val_recentes = val_recentes[parametres + meta_features]

np.save(path + '/np_xtrain_' + fichier_suffixe + '.npy', X_train)
np.save(path + '/np_xtest_' + fichier_suffixe + '.npy', X_test)
np.save(path + '/np_xval_' + fichier_suffixe + '.npy', X_val)
np.save(path + '/np_xval_recentes_' + fichier_suffixe + '.npy', X_val_recentes)
joblib.dump(scaler_param, path + '/joblib_scaler_param_' + fichier_suffixe + '.gz')


val = pd.read_csv(path + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
val[parametres] = scaler_param.transform(val[parametres])

# on recherche les valeurs minimales de correction (décile 1 pour retirer les cas extrêmes)
val[[param + '_corrige' for param in parametres]] = scaler_param.transform(val[[param + '_corrige' for param in parametres]].rename(columns = dict(zip([param + '_corrige' for param in parametres], parametres))))
quantiles_corrections = []
for param in parametres:
    correction = np.abs(val[param + '_corrige'] - val[param])
    quantiles_corrections.append(correction.loc[correction > 0].quantile([0.1])[0.1])
print(quantiles_corrections)
# [0.031959629941127, 0.015307964473969246, 0.007352941176470562, 0.0034602076124566894]
# sans ETP [0.018672828363058707, 0.007462686567164201, 0.0034602076124566894]

################################
""" 
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

 """

