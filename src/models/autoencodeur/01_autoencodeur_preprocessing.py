
""" 
Préprocessing spécial pour autoencodeur

Pour utiliser ce script
choisissez SEUIL_ANOMALIE et modifier la variable DONNEES pour définir les données à utiliser
choisissez les PARAMETRES à utiliser
Les données seront écrite dans le dossier PATH spécifié

"""
METHOD  = 'autoencodeur'
PATH    = './data/processed/' + METHOD
SEUIL_ANOMALIE = 0.1
DONNEES = f"data/processed/meteo_pivot_cleaned_2010-2024_{SEUIL_ANOMALIE}.csv" 
PARAMETRES =  ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant # ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# définition des critères de séparation train - val - test
## On se limite dans un premier temps à 2010-2022: éviter les nouvelles stations météo
train_years = [a for a in range(2010, 2021)]
# séparation train-test: par les années: on garde les années les plus récentes pour le test
val_years = [a for a in range(2021, 2023)]
# proportion de station météo complètes pour le test: on garde 10% pour le test
stations_split_test = 0.1

# suffixe ajouté aux noms des fichiers générés pour les conserver entre les tests
fichier_suffixe = '-'.join(PARAMETRES) + '_' + str(np.min(train_years + val_years)) + '-' + str(np.max(train_years + val_years)) + '-' + str(SEUIL_ANOMALIE)

# features à utiliser par l'autoencodeur sans les reproduire en sortie
meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']

# nom des colonnes contenant les paramètres non corrigés dans FICHIER
parametres_origine = [ param + '_origine' for param in PARAMETRES ]

# lecture
meteobydate = pd.read_csv(DONNEES, sep = ';', parse_dates = ['datemesure'])

# la colonne 'anomaly' contient le nombre d'anomalies, on la remet sous forme binaire
meteobydate['anomaly']  = np.where(meteobydate['anomaly'] > 0, 1, 0)

## ETAPE séparation des données train val test
# jeu de test: retrait des années utilisées pour train et val
test = meteobydate[~meteobydate.datemesure.dt.year.isin(train_years + val_years)]
# jeu train-val : uniquement les années sélectionnées
meteobydate = meteobydate[meteobydate.datemesure.dt.year.isin(train_years + val_years)]
stations_rand = meteobydate.codearvalis.unique()
# on tire un échantillon aléatoire de nb_stations_test stations pour le test
np.random.shuffle(stations_rand)
nb_stations_test = int(np.round(len(stations_rand) * stations_split_test))
meteobydate['val'] = meteobydate.codearvalis.isin(stations_rand[:nb_stations_test]) | meteobydate.datemesure.dt.year.isin(val_years)
print("Proportions de données de test:")
print(meteobydate.val.value_counts(normalize = True))

# Le jeu de validation est désormais défini dans val
val = meteobydate[meteobydate.val].drop(columns = 'val')
# on mélange les données d'entrainement pour éviter d'avoir trop d'homogénéité dans les batchs
train = meteobydate[~meteobydate.val].sample(frac=1).drop(columns = 'val')

# pour faciliter les traitements, on renomme les colonnes des données test pour correspondre à celles utilisées pour l'entrainement
# les colonnes 'param' sont renommées en 'param_corrige
# les colonnes 'param_origine' sont renommées en 'param'
test = test.rename(columns = dict(zip(PARAMETRES, [param + '_corrige' for param in PARAMETRES])))
test = test.rename(columns = dict(zip(parametres_origine, PARAMETRES)))

# on sauvegarde les différents jeux de données (pas d'usage pour la suite des scripts)
train.to_csv(PATH + '/train_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
val.to_csv(PATH + '/val_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)
test.to_csv(PATH + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';', index = False)

# On normalise avant transformation du jeu de données: c'est plus simple. Cela normalise aussi les variables à prédire!
scaler_meta = MinMaxScaler()
train[meta_features] = scaler_meta.fit_transform(train[meta_features])
val[meta_features] = scaler_meta.transform(val[meta_features])
test[meta_features] = scaler_meta.transform(test[meta_features])

scaler_param = MinMaxScaler()
train[PARAMETRES] = scaler_param.fit_transform(train[PARAMETRES])
val[PARAMETRES] = scaler_param.transform(val[PARAMETRES])
test[PARAMETRES] = scaler_param.transform(test[PARAMETRES])

X_train = train[PARAMETRES + meta_features]
X_val  = val[PARAMETRES + meta_features]
X_test   = test[PARAMETRES + meta_features]

# sauvegarde des données au format numpy pour ne charger que le minimum de donnée et limiter l'occupation en mémoire vive
np.save(PATH + '/np_xtrain_' + fichier_suffixe + '.npy', X_train)
np.save(PATH + '/np_xval_' + fichier_suffixe + '.npy', X_val)
np.save(PATH + '/np_xtest_' + fichier_suffixe + '.npy', X_test)
joblib.dump(scaler_param, PATH + '/joblib_scaler_param_' + fichier_suffixe + '.gz')

#####################
test = pd.read_csv(PATH + '/test_preprocessed_' + fichier_suffixe + '.csv', sep = ';')
test[PARAMETRES] = scaler_param.transform(test[PARAMETRES])




