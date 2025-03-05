import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import joblib

parametres = ['ETP', 'GLOT', 'TN', 'TX'] # pas RR pour l'instant
nb_jours = 10
seed = 42

periode = '2010-2024'
path    = './data/processed'
fichier = path + '/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')
# on ne fait pas de : cela sera géré dans la construction du jeu : meteobydate = meteobydate.dropna()
meteobydate = meteobydate.sort_values(['codearvalis', 'datemesure'])

n_features = len(parametres)
nb_stations = len(meteobydate.codearvalis.unique())

# construction d'un jeu de données
def split_sequence(sequence, n_steps, sequence_index):
    X, y, ix = list(), list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
        ix.append(sequence_index[end_ix])
    return np.array(X), np.array(y), np.array(ix)

# séparation train-test: par les années: on garde les années les plus récentes pour le test
#meteobydate['test'] = meteobydate.datemesure.dt.year.isin([2023, 2024])
# séparation train-test: par les stations : on garde 10% pour le test
stations_rand = meteobydate.codearvalis.unique()
np.random.shuffle(stations_rand)
meteobydate['test'] = meteobydate.codearvalis.isin(stations_rand[(len(stations_rand) * 9 // 10): ]) | meteobydate.datemesure.dt.year.isin([2023, 2024])
print("on veut max 30% de données de test")
print(meteobydate.test.value_counts(normalize = True))

# On normalise avant transformation du jeu de données: c'est plus simple. Cela normalise aussi les variables à prédire!
scaler = MinMaxScaler()
meteobydate.loc[meteobydate.test == False, parametres] = scaler.fit_transform(meteobydate.loc[meteobydate.test == False, parametres])
meteobydate.loc[meteobydate.test == True, parametres]  = scaler.transform    (meteobydate.loc[meteobydate.test == True, parametres])

anomalies_true = (meteobydate[[param + '_anomalie' for param in parametres]] > 0)
threshold = meteobydate[[param + '_origine' for param in parametres]].std() * 0.1

for param in parametres:
    meteobydate[param + '_anomalie_threshold'] = np.where(np.abs(meteobydate[param + '_difference']) > threshold[param + '_origine'], 1, 0)

X_train = []
X_test  = []
y_train = []
y_test  = []
indexes_train = []
indexes_test = []
for i, station in enumerate(meteobydate.codearvalis.unique()):
    print((i + 1), "/", nb_stations, "station en cours : ", station, " - jeu d'entrainement")
    X_i, y_i, df_index = split_sequence(meteobydate.loc[(meteobydate.codearvalis == station) & (meteobydate.test == False), parametres].values, nb_jours,
                                        meteobydate.loc[(meteobydate.codearvalis == station) & (meteobydate.test == False), parametres].index)
    X_train.extend(X_i)
    y_train.extend(y_i)
    indexes_train.extend(df_index)

for i, station in enumerate(meteobydate.codearvalis.unique()):
    print((i + 1), "/", nb_stations, "station en cours : ", station, " - jeu de test")
    X_i, y_i, df_index = split_sequence(meteobydate.loc[(meteobydate.codearvalis == station) & (meteobydate.test == True), parametres].values, nb_jours,
                                        meteobydate.loc[(meteobydate.codearvalis == station) & (meteobydate.test == True), parametres].index)
    X_test.extend(X_i)
    y_test.extend(y_i)
    indexes_test.extend(df_index)

# ordre aléatoire pour train
X_train, y_train, indexes_train = shuffle(X_train, y_train, indexes_train)


## meta données météos
X_meta_train = meteobydate.loc[indexes_train, ['jourjulien', 'Altitude', 'Lambert93x', 'Lambert93y']]
X_meta_test = meteobydate.loc[indexes_test, ['jourjulien', 'Altitude', 'Lambert93x', 'Lambert93y']]
scaler_meta = MinMaxScaler()
X_meta_train = scaler_meta.fit_transform(X_meta_train)
X_meta_test  = scaler_meta.transform(X_meta_test)

################################

# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train = np.reshape(X_train, (len(X_train), nb_jours, n_features))
X_test  = np.reshape(X_test, (len(X_test), nb_jours, n_features))
y_train = np.array(y_train)
y_test = np.array(y_test)

## jeux d'évaluation
balanced_df_TN = meteobydate.loc[indexes_test]
balanced_df_TN['indexes_test'] = indexes_test
nb_anomalies_TN = (balanced_df_TN.TN_anomalie_threshold == 1).sum()
balanced_df_TN = pd.concat(
    [balanced_df_TN[balanced_df_TN.TN_anomalie_threshold == 0].sample(nb_anomalies_TN),
    balanced_df_TN[balanced_df_TN.TN_anomalie_threshold == 1]])

X_val_TN = []
y_val_TN = []
index_test = np.array(indexes_test)
for i in balanced_df_TN.indexes_test:
    j = np.where(index_test == i)
    X_val_TN.append(X_test[j]) faux: il faut prendre TN_origine!
    y_val_TN.append(y_test[j])
X_meta_val_TN = balanced_df_TN.loc[['jourjulien', 'Altitude', 'Lambert93x', 'Lambert93y']]



np.save(path + '/np_tsforecast3_xtrain.npy', X_train)
np.save(path + '/np_tsforecast3_xtest.npy', X_test)
np.save(path + '/np_tsforecast3_indexestrain.npy', indexes_train)

np.save(path + '/np_tsforecast3_ytrain.npy', y_train)
np.save(path + '/np_tsforecast3_ytest.npy', y_test)
np.save(path + '/np_tsforecast3_indexestrain.npy', indexes_test)

np.save(path + '/np_tsforecast3_xval_tn.npy', X_val_TN)
np.save(path + '/np_tsforecast3_yval_tn.npy', y_val_TN)
np.save(path + '/np_tsforecast3_indexesval_tn.npy', balanced_df_TN.indexes_test.values)
np.save(path + '/np_tsforecast3_xval_meta', X_meta_val_TN)

joblib.dump(scaler, path + '/joblib_tsforecast3_scaler.gz')
joblib.dump(scaler_meta, path + '/joblib_tsforecast3_scaler_meta.gz')

np.save(path + '/np_tsforecast3_xtrain_meta', X_meta_train)
np.save(path + '/np_tsforecast3_xtest_meta', X_meta_test)

col_to_save = ['test']
col_to_save.extend(param + '_anomalie_threshold' for param in parametres)
meteobydate[col_to_save].to_csv(path + '/pd_tsforecast3_anomalies_threshold.csv', sep =';', index = False)