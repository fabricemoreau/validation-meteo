""" timeseriesforecasting pour un paramètre """
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef


periode = '2010-2024'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')
# on ne fait pas de : cela sera géré dans la construction du jeu : meteobydate = meteobydate.dropna()
meteobydate = meteobydate.sort_values(['codearvalis', 'datemesure'])

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

parametre = 'TN'
n_features = 1
nb_jours = 5

# construction d'un jeu de données
def split_sequence(sequence, n_steps):
    X, y, ix = list(), list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        ix.append(end_ix)
    return np.array(X), np.array(y), np.array(ix)

nb_stations = len(meteobydate.codearvalis.unique())
X = []
y = []
indexes = []
for i, station in enumerate(meteobydate.codearvalis.unique()):
    print((i + 1), "/", nb_stations, "station en cours : ", station)
    X_i, y_i, df_index = split_sequence(meteobydate.loc[meteobydate.codearvalis == station, parametre].tolist(), nb_jours)
    X.extend(X_i)
    y.extend(y_i)
    indexes.extend(df_index)
    
anomalies_true = (meteobydate['TN_anomalie']) > 0
threshold = meteobydate['TN_origine'].std() * 0.1
anomalies_thresold = ((np.abs(meteobydate['TN'] - meteobydate['TN_origine']) > threshold)) > 0

del meteobydate

# revoir le split: mettre les années récentes en test, ne pas mélanger années ou stations entre train et test
# remplacer par TimeSeriesSplit
# remplacer pour avoir y test équilibré en anomalies
X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(X, y, indexes, test_size=0.33, shuffle = True, random_state = 42)

anomalies_true_test = anomalies_true[index_test]
anomalies_thresold_test = anomalies_thresold[index_test]
del X
del y

scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#X_train_scale = np.array(X_train)
#X_test_scale = np.array(X_test)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train_scale = X_train_scale.reshape((X_train_scale.shape[0], X_train_scale.shape[1], n_features))
X_test_scale = X_test_scale.reshape((X_test_scale.shape[0], X_test_scale.shape[1], n_features))
y_train = np.array(y_train)
y_test = np.array(y_test)


# define model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(nb_jours, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation = 'linear'))
model.compile(optimizer= Adam(learning_rate=0.1), loss='mse')

early_stopping = EarlyStopping(monitor = 'val_loss',
                           min_delta = 0.01,
                           patience = 3,
                           mode = 'min',
                           verbose = 1)
reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss',
                               min_delta = 0.1,
                               patience = 2,
                               factor = 0.5, 
                               #cooldown = 2,						
                               verbose = 1)
checkpoint = ModelCheckpoint('data/models/tsforecast_v1.keras', 
                                save_best_only=True, 
                                monitor='val_loss',
                                mode='min')
# ajouter sauvegarde du meilleur modèle
history = model.fit(X_train_scale, y_train, batch_size=128, validation_split= 0.2, epochs=20,
                    callbacks = [early_stopping, reduce_learning_rate, checkpoint])
y_pred_test = model.predict(X_test_scale)
ydf = pd.DataFrame({"y_test": y_test, "y_pred_test": y_pred_test.reshape(-1)})

print(root_mean_squared_error(ydf.y_test, ydf.y_pred_test))
seuil = root_mean_squared_error(ydf.y_test, ydf.y_pred_test)
# rmse 3 pour nb_jours = 5, filters = 16 dense = 20

# anomalies détectées
anomalies_pred = np.where(np.abs(ydf.y_test - ydf.y_pred_test) > seuil, 1, 0)
anomalies_thresold_test = np.where(anomalies_thresold_test, 1, 0)

pd.crosstab(anomalies_thresold_test, anomalies_pred)
print(classification_report(anomalies_thresold_test, anomalies_pred))