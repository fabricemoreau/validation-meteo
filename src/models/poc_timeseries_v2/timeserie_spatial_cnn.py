
# export QT_QPA_PLATFORM=offscreen
# d'après : https://www.tensorflow.org/tutorials/structured_data/time_series?hl=fr
# à regarder ensuite: https://medium.com/@kavyamalla/extending-tensorflows-window-generator-for-multiple-time-series-8b15eba57858
import os
import datetime

#import IPython
#import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

parametres = ['ETP', 'GLOT', 'TN', 'TX']
TN_ANOMALIE_SEUIL = 2.5
coords = ['Altitude', 'Lambert93x', 'Lambert93y'] # ajouter distance!
df = pd.read_csv('data/processed/meteo_pivot_cleaned_2010-2024.csv', parse_dates=["datemesure"], sep = ';')
df = df.sort_values(['codearvalis', 'datemesure'])
#df.index = df.datemesure

stations = pd.read_csv('./data/raw/stationsmeteo_2010-2024.csv', sep = ';')
stations = stations.drop(columns = 'Unnamed: 0')
stations = stations.merge(df.groupby('codearvalis').datemesure.agg(['min', 'max']), left_on='Station', right_on='codearvalis')
distance_2d = euclidean_distances(stations[['Lambert93x', 'Lambert93y']]) / 1000 # en km, sans tenir compte altitude
distance_2d = pd.DataFrame(distance_2d, index = stations.Station, columns = stations.Station)

# une station météo
NB_STATIONS_PROCHES = 3


def build_features_stations(df: pd.DataFrame, stations: pd.DataFrame, distances : pd.DataFrame, parametres_meteo, nb_stations_proches: int = NB_STATIONS_PROCHES):
    param_proches = [f"{param}_{i}" for i in range(NB_STATIONS_PROCHES) for param in parametres_meteo]
    coord_proches = [f"{param}_{i}" for i in range(nb_stations_proches) for param in coords]
    df_9151 =df.loc[df.codearvalis == 9151, ['datemesure', 'day_sin', 'day_cos'] + parametres_meteo]
    df_9151[param_proches + coord_proches] = None
    stations_proches = distances[9151].sort_values().drop(9151)
    stations_proches = stations_proches.head(nb_stations_proches * 10)
    stations_proches = pd.DataFrame(stations_proches).reset_index().merge(stations)
    date_i = stations.loc[stations.Station == 9151, 'min'].item()
    date_max = stations.loc[stations.Station == 9151, 'max'].item()
    while date_i < date_max:
        print(date_i)
        stations_proches_i = stations_proches[(stations_proches['min'] <= date_i) & (stations_proches['max'] > date_i)].head(nb_stations_proches)
        date_fin = stations_proches_i['max'].min()
        for i in range(nb_stations_proches):
            print(stations_proches_i.iloc[i].Station)
            df_9151.loc[(df_9151.datemesure >= date_i) &
                        (df_9151.datemesure <= date_fin), [f"{param}_{i}" for param in parametres_meteo]] = df.loc[(df.codearvalis == stations_proches_i.iloc[i].Station) & 
                    (df.datemesure >= date_i) &
                    (df.datemesure <= date_fin), parametres_meteo].values
            df_9151.loc[(df_9151.datemesure >= date_i) &
                        (df_9151.datemesure <= date_fin), [f"{param}_{i}" for param in coords]] = stations_proches_i.iloc[i][coords].values    
        date_i = date_fin + pd.Timedelta(days=1)
    return df_9151
df_9151 = build_features_stations(df, stations, distance_2d, parametres)
parametres_origine = [f"{param}_origine" for param in parametres]
df_val = df
df_val = df_val.rename(columns = dict(zip(parametres, [param + '_corrige' for param in parametres])))
df_val = df_val.rename(columns = dict(zip(parametres_origine, parametres)))
df_9151_val = build_features_stations(df_val, stations, distance_2d, parametres)

TN_anomalies = np.where(np.abs(df_val[df_val.codearvalis == 9151].TN - df_val[df_val.codearvalis == 9151].TN_corrige) > TN_ANOMALIE_SEUIL, 1, 0)

plot_features = df_9151[parametres]
#plot_features.index = df_9151.datemesure
_ = plot_features.plot(subplots=True)
plt.savefig('data/timeserie/plot1.png')

plot_features = df_9151[parametres][:480]
#plot_features.index = df_9151.datemesure[:480]
_ = plot_features.plot(subplots=True)
plt.savefig('data/timeserie/plot2.png')

# on divise les données
df_9151 = df_9151.drop(columns = 'datemesure')
df_9151_val = df_9151_val.drop(columns = 'datemesure')
# on ne conserve pas les infos des stations voisines: pas assez variance quand une seule station
df_9151 = df_9151.drop(columns = [f"{param}_{i}" for i in range(NB_STATIONS_PROCHES) for param in coords])
df_9151_val = df_9151_val.drop(columns = [f"{param}_{i}" for i in range(NB_STATIONS_PROCHES) for param in coords])
column_indices = {name: i for i, name in enumerate(df_9151.columns)}

n = len(df_9151)
train_df = df_9151[0:int(n*0.3)]
val_df = df_9151[int(n*0.3):int(n*0.5)]
# discutable: on utilise les données brut pour validation (ce qui est ok), mais vu le peu d'anomalies, on utilise toute la série temporelle
# il serait plus juste de prendre une partie qui n'a servi ni aux tests, ni à l'entrainement
test_df = df_9151_val[int(n*0.5):]
TN_anomalies = TN_anomalies[int(n*0.5):]

num_features = df_9151.shape[1]

# on normalise
train_mean = train_df.mean()
train_std = train_df.std()
# pour une seule station, on a des écarts-types nuls
train_std[train_std == 0] = 1

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

y_train = train_df.TN
y_val  = val_df.TN
y_test = test_df.TN

# visualisation
df_std = (df_9151 - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df_9151.keys(), rotation=90)
plt.savefig('data/timeserie/plot3.png')


## fenêtrage des données
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def plot(self, model=None, plot_col='TN', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [day]')
    return None

  def make_dataset(self, data, shuffle = True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df, shuffle = False)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

w2 = WindowGenerator(input_width=12, label_width=1, shift=1,
                     label_columns=['TN'])

example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

"""
w2._example = example_inputs, example_labels
w2.plot()
w2.plot(example_inputs, example_labels)
plt.savefig('data/timeserie/plot4.png')
w2.plot(example_inputs, example_labels, plot_col='ETP')
plt.savefig('data/timeserie/plot5.png')
"""
w2.train.element_spec

for example_inputs, example_labels in w2.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')

# Modèle une seule étape
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['TN'])
single_step_window


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    def bidon(): # ajouté pour problèmes interprétation
        return None

baseline = Baseline(label_index=column_indices['TN'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])


def detection_anomaly(model, window, y_train, y_test, TN_anomalies):
    y_train_pred = model.predict(window.train).reshape(-1)
    y_train2 = y_train[window.input_width:]
    residus = np.abs(y_train2 - y_train_pred)
    seuil = residus.max() # 2.5°C = 0.45
    seuil = model.evaluate(window.train)[1]
    seuil = np.quantile(residus, 0.5) # 3.26 : bien trop élevé!!
    y_test2 = y_test[window.input_width:]
    y_test_pred = model.predict(window.test) # renvoie les sorties aléatoirement!!!
    y_test_pred = y_test_pred.reshape(-1)
    residus_test = np.abs(y_test2 - y_test_pred)
    TN_anomalies2 = TN_anomalies[window.input_width:]
    TN_anomalies_pred = np.where(residus_test > seuil, 1, 0)
    print("recall: ", recall_score(TN_anomalies2, TN_anomalies_pred))
    print("f1_score: ", f1_score(TN_anomalies2, TN_anomalies_pred))
    print("precision: ", precision_score(TN_anomalies2, TN_anomalies_pred))
    print(classification_report(TN_anomalies2, TN_anomalies_pred))
    print(
        pd.crosstab(
            TN_anomalies2, TN_anomalies_pred, rownames=["Classe réelle"], colnames=["Classe prédite"]
        )
    )
    return recall_score(TN_anomalies2, TN_anomalies_pred), f1_score(TN_anomalies2, TN_anomalies_pred), accuracy_score(TN_anomalies2, TN_anomalies_pred)
    

val_performance = {}
performance = {}
detection_recall = {}
detection_f1 = {}
accuracy = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
detection_recall['Baseline'], detection_f1['Baseline'], accuracy['Baseline']  = detection_anomaly(baseline, single_step_window, y_train, y_test, TN_anomalies)

# sur 30 jours
wide_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    label_columns=['TN'])

wide_window
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

# à revoir: plot
wide_window.plot(baseline)
plt.savefig('data/timeserie/plot6.png')


### modèle linéaire
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

MAX_EPOCHS = 60

def compile_and_fit(model, window, patience=3):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        min_delta = 1E-4,
                                                        patience=patience,
                                                        mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=patience - 1,
        verbose=1,
        min_delta=1E-3,
        cooldown=0,
        min_lr=1E-6
    )

    model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate = 1E-4),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])    
    return history

history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)
detection_recall['Linear'], detection_f1['Linear'], accuracy['Linear'] = detection_anomaly(linear, single_step_window, y_train, y_test, TN_anomalies)


wide_window.plot(linear)
plt.savefig('data/timeserie/plot7.png')



# explications
plt.figure()
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)
plt.savefig('data/timeserie/plot8.png')

### dense
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
detection_recall['Dense'], detection_f1['Dense'], accuracy['Dense'] = detection_anomaly(dense, single_step_window, y_train, y_test, TN_anomalies)

CONV_WIDTH = 10
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['TN'])

conv_window
conv_window.plot()
plt.title("Given 3 hours of inputs, predict 1 hour into the future.")
plt.savefig('data/timeserie/plot9.png')


multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
detection_recall['Multi step dense'], detection_f1['Multi step dense'], accuracy['Multi step dense'] = detection_anomaly(multi_step_dense, conv_window, y_train, y_test, TN_anomalies)

conv_window.plot(multi_step_dense)
plt.savefig('data/timeserie/plot10.png')


####convolutif
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)

history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
detection_recall['Conv'], detection_f1['Conv'], accuracy['Conv'] = detection_anomaly(conv_model, conv_window, y_train, y_test, TN_anomalies)

LABEL_WIDTH = 30
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['TN'])

wide_conv_window
print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)
wide_conv_window.plot(conv_model)
plt.savefig('data/timeserie/plot11.png')

### récurrent
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])
print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
#detection_recall['LSTM'], detection_f1['LSTM'] = detection_anomaly(lstm_model, wide_window, y_train, y_test, TN_anomalies)

wide_window.plot(lstm_model)
plt.savefig('data/timeserie/plot13.png')


#### performances
plt.figure()
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = conv_model.metrics_names.index('compile_metrics')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.savefig('data/timeserie/plot99.png')

for name, value in performance.items():
  print(f'{name:12s} normalized: {value[1]:0.4f}')
  print(f'{name:12s} denormalized: {value[1] * train_std['TN']:0.4f}')

#### recap
# Préparation des données pour le graphique
models = list(detection_recall.keys())
recall_values = list(detection_recall.values())
accuracy_values = list(accuracy.values())

# Création du graphique en barres
x = np.arange(len(models))  # Position des barres
width = 0.35  # Largeur des barres

plt.figure(figsize=(10, 6))
# Barres pour detection_recall
plt.bar(x - width/2, recall_values, width, label='Recall', color='skyblue')
# Barres pour accuracy
plt.bar(x + width/2, accuracy_values, width, label='Accuracy', color='orange')

# Ajout des labels et du titre
plt.xlabel('Modèles')
plt.ylabel('Valeurs')
plt.title('Performance des modèles : Recall et Accuracy')
plt.xticks(ticks=x, labels=models, rotation=45)
plt.legend()

# Sauvegarde et affichage
plt.tight_layout()
plt.savefig('data/timeserie/detection_performance.png')
#plt.show()