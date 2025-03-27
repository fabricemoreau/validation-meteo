import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances 

DONNEES = "data/processed/meteo_pivot_cleaned_2010-2024_0.1.csv" 
STATIONS = "data/processed/stationsmeteo_processed_2010-2024.csv"

# chargement mesures
df = pd.read_csv(DONNEES, sep = ';', parse_dates = ['datemesure'])
df.loc[df.anomaly > 0, 'anomaly'] = 1

stations_timetable = df[['datemesure', 'codearvalis', 'TN']]
stations_timetable = stations_timetable.drop_duplicates()
stations_timetable = stations_timetable.pivot(index = 'codearvalis', columns = 'datemesure', values = 'TN')
# on remplace par valeur binaire 0 ou 1
stations_timetable = stations_timetable.notnull().astype(int)
stations_timetable = stations_timetable.fillna(0)

fig = plt.figure()
plt.matshow(stations_timetable, cmap = 'gray_r', aspect = 2)
plt.xticks([i for i in range(0, stations_timetable.shape[1]+1, 366)], [stations_timetable.columns[i].year for i in range(0, stations_timetable.shape[1]+1, 366)])
plt.yticks([i for i in range(0, stations_timetable.shape[0]+1, 100)], [stations_timetable.index[i] for i in range(0, stations_timetable.shape[0]+1, 100)])
plt.xlabel("Temps (années)")
plt.ylabel("Numéro de station météo")
plt.savefig('./reports/figures/series_stations_meteo.png')


# chargement liste stations
stations = pd.read_csv(STATIONS, sep = ';')
dates_presence =  df.groupby('codearvalis').datemesure.agg(['min', 'max'])
stations = stations.merge(dates_presence, left_on = 'Station', right_index = True)
stations.to_csv('reports/streamlit-data/stations.csv', index = False, sep = ';')



from imblearn.under_sampling import RandomUnderSampler

# Définir les caractéristiques (X) et la cible (y)
X = df.drop(columns=['anomaly'])  # Toutes les colonnes sauf "anomaly"
y = df['anomaly']  # Colonne cible

# Initialiser le sous-échantillonneur
rus = RandomUnderSampler()

# Appliquer le sous-échantillonnage
X_resampled, y_resampled = rus.fit_resample(X, y)

# Reconstituer un DataFrame équilibré
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Afficher la répartition des classes après sous-échantillonnage
print(df_resampled['anomaly'].value_counts())