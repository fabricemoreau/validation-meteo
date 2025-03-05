'''
Ajout des informations spatio-temporelles: utiles pour modèles linéaires
'''
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

periode = '2010-2024'
fichier = './data/processed/meteo_pivot_cleaned_' + periode + '.csv'
meteobydate = pd.read_csv(fichier, sep = ';', parse_dates = True)
meteobydate.datemesure = pd.to_datetime(meteobydate.datemesure).round('d')

stations = pd.read_csv('./data/raw/stationsmeteo_' + periode +'.csv', sep = ';')
stations = stations.drop(columns = 'Unnamed: 0')

## distances entre les stations
distance_2d = euclidean_distances(stations[['Lambert93x', 'Lambert93y']]) / 1000 # en km, sans tenir compte altitude
distance_2d = pd.DataFrame(distance_2d, index = stations.Station, columns = stations.Station)
# les 5 distances les plus courtes à chaque station (en excluant la station elle-même)
distances = distance_2d.apply(lambda x: x.nsmallest(6)[1:].to_list(), axis = 0).T
distances.columns = ['distance_station1', 'distance_station2', 'distance_station3', 'distance_station4', 'distance_station5']
stations_proches = distance_2d.apply(lambda x: x.nsmallest(6)[1:].index.to_list(), axis = 0).T
stations_proches = pd.DataFrame(stations_proches, index = stations.Station)
stations_proches.columns = ['stationproche1', 'stationproche2', 'stationproche3', 'stationproche4', 'stationproche5']

stations = pd.concat([stations, stations_proches.reset_index(drop = True), distances.reset_index(drop = True)], axis = 1)
stations.to_csv('./data/processed/stationsmeteo_processed_' + periode + '.csv', sep = ';', index = False)

# ajout des 5 derniers jours
df_time = pd.concat([meteobydate, meteobydate[parametres].shift([1,2,3,4,5])], axis = 1)
# suppression des 5 premiers jours de chaque station: le df est trié par codearvalis puis station. On peut donc supprimer quand il y a une discontinuité
df_time = df_time.sort_values(['codearvalis', 'datemesure'])

df_time = df_time.reset_index(drop = True)
df_time.to_csv('./data/processed/meteo_pivot_cleaned_time_' + periode + '.csv', sep = ';', index = False)

df_time = pd.read_csv('./data/processed/meteo_pivot_cleaned_time_' + periode + '.csv', sep = ';')
df_time.datemesure = pd.to_datetime(df_time.datemesure).round('d')

# ajout des valeurs des stations voisines et distance
'''df_space_time = df_time.merge(stations[['Station', 
                        'distance_station1', 'distance_station2', 'distance_station3', 'distance_station4', 'distance_station5',
                        'stationproche1', 'stationproche2', 'stationproche3', 'stationproche4', 'stationproche5']], left_on = 'codearvalis', right_on = "Station").drop(columns = 'Station')
for p in parametres:
    for d in range(1, 3):
        for t in range(1, 3):
            df_space_time[p + '_j' + str(d) + '_t' + str(t)] = np.nan
df_space_time = df_space_time.drop(columns = ['distance_station3', 'distance_station4', 'distance_station5', 'stationproche3', 'stationproche4', 'stationproche5'])
df_space_time.to_csv('./data/processed/meteo_pivot_cleaned_time_space_' + periode + '.csv', sep = ';', index = False)

df_space_time = pd.read_csv('./data/processed/meteo_pivot_cleaned_time_space_' + periode + '.csv', sep = ';')
df_space_time.datemesure = pd.to_datetime(df_space_time.datemesure).round('d')
'''
df_space_time = df_time
for p in parametres:
    for d in range(1, 4):
        for t in range(1, 4):
            df_space_time[p + '_j' + str(d) + '_t' + str(t)] = np.nan

#for i in df_space_time.index:
for i in df_space_time[5400:].index:
    if (i % 100) == 0:
        print(i, '/', len(df_space_time.index))
    for p in parametres:
        for d in range(1, 4):
            station_proche = stations.loc[stations.Station == df_space_time.loc[i, "codearvalis"], "stationproche" + str(d)].iloc[0]
            for t in range(1, 4):
                jour = df_space_time.loc[i, "datemesure"] - pd.Timedelta(t,"d")
                valeur = df_space_time.loc[(df_space_time.codearvalis == station_proche) & (df_space_time.datemesure == jour), p]
                if (len(valeur) == 1):
                    df_space_time.loc[i, p + '_j' + str(d) + '_t' + str(t)] = valeur.iloc[0]

# faire une sauvegarde
df_space_time.to_csv('./data/processed/meteo_pivot_cleaned_time_space_' + periode + '_77500.csv', sep = ';', index = False)

# pas encore exécuté
# index des changements de séries temporelles
index_to_drop = df_time[df_time.datemesure - pd.Timedelta(days = 1) != df_time.datemesure.shift(1)].index
for index in index_to_drop:
    df_time.drop(index = [i for i in range(index, index + 5)], inplace = True, errors = 'ignore')

