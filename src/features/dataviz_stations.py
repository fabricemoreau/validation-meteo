import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

periode = '2010-2012'
fichier = './data/raw/stationsmeteo_' + periode + '.csv'
stations = pd.read_csv('./data/raw/stationsmeteo_2010-2024.csv', sep = ';')
stations = pd.read_csv(fichier, sep = ';')

# si on veut se restreindre aux stations de l'échantillon
#stations = stations[stations.Station.isin(donnees.codearvalis.unique())]
# stations sont un code à 4 chiffres, les 2 premiers chiffres correspondent au département (9151 = 91)
stations['Departement'] = stations.Station // 100

# Altitude
print(stations.Altitude.describe())
# pas d'altitude négative
print(stations[stations.Altitude == 0]) #station en bord de mer, zone lagunaire, donc cohérent
print(stations[stations.Altitude > 800]) # 200 stations au-dessus de 800m

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
sns.boxplot(y = "Altitude", data = stations, ax = axes[0])
axes[0].title("Altitude des stations")
axes[0].ylabel("Altitude (m)")
axes[0].xlabel("Stations")

sns.histplot(x = "Altitude", data = stations, ax = axes[1])
axes[1].title("distribution des altitudes des stations, période " + periode)
axes[1].ylabel("effectif")
axes[1].xlabel("Altitude (m)")
plt.savefig('./reports/figures/altitudes_' + periode + '.png')
plt.show();
# Les stations les plus hautes sont rares

## distances entre les stations
distance_2d = euclidean_distances(stations[['Lambert93x', 'Lambert93y']]) / 1000 # en km, sans tenir compte altitude
distance_2d = pd.DataFrame(distance_2d, index = stations.Station, columns = stations.Station)
# les 5 distances les plus courtes à chaque station (en excluant la station elle-même)
distances = distance_2d.apply(lambda x: x.nsmallest(6)[1:].to_list(), axis = 0).T
stations_proches = distance_2d.apply(lambda x: x.nsmallest(6)[1:].index.to_list(), axis = 0).T
stations_proches = pd.DataFrame(stations_proches, index = stations.Station)
stations_proches.columns = ['stationproche1', 'stationproche2', 'stationproche3', 'stationproche4', 'stationproche5']

# distances moyennes
print(distances.describe())
plt.figure(figsize=(10, 10))
sns.boxplot( data = distances)
plt.title("Distances des stations les plus proches, periode " + periode)
plt.ylabel("Distance (km)")
plt.xticks([0, 1, 2, 3, 4],labels = ["stationproche1", "stationproche2", "stationproche3", "stationproche4", "stationproche5"], rotation = 45)
plt.tit
plt.savefig('./reports/figures/distances_moyennes' + periode + '.png')
plt.show();

# format plus facile à exploiter pour les graphiques
distances_graph = distances.melt(var_name='rang', value_name='distance')

plt.figure(figsize=(10, 10))
sns.boxplot(x = "rang", y = "distance", data = distances_graph)
plt.title("Distance aux 5 stations les plus proches, periode " + periode)
plt.ylabel("Distance (km)")
plt.xlabel("Rang de la station")
plt.savefig('./reports/figures/distances_rang_' + periode + '.png')
plt.show();


# les stations en altitude sont elles plus éloignées des autres ? 
# Extraire les altitudes des stations proches
altitudes_stations_proches = stations_proches.apply(
    lambda x: [stations.loc[stations.Station == station, "Altitude"].values[0] for station in x], axis=1
)

# Créer un DataFrame avec les altitudes
altitudes_stations_proches_df = pd.DataFrame(altitudes_stations_proches.tolist(), index=stations_proches.index)
altitudes_stations_proches_df.columns = ['altitude_stationproche1', 'altitude_stationproche2', 'altitude_stationproche3', 'altitude_stationproche4', 'altitude_stationproche5']
altitudes_stations_proches_df.T.describe()

difference_altitude = euclidean_distances(stations[['Altitude']]) # en m
difference_altitude = pd.DataFrame(difference_altitude, index = stations.Station, columns = stations.Station)                                          

distance_altitude_df = distance_2d.stack()
distance_altitude_df.index.names = ['Station1', 'Station2']
distance_altitude_df = distance_altitude_df.reset_index()
distance_altitude_df.columns = ['Station1', 'Station2', 'Distance']
distance_altitude_df['Difference_Altitude'] = difference_altitude.stack().values
distance_altitude_df['departement'] = distance_altitude_df.Station1 // 100

# différence d'altitude en fonction de la distance          
plt.figure(figsize=(10, 10))
sns.scatterplot(x='Distance', y='Difference_Altitude', data=distance_altitude_df, )
sns.lineplot(x= [0, 50], y = [150, 150], color='red', label = "seuil tolérance")
plt.title("Différence d'altitude en fonction de la distance, période " + periode)
plt.xlabel("Distance (km)")
plt.ylabel("Différence d'altitude (m)")
plt.xlim(0, 50)
plt.savefig('./reports/figures/distance_altitude_' + periode + '.png')
plt.show();
print(distance_altitude_df.corr()) # pas de corrélation entre distance et altitude


fig = plt.figure(figsize=(20, 20))
g = sns.FacetGrid(distance_altitude_df, col="departement", col_wrap=9, height=2)
g.map(sns.scatterplot, "Distance", "Difference_Altitude")
g.add_legend()
plt.show()
#plt.savefig()