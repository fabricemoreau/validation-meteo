import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

stations = pd.read_csv('./data/raw/stationsmeteo.csv', sep = ';')
# On se restreint aux stations météo ouvertes actuellement
sm_ouverte = stations[stations['En Service'] == 'Ouverte']
plt.figure(figsize=(10, 10))
m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
m.drawcountries()
m.shadedrelief()
x, y = m(sm_ouverte.Longitude, sm_ouverte.Latitude)
m.scatter(x, y, marker='D',color='m')
plt.title('Carte des stations météo ouvertes fin 2024')
plt.savefig('./reports/figures/carte_stations_meteo.png')
plt.show();


# si on veut se restreindre aux stations de l'échantillon
stations363 = pd.read_csv('./data/raw/stationsmeteo_363.csv', sep = ';')
stationstotales = pd.read_csv('./data/raw/stationsmeteo_2010-2024.csv', sep = ';')
plt.figure(figsize=(10, 10))
m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
m.drawcountries()
m.shadedrelief()
x, y = m(stations363.Longitude, stations363.Latitude)
x2, y2 = m(stationstotales.Longitude, stationstotales.Latitude)
m.scatter(x2, y2, marker='D',color='lightgrey', label = "absente échantillon")
m.scatter(x, y, marker='D',color='m', label = "présente dans échantillon")
plt.title("Carte de l'échantillon de 363 stations")
plt.legend()
plt.savefig('./reports/figures/carte_363_stations_meteo.png')
plt.show();
# Pas terrible

# si on veut se restreindre aux stations de l'échantillon par dates
# sélection par groupe d'années: 3 ans
duree = 3
stationstotales = pd.read_csv('./data/raw/stationsmeteo_2010-2024.csv', sep = ';')
for annee_deb in range(2010, 2025, duree):
    annee_fin = annee_deb + 2
    stations_dates = pd.read_csv('./data/raw/stationsmeteo_' + str(annee_deb) + '-' + str(annee_fin) + '.csv', sep = ';')
    plt.figure(figsize=(10, 10))
    m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

    #m.drawmapboundary(fill_color='aqua')
    #m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcoastlines()
    m.drawcountries()
    m.shadedrelief()
    x, y = m(stations_dates.Longitude, stations_dates.Latitude)
    x2, y2 = m(stationstotales.Longitude, stationstotales.Latitude)
    m.scatter(x2, y2, marker='D',color='lightgrey', label = "présente en 2010-2024")
    m.scatter(x, y, marker='D',color='m', label = "présente en " + str(annee_deb) + '-' + str(annee_fin))
    plt.title("Répartition des stations météo en " + str(annee_deb) + '-' + str(annee_fin))
    plt.legend()
    plt.savefig('./reports/figures/carte_stations_meteo_' + str(annee_deb) + '-' + str(annee_fin) + '.png')
    plt.show();
# Pas terrible

# carte de stations spécifiques

def affiche_carte_stations(stations_df: pd.DataFrame, titre, filename):
    latmin = stations_df.Latitude.min() - 0.1
    latmax = stations_df.Latitude.max() + 0.1
    lngmin = stations_df.Longitude.min() - 0.5
    lngmax = stations_df.Longitude.max() + 1.5
    plt.figure(figsize=(5, 5))
    m =  Basemap(llcrnrlon=lngmin,llcrnrlat=latmin,urcrnrlon=lngmax,urcrnrlat=latmax,
                resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcoastlines()
    m.drawcountries()
    #m.shadedrelief()
    x, y = m(stations_df.Longitude, stations_df.Latitude)
    m.scatter(x, y, marker='D',color='m')
    plt.title(titre)
    plt.savefig(filename)
    plt.show();
affiche_carte_stations(stations[stations.Altitude == 0 ], "Station au niveau de la mer", './reports/figures/carte_stations_0m.png')
affiche_carte_stations(stations[stations.Altitude > 800], "Stations à plus de 800 mètres d'altitude", './reports/figures/carte_stations_800m.png')


#### cartes des pluies extrêmes
fichier = './data/processed/meteo_pivot_2010-2024.csv'
meteobydate = pd.read_csv(fichier, sep = ';')
plt.figure(figsize=(10, 10))
mapFrance =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

#mapFrance.drawmapboundary(fill_color='aqua')
#mapFrance.fillcontinents(color='coral',lake_color='aqua')
mapFrance.drawcoastlines()
mapFrance.drawcountries()
mapFrance.shadedrelief()
x, y = mapFrance(stations363.Longitude, stations363.Latitude)
x2, y2 = mapFrance(stationstotales.Longitude, stationstotales.Latitude)
mapFrance.scatter(x2, y2, marker='D',color='lightgrey', label = "absente échantillon")
mapFrance.scatter(x, y, marker='D',color='m', label = "présente dans échantillon")
plt.title("Carte de l'échantillon de 363 stations")
plt.legend()
plt.savefig('./reports/figures/carte_363_stations_meteo.png')
plt.show();