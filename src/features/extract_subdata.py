'''
Script pour extraire un sous-ensemble du jeu de données de 5Go
'''
import pandas as pd
import duckdb

duckdb.read_csv("./data/raw/donneesmeteo_2010-2024_completes.csv", sep=';')
query = """
    SELECT * 
    FROM './data/raw/donneesmeteo_2010-2024_completes.csv'
    WHERE EXTRACT('YEAR' FROM datemesure) IN (2011, 2012, 2013, 2014, 2015) 
       AND libellecourt IN ['TN', 'TX', 'TM', 'RR', 'GLOT', 'ETP']
"""
df = duckdb.sql(query)
df.to_csv('./data/raw/donneesmeteo_2011-2015_completes_v2.csv', sep=';', timestamp_format='%Y-%m-%d')

stations = pd.read_csv("./data/raw/stationsmeteo.csv", sep=';')
stations_selected = stations.Station.sample(500,  random_state = 42)

query = """
    SELECT * 
    FROM './data/raw/donneesmeteo_2010-2024_completes.csv'
    WHERE codearvalis IN ("""
query += ','.join(map(str, stations_selected.tolist())) + ")"
df = duckdb.sql(query)
df.to_csv('./data/raw/donneesmeteo_2010-2024_500stations.csv', sep=';', timestamp_format='%Y-%m-%d')

# pip install basemap
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
sm = pd.read_csv('./data/raw/stationsmeteo.csv', sep = ';')
# On se restreint aux stations météo ouvertes actuellement
sm_ouverte = sm[sm['En Service'] == 'Ouverte']
sm_ouverte = stations[stations.Station.isin(stations_selected)]
plt.figure(figsize=(10, 10))
m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
             resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)
#m.etopo()
m.drawmapboundary(fill_color='aqua')
m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
m.drawcountries()
x, y = m(sm_ouverte.Longitude, sm_ouverte.Latitude)
m.scatter(x, y, marker='D',color='m')
plt.show()