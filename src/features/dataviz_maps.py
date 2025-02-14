# pip install basemap
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

sm = pd.read_csv('./data/raw/stationsmeteo.csv', sep = ';')
# On se restreint aux stations météo ouvertes actuellement
sm_ouverte = sm[sm['En Service'] == 'Ouverte']
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