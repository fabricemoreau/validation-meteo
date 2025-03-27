import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from mpl_toolkits.basemap import Basemap # afficher les cartes



def stations_map(stations: pd.DataFrame, title = "Carte des stations météo", figsize=(7, 7)):
    fig = plt.figure(figsize = figsize)
    m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
                resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

    #m.drawcoastlines()
    #m.drawcountries()
    #m.shadedrelief()
    m.drawmapboundary(fill_color='aqua')
    m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcoastlines()
    m.drawcountries()
    x, y = m(stations.Longitude, stations.Latitude)
    m.scatter(x, y, marker='D',color='m')
    plt.title(title)
    return fig

def plot_distances(distances, figsize = (10, 10)):
    fig = px.box(distances)
    fig.update_layout(
        title={
            'text': "Distances moyenne des stations les plus proches"
        },
        xaxis={
            'title': {'text': "rang de proximité"}
        }
        ,
        yaxis={
            'title': {'text': "distance (km)"}
        }
    )
    return fig
