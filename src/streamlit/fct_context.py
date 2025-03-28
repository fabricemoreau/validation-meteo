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
    fig = px.box(distances) # px.scattermap?
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

def plot_anomalies():
    return 1


def plot_custom_hover():
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(
        x = [1,2,3,4,5],
        y = [2.02825,1.63728,6.83839,4.8485,4.73463],
        hovertemplate =
        '<i>Price</i>: $%{y:.2f}'+
        '<br><b>X</b>: %{x}<br>'+
        '<b>%{text}</b>',
        text = ['Custom text {}'.format(i + 1) for i in range(5)],
        showlegend = False))

    fig.add_trace(go.Scatter(
        x = [1,2,3,4,5],
        y = [3.02825,2.63728,4.83839,3.8485,1.73463],
        hovertemplate = 'Price: %{y:$.2f}<extra></extra>',
        showlegend = False))

    fig.update_layout(
        hoverlabel_align = 'right',
        title = "Set hover text with hovertemplate")

    fig.show()
