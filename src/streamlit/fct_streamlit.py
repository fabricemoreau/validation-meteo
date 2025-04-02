import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def plot_jours(jours):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=jours.datemesure, y=jours.day_sin, name='sinus'))
    fig.add_trace(go.Scatter(x=jours.datemesure, y=jours.day_cos, name='cosinus'))
    fig.update_layout(hovermode='x unified')
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
    
def plot_hyperparam_model(df):
    fig = go.Figure(go.Scatter(x = df.recall, y = df.accuracy, 
                               mode='markers',
                               customdata= df[['param', 'TP', 'TN', 'FP', 'FN']],
                               hovertemplate = 
                                '<b>Recall:</b>: %{x:.1%}' + 
                                '<br>Accuracy: %{y:.1%}' + 
                                '<br><i>Hyperparameters</i>: %{customdata[0]}' +
                                '<br>TN: %{customdata[2]:.1%}' +
                                '<br>TP: %{customdata[1]:.1%}' +
                                '<br><b>FN: %{customdata[4]:.1%}</b>' +
                                '<br><b>FP: %{customdata[3]:.1%}</b>'
                     ) 
    )
    return fig
    
def plot_anomaly(resultats : pd.DataFrame, param: str, logomf):
    subdata_anomaly = resultats[resultats[f"{param}_anomaly"] == 1]
    fig = go.Figure([
        #   marker = dict(size = 5, line = dict(color = subdata_anomaly['anomaly_nbmodele'], width = 1))
        go.Scatter(name = 'Réelle', x = subdata_anomaly.datemesure, y = subdata_anomaly[f"{param}_origine"], mode = 'markers', showlegend = True),
        go.Scatter(name = 'Correction', x = subdata_anomaly.datemesure, y = subdata_anomaly[param], mode = 'markers', showlegend = True),
        go.Scatter(name = 'DBSCAN détectée', x = subdata_anomaly[subdata_anomaly.anomaly_dbs == 1].datemesure, y = subdata_anomaly.loc[subdata_anomaly.anomaly_dbs == 1, f"{param}_origine"], mode = 'markers', showlegend = True),
        go.Scatter(name = 'Isolation Forest détectée', x = subdata_anomaly[subdata_anomaly.anomaly_if == 1].datemesure, y = subdata_anomaly.loc[subdata_anomaly.anomaly_if == 1, f"{param}_origine"], mode = 'markers', showlegend = True),
        go.Scatter(name = 'Autoencodeur détectée', x = subdata_anomaly[subdata_anomaly.anomaly_autoenc == 1].datemesure, y = subdata_anomaly.loc[subdata_anomaly.anomaly_autoenc == 1, f"{param}_origine"], mode = 'markers', showlegend = True),
    ])
    fig.update_layout(
        title = f"Anomalies pour le paramètre {param} sur l'année 2024",
        hoversubplots="axis",
        hovermode="x unified",
        images = [{
            "source": logomf,
            "x": 1,
            "y": 1,
            "sizex": 0.1,
            "sizey": 0.1,
            "xanchor": "right",
            "yanchor": "top",
            "layer": "above"
        }]
    )
    return fig

def plot_station(station: pd.DataFrame, logomf):
    params = ['ETP', 'GLOT', 'TN', 'TX']
    
    # Création de la figure avec 4 sous-graphiques empilés horizontalement
    fig = make_subplots(rows=4, cols=1)
    nb_anomalies = len(station[station.anomaly == 1])

    for i, param in enumerate(params):
        # Ajout des scatterplots pour chaque paramètre
        fig.add_trace(go.Bar(x=station.loc[station.anomaly == 1, 'datemesure'], 
                             y = np.repeat(station[param].max(), nb_anomalies), 
                             width= 3,
                             name = 'Anomalie (tout paramètre confondu)', 
                             marker=dict(color='magenta', opacity=0.3),
                             legendgroup = 'Anomalies',
                             hovertemplate='Anomalie',
                             showlegend = (i == 0)), # on affiche seulement la légende pour le premier
                      row=i+1, col=1)
        nb_anomalies_autoenc =len(station[station[f"{param}_anomaly_autoenc"] == 1])
        fig.add_trace(go.Bar(x=station.loc[station[f"{param}_anomaly_autoenc"] == 1, 'datemesure'], 
                             y = np.repeat(station[param].max(), nb_anomalies_autoenc), 
                             width= 2,
                             name = 'Anomalie détectée autoencodeur', 
                             hovertemplate=f"Anomalie détectée autoencodeur sur {param}",
                             marker=dict(color='brown', opacity=0.3),
                             legendgroup = 'Anomalies détectées autoencodeur',
                             showlegend = (i == 0)), # on affiche seulement la légende pour le premier
                      row=i+1, col=1)
        fig.add_trace(go.Scatter(
            x=station['datemesure'], 
            y=station[f"{param}_origine"], 
            mode='markers', 
            name=f'{param} Origine',
            marker=dict(size=5, color = 'grey'),
            legendgroup="Origine",
            showlegend=True
        ), row=i+1, col=1)
        
        fig.add_trace(go.Scatter(
            x=station.loc[station[f"{param}_anomaly"] == 1, 'datemesure'], 
            y=station.loc[station[f"{param}_anomaly"] == 1, param], 
            mode='markers', 
            name=f'{param} Corrigé',
            marker=dict(size=10, color = 'red'),
            legendgroup="Corrigé",
            showlegend=True
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=station['datemesure'], 
            y=station[f"{param}_pred_autoenc"], 
            mode='markers', 
            name=f'{param} Prédit autoencodeur',
            marker=dict(size=5, color='dodgerblue'),
            legendgroup="Prédit autoencodeur",
            showlegend=True
        ), row=i+1, col=1)
            # Mise en page pour empiler les sous-graphiques horizontalement
    fig.update_layout(
        title="Comparaison des valeurs pour chaque paramètre",
        xaxis_title="Date de mesure",
        yaxis_title="Valeur",
        grid=dict(rows=4, columns=1, pattern="independent"),  # 4 colonnes indépendantes
        hovermode="x",
        hoversubplots="axis", # actuellement sans effet à cause d'un bug
        images=[{
            "source": logomf,
            "xref": "paper",
            "yref": "paper",
            "x": 1,
            "y": 1,
            "sizex": 0.1,
            "sizey": 0.1,
            "xanchor": "right",
            "yanchor": "top",
            "layer": "above"
        }]
    )

    # Mise à jour des axes pour chaque sous-graphique
    for i, param in enumerate(params):
        fig.update_yaxes(title_text=param, row=i + 1, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)    
    return fig

# renvoie y_pred (4 valeurs) + liste d'anomalie
# X = [ETP, GLOT, TN, TX, 'Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']
def autoencodeur_predict(model, scaler_meta, scaler_param, X):
    seuil_anomalie_autoencodeur = [0.00493416, 0.00489685, 0.00538372, 0.00484346]
    X.loc[:,['Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']] = scaler_meta.transform(X[['Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']])
    X.loc[:,['ETP', 'GLOT', 'TN', 'TX']] = scaler_param.transform(X[['ETP', 'GLOT', 'TN', 'TX']])
    y_pred = model.predict(X)
    ae = np.abs(X[['ETP', 'GLOT', 'TN', 'TX']] - y_pred)
    is_anomaly = np.where(ae > seuil_anomalie_autoencodeur, 1, 0).reshape(-1)
    predict_denormalise = scaler_param.inverse_transform(y_pred).reshape(-1)
    return predict_denormalise, is_anomaly
