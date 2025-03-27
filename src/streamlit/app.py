
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime #cartes


#from src.streamlit.fct_context import stations_map, plot_distances
from fct_context import stations_map, plot_distances

@st.cache_data
def load_data():
    return pd.read_csv('./reports/streamlit-data/stations.csv', sep = ';', parse_dates=['min', 'max'])

stations = load_data()

st.title("Projet détection d'anomalie sur des données météo")
st.sidebar.title("Sommaire")
pages = ["Contexte", "Le jeu de données", "Visualisation", "Préparation des données", "Modélisation", "Exemples", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

# objectif: le moins de texte possible

# Contexte
if page == pages[0] :
    st.write("### Contexte")
# Le jeu de données
elif page == pages[1] :
    st.write("## Le jeu de données")
    
    st.write('### Stations météo')
    if st.checkbox("Une couverture nationale en France continentale"):
        date_consultation = st.slider(
            "Date de consultation : ",
            min_value=datetime(2010, 1, 1),
            max_value=datetime(2024, 12, 31),
            value=datetime(2020, 1, 1),
            format="MM/DD/YY",
        )
        # On se restreint aux stations météo ouvertes actuellement
        sm_ouvertes = stations[(stations['min'] <= date_consultation) & (stations['max'] >= date_consultation)]
        fig = stations_map(sm_ouvertes, f"Carte des stations météo ouvertes au {date_consultation}", figsize = (7,7))
        st.pyplot(fig)
    
    if st.checkbox("des séries temporelles non synchrones"):
        st.image('./reports/figures/series_stations_meteo.png', caption = 'ensemble des séries temporelles')
    # distances
    if st.checkbox("densité du réseau"):
        fig = plot_distances(stations[[f"distance_station{i}" for i in range(1, 6)]])
        st.plotly_chart(fig)
    # séries
    
# Visualisation
elif page == pages[2] :
    st.write("### Visualisation")
    # répartition des paramètres 
    # évolution temporelle
    # spatiales
    # corrections temporelle (année, mois), répartition des corrections, corrections dans le jeu de données
# Préparation des données
elif page == pages[3] :
    st.write("### Préparation des données")
    # structure jeu de données
    # jours sin cos
    # seuillage anomalies
    
# Modélisation
elif page == pages[4] :
    st.write("### Modélisation")
    # choix modélisation, choix préprocessing
    # exploration hyperparamètres IF: règles plus précises
    # exploration hyperparamètres DBSCAN densité => anomalies dans les données
    # FP/FN par paramètre
    # tester une valeur
    # page dédiée à autoencodeur? 
# Exemples
elif page == pages[5] :
    st.write("### Exemples")
    # récapitulatif des sorties modèles?
# Conclusion
elif page == pages[6] :
    st.write("### Conclusion")
    # preprocessing spatiotemporel





