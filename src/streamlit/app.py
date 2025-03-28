
import streamlit as st
import streamlit_mermaid as stmd
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
from pathlib import Path


#from src.streamlit.fct_context import stations_map, plot_distances
import fct_context as fct

@st.cache_resource
def load_data():
    return pd.read_csv('./reports/streamlit-data/stations.csv', sep = ';', parse_dates=['min', 'max'])

@st.cache_resource
def load_example_struct():
    return (pd.read_csv('./reports/streamlit-data/df_brut.csv', sep = ';', parse_dates=['datemesure']),
           pd.read_csv('./reports/streamlit-data/df_sample.csv', sep = ';', parse_dates=['datemesure']) )

@st.cache_resource
def load_jours():
    return pd.read_csv('./reports/streamlit-data/jours.csv', sep = ';', parse_dates=['datemesure'])

stations = load_data()
example_struct = load_example_struct()
jours = load_jours()

st.title("Projet détection d'anomalie sur des données météo")
st.sidebar.title("Sommaire")
pages = ["Accueil", "Contexte", "Le jeu de données", "Visualisation", "Préparation des données", "Modélisation", "Exemples", "Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

# objectif: le moins de texte possible

# Contexte
if page == pages[0] :
    st.write("### Accueil")
elif page == pages[1] :
    st.write("### Contexte")
# Le jeu de données
elif page == pages[2] :
    st.write("### Le jeu de données")
    
    if st.checkbox("Structure", value = True):
        stmd.st_mermaid(Path('reports/figures/schema-donnees.mmd').read_text())
        
    if st.checkbox("Une couverture nationale en France continentale", value = True):
        date_consultation = st.slider(
            "Date de consultation : ",
            min_value=datetime(2010, 1, 1),
            max_value=datetime(2024, 12, 31),
            value=datetime(2020, 1, 1),
            format="MM/DD/YY",
        )
        # On se restreint aux stations météo ouvertes actuellement
        sm_ouvertes = stations[(stations['min'] <= date_consultation) & (stations['max'] >= date_consultation)]
        fig = fct.stations_map(sm_ouvertes, f"Carte des stations météo ouvertes au {date_consultation}", figsize = (7,7))
        st.pyplot(fig)
    
    if st.checkbox("des séries temporelles non synchrones", value = True):
        st.image('./reports/figures/series_stations_meteo.png', caption = 'ensemble des séries temporelles')
    
    if st.checkbox("densité du réseau", value = True):
        fig = fct.plot_distances(stations[[f"distance_station{i}" for i in range(1, 6)]])
        st.plotly_chart(fig)
    
# Visualisation
elif page == pages[3] :
    st.write("### Visualisation")
    param = st.radio(
            "Choisissez un paramètre météo",
            ['ETP', 'GLOT', 'RR', 'TN', 'TX'],
            captions=['ETP', 'Rayonnement global', 'Cumul de pluie', 'Température minimale', 'Température maximale'],
            horizontal=True
        )
    
    if st.checkbox("Les paramètres météo", value = True):
        st.image(f"./reports/figures/{param}.png", caption = f"Répartition du paramètre météo {param}")
    # TODO: revoir RR
    
    if st.checkbox("Répartition des valeurs des anomalies", value = True):
        st.image(f"./reports/figures/{param}_time.png", caption = f"Répartition du paramètre météo {param}")
    
    if st.checkbox("Fréquence des anomalies", value = True):
        st.write("Des anomalies sont présentes sur **2%** des données")
        st.image("./reports/figures/Corrections per year.png", caption = "Répartition des corrections, cumul par an")
        st.image("./reports/figures/Corrections per month.png", caption = "Répartition des corrections, cumul par mois")
        # corrections temporelle (année, mois), répartition des corrections, corrections dans le jeu de données
    if st.checkbox("Répartition des corrections", value = True):
        st.image(f"./reports/figures/{param}_corrections.png", caption = f"Corrections du paramètre météo {param}")
# Préparation des données
elif page == pages[4] :
    st.write("### Préparation des données")
    # structure jeu de données
    if st.checkbox("Changement de structure du jeu de données", value = True):
        rows = st.columns(2)
        rows[0].markdown('#### Format origine')
        rows[0].dataframe(example_struct[0])
        rows[1].markdown('#### Format transformé')
        rows[1].dataframe(example_struct[1])
    # jours sin cos
    if st.checkbox("Transformation des dates", value = True):
        st.write("Objectif: une expression de la date qui permet de garder la proximité entre les dates aux changements d'années et garder informations saisonnière ")
        st.markdown('$daysin = \\sin\\left(\\frac{2 \\pi d}{366}\\right)$')
        st.markdown('$daycos = \\cos\\left(\\frac{2 \\pi d}{366}\\right)$')
        st.write('`d` est le numéro du jour dans l''année (1 au premier janvier, 365 ou 366 au 31 décembre)')
        st.plotly_chart(fct.plot_jours(jours))
        
    # seuillage anomalies
    if st.checkbox("Seuillage des anomalies", value = True):
        st.write('Beaucoup de très faibles corrections. On retire des anomalies celles qui sont inférieures à 10% de l''écart-type')
        st.dataframe(pd.DataFrame({'ETP' : [0.18],
                                   'GLOT (J/cm²)': [83],
                                   'RR (mm)': [0.6],
                                   'TN (°C)': [0.6],
                                   'TX (°C)': [0.8]
                                   }))
# Modélisation
elif page == pages[5] :
    st.write("### Modélisation")
    # choix modélisation, choix préprocessing
    # exploration hyperparamètres IF: règles plus précises
    # exploration hyperparamètres DBSCAN densité => anomalies dans les données
    # FP/FN par paramètre
    # tester une valeur
    # page dédiée à autoencodeur? 
    # Exemple de données
# Exemples
elif page == pages[6] :
    st.write("### Exemples")
    # récapitulatif des sorties modèles?
# Conclusion
elif page == pages[7] :
    st.write("### Conclusion")
    # preprocessing spatiotemporel





