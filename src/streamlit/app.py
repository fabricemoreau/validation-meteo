
import streamlit as st
import pandas as pd
from mpl_toolkits.basemap import Basemap # afficher les cartes
import matplotlib.pyplot as plt
from datetime import datetime #cartes

@st.cache_data
def load_data():
    return pd.read_csv('./src/streamlit/data/stationsmeteo_2010-2024.csv', sep = ';', parse_dates=['Min', 'Max'])
    
stations = load_data()

st.title("Projet détection d'anomalie sur des données méto")
st.sidebar.title("Sommaire")
pages=["Contexte", "Le jeu de données", "Visualisation", "Préparation des données", "Modélisation", "Exemples", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)

# Contexte
if page == pages[0] :
    st.write("### Contexte")
# Le jeu de données
elif page == pages[1] :
    st.write("## Le jeu de données")
    
    st.write('### Stations météo')
    
    date_consultation = st.slider(
        "Jour de consultation?",
        min_value=datetime(2010, 1, 1),
        max_value=datetime(2024, 12, 31),
        value=datetime(2020, 1, 1),
        format="MM/DD/YY",
    )
    st.write("Start time:", date_consultation)
    # On se restreint aux stations météo ouvertes actuellement
    sm_ouverte = stations[(stations['Min'] <= date_consultation) & (stations['Max'] >= date_consultation)]
    fig = plt.figure(figsize=(7, 7))
    m =  Basemap(llcrnrlon=-5.,llcrnrlat=42.,urcrnrlon=9.5,urcrnrlat=51.,
                resolution='i', projection='tmerc', lat_0 = 39.5, lon_0 = -3.25)

    m.drawcoastlines()
    m.drawcountries()
    m.shadedrelief()
    x, y = m(sm_ouverte.Longitude, sm_ouverte.Latitude)
    m.scatter(x, y, marker='D',color='m')
    plt.title(f"Carte des stations météo ouvertes au {date_consultation}")
    st.pyplot(fig)
# Visualisation
elif page == pages[2] :
    st.write("### Visualisation")
# Préparation des données
elif page == pages[3] :
    st.write("### Préparation des données")
# Modélisation
elif page == pages[4] :
    st.write("### Modélisation")
# Exemples
elif page == pages[5] :
    st.write("### Exemples")
# Conclusion
elif page == pages[6] :
    st.write("### Conclusion")





