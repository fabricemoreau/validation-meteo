
import streamlit as st
import streamlit_mermaid as stmd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime 
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model 
import joblib

import fct_context as fct

@st.cache_resource
def load_logomf():
    return Image.open('src/streamlit/assets/LOGO_MF.png')

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

@st.cache_resource
def load_dbscan_hyperparameters():
    data = pd.read_csv('./reports/streamlit-data/dbscan.csv', sep = ';')
    total = data[['TP', 'TN', 'FP', 'FN']].sum(axis = 1)[0]
    data[['TP', 'TN', 'FP', 'FN']] = data[['TP', 'TN', 'FP', 'FN']] / total
    return data

@st.cache_resource
def load_isolationforest_hyperparameters():
    data = pd.read_csv('./reports/streamlit-data/isolationforest.csv', sep = ';')
    total = data[['TP', 'TN', 'FP', 'FN']].sum(axis = 1)[0]
    data[['TP', 'TN', 'FP', 'FN']] = data[['TP', 'TN', 'FP', 'FN']] / total
    return data

@st.cache_resource
def load_anomaly_2025():
    data = pd.read_csv('./reports/streamlit-data/anomalies2025.csv', sep = ';', parse_dates=['datemesure'])
    return data

@st.cache_resource
def load_station4501():
    data = pd.read_csv('./reports/streamlit-data/station4501.csv', sep = ';', parse_dates=['datemesure'])
    return data

@st.cache_resource
def load_autoencodeur_scalers():
    model = load_model('./models/autoencodeur.keras')
    scaler_param = joblib.load('models/joblib_scaler_param_ETP-GLOT-TN-TX_2010-2022-0.1.gz')
    scaler_meta = joblib.load('models/joblib_scaler_meta.gz')
    return model, scaler_param, scaler_meta

logomf = load_logomf()
stations = load_data()
example_struct = load_example_struct()
jours = load_jours()
dbscan_hyperparameters = load_dbscan_hyperparameters()
isolationforest_hyperparameters = load_isolationforest_hyperparameters()
anomaly2025 = load_anomaly_2025()
station = load_station4501()
autoencodeur, scaler_param, scaler_meta = load_autoencodeur_scalers()

seuil_anomalie_data = pd.DataFrame({'ETP' : [0.18],
                                   'GLOT (J/cm²)': [83],
                                   'RR (mm)': [0.6],
                                   'TN (°C)': [0.6],
                                   'TX (°C)': [0.8]
                                   })

st.title("Projet détection d'anomalie sur des données météo")
st.sidebar.title("Sommaire")
pages = ["Accueil", "Contexte", "Le jeu de données", "Visualisation", "Préparation des données", "Modèles clustering", "Modèle autoencodeur", "Exemples", "Conclusion"]
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
        st.dataframe(seuil_anomalie_data)
# Modélisation
elif page == pages[5] :
    st.write("### Modèles de clustering")
    # exploration hyperparamètres DBSCAN densité => anomalies dans les données
    if st.checkbox("DBSCAN", value = True):
        st.write("Détection d'anomalie par la densité")
        st.write("Données utilisées: coordonnées géographiques, altitude, day_sin, day_cos et paramètres non corrigés")
        st.write("Normalisation par centrage-réduction")
        st.write("Utilisation des données sur 1/5e des données en raison des capacités de traitement de la machine")
        st.plotly_chart(fct.plot_hyperparam_model(dbscan_hyperparameters))
        # ajouter graphe des anomalies par paramètre
        
    # exploration hyperparamètres IF: règles plus précises
    if st.checkbox("Isolation Forest", value = True):
        st.write("Détection d'anomalie par la recherche de règles de séparations pour isoler les anomalies")
        st.write("Données utilisées: coordonnées géographiques, altitude, day_sin, day_cos et paramètres non corrigés")
        st.write("Normalisation par centrage-réduction")
        st.write("Toutes les données sont utilisées")
        st.plotly_chart(fct.plot_hyperparam_model(isolationforest_hyperparameters))
        # ajouter graphe des anomalies par paramètre
# Modèle autoencodeur
elif page == pages[6] :
    st.write("### Modèle autoencodeur")
    if st.checkbox("Séparation du jeu de données en trois parties", value = True):
        st.write("Pour les trois étapes suivantes.")
        st.image("./reports/figures/series_train_val_test.png", caption = "Séparation du jeu de données pour l'entrainement, la validation, le test")
        st.write("*en rouge: données entrainement de l'autoencodeur* **(valeurs corrigées)**")
        st.write("*en bleu: données pour validation de l'autoencodeur* **(valeurs corrigées)**")
        st.write("*en vert: données pour tester la détection d'anomalie* **(valeurs brutes)**")
    if st.checkbox("Etape 1: entrainement de l'autoencodeur", value = True):
        st.write('Entrées: **ETP, GLOT, TN, TX** + altitude, Lambert93x, Lambert93y, day_sin, day_cos')
        st.write('Sorties: **ETP, GLOT, TN, TX**')
        st.image('./reports/figures/autoencodeur_schema.png', caption='Schéma de l''autoencodeur')
        st.write('Fonction de coût: MSE')
        st.image('./reports/figures/autoencodeur_train_history.png', caption='historique d''entrainement')
    if st.checkbox("Etape 2: détermination d'un seuil de détection", value = True):
        st.write("Bonne performances sur le jeu de validation. RMSE dénormalisée: ")
        st.dataframe(pd.DataFrame({'ETP' : [0.04],
                                   'GLOT (J/cm²)': [13.3],
                                   'TN (°C)': [0.2],
                                   'TX (°C)': [0.2]
                                   })) 
        st.write("Un écart supérieure à l'**erreur absolue maximale** sur un paramètre est une anomalie")
        st.image('./reports/figures/autoencodeur_val_ae_loss.png', caption='erreurs absolues maximales sur jeu de validation')
        st.write("Pour minimiser les faux négatifs, on choisit le décile 9 de l'erreur minimale absolue. En valeur dénormalisée:")
        st.dataframe(pd.DataFrame({'ETP' : [0.06],
                                   'GLOT (J/cm²)': [20.4],
                                   'TN (°C)': [0.30],
                                   'TX (°C)': [0.29]
                                   })) 
    if st.checkbox("Etape 3: détection d'anomalie", value = True):
        st.write("On effectue les prédiction sur le jeu de test. Tout écart entre les données d'origine et les données prédites supérieur au seuil est une anomalie")
        crosstab = pd.DataFrame({0: [71.9, 2.4], 1: [24.6, 1.1]})
        crosstab.columns.name = "Prédit"
        crosstab.index.name = "Observé"
        st.dataframe(crosstab) 
        st.write("**recall: 30.4%, accuracy: 73.0%**")
# Exemples
elif page == pages[7] :
    st.write("### Exemples")
    if st.checkbox("Les anomalies détectées", value = True):
        param = st.radio(
                "Choisissez un paramètre météo",
                ['ETP', 'GLOT', 'RR', 'TN', 'TX'],
                captions=['ETP', 'Rayonnement global', 'Cumul de pluie', 'Température minimale', 'Température maximale'],
                horizontal=True
            )
        fig = fct.plot_anomaly(anomaly2025, param, logomf)
        st.plotly_chart(fig)
    if st.checkbox("Autoencodeur sur une station météo", value = True):
        st.plotly_chart(fct.plot_station(station, logomf))
        st.text("matrice de confusion:")
        st.dataframe(pd.crosstab(station['anomaly'], station['anomaly_autoenc'], rownames=["Observé"], colnames=["Prédit"]))
    # autoencodeur: si on modifie des valeurs
    if st.checkbox("Test de prédiction de l'autoencodeur", value = True):
        date_consultation_station = st.slider(
            "Choisir une date : ",
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31),
            value=datetime(2024, 11, 23),
            format="MM/DD/YY",
        )
        rows_autoenc = st.columns(6, vertical_alignment = 'center')
        input_val = [0,0,0,0]
        parametres = ['ETP', 'GLOT', 'TN', 'TX']
        seuil_anomalie_data_autoenc = seuil_anomalie_data.drop(columns = 'RR (mm)').iloc[0].values
        
        rows_autoenc[0].markdown('**Paramètre**')
        for param in parametres:
            rows_autoenc[0].text(param)
            rows_autoenc[0].text('') # pour aligner avec les inputs
        rows_autoenc[1].markdown('**Valeur station**')
        for param in parametres:
            rows_autoenc[1].text(station.loc[station.datemesure == date_consultation_station, f"{param}_origine"].values[0])
            rows_autoenc[1].text('') # pour aligner avec les inputs
        rows_autoenc[2].markdown('**Valeur à tester**')
        for i, param in enumerate(parametres):
            input_val[i] = rows_autoenc[2].number_input(label= param, 
                                                        label_visibility='collapsed',
                                                        step = 0.5,
                                                        format = "%0.1f",
                                                        placeholder = station.loc[station.datemesure == date_consultation_station, param].values[0],
                                                        value = station.loc[station.datemesure == date_consultation_station, param].values[0])
        rows_autoenc[3].markdown('**Anomalie**')
        for i, param in enumerate(parametres):
            dif = np.abs(station.loc[station.datemesure == date_consultation_station, f"{param}_origine"].values[0] - input_val[i])
            if dif > seuil_anomalie_data_autoenc[i]:
                rows_autoenc[3].markdown("**anomalie**")
            else:
                rows_autoenc[3].text("ok")
            rows_autoenc[3].text('') # pour aligner avec les inputs
        #calcul
        X = station.loc[station.datemesure == date_consultation_station, 
                        ['ETP',
                        'GLOT',
                        'TN',
                        'TX',
                        'Altitude',
                        'Lambert93x',
                        'Lambert93y',
                        'day_sin',
                        'day_cos']]
        X = X.copy() # éviter de modifier le jeu de données en cache de streamlit
        X.loc[:,parametres] = input_val
        y_pred, anomalie_pred = fct.autoencodeur_predict(autoencodeur, scaler_meta, scaler_param, X)
        rows_autoenc[4].markdown('**Prédiction**')
        for i, param in enumerate(parametres):
            rows_autoenc[4].text('{:0.1f}'.format(y_pred[i]))
            rows_autoenc[4].text('') # pour aligner avec les inputs
        rows_autoenc[5].markdown('**Détection**')
        for i, param in enumerate(parametres):
            if anomalie_pred[i] > 0:
                rows_autoenc[5].markdown('**Anomalie**')
            else:
                rows_autoenc[5].text('ok')
            rows_autoenc[5].text('') # pour aligner avec les inputs
            
        # modif de valeur
        # prédiction
# Conclusion
elif page == pages[8] :
    st.write("### Conclusion")
    # preprocessing spatiotemporel





