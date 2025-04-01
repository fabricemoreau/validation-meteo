import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
import joblib
import visualkeras

DONNEES = "data/processed/meteo_pivot_cleaned_2010-2024_0.1.csv" 
STATIONS = "data/processed/stationsmeteo_processed_2010-2024.csv"
PARAMETRES = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

# logos
logomf = plt.imread('src/streamlit/assets/LOGO_MF.png')
logoarvalis = plt.imread('src/streamlit/assets/logo-arvalis.png')

# chargement mesures
df = pd.read_csv(DONNEES, sep = ';', parse_dates = ['datemesure'])
df.loc[df.anomaly > 0, 'anomaly'] = 1

stations_timetable = df[['datemesure', 'codearvalis', 'TN']]
stations_timetable = stations_timetable.drop_duplicates()
stations_timetable = stations_timetable.pivot(index = 'codearvalis', columns = 'datemesure', values = 'TN')
# on remplace par valeur binaire 0 ou 1
stations_timetable = stations_timetable.notnull().astype(int)
stations_timetable = stations_timetable.fillna(0)

# séries temporelles des stations
fig = plt.figure()
plt.matshow(stations_timetable, cmap = 'gray_r', aspect = 2)
plt.xticks([i for i in range(0, stations_timetable.shape[1]+1, 366)], [stations_timetable.columns[i].year for i in range(0, stations_timetable.shape[1]+1, 366)])
plt.yticks([i for i in range(0, stations_timetable.shape[0]+1, 100)], [stations_timetable.index[i] for i in range(0, stations_timetable.shape[0]+1, 100)])
plt.xlabel("Temps (années)")
plt.ylabel("Numéro de station météo")
plt.savefig('./reports/figures/series_stations_meteo.png')

# description des paramètres
for param in PARAMETRES:
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    ax.boxplot(df[param])
    ax.set_title(f"Variation de {param}")
    ax = plt.subplot(1, 2, 2)
    ax.hist(df[param], bins = 50)
    ax.set_title(f"Distribution de {param}")
    fig.subplots_adjust(bottom=0.15) 
    
    #ax = fig.add_subplot()
    addLogo = OffsetImage(logomf, zoom=0.05)
    addLogo.set_offset((100,15)) 
    ax.add_artist(addLogo)
    #ax = fig.add_subplot()
    addLogo = OffsetImage(logoarvalis, zoom=0.3)
    addLogo.set_offset((200,15)) 
    ax.add_artist(addLogo)
    plt.savefig('./reports/figures/' + param + '.png')
    
# anomalies 
for param in PARAMETRES:
    df_valides = df[df[f"{param}_anomaly"] == 0]
    df_anomalies = df[df[f"{param}_anomaly"] == 1]
    fig = plt.figure(figsize=(20,10))
    plt.scatter(df_valides.datemesure, df_valides[param], label = f"{param} valides", c = "blue")
    #plt.scatter(df_anomalies.datemesure, df_anomalies[f"{param}"], label = f"{param} corrections", c="green")
    plt.scatter(df_anomalies.datemesure, df_anomalies[f"{param}_origine"], label = f"{param} anomalies", c="red")
    plt.legend()
    plt.savefig('./reports/figures/' + param + '_time.png')

# corrections 
for param in PARAMETRES:
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, 2, 1)
    df_corrections = df[df[f"{param}_anomaly"] == 1]
    ax.boxplot(df_corrections[f"{param}_difference"])
    ax.set_title(f"Corrections de {param}")
    ax = plt.subplot(1, 2, 2)
    ax.hist(df_corrections[f"{param}_difference"], bins = 50)
    ax.set_title(f"Distribution des corrections de {param}")
    fig.subplots_adjust(bottom=0.15) 
    plt.savefig('./reports/figures/' + param + '_corrections.png')

# days_sin_cos
jours = df.loc[(df.codearvalis == 9151) & df.datemesure.dt.year.isin([2014, 2015]), ['datemesure', 'day_sin', 'day_cos']]
jours['numero_jour'] = jours.datemesure.dt.day_of_year
jours.to_csv('reports/streamlit-data/jours.csv', index = False, sep = ';')

# génération tableaux données
brut = pd.read_csv('data/raw/donneesmeteo_2010-2024_363stations.csv', sep = ';', parse_dates = ['datemesure'])
brut.sort_values(['datemesure', 'codearvalis']).head(50).to_csv('reports/streamlit-data/df_brut.csv', index = False, sep =';')

sample = pd.read_csv('data/processed/meteo_pivot_2010-2012.csv', sep = ';', parse_dates = ['datemesure'])
sample.sort_values(['datemesure', 'codearvalis']).head(10).to_csv('reports/streamlit-data/df_sample.csv', index = False, sep =';')

# chargement liste stations
stations = pd.read_csv(STATIONS, sep = ';')
dates_presence =  df.groupby('codearvalis').datemesure.agg(['min', 'max'])
stations = stations.merge(dates_presence, left_on = 'Station', right_index = True)
stations.to_csv('reports/streamlit-data/stations.csv', index = False, sep = ';')

# schema autoencodeur
model = load_model('models/autoencodeur.keras')
visualkeras.layered_view(model, 
                         to_file='reports/figures/autoencodeur_schema.png',
                         legend = True,
                         draw_volume = False,
                         scale_xy= 2)

# préparation du fichier de résultat
resultats_if = pd.read_csv('data/testModelResults/isolationForest_ETP-GLOT-RR-TN-TX_False_anomaly_results.csv', parse_dates=['datemesure'])
resultats_if = resultats_if.rename(columns = {'anomaly_pred': 'anomaly_if'})
resultats_dbs = pd.read_csv('data/testModelResults/dbscan_ETP-GLOT-RR-TN-TX_True_anomaly_results.csv', parse_dates=['datemesure'])
resultats_dbs = resultats_dbs.rename(columns = {'anomaly_pred': 'anomaly_dbs'})
# uniquement l'année 2024 qui n'est pas utilisée pour entrainer l'autoencodeur
resultats_dbs = resultats_dbs[resultats_dbs.datemesure.dt.year.isin([2024])]
resultats = resultats_dbs.merge(resultats_if[['codearvalis', 'datemesure', 'anomaly_if']], how = 'left', left_on=['codearvalis', 'datemesure'], right_on = ['codearvalis', 'datemesure'])
resultats.loc[resultats.anomaly > 0, "anomaly"] = 1

# prédiction autoencodeur
scaler_param = joblib.load('models/joblib_scaler_param_ETP-GLOT-TN-TX_2010-2022-0.1.gz')
threshold = [0.00493416, 0.00489685, 0.00538372, 0.00484346]
scaler_meta = joblib.load('models/joblib_scaler_meta.gz')
meta_features = ['Altitude', 'Lambert93x', 'Lambert93y', 'day_sin', 'day_cos']
meta_features_scaled = pd.DataFrame(scaler_meta.transform(resultats[meta_features]), columns = meta_features)
data = pd.DataFrame(scaler_param.transform(resultats[[f"{param}_origine" for param in ['ETP', 'GLOT', 'TN', 'TX']]].values), columns = ['ETP', 'GLOT', 'TN', 'TX'])
meta_features_scaled = pd.concat([data, meta_features_scaled], ignore_index = True, axis = 1)
y_pred_autoenc = model.predict(meta_features_scaled.values)
resultats[[f"{param}_pred_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = y_pred_autoenc
resultats[[f"{param}_pred_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = scaler_param.inverse_transform(resultats[[f"{param}_pred_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]])
resultats[[f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = np.where(np.abs(data - y_pred_autoenc) > threshold, 1, 0)
resultats['anomaly_autoenc'] = np.where(resultats[[f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]].sum(axis = 1) > 0, 1, 0)

resultats['anomaly_nbmodele'] = resultats[['anomaly_dbs', 'anomaly_if', 'anomaly_autoenc']].sum(axis = 1)
resultats.to_csv('reports/streamlit-data/anomalies2025.csv', index = False, sep = ';')

###suivi d'une station météo
station4501 = df[(df.datemesure.dt.year == 2024) & (df.codearvalis == 4501)]
meta_features_scaled = pd.DataFrame(scaler_meta.transform(station4501[meta_features]), columns = meta_features)
data = pd.DataFrame(scaler_param.transform(station4501[[f"{param}_origine" for param in ['ETP', 'GLOT', 'TN', 'TX']]].values), columns = ['ETP', 'GLOT', 'TN', 'TX'])
meta_features_scaled = pd.concat([data, meta_features_scaled], ignore_index = True, axis = 1)
y_pred_autoenc = model.predict(meta_features_scaled.values)
station4501[[f"{param}_pred_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = scaler_param.inverse_transform(y_pred_autoenc)
station4501[[f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = np.where(np.abs(data - y_pred_autoenc) > threshold, 1, 0)
station4501['anomaly_autoenc'] = np.where(station4501[[f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]].sum(axis = 1) > 0, 1, 0)
# on retire des anomalies prédite les différences inférieures au seuil d'anomalie
diff = np.abs(station4501.loc[station4501.anomaly_autoenc ==1, [f"{param}_pred_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]].values -
              station4501.loc[station4501.anomaly_autoenc ==1, [f"{param}_origine" for param in ['ETP', 'GLOT', 'TN', 'TX']]].values)
seuil_anomalie = [0.18, 83, 0.6, 0.8]
seuil_anomalie = [seuil / 2 for seuil in seuil_anomalie ]
diff_sup_seuil = np.where(diff > seuil_anomalie, 1, 0)
station4501.loc[station4501.anomaly_autoenc ==1, [f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]] = diff_sup_seuil
station4501['anomaly_autoenc'] = np.where(station4501[[f"{param}_anomaly_autoenc" for param in ['ETP', 'GLOT', 'TN', 'TX']]].sum(axis = 1) > 0, 1, 0)

station4501.to_csv('reports/streamlit-data/station4501.csv', index = False, sep = ';')

"""
from imblearn.under_sampling import RandomUnderSampler

# Définir les caractéristiques (X) et la cible (y)
X = df.drop(columns=['anomaly'])  # Toutes les colonnes sauf "anomaly"
y = df['anomaly']  # Colonne cible

# Initialiser le sous-échantillonneur
rus = RandomUnderSampler()

# Appliquer le sous-échantillonnage
X_resampled, y_resampled = rus.fit_resample(X, y)

# Reconstituer un DataFrame équilibré
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# Afficher la répartition des classes après sous-échantillonnage
print(df_resampled['anomaly'].value_counts())
df_resampled.to_csv('reports/streamlit-data/df.csv', index = False, sep = ';')
"""